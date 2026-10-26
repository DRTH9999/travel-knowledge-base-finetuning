from __future__ import annotations

from dataclasses import dataclass
import time

from .common import ServiceError, check_candidate, json_bytes, sha


@dataclass
class Prepared:
    ids: list[int]
    input_sha256: str


class TextEncoder:
    def __init__(self, tokenizer, candidate, max_total_tokens):
        self.tokenizer, self.max_total_tokens = tokenizer, max_total_tokens
        if sha(tokenizer.chat_template.encode("utf-8")) != candidate["chat_template_sha256"]:
            raise RuntimeError("Chat template identity mismatch.")
        if tokenizer.eos_token_id != candidate["eos_token_id"]:
            raise RuntimeError("EOS identity mismatch.")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    def prepare(self, prompt, max_new_tokens):
        text = self.tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) + max_new_tokens > self.max_total_tokens:
            raise ServiceError("Input plus requested output exceeds the token budget; nothing was truncated.",
                               400, "context_length_exceeded")
        return Prepared(ids, sha(json_bytes(ids)))


class TransformersEngine:
    """Same loading and GenerationConfig as evaluation/evaluate.py; no remote code."""
    def __init__(self, settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        identity = check_candidate(settings)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required; no CPU fallback is permitted.")
        self.encoder = TextEncoder(AutoTokenizer.from_pretrained(str(settings.model_dir), local_files_only=True,
                                                               trust_remote_code=False),
                                   identity["candidate"], settings.max_total_tokens)
        model = AutoModelForCausalLM.from_pretrained(
            str(settings.model_dir), local_files_only=True, trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        if model.config.model_type != "qwen3" or not getattr(model, "is_loaded_in_4bit", False):
            raise RuntimeError("Expected the 4-bit Qwen3 base model.")
        self.model = PeftModel.from_pretrained(model, str(settings.adapter_dir), is_trainable=False,
                                             autocast_adapter_dtype=False, local_files_only=True).eval()
        self.model.config.use_cache = True
        self.identity = dict(identity, backend="transformers_peft_nf4", enable_thinking=False, do_sample=False,
                             num_beams=1, max_new_tokens=512, max_total_tokens=settings.max_total_tokens,
                             gpu=torch.cuda.get_device_name(0),
                             gpu_total_gib=round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3))
        self.failed = False

    def prepare(self, prompt, max_new_tokens):
        return self.encoder.prepare(prompt, max_new_tokens)

    def generate(self, prepared, max_new_tokens, cancel, deadline, emit):
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList, TextStreamer
        tokenizer = self.encoder.tokenizer

        class CancelOrDeadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancel.is_set() or time.monotonic() >= deadline

        class CallbackStreamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                if text:
                    emit(text)

        inputs = mask = tokens = None
        try:
            inputs = torch.tensor([prepared.ids], dtype=torch.long, device="cuda:0")
            mask = torch.ones_like(inputs)
            config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                      use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                      eos_token_id=tokenizer.eos_token_id)
            streamer = CallbackStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with torch.inference_mode():
                tokens = self.model.generate(input_ids=inputs, attention_mask=mask, generation_config=config,
                    streamer=streamer, stopping_criteria=StoppingCriteriaList([CancelOrDeadline()]))
            torch.cuda.synchronize()
            new_ids = tokens[0, inputs.shape[-1]:].tolist()
            if cancel.is_set() or time.monotonic() >= deadline:
                raise ServiceError("Generation cancelled or timed out.", 504, "generation_timeout")
            reason = "stop" if new_ids and new_ids[-1] == tokenizer.eos_token_id else "length"
            return dict(answer=tokenizer.decode(new_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False),
                        output_token_ids=new_ids, completion_tokens=len(new_ids), finish_reason=reason,
                        generation_seconds=time.monotonic() - started,
                        peak_allocated_vram_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3))
        except ServiceError:
            raise
        except Exception:
            # A CUDA failure requires an explicit restart; do not claim subsequent requests are healthy.
            self.failed = True
            raise
        finally:
            del inputs, mask, tokens
