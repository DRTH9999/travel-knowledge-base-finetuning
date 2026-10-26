"""Download the selected official Qwen3-8B repository from ModelScope."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-8B'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path,
                        default=Path('/root/autodl-tmp/models/Qwen3-8B'))
    parser.add_argument('--revision', default='master')
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(args.model_dir).free / 1024**3
    print(f'ModelScope model: {MODEL_ID}; requested revision: {args.revision}')
    print(f'Destination: {args.model_dir}; free disk space: {free_gib:.2f} GiB')

    from modelscope import snapshot_download

    location = Path(snapshot_download(
        model_id=MODEL_ID, revision=args.revision,
        local_dir=str(args.model_dir), max_workers=4,
    )).resolve()

    for filename in ['config.json', 'tokenizer_config.json', 'tokenizer.json']:
        if not (location / filename).is_file():
            raise RuntimeError(f'Missing required model file: {filename}')
    config = json.loads((location / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'qwen3':
        raise RuntimeError('The downloaded configuration is not a qwen3 model.')

    index_path = location / 'model.safetensors.index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        filenames = sorted(set(index['weight_map'].values()))
    else:
        filenames = ['model.safetensors']
    if not filenames:
        raise RuntimeError('No model weight files were listed.')
    weights = []
    for filename in filenames:
        path = (location / filename).resolve()
        if not path.is_relative_to(location):
            raise RuntimeError(f'Invalid weight path in index: {filename}')
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Missing or empty model weight: {filename}')
        weights.append({'file': filename, 'bytes': path.stat().st_size})

    manifest = {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'ModelScope', 'model_id': MODEL_ID,
        'requested_revision': args.revision,
        'revision_note': 'A branch name such as master is not an immutable commit ID. '
                         'Preserve SDK metadata and this downloaded snapshot.',
        'modelscope_version': importlib.metadata.version('modelscope'),
        'directory': str(location), 'model_type': config['model_type'],
        'config_sha256': hashlib.sha256((location / 'config.json').read_bytes()).hexdigest(),
        'weight_files': weights,
        'weight_bytes': sum(w['bytes'] for w in weights),
        'check_scope': 'Required files and nonempty indexed shards; '
                       'not a complete cryptographic hash check of all weights.',
    }
    (location / 'download_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print(f"Weight files: {len(weights)}; total: {manifest['weight_bytes'] / 1024**3:.2f} GiB")
    print(f'MODEL_DOWNLOAD_OK: {location}')


if __name__ == '__main__':
    main()
