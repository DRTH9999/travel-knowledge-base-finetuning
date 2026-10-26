$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuilderPython = 'E:\Anaconda3\python.exe'
$TokenizerPython = 'E:\260515\travel_knowledge_base\.venv\Scripts\python.exe'
$TokenizerDirectory = Join-Path $ProjectRoot '.cache\qwen3-8b-tokenizer'
foreach ($RequiredPath in @($BuilderPython, $TokenizerPython, $TokenizerDirectory)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Missing required path: $RequiredPath"
    }
}
& $BuilderPython (Join-Path $PSScriptRoot 'build_pilot.py')
if ($LASTEXITCODE -ne 0) { throw 'Source/sample build failed; no training started.' }
& $BuilderPython (Join-Path $PSScriptRoot 'validate_pilot.py')
if ($LASTEXITCODE -ne 0) { throw 'Source/format validation failed; no training started.' }
& $TokenizerPython (Join-Path $PSScriptRoot 'tokenizer_export.py') --model-dir $TokenizerDirectory
if ($LASTEXITCODE -ne 0) { throw 'Tokenizer export failed; no training started.' }
& $BuilderPython (Join-Path $PSScriptRoot 'validate_pilot.py') --require-tokenized
if ($LASTEXITCODE -ne 0) { throw 'Final validation failed; no training started.' }
& $BuilderPython (Join-Path $PSScriptRoot 'package_pilot.py')
if ($LASTEXITCODE -ne 0) { throw 'Packaging failed; no training started.' }
Write-Output 'PILOT_BUILD_AND_CHECK_OK'
