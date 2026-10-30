# Bootstrap a fully project-local environment in .tools\ (no global installs).
#   .tools\uv\        portable uv
#   .tools\python\    managed CPython
#   .tools\venv\      main virtualenv: CUDA PyTorch + web server + in-process backends
#   .tools\hf-home\   Hugging Face model cache
# Model-specific envs (.tools\envs\<name>) are built later by scripts\prepare.py or on first load.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$tools = Join-Path $PSScriptRoot ".tools"
$env:UV_PYTHON_INSTALL_DIR = "$tools\python"
$env:UV_CACHE_DIR = "$tools\uv-cache"
$env:UV_PYTHON_PREFERENCE = "only-managed"
$uv = "$tools\uv\uv.exe"
$uvVersion = "0.13.6"

if (-not (Test-Path $uv)) {
    New-Item -ItemType Directory -Force "$tools\uv" | Out-Null
    $zip = "$tools\uv\uv.zip"
    Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
    Expand-Archive $zip "$tools\uv" -Force
    Remove-Item $zip
}
if (-not (Test-Path "$tools\venv\Scripts\python.exe")) {
    & $uv venv "$tools\venv" --python 3.11
}
$env:VIRTUAL_ENV = "$tools\venv"
& $uv pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE) { throw "torch install failed" }
& $uv pip install -r requirements.txt
if ($LASTEXITCODE) { throw "requirements install failed" }
Write-Host ""
Write-Host "Setup done."
Write-Host "Optional - build model envs + download weights now:  .tools\venv\Scripts\python.exe scripts\prepare.py"
Write-Host "Start the app with run.cmd"
