@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not defined ANIMA_WEB_HOST set "ANIMA_WEB_HOST=127.0.0.1"
if not defined ANIMA_WEB_PORT set "ANIMA_WEB_PORT=20103"
set "PY=.venv\Scripts\python.exe"
set "HF=.venv\Scripts\hf.exe"
set "PATH=%CD%\.venv\Scripts;%PATH%"

echo [Anima LoRA] Setup and start WebUI
echo [Anima LoRA] Project: %CD%

where uv >nul 2>nul
if errorlevel 1 (
    echo [Anima LoRA] uv not found. Installing uv for the current user...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto fail
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
)

where git >nul 2>nul
if not errorlevel 1 (
    git rev-parse --is-inside-work-tree >nul 2>nul
    if not errorlevel 1 (
        git lfs version >nul 2>nul
        if not errorlevel 1 (
            echo [Anima LoRA] Syncing Git LFS files...
            git lfs install --local >nul 2>nul
            git lfs pull
        ) else (
            echo [Anima LoRA] Git LFS is not installed; skipping git lfs pull.
        )
    )
)

echo [Anima LoRA] Syncing Python environment with uv...
uv sync
if errorlevel 1 goto fail

if not exist "%PY%" (
    echo [Anima LoRA] Python venv not found at %PY% after uv sync.
    goto fail
)

if not exist ".env" if exist ".env.example" (
    echo [Anima LoRA] Creating .env from .env.example...
    copy ".env.example" ".env" >nul
)

echo [Anima LoRA] Python version:
"%PY%" --version

echo [Anima LoRA] CUDA check:
"%PY%" -c "import importlib.util; s=importlib.util.find_spec('torch'); exec('print(\"torch check skipped: torch is not installed\")' if s is None else 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available())')"

if "%ANIMA_DOWNLOAD_MODELS%"=="1" (
    echo [Anima LoRA] ANIMA_DOWNLOAD_MODELS=1, downloading default models...
    if not exist "%HF%" (
        echo [Anima LoRA] Hugging Face CLI not found at %HF% after uv sync.
        goto fail
    )
    "%HF%" auth whoami >nul 2>nul
    if errorlevel 1 (
        echo [Anima LoRA] Hugging Face login is required.
        "%HF%" auth login
        if errorlevel 1 goto fail
    )
    "%PY%" tasks.py download-models
    if errorlevel 1 goto fail
) else (
    if not exist "models\diffusion_models\anima-base-v1.0.safetensors" echo [Anima LoRA] Model file not found yet: models\diffusion_models\anima-base-v1.0.safetensors
    if not exist "models\text_encoders\qwen_3_06b_base.safetensors" echo [Anima LoRA] Model file not found yet: models\text_encoders\qwen_3_06b_base.safetensors
    if not exist "models\vae\qwen_image_vae.safetensors" echo [Anima LoRA] Model file not found yet: models\vae\qwen_image_vae.safetensors
    echo [Anima LoRA] To auto-download default models next time, run: set ANIMA_DOWNLOAD_MODELS=1
)

echo [Anima LoRA] Starting WebUI at http://%ANIMA_WEB_HOST%:%ANIMA_WEB_PORT%/
"%PY%" -m web --host "%ANIMA_WEB_HOST%" --port "%ANIMA_WEB_PORT%" %*
if errorlevel 1 goto fail
goto end

:fail
echo.
echo [Anima LoRA] Failed. Please check the messages above.
pause
exit /b 1

:end
endlocal
