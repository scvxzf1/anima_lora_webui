@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not defined ANIMA_WEB_HOST set "ANIMA_WEB_HOST=127.0.0.1"
if not defined ANIMA_WEB_PORT set "ANIMA_WEB_PORT=20103"
set "PY=.venv\Scripts\python.exe"

echo [Anima LoRA] Start WebUI
echo [Anima LoRA] Project: %CD%

if not exist "%PY%" (
    echo [Anima LoRA] Existing environment was not found at %PY%.
    echo [Anima LoRA] Run setup_and_start.bat first, or run uv sync manually.
    goto fail
)

"%PY%" -c "import aiohttp" >nul 2>nul
if errorlevel 1 (
    echo [Anima LoRA] aiohttp is missing from the venv.
    echo [Anima LoRA] Run setup_and_start.bat first, or run uv sync manually.
    goto fail
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
