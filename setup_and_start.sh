#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

HOST="${ANIMA_WEB_HOST:-127.0.0.1}"
PORT="${ANIMA_WEB_PORT:-20103}"
PY=".venv/bin/python"
HF=".venv/bin/hf"

echo "[Anima LoRA] Setup and start WebUI"
echo "[Anima LoRA] Project: $(pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "[Anima LoRA] uv not found. Installing uv for the current user..."
  if ! command -v curl >/dev/null 2>&1; then
    echo "[Anima LoRA] curl is required to install uv. Please install curl first." >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ -f "$HOME/.local/bin/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
  fi
fi

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git lfs version >/dev/null 2>&1; then
    echo "[Anima LoRA] Syncing Git LFS files..."
    git lfs install --local >/dev/null 2>&1 || git lfs install >/dev/null 2>&1 || true
    git lfs pull || true
  else
    echo "[Anima LoRA] Git LFS is not installed; skipping git lfs pull."
  fi
fi

echo "[Anima LoRA] Syncing Python environment with uv..."
uv sync

if [ ! -x "$PY" ]; then
  echo "[Anima LoRA] Python venv not found at $PY after uv sync." >&2
  exit 1
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  echo "[Anima LoRA] Creating .env from .env.example..."
  cp ".env.example" ".env"
fi

echo "[Anima LoRA] Python version:"
"$PY" --version

echo "[Anima LoRA] CUDA check:"
"$PY" - <<'PY' || true
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
except Exception as exc:
    print("torch check skipped:", exc)
PY

if [ "${ANIMA_DOWNLOAD_MODELS:-0}" = "1" ]; then
  echo "[Anima LoRA] ANIMA_DOWNLOAD_MODELS=1, downloading default models..."
  if [ ! -x "$HF" ]; then
    echo "[Anima LoRA] Hugging Face CLI not found at $HF after uv sync." >&2
    exit 1
  fi
  if ! "$HF" auth whoami >/dev/null 2>&1; then
    echo "[Anima LoRA] Hugging Face login is required."
    "$HF" auth login
  fi
  "$PY" tasks.py download-models
else
  for path in \
    "models/diffusion_models/anima-base-v1.0.safetensors" \
    "models/text_encoders/qwen_3_06b_base.safetensors" \
    "models/vae/qwen_image_vae.safetensors"
  do
    if [ ! -f "$path" ]; then
      echo "[Anima LoRA] Model file not found yet: $path"
    fi
  done
  echo "[Anima LoRA] To auto-download default models next time, run: ANIMA_DOWNLOAD_MODELS=1 ./setup_and_start.sh"
fi

echo "[Anima LoRA] Starting WebUI at http://$HOST:$PORT/"
exec "$PY" -m web --host "$HOST" --port "$PORT" "$@"
