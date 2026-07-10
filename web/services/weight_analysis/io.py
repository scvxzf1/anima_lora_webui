"""Safetensors header and tensor loading helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from web.services import path_safety


def _read_safetensors_header(path: Path) -> tuple[dict[str, str], list[str]]:
    return path_safety.read_safetensors_header(path)


def _load_safetensors_tensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors import safe_open

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[str(key)] = f.get_tensor(key).detach().cpu()
        return tensors
    except Exception as exc:
        raise ValueError(f"读取 safetensors 张量失败: {exc}") from exc


def _read_safetensors_header_bytes(data: bytes) -> tuple[dict[str, str], list[str]]:
    return path_safety.read_safetensors_header_bytes(data)


def _load_safetensors_tensors_bytes(data: bytes) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load

        return {str(key): tensor.detach().cpu() for key, tensor in load(data).items()}
    except Exception as exc:
        raise ValueError(f"读取 safetensors 上传张量失败: {exc}") from exc

