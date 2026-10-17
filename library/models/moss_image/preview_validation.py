"""Validate MOSS preview values before loading model components."""

import math


def validate_moss_preview(prompt, sampler="euler"):
    if str(prompt.get("sample_sampler") or sampler) != "euler":
        raise ValueError("MOSS-Image 采样仅支持 Euler")
    for key in ("width", "height"):
        raw = prompt.get(key, 512)
        value = int(raw)
        if isinstance(raw, bool) or float(raw) != value or value <= 0 or value % 16:
            raise ValueError("MOSS-Image 采样宽高须为正整数且为 16 的倍数")
    raw = prompt.get("sample_steps", 50)
    if isinstance(raw, bool) or int(raw) != float(raw) or not 1 <= int(raw) <= 1000:
        raise ValueError("MOSS-Image 采样步数须为 1..1000 的整数")
    scale = float(prompt.get("guidance_scale", prompt.get("scale", 3.0)))
    shift = float(prompt.get("flow_shift") or 5.0)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("MOSS-Image CFG 须为有限非负数")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("MOSS-Image Flow shift 须为有限正数")
    layers = prompt.get("uncond_skip_layers", [22])
    if not isinstance(layers, (list, tuple)) or any(type(i) is not int or not 0 <= i < 42 for i in layers):
        raise ValueError("MOSS-Image uncond_skip_layers 须为 0..41 的整数列表")
