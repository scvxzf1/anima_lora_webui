"""Configuration contract for frozen Qwen3-VL cache execution only."""

QWEN_TEXT_ENCODER_CACHE_POLICIES = ("auto", "cpu_offload", "gpu", "cpu")


def validate_cache_policy(policy: str) -> str:
    if policy not in QWEN_TEXT_ENCODER_CACHE_POLICIES:
        raise ValueError(f"Invalid qwen_text_encoder_cache_policy: {policy!r}")
    return policy


def resolve_cache_policy(policy: str, *, device: str = "auto", offload: str = "auto") -> tuple[str, str]:
    """Keep legacy CLI overrides for auto, reject contradictory explicit modes."""
    validate_cache_policy(policy)
    if policy == "auto":
        return device, offload
    resolved = {
        "cpu_offload": ("cuda", "on"),
        "gpu": ("cuda", "off"),
        "cpu": ("cpu", "off"),
    }[policy]
    if (device != "auto" and device.split(":", 1)[0] != resolved[0]) or (offload != "auto" and offload != resolved[1]):
        raise ValueError(f"Cache policy {policy!r} conflicts with --device/--offload overrides")
    return (resolved[0] if device == "auto" else device), resolved[1]
