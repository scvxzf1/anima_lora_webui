"""Resolve task preprocessing profiles without importing GPU libraries."""

PROFILES = {
    "auto": ("auto", "auto"),
    "low_vram": (1, 4),
    "balanced": (2, 8),
    "speed": (4, 16),
}


def resolve_cache_batch_sizes(overrides):
    profile = str(overrides.get("preprocess_memory_profile") or "auto").strip().lower()
    defaults = PROFILES.get(profile.replace("-", "_"), PROFILES["auto"])
    values = []
    for key, default in zip(
        ("preprocess_vae_cache_batch_size", "preprocess_text_cache_batch_size"),
        defaults,
        strict=True,
    ):
        value = overrides.get(key)
        if value is None or str(value).strip().lower() in {"", "auto"}:
            values.append(default)
        else:
            try:
                parsed = int(str(value).strip())
            except (TypeError, ValueError):
                raise ValueError(
                    f"{key} must be 'auto' or a positive integer"
                ) from None
            if parsed < 1:
                raise ValueError(f"{key} must be 'auto' or a positive integer")
            values.append(parsed)
    return tuple(values)
