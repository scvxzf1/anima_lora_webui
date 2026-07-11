"""Tests for cache reuse policy parsing and schema registration."""

from __future__ import annotations

from library.cache_pool.policy import parse_cache_reuse_policy
from library.config import schema as config_schema


def test_defaults_all_reuse_light() -> None:
    p = parse_cache_reuse_policy({})
    assert p.reuse_dataset_cache_copy is True
    assert p.reuse_vae_latents is True
    assert p.reuse_text_encoder_cache is True
    assert p.fingerprint_mode == "light"
    assert p.force_rebuild is False


def test_parse_overrides() -> None:
    p = parse_cache_reuse_policy(
        {
            "reuse_dataset_cache_copy": False,
            "reuse_vae_latents": False,
            "cache_fingerprint_mode": "content",
            "force_rebuild_preprocess_cache": True,
        }
    )
    assert p.reuse_dataset_cache_copy is False
    assert p.reuse_vae_latents is False
    assert p.reuse_text_encoder_cache is True
    assert p.fingerprint_mode == "content"
    assert p.force_rebuild is True


def test_schema_registers_cache_reuse_keys() -> None:
    # Ensure extras populated
    if not config_schema.CONFIG_SCHEMA:
        # populate via train parser path used elsewhere
        try:
            from library.config.schema import populate_schema
            import train

            parser = train.setup_parser()
            populate_schema(parser)
        except Exception:
            pass
    # Manual keys should exist after module import of schema extras
    from library.config import schema as sch

    # Force re-import side effects: call known registration by importing train_util path
    # Directly ensure keys via ensure function if we add one; else check after loading extras.
    keys = {
        "reuse_dataset_cache_copy",
        "reuse_vae_latents",
        "reuse_text_encoder_cache",
        "cache_fingerprint_mode",
        "force_rebuild_preprocess_cache",
    }
    # Register by importing schema and calling ensure helper
    from library.config.schema import ensure_cache_reuse_schema_keys

    ensure_cache_reuse_schema_keys()
    for key in keys:
        assert key in sch.CONFIG_SCHEMA, key
        assert sch.CONFIG_SCHEMA[key].name == key
