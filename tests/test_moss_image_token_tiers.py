"""Static (CPU-only) contracts for MOSS-Image token-budget tiers.

No checkpoints, no CUDA and no dataset are touched here. These tests pin the
bucket tables that configs/methods/moss_image_lora*.toml actually resolve to,
so that

* a resolution tier cannot silently collapse the aspect ladder, and
* a tier variant cannot silently drift away from the base method config.

Thermal / VRAM / s-per-step numbers are measurement, not contract; they live in
docs/findings/moss_image_token_budget_tiers_20261017.md.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS, BucketManager
from library.preprocess.bucket_geometry import select_resize_bucket
from tests.conftest import iter_method_names

REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = REPO_ROOT / "configs" / "methods"
BASE_PATH = METHODS / "moss_image_lora.toml"
TIER_768_PATH = METHODS / "moss_image_lora_t768.toml"
TIER_512_PATH = METHODS / "moss_image_lora_t512.toml"
TECAP_PATH = METHODS / "moss_image_lora_tecap1024.toml"

MOSS_STRIDE = 16  # FLUX.2 VAE /8 then 2x2 packing -> 16 px per image token


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _dataset(cfg: dict) -> dict:
    return cfg["datasets"][0]


def _manager(cfg: dict) -> BucketManager:
    ds = _dataset(cfg)
    manager = BucketManager(
        max_reso=(ds["resolution"], ds["resolution"]),
        min_size=ds["min_bucket_reso"],
        max_size=ds["max_bucket_reso"],
        reso_steps=ds["bucket_reso_steps"],
    )
    manager.make_buckets(constant_token_buckets=True)
    return manager


def _resos(cfg: dict) -> list[tuple[int, int]]:
    return list(_manager(cfg).predefined_resos)


def _tokens(resos) -> set[int]:
    return {(w // MOSS_STRIDE) * (h // MOSS_STRIDE) for w, h in resos}


def _aspects(resos) -> list[float]:
    return [w / h for w, h in resos]


def _flatten(node, prefix: str = "") -> dict:
    flat: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            flat.update(_flatten(value, f"{prefix}.{index}" if prefix else str(index)))
    else:
        flat[prefix] = node
    return flat


def _diff_paths(left: dict, right: dict) -> set[str]:
    a, b = _flatten(left), _flatten(right)
    return {key for key in set(a) | set(b) if a.get(key) != b.get(key)}


# ---------------------------------------------------------------------------
# Base tier: what the shipped default actually resolves to
# ---------------------------------------------------------------------------


def test_base_tier_resolves_to_canonical_constant_token_table():
    cfg = _load(BASE_PATH)
    resos = _resos(cfg)
    assert resos == list(CONSTANT_TOKEN_BUCKETS)
    assert len(resos) == 24
    assert _tokens(resos) == {4032, 4200}


def test_base_tier_min_max_steps_are_inert_on_constant_token_path():
    """resolution == 1024 early-returns before the min/max filter runs.

    This is the documented-but-surprising behaviour the base config comment
    warns about; if the implementation ever starts honouring min/max here, this
    test is the tripwire that forces the comment and the findings doc to change.
    """
    cfg = _load(BASE_PATH)
    loose = _load(BASE_PATH)
    _dataset(loose)["min_bucket_reso"] = 256
    _dataset(loose)["max_bucket_reso"] = 2048
    assert _resos(cfg) == _resos(loose) == list(CONSTANT_TOKEN_BUCKETS)

    generic = BucketManager(
        max_reso=(1024, 1024), min_size=672, max_size=1536, reso_steps=64
    )
    generic.make_buckets(constant_token_buckets=False)
    assert list(generic.predefined_resos) != list(CONSTANT_TOKEN_BUCKETS)


@pytest.mark.parametrize("resolution, expected", [(768, 8), (512, 1)])
def test_changing_resolution_alone_collapses_the_aspect_ladder(resolution, expected):
    """Trap guard: the base min/max do not scale with resolution."""
    cfg = _load(BASE_PATH)
    ds = _dataset(cfg)
    ds["resolution"] = resolution
    ds["bucket_reso_steps"] = 64
    assert len(_resos(cfg)) == expected


# ---------------------------------------------------------------------------
# Scaled tiers: the shipped variants must keep the ladder
# ---------------------------------------------------------------------------


def test_tier_768_keeps_aspect_ladder_at_scaled_area():
    cfg = _load(TIER_768_PATH)
    ds = _dataset(cfg)
    assert (ds["resolution"], ds["min_bucket_reso"], ds["max_bucket_reso"]) == (
        768,
        504,
        1152,
    )
    resos = _resos(cfg)
    assert len(resos) == 14
    tokens = _tokens(resos)
    assert min(tokens) == 2256 and max(tokens) == 2394
    assert max(tokens) / min(tokens) < 1.07  # still one narrow token band
    aspects = _aspects(resos)
    assert min(aspects) <= 0.45 and max(aspects) >= 2.2


def test_tier_512_keeps_aspect_ladder_at_scaled_area():
    cfg = _load(TIER_512_PATH)
    ds = _dataset(cfg)
    assert (ds["resolution"], ds["min_bucket_reso"], ds["max_bucket_reso"]) == (
        512,
        336,
        768,
    )
    resos = _resos(cfg)
    assert len(resos) == 13
    tokens = _tokens(resos)
    assert min(tokens) == 1008 and max(tokens) == 1064
    assert max(tokens) / min(tokens) < 1.07
    aspects = _aspects(resos)
    assert min(aspects) <= 0.45 and max(aspects) >= 2.2


@pytest.mark.parametrize("path", [BASE_PATH, TIER_768_PATH, TIER_512_PATH])
def test_every_tier_bucket_is_16px_aligned(path):
    """MOSS latents demand H,W % 16 == 0 (library/models/moss_image/latent.py)."""
    for width, height in _resos(_load(path)):
        assert width % MOSS_STRIDE == 0, (path.name, width, height)
        assert height % MOSS_STRIDE == 0, (path.name, width, height)


# ---------------------------------------------------------------------------
# Variant hygiene: no silent drift from the base method config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, expected_diff",
    [
        (
            TIER_768_PATH,
            {
                "output_name",
                "datasets.0.resolution",
                "datasets.0.min_bucket_reso",
                "datasets.0.max_bucket_reso",
            },
        ),
        (
            TIER_512_PATH,
            {
                "output_name",
                "datasets.0.resolution",
                "datasets.0.min_bucket_reso",
                "datasets.0.max_bucket_reso",
            },
        ),
        (TECAP_PATH, {"output_name", "max_token_length"}),
    ],
)
def test_tier_variant_diff_whitelist(path, expected_diff):
    assert _diff_paths(_load(BASE_PATH), _load(path)) == expected_diff


def test_text_cap_tier_keeps_image_buckets_identical():
    base_cfg = _load(BASE_PATH)
    cap_cfg = _load(TECAP_PATH)
    assert cap_cfg["max_token_length"] == 1024
    assert base_cfg["max_token_length"] == 4096
    assert _resos(cap_cfg) == _resos(base_cfg)


def test_tiers_are_loadable_as_method_configs():
    from library.config.io import load_method_preset

    names = set(iter_method_names())
    for stem in (
        "moss_image_lora",
        "moss_image_lora_t768",
        "moss_image_lora_t512",
        "moss_image_lora_tecap1024",
    ):
        assert stem in names
        merged = load_method_preset(stem, preset="default")
        assert merged["model_family"] == "moss_image"


# ---------------------------------------------------------------------------
# Documented cost model (kept in sync with the findings doc)
# ---------------------------------------------------------------------------


def _seq(tier_tokens: int, text_tokens: int) -> int:
    return tier_tokens + text_tokens


@pytest.mark.parametrize(
    "text_tokens, ratio_768, ratio_512",
    [
        (128, 0.34, 0.076),
        (512, 0.380, 0.112),
        (1024, 0.428, 0.160),
        (4096, 0.612, 0.387),
    ],
)
def test_documented_attention_cost_ratios(text_tokens, ratio_768, ratio_512):
    """Attention cost is ~ seq^2, with one constant token band per tier."""
    ref = _seq(4200, text_tokens)
    assert (_seq(2394, text_tokens) / ref) ** 2 == pytest.approx(ratio_768, abs=0.01)
    assert (_seq(1064, text_tokens) / ref) ** 2 == pytest.approx(ratio_512, abs=0.01)


@pytest.mark.parametrize("cap, ratio", [(512, 0.32), (1024, 0.40), (2048, 0.57)])
def test_documented_text_cap_ratios(cap, ratio):
    ref = _seq(4200, 4096)
    assert (_seq(4200, cap) / ref) ** 2 == pytest.approx(ratio, abs=0.01)


# ---------------------------------------------------------------------------
# Known limitation, pinned on purpose
# ---------------------------------------------------------------------------


def test_short_side_below_768_escapes_the_1024_token_band():
    """Sources whose short side is under 768 px leave the 4032/4200 band.

    select_resize_bucket() honours no-upscale only while some predefined bucket
    still fits the source; when none fits it falls back to a min_size-floored
    size. Consequences, all measured on the shipped 1024-tier table:

    * 512x512  -> 672x672  (1764 tokens, and UPSCALED despite no-upscale);
    * 640x960  -> 672x960  (2520 tokens, also upscaled);
    * 672x1400 -> 672x1344 (3528 tokens, off-table size, not in the 24 buckets).

    Data-design contract: keep source short sides >= 768 px for the 1024 tier
    if the run is meant to sit in one token band. Documented in
    docs/findings/moss_image_token_budget_tiers_20261017.md; changing this is a
    preprocessing-policy decision, not a bucket-table change.
    """
    cfg = _load(BASE_PATH)

    def bucket_for(width: int, height: int) -> tuple[int, int]:
        return select_resize_bucket(
            _manager(cfg), width, height, enable_bucket=True, bucket_no_upscale=True
        )

    def tokens(reso: tuple[int, int]) -> int:
        return (reso[0] // MOSS_STRIDE) * (reso[1] // MOSS_STRIDE)

    small = bucket_for(512, 512)
    assert small == (672, 672)
    assert tokens(small) == 1764
    assert small[0] > 512  # min_size floor wins over bucket_no_upscale

    middle = bucket_for(672, 1400)
    assert middle == (672, 1344)
    assert middle not in CONSTANT_TOKEN_BUCKETS
    assert tokens(middle) == 3528

    for width, height in ((2048, 2048), (1200, 1600), (900, 1600)):
        assert tokens(bucket_for(width, height)) in {4032, 4200}


# ---------------------------------------------------------------------------
# Tier knobs survive the production config chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, resolution, min_reso, max_reso",
    [(TIER_768_PATH, 768, 504, 1152), (TIER_512_PATH, 512, 336, 768)],
)
def test_tier_dataset_params_reach_the_blueprint(path, resolution, min_reso, max_reso):
    """The sanitizer must not drop the tier keys on the way to DatasetParams."""
    import argparse

    from library.config.loader import BlueprintGenerator, ConfigSanitizer

    ds = dict(_dataset(_load(path)))
    ds["subsets"] = [{"image_dir": "post_image_dataset/resized"}]
    user_config = {"general": {"caption_extension": ".txt"}, "datasets": [ds]}
    args = argparse.Namespace(
        train_batch_size=None,
        debug_dataset=False,
        max_token_length=None,
        prior_loss_weight=1.0,
    )

    blueprint = BlueprintGenerator(ConfigSanitizer(support_dropout=True)).generate(
        user_config, args
    )
    params = blueprint.dataset_group.datasets[0].params

    assert params.resolution == resolution
    assert params.min_bucket_reso == min_reso
    assert params.max_bucket_reso == max_reso
    assert params.bucket_reso_steps == 64
    assert params.enable_bucket is True
    assert params.bucket_no_upscale is True


@pytest.mark.parametrize(
    "path, low, high",
    [(TIER_768_PATH, 2256, 2394), (TIER_512_PATH, 1008, 1064)],
)
def test_common_source_size_lands_inside_the_tier_band(path, low, high):
    """A 1200x1600 source preprocesses into the tier's own token band."""
    cfg = _load(path)
    resos = _resos(cfg)
    bucket = select_resize_bucket(
        _manager(cfg), 1200, 1600, enable_bucket=True, bucket_no_upscale=True
    )
    assert bucket in resos
    tokens = (bucket[0] // MOSS_STRIDE) * (bucket[1] // MOSS_STRIDE)
    assert low <= tokens <= high


def test_base_tier_common_source_lands_in_the_4032_4200_band():
    cfg = _load(BASE_PATH)
    bucket = select_resize_bucket(
        _manager(cfg), 1200, 1600, enable_bucket=True, bucket_no_upscale=True
    )
    assert bucket == (896, 1200)
    assert (bucket[0] // MOSS_STRIDE) * (bucket[1] // MOSS_STRIDE) == 4200


@pytest.mark.parametrize(
    "method, resolution, min_reso, max_reso",
    [
        ("moss_image_lora", 1024, 672, 1536),
        ("moss_image_lora_t768", 768, 504, 1152),
        ("moss_image_lora_t512", 512, 336, 768),
        ("moss_image_lora_tecap1024", 1024, 672, 1536),
    ],
)
def test_tier_dataset_blueprint_resolves_via_method_override(
    method, resolution, min_reso, max_reso
):
    """The real activation path: base blueprint + method TOML override.

    library/config/io.py::load_dataset_config_from_base() harvests
    [general]/[[datasets]] from base.toml and shallow-overrides them with the
    method file's own sections. This is how a tier reaches DatasetParams when
    the run has no explicit --dataset_config.
    """
    from library.config.io import load_dataset_config_from_base

    cfg = load_dataset_config_from_base(method=method)
    dataset = (cfg or {}).get("datasets", [{}])[0]
    assert dataset["resolution"] == resolution
    assert dataset["min_bucket_reso"] == min_reso
    assert dataset["max_bucket_reso"] == max_reso
    assert dataset["bucket_reso_steps"] == 64
    assert dataset["bucket_no_upscale"] is True
