"""Sample-prompt token counts feeding the torch.compile budget.

A training run compiled for the buckets the dataset populates (e.g. the 1024
tier's 4032/4200 range) crashes mid-training when a sample prompt requests
1024x1536 — 6144 tokens, outside the dynamic-seq mark_dynamic range. Sampling
now widens the compiled range instead (``sample_images`` →
``_ensure_sample_compile_range`` → ``ensure_training_compile_seq_range``);
these tests lock the pure count/snap math and the bridge that feeds it.
"""

from pathlib import Path

from library.datasets.buckets import (
    snap_sample_size,
    token_counts_for_sample_prompts,
)
from library.runtime.token_counts import pixel_bucket_token_counts

ANIMA_TRAINING = Path(__file__).resolve().parents[1] / "library" / "anima" / "training.py"


def test_out_of_range_sample_resolution():
    # 1024x1536 → (1024//16)*(1536//16) = 64*96 = 6144, the count that falls
    # outside a (4032, 4200) compiled range.
    counts = token_counts_for_sample_prompts([{"width": 1024, "height": 1536}])
    assert counts == {6144}


def test_union_with_bucket_counts_widens_range():
    # The startup budget unions bucket counts with sample counts; the 1024-tier
    # buckets (4032/4200 families) plus the 1024x1536 sample must yield a range
    # covering 6144. patch_spatial=2 × vae 8 = the same //16 stride.
    bucket_counts = pixel_bucket_token_counts(
        [(768, 1344), (800, 1344), (896, 1200), (1344, 768), (1344, 800)],
        patch_spatial=2,
    )
    assert bucket_counts == {4032, 4200}
    merged = bucket_counts | token_counts_for_sample_prompts(
        [{"width": 1024, "height": 1536}]
    )
    assert (min(merged), max(merged)) == (4032, 6144)
    assert len(merged) == 3


def test_defaults_match_sample_inference():
    # _sample_image_inference defaults width/height to 512 when a prompt omits
    # --w/--h; the budget must count the same resolution.
    assert token_counts_for_sample_prompts([{"prompt": "1girl"}]) == {
        (512 // 16) * (512 // 16)
    }


def test_snap_matches_inference_formula():
    # snap_sample_size is the shared definition of _sample_image_inference's
    # pre-sampling snap: dim → max(64, dim - dim % 16).
    assert snap_sample_size(1000, 1000) == (992, 992)
    assert snap_sample_size(1024, 1536) == (1024, 1536)
    assert snap_sample_size(10, 30) == (64, 64)
    counts = token_counts_for_sample_prompts([{"width": 1000, "height": 1000}])
    assert counts == {(992 // 16) * (992 // 16)}


def test_duplicate_resolutions_dedup():
    prompts = [
        {"width": 1024, "height": 1536},
        {"width": 1536, "height": 1024},  # same token count, mirrored
        {"width": 1024, "height": 1536, "prompt": "another"},
    ]
    assert token_counts_for_sample_prompts(prompts) == {6144}


def test_malformed_prompt_entries_are_skipped():
    # Prompt files are user-edited between sampling events; a bad entry must
    # not take down the run before the preview even starts.
    prompts = [
        {"width": "wide", "height": 512},
        "not a dict",
        {"width": 1024, "height": 1536},
    ]
    assert token_counts_for_sample_prompts(prompts) == {6144}


def test_sample_compile_range_bridge_uses_current_prompt_sizes(monkeypatch):
    import library.anima.training as anima_training
    import library.runtime.harness as harness

    captured = {}

    def fake_ensure(dit, network, seq_lens, *, logger):
        captured["dit"] = dit
        captured["network"] = network
        captured["seq_lens"] = set(seq_lens)
        captured["logger"] = logger
        return True

    monkeypatch.setattr(harness, "ensure_training_compile_seq_range", fake_ensure)

    dit = object()
    network = object()
    changed = anima_training._ensure_sample_compile_range(
        dit,
        network,
        [{"width": 768, "height": 1152}],
    )

    assert changed is True
    assert captured["dit"] is dit
    assert captured["network"] is network
    assert captured["seq_lens"] == {3456}


def test_range_widening_precedes_the_block_swap_pause():
    """pause_block_swap() zeroes blocks_to_swap; a recompile after it would see
    every block as resident and compile the swapped tail too, permanently
    widening compile_block_scope="resident" for the rest of the run."""
    source = ANIMA_TRAINING.read_text(encoding="utf-8")
    body = source[
        source.index("def sample_images(") : source.index("def _sample_image_inference(")
    ]

    assert body.index("_ensure_sample_compile_range(dit, net, prompts)") < body.index(
        "dit.pause_block_swap()"
    )
    assert body.index("_ensure_sample_compile_range(dit, net, prompts)") < body.index(
        "dit.switch_block_swap_for_inference()"
    )
