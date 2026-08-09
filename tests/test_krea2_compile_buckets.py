from __future__ import annotations

from scripts.krea2.probe_nf4_compile_buckets import CASES


def test_krea2_constant_bucket_representatives_collapse_to_two_padded_graphs() -> None:
    padded_lengths = {}
    for _label, family, (width, height) in CASES:
        image_tokens = (width // 16) * (height // 16)
        combined = image_tokens + 512
        padded = combined + (-combined % 256)
        padded_lengths.setdefault(family, set()).add(padded)

    assert padded_lengths == {
        "tokens4032": {4608},
        "tokens4200": {4864},
    }
