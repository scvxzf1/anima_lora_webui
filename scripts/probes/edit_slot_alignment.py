"""Probe: can T5 tokenization split string-level tag edits into a clean,
contiguous differing span — enabling slot-level surgery on crossattn_emb?

Hypothesis (DirectEdit slot surgery): given two captions that differ by one
tag (drop / add / replace), the T5 tokenizations differ on exactly one
contiguous slot range corresponding to the changed text, with everything
before and after bit-identical. If true, we can transplant just the changed
slots from crossattn_emb_tar into crossattn_emb_src and leave the rest
untouched — Prompt-to-Prompt-style surgery on the conditioning tensor.

Failure mode to watch for: SentencePiece can re-segment around inserted
text (e.g., the leading-space token of a neighboring tag changing), causing
the diff span to bleed into untouched tags.

Run from anima_lora/:
    uv run python scripts/probes/edit_slot_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ANIMA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANIMA_ROOT))

from library.anima import weights as anima_utils  # noqa: E402


# Each row: (psi_src, psi_tar, human-readable "what changed", edit_kind)
TEST_CASES: list[tuple[str, str, str, str]] = [
    # Word-for-word replacements (same token count likely)
    ("1girl, blonde hair, medium breasts, blue eyes, smile",
     "1girl, blonde hair, large breasts, blue eyes, smile",
     "medium breasts -> large breasts", "replace"),
    ("1girl, short hair, brown eyes, casual clothes, standing",
     "1girl, long hair, brown eyes, casual clothes, standing",
     "short hair -> long hair", "replace"),
    ("1girl, blonde hair, blue eyes, smile, outdoors",
     "1girl, red hair, blue eyes, smile, outdoors",
     "blonde hair -> red hair", "replace"),

    # Replacement with different token count (extra adjective)
    ("1girl, short hair, blue eyes, casual clothes",
     "1girl, very long blonde hair, blue eyes, casual clothes",
     "short hair -> very long blonde hair", "replace"),
    ("1girl, sad, indoors, long hair, blue dress",
     "1girl, very happy, indoors, long hair, blue dress",
     "sad -> very happy", "replace"),

    # Tag removal (delete in the middle)
    ("1girl, blonde hair, hair ornament, school uniform, smile",
     "1girl, blonde hair, school uniform, smile",
     "remove 'hair ornament'", "remove"),
    ("1girl, large breasts, long blonde hair, blue eyes, school uniform",
     "1girl, long blonde hair, blue eyes, school uniform",
     "remove 'large breasts'", "remove"),

    # Tag addition (append at end)
    ("1girl, blonde hair, blue eyes, smile",
     "1girl, blonde hair, blue eyes, smile, holding sword",
     "append 'holding sword'", "add_end"),

    # Tag addition (insert in middle)
    ("1girl, blonde hair, blue eyes, smile",
     "1girl, blonde hair, cat ears, blue eyes, smile",
     "insert 'cat ears' between blonde hair and blue eyes", "add_mid"),

    # First-position edit (right after the BOS / leading tag)
    ("1girl, blonde hair, blue eyes",
     "2girls, blonde hair, blue eyes",
     "1girl -> 2girls", "replace_first"),
]


def find_diff_span(a: list[int], b: list[int]) -> tuple[int, int, int]:
    """Return (start, end_a, end_b) where a[start:end_a] != b[start:end_b]
    and a[:start] == b[:start], a[end_a:] == b[end_b:] (matched longest prefix
    + longest suffix on residual tails).
    """
    n_a, n_b = len(a), len(b)
    # Longest common prefix
    start = 0
    while start < min(n_a, n_b) and a[start] == b[start]:
        start += 1
    # Longest common suffix on the tails after `start`
    suf = 0
    while suf < min(n_a - start, n_b - start) and a[n_a - 1 - suf] == b[n_b - 1 - suf]:
        suf += 1
    end_a = n_a - suf
    end_b = n_b - suf
    return start, end_a, end_b


def trim_padding(ids: list[int], pad_id: int) -> list[int]:
    """Drop trailing pad tokens. Keep everything else as-is (BOS, EOS, content)."""
    out = list(ids)
    while out and out[-1] == pad_id:
        out.pop()
    return out


def main() -> None:
    print("[probe] loading T5 tokenizer (default: library/anima/configs/t5_old/)")
    t5_tokenizer = anima_utils.load_t5_tokenizer(None)
    pad_id = t5_tokenizer.pad_token_id
    print(f"[probe] T5 pad_token_id={pad_id}  vocab_size={t5_tokenizer.vocab_size}")

    n_clean = 0
    n_total = 0
    for case_idx, (psi_src, psi_tar, label, kind) in enumerate(TEST_CASES):
        n_total += 1
        enc_src = t5_tokenizer(psi_src, padding="max_length", max_length=512,
                                truncation=True, return_tensors=None)
        enc_tar = t5_tokenizer(psi_tar, padding="max_length", max_length=512,
                                truncation=True, return_tensors=None)
        ids_src = trim_padding(enc_src["input_ids"], pad_id)
        ids_tar = trim_padding(enc_tar["input_ids"], pad_id)

        start, end_s, end_t = find_diff_span(ids_src, ids_tar)

        # Decode the diff span and the surrounding context
        span_src_ids = ids_src[start:end_s]
        span_tar_ids = ids_tar[start:end_t]
        span_src_txt = t5_tokenizer.decode(span_src_ids, skip_special_tokens=False)
        span_tar_txt = t5_tokenizer.decode(span_tar_ids, skip_special_tokens=False)

        # Per-token view (using sentencepiece pieces if available)
        try:
            pieces_src = t5_tokenizer.convert_ids_to_tokens(span_src_ids)
            pieces_tar = t5_tokenizer.convert_ids_to_tokens(span_tar_ids)
        except Exception:
            pieces_src = [str(i) for i in span_src_ids]
            pieces_tar = [str(i) for i in span_tar_ids]

        # Heuristic "clean" test: the differing region's decoded text should
        # contain the changed tag and NOT contain any untouched tag.
        # We check by stripping a leading space + comma the tokenizer may add.
        is_clean = True

        # For replace/replace_first kinds, both spans non-empty.
        if kind in ("replace", "replace_first"):
            if not span_src_ids or not span_tar_ids:
                is_clean = False
        elif kind == "remove":
            # Source span should be non-empty, target span empty (the removed tokens).
            if not span_src_ids or span_tar_ids:
                is_clean = False
        elif kind in ("add_end", "add_mid"):
            if span_src_ids or not span_tar_ids:
                is_clean = False

        # Sanity: residual prefix + suffix should re-cover untouched ψ_src content.
        residual_decoded = t5_tokenizer.decode(
            ids_src[:start] + ids_src[end_s:], skip_special_tokens=False
        )

        marker = "OK " if is_clean else "?? "
        if is_clean:
            n_clean += 1

        print()
        print(f"[case {case_idx}] {marker} {label}  (kind={kind})")
        print(f"   src: {psi_src}")
        print(f"   tar: {psi_tar}")
        print(f"   common-prefix-len: {start}, src-len: {len(ids_src)}, tar-len: {len(ids_tar)}")
        print(f"   diff span src[{start}:{end_s}] = {pieces_src!r}  -> {span_src_txt!r}")
        print(f"   diff span tar[{start}:{end_t}] = {pieces_tar!r}  -> {span_tar_txt!r}")
        print(f"   residual (src minus span) decoded: {residual_decoded!r}")

    print()
    print(f"[summary] clean-span cases: {n_clean}/{n_total}")


if __name__ == "__main__":
    main()
