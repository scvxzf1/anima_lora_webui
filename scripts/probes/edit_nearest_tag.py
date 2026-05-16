"""Probe: does the Qwen3 text-encoder's geometry reliably identify the
mutually-exclusive source tag that a DirectEdit edit phrase should replace?

Hypothesis (DirectEdit option B): given a comma-separated ψ_src caption and a
short edit phrase, the source tag with highest cosine similarity to the edit
phrase (in mean-pooled Qwen3 last_hidden_state space) is the tag the edit
should drop. If true, we can do tag-family conflict resolution without
maintaining a hand-curated families YAML.

Run from anima_lora/:
    uv run python scripts/probes/edit_nearest_tag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ANIMA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANIMA_ROOT))

from library.anima import weights as anima_utils  # noqa: E402
from library.anima.strategy import (  # noqa: E402
    AnimaTextEncodingStrategy,
    AnimaTokenizeStrategy,
)


QWEN3_PATH = str(ANIMA_ROOT / "models/text_encoders/qwen_3_06b_base.safetensors")


# (psi_src, edit_phrase, "what we expect the nearest tag to be" or None)
TEST_CASES: list[tuple[str, str, str | None]] = [
    # Easy mutually-exclusive replacements
    ("1girl, blonde hair, medium breasts, blue eyes, school uniform, smile",
     "large breasts",
     "medium breasts"),
    ("1girl, short hair, brown eyes, casual clothes, standing",
     "long hair",
     "short hair"),
    ("1girl, blonde hair, blue eyes, smile, outdoors",
     "red hair",
     "blonde hair"),
    ("1girl, medium breasts, blonde hair, school uniform, smile",
     "huge breasts",
     "medium breasts"),
    ("1girl, sad, indoors, long hair, blue dress",
     "happy",
     "sad"),

    # Trickier — multiple plausible "neighbors"
    ("1girl, large breasts, long blonde hair, blue eyes, school uniform",
     "twintails",
     "long blonde hair"),  # twintails is a hairstyle, conflicts with "long hair"
    ("1girl, sitting, indoors, casual clothes, holding cup",
     "standing",
     "sitting"),  # pose conflict

    # Pure addition — edit doesn't conflict with any source tag.
    # We want the top similarity to be NOTABLY LOWER than the easy cases above.
    ("1girl, blonde hair, blue eyes, school uniform, standing, outdoors",
     "holding sword",
     None),
    ("1girl, short hair, smile, indoors, casual clothes",
     "cat ears",
     None),

    # Spurious-match risk
    ("1girl, blonde hair, blue eyes, breast tattoo, school uniform",
     "large breasts",
     "breast tattoo"),  # we WORRY this happens; ideally similarity is mediocre

    # Removal-style edit (the user proposed "-hair ornaments" / "no hair ornaments").
    # For the nearest-neighbor probe we just check that the literal tag matches itself.
    ("1girl, blonde hair, hair ornament, school uniform, smile",
     "hair ornament",
     "hair ornament"),

    # --- Real-caption cases (image_dataset/) ---
    # Trimmed from actual training captions to exercise the probe under
    # realistic tag density and distractor noise.

    # 11036780.txt — asuna (blue archive) swimsuit set; light-brown → red.
    ("sensitive, 1girl, light brown hair, long hair, hair over one eye, "
     "blue eyes, smile, large breasts, swimsuit, bikini, halo, cleavage, "
     "sideboob, underboob",
     "red hair",
     "light brown hair"),

    # 10302865.txt — kikyou (blue archive); short → long under cat-girl tags.
    ("explicit, 1girl, black hair, short hair, cat ears, animal ears, "
     "large breasts, lying, on back, nipples, halo",
     "long hair",
     "short hair"),

    # 11703691.txt — juri (blue archive); both "huge breasts" and
    # "large breasts" present, edit "small breasts". Either is a defensible
    # top-1 — leave expected unset and inspect the ranking.
    ("sensitive, 1girl, pink hair, long hair, huge breasts, large breasts, "
     "maid, smile, blush, purple dress, halo, horns",
     "small breasts",
     None),

    # 11146287.txt — pipkin pippa; pose swap inside a busy caption.
    ("sensitive, 1girl, pink hair, long hair, pink eyes, rabbit ears, "
     "animal ears, sitting, looking at viewer, hair bow",
     "standing",
     "sitting"),

    # 12455572.txt — fujiwara chika; emotion replace.
    ("sensitive, 1girl, pink hair, long hair, blue eyes, school uniform, "
     "smile, open mouth, large breasts, standing",
     "angry",
     "smile"),

    # 7221919.txt — kashima (kancolle); medium → long (harder than short → long
    # since "medium hair" and "long hair" are closer neighbours).
    ("sensitive, 1girl, grey hair, medium hair, blue eyes, maid, smile, "
     "indoors, medium breasts, two side up",
     "long hair",
     "medium hair"),

    # 8526178.txt — shiroko (swimsuit); both "long hair" and "ponytail" present,
    # edit is "twintails". Either is defensible as the hairstyle conflict —
    # log the ranking instead of asserting.
    ("sensitive, 1girl, grey hair, long hair, ponytail, cat ears, aqua eyes, "
     "swimsuit, large breasts, halo",
     "twintails",
     None),

    # Real-style APPEND — edit doesn't conflict with anything in ψ_src.
    ("sensitive, 1girl, pink hair, long hair, blue eyes, school uniform, "
     "smile, large breasts, standing, white background",
     "holding sword",
     None),

    # Spurious-match in a realistic caption: "breast tattoo" sitting alongside
    # "medium breasts"; edit "large breasts" should hit "medium breasts".
    ("sensitive, 1girl, blonde hair, long hair, blue eyes, breast tattoo, "
     "medium breasts, school uniform",
     "large breasts",
     "medium breasts"),

    # Feature-axis isolation: "blue eyes" + "blue dress" present, edit
    # "red eyes" should target the eye tag, not the dress.
    ("sensitive, 1girl, blonde hair, blue eyes, blue dress, smile, "
     "school uniform, long hair, standing",
     "red eyes",
     "blue eyes"),

    # Small-lexical-distance semantic opposite: open ↔ closed mouth.
    ("sensitive, 1girl, pink hair, long hair, blue eyes, open mouth, "
     "school uniform, smile, large breasts",
     "closed mouth",
     "open mouth"),
]


@torch.no_grad()
def encode_phrases(
    phrases: list[str],
    text_encoder,
    tokenize_strategy: AnimaTokenizeStrategy,
    encoding_strategy: AnimaTextEncodingStrategy,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode phrases and return (mean_pooled, last_token_pooled), both (N, D) fp32.

    Qwen3 is decoder-only with causal attention, so the last non-padding token
    summarizes the full phrase; this is typically a stronger sentence embedding
    than the mean over short phrases (which gets dominated by per-token noise).
    """
    tokens = tokenize_strategy.tokenize(phrases)
    prompt_embeds, attn_mask, _t5_ids, _t5_mask = encoding_strategy.encode_tokens(
        tokenize_strategy, [text_encoder], tokens
    )
    # prompt_embeds: (N, 512, D), attn_mask: (N, 512) int
    mask_f = attn_mask.to(prompt_embeds.device).to(prompt_embeds.dtype).unsqueeze(-1)
    summed = (prompt_embeds * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1.0)
    mean_pooled = (summed / counts).float()

    # Last non-padding token index per row.
    lengths = attn_mask.to(prompt_embeds.device).sum(dim=1)  # (N,)
    last_idx = (lengths - 1).clamp(min=0).long()
    last_pooled = prompt_embeds[torch.arange(prompt_embeds.size(0), device=prompt_embeds.device), last_idx].float()

    return mean_pooled, last_pooled


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[probe] device={device} dtype={dtype}")
    print(f"[probe] loading Qwen3 from {QWEN3_PATH}")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        QWEN3_PATH, dtype=dtype, device=str(device)
    )
    text_encoder.eval()
    t5_tokenizer = anima_utils.load_t5_tokenizer(None)

    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    stats = {"mean": [0, 0], "last": [0, 0]}  # [correct, total]

    for case_idx, (psi_src, edit, expected) in enumerate(TEST_CASES):
        src_tags = [t.strip() for t in psi_src.split(",") if t.strip()]
        phrases = [edit] + src_tags
        mean_emb, last_emb = encode_phrases(
            phrases, text_encoder, tokenize_strategy, encoding_strategy, device
        )

        print()
        print(f"[case {case_idx}] edit: '{edit}'  expected: {expected or '(no-conflict)'}")
        print(f"   src: {psi_src}")

        for name, embeds in (("mean", mean_emb), ("last", last_emb)):
            edit_vec = embeds[0:1]
            tag_vecs = embeds[1:]
            sims = F.cosine_similarity(edit_vec, tag_vecs, dim=-1).cpu().tolist()
            ranked = sorted(zip(src_tags, sims), key=lambda x: -x[1])
            top1_tag = ranked[0][0]
            if expected is not None:
                stats[name][1] += 1
                hit = top1_tag == expected
                if hit:
                    stats[name][0] += 1
                tag_flag = "OK" if hit else f"MISS (expected '{expected}')"
            else:
                tag_flag = "no-conflict"
            top4 = "  ".join(f"{s:+.3f} {t}" for t, s in ranked[:4])
            print(f"   [{name}] {tag_flag}")
            print(f"     {top4}")

    print()
    for name, (c, n) in stats.items():
        if n:
            print(f"[summary] {name}-pool top-1 conflict match: {c}/{n} "
                  f"({100.0 * c / n:.0f}%)")


if __name__ == "__main__":
    main()
