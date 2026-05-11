# DirectEdit smart-edit integration plan

Make DirectEdit's edit-instruction handling do replacement and removal (not just
appending), with the auto-replace driven by Qwen3 text-encoder geometry rather
than a hand-curated tag-families YAML.

## TL;DR

- **Detection**: Qwen3 last-non-padding-token cosine. Best-effort. Probe scored 6/9.
- **Execution**: slot-level surgery on `crossattn_emb`. Probe scored 10/10 clean diff spans.
- **Dispatcher**: default APPEND, explicit `-X` / `no X` for REMOVE, last-token similarity for REPLACE (only when both top-1 sim and gap exceed thresholds).
- **Failure mode is graceful**: when detection is uncertain, fall through to APPEND. We never silently replace the wrong tag.

## Decisions and the reasoning behind them

| Decision | Why |
|---|---|
| Drop YAML tag-families. | Brittle, never complete; can't handle out-of-vocab concepts. |
| Use last-non-padding-token pool, not mean pool. | Probe: mean-pool 4/9, last-pool 6/9. Decoder-only LMs concentrate phrase semantics in the final token. Common tags ("blue eyes", "indoors") dominate mean-pool similarity. |
| Threshold + gap gating on detection. | Probe shows true-conflict cosine spans ~0.75–0.98, no-conflict spans ~0.78–0.88 — overlapping. A single threshold can't separate them; require BOTH high absolute sim AND a notable gap to second-best before triggering REPLACE. |
| Slot-level surgery on crossattn_emb (not full re-encode). | Probe: T5 tokenization produces a clean contiguous diff span for every edit kind (replace/remove/add). Surgery preserves untouched slots from ψ_src's encoding; only the diff range is transplanted from ψ_tar's encoding. |
| Default to APPEND when uncertain. | Honest failure mode. The current code already appends; this preserves baseline behavior when detection isn't confident. |
| Skip the LLM-dispatcher option for now. | Adds latency + dep + hallucination. Revisit only if last-token + slot surgery isn't enough in practice. |

## Architecture

### New: `library/inference/edit_dispatcher.py` (~150 LOC)

```python
@dataclass
class EditPlan:
    tar_caption: str
    intent: Literal["append", "remove", "replace"]
    detected_conflict_tag: str | None   # for "replace"
    detection_top1_sim: float | None
    detection_gap: float | None         # top1 - top2

def derive_target_caption(
    src_caption: str,
    edit_instruction: str,
    *,
    text_encoder,
    tokenize_strategy,
    encoding_strategy,
    device: torch.device,
    replace_threshold: float = 0.92,   # top1 cosine must exceed this
    replace_gap: float = 0.04,          # top1 - top2 must exceed this
) -> EditPlan
```

Behavior:

1. Parse `edit_instruction` for explicit syntax:
   - `-X` or `no X` → REMOVE. Drop the literal tag from `src_caption` (case-insensitive, leading-space-tolerant).
   - Else → continue to detection.
2. Detection:
   - Split `src_caption` into tags by `,`.
   - Encode `[edit_instruction, *src_tags]` through Qwen3, last-non-padding-token pool.
   - Cosine to each src tag; rank.
   - If `top1_sim > replace_threshold` AND `(top1_sim − top2_sim) > replace_gap`: REPLACE — produce `tar_caption` by string-level substitution of the top-1 tag with `edit_instruction`.
   - Else: APPEND — `tar_caption = src_caption + ", " + edit_instruction`.
3. Return the plan with diagnostics so callers can log which branch fired.

Thresholds tuned against the probe set (`scripts/probes/edit_nearest_tag.py`); should be revisited once we have a larger labeled set.

### New: `library/inference/directedit_splice.py` (~80 LOC)

```python
def find_t5_diff_span(
    src_ids: list[int], tar_ids: list[int], pad_id: int,
) -> tuple[int, int, int]:
    """Longest common prefix + longest common suffix on trimmed token sequences.
    Returns (common_start, src_end, tar_end)."""

def splice_crossattn_emb(
    *,
    crossattn_emb_src: torch.Tensor,   # (1, 512, D)
    crossattn_emb_tar: torch.Tensor,   # (1, 512, D)
    t5_ids_src: torch.Tensor,           # (1, 512)
    t5_ids_tar: torch.Tensor,           # (1, 512)
    pad_id: int,
) -> torch.Tensor:
    """Build edit conditioning by keeping src slots outside the diff span and
    overwriting the span with tar slots. Re-pads to 512 with zeros."""
```

Reused logic already prototyped in `scripts/probes/edit_slot_alignment.py`.

### Wiring changes

| File | Change |
|---|---|
| `scripts/experimental_tasks/inference.py:263-265` (`cmd_test_directedit`) | Replace `tar_caption = f"{src_caption}, {edit_prompt}"` with `derive_target_caption(...)`. Log the chosen intent. Pass `--prompt_src` AND `--prompt_tar` to `scripts/edit.py` as today; surgery happens downstream. |
| `scripts/edit.py:350-357` (`_load_embed_variants`) | Add `--use_slot_surgery` flag. When set: encode ψ_src AND ψ_tar fully through Qwen3 + LLMAdapter, call `splice_crossattn_emb(...)`, pass the spliced tensor into `invert()` / `edit_forward()` as `embed_tar`. |
| `library/inference/directedit.py` | No API change. Continues to accept pre-encoded `embed_src` / `embed_tar`; the splicing is upstream. |
| `custom_nodes/comfyui-anima-directedit/nodes.py:305` | Same dispatcher call. Add input sockets for `replace_threshold`, `replace_gap`, `use_slot_surgery` (with sensible defaults so basic users don't need to touch them). Bundle `library/inference/edit_dispatcher.py` + `directedit_splice.py` into `_vendor/` via `make vendor-sync`. |

## Phases

1. **Library: dispatcher** — `edit_dispatcher.py` + unit tests reusing the 11 probe cases as a regression suite. ~150 LOC.
2. **Library: splice** — `directedit_splice.py`. ~80 LOC.
3. **CLI: `scripts/edit.py`** — `--edit_instruction` flag (sugar for "derive ψ_tar from ψ_src + edit") and `--use_slot_surgery` flag. ~50 LOC.
4. **Task wrapper: `scripts/experimental_tasks/inference.py`** — call dispatcher, surface logs. ~20 LOC change.
5. **ComfyUI node: `custom_nodes/comfyui-anima-directedit/nodes.py`** — dispatcher + optional surgery. ~30 LOC change. Refresh `_vendor/` with `make vendor-sync`.
6. **Empirical validation** — bench-style script in `bench/directedit/` (or `scripts/probes/`): run a fixed set of reference images × edit instructions × {current append-only, dispatcher-only string-level, dispatcher + slot surgery}; subjective image grid for review.
7. **Docs** — update `docs/experimental/directedit_editing_v3.md` with the new edit-instruction syntax and dispatcher behavior. Add a "failure modes" section honestly listing the 3/9 probe misses.

## Open questions / risks

- **Threshold calibration is undersized.** 11 hand-written probe cases is enough to set sane defaults but not enough to trust the operating point. Phase 6 should expand to ~50 cases including failures from real use (twintails-vs-long-hair, sitting-vs-standing — the probe's known misses).
- **Cross-attn drift outside the diff span.** The LLM Adapter is cross-attention, so even slots OUTSIDE the diff range can have slightly different values between ψ_src's and ψ_tar's encodings. We're keeping ψ_src's slots there. Whether that introduces visible artifacts is empirical — Phase 6 must compare slot-surgery vs full-re-encode head-to-head, not just both vs the current baseline.
- **T5 tokenizer leading-space quirk for the REPLACE substitution.** When we do string-level replacement of "medium breasts" with "large breasts" inside ψ_src, T5 retokenization should produce the same prefix/suffix; the slot-alignment probe confirms this for our cases but is not exhaustive. Edge case: a tag at the very start of the caption (no leading comma) might tokenize differently.
- **REMOVE syntax disambiguation.** `no X` is ambiguous — could be a removal directive OR could literally be the user wanting to add a tag like "no shoes" (which is a valid danbooru tag). Initial heuristic: `-X` is unambiguous removal; `no X` is only treated as removal if `X` matches a tag in ψ_src exactly. Otherwise treat as a literal add.
- **The 3/9 detection misses are not random.** Probe misses are: "twintails"/"long blonde hair" (no surface word overlap), "standing"/"sitting" (semantic opposites but cosine drops them), "large breasts" with "breast tattoo" present (spurious match risk on a different tag). All three are real failure modes; APPEND-fallback is the safety net.
- **What does the dispatcher say in logs?** Should print one line: `[dispatcher] intent=REPLACE src_tag='medium breasts' top1=0.96 gap=0.12` so users can audit and tune.

## Probe-case regression set

Reused from `scripts/probes/edit_nearest_tag.py` as the initial test suite. Last-pool results from the run:

| # | ψ_src | edit | expected intent | mean | last |
|---|---|---|---|---|---|
| 0 | …medium breasts… | large breasts | REPLACE medium breasts | OK | OK |
| 1 | …short hair… | long hair | REPLACE short hair | OK | OK |
| 2 | …blonde hair… | red hair | REPLACE blonde hair | MISS | OK |
| 3 | …medium breasts… | huge breasts | REPLACE medium breasts | OK | OK |
| 4 | …sad… | happy | REPLACE sad | MISS | OK |
| 5 | …long blonde hair… | twintails | REPLACE long blonde hair | MISS | MISS |
| 6 | …sitting… | standing | REPLACE sitting | MISS | MISS |
| 7 | …(no holding tag)… | holding sword | APPEND | OK | OK |
| 8 | …(no ears tag)… | cat ears | APPEND | OK | OK |
| 9 | …breast tattoo… | large breasts | (spurious-match risk) | MISS | MISS |
| 10 | …hair ornament… | hair ornament | REPLACE (self) / removal sanity | OK | OK |

Last-pool 6/9; the 3 misses fall back to APPEND, which is benign.

## Explicitly out of scope (deferred)

- **LLM dispatcher.** Falls into the toolbox if detection accuracy proves insufficient after Phase 6.
- **Post-LLM-Adapter (crossattn_emb) similarity probe.** Last-token-pool on Qwen3 is good enough; the marginal lift from the post-adapter space isn't worth loading the DiT for detection.
- **SEGA-style score-space guidance.** Different mechanic; could compose later as an independent feature.
- **Continuous-slider style edits ("30% larger").** Out of scope; APPEND/REMOVE/REPLACE only.
- **General paraphrasing / natural-language edit rewriting.** Out of scope; edit-instruction is treated as a single tag or short tag phrase.
