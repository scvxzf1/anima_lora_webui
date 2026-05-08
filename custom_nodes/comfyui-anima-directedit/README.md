# Anima DirectEdit (ComfyUI)

A single ComfyUI node that edits an image by adding tag(s) to its caption. Drop in an image, type `glasses` (or `double peace`, or `school_uniform`, …), and out comes the same image with that change applied — backgrounds, composition, and unchanged subject details preserved.

Built on **DirectEdit** (Yang & Ye, [arXiv:2605.02417](https://arxiv.org/abs/2605.02417v1)) — a training-free flow-inversion editor. Reference implementation: [Tr1stesse/DirectEdit](https://github.com/Tr1stesse/DirectEdit).

This node ports that idea to the Anima (DiT, flow-matching) model and bundles **AnimaTagger** so the user only has to think about *what to add*, not *what was originally there*.

## What the node does

```
IMAGE ─► AnimaTagger ─► ψ_src ──┐
                                 ├─► ψ_tar = ψ_src + ", " + edit_text
              edit_text ─────────┘                │
                                                  ▼
                       ┌─────────────────────────────────────┐
                       │ DirectEdit                          │
                       │   1. invert(image, ψ_src)           │
                       │      ─► z_inv, Δz                   │
                       │   2. edit_forward(z_inv[0], Δz, ψ_tar) │
                       │      ─► z_edit                      │
                       └─────────────────────────────────────┘
                                                  │
                                                  ▼
                                          edited IMAGE
```

Two passes through the DiT:

1. **Inversion** (clean → noise): step backward through the same Euler ODE the generator runs forward, querying v_θ at each step's input. Record per-step residuals `Δz_i = z_inv[i+1] − z_inv[i]` — these are the "anchor" the paper uses to make reconstruction bit-exact.
2. **Editing** (noise → clean): standard generation loop, but every model call is queried at `z[i] + Δz[i]` instead of `z[i]`. The cross-attn prompt is the edit target ψ_tar; the residual Δz pins the trajectory to the source.

Result: regions the prompt *doesn't change* stay locked to the source; regions it *does* change get re-rendered under the target prompt.

## Install

Drop `custom_nodes/comfyui-anima-directedit/` (this directory) into your ComfyUI `custom_nodes/`, restart ComfyUI. The node appears as **Anima DirectEdit** in the `anima` category.

The node imports from the parent `anima_lora/` repo (it lives at `anima_lora/custom_nodes/comfyui-anima-directedit/`), so keep the directory inside the repo or symlink in such a way that `Path(__file__).resolve().parents[2]` lands on the `anima_lora/` root.

## Inputs

| Input | Type | Notes |
|-------|------|-------|
| `image` | IMAGE | Source image. Auto-snapped to the closest `CONSTANT_TOKEN_BUCKETS` aspect ratio. |
| `edit_text` | STRING (multiline) | Tag(s) to add. `ψ_tar = ψ_src + ", " + edit_text`. Empty → reconstruction sanity check. |
| `dit` | dropdown | Anima DiT checkpoint (`anima-preview3-base.safetensors`). |
| `text_encoder` | dropdown | Anima text encoder (Qwen3 06B). |
| `vae` | dropdown | Qwen Image VAE. |
| `tagger_dir` | STRING | AnimaTagger checkpoint directory. Relative to `anima_lora/` or absolute. |
| `negative_prompt` | STRING | CFG negative for the edit pass. Default `"worst quality"`. |
| `infer_steps` | INT | Both inversion and edit step count. Default 28. |
| `flow_shift` | FLOAT | Sigma-shift schedule. Default 1.0 (Anima preview3 standard). |
| `guidance_scale` | FLOAT | CFG for the edit (target) pass. Default 4.0. |
| `invert_guidance` | FLOAT | CFG during inversion. Default 1.0 (no CFG). |
| `seed` | INT | Reserved for future stochastic hooks; current pipeline is deterministic. |
| `prompt_src_override` | STRING (optional) | Replace the tagger's caption with your own ψ_src. Useful when the source is an Anima-generated image and you already know the original prompt. |

## Outputs

| Output | Type | Notes |
|--------|------|-------|
| `image` | IMAGE | Edited image, decoded internally (the node owns its own VAE). |
| `prompt_src` | STRING | What the tagger derived (or `prompt_src_override` if set). Useful for debugging — if the edit fails, check whether ψ_src actually describes the source. |
| `prompt_tar` | STRING | The full target caption fed to the edit pass. |

## Self-contained design (no MODEL/CLIP/VAE wires)

The node loads its own DiT / text encoder / VAE / tagger from disk per invocation rather than consuming the canvas's `MODEL` / `CLIP` / `VAE` handles. Two reasons:

1. **Forward signature mismatch.** The DirectEdit primitives target `library/anima/models.py::Anima`, whose `forward(x, t, context, padding_mask=...)` expects an already-T5-projected, 512-padded crossattn embed. ComfyUI's `comfy.ldm.anima.model.Anima` instead takes raw Qwen3 hidden states + `t5xxl_ids` in kwargs and runs the LLM adapter internally. Adapting the inversion + edit-forward loops to ComfyUI's `apply_model` API is a real model-integration project, not a packaging task.
2. **CONDITIONING shape.** `CLIPTextEncode → CONDITIONING` for Anima carries pre-adapter Qwen3 hidden states. `_v_pred` needs the post-adapter, padded embed. We'd have to manually drive the adapter to extract that.

Cold-loading every run matches `scripts/edit.py` semantics. Caching across invocations is a clear v1 improvement once you've validated the output.

## Usage

```
[Load Image] ──► [Anima DirectEdit] ──► [Save Image]
                        │
                        edit_text: "double peace"
                        dit: anima-preview3-base.safetensors
                        text_encoder: qwen_3_06b_base.safetensors
                        vae: qwen_image_vae.safetensors
                        tagger_dir: models/captioners/anima-tagger-v1
```

CLI equivalent (for reference):

```bash
make exp-test-directedit PROMPT='double peace'
```

## When to use `prompt_src_override`

The tagger is great for *external* images (web rips, screenshots) where you have no recorded prompt. For images you generated with Anima yourself, the original prompt is already a much better ψ_src than anything a tagger can recover. Paste it into `prompt_src_override` and the node skips the tagger entirely.

If the tagger checkpoint isn't on disk yet (training pending), `prompt_src_override` is also the only way to use the node — leave it empty and the node will raise on `AnimaTagger.__init__`.

## Caveats (v0)

- **No V-injection / no mask blending.** v1 of the underlying DirectEdit primitive (`library/inference/directedit.py`) is the paper's pure ΔZ-anchored edit at `t_inj=0, mask=None`. V-injection and background-lock are deferred to v2.
- **Inversion runs at `invert_guidance=1.0`** by default (no CFG). Raise only if you need the inverted noise to match a high-CFG generation seed.
- **Cold-loads ~7 GB of weights per invocation.** Don't hammer the node — each call pays the full DiT/TE/VAE/tagger load cost. A module-level cache would help; intentionally not implemented in v0 to keep the code path identical to `scripts/edit.py`.
- **Single-frame only.** Anima's qwen-image VAE is a video VAE that this pipeline drives at `T=1`. If you wire in a video-shaped IMAGE, only the first frame is processed.

## Files

| File | Role |
|------|------|
| `nodes.py` | The `AnimaDirectEdit` node — full pipeline (tagger → encode → invert → edit → decode). |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. |
| `pyproject.toml` | ComfyUI Registry metadata. |

## References

- **DirectEdit paper.** Yang & Ye, "Direct flow-inversion image editing for rectified flow models." [arXiv:2605.02417](https://arxiv.org/abs/2605.02417v1).
- **Reference implementation.** [Tr1stesse/DirectEdit](https://github.com/Tr1stesse/DirectEdit) — original PyTorch reference, source for the inversion/edit-forward step rules ported here.
- **AnimaTagger.** `docs/experimental/anima_tagger.md` (architecture) and `docs/experimental/directedit_editing_v3.md` (integration) in the parent repo.
- **Anima editing pipeline.** `scripts/edit.py` (CLI) and `library/inference/directedit.py` (primitives).
