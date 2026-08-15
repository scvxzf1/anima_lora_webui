# `custom_nodes/comfyui-hydralora/` development rules

This file adds node-specific constraints to the repository-wide [`AGENTS.md`](../../AGENTS.md). User-facing installation, loader behavior, and release history belong in [`README.md`](README.md); this file contains only maintenance invariants.

## Scope

| File | Responsibility |
| --- | --- |
| [`adapter.py`](adapter.py) | Plain LoRA, HydraLoRA, ReFT, and Chimera parsing/application |
| [`chimera.py`](chimera.py) | Chimera metadata, content/frequency routers, single-A/dual-A hooks |
| [`fera.py`](fera.py) | Author-faithful and plan2 stacked-expert FeRA parsing/application |
| [`soft_tokens.py`](soft_tokens.py) | Soft-token checkpoint parsing and per-block splice hooks |
| [`nodes.py`](nodes.py) | ComfyUI node definitions and public mappings |
| [`_vendor/`](_vendor/) | Generated standalone fallback; never an independent source |

`AnimaAdapterLoader` and `AnimaFeraLoader` are alternative routing loaders for a checkpoint and must not both patch the same adapter. `AnimaSoftTokensLoader` may be chained after either loader because it owns a separate conditioning path.

## Hook invariant

- Never replace `diffusion_model.forward`, a DiT block's `forward`, or a Linear's `forward`.
- Use `ModelPatcher.add_object_patch` to install copied `_forward_hooks` or `_forward_pre_hooks` mappings. This preserves ComfyUI cast-weight/dynamic-VRAM traversal, compile behavior, and `unpatch_model` restoration.
- Hydra, FeRA, Chimera, and ReFT add outputs with `forward_hook`. Soft tokens and per-step router state use `forward_pre_hook`; the global Chimera content router observes the LLM adapter with `forward_hook` because text features do not exist at the model pre-hook boundary.
- When multiple node features touch one hook mapping, merge with the already patched mapping returned by ComfyUI. Never replace the mapping with a fresh dict that drops earlier hooks.

Replacing `forward` can leave cast-weight Linears on CPU and fail only at runtime. Treat this as a correctness rule, not a style preference.

## Checkpoint and routing contracts

- Dispatch from explicit metadata plus validated key shapes. A router's input width alone cannot distinguish sigma columns from FEI columns.
- Preserve router input ordering exactly. Hydra-style inputs are `[pooled, sinusoidal(sigma), FEI]`; Chimera's frequency router uses `[FEI, sinusoidal(sigma)]`.
- Preserve FEI band ordering by checkpoint family: author-faithful FeRA and plan2 routing use different canonical orders. Import the canonical kernels rather than reimplementing blur, band energy, sigma features, or masks in this node.
- Author-faithful FeRA resolves targets by walking `diffusion_model.named_modules()`. Do not switch it to `model_lora_keys_unet`, which can miss fused projections and LLM-adapter submodules.
- Chimera single-A and dual-A formats are mutually exclusive by key shape. Global content routing requires its `content_router.net.*` weights and an Anima `llm_adapter`; malformed declarations must raise a clear error.
- Routing-aware adapters are not mergeable into one static DiT delta. Do not add a path that silently drops router, ReFT, Chimera, FeRA, or soft-token state.

## Soft-token contracts

- ComfyUI FLOW timesteps arrive as `sigma * 1000`; divide by `_FLOW_MULTIPLIER` before t-bucket selection so the node matches training's `[0, 1]` sigma domain.
- Apply the selected per-layer token bank to the whole batch, including cond and uncond rows, exactly as training did.
- Preserve the configured splice mode and padded-text assumptions. Do not shorten or mask Anima's max-padded text encoder output.
- Infer dimensions only from the documented tensor shapes and metadata; reject missing or inconsistent banks instead of guessing.

## Router compute and vendor tree

The live source of truth is [`../../library/inference/router_compute.py`](../../library/inference/router_compute.py) and its canonical dependencies. In-repo loading uses that live source; standalone installation falls back to `_vendor/`.

Trained gates are sensitive to band order and sigma features, so live/vendor drift can silently change output without a shape error.

- Modify live `library/` or `networks/` sources first; never hand-edit `_vendor/`.
- Regenerate with `python tasks.py vendor-sync` from the repository root.
- Check release drift with `python tasks.py vendor-sync --check`.
- If a change does not affect this node's vendored dependency closure, record that conclusion rather than editing generated files anyway.

## Validation and release checklist

- Run focused loader tests such as `tests/test_chimera_node_loader.py`, `tests/test_soft_tokens_node_loader.py`, and `tests/test_router_compute.py`.
- Add a regression test for metadata dispatch, key mapping, hook coexistence, dtype/device behavior, or numerical equivalence whenever that contract changes.
- Exercise standalone fallback when changing imports or the vendor manifest; an in-repo import success is not sufficient evidence.
- Keep `README.md` loader behavior and changelog synchronized with user-visible changes.
- Before publishing, verify the generated vendor tree, bump the package version deliberately, inspect the archive contents, and keep registry credentials outside source, docs, logs, and command examples.
