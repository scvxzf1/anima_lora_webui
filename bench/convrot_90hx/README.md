# CMP 90HX validation tools

Experimental Anima validation harness; no production precision dispatch or defaults changed.
Completed run: [findings](../../docs/findings/convrot_90hx_validation_20261002.md).
Next work: [development plan](../../docs/proposal/convrot_90hx_followup_development_plan_20261002.md).

```bash
.venv/bin/python -m bench.convrot_90hx.runner prepare --out-dir output/convrot-90hx-<new>/prepared
CUDA_VISIBLE_DEVICES=GPU-806debd9-d254-da57-d515-ccae15b4ac6a .venv/bin/python -m bench.convrot_90hx.runner run --kind smoke --mode bf16 --warmup 0 --steps 3 --out-dir output/convrot-90hx-<new>/bf16-smoke
CUDA_VISIBLE_DEVICES=GPU-806debd9-d254-da57-d515-ccae15b4ac6a .venv/bin/python -m bench.convrot_90hx.runner run --kind performance --mode w8a8 --warmup 10 --steps 30 --out-dir output/convrot-90hx-<new>/w8a8-performance
```

Defaults: explicitly specified Anima base-v1.0 checkpoint,
`output/ab-anima-20261002/{cache,images-1024}`, BF16 compute, FP32 plain LoRA
rank/alpha32, Flash, full checkpoint, swap12, foreach BF16 restore, compile off.
Use a fresh output directory; errors preserve `failure.json`. No automatic
swap reduction/backend fallback. Strict INT8 and FP32 matmul/TF32 settings
are fixed by the batch orchestrator; inspect the saved manifest for each run.
`--timestep` is flow sigma in (0,1), not a sampler index. CPU noise and hashes
make inputs replayable; text keeps all 512 max-padded positions.

| Module | Purpose and evidence boundary |
|---|---|
| `runner` | Smoke, independent single-update numerical, performance, profile and synchronized diagnostic latency; experimental forward/backward flags are explicit opt-in |
| `batch` | New process per job, device UUID, fixed numeric environment, status and failure preservation |
| `grid` | Official 2 cache ×3 seed ×3 sigma independent updates; custom subsets are incomplete; optional deterministic Flash control |
| `assessment` | Frozen whole/per-layer output, gradient and update gates; requires full18 sample/parameter coverage; missing data cannot pass |
| `microbench` | Quant/GEMM exact contracts, timing and extra allocation for actual/synthetic shapes |
| `event_profile` | CUDA Event local diagnostic fallback; not end-to-end throughput |
| `lifetime_probe` | Real checkpointed graph/owner payload residency around CPU park and graph deletion; diagnostic synchronization only |
| `resume_probe` | Fixed replay cursor plus optimizer/RNG restore, explicitly deterministic Flash; does not establish production dataloader cursor restoration |
| `trajectory` | Prepared three-seed continuous20-update tool; conditional experiment NOT run in this validation, no long-trajectory acceptance claimed |
| `production_lifecycle` | Actual train CLI with graph/CPU park/VAE/restore phase traces; synchronized diagnostics, no throughput ranking |
| `lifecycle_replay` | Four deterministic updates with/without one real preview/VAE lifecycle; complete parameter/gradient/optimizer/RNG fingerprints |

Numerical mode resets adapter and AdamW for every sample and saves complete
FP32 named gradients/updates/output. Undefined relative errors/cosines for
zero reference norms are null, not fabricated passes. Grid execution PASS
means finite and complete, distinct from assessment quality gates. Default
Flash backward is nondeterministic: self-repeat AdamW updates exceeded frozen
gates, so candidate update failures require calibrated controls. The executed
deterministic control was only one sample, not full18 acceptance.

Performance wall time brackets offloader-future and CUDA drains, without
per-step host scalar reductions. `host_submission_sec` in step JSONL is
asynchronous submission time, not full-step latency. Separate synchronized
latency diagnostics provide p50/p95; never mix them into throughput ranking.
Finite checks for performance run outside the measured window. Loading and
training-window peaks are separate; sampling full-process peaks require the
production phase traces. Read-only NVML samples every second and does not
measure exact instantaneous peaks.

CMP90HX CUPTI reports `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED(42)`; exported
Chrome traces contain CPU events only. CUDA Event diagnostics do not replace
a full CUDA trace. The first failed profiler attempt remains in the run,
with completed retries recorded separately.

Production configs/phase traces, 1024 sample images and adapter/optimizer
save/resume audits are in `output/convrot-90hx-20261002-210307`. Eight-step
sampling is functional smoke, not training quality. Production resume5→10
checks counters/state; exact continuous10==5+5 was demonstrated only by the
separate controlled replay. Source captions/sample prompt hashes are saved
in `input-plan-manifest.json`. All60 images belong to training diagnostics,
with no held-out split. S5 500-step configs were prepared but NOT executed:
`SKIPPED_NO_ACCEPTED_CANDIDATE`. Candidate compile/production policy integration
and the continuous20 trajectory were likewise conditional and not executed.

2026-10-03 lifecycle repair and short retest: [report](../../docs/findings/convrot_lifecycle_fix_90hx_20261003.md).
These tools require fresh output paths and do not dispatch500-step training.
