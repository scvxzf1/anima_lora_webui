"""Misc utility entry-points: merge, comfy-batch, distill-prep, distill-mod,
test-unit, test-fast, test-focused, test-slow, type-check, update, export-logs,
print-config, explain-config, config-compat, training-hot."""

from __future__ import annotations

import importlib.util
import os
import sys

from ._common import PY, _path, _preset, bespoke_preset_flags, run


FAST_TEST_TARGETS = [
    "tests/test_training_hot_runner.py",
    "tests/test_plain_lora_speed_runner.py",
    "tests/test_signal_probe_runner.py",
    "tests/test_mfu_bench.py",
    "tests/test_mfu_gpu_theoretical.py",
]

TYPE_CHECK_TARGETS = [
    "library/config",
    "tasks.py",
    "scripts/config_compat.py",
    "scripts/config_explain.py",
    "scripts/tasks/_common.py",
    "scripts/tasks/utilities.py",
    "web/services/config/common.py",
    "web/services/config/estimation.py",
    "web/services/config/file_groups.py",
    "web/services/config/merge.py",
    "web/services/config/metadata.py",
    "web/services/config/output_runs.py",
    "web/services/config/paths.py",
    "web/services/config/preflight.py",
    "web/services/config/raw_files.py",
    "web/services/config/sample_prompts.py",
]


def cmd_merge(extra):
    """Bake latest LoRA in ADAPTER_DIR (env, default 'output/ckpt') into the base DiT."""
    adapter_dir = os.environ.get("ADAPTER_DIR", "output/ckpt")
    multiplier = os.environ.get("MULTIPLIER", "1.0")
    run(
        [
            PY,
            "scripts/merge_to_dit.py",
            "--adapter_dir",
            adapter_dir,
            "--multiplier",
            multiplier,
            *extra,
        ]
    )


def cmd_comfy_batch(extra):
    workflow = extra[0] if extra else "workflows/modhydra.json"
    remaining = extra[1:] if extra else []
    run([PY, "scripts/comfy_batch.py", workflow, *remaining])


def cmd_distill_prep(extra):
    """Pre-stage artifacts for ``make distill-mod``.

    Phase 1: emits ``post_image_dataset/_anima_uncond_te.safetensors``
    (T5("") cross-attn baseline) — consumed as the student's unconditional
    text input, replacing the zeroed-crossattn shortcut. ``make preprocess-te``
    already produces this for free; this Phase 1 block is the explicit
    re-stager (useful with ``--overwrite`` after a model swap).

    Phase 2: emits teacher-synthesized clean latents under
    ``post_image_dataset/distill_mod_synth/`` (same NPZ layout as
    ``cache_latents.py``). Train with
    ``make distill-mod ARGS='--synth_data_dir post_image_dataset/distill_mod_synth'``
    to fit on the teacher's manifold (paper-faithful; removes real-vs-teacher
    gap that floors val loss).

    Skip flags forwarded via ``extra``: ``--skip_uncond``, ``--skip_synth``,
    ``--max_samples N``, etc.
    """
    run([PY, "-m", "scripts.distill_mod.prep", *extra])


def cmd_distill_mod(extra):
    """Distill the pooled_text_proj MLP for modulation guidance.

    Honors ``PRESET`` (default ``default``) — translates ``blocks_to_swap`` and
    ``gradient_checkpointing`` from ``configs/presets.toml`` into CLI flags so
    ``make distill-mod PRESET=low_vram`` enables grad ckpt + unsloth offload.
    Trailing ``extra`` args are appended last, so user CLI overrides win.

    Saves to ``output/ckpt/pooled_text_proj.safetensors`` so ``make test MOD=1``
    picks it up automatically.
    """
    preset_flags = bespoke_preset_flags(_preset())
    run(
        [
            PY,
            "-m",
            "scripts.distill_mod.distill",
            "--data_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--dit_path",
            _path(
                "pretrained_model_name_or_path",
                "models/diffusion_models/anima-base-v1.0.safetensors",
            ),
            "--output_path",
            "output/ckpt/pooled_text_proj.safetensors",
            "--attn_mode",
            "flash",
            *preset_flags,
            *extra,
        ]
    )


def cmd_test_unit(extra):
    run([PY, "-m", "pytest", "-q", "tests/", *extra])


def cmd_test_backend_smoke(extra):
    """Backend web/training smoke subset for durable optimization gates."""
    targets = [
        "tests/test_web_http_contracts.py",
        "tests/test_training_websocket.py",
        "tests/test_training_queue_retry_wake.py",
        "tests/test_training_retry_classification.py",
        "tests/test_training_retry_integration.py",
        "tests/test_queue_policy_layers.py",
        "tests/test_settings_image_test_flags.py",
        "tests/test_path_safety.py",
        "tests/test_path_resolve_unified.py",
        "tests/test_method_discovery.py",
        "tests/test_cross_domain_delete_boundaries.py",
        "tests/test_stage_schedule.py",
        "tests/test_env_config_paths.py",
        "tests/test_global_settings_runtime.py",
        "tests/test_web_config_raw_files.py",
        "tests/test_schema_gate_observability.py",
        "tests/test_raw_file_warnings_contract.py",
        "tests/test_queue_item_retry_override.py",
        "tests/test_settings_image_test_save_root.py",
        "tests/test_path_allowlist_freeze.py",
        "tests/test_image_test_service.py",
    ]
    run([PY, "-m", "pytest", "-q", *targets, *extra])


def cmd_test_fast(extra):
    """Run the fast smoke layer for task runners and bench safety guards."""
    run([PY, "-m", "pytest", "-q", "-m", "fast and not slow", *FAST_TEST_TARGETS, *extra])


def cmd_test_focused(extra):
    """Run a caller-selected pytest slice.

    Pass an explicit file, node id, marker, or ``-k`` expression after the
    command. This guard prevents accidentally turning a focused run into a
    full-repo test run.
    """
    if not extra:
        print(
            "Usage: python tasks.py test-focused -- <pytest target, -k expr, or -m marker>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    run([PY, "-m", "pytest", "-q", *extra])


def cmd_test_slow(extra):
    """Run tests explicitly marked as slow."""
    run([PY, "-m", "pytest", "-q", "-m", "slow", "tests/", *extra])


def cmd_type_check(extra):
    """Run the configured pyright pilot gate.

    Defaults to the current config/WebUI pilot surface. Pass explicit paths or
    pyright flags after the command to expand/narrow the checked surface.
    """
    if extra and extra[0] == "--":
        extra = extra[1:]
    if importlib.util.find_spec("pyright") is None:
        print(
            "pyright is not installed in this Python environment. "
            "Install the repo dev type-check dependency first, then rerun: "
            "python tasks.py type-check",
            file=sys.stderr,
        )
        raise SystemExit(2)
    targets = extra or TYPE_CHECK_TARGETS
    run([PY, "-m", "pyright", *targets])


def cmd_update(extra):
    """Update anima_lora from a GitHub release (preserves datasets/output/models;
    prompts on configs/methods/ + configs/gui-methods/ conflicts; runs uv sync)."""
    run([PY, "scripts/update.py", *extra])


def cmd_vendor_sync(extra):
    """Refresh custom_nodes/*/_vendor/ trees from the live library.* sources.

    Run before bumping a custom-node version / publishing — the bundled
    vendor copies (tagger + directedit) are how the ComfyUI nodes import
    their inference subset when not running inside the anima_lora repo.
    Pass --check to compare in a temporary directory without touching _vendor/.
    """
    run([PY, "scripts/sync_vendor.py", *extra])


def cmd_export_logs(extra):
    """Dump TB scalar logs to JSON. RUN=<dir> (default output/logs), ALL=1, JSONL=1."""
    run_path = os.environ.get("RUN", "output/logs")
    cmd = [PY, "scripts/export_logs_json.py", run_path]
    if os.environ.get("ALL"):
        cmd.append("--all")
    if os.environ.get("JSONL"):
        cmd.append("--jsonl")
    run([*cmd, *extra])


def cmd_print_config(extra):
    method = os.environ.get("METHOD", "lora")
    preset = _preset()
    run(
        [
            PY,
            "train.py",
            "--method",
            method,
            "--preset",
            preset,
            "--print-config",
            "--no-config-snapshot",
            *extra,
        ]
    )


def cmd_explain_config(extra):
    """Trace config layer history without launching the trainer.

    Env:
      METHOD=<name> (default lora)
      PRESET=<name> (default default)
      METHODS_SUBDIR=<dir> (default methods)
      CONFIGS_DIR=<dir> (default configs)
      ANIMA_RUNTIME_CONFIG=<path> (optional runtime TOML layer)
    """
    if extra and extra[0] == "--":
        extra = extra[1:]
    method = os.environ.get("METHOD", "lora")
    preset = _preset()
    methods_subdir = os.environ.get("METHODS_SUBDIR", "methods")
    configs_dir = os.environ.get("CONFIGS_DIR", "configs")
    cmd = [
        PY,
        "scripts/config_explain.py",
        "--method",
        method,
        "--preset",
        preset,
        "--methods-subdir",
        methods_subdir,
        "--configs-dir",
        configs_dir,
    ]
    runtime_config = os.environ.get("ANIMA_RUNTIME_CONFIG")
    if runtime_config:
        cmd += ["--runtime-config", runtime_config]
    run([*cmd, *extra])


def cmd_config_compat(extra):
    """Print compile/checkpoint/block-swap compatibility diagnostics.

    Env:
      METHOD=<name> (default lora)
      PRESET=<name> (default default)
      METHODS_SUBDIR=<dir> (default methods)
      CONFIGS_DIR=<dir> (default configs)
      ANIMA_RUNTIME_CONFIG=<path> (optional runtime TOML layer)
    """
    if extra and extra[0] == "--":
        extra = extra[1:]
    method = os.environ.get("METHOD", "lora")
    preset = _preset()
    methods_subdir = os.environ.get("METHODS_SUBDIR", "methods")
    configs_dir = os.environ.get("CONFIGS_DIR", "configs")
    cmd = [
        PY,
        "scripts/config_compat.py",
        "--method",
        method,
        "--preset",
        preset,
        "--methods-subdir",
        methods_subdir,
        "--configs-dir",
        configs_dir,
    ]
    runtime_config = os.environ.get("ANIMA_RUNTIME_CONFIG")
    if runtime_config:
        cmd += ["--runtime-config", runtime_config]
    run([*cmd, *extra])


def cmd_training_hot(extra):
    """Run generic short training hot tests.

    Examples:
      python tasks.py training-hot -- --dry-run --suite plugins_nonlokr
      python tasks.py training-hot -- --steps 12 --case gui:loha
      python tasks.py training-hot -- --steps 12 --case config:output/runs/x/config.runtime.toml
    """
    if extra and extra[0] == "--":
        extra = extra[1:]
    run([PY, "-m", "bench.training_hot.run_matrix", *extra])
