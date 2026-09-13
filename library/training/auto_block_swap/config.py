"""Opt-in contract and isolated probe argument construction."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import math
from pathlib import Path
from types import SimpleNamespace

from library.env import resolve_model_family


def add_arguments(parser) -> None:
    parser.add_argument(
        "--auto_block_swap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="[EXPERIMENTAL] Automatically adjust block swapping; overrides the manual count for this run.",
    )
    parser.add_argument(
        "--auto_block_swap_max_trials",
        type=int,
        default=6,
        help="Maximum AUTO search candidates (1-16), plus one confirmation run.",
    )
    parser.add_argument(
        "--auto_block_swap_mode",
        choices=["startup", "dynamic"],
        default="startup",
        help="Startup calibration or continuous Krea-2 runtime adaptation.",
    )
    parser.add_argument(
        "--auto_block_swap_interval",
        type=int,
        default=8,
        help="Usable optimizer updates per dynamic timing window (4-256).",
    )
    parser.add_argument(
        "--auto_block_swap_vram_reserve_percent", type=float, default=10.0,
        help="Target free VRAM as percent of total GPU capacity (0-90); minimum 1 GiB safety reserve.",
    )
    parser.add_argument(
        "--auto_block_swap_preference", choices=["balanced", "vram", "ram"],
        default="balanced", help="Balanced speed, save VRAM, or save host RAM (Krea-2).",
    )
    parser.add_argument(
        "--auto_block_swap_timeout",
        type=int,
        default=1800,
        help="Timeout in seconds per AUTO inventory/candidate process.",
    )
    parser.add_argument(
        "--auto_block_swap_swap_io_limit_mb",
        type=float,
        default=1024.0,
        help="Maximum system swap IO per AUTO probe/runtime window in MiB; 0 disables this cap but RAM reserve still applies.",
    )


def configuration_errors(config, *, world_size: int = 1) -> list[str]:
    get = (
        config.get
        if isinstance(config, Mapping)
        else lambda key, default=None: getattr(config, key, default)
    )
    if not get("auto_block_swap", False):
        return []
    errors = []
    try:
        percent = get("auto_block_swap_vram_reserve_percent", 10.0)
        if isinstance(percent, bool) or not math.isfinite(float(percent)) or not 0 <= float(percent) <= 90:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        errors.append("auto_block_swap_vram_reserve_percent must be between 0 and 90")
    preference = get("auto_block_swap_preference", "balanced")
    if preference not in ("balanced", "vram", "ram"):
        errors.append("auto_block_swap_preference must be balanced, vram or ram")
    try:
        family = resolve_model_family(SimpleNamespace(model_family=get("model_family")))
    except ValueError as exc:
        errors.append(str(exc))
        family = None
    if preference == "ram" and family != "krea2_raw":
        errors.append("AUTO RAM preference currently supports Krea-2 only")
    mode = get("auto_block_swap_mode", "startup")
    if mode not in ("startup", "dynamic"):
        errors.append("auto_block_swap_mode must be startup or dynamic")
    if mode == "dynamic":
        if family != "krea2_raw":
            errors.append("Dynamic AUTO currently supports Krea-2 only")
        if get("compile_block_scope", "resident") != "resident":
            errors.append("Dynamic AUTO requires resident compile scope")
        try:
            if not 4 <= int(get("auto_block_swap_interval", 8)) <= 256:
                errors.append("auto_block_swap_interval must be between 4 and 256")
        except (TypeError, ValueError):
            errors.append("auto_block_swap_interval must be an integer")
    if world_size != 1 or get("pipeline_parallel", False):
        errors.append("AUTO block swap currently requires a single training process")
    if not get("use_vae_cache", True) or not get("use_text_cache", True):
        errors.append("AUTO block swap requires complete disk latent and text caches")
    if get("max_data_loader_n_workers", 0) not in (None, 0):
        errors.append(
            "AUTO v1 requires max_data_loader_n_workers=0 for reproducible host budgeting"
        )
    if not get("network_train_unet_only", True):
        errors.append("AUTO v1 requires network_train_unet_only=true")
    if get("network_module", "networks.lora_anima") != "networks.lora_anima" or get(
        "network_args"
    ):
        errors.append(
            "AUTO v1 requires networks.lora_anima without custom network_args"
        )
    if get("lora_adapter_kind", "lora") != "lora":
        errors.append("AUTO v1 supports plain LoRA only")
    flags = (
        "use_ortho",
        "use_timestep_mask",
        "use_moe_style",
        "add_reft",
        "use_glora",
        "use_loha",
        "use_lokr",
        "use_vera",
        "use_dora",
        "dora_wd",
        "use_ip_adapter",
        "use_easycontrol",
        "use_chimera_hydra",
        "cpu_offload_checkpointing",
        "unsloth_offload_checkpointing",
        "anima_freeze_blocks",
        "train_llm_adapter",
        "network_train_text_encoder_only",
        "prior_preservation_weight",
        "inverted_mask_prior_weight",
        "artist_filter",
        "debug_dataset",
    )
    active = [key for key in flags if get(key, False)]
    if active:
        errors.append("AUTO v1 does not support: " + ", ".join(active))
    if get("base_compute", "bf16") not in ("bf16", "nf4"):
        errors.append("AUTO v1 supports BF16 or prequantized NF4 base compute")
    if get("block_swap_transfer_dtype", "bf16") != "bf16":
        errors.append("AUTO v1 requires block_swap_transfer_dtype=bf16")
    if (
        get("stage_schedule")
        or get("resume")
        or get("initial_step")
        or get("initial_epoch")
    ):
        errors.append("AUTO v1 does not support resume or stage schedules")
    if get("sample_prompts") and any(
        get(k)
        for k in (
            "sample_at_first",
            "sample_every_n_steps",
            "sample_every_n_epochs",
        )
    ):
        errors.append(
            "AUTO v1 calibrates training only; disable preview sampling first"
        )
    if get("validate_every_n_steps") or get("validate_every_n_epochs"):
        errors.append("AUTO v1 requires validation cadence to be disabled")
    try:
        if not 1 <= int(get("auto_block_swap_max_trials", 6)) <= 16:
            errors.append("auto_block_swap_max_trials must be between 1 and 16")
        if int(get("auto_block_swap_timeout", 1800)) <= 0:
            errors.append("auto_block_swap_timeout must be positive")
    except (ValueError, TypeError):
        errors.append("AUTO trial count and timeout must be integers")
    try:
        swap_limit = get("auto_block_swap_swap_io_limit_mb", 1024.0)
        if isinstance(swap_limit, bool) or not math.isfinite(float(swap_limit)) or float(swap_limit) < 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        errors.append("auto_block_swap_swap_io_limit_mb must be non-negative")
    return errors


def probe_arguments(args, directory: Path, *, blocks: int, inventory: bool = False):
    probe = copy.deepcopy(args)
    probe.auto_block_swap = False
    probe.blocks_to_swap = blocks
    probe._auto_swap_probe = {"directory": str(directory), "inventory": inventory}
    # Private args are serialized as JSON, never as an executable pickle.
    overrides = {
        "output_dir": str(directory / "output"),
        "output_name": "probe",
        "logging_dir": None,
        "log_with": None,
        "log_tracker_config": None,
        "huggingface_repo_id": None,
        "huggingface_token": None,
        "wandb_api_key": None,
        "sample_at_first": False,
        "sample_every_n_steps": None,
        "sample_every_n_epochs": None,
        "validate_every_n_steps": None,
        "validate_every_n_epochs": None,
        "use_cmmd": False,
        "save_state": False,
        "save_state_on_train_end": False,
        "save_every_n_steps": None,
        "save_every_n_epochs": None,
        "save_n_epoch_ratio": None,
        "checkpointing_epochs": None,
        "progress_jsonl": "off",
        "memory_probe_jsonl": "off",
        "peak_probe_jsonl": "off",
        "gradient_flow_probe_jsonl": "off",
        "block_swap_profile_jsonl": str(directory / "transfers.jsonl"),
        "profile_steps": None,
        "debug_dataset": False,
        "debug_finite_checks": True,
        "skip_cache_check": False,
        "max_data_loader_n_workers": 0,
        "persistent_data_loader_workers": False,
    }
    for key, value in overrides.items():
        setattr(probe, key, value)
    return probe
