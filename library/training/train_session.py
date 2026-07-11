"""Training session setup through loop entry and teardown."""

from __future__ import annotations

import logging
import math
import os
import random
import time

import torch
from accelerate.utils import set_seed

from library import train_util
from library.datasets import debug_dataset
from library.anima import training as anima_train_utils
from library.anima import text_strategies
from library.runtime.accelerator import prepare_accelerator, prepare_dtype, resume_from_local_or_hf_if_specified
from library.runtime.device import clean_memory_on_device
from library.training.adapter_resolver import resolve_adapters
from library.training.checkpoints import (
    CheckpointSaver,
    plan_resume_start,
    save_state_on_train_end,
)
from library.training.loop import build_loop_state, run_training_loop
from library.log import setup_logging
from library.runtime.peak_probe import PeakProbe
from library.training.cli_args import verify_training_args
from library.training.memory_probe import MemoryProbe
from library.training.metadata import (
    add_dataset_metadata,
    add_model_hash_metadata,
    build_training_metadata,
    finalize_metadata,
)
from library.training.method_adapter import SetupCtx
from library.training.precision_policy import (
    resolve_mixed_precision,
    resolve_vae_dtype,
)
from library.training.progress import ProgressSink, run_scope
from library.training.probes import attach_peak_probe_to_network, maybe_probe, maybe_probe_components
from library.training.train_bootstrap import (
    collect_compile_resolutions,
    decode_deferred_samples_safely,
    normalize_sample_args,
    sample_preview_enabled,
)

logger = logging.getLogger(__name__)


def run_training_session(trainer, args) -> None:
    session_id = random.randint(0, 2**32)
    training_started_at = time.time()
    normalize_sample_args(args)
    verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    setup_logging(args, reset=True)

    cache_latents = args.cache_latents

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)

    # Whether inductor will have CUDAGraphs active -- governs whether the
    # training loop needs to call torch.compiler.cudagraph_mark_step_begin()
    # each step (see the call site inside the accumulate block).
    trainer._cudagraph_mark_step = bool(
        getattr(args, "torch_compile", False)
        and getattr(args, "compile_inductor_mode", None)
        in ("reduce-overhead", "max-autotune")
    )

    tokenize_strategy = trainer.get_tokenize_strategy(args)
    text_strategies.TokenizeStrategy.set_strategy(tokenize_strategy)
    tokenizers = trainer.get_tokenizers(
        tokenize_strategy
    )  # will be removed after sample_image is refactored

    # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
    latents_caching_strategy = trainer.get_latents_caching_strategy(args)
    text_strategies.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    (
        train_dataset_group,
        val_dataset_group,
        current_epoch,
        current_step,
        collator,
        use_user_config,
        use_dreambooth_method,
    ) = trainer._prepare_dataset(args)
    # Preview images use the same compiled DiT blocks as training, so the
    # compile shape set must cover both dataset buckets and sample prompt
    # preview resolutions.
    args.bucket_resolutions = collect_compile_resolutions(
        train_dataset_group,
        val_dataset_group,
        sample_prompts=getattr(args, "sample_prompts", None),
    )

    if args.debug_dataset:
        train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
        debug_dataset(train_dataset_group)

        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            debug_dataset(val_dataset_group)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify arguments (train_data_dir must be the parent of folders with images)"
        )
        return

    if cache_latents:
        assert train_dataset_group.is_latent_cacheable(), (
            "when caching latents, either color_aug or random_crop cannot be used"
        )
        if val_dataset_group is not None:
            assert val_dataset_group.is_latent_cacheable(), (
                "when caching latents, either color_aug or random_crop cannot be used"
            )

    trainer.assert_extra_args(
        args, train_dataset_group, val_dataset_group
    )  # may change some args

    # Set the text-encoder-outputs caching strategy now (before the model
    # load) so the cache-completeness probe below can use it to decide
    # whether the Qwen3 text encoder needs loading at all.
    weight_dtype, _save_dtype = prepare_dtype(args)
    text_encoder_outputs_caching_strategy = (
        trainer.get_text_encoder_outputs_caching_strategy(args, weight_dtype)
    )
    if text_encoder_outputs_caching_strategy is not None:
        text_strategies.TextEncoderOutputsCachingStrategy.set_strategy(
            text_encoder_outputs_caching_strategy
        )

    # Decide whether the heavy encoders are actually needed. When caching is
    # enabled the caches MUST already be complete on disk (run `make
    # preprocess` first) — train.py no longer encodes missing latents / TE
    # outputs on the fly. With complete caches and nothing else needing them
    # we skip loading the encoders entirely (saves the disk read, RAM, and
    # the GPU round-trip). `cache_latents = false` (e.g. IP-Adapter) is a
    # separate, explicit live-encoding mode, not a fallback.
    sampling_enabled = sample_preview_enabled(args)

    def _latents_complete(group):
        return group is None or group.is_latents_cache_complete()

    def _te_complete(group):
        return group is None or group.is_text_encoder_outputs_cache_complete()

    if cache_latents and not (
        _latents_complete(train_dataset_group)
        and _latents_complete(val_dataset_group)
    ):
        raise RuntimeError(
            "Latent cache is incomplete. train.py requires a completed "
            "preprocess pass — run `make preprocess` (or set "
            "cache_latents = false for live VAE encoding)."
        )

    if args.cache_text_encoder_outputs and not (
        _te_complete(train_dataset_group) and _te_complete(val_dataset_group)
    ):
        raise RuntimeError(
            "Text-encoder cache is incomplete. train.py requires a completed "
            "preprocess pass — run `make preprocess` (or set "
            "cache_text_encoder_outputs = false for live encoding)."
        )

    # CMMD validation generates samples and decodes them through the VAE
    # (see library/training/validation.py). It reads cached TE outputs, so
    # it needs the VAE but not the text encoder.
    cmmd_validation = val_dataset_group is not None and getattr(
        args, "use_cmmd", True
    )
    # VAE: needed only to live-encode (caching off), to decode training
    # samples, or to decode CMMD validation samples. With caching on the
    # cache is guaranteed complete above, so no encode pass is required.
    vae_needed = (not cache_latents) or sampling_enabled or cmmd_validation

    # Qwen3 TE: needed only to live-encode (caching off), to encode active
    # sample prompts, or when the text encoder itself is being trained.
    qwen3_needed = (
        (not args.cache_text_encoder_outputs)
        or sampling_enabled
        or trainer.is_train_text_encoder(args)
    )

    # Resolve mixed precision BEFORE prepare_accelerator: Accelerator() bakes
    # the autocast dtype at construction time.
    resolve_mixed_precision(args)

    # Prepare accelerator
    logger.info("preparing accelerator")
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process
    trainer.memory_probe = MemoryProbe.from_args(
        args,
        is_main_process=is_main_process,
        t0=training_started_at,
    )
    trainer.peak_probe = PeakProbe.from_args(
        args,
        is_main_process=is_main_process,
        t0=training_started_at,
    )
    maybe_probe(
        trainer,
        "accelerator_ready",
        device=accelerator.device,
        phase="setup",
        session_id=session_id,
        method=getattr(args, "method", None),
        preset=getattr(args, "preset", None),
        network_module=getattr(args, "network_module", None),
        torch_compile=getattr(args, "torch_compile", False),
        compile_inductor_mode=getattr(args, "compile_inductor_mode", None),
        attn_mode=getattr(args, "attn_mode", None),
        blocks_to_swap=getattr(args, "blocks_to_swap", None),
        selective_checkpoint=getattr(args, "selective_checkpoint", "off"),
        selective_checkpoint_blocks=getattr(
            args, "selective_checkpoint_blocks", ""
        ),
        memory_probe_jsonl=getattr(trainer.memory_probe, "path", None),
        peak_probe_jsonl=getattr(trainer.peak_probe, "path", None),
    )
    if trainer.peak_probe is not None:
        trainer.peak_probe.write(
            {
                "ev": "peak_probe_config",
                "label": "accelerator_ready",
                "phase": "setup",
                "path": trainer.peak_probe.path,
                "max_steps": trainer.peak_probe.max_steps,
                "level": trainer.peak_probe.level,
                "torch_compile": getattr(args, "torch_compile", False),
                "compile_inductor_mode": getattr(args, "compile_inductor_mode", None),
                "attn_mode": getattr(args, "attn_mode", None),
                "blocks_to_swap": getattr(args, "blocks_to_swap", None),
                "selective_checkpoint": getattr(args, "selective_checkpoint", "off"),
                "selective_checkpoint_blocks": getattr(
                    args, "selective_checkpoint_blocks", ""
                ),
            }
        )

    # mixed precision dtype
    weight_dtype, save_dtype = prepare_dtype(args)
    vae_dtype = (
        resolve_vae_dtype(args, weight_dtype)
        if trainer.cast_vae(args)
        else None
    )

    # load target models: unet may be None for lazy loading
    model_version, text_encoder, vae, unet = trainer.load_target_model(
        args,
        weight_dtype,
        accelerator,
        load_qwen3=qwen3_needed,
        load_vae=vae_needed,
    )
    maybe_probe(
        trainer,
        "target_models_loaded",
        device=accelerator.device,
        phase="setup",
        qwen3_loaded=bool(qwen3_needed),
        vae_loaded=bool(vae_needed),
        vae_dtype=getattr(vae, "dtype", None) if vae is not None else None,
    )
    if vae_dtype is None:
        vae_dtype = vae.dtype if vae is not None else weight_dtype
        logger.info(
            f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false"
        )

    # text_encoder is List[CLIPTextModel] or CLIPTextModel
    text_encoders = (
        text_encoder if isinstance(text_encoder, list) else [text_encoder]
    )

    # prepare dataset for latents caching if needed. When vae is None the
    # latents are already fully cached -- new_cache_latents still runs to
    # populate each ImageInfo.latents_npz path the dataloader reads, but
    # forms no encode batches so the (absent) VAE is never touched.
    if cache_latents:
        if vae is not None:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)
        if val_dataset_group is not None:
            val_dataset_group.new_cache_latents(vae, accelerator)

        if vae is not None:
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
    text_encoding_strategy = trainer.get_text_encoding_strategy(args)
    text_strategies.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    trainer.cache_text_encoder_outputs_if_needed(
        args,
        accelerator,
        text_encoders,
        train_dataset_group,
    )
    if val_dataset_group is not None:
        trainer.cache_text_encoder_outputs_if_needed(
            args,
            accelerator,
            text_encoders,
            val_dataset_group,
        )

    if unet is None:
        # lazy load unet if needed. text encoders may be freed or replaced with dummy models for saving memory
        unet, text_encoders = trainer.load_unet_lazily(
            args, weight_dtype, accelerator, text_encoders
        )

    # Stage the T5("") sidecar once if caption dropout is on — dropped
    # rows then get the same crossattn embedding Anima feeds at
    # CFG-uncond inference instead of all-zeros (which is out-of-dist).
    if trainer._state.caption_dropout_enabled or trainer._blank_prompt_preservation_enabled(args):
        trainer._ensure_uncond_crossattn(args, accelerator, weight_dtype)

    network_result = trainer._create_and_apply_network(
        args, accelerator, vae, text_encoder, unet, text_encoders, weight_dtype
    )
    if network_result is None:
        return
    network, net_kwargs, train_unet, train_text_encoder = network_result
    peak_probe_lokr_modules = attach_peak_probe_to_network(network, trainer.peak_probe)
    if trainer.peak_probe is not None:
        trainer.peak_probe.write(
            {
                "ev": "peak_probe_config",
                "label": "network_peak_probe_attached",
                "phase": "setup",
                "module_count": peak_probe_lokr_modules,
            }
        )

    # Resolve and run on_network_built for each method adapter (EasyControl,
    # IP-Adapter, …). Each adapter validates its runtime contract and
    # logs/sets up auxiliary state before optimizer / accelerator wiring.
    trainer._adapters = resolve_adapters(args, network)
    if trainer._adapters:
        setup_ctx = SetupCtx(
            args=args,
            accelerator=accelerator,
            network=network,
            unet=unet,
            text_encoders=text_encoders,
            weight_dtype=weight_dtype,
        )
        for adapter in trainer._adapters:
            adapter.on_network_built(setup_ctx)
    maybe_probe_components(
        trainer,
        "network_applied",
        network=network,
        unet=unet,
        device=accelerator.device,
        phase="setup",
        adapter_count=len(trainer._adapters),
        use_lokr=getattr(args, "use_lokr", None),
        lokr_factor=getattr(args, "lokr_factor", None),
    )

    (
        optimizer,
        optimizer_name,
        optimizer_args,
        optimizer_train_fn,
        optimizer_eval_fn,
        text_encoder_lr,
        lr_descriptions,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = trainer._setup_optimizer_and_dataloader(
        args,
        accelerator,
        network,
        train_dataset_group,
        val_dataset_group,
        collator,
    )
    maybe_probe_components(
        trainer,
        "optimizer_built",
        network=network,
        unet=unet,
        optimizer=optimizer,
        device=accelerator.device,
        phase="setup",
        optimizer_name=optimizer_name,
    )

    (
        network,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        training_model,
        unet,
        text_encoders,
        text_encoder,
        unet_weight_dtype,
    ) = trainer._prepare_with_accelerator(
        args,
        accelerator,
        network,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        unet,
        text_encoders,
        text_encoder,
        vae,
        vae_dtype,
        weight_dtype,
        train_unet,
        train_text_encoder,
        cache_latents,
    )
    maybe_probe_components(
        trainer,
        "accelerator_prepared",
        network=network,
        unet=unet,
        optimizer=optimizer,
        device=accelerator.device,
        phase="setup",
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Structured progress sink (Phase 0): a JSONL event stream next to the
    # checkpoint that the GUI / daemon can tail instead of regex-parsing
    # tqdm. Main-process only; default on, gated by --progress_jsonl.
    trainer.progress_sink = None
    if is_main_process:
        progress_path = ProgressSink.resolve_path(args)
        if progress_path is not None:
            trainer.progress_sink = ProgressSink(
                progress_path,
                run=args.output_name or "run",
                method=getattr(args, "method", None),
                preset=getattr(args, "preset", None),
                t0=training_started_at,
            )
            trainer.progress_sink.run_start(
                total_steps=args.max_train_steps,
                total_epochs=num_train_epochs,
                pid=os.getpid(),
            )

    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = (
            math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
        )

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    accelerator.print("running training")
    accelerator.print("  num train images * repeats")
    accelerator.print("  num validation images * repeats")
    accelerator.print("  num reg images")
    accelerator.print("  num batches per epoch")
    accelerator.print("  num epochs")
    accelerator.print("  batch size per device")
    accelerator.print("  gradient accumulation steps")
    accelerator.print("  total optimization steps")

    metadata = build_training_metadata(
        args,
        session_id=session_id,
        training_started_at=training_started_at,
        text_encoder_lr=text_encoder_lr,
        optimizer_name=optimizer_name,
        optimizer_args=optimizer_args,
        model_version=model_version,
        num_train_images=train_dataset_group.num_train_images,
        num_val_images=val_dataset_group.num_train_images
        if val_dataset_group is not None
        else 0,
        num_reg_images=train_dataset_group.num_reg_images,
        num_batches_per_epoch=len(train_dataloader),
        num_train_epochs=num_train_epochs,
    )
    trainer.update_metadata(metadata, args)  # architecture specific metadata
    add_dataset_metadata(
        metadata,
        train_dataset_group,
        args,
        use_user_config=use_user_config,
        use_dreambooth_method=use_dreambooth_method,
        total_batch_size=total_batch_size,
    )
    add_model_hash_metadata(metadata, args)
    metadata, minimum_metadata = finalize_metadata(
        metadata, net_kwargs=net_kwargs if args.network_args else None
    )

    # Saver owns every save / remove operation plus the accelerator
    # save/load pre-hooks that persist train_state.json. Hooks must be
    # registered before resume_from_local_or_hf_if_specified() so the
    # load hook fires and populates saver.steps_from_state.
    saver = CheckpointSaver(
        args=args,
        accelerator=accelerator,
        save_dtype=save_dtype,
        metadata=metadata,
        minimum_metadata=minimum_metadata,
        get_sai_model_spec_fn=trainer.get_sai_model_spec,
        current_epoch=current_epoch,
        current_step=current_step,
        progress_sink=trainer.progress_sink,
    )
    saver.register_hooks(network)

    # auto-resume from the resumable checkpoint if one exists
    saver.auto_resume(network)

    # resume
    resume_from_local_or_hf_if_specified(accelerator, args)
    resume_plan = plan_resume_start(
        args,
        steps_from_state=saver.steps_from_state,
        batches_per_epoch=len(train_dataloader),
        num_processes=accelerator.num_processes,
    )
    initial_step = resume_plan.initial_step
    epoch_to_start = resume_plan.epoch_to_start

    # Keep train_dataset_group when stage schedule needs mid-run rebuilds.
    # Otherwise drop it before loop entry — the dataloader already holds
    # the data it needs. Keep val_dataset_group alive: CMMD validation
    # enumerates its image_data to pair held-out references with generated samples.
    stage_dataset = getattr(args, "_stage_train_dataset_group", None) or train_dataset_group
    stage_loader_kwargs = getattr(args, "_stage_dataloader_kwargs", None)
    from library.training.stage_schedule import stage_schedule_enabled

    if not stage_schedule_enabled(args):
        del train_dataset_group
        stage_dataset = None
        stage_loader_kwargs = None

    loop_state = build_loop_state(
        trainer,
        args=args,
        accelerator=accelerator,
        saver=saver,
        network=network,
        unet=unet,
        text_encoder=text_encoder,
        text_encoders=text_encoders,
        vae=vae,
        tokenizers=tokenizers,
        training_model=training_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_dataset_group=val_dataset_group,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_descriptions=lr_descriptions,
        optimizer_train_fn=optimizer_train_fn,
        optimizer_eval_fn=optimizer_eval_fn,
        weight_dtype=weight_dtype,
        unet_weight_dtype=unet_weight_dtype,
        vae_dtype=vae_dtype,
        text_encoding_strategy=text_encoding_strategy,
        tokenize_strategy=tokenize_strategy,
        train_text_encoder=train_text_encoder,
        train_unet=train_unet,
        current_epoch=current_epoch,
        current_step=current_step,
        num_train_epochs=num_train_epochs,
        epoch_to_start=epoch_to_start,
        initial_step=initial_step,
        metadata=metadata,
    )
    loop_state.train_dataset_group = stage_dataset
    loop_state.dataloader_kwargs = stage_loader_kwargs
    maybe_probe(
        trainer,
        "loop_ready",
        device=accelerator.device,
        phase="setup",
        global_step=loop_state.global_step,
        max_train_steps=args.max_train_steps,
        num_train_epochs=num_train_epochs,
    )

    # run_scope emits the matching run_end (ok / stopped / error) on exit;
    # run_start already fired when the sink was constructed above.
    with run_scope(trainer.progress_sink, final_step=lambda: loop_state.global_step):
        training_loop_completed = False
        try:
            run_training_loop(trainer, loop_state)
            training_loop_completed = True
        finally:
            if not training_loop_completed:
                decode_deferred_samples_safely(
                    accelerator,
                    args,
                    loop_state,
                    vae,
                    optimizer_eval_fn=optimizer_eval_fn,
                )

        accelerator.end_training()
        optimizer_eval_fn()
        decode_deferred_samples_safely(accelerator, args, loop_state, vae)

        if is_main_process and sample_preview_enabled(args):
            try:
                accelerator.unwrap_model(loop_state.unet).to("cpu")
            except Exception:
                pass
            clean_memory_on_device(accelerator.device)
            anima_train_utils.decode_pending_samples(accelerator, args, vae)

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            save_state_on_train_end(args, accelerator)

        saver.cleanup_resumable()

