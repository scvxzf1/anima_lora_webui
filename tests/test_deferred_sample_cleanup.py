from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"
ANIMA_TRAINING = ROOT / "library" / "anima" / "training.py"


def _section(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    return source[start_index: source.index(end, start_index)]


def test_deferred_sample_decode_runs_on_interrupted_training_loop() -> None:
    source = TRAIN_PY.read_text(encoding="utf-8")
    train_tail = source[
        source.index("with run_scope(self.progress_sink")
        : source.index("if is_main_process and (args.save_state", source.index("with run_scope(self.progress_sink"))
    ]

    assert "training_loop_completed = False" in train_tail
    assert "finally:" in train_tail
    assert "if not training_loop_completed:" in train_tail
    assert "_decode_deferred_samples_safely(" in train_tail
    assert "optimizer_eval_fn=optimizer_eval_fn" in train_tail


def test_deferred_sample_decode_still_runs_on_normal_completion() -> None:
    source = TRAIN_PY.read_text(encoding="utf-8")
    train_tail = source[
        source.index("accelerator.end_training()")
        : source.index("if is_main_process and (args.save_state", source.index("accelerator.end_training()"))
    ]

    assert "optimizer_eval_fn()" in train_tail
    assert "_decode_deferred_samples_safely(accelerator, args, loop_state, vae)" in train_tail


def test_sigterm_uses_keyboard_interrupt_cleanup_path() -> None:
    source = TRAIN_PY.read_text(encoding="utf-8")

    assert "signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)" in source
    assert "_install_stop_signal_handlers()" in source


def test_live_sample_preview_decode_runs_after_latent_save() -> None:
    source = ANIMA_TRAINING.read_text(encoding="utf-8")
    sample_body = source[
        source.index("def sample_images(")
        : source.index("def _sample_image_inference(")
    ]

    assert "saved_latents: list[str] = []" in sample_body
    assert "saved_latents.append(latent_path)" in sample_body
    assert "decode_samples_for_live_preview(accelerator, args, vae" in sample_body


def test_live_sample_preview_restores_block_swap_safely() -> None:
    source = ANIMA_TRAINING.read_text(encoding="utf-8")

    assert "def _restore_dit_training_device" in source
    assert "move_to_device_except_swap_blocks" in source
    assert "_restore_dit_training_device(dit, dit_device)" in source


def test_training_sample_sampler_is_consumed_by_preview_loop() -> None:
    source = ANIMA_TRAINING.read_text(encoding="utf-8")
    sample_fn = _section(source, "def do_sample(", "def sample_images(")
    inference_fn = _section(source, "def _sample_image_inference(", "def _module_device(")

    assert '_TRAINING_SAMPLE_SAMPLERS = {"euler", "er_sde", "lcm"}' in source
    assert "def normalize_training_sample_sampler" in source
    assert 'sampler: str = "euler"' in sample_fn
    assert "sampler = normalize_training_sample_sampler(sampler)" in sample_fn
    assert "inference_sampling.ERSDESampler" in sample_fn
    assert "inference_sampling.LCMSampler" in sample_fn
    assert "sampler_stepper.step(x, denoised, i)" in sample_fn
    assert 'getattr(args, "sample_sampler", "euler")' in inference_fn
    assert "sample_sampler," in inference_fn


def test_training_sample_prompts_cfg_and_dual_schedule_are_used() -> None:
    source = ANIMA_TRAINING.read_text(encoding="utf-8")
    sample_images_fn = _section(source, "def sample_images(", "def _sample_image_inference(")
    inference_fn = _section(source, "def _sample_image_inference(", "def _module_device(")

    assert 'scale = prompt_dict.get("guidance_scale", prompt_dict.get("scale", 7.5))' in inference_fn
    assert "sample_this_call = False" in sample_images_fn
    assert "if epoch is not None and args.sample_every_n_epochs is not None:" in sample_images_fn
    assert "sample_this_call = epoch % args.sample_every_n_epochs == 0" in sample_images_fn
    assert "if epoch is None and args.sample_every_n_steps is not None:" in sample_images_fn
    assert "sample_this_call = steps % args.sample_every_n_steps == 0" in sample_images_fn


def test_sample_prompts_without_schedule_do_not_trigger_qwen3_or_te_cache() -> None:
    source = TRAIN_PY.read_text(encoding="utf-8")
    cache_fn = _section(source, "def cache_text_encoder_outputs_if_needed(", "    # endregion")
    encoder_decision = _section(source, "sampling_enabled = _sample_preview_enabled(args)", "# Prepare accelerator")

    assert "def _sample_preview_enabled(args) -> bool:" in source
    assert "if text_encoders[0] is not None and _sample_preview_enabled(args):" in cache_fn
    assert "sampling_enabled = _sample_preview_enabled(args)" in encoder_decision
    assert "or sampling_enabled" in encoder_decision
    assert 'getattr(args, "sample_prompts", None)' not in encoder_decision
