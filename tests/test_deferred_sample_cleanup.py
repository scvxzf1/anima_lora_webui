from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"
ANIMA_TRAINING = ROOT / "library" / "anima" / "training.py"


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
