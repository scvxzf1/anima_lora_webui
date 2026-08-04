"""Checkpoint saver / plan_resume_start tests split from test_training_resume.py."""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

# Split: checkpointing

def test_save_last_n_epochs_keeps_recent_weight_files(tmp_path):
    args = _weight_checkpoint_args(tmp_path, keep_last=2)
    saver = _weight_checkpoint_saver(args)
    network = _TinySaveNetwork()

    for epoch_no, step in ((1, 10), (2, 20), (3, 30)):
        saver.maybe_save_epoch(
            network, global_step=step, epoch=epoch_no - 1, num_train_epochs=5
        )

    assert not (tmp_path / "demo-000001.safetensors").exists()
    assert (tmp_path / "demo-000002.safetensors").exists()
    assert (tmp_path / "demo-000003.safetensors").exists()

def test_save_last_n_epochs_does_not_remove_preexisting_weight_files(tmp_path):
    old_weight = tmp_path / "demo-000001.safetensors"
    old_weight.write_bytes(b"old")
    args = _weight_checkpoint_args(tmp_path, keep_last=1)
    saver = _weight_checkpoint_saver(args)

    saver.maybe_save_epoch(
        _TinySaveNetwork(), global_step=20, epoch=1, num_train_epochs=4
    )

    assert old_weight.read_bytes() == b"old"
    assert (tmp_path / "demo-000002.safetensors").exists()

def test_save_last_n_epochs_minus_one_keeps_all_weight_files(tmp_path):
    args = _weight_checkpoint_args(tmp_path, keep_last=-1)
    saver = _weight_checkpoint_saver(args)
    network = _TinySaveNetwork()

    for epoch_no, step in ((1, 10), (2, 20), (3, 30)):
        saver.maybe_save_epoch(
            network, global_step=step, epoch=epoch_no - 1, num_train_epochs=5
        )

    assert (tmp_path / "demo-000001.safetensors").exists()
    assert (tmp_path / "demo-000002.safetensors").exists()
    assert (tmp_path / "demo-000003.safetensors").exists()


def test_final_epoch_writes_numbered_weight_file(tmp_path):
    args = _weight_checkpoint_args(tmp_path, keep_last=-1)
    saver = _weight_checkpoint_saver(args)

    saver.maybe_save_epoch(
        _TinySaveNetwork(), global_step=60, epoch=5, num_train_epochs=6
    )

    assert (tmp_path / "demo-000006.safetensors").exists()

def test_save_checkpoint_state_replaces_state_after_success(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 1}),
        encoding="utf-8",
    )

    save_checkpoint_state(_checkpoint_args(tmp_path), _FakeAccelerator(step=7))

    assert json.loads((state_dir / "train_state.json").read_text(encoding="utf-8"))["current_step"] == 7
    assert not (tmp_path / "demo-checkpoint-state.tmp").exists()
    assert not (tmp_path / "demo-checkpoint-state.backup").exists()

def test_save_checkpoint_state_keeps_old_state_on_failure(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 1}),
        encoding="utf-8",
    )

    try:
        save_checkpoint_state(_checkpoint_args(tmp_path), _FakeAccelerator(step=8, fail=True))
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("save_state 失败时应继续抛出异常")

    assert json.loads((state_dir / "train_state.json").read_text(encoding="utf-8"))["current_step"] == 1
    assert not (tmp_path / "demo-checkpoint-state.tmp").exists()
    assert not (tmp_path / "demo-checkpoint-state.backup").exists()

def test_save_checkpoint_state_recovers_leftover_backup(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    backup_dir = tmp_path / "demo-checkpoint-state.backup"
    tmp_dir = tmp_path / "demo-checkpoint-state.tmp"
    backup_dir.mkdir()
    tmp_dir.mkdir()
    (backup_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 3}),
        encoding="utf-8",
    )
    (tmp_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 99}),
        encoding="utf-8",
    )

    save_checkpoint_state(_checkpoint_args(tmp_path), _FakeAccelerator(step=4))

    assert json.loads((state_dir / "train_state.json").read_text(encoding="utf-8"))["current_step"] == 4
    assert not backup_dir.exists()
    assert not tmp_dir.exists()

def test_save_checkpoint_state_writes_latest_marker_on_main_process_only(tmp_path):
    args = _checkpoint_args(tmp_path)

    save_checkpoint_state(
        args, _FakeAccelerator(step=5, is_main_process=False), epoch_no=1
    )
    assert (tmp_path / "demo-checkpoint-000001-state" / "train_state.json").exists()
    assert not (tmp_path / "demo-checkpoint-latest.json").exists()

    save_checkpoint_state(
        args, _FakeAccelerator(step=6, is_main_process=True), epoch_no=2
    )
    marker = json.loads(
        (tmp_path / "demo-checkpoint-latest.json").read_text(encoding="utf-8")
    )
    assert marker["state_dir"] == str((tmp_path / "demo-checkpoint-000002-state").resolve())
    assert marker["epoch"] == 2

def test_checkpointing_last_n_epochs_keeps_recent_resumable_states(tmp_path):
    args = _checkpointing_args(tmp_path, keep_last=2)
    accelerator = _FakeAccelerator()
    saver = CheckpointSaver(
        args=args,
        accelerator=accelerator,
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda _args: {},
        current_epoch=SimpleNamespace(value=0),
        current_step=SimpleNamespace(value=0),
    )
    network = _TinySaveNetwork()

    for epoch_no, step in ((1, 10), (2, 20), (3, 30)):
        accelerator.step = step
        saver.maybe_save_resumable(
            network, global_step=step, epoch=epoch_no - 1, num_train_epochs=5
        )

    assert not (tmp_path / "demo-checkpoint-000001-state").exists()
    assert not (tmp_path / "demo-checkpoint-000001.safetensors").exists()
    assert (tmp_path / "demo-checkpoint-000002-state" / "train_state.json").exists()
    assert (tmp_path / "demo-checkpoint-000002.safetensors").exists()
    latest_state = tmp_path / "demo-checkpoint-000003-state"
    assert latest_state.exists()
    assert json.loads((latest_state / "train_state.json").read_text(encoding="utf-8"))["current_step"] == 30

    resume_args = _checkpointing_args(tmp_path, keep_last=2)
    _resume_saver(resume_args).auto_resume()

    assert resume_args.resume == str(latest_state.resolve())
    assert resume_args.skip_until_initial_step is True

    saver.cleanup_resumable()
    completed_args = _checkpointing_args(tmp_path, keep_last=2)
    _resume_saver(completed_args).auto_resume()
    assert completed_args.resume is None
    assert latest_state.exists()


def test_final_epoch_writes_numbered_resumable_checkpoint(tmp_path):
    args = _checkpointing_args(tmp_path, keep_last=1)
    accelerator = _FakeAccelerator(step=60)
    saver = CheckpointSaver(
        args=args,
        accelerator=accelerator,
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda _args: {},
        current_epoch=SimpleNamespace(value=6),
        current_step=SimpleNamespace(value=59),
    )

    saver.maybe_save_resumable(
        _TinySaveNetwork(), global_step=60, epoch=5, num_train_epochs=6
    )

    state_dir = tmp_path / "demo-checkpoint-000006-state"
    assert (tmp_path / "demo-checkpoint-000006.safetensors").exists()
    state = json.loads((state_dir / "train_state.json").read_text(encoding="utf-8"))
    assert state["current_step"] == 60

def test_checkpointing_cleanup_does_not_remove_preexisting_resume_points(tmp_path):
    old_state = tmp_path / "demo-checkpoint-000001-state"
    old_state.mkdir()
    (old_state / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 10}),
        encoding="utf-8",
    )
    old_weight = tmp_path / "demo-checkpoint-000001.safetensors"
    old_weight.write_bytes(b"old")
    args = _checkpointing_args(tmp_path, keep_last=1)
    accelerator = _FakeAccelerator(step=20)
    saver = CheckpointSaver(
        args=args,
        accelerator=accelerator,
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda _args: {},
        current_epoch=SimpleNamespace(value=0),
        current_step=SimpleNamespace(value=0),
    )

    saver.maybe_save_resumable(
        _TinySaveNetwork(), global_step=20, epoch=1, num_train_epochs=4
    )

    assert (old_state / "train_state.json").exists()
    assert old_weight.read_bytes() == b"old"
    assert (tmp_path / "demo-checkpoint-000002-state" / "train_state.json").exists()
    assert (tmp_path / "demo-checkpoint-000002.safetensors").exists()

def test_checkpointing_last_n_epochs_minus_one_keeps_all_resumable_states(tmp_path):
    args = _checkpointing_args(tmp_path, keep_last=-1)
    accelerator = _FakeAccelerator()
    saver = CheckpointSaver(
        args=args,
        accelerator=accelerator,
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda _args: {},
        current_epoch=SimpleNamespace(value=0),
        current_step=SimpleNamespace(value=0),
    )
    network = _TinySaveNetwork()

    for epoch_no, step in ((1, 10), (2, 20), (3, 30)):
        accelerator.step = step
        saver.maybe_save_resumable(
            network, global_step=step, epoch=epoch_no - 1, num_train_epochs=5
        )

    assert (tmp_path / "demo-checkpoint-000001-state").exists()
    assert (tmp_path / "demo-checkpoint-000002-state").exists()
    assert (tmp_path / "demo-checkpoint-000003-state").exists()

def test_auto_resume_skips_incompatible_network_state(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 3}),
        encoding="utf-8",
    )
    save_file({"lora_down.weight": torch.zeros(1)}, str(state_dir / "model.safetensors"))
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="demo",
        checkpointing_epochs=1,
        resume=None,
        max_train_steps=10,
        skip_until_initial_step=False,
    )

    _resume_saver(args).auto_resume(_TinyResumeNetwork())

    assert args.resume is None
    assert args.skip_until_initial_step is False

def test_auto_resume_uses_compatible_network_state(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 3}),
        encoding="utf-8",
    )
    save_file({"weight": torch.zeros(1)}, str(state_dir / "model.safetensors"))
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="demo",
        checkpointing_epochs=1,
        resume=None,
        max_train_steps=10,
        skip_until_initial_step=False,
    )

    _resume_saver(args).auto_resume(_TinyResumeNetwork())

    assert args.resume == str(state_dir)
    assert args.skip_until_initial_step is True

def test_cleanup_resumable_keeps_explicit_resume_state(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 3}),
        encoding="utf-8",
    )
    latest_marker = tmp_path / "demo-checkpoint-latest.json"
    latest_marker.write_text(json.dumps({"state_dir": str(state_dir)}), encoding="utf-8")
    checkpoint_weight = tmp_path / "demo-checkpoint.safetensors"
    checkpoint_weight.write_bytes(b"checkpoint")
    args = _checkpointing_args(tmp_path, keep_last=2)
    args.resume = str(state_dir)

    _resume_saver(args).cleanup_resumable()

    assert (state_dir / "train_state.json").exists()
    assert latest_marker.exists()
    assert checkpoint_weight.exists()

def test_cleanup_resumable_keeps_preexisting_legacy_checkpoint_files(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 3}),
        encoding="utf-8",
    )
    latest_marker = tmp_path / "demo-checkpoint-latest.json"
    latest_marker.write_text(json.dumps({"state_dir": str(state_dir)}), encoding="utf-8")
    checkpoint_weight = tmp_path / "demo-checkpoint.safetensors"
    checkpoint_weight.write_bytes(b"checkpoint")
    args = _checkpointing_args(tmp_path, keep_last=2)

    _resume_saver(args).cleanup_resumable()

    assert (state_dir / "train_state.json").exists()
    assert latest_marker.exists()
    assert checkpoint_weight.exists()

def test_plan_resume_start_uses_steps_from_state():
    args = SimpleNamespace(
        initial_epoch=None,
        initial_step=None,
        gradient_accumulation_steps=2,
        max_train_steps=100,
        skip_until_initial_step=True,
        resume="state-dir",
    )

    plan = plan_resume_start(
        args,
        steps_from_state=8,
        batches_per_epoch=10,
        num_processes=1,
    )

    assert plan.initial_step == 16
    assert plan.epoch_to_start == 3
    assert plan.steps_from_state is None

def test_plan_resume_start_auto_enables_skip_for_resume_state():
    args = SimpleNamespace(
        initial_epoch=None,
        initial_step=None,
        gradient_accumulation_steps=2,
        max_train_steps=100,
        skip_until_initial_step=False,
        resume="state-dir",
    )

    plan = plan_resume_start(
        args,
        steps_from_state=8,
        batches_per_epoch=10,
        num_processes=1,
    )

    assert args.skip_until_initial_step is True
    assert plan.initial_step == 16
    assert plan.epoch_to_start == 3
    assert plan.steps_from_state is None

def test_plan_resume_start_rejects_completed_resume_step():
    args = SimpleNamespace(
        initial_epoch=None,
        initial_step=None,
        gradient_accumulation_steps=1,
        max_train_steps=8,
        skip_until_initial_step=True,
        resume="state-dir",
    )

    with pytest.raises(ValueError, match="恢复点已训练到 step 8"):
        plan_resume_start(
            args,
            steps_from_state=8,
            batches_per_epoch=10,
            num_processes=1,
        )

def test_plan_resume_start_initial_step_overrides_state():
    args = SimpleNamespace(
        initial_epoch=None,
        initial_step=6,
        gradient_accumulation_steps=2,
        max_train_steps=100,
        skip_until_initial_step=False,
        resume="state-dir",
    )

    plan = plan_resume_start(
        args,
        steps_from_state=42,
        batches_per_epoch=10,
        num_processes=1,
    )

    assert plan.initial_step == 0
    assert plan.epoch_to_start == 1
    assert plan.steps_from_state == 42

def test_plan_resume_start_skip_until_initial_step_scales_by_grad_accum():
    args = SimpleNamespace(
        initial_epoch=3,
        initial_step=None,
        gradient_accumulation_steps=3,
        max_train_steps=100,
        skip_until_initial_step=True,
        resume=None,
    )

    plan = plan_resume_start(
        args,
        steps_from_state=None,
        batches_per_epoch=12,
        num_processes=2,
    )

    assert plan.initial_step == 12
    assert plan.epoch_to_start == 3


def test_plan_resume_start_stage_uses_full_update_budget():
    args = SimpleNamespace(
        initial_epoch=None,
        initial_step=None,
        gradient_accumulation_steps=2,
        max_train_steps=200,
        skip_until_initial_step=False,
        resume="state-dir",
    )

    plan = plan_resume_start(
        args,
        steps_from_state=50,
        batches_per_epoch=10,
        num_processes=1,
        updates_per_epoch=100,
    )

    assert args.skip_until_initial_step is True
    assert plan.initial_step == 0
    assert plan.epoch_to_start == 0
    assert plan.global_step == 50


def test_plan_resume_start_stage_exact_epoch_boundary():
    args = SimpleNamespace(
        initial_epoch=None,
        initial_step=None,
        gradient_accumulation_steps=1,
        max_train_steps=300,
        skip_until_initial_step=True,
        resume="state-dir",
    )

    plan = plan_resume_start(
        args,
        steps_from_state=100,
        batches_per_epoch=10,
        num_processes=2,
        updates_per_epoch=100,
    )

    assert plan.initial_step == 0
    assert plan.epoch_to_start == 1
    assert plan.global_step == 100


def test_plan_resume_start_stage_initial_epoch_keeps_progress_aligned():
    args = SimpleNamespace(
        initial_epoch=3,
        initial_step=None,
        gradient_accumulation_steps=2,
        max_train_steps=400,
        skip_until_initial_step=False,
        resume=None,
    )

    plan = plan_resume_start(
        args,
        steps_from_state=None,
        batches_per_epoch=10,
        num_processes=1,
        updates_per_epoch=100,
    )

    assert plan.initial_step == 0
    assert plan.epoch_to_start == 2
    assert plan.global_step == 200
