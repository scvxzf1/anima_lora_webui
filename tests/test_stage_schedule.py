"""Unit tests for percent-based stage schedule helpers."""

from __future__ import annotations

from types import SimpleNamespace

from library.training.stage_schedule import (
    StageSpec,
    active_subset_indices_for_step,
    normalize_stage_dicts,
    parse_stage_specs,
    progress_from_steps,
    resolve_stage_index,
    stage_schedule_enabled,
    validate_stage_specs,
)


def test_normalize_accepts_percent_0_100_and_fraction():
    stages = normalize_stage_dicts(
        [
            {"name": "a", "subset_index": 0, "start_pct": 0, "end_pct": 50},
            {"name": "b", "subset_index": 1, "start_pct": 0.5, "end_pct": 1.0},
        ]
    )
    assert stages[0]["start_pct"] == 0.0
    assert stages[0]["end_pct"] == 0.5
    assert stages[1]["start_pct"] == 0.5
    assert stages[1]["end_pct"] == 1.0


def test_validate_two_stage_cover():
    specs = parse_stage_specs(
        [
            {"subset_index": 0, "start_pct": 0.0, "end_pct": 0.4},
            {"subset_index": 1, "start_pct": 0.4, "end_pct": 1.0},
        ]
    )
    assert validate_stage_specs(specs, subset_count=2) == []


def test_validate_rejects_gap_and_bad_subset():
    specs = parse_stage_specs(
        [
            {"subset_index": 0, "start_pct": 0.0, "end_pct": 0.3},
            {"subset_index": 9, "start_pct": 0.5, "end_pct": 1.0},
        ]
    )
    problems = validate_stage_specs(specs, subset_count=2)
    assert any("贴齐" in p for p in problems)
    assert any("subset_index" in p for p in problems)


def test_resolve_stage_index_boundaries():
    specs = [
        StageSpec(0, 0.0, 0.5, "low"),
        StageSpec(1, 0.5, 1.0, "high"),
    ]
    assert resolve_stage_index(specs, 0.0) == 0
    assert resolve_stage_index(specs, 0.499) == 0
    assert resolve_stage_index(specs, 0.5) == 1
    assert resolve_stage_index(specs, 1.0) == 1


def test_active_subset_for_step():
    args = SimpleNamespace(
        stage_schedule_enabled=True,
        max_train_steps=1000,
        stage_schedule=[
            {"subset_index": 0, "start_pct": 0.0, "end_pct": 0.3},
            {"subset_index": 2, "start_pct": 0.3, "end_pct": 1.0},
        ],
    )
    assert stage_schedule_enabled(args)
    assert active_subset_indices_for_step(args, 0) == {0}
    assert active_subset_indices_for_step(args, 299) == {0}
    assert active_subset_indices_for_step(args, 300) == {2}
    assert progress_from_steps(300, 1000) == 0.3


def test_disabled_when_flag_off():
    args = SimpleNamespace(
        stage_schedule_enabled=False,
        max_train_steps=100,
        stage_schedule=[{"subset_index": 0, "start_pct": 0, "end_pct": 1}],
    )
    assert not stage_schedule_enabled(args)
    assert active_subset_indices_for_step(args, 50) is None


class _FakeInfo:
    def __init__(self, key: str, repeats: int = 1, is_reg: bool = False):
        self.image_key = key
        self.num_repeats = repeats
        self.is_reg = is_reg
        self.bucket_reso = (64, 64)
        self.image_size = (64, 64)
        self.resized_size = (64, 64)
        self.absolute_path = key
        self.mask_path = None
        self.preloaded_alpha_mask = None
        self.resize_interpolation = None


class _FakeSubset:
    def __init__(self, name: str):
        self.name = name
        self.sample_ratio = 1.0
        self.random_crop = False


class _FakeBucketDataset:
    """Minimal stand-in for DatasetBucketsMixin rebuild/snapshot semantics."""

    def __init__(self):
        self.subsets = [_FakeSubset("a"), _FakeSubset("b")]
        self.image_data = {
            "a1": _FakeInfo("a1"),
            "a2": _FakeInfo("a2"),
            "b1": _FakeInfo("b1"),
        }
        self.image_to_subset = {
            "a1": self.subsets[0],
            "a2": self.subsets[0],
            "b1": self.subsets[1],
        }
        self._all_image_data = None
        self._all_image_to_subset = None
        self._constant_token_buckets = True
        self.bucket_manager = None
        self._largest_bucket_index = None
        self.num_train_images = 3
        self.num_reg_images = 0
        self._length = 3
        self.seed = 0
        self.current_epoch = 0
        self.batch_size = 1
        self.enable_bucket = False
        self.resolution = 64
        self.min_bucket_reso = 64
        self.max_bucket_reso = 64
        self.bucket_reso_steps = 64
        self.bucket_no_upscale = False
        self.buckets_indices = []

    def snapshot_full_image_data(self, *, force: bool = False):
        if not force and self._all_image_data is not None:
            return
        self._all_image_data = dict(self.image_data)
        self._all_image_to_subset = dict(self.image_to_subset)

    def has_full_image_data_snapshot(self) -> bool:
        return self._all_image_data is not None

    def make_buckets(self, constant_token_buckets: bool = False):
        # Lightweight: one index entry per image key (no real bucket manager).
        self.buckets_indices = list(self.image_data.keys())
        self._length = len(self.buckets_indices)

    def rebuild_buckets_for_subsets(self, active_subset_indices=None) -> bool:
        # Mirror production: filter from snapshot, not from already-filtered live map.
        if not self.has_full_image_data_snapshot():
            if not self.image_data:
                return False
            self.snapshot_full_image_data()
        if active_subset_indices is None:
            allowed = None
        else:
            allowed_ids = {int(i) for i in active_subset_indices}
            allowed = {
                subset
                for index, subset in enumerate(self.subsets)
                if index in allowed_ids
            }
            if not allowed:
                return False
        source = self._all_image_data
        source_map = self._all_image_to_subset
        if allowed is None:
            filtered = source
            filtered_map = source_map
        else:
            filtered = {
                k: v for k, v in source.items() if source_map.get(k) in allowed
            }
            filtered_map = {k: source_map[k] for k in filtered}
        if not filtered:
            return False
        self.image_data = dict(filtered)
        self.image_to_subset = dict(filtered_map)
        self.num_train_images = len(filtered)
        self.make_buckets()
        return True


def test_snapshot_before_filter_allows_later_stage_recovery():
    """P0 regression: filtering stage0 first must not freeze stage0 as full set."""
    from library.training.stage_schedule import (
        apply_active_subsets_to_dataset,
        snapshot_full_image_data,
    )

    ds = _FakeBucketDataset()
    # Correct order: snapshot full map, then filter stage0, then stage1.
    snapshot_full_image_data(ds, force=True)
    assert apply_active_subsets_to_dataset(ds, {0}) is True
    assert set(ds.image_data) == {"a1", "a2"}
    assert apply_active_subsets_to_dataset(ds, {1}) is True
    assert set(ds.image_data) == {"b1"}
    # Restore full.
    assert apply_active_subsets_to_dataset(ds, None) is True
    assert set(ds.image_data) == {"a1", "a2", "b1"}


def test_partial_snapshot_blocks_later_stage_recovery():
    """If the frozen map was captured after a partial filter, later stages fail."""
    from library.training.stage_schedule import apply_active_subsets_to_dataset

    ds = _FakeBucketDataset()
    # Corrupt order: shrink live map, then force-snapshot that partial map.
    ds.image_data = {"a1": ds.image_data["a1"], "a2": ds.image_data["a2"]}
    ds.image_to_subset = {
        "a1": ds.image_to_subset["a1"],
        "a2": ds.image_to_subset["a2"],
    }
    ds.snapshot_full_image_data(force=True)
    assert set(ds._all_image_data) == {"a1", "a2"}
    assert apply_active_subsets_to_dataset(ds, {1}) is False
    assert set(ds.image_data) == {"a1", "a2"}
