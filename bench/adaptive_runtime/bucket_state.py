"""Bucket-order prototype. Structural validation is NOT cache-content identity."""

from collections import Counter
from copy import deepcopy

from library.datasets.buckets import BucketBatchIndex


def capture_buckets(group):
    datasets = []
    for dataset in group.datasets:
        manager = dataset.bucket_manager
        if manager is None or dataset.caching_mode is not None:
            raise ValueError("Expected initialized training buckets, not caching/stage mode")
        datasets.append({
            "seed": dataset.seed, "max_train_steps": dataset.max_train_steps,
            "current_epoch": dataset.current_epoch, "current_step": dataset.current_step,
            "length": len(dataset), "resos": [list(r) for r in manager.resos],
            "buckets": deepcopy(manager.buckets),
            "indices": [list(index) for index in dataset.buckets_indices],
            "largest": dataset._largest_bucket_index,
        })
    return {"schema": "bucket_order_prototype_v1", "datasets": datasets}


def _validate_candidate(expected, candidate):
    for key in ("seed", "max_train_steps", "length", "resos"):
        if candidate[key] != expected[key]:
            raise ValueError(f"Bucket structure changed: {key}")
    if len(candidate["buckets"]) != len(expected["buckets"]):
        raise ValueError("Bucket count changed")
    for saved, current in zip(candidate["buckets"], expected["buckets"]):
        # Repeated image keys are legitimate num_repeats, not aliases to reject.
        if Counter(saved) != Counter(current):
            raise ValueError("Bucket sample membership or repeats changed")
    if Counter(map(tuple, candidate["indices"])) != Counter(map(tuple, expected["indices"])):
        raise ValueError("Bucket batch inventory changed")
    for key in ("current_epoch", "current_step"):
        if type(candidate[key]) is not int or candidate[key] < 0:
            raise ValueError(f"Invalid bucket progress: {key}")
    largest = candidate["largest"]
    present = {row[0] for row in candidate["indices"]}
    if present:
        if type(largest) is not int or largest not in present:
            raise ValueError("Invalid largest bucket")
        def area(i):
            return candidate["resos"][i][0] * candidate["resos"][i][1]
        if area(largest) != max(area(i) for i in present):
            raise ValueError("Largest bucket is not maximal")


def restore_buckets(group, snapshot):
    snapshot = deepcopy(snapshot)
    current = capture_buckets(group)
    if (snapshot.get("schema") != current["schema"]
            or len(snapshot.get("datasets", [])) != len(current["datasets"])):
        raise ValueError("Bucket snapshot schema or dataset count changed")
    for expected, candidate in zip(current["datasets"], snapshot["datasets"]):
        _validate_candidate(expected, candidate)
    # Validate every member before mutating any dataset in the group.
    for dataset, saved in zip(group.datasets, snapshot["datasets"]):
        dataset.bucket_manager.buckets = saved["buckets"]
        dataset.buckets_indices = [BucketBatchIndex(*row) for row in saved["indices"]]
        dataset._largest_bucket_index = saved["largest"]
        dataset.current_epoch = saved["current_epoch"]
        dataset.current_step = saved["current_step"]
