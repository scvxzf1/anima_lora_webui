"""Shared resize target selection for preprocessing and read-only estimates."""

from library.datasets.buckets import BucketManager


def select_resize_bucket(
    manager: BucketManager,
    width: int,
    height: int,
    *,
    enable_bucket: bool = True,
    bucket_no_upscale: bool = False,
) -> tuple[int, int]:
    if not enable_bucket:
        return manager.max_reso
    bucket, _, _ = manager.select_bucket(width, height)
    if bucket_no_upscale and (bucket[0] > width or bucket[1] > height):
        candidates = [
            reso for reso in manager.predefined_resos
            if reso[0] <= width and reso[1] <= height
        ]
        if candidates:
            aspect = width / height
            bucket = min(
                candidates,
                key=lambda reso: (abs(reso[0] / reso[1] - aspect), -reso[0] * reso[1]),
            )
        else:
            bucket = (
                max(manager.min_size, (min(bucket[0], width) // manager.reso_steps) * manager.reso_steps),
                max(manager.min_size, (min(bucket[1], height) // manager.reso_steps) * manager.reso_steps),
            )
        manager.add_if_new_reso(bucket)
    return bucket
