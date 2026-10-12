"""Static contract shared by the payload, runtime, and BF16 comparison."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NMConfig:
    group_size: int = 128
    mask_mode: str = "bidirectional"
    backend: str = "reference"
    cache_sparse: bool = False

    def __post_init__(self):
        if self.group_size < 4 or self.group_size % 4:
            raise ValueError("group_size must be a positive multiple of four")
        if self.mask_mode not in {"forward", "bidirectional"}:
            raise ValueError("mask_mode must be forward or bidirectional")
        if self.backend not in {"reference", "cusparselt"}:
            raise ValueError("backend must be reference or cusparselt")
        if self.cache_sparse and self.backend != "cusparselt":
            raise ValueError("cache_sparse is only valid for cusparselt")

    def validate_shape(self, out_features: int, in_features: int):
        if out_features <= 0 or in_features <= 0:
            raise ValueError("Linear dimensions must be positive")
        if out_features % 4 or in_features % self.group_size or in_features % 8:
            raise ValueError("out_features must divide by 4; in_features by group_size and 8")
        if self.backend == "cusparselt":
            if self.mask_mode != "bidirectional":
                raise ValueError("cusparselt requires a transpose-safe bidirectional mask")
            if out_features % 16 or in_features % 16:
                raise ValueError("cusparselt requires both weight dimensions divisible by 16")
