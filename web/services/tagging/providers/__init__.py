"""Optional local WD14/CLTagger providers."""

from .base import LocalTaggingError, ProgressFn, TagResult, Tagger
from .factory import get_tagger
from .onnx_base import OnnxTaggerBase, onnxruntime_status, silenced_fd_stderr

__all__ = [
    "CLTagger",
    "LocalTaggingError",
    "OnnxTaggerBase",
    "ProgressFn",
    "TagResult",
    "Tagger",
    "WD14Tagger",
    "get_tagger",
    "onnxruntime_status",
    "silenced_fd_stderr",
]


def __getattr__(name: str):
    if name == "WD14Tagger":
        from .wd14 import WD14Tagger

        return WD14Tagger
    if name == "CLTagger":
        from .cltagger import CLTagger

        return CLTagger
    raise AttributeError(name)
