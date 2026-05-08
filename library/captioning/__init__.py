"""Image-to-caption helpers used by editing/inversion paths.

Two taggers ship: :class:`WDTagger` (SmilingWolf/wd-swinv2-tagger-v3 — generic
booru-distribution tagger, used as the case-1 fallback) and
:class:`AnimaTagger` (trained on the Anima caption distribution; the default
ψ_src provider for DirectEdit when a checkpoint is present at
``models/captioners/anima-tagger-v1/``).

Both expose the same surface: ``predict(pil_img)`` and
``predict_caption(pil_img)`` for a comma-separated tag string.
"""

from library.captioning.wd_tagger import WDTagger

# AnimaTagger pulls in PE-Core lazily on import (it doesn't load weights
# until first predict), but the import itself touches torch/safetensors —
# keep it eager so that ``from library.captioning import AnimaTagger`` works
# but in environments without a built checkpoint, callers handle the
# ``FileNotFoundError`` from ``AnimaTagger.__init__``.
from library.captioning.anima_tagger import AnimaTagger

__all__ = ["WDTagger", "AnimaTagger"]
