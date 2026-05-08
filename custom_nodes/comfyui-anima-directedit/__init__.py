"""Anima DirectEdit ComfyUI custom node.

Single node (``AnimaDirectEdit``) that takes an image and an
"edit text" (tags to add), runs the AnimaTagger to derive ``psi_src``,
forms ``psi_tar = psi_src + ", " + edit_text``, then invokes the
DirectEdit invert + edit_forward primitives on a freshly-loaded Anima DiT
to produce an edited image. Loads DiT / TE / VAE / tagger from disk per
invocation; does NOT consume ComfyUI's MODEL / CLIP / VAE handles
(the underlying primitives target ``library/anima/models.py::Anima``,
which has a different forward signature than ``comfy.ldm.anima.model.Anima``).
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
