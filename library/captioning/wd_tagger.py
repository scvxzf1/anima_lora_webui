"""SmilingWolf wd-swinv2-tagger-v3 wrapper (timm + safetensors).

Multi-label booru-style tagger (https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3).
Used as the case-1 fallback captioner for DirectEdit when an external image
arrives without a recorded ψ_src — produces a comma-separated tag string that
gets fed through T5 to seed inversion.

Usage:
    from library.captioning.wd_tagger import WDTagger
    tagger = WDTagger()                        # downloads on first use, cached
    caption = tagger.predict_caption(pil_img)  # → "1girl, smile, school_uniform, ..."

Loads via ``timm.create_model("hf_hub:SmilingWolf/wd-swinv2-tagger-v3", pretrained=True)``
which pulls ``model.safetensors`` + ``config.json`` from HF and assembles the
right SwinV2 architecture. The tag CSV is fetched alongside via ``hf_hub_download``.

Preprocessing matches SmilingWolf's published pipeline: square-pad with white
fill, bicubic resize, RGB→BGR, kept in 0–255 range (no per-channel mean/std
normalization). The ONNX model in the same repo uses NHWC, the timm/torch
model uses NCHW — content is identical.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

REPO_ID = "SmilingWolf/wd-swinv2-tagger-v3"
TAGS_FILENAME = "selected_tags.csv"

# Default per-category confidence thresholds. Match the values
# SmilingWolf's reference Gradio space ships with.
DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85

# Tag categories in selected_tags.csv. The CSV's `category` column is integer:
#   0 = general, 4 = character, 9 = rating
_CATEGORY_GENERAL = 0
_CATEGORY_CHARACTER = 4
_CATEGORY_RATING = 9

# Tags the model fires on too eagerly for the Anima dataset (manga / stylized
# anime). The bench at bench/wd_tagger_dataset/ shows these as the dominant
# false positives — `no_humans` is plain wrong on character art, the two
# "background" tags overfit to manga page artifacts (white pages → predicted
# as transparent/simple). Filtered by default at the predict() boundary so
# both predict() and predict_caption() get a clean output. Pass
# ``tag_blocklist=()`` (or any custom iterable) to override.
DEFAULT_TAG_BLOCKLIST: Tuple[str, ...] = (
    "no_humans",
    "transparent_background",
    "greyscale",
    "monochrome"
)


@dataclass
class _TagSchema:
    names: List[str]
    rating_idx: List[int]
    general_idx: List[int]
    character_idx: List[int]


def _local_dir() -> Path:
    """Local cache directory for the tagger files (mirrors `models/` convention)."""
    # ROOT = anima_lora/
    root = Path(__file__).resolve().parents[2]
    return root / "models" / "captioners" / "wd-swinv2-tagger-v3"


def _download_tags_csv() -> Path:
    """Resolve `selected_tags.csv` from the local cache or pull from HF on first use.

    The model + config download is handled by timm's HF integration; this is
    the one file timm doesn't fetch for us.
    """
    local = _local_dir() / TAGS_FILENAME
    if local.is_file():
        return local

    from huggingface_hub import hf_hub_download

    local.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"WDTagger: downloading {TAGS_FILENAME} from {REPO_ID}")
    fetched = hf_hub_download(
        repo_id=REPO_ID,
        filename=TAGS_FILENAME,
        local_dir=str(local.parent),
    )
    return Path(fetched)


def _load_tag_schema(csv_path: Path) -> _TagSchema:
    names: List[str] = []
    rating_idx: List[int] = []
    general_idx: List[int] = []
    character_idx: List[int] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            names.append(row["name"])
            cat = int(row["category"])
            if cat == _CATEGORY_RATING:
                rating_idx.append(i)
            elif cat == _CATEGORY_CHARACTER:
                character_idx.append(i)
            elif cat == _CATEGORY_GENERAL:
                general_idx.append(i)
    return _TagSchema(
        names=names,
        rating_idx=rating_idx,
        general_idx=general_idx,
        character_idx=character_idx,
    )


def _square_pad(
    img: Image.Image, fill: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Pad to square with `fill` (white by default — matches SmilingWolf's pipeline)."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    bg = Image.new("RGB", (side, side), fill)
    bg.paste(img, ((side - w) // 2, (side - h) // 2))
    return bg


class WDTagger:
    """SmilingWolf wd-swinv2-tagger-v3 inference wrapper.

    Lazy-loads the timm SwinV2 model on first ``predict``/``predict_caption``
    call. Constructing the object is cheap and safe to do at import time.
    """

    def __init__(
        self,
        general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
        character_threshold: float = DEFAULT_CHARACTER_THRESHOLD,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        tag_blocklist: Optional[Iterable[str]] = None,
    ):
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        # None → use module default; pass () to disable filtering. Normalize
        # to underscore form so callers can pass either "no humans" or
        # "no_humans" interchangeably.
        bl = DEFAULT_TAG_BLOCKLIST if tag_blocklist is None else tag_blocklist
        self.tag_blocklist: frozenset[str] = frozenset(
            t.strip().lower().replace(" ", "_") for t in bl
        )
        self._model: Optional[torch.nn.Module] = None
        self._schema: Optional[_TagSchema] = None
        self._input_size: int = 448

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import timm
        except ImportError as e:
            raise RuntimeError(
                "timm is required for WDTagger. Install with `uv add timm`."
            ) from e

        logger.info(f"WDTagger: loading {REPO_ID} via timm.create_model(pretrained=True)")
        # timm's HF integration pulls model.safetensors + config.json directly.
        # No onnxruntime needed.
        model = timm.create_model(f"hf_hub:{REPO_ID}", pretrained=True)
        model.eval()
        model.to(self._device, dtype=self._dtype)
        self._model = model

        # Resolve input resolution from the timm pretrained config (so we don't
        # hardcode 448 in case SmilingWolf ever bumps it). Falls back to 448.
        cfg = getattr(model, "pretrained_cfg", None) or {}
        input_size = cfg.get("input_size") or (3, 448, 448)
        self._input_size = int(input_size[-1])

        tags_path = _download_tags_csv()
        self._schema = _load_tag_schema(tags_path)

        n_out = (
            model.num_classes
            if hasattr(model, "num_classes")
            else len(self._schema.names)
        )
        logger.info(
            f"WDTagger: ready ({self._input_size}x{self._input_size}, "
            f"{len(self._schema.names)} tags, {n_out} model outputs, device={self._device})"
        )

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        """RGB PIL -> [1, 3, S, S] float tensor in BGR 0-255 (no mean/std normalize)."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = _square_pad(img)
        img = img.resize((self._input_size, self._input_size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32)  # H, W, C — RGB, 0..255
        arr = arr[:, :, ::-1]  # RGB -> BGR (training-time convention)
        arr = np.ascontiguousarray(arr)
        # NHWC -> NCHW
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t.to(self._device, dtype=self._dtype)

    @torch.no_grad()
    def predict(
        self,
        img: Image.Image,
    ) -> Tuple[
        List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]
    ]:
        """Return ``(rating, character, general)`` lists of (name, prob) above threshold.

        Rating is sorted descending and unfiltered (always returns 4 entries —
        general/sensitive/questionable/explicit). Character and general are
        thresholded and sorted descending.
        """
        self._ensure_loaded()
        assert self._model is not None and self._schema is not None

        x = self._preprocess(img)
        logits = self._model(x)  # [1, N_tags]
        probs = torch.sigmoid(logits)[0].float().cpu().numpy()

        names = self._schema.names

        rating = sorted(
            [(names[i], float(probs[i])) for i in self._schema.rating_idx],
            key=lambda t: t[1],
            reverse=True,
        )
        character = sorted(
            [
                (names[i], float(probs[i]))
                for i in self._schema.character_idx
                if float(probs[i]) >= self.character_threshold
                and names[i] not in self.tag_blocklist
            ],
            key=lambda t: t[1],
            reverse=True,
        )
        general = sorted(
            [
                (names[i], float(probs[i]))
                for i in self._schema.general_idx
                if float(probs[i]) >= self.general_threshold
                and names[i] not in self.tag_blocklist
            ],
            key=lambda t: t[1],
            reverse=True,
        )
        return rating, character, general

    def predict_caption(
        self,
        img: Image.Image,
        include_rating: bool = False,
        max_tags: Optional[int] = None,
    ) -> str:
        """Comma-separated caption: characters first, then general tags.

        Tag ``_`` is mapped to space (``"long_hair"`` -> ``"long hair"``) and
        parens are escaped (``"\\("``) so the string is safe to drop straight
        into a T5 prompt.
        """
        rating, character, general = self.predict(img)
        ordered: List[str] = []
        if include_rating and rating:
            ordered.append(rating[0][0])  # top rating only
        ordered += [name for name, _ in character]
        ordered += [name for name, _ in general]
        if max_tags is not None:
            ordered = ordered[:max_tags]

        def _normalize(tag: str) -> str:
            return tag.replace("_", " ").replace("(", "\\(").replace(")", "\\)")

        return ", ".join(_normalize(t) for t in ordered)


# Tiny CLI: `python -m library.captioning.wd_tagger <image>` for smoke testing.
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m library.captioning.wd_tagger <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    tagger = WDTagger()
    caption = tagger.predict_caption(Image.open(path))
    print(caption)
