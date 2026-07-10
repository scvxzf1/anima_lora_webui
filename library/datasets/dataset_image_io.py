"""Image size / face-info loading helpers for BaseDataset.

Extracted from the core dataset class so image IO stays separate from caption
and batch-assembly logic.
"""

from __future__ import annotations

import logging
import os

import imagesize
from PIL import Image

from library.datasets.image_utils import load_image
from library.datasets.subsets import BaseSubset

logger = logging.getLogger(__name__)


class DatasetImageIOMixin:
    """Mixin providing image size probing and face-crop metadata loading."""

    def get_image_size(self, image_path):
        if image_path.endswith(".jxl") or image_path.endswith(".JXL"):
            from library.jpeg_xl_util import get_jxl_size

            return get_jxl_size(image_path)
        image_size = imagesize.get(image_path)
        if image_size[0] <= 0:
            try:
                with Image.open(image_path) as img:
                    image_size = img.size
            except Exception as e:
                logger.warning(f"failed to get image size: {image_path}, error: {e}")
                image_size = (0, 0)
        return image_size

    def load_image_with_face_info(
        self, subset: BaseSubset, image_path: str, alpha_mask=False
    ):
        img = load_image(image_path, alpha_mask)

        face_cx = face_cy = face_w = face_h = 0
        if subset.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h
