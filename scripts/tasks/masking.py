"""Mask generation: SAM3 + MIT/ComicTextDetector → merged.

Outputs to masks/{sam,mit,merged}/. ``cmd_mask`` runs SAM and MIT only if
their per-tool dirs are missing, then always runs the merge.
"""

from __future__ import annotations

import os
import shutil

from ._common import PY, ROOT, run


def cmd_mask_sam(extra):
    run(
        [
            PY,
            "preprocess/generate_masks.py",
            "--config",
            "configs/sam_mask.yaml",
            "--image-dir",
            "post_image_dataset/resized",
            "--mask-dir",
            "masks/sam",
            "--checkpoint",
            "models/sam3/sam3.pt",
            "--batch-size",
            "2",
            *extra,
        ]
    )


def cmd_mask_mit(extra):
    # MIT_TEXT_THRESHOLD / MIT_DILATE let the GUI's Preprocessing tab tune
    # the MIT masker without editing this file. Defaults match the script's
    # own argparse defaults so direct CLI use is unchanged.
    cmd = [
        PY,
        "preprocess/generate_masks_mit.py",
        "--image-dir",
        "post_image_dataset/resized",
        "--mask-dir",
        "masks/mit",
        "--model-path",
        "models/mit/model.pth",
    ]
    text_threshold = os.environ.get("MIT_TEXT_THRESHOLD")
    if text_threshold:
        cmd += ["--text-threshold", text_threshold]
    dilate = os.environ.get("MIT_DILATE")
    if dilate:
        cmd += ["--dilate", dilate]
    cmd += list(extra)
    run(cmd)


def cmd_mask(extra):
    if not (ROOT / "masks" / "sam").is_dir():
        cmd_mask_sam([])
    if not (ROOT / "masks" / "mit").is_dir():
        cmd_mask_mit([])
    run(
        [
            PY,
            "preprocess/merge_masks.py",
            "masks/sam",
            "masks/mit",
            "--output-dir",
            "masks/merged",
            *extra,
        ]
    )


def cmd_mask_clean(_extra):
    p = ROOT / "masks"
    if p.exists():
        shutil.rmtree(p)
        print("  Removed masks/")
