"""Color-only caption filter for the colorization EasyControl task.

Guards :func:`color_caption.filter_to_colors` — the transform that reduces a
full Anima caption to its color tags before re-encoding into the colorize
text_cache_dir. A regression here would silently change what the colorize
adapter learns to steer on.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COLORIZE_DIR = REPO_ROOT / "easycontrol_adapters" / "colorization"
if str(COLORIZE_DIR) not in sys.path:
    sys.path.insert(0, str(COLORIZE_DIR))

from color_caption import filter_to_colors, is_color_tag  # noqa: E402


def test_keeps_hair_eye_skin_colors():
    assert is_color_tag("blue hair")
    assert is_color_tag("aqua eyes")
    assert is_color_tag("light brown hair")
    assert is_color_tag("grey eyes")
    assert is_color_tag("dark skin")
    # multi-color escapes
    assert is_color_tag("two-tone hair")
    assert is_color_tag("heterochromia")


def test_keeps_color_noun_tags():
    assert is_color_tag("yellow shirt")
    assert is_color_tag("red ribbon")
    assert is_color_tag("white background")
    assert is_color_tag("dark green dress")  # modifier + color
    assert is_color_tag("monochrome")


def test_drops_non_color_tags():
    for tag in (
        "1girl",
        "looking at viewer",
        "long hair",  # length, not color
        "closed eyes",  # state, not color
        "open mouth",
        "smile",
        "@eufoniuz",
        "ponytail",
    ):
        assert not is_color_tag(tag), tag


def test_filter_extracts_colors_in_order():
    caption = (
        "explicit, 1boy, 1girl, suou yuki, @eufoniuz, brown eyes, brown hair, "
        "looking at viewer, ponytail, smile, yellow shirt, green shorts"
    )
    out = filter_to_colors(caption)
    assert out == "brown eyes, brown hair, yellow shirt, green shorts"


def test_filter_empty_when_no_color():
    # No color tag → empty caption (trains the unconditional auto-color mode).
    assert filter_to_colors("1girl, solo, looking at viewer, smile") == ""
    assert filter_to_colors("") == ""


def test_long_hair_not_kept_but_blue_hair_is():
    # 'long hair' shares the 'hair' suffix but has no color word → dropped.
    assert filter_to_colors("long hair, blue hair") == "blue hair"
