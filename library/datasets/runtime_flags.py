"""Runtime flags shared by dataset loading helpers."""

HIGH_VRAM = False


def enable_high_vram() -> None:
    global HIGH_VRAM
    HIGH_VRAM = True
