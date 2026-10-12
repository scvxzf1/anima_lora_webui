"""Exception-safe block swap transitions for consecutive inference forwards."""

from contextlib import contextmanager


@contextmanager
def inference_block_swap(model):
    """Reuse forward-only swapping and restore the entry mode on exit.

    Switch methods drain pending transfers and prepare a canonical resident
    layout. Recovery restores that layout, rather than a rotated physical
    layout from an interrupted forward. Model and adapter train/eval state is
    owned by the caller.
    """
    if not getattr(model, "blocks_to_swap", 0):
        yield
        return

    offloader = getattr(model, "offloader", None)
    original_mode = getattr(offloader, "forward_only", None)
    switch_inference = getattr(model, "switch_block_swap_for_inference", None)
    switch_training = getattr(model, "switch_block_swap_for_training", None)
    if (
        offloader is None
        or not isinstance(original_mode, bool)
        or not callable(switch_inference)
        or not callable(switch_training)
    ):
        raise RuntimeError(
            "Inference block swap requires an offloader with a boolean "
            "forward_only mode and inference/training switch methods"
        )

    restore = switch_inference if original_mode else switch_training
    error = None
    try:
        # Preparation is part of the transition and can fail after mutation.
        switch_inference()
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            restore()
        except BaseException as exc:
            if error is None:
                raise
            error.add_note(f"Inference block swap restoration failed: {exc!r}")
