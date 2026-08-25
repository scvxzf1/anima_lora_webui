"""Fail-closed model-family preparation for the inference entrypoint."""

from __future__ import annotations

from library.models.family_registry import dispatch_model_family, get_model_family_spec


def _validate_anima(args, *, mode: str) -> None:
    return None


def _validate_krea2(args, *, mode: str) -> None:
    from library.models.krea2_raw import inference_runner

    inference_runner.validate_krea2_inference_args(args, mode=mode)


def _validate_z_image(args, *, mode: str) -> None:
    raise SystemExit("Z-Image inference is not implemented in the training v1 path.")


def _install_anima_strategies(args) -> None:
    from library.anima import strategy as strategy_anima
    from library.anima import text_strategies
    from library.inference.text import MAX_CROSSATTN_TOKENS

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.text_encoder,
        t5_tokenizer_path=None,
        qwen3_max_length=MAX_CROSSATTN_TOKENS,
        t5_max_length=MAX_CROSSATTN_TOKENS,
    )
    text_strategies.TokenizeStrategy.set_strategy(tokenize_strategy)
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    text_strategies.TextEncodingStrategy.set_strategy(encoding_strategy)


def _install_krea2_strategies(args) -> None:
    # Krea single-prompt inference owns its strategies inside inference_runner.
    return None


def _install_z_image_strategies(args) -> None:
    return None


def prepare_inference_family(args, family: str, mode: str) -> None:
    spec = get_model_family_spec(family)
    if mode not in spec.supported_inference_modes:
        supported = (
            "single-prompt"
            if spec.supported_inference_modes == frozenset({"single"})
            else ", ".join(sorted(spec.supported_inference_modes))
        )
        raise SystemExit(
            f"{spec.display_name} inference does not support {mode!r} mode; "
            f"supported: {supported}."
        )

    validator = dispatch_model_family(
        family,
        operation="inference capability validation",
        handlers={
            "anima": _validate_anima,
            "krea2_raw": _validate_krea2,
            "z_image": _validate_z_image,
        },
    )
    validator(args, mode=mode)
    installer = dispatch_model_family(
        family,
        operation="inference text strategy",
        handlers={
            "anima": _install_anima_strategies,
            "krea2_raw": _install_krea2_strategies,
            "z_image": _install_z_image_strategies,
        },
    )
    installer(args)
