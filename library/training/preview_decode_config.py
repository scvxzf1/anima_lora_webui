"""Shared VAE preview policy and the legacy Qwen configuration alias."""

from collections.abc import Mapping

DEFAULT_DECODE_POLICY = "tiled"
DECODE_KEY = "sample_vae_decode"
LEGACY_DECODE_KEY = "qwen_sample_vae_decode"
DECODE_POLICIES = ("auto", "tiled", "cpu")


def decode_policy(config):
    get = config.get if isinstance(config, Mapping) else lambda key: getattr(config, key, None)
    value = get(DECODE_KEY)
    if value is None:
        value = get(LEGACY_DECODE_KEY)
    return DEFAULT_DECODE_POLICY if value is None else value


def configuration_errors(config):
    if decode_policy(config) not in DECODE_POLICIES:
        return [(DECODE_KEY, "Expected auto, tiled, or cpu")]
    return []


def add_arguments(parser):
    parser.add_argument("--sample_vae_decode", "--qwen_sample_vae_decode", dest=DECODE_KEY,
                        choices=DECODE_POLICIES, default=DEFAULT_DECODE_POLICY,
                        help="Training-preview VAE decoding; default tiled GPU. Auto retries CUDA OOM with tiling then CPU.")
