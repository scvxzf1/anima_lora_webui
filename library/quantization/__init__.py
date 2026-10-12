"""Shared mixed NF4 API; ordinary loading does not import bitsandbytes."""
from importlib import import_module

from .constants import STATE_KEYS
from .format import MANIFEST_KEY, inspect_mixed_nf4
from .policy import ALL_LINEAR_POLICY, POLICY

__all__ = ['ALL_LINEAR_POLICY', 'MANIFEST_KEY', 'POLICY', 'STATE_KEYS', 'BF16AutocastLinear4bit',
           'inspect_mixed_nf4', 'load_mixed_nf4_into', 'save_mixed_nf4',
           'quant_state', 'restore_nf4_linear']


def __getattr__(name):
    if name in {'load_mixed_nf4_into', 'save_mixed_nf4'}:
        return getattr(import_module('.mixed_nf4', __name__), name)
    if name in {'BF16AutocastLinear4bit', 'quant_state', 'restore_nf4_linear'}:
        return getattr(import_module('.nf4', __name__), name)
    raise AttributeError(name)
