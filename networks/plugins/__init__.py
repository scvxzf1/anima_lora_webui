"""Bundled network plugins.

Each child package registers itself through ``networks.registry`` on import.
Core LoRA code discovers these packages at startup and otherwise knows only
the registry interface.
"""
