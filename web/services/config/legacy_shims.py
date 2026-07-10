"""Compatibility facade for legacy shim installers.

Domain installers live in sibling modules; this module re-exports the public
API expected by ``web.services.config._legacy``.
"""

from __future__ import annotations

from web.services.config.legacy_shims_common import (
    install_common_public_shims,
    restore_raw_files_shims,
)
from web.services.config.legacy_shims_dataset import install_dataset_public_shims
from web.services.config.legacy_shims_estimation import install_estimation_public_shims
from web.services.config.legacy_shims_file_groups import install_file_groups_public_shims
from web.services.config.legacy_shims_merge import install_merge_public_shims
from web.services.config.legacy_shims_output_runs import install_output_runs_public_shims
from web.services.config.legacy_shims_preflight import install_preflight_public_shims
from web.services.config.legacy_shims_raw_files import install_raw_files_public_shims
from web.services.config.legacy_shims_sample_prompts import (
    install_sample_prompts_public_shims,
)

__all__ = [
    "restore_raw_files_shims",
    "install_common_public_shims",
    "install_merge_public_shims",
    "install_preflight_public_shims",
    "install_dataset_public_shims",
    "install_sample_prompts_public_shims",
    "install_file_groups_public_shims",
    "install_raw_files_public_shims",
    "install_output_runs_public_shims",
    "install_estimation_public_shims",
]
