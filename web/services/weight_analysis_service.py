"""Static safetensors ΔW analysis for LoRA / LoHa / LoKr weights.

The service deliberately stays CPU-only and model-free: it reads tensors from a
single ``.safetensors`` file, reconstructs equivalent adapter deltas where the
layout is supported, and summarizes static weight energy.  It does not load the
DiT, run prompt inference, or write back to user checkpoints.

Implementation lives under ``web.services.weight_analysis``; this module remains
the stable public import surface for routes, services, and tests.
"""

from __future__ import annotations

from web.services.weight_analysis import constants as _wa_constants
from web.services.weight_analysis import detect as _wa_detect
from web.services.weight_analysis import inspect as _wa_inspect
from web.services.weight_analysis import io as _wa_io
from web.services.weight_analysis import layers as _wa_layers
from web.services.weight_analysis import paths as _wa_paths
from web.services.weight_analysis import summary as _wa_summary

ROOT = _wa_constants.ROOT
WEIGHT_EXTS = _wa_constants.WEIGHT_EXTS
MAX_ANALYSIS_WEIGHT_LIMIT = _wa_constants.MAX_ANALYSIS_WEIGHT_LIMIT
MAX_METADATA_ITEMS = _wa_constants.MAX_METADATA_ITEMS
MAX_UPLOAD_WEIGHT_BYTES = _wa_constants.MAX_UPLOAD_WEIGHT_BYTES
TOP_LAYER_LIMIT = _wa_constants.TOP_LAYER_LIMIT
STYLE_PRIORITY = _wa_constants.STYLE_PRIORITY
CHARACTER_PRIORITY = _wa_constants.CHARACTER_PRIORITY
UNSUPPORTED_SPEC_TOKENS = _wa_constants.UNSUPPORTED_SPEC_TOKENS
UNSUPPORTED_KEY_FRAGMENTS = _wa_constants.UNSUPPORTED_KEY_FRAGMENTS
BLOCK_RE = _wa_constants.BLOCK_RE
LORA_SUFFIX = _wa_constants.LORA_SUFFIX
LOHA_SUFFIX = _wa_constants.LOHA_SUFFIX
LOKR_SUFFIX = _wa_constants.LOKR_SUFFIX

WeightListingContext = _wa_paths.WeightListingContext
list_analysis_weights = _wa_paths.list_analysis_weights
resolve_analysis_weight = _wa_paths.resolve_analysis_weight
_normalize_user_path_value = _wa_paths._normalize_user_path_value
_analysis_source_tasks = _wa_paths._analysis_source_tasks
_source_task_meta = _wa_paths._source_task_meta
_analysis_weight_meta = _wa_paths._analysis_weight_meta
_allowed_weight_dirs = _wa_paths._allowed_weight_dirs
_is_under_allowed_weight_dir = _wa_paths._is_under_allowed_weight_dir
_resolve_display_path = _wa_paths._resolve_display_path
_display_path = _wa_paths._display_path

_read_safetensors_header = _wa_io._read_safetensors_header
_load_safetensors_tensors = _wa_io._load_safetensors_tensors
_read_safetensors_header_bytes = _wa_io._read_safetensors_header_bytes
_load_safetensors_tensors_bytes = _wa_io._load_safetensors_tensors_bytes

_detect_adapter_type = _wa_detect._detect_adapter_type
_unsupported_reason = _wa_detect._unsupported_reason
_label_from_meta_or_keys = _wa_detect._label_from_meta_or_keys
_truthy = _wa_detect._truthy
_base_payload = _wa_detect._base_payload
_uploaded_base_payload = _wa_detect._uploaded_base_payload
_safe_metadata = _wa_detect._safe_metadata
_int_or_none = _wa_detect._int_or_none

_compute_layers = _wa_layers._compute_layers
_compute_lora_layers = _wa_layers._compute_lora_layers
_compute_loha_layers = _wa_layers._compute_loha_layers
_compute_lokr_layers = _wa_layers._compute_lokr_layers
_lora_delta = _wa_layers._lora_delta
_alpha_value = _wa_layers._alpha_value
_rank_from_down = _wa_layers._rank_from_down
_rank_from_loha = _wa_layers._rank_from_loha
_metadata_network_dim = _wa_layers._metadata_network_dim
_layer_stats = _wa_layers._layer_stats
_layer_error = _wa_layers._layer_error
parse_layer_name = _wa_layers.parse_layer_name
_normalize_component = _wa_layers._normalize_component
_finalize_layer_contributions = _wa_layers._finalize_layer_contributions
_candidate_score = _wa_layers._candidate_score
_layer_notes = _wa_layers._layer_notes

_summary = _wa_summary._summary
_average_fro_for_blocks = _wa_summary._average_fro_for_blocks
_component_summary = _wa_summary._component_summary
_block_summary = _wa_summary._block_summary
_aggregate_group = _wa_summary._aggregate_group
_top_candidates = _wa_summary._top_candidates
_candidate_reason = _wa_summary._candidate_reason
_heatmap = _wa_summary._heatmap
_ordered_components = _wa_summary._ordered_components
_empty_summary = _wa_summary._empty_summary
_empty_heatmap = _wa_summary._empty_heatmap

inspect_weight = _wa_inspect.inspect_weight
inspect_weight_bytes = _wa_inspect.inspect_weight_bytes
_inspect_loaded_weight = _wa_inspect._inspect_loaded_weight
