"""Shared preprocess cache pool (content-addressed resized + VAE/TE)."""

from library.cache_pool.fingerprint import (
    SCHEMA_VERSION,
    build_preprocess_signature,
    compute_fingerprint,
    scan_input_inventory,
)
from library.cache_pool.gc import cleanup_orphan_cache_pool, safe_rmtree_run_dir
from library.cache_pool.mount import mount_dir
from library.cache_pool.policy import CacheReusePolicy, parse_cache_reuse_policy
from library.cache_pool.refs import acquire_ref, list_orphans, release_ref
from library.cache_pool.store import (
    default_pool_root,
    pool_entry_dir,
    publish_pool_entry,
    read_manifest,
    write_manifest,
)

__all__ = [
    "SCHEMA_VERSION",
    "acquire_ref",
    "build_preprocess_signature",
    "cleanup_orphan_cache_pool",
    "compute_fingerprint",
    "default_pool_root",
    "list_orphans",
    "mount_dir",
    "parse_cache_reuse_policy",
    "CacheReusePolicy",
    "pool_entry_dir",
    "publish_pool_entry",
    "read_manifest",
    "release_ref",
    "safe_rmtree_run_dir",
    "scan_input_inventory",
    "write_manifest",
]
