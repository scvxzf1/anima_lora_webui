from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LORA_MODULES_DIR = REPO_ROOT / "networks" / "lora_modules"


def _iter_absolute_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append(node.module)
    return out


def test_lora_modules_do_not_import_networks_package_root():
    """lora_modules 可以依赖 networks.attn_fuse，但不能 import 包根 networks。"""
    offenders: list[str] = []
    for path in sorted(LORA_MODULES_DIR.rglob("*.py")):
        for mod in _iter_absolute_imports(path):
            if mod == "networks" or mod.startswith("networks.registry"):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {mod}")
            if mod.startswith("networks.lora_modules"):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {mod}")
    assert offenders == []


def test_registry_module_body_does_not_import_lora_modules_eagerly():
    """registry 顶层模块体不应直接 from .lora_modules import ...。"""
    path = REPO_ROOT / "networks" / "registry.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            top_level_imports.append(mod)
        elif isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
    assert not any(
        item in {".lora_modules", "networks.lora_modules", "lora_modules"}
        for item in top_level_imports
    )


def test_import_networks_then_lora_modules_is_stable():
    """包根和实现层应可按任意顺序导入，且 registry 可用。"""
    for name in list(sys.modules):
        if name == "networks" or name.startswith("networks."):
            del sys.modules[name]

    networks = importlib.import_module("networks")
    lora_modules = importlib.import_module("networks.lora_modules")
    registry = importlib.import_module("networks.registry")

    assert "lora" in networks.NETWORK_REGISTRY
    assert hasattr(lora_modules, "LoRAModule")
    assert registry.NETWORK_REGISTRY is networks.NETWORK_REGISTRY


def test_public_facade_still_exports_registry_api():
    import networks
    for name in (
        "NETWORK_REGISTRY",
        "NetworkSpec",
        "resolve_network_spec",
        "register_network_spec",
        "ensure_builtin_plugins_loaded",
        "ModuleCreationContext",
    ):
        assert hasattr(networks, name)


def test_resolve_network_spec_still_selects_lora_by_default():
    from networks import resolve_network_spec
    spec = resolve_network_spec({})
    assert spec.name == "lora"


def test_registry_leaf_path_has_no_absolute_networks_imports():
    """Critical path modules must not absolute-import networks.* package paths."""
    offenders: list[str] = []
    for rel in (
        "networks/registry.py",
        "networks/core_specs.py",
        "networks/registry_api.py",
        "networks/lora_modules/lora.py",
        "networks/lora_modules/hydra.py",
        "networks/lora_modules/chimera.py",
        "networks/lora_modules/stacked_experts.py",
        "networks/plugins/step_expert/__init__.py",
        "networks/plugins/lokr/__init__.py",
    ):
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        for mod in _iter_absolute_imports(path):
            if mod == "networks" or mod.startswith("networks."):
                offenders.append(f"{rel}: {mod}")
    assert offenders == []


def test_import_lora_modules_before_networks_is_stable():
    """Reverse import order should also leave registry usable."""
    for name in list(sys.modules):
        if name == "networks" or name.startswith("networks."):
            del sys.modules[name]

    lora_modules = importlib.import_module("networks.lora_modules")
    networks = importlib.import_module("networks")
    registry = importlib.import_module("networks.registry")

    assert hasattr(lora_modules, "LoRAModule")
    assert "lora" in networks.NETWORK_REGISTRY
    assert registry.NETWORK_REGISTRY is networks.NETWORK_REGISTRY
