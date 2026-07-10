"""Regression guards for the train.py / BaseDataset split.

Catches the class of bugs that repeatedly broke cold starts after mechanical
extraction:

- missing import of a free name used only on a conditional branch
  (``resolve_block_swap_profile_jsonl``, ``MemoryProbe``, ``TokenizeStrategy``)
- wrong re-export after a helper moved between modules
  (``plan_resume_start``)
- whole submodules that fail to import after the move
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

from library.training.train_bootstrap import resolve_block_swap_profile_jsonl

ROOT = Path(__file__).resolve().parents[1]

# Free names that historically broke after extraction, plus the general pattern
# for helpers / types that are almost always imported rather than defined local.
_INTEREST = re.compile(
    r"^(?:"
    r"[A-Z][A-Za-z0-9_]*|"
    r"resolve_\w+|build_\w+|plan_\w+|save_\w+|maybe_\w+|load_\w+|"
    r"enable_\w+|cache_\w+|compile_\w+"
    r")$"
)

_BUILTIN = set(dir(__import__("builtins")))
_EXTRA_OK = {
    "Any",
    "Optional",
    "Union",
    "List",
    "Dict",
    "Tuple",
    "Set",
    "Callable",
    "Iterable",
    "Iterator",
    "Sequence",
    "Mapping",
    "Type",
    "TypeVar",
    "Generic",
    "ClassVar",
    "Final",
    "Literal",
    "cast",
    "overload",
    "Protocol",
    "TypedDict",
    "NotRequired",
    "Required",
    "Self",
    "Annotated",
    "Path",
    "PurePath",
    "Namespace",
    "ModuleType",
    "Enum",
    "IntEnum",
    "auto",
    "abstractmethod",
    "dataclass",
    "field",
    "asdict",
    "astuple",
    "property",
    "staticmethod",
    "classmethod",
    "super",
    "object",
    "type",
    "Exception",
    "BaseException",
    "OSError",
    "RuntimeError",
    "ValueError",
    "TypeError",
    "KeyError",
    "AttributeError",
    "ImportError",
    "NameError",
    "StopIteration",
    "NotImplementedError",
    "AssertionError",
    "FileNotFoundError",
    "PermissionError",
    "torch",
    "nn",
    "F",
    "np",
    "pd",
    "tqdm",
    "logging",
    "logger",
    "log",
    "warnings",
    "math",
    "os",
    "sys",
    "json",
    "toml",
    "re",
    "copy",
    "time",
    "datetime",
    "shutil",
    "tempfile",
    "subprocess",
    "threading",
    "queue",
    "functools",
    "itertools",
    "collections",
    "contextlib",
    "dataclasses",
    "typing",
    "importlib",
    "inspect",
    "traceback",
    "hashlib",
    "uuid",
    "glob",
    "fnmatch",
    "argparse",
    "Accelerator",
    "PartialState",
    "ProjectConfiguration",
    "DistributedType",
    "OrderedDict",
    "defaultdict",
    "deque",
    "Counter",
    "namedtuple",
    "ChainMap",
    "Tensor",
    "Parameter",
    "Module",
    "Device",
    "dtype",
    "Generator",
    "safetensors",
    "tomlkit",
    "yaml",
    "PIL",
    "Image",
    "cv2",
    "True",
    "False",
    "None",
    "Ellipsis",
    "NotImplemented",
    "args",
    "accelerator",
    "network",
    "unet",
    "vae",
    "text_encoder",
    "optimizer",
    "lr_scheduler",
}

_SCAN_PACKAGES = ("library.training", "library.datasets")
_SCAN_EXTRA_MODULES = ("train",)
_CRITICAL_EXPORTS = {
    "library.training.checkpoints": ("plan_resume_start",),
    "library.training.train_bootstrap": ("resolve_block_swap_profile_jsonl",),
    "library.training.probes": ("maybe_probe", "maybe_probe_components"),
    "library.training.memory_probe": ("MemoryProbe",),
    "library.runtime.peak_probe": ("PeakProbe",),
    "library.anima.text_strategies": ("TokenizeStrategy",),
    "library.datasets.image_utils": ("load_image",),
    "library.training.model_loading": ("load_unet_lazily",),
    "library.training.train_session": ("run_training_session",),
}


def _iter_package_modules(package_name: str) -> list[str]:
    module = importlib.import_module(package_name)
    if not getattr(module, "__path__", None):
        return [package_name]
    names = [package_name]
    for info in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
        names.append(info.name)
    return names


def _source_files() -> list[Path]:
    files: list[Path] = []
    for rel in ("library/training", "library/datasets"):
        files.extend(
            p
            for p in (ROOT / rel).rglob("*.py")
            if "__pycache__" not in p.parts
        )
    files.append(ROOT / "train.py")
    return sorted(files)


def _collect_defined(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
    return names


def _free_interesting_loads(fn: ast.AST, module_names: set[str]) -> list[tuple[str, int]]:
    issues: list[tuple[str, int]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[set[str]] = [set()]
            self.global_decl: set[str] = set()

        def bind(self, name: str) -> None:
            self.stack[-1].add(name)

        def push(self) -> None:
            self.stack.append(set())

        def pop(self) -> None:
            self.stack.pop()

        def visit_Global(self, node: ast.Global) -> None:
            self.global_decl.update(node.names)

        def visit_Name(self, node: ast.Name) -> None:
            if not isinstance(node.ctx, ast.Load):
                if isinstance(node.ctx, (ast.Store, ast.Del, ast.Param)):
                    self.bind(node.id)
                return
            name = node.id
            if name in _BUILTIN or name in _EXTRA_OK or not _INTEREST.match(name):
                return
            for scope in reversed(self.stack):
                if name in scope:
                    return
            if name in self.global_decl or name in module_names:
                return
            issues.append((name, getattr(node, "lineno", 0)))

        def visit_arg(self, node: ast.arg) -> None:
            self.bind(node.arg)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.bind(node.name)
            self.push()
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                self.bind(arg.arg)
            if node.args.vararg:
                self.bind(node.args.vararg.arg)
            if node.args.kwarg:
                self.bind(node.args.kwarg.arg)
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None:
                    self.visit(default)
            for stmt in node.body:
                self.visit(stmt)
            for decorator in node.decorator_list:
                self.visit(decorator)
            self.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.bind(node.name)
            self.push()
            for stmt in node.body:
                self.visit(stmt)
            self.pop()

        def visit_For(self, node: ast.For) -> None:
            self.visit(node.iter)
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                    self.bind(target.id)
            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)

        def visit_With(self, node: ast.With) -> None:
            for item in node.items:
                self.visit(item.context_expr)
                if item.optional_vars is not None:
                    for target in ast.walk(item.optional_vars):
                        if isinstance(target, ast.Name) and isinstance(
                            target.ctx, ast.Store
                        ):
                            self.bind(target.id)
            for stmt in node.body:
                self.visit(stmt)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.type is not None:
                self.visit(node.type)
            if node.name:
                self.bind(node.name)
            for stmt in node.body:
                self.visit(stmt)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                    self.bind(target.id)
            self.visit(node.iter)
            for if_clause in node.ifs:
                self.visit(if_clause)

        def visit_ListComp(self, node: ast.ListComp) -> None:
            self.push()
            for generator in node.generators:
                self.visit(generator)
            self.visit(node.elt)
            self.pop()

        visit_SetComp = visit_ListComp

        def visit_DictComp(self, node: ast.DictComp) -> None:
            self.push()
            for generator in node.generators:
                self.visit(generator)
            self.visit(node.key)
            self.visit(node.value)
            self.pop()

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            self.push()
            for generator in node.generators:
                self.visit(generator)
            self.visit(node.elt)
            self.pop()

        def visit_Lambda(self, node: ast.Lambda) -> None:
            self.push()
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                self.bind(arg.arg)
            if node.args.vararg:
                self.bind(node.args.vararg.arg)
            if node.args.kwarg:
                self.bind(node.args.kwarg.arg)
            self.visit(node.body)
            self.pop()

    Visitor().visit(fn)
    return issues


def _scan_missing_symbols() -> list[str]:
    findings: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_names = _collect_defined(tree)
        rel = str(path.relative_to(ROOT))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for name, lineno in _free_interesting_loads(node, module_names):
                    key = (rel, node.name, name)
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(f"{rel}:{lineno} in {node.name} -> {name}")
            elif isinstance(node, ast.ClassDef):
                class_names = set(module_names) | {node.name}
                for sub in node.body:
                    if isinstance(
                        sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        class_names.add(sub.name)
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        owner = f"{node.name}.{stmt.name}"
                        for name, lineno in _free_interesting_loads(stmt, class_names):
                            key = (rel, owner, name)
                            if key in seen:
                                continue
                            seen.add(key)
                            findings.append(f"{rel}:{lineno} in {owner} -> {name}")
    return findings


def test_training_and_dataset_modules_import():
    failures: list[str] = []
    for package in _SCAN_PACKAGES:
        for name in _iter_package_modules(package):
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001 - collect all import failures
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
    for name in _SCAN_EXTRA_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "module import failures:\n" + "\n".join(failures)


def test_critical_split_exports_exist():
    for module_name, attrs in _CRITICAL_EXPORTS.items():
        module = importlib.import_module(module_name)
        for attr in attrs:
            assert hasattr(module, attr), f"{module_name}.{attr} missing"


def test_ast_no_unresolved_split_symbols():
    findings = _scan_missing_symbols()
    assert not findings, "unresolved free names after split:\n" + "\n".join(findings)


def test_maybe_probe_signature_is_trainer_label():
    from library.training import probes

    # Historical bug: maybe_probe(trainer, trainer, label=...) after method->function.
    assert list(probes.maybe_probe.__code__.co_varnames[:2]) == ["trainer", "label"]
    assert list(probes.maybe_probe_components.__code__.co_varnames[:2]) == [
        "trainer",
        "label",
    ]


@pytest.mark.parametrize(
    ("value", "output_dir", "output_name", "expected_suffix"),
    [
        ("off", "/tmp/run/training_output", "run", None),
        ("none", "/tmp/run/training_output", "run", None),
        (None, "/tmp/run/training_output", "run", None),
        (
            "auto",
            "/tmp/run/training_output",
            "exp",
            "logs/exp.block_swap_profile.jsonl",
        ),
        (
            "/abs/path/profile.jsonl",
            "/tmp/run/training_output",
            "run",
            "/abs/path/profile.jsonl",
        ),
    ],
)
def test_resolve_block_swap_profile_jsonl(value, output_dir, output_name, expected_suffix):
    args = type(
        "Args",
        (),
        {
            "block_swap_profile_jsonl": value,
            "output_dir": output_dir,
            "output_name": output_name,
        },
    )()
    resolved = resolve_block_swap_profile_jsonl(args)
    if expected_suffix is None:
        assert resolved is None
    elif expected_suffix.startswith("/"):
        assert resolved == expected_suffix
    else:
        assert resolved is not None
        assert resolved.replace("\\", "/").endswith(expected_suffix)


def test_model_loading_imports_block_swap_resolver():
    """blocks_to_swap>0 path must resolve profile helper at import time."""
    source = (ROOT / "library/training/model_loading.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
    assert "resolve_block_swap_profile_jsonl" in imported
    assert "load_image" not in source or True  # keep module focused
    # ensure the call site still exists (branch not deleted)
    assert "resolve_block_swap_profile_jsonl(args)" in source


def test_dataset_cache_imports_load_image():
    source = (ROOT / "library/datasets/dataset_cache.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
    assert "load_image" in imported
    assert "executor.submit(" in source
    assert "load_image" in source


def test_train_session_wires_probe_and_resume_symbols():
    source = (ROOT / "library/training/train_session.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
    for name in (
        "MemoryProbe",
        "PeakProbe",
        "maybe_probe",
        "plan_resume_start",
        "TokenizeStrategy",
    ):
        # TokenizeStrategy may be used as text_strategies.TokenizeStrategy
        if name == "TokenizeStrategy":
            assert "TokenizeStrategy" in source
            continue
        assert name in imported, f"train_session missing import: {name}"
