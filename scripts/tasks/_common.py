"""Shared helpers for ``scripts/tasks/*`` command modules.

Centralizes:
- ``ROOT`` (project root, regardless of where the calling module lives)
- ``PY`` resolution (venv-aware, pythonw.exe-safe)
- ``run`` / ``build_launch_cmd`` / ``accelerate_launch`` / ``train`` subprocess helpers
- ``latest_output`` / ``latest_lora`` / ``latest_hydra`` checkpoint pickers
- ``INFERENCE_BASE`` — shared inference.py argv prefix
- ``_path`` / ``_preset`` config-overlay helpers
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None
    try:
        import toml as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover
        _toml = None
else:
    _toml = None

ROOT = Path(__file__).resolve().parents[2]


def _python_exe() -> str:
    """Resolve the venv's ``python.exe`` even if this process runs as pythonw.exe.

    Why python.exe and not just ``sys.executable``: when the GUI is launched
    via the desktop shortcut, sys.executable is pythonw.exe. pythonw children
    don't surface a working ``sys.stdout``/``sys.stderr`` to inherited pipes
    the way python.exe does — tqdm progress (which writes to stderr) silently
    drops, breaking the GUI's progress bar. python.exe + ``CREATE_NO_WINDOW``
    (set in ``run()`` when this process has no console) gives us both no
    console popup AND working stdio for grandchildren.
    """
    if sys.platform == "win32":
        cand = Path(sys.executable).with_name("python.exe")
        if cand.exists():
            return str(cand)
    return sys.executable


PY = _python_exe()


def _preset(default: str = "default") -> str:
    return os.environ.get("PRESET", default)


_PATH_OVERRIDES_CACHE: dict | None = None
_PATH_OVERRIDES_CACHE_KEY: tuple[str, str, str, str] | None = None


def _load_toml_file(path: Path) -> dict:
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f)
    if _toml is not None:
        return _toml.loads(path.read_text(encoding="utf-8"))
    raise ModuleNotFoundError("No TOML parser available; install toml or use Python >= 3.11")


def _flat_config_scalars(data: dict) -> dict:
    return {
        key: value
        for key, value in data.items()
        if key not in {"general", "datasets", "variant"}
        and not isinstance(value, (dict, list))
    }


def _fallback_preset_section(preset: str) -> dict:
    presets_path = ROOT / "configs" / "presets.toml"
    if presets_path.exists():
        presets = _load_toml_file(presets_path)
        section = presets.get(preset)
        if isinstance(section, dict):
            return dict(section)
    custom_path = ROOT / "configs" / "custom" / f"{preset}.toml"
    if custom_path.exists():
        data = _load_toml_file(custom_path)
        if isinstance(data, dict):
            return data
    raise KeyError(preset)


def _fallback_path_overrides(cache_key: tuple[str, str, str, str]) -> dict:
    _runtime_config, preset, method, methods_subdir = cache_key
    overrides: dict = {}
    base_path = ROOT / "configs" / "base.toml"
    if base_path.exists():
        overrides.update(_flat_config_scalars(_load_toml_file(base_path)))
    try:
        overrides.update(_flat_config_scalars(_fallback_preset_section(preset)))
    except (FileNotFoundError, KeyError, ModuleNotFoundError):
        pass
    if method:
        method_path = ROOT / "configs" / methods_subdir / f"{method}.toml"
        if method_path.exists():
            overrides.update(_flat_config_scalars(_load_toml_file(method_path)))
    return overrides


def _path_overrides() -> dict:
    """Top-level path scalars from runtime config or base.toml → preset → method file.

    When ``ANIMA_RUNTIME_CONFIG`` is set, preprocess uses that runtime TOML
    first so Web training/preprocess share the same generated run directory.
    Missing env vars → just base + preset + method.

    Defers the import of ``library.config.io`` so commands that don't touch
    preprocess (e.g. ``test-merge``) keep the module-load surface small.
    """
    global _PATH_OVERRIDES_CACHE, _PATH_OVERRIDES_CACHE_KEY
    runtime_config = os.environ.get("ANIMA_RUNTIME_CONFIG") or ""
    cache_key = (
        runtime_config,
        _preset(),
        os.environ.get("METHOD") or "",
        os.environ.get("METHODS_SUBDIR") or "methods",
    )
    if _PATH_OVERRIDES_CACHE is not None:
        if not runtime_config or _PATH_OVERRIDES_CACHE_KEY is None or _PATH_OVERRIDES_CACHE_KEY == cache_key:
            return _PATH_OVERRIDES_CACHE
    if runtime_config:
        runtime_path = Path(runtime_config)
        if not runtime_path.is_absolute():
            runtime_path = (ROOT / runtime_path).resolve()
        try:
            if runtime_path.exists():
                data = _load_toml_file(runtime_path)
                overrides = {
                    k: v
                    for k, v in data.items()
                    if k not in {"general", "datasets"}
                    and not isinstance(v, (dict, list))
                }
                _PATH_OVERRIDES_CACHE = overrides
                _PATH_OVERRIDES_CACHE_KEY = cache_key
                return overrides
        except Exception as e:  # noqa: BLE001 — fall back silently to defaults
            print(f"warn: could not read runtime config overrides: {e}", file=sys.stderr)
    sys.path.insert(0, str(ROOT))
    try:
        from library.config.io import load_path_overrides

        overrides = load_path_overrides(
            preset=_preset(),
            method=os.environ.get("METHOD") or None,
            methods_subdir=os.environ.get("METHODS_SUBDIR") or "methods",
        )
    except Exception as e:  # noqa: BLE001 — fall back silently to defaults
        try:
            overrides = _fallback_path_overrides(cache_key)
        except Exception:  # noqa: BLE001
            print(f"warn: could not read base.toml path overrides: {e}", file=sys.stderr)
            overrides = {}
    _PATH_OVERRIDES_CACHE = overrides
    _PATH_OVERRIDES_CACHE_KEY = cache_key
    return overrides


def _path(key: str, default: str) -> str:
    """Fetch a path key from base.toml/preset overrides, with hardcoded fallback."""
    val = _path_overrides().get(key, default)
    return str(val) if val is not None else default


def bespoke_preset_flags(preset: str) -> list[str]:
    """Translate ``configs/presets.toml[<preset>]`` into CLI flags for the
    bespoke distillation loops (``scripts/distill_mod/distill.py`` / ``distill_turbo.py``)
    that bypass ``train.py``'s config merge chain.

    Honored keys:
      - ``blocks_to_swap`` → ``--blocks_to_swap N``
      - ``gradient_checkpointing`` (bool) → ``--grad_ckpt`` / ``--no_grad_ckpt``
      - ``sample_ratio`` → ``--sample_ratio R`` (per-bucket subsample; makes
        ``PRESET=debug/half/quarter/tenth`` actually run on a small slice).

    When the preset omits ``gradient_checkpointing`` we default to
    ``--no_grad_ckpt`` (the trainable footprints here are tiny; ckpt is a perf
    loss when VRAM isn't tight). Other preset keys are silently dropped.
    """
    sys.path.insert(0, str(ROOT))
    try:
        from library.config.io import load_preset_section
    except Exception as e:  # noqa: BLE001
        try:
            section = _fallback_preset_section(preset)
        except Exception:  # noqa: BLE001
            print(f"warn: could not import preset loader: {e}", file=sys.stderr)
            return ["--no_grad_ckpt"]
    else:
        try:
            section = load_preset_section(preset)
        except (FileNotFoundError, KeyError) as e:
            print(
                f"warn: preset '{preset}' not found ({e}); using bespoke-loop defaults",
                file=sys.stderr,
            )
            return ["--no_grad_ckpt"]

    flags: list[str] = []
    if "blocks_to_swap" in section:
        flags += ["--blocks_to_swap", str(int(section["blocks_to_swap"]))]
    if "gradient_checkpointing" in section:
        flags.append("--grad_ckpt" if section["gradient_checkpointing"] else "--no_grad_ckpt")
    else:
        flags.append("--no_grad_ckpt")
    if "sample_ratio" in section:
        flags += ["--sample_ratio", str(float(section["sample_ratio"]))]
    return flags


def latest_output(prefix: str = "", exclude: str | None = None) -> Path:
    """Return the most recently modified .safetensors file in output/ckpt/ matching prefix.

    If `exclude` is given, any filename containing that substring is skipped. Useful
    to disambiguate overlapping prefixes (e.g. anima_postfix vs anima_postfix_exp).
    HydraLoRA multi-head sibling files (`*_moe.safetensors`) and backup files
    (containing `.bak.`) are always excluded.
    """
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("*.safetensors")
            if f.name.startswith(prefix)
            and not f.name.endswith("_moe.safetensors")
            and ".bak." not in f.name
            and (exclude is None or exclude not in f.name)
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        label = f"'{prefix}*.safetensors'" if prefix else "*.safetensors"
        print(f"No {label} files found in output/ckpt/", file=sys.stderr)
        sys.exit(1)
    return outputs[0]


def latest_lora() -> Path:
    # Exclude pooled_text_proj heads: they live in output/ckpt/ too but are
    # not LoRAs — picking the newest `.safetensors` blindly grabs them right
    # after `make distill-mod`. They're resolved separately by MOD=1.
    return latest_output(exclude="pooled_text_proj")


def latest_hydra() -> Path:
    """Latest HydraLoRA multi-head file (`anima_hydra*_moe.safetensors`)."""
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("anima_hydra*_moe.safetensors")
            if ".bak." not in f.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        print(
            "No 'anima_hydra*_moe.safetensors' files found in output/ckpt/ "
            "(enable the HydraLoRA block in configs/methods/lora.toml and run `make lora`)",
            file=sys.stderr,
        )
        sys.exit(1)
    return outputs[0]


def _has_console() -> bool:
    """True if this process is attached to a Windows console (or is non-Windows).

    Used to decide whether to suppress new console popups for child processes.
    A pythonw.exe-launched process (e.g. desktop GUI shortcut) has no console.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:  # noqa: BLE001 — err on the safe side: keep output visible
        return True


def _prepend_env_path(env: dict[str, str], key: str, path: Path | str) -> None:
    """Prepend a path-like environment value without duplicating entries."""
    value = str(path)
    parts = [part for part in env.get(key, "").split(os.pathsep) if part]
    if value not in parts:
        env[key] = os.pathsep.join([value, *parts])


def run(cmd: list[str], **kwargs):
    """Run a subprocess, exit on failure.

    Prepends the venv's Scripts/bin directory to PATH (in both the child env
    and our own lookup) so venv-installed CLIs (``accelerate``, ``hf``, ...)
    resolve even when this process was started via a desktop shortcut that
    invokes ``pythonw.exe`` directly, bypassing venv activation.

    On Windows, ``subprocess.run`` uses the parent's PATH to locate the exe —
    setting ``env["PATH"]`` only affects the *child's* environment, not the
    lookup. We resolve the first arg to an absolute path with ``shutil.which``
    against the boosted PATH so the lookup works regardless.

    When this process has no console (pythonw.exe), Windows would allocate a
    new console for any console-subsystem child (python.exe, hf.exe, ...).
    We pass ``CREATE_NO_WINDOW`` to suppress that popup so GUI users don't
    see a terminal flash for every subprocess.
    """
    print(f"  > {' '.join(cmd)}")
    env = kwargs.pop("env", None)
    if env is None:
        env = os.environ.copy()
    venv_bin = str(Path(PY).parent)
    if venv_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    # Block-buffered stdio over pipes makes progress output (tqdm, training
    # logs) appear in chunks instead of streaming live. PYTHONUNBUFFERED keeps
    # children's Python stdio line-/un-buffered so the GUI sees output as it
    # happens. Inherited by grandchildren too.
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Preprocess helpers are executed as ``python scripts/preprocess/*.py``.
    # In that mode Python places ``scripts/preprocess`` on sys.path, not the
    # repository root, so imports like ``from library...`` need PYTHONPATH.
    _prepend_env_path(env, "PYTHONPATH", ROOT)
    cmd = list(cmd)
    if cmd and not Path(cmd[0]).is_absolute():
        resolved = shutil.which(cmd[0], path=env["PATH"])
        if resolved:
            cmd[0] = resolved
    if sys.platform == "win32" and not _has_console():
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        # Explicit stdio inheritance: when this process runs under pythonw.exe
        # (e.g. GUI shortcut), pythonw's fd 1/2 aren't exposed to children the
        # standard way — subprocess.run's default inheritance silently drops
        # the grandchild's output. Passing sys.stdout/sys.stderr directly hands
        # over Python's wrapped file objects, which DO route to the pipes our
        # parent (QProcess) set up. Only set when the caller hasn't.
        if sys.stdout is not None:
            kwargs.setdefault("stdout", sys.stdout)
        if sys.stderr is not None:
            kwargs.setdefault("stderr", sys.stderr)
    result = subprocess.run(cmd, cwd=kwargs.pop("cwd", ROOT), env=env, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _nsys_wrapper() -> tuple[list[str], Path] | tuple[None, None]:
    """Build an ``nsys profile`` prefix when PROFILE_STEPS is set.

    Returns ``(prefix, out_path)`` when active so the caller can both wrap the
    launch AND run ``nsys stats`` against the resulting report afterward.
    Returns ``(None, None)`` when PROFILE_STEPS is unset. Honors NSYS_OUT for
    the report path (default ``output/nsys/profile.nsys-rep``).

    Why ``--capture-range-end=stop`` (not ``stop-shutdown``) and ``--wait=primary``:
    the wrapped tree is ``nsys → accelerate launcher → train.py worker``. With
    ``stop-shutdown`` nsys SIGTERMs the launcher the moment ``cuProfilerStop``
    fires, the launcher dies before reaping the worker, the worker gets
    reparented to init, and the default ``--wait=all`` blocks forever waiting
    for it. Instead: the worker calls ``cuProfilerStop`` and then voluntarily
    ``sys.exit(0)`` (see ``library/training/loop.py`` ``_profiler_step_end``),
    the launcher exits naturally, and ``--wait=primary`` lets nsys finalize
    the report as soon as the launcher (its primary target) is gone — no
    leftover ``/tmp/*.qdstrm``.
    """
    if not os.environ.get("PROFILE_STEPS"):
        return None, None
    nsys = shutil.which("nsys")
    if nsys is None:
        print(
            "warn: PROFILE_STEPS set but `nsys` not found on PATH; "
            "running without profiler wrapper",
            file=sys.stderr,
        )
        return None, None
    out = os.environ.get("NSYS_OUT", "output/nsys/profile.nsys-rep")
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  > nsys report -> {out_path}")
    # Profile config tuned for kernel optimization + bottleneck analysis.
    #
    # Bottleneck-analysis additions (none of these need symbol downloads):
    #   --gpu-metrics-devices=cuda-visible  HW perf counters: SM occupancy,
    #       tensor-core util, DRAM/L2 bandwidth, warp stall reasons. The single
    #       most useful signal for "is this kernel compute- or memory-bound".
    #       nsys auto-picks the metric set (gb20x for Blackwell, ad10x for Ada,
    #       etc.); override with NSYS_GPU_METRICS_SET if needed.
    #   --gpu-metrics-frequency=10000       10 kHz sampling — fine enough to
    #       see per-step variation in a 3-step capture window.
    #   --cuda-graph-trace=node             per-node timing inside CUDA graphs
    #       (torch.compile emits these). Without it you only see the whole
    #       graph as one opaque blob.
    #   --cuda-memory-usage=true            tracks cudaMalloc/Free over time so
    #       you can correlate VRAM spikes with NVTX step ranges. Marked
    #       "significant runtime overhead" by nsys but fine inside a 3-step
    #       window — and essential for catching allocator thrash.
    #   --python-sampling=true @ 1 kHz      Python-side IP samples. Catches
    #       "Python is the bottleneck" cases (data loader, cache misses,
    #       config merging) that pure CUDA traces miss. Uses Python's own
    #       frame metadata, no debug-symbol download.
    #   --stats=true                        emit a sqlite next to the .nsys-rep
    #       so you can grep/SQL kernel timings without opening the GUI.
    #
    # Symbol-resolution is still OFF (--resolve-symbols=false + the three
    # *=none flags below). Without these, nsys finalize stalls for many
    # minutes on "Press Ctrl-C to stop symbol files downloading" reaching
    # out to NVIDIA's symbol servers — VRAM stays reserved, CPU sits at 0%.
    # The additions above are perf-counter and Python-frame data; none of
    # them need C++/CUDA-API symbol resolution.
    metrics_set = os.environ.get("NSYS_GPU_METRICS_SET")
    cmd = [
        nsys,
        "profile",
        "-o",
        str(out_path.with_suffix("")),  # nsys appends .nsys-rep
        "--force-overwrite=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--wait=primary",
        "--trace=cuda,nvtx,cudnn,cublas",
        "--cuda-graph-trace=node",
        "--cuda-memory-usage=true",
        "--python-sampling=true",
        "--python-sampling-frequency=1000",
        "--stats=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--cudabacktrace=none",
        "--resolve-symbols=false",
    ]
    if _nsys_gpu_metrics_available(nsys):
        cmd += [
            "--gpu-metrics-devices=cuda-visible",
            "--gpu-metrics-frequency=10000",
        ]
        if metrics_set:
            cmd.append(f"--gpu-metrics-set={metrics_set}")
    else:
        print(
            "  > nsys: GPU metrics disabled (perf counters restricted to admin). "
            "To enable SM occupancy / tensor-core / memory-bandwidth counters:\n"
            "      sudo tee /etc/modprobe.d/nvidia-perf.conf <<<'options nvidia "
            '"NVreg_RestrictProfilingToAdminUsers=0"\'\n'
            "      sudo update-initramfs -u && sudo reboot\n"
            "    See https://developer.nvidia.com/ERR_NVGPUCTRPERM",
            file=sys.stderr,
        )
    return cmd, out_path


def _nsys_gpu_metrics_available(nsys: str) -> bool:
    """Probe whether nsys can collect GPU metrics on this host.

    nsys validates ``--gpu-metrics-devices`` at argv-parse time and aborts the
    whole run if the perf-counter ioctl is restricted to root (the default on
    most distros — see ERR_NVGPUCTRPERM). Probing first lets us silently skip
    the flag instead of crashing the training task.
    """
    try:
        out = subprocess.run(
            [nsys, "profile", "--gpu-metrics-devices=help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    blob = (out.stdout or "") + (out.stderr or "")
    return "Insufficient privilege" not in blob and "None of the installed GPUs" not in blob


# nsys stats reports auto-generated after profiling. Tuned for kernel
# optimization + bottleneck analysis on a per-step NVTX trace:
#   cuda_gpu_kern_sum     — top kernels by total GPU time (the "what to
#                           optimize" list)
#   nvtx_kern_sum         — kernels grouped under our `step=N` NVTX ranges
#                           (which step is slow + which kernels caused it)
#   cuda_gpu_mem_time_sum — host↔device mem ops by total time (catches
#                           transfer-bound steps, e.g. uncached latents)
#   cuda_gpu_mem_size_sum — same ops by bytes moved (cross-check time vs size
#                           to spot small-but-frequent thrash)
#   cuda_api_sum          — host-side CUDA API calls (cudaLaunchKernel,
#                           cudaStreamSynchronize blocking, etc.)
#   cuda_kern_exec_sum    — per-kernel queue/exec timings (launch overhead
#                           vs. on-GPU runtime — small kernels dominated by
#                           launch latency show up here)
_NSYS_STATS_REPORTS = (
    "cuda_gpu_kern_sum",
    "nvtx_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_api_sum",
    "cuda_kern_exec_sum",
)


def _nsys_run_stats(rep_path: Path) -> None:
    """Generate textual ``nsys stats`` reports next to the .nsys-rep.

    Writes one ``<stem>_<report>.txt`` per report into the same directory as
    the .nsys-rep. Best-effort: if the .nsys-rep didn't materialize (e.g. nsys
    aborted before finalizing) or stats fails, prints a warning and returns —
    a missing summary shouldn't fail the training task itself.
    """
    if not rep_path.exists():
        print(
            f"warn: nsys report not found at {rep_path}; skipping stats",
            file=sys.stderr,
        )
        return
    nsys = shutil.which("nsys")
    if nsys is None:
        return
    out_prefix = rep_path.with_suffix("")  # strip .nsys-rep
    cmd = [
        nsys,
        "stats",
        "--force-export=true",
        "--force-overwrite=true",
        "--format=column",
        "--output",
        str(out_prefix),
    ]
    for report in _NSYS_STATS_REPORTS:
        cmd += ["--report", report]
    cmd.append(str(rep_path))
    print(f"  > nsys stats -> {out_prefix.parent}/")
    # Don't sys.exit on failure — best-effort summary, the .nsys-rep is the
    # canonical artifact and the GUI can always open it directly.
    try:
        subprocess.run(cmd, cwd=ROOT, check=False)
    except OSError as e:
        print(f"warn: nsys stats failed: {e}", file=sys.stderr)


def build_launch_cmd(*args: str, python_exe: str | None = None) -> list[str]:
    """Build the training launch command list (no side effects).

    Default single-GPU path runs ``train.py`` directly. Set
    ``ANIMA_ACCELERATE_LAUNCH=1`` for multi-GPU / distributed launches where
    accelerate must provide rank and world-size environment variables. The
    daemon reuses this pure builder so CLI, queue, and WebUI launches stay on
    one command path.
    """
    from library.runtime.launch import accelerate_training_command_prefix

    py = python_exe or PY
    return [*accelerate_training_command_prefix(py, "train.py", os.environ), *args]


def accelerate_launch(*args: str):
    """Launch training with extra CLI args forwarded.

    Builds the command via ``build_launch_cmd`` and runs it through ``run``.
    When PROFILE_STEPS is set, wraps the launch with ``nsys profile`` so
    ``make <method> PROFILE_STEPS=3-5`` produces a navigable Nsight report
    at ``output/nsys/profile.nsys-rep`` (override with NSYS_OUT). After the
    run, generates per-report textual summaries via ``nsys stats`` next to
    the .nsys-rep.
    """
    cmd = build_launch_cmd(*args)
    nsys_prefix, nsys_out = _nsys_wrapper()
    if nsys_prefix is not None:
        cmd = nsys_prefix + ["--"] + cmd
    run(cmd)
    if nsys_out is not None:
        _nsys_run_stats(nsys_out)


def build_method_args(
    method: str,
    *,
    preset: str,
    methods_subdir: str | None = None,
    extra=None,
    artist: str | None = None,
    profile_steps: str | None = None,
) -> list[str]:
    """Assemble the ``train.py`` method/preset argument list.

    Kept pure so both CLI and daemon can use one source of truth.
    """
    extra = list(extra or [])
    args = ["--method", method, "--preset", preset]
    if methods_subdir:
        args += ["--methods_subdir", methods_subdir]
    if artist and not any(a == "--artist_filter" for a in extra):
        args += ["--artist_filter", artist]
    if profile_steps and not any(a == "--profile_steps" for a in extra):
        args += ["--profile_steps", profile_steps]
    return [*args, *extra]


def _queue_submit(
    method: str,
    *,
    preset: str,
    methods_subdir: str | None,
    extra: list[str],
    artist: str | None,
    profile_steps: str | None,
) -> None:
    """Enqueue a training job on the local daemon instead of running inline."""
    extra = list(extra)
    if artist and "--artist_filter" not in extra:
        extra += ["--artist_filter", artist]
    if profile_steps and "--profile_steps" not in extra:
        extra += ["--profile_steps", profile_steps]

    from scripts.daemon import client as _daemon_client

    cl = _daemon_client.ensure_daemon()
    resp = cl.submit(
        method=method,
        preset=preset,
        methods_subdir=methods_subdir,
        extra=extra,
    )
    job_id = resp.get("job_id")
    print(
        f"queued job {job_id} (method={method}, preset={preset}). "
        f"daemon: {cl.base}\n"
        f"  make daemon-attach JOB={job_id}   # follow this job's output\n"
        f"  make daemon-attach                # follow queue/lifecycle events\n"
        f"  make daemon-kill JOB={job_id}     # cancel it\n"
        f"  make daemon-terminate             # stop the daemon + discard queue"
    )


def train(
    method: str, extra, preset: str | None = None, methods_subdir: str | None = None
):
    """Launch training for a given method + preset (PRESET env overrides default).

    `methods_subdir` selects the folder under `configs/` that holds the method
    file (default ``"methods"``; pass ``"gui-methods"`` for the clean per-variant
    files used by the `lora-gui` path).

    ARTIST env var trains an artist-only LoRA — equivalent to passing
    `--artist_filter <name>` (filters dataset to `@<name>`-tagged captions and
    redirects output to `output/ckpt-artist/`).

    ``--queue`` anywhere in ``extra`` enqueues the job on the local training
    daemon and returns immediately instead of running inline.
    """
    preset = preset or _preset()
    extra = list(extra or [])
    artist = os.environ.get("ARTIST")
    profile_steps = os.environ.get("PROFILE_STEPS")

    if "--queue" in extra:
        extra.remove("--queue")
        _queue_submit(
            method,
            preset=preset,
            methods_subdir=methods_subdir,
            extra=extra,
            artist=artist,
            profile_steps=profile_steps,
        )
        return

    args = build_method_args(
        method,
        preset=preset,
        methods_subdir=methods_subdir,
        extra=extra,
        artist=artist,
        profile_steps=profile_steps,
    )
    accelerate_launch(*args)


def build_inference_base() -> list[str]:
    return [
        PY,
        "inference.py",
        "--dit",
        _path(
            "pretrained_model_name_or_path",
            "models/diffusion_models/anima-base-v1.0.safetensors",
        ),
        "--text_encoder",
        _path("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
        "--vae",
        _path("vae", "models/vae/qwen_image_vae.safetensors"),
        "--vae_chunk_size",
        "64",
        "--vae_disable_cache",
        "--attn_mode",
        "flash",  # flash4 not supported yet (flash-attention-sm120 disabled)
        "--lora_multiplier",
        "1.0",
        "--prompt",
        "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top"
        " and denim shorts is standing outdoors. She's holding a rectangular sign out in"
        ' front of her that reads "ANIMA". She\'s looking at the viewer with a smile. The'
        " background features some trees and blue sky with clouds.",
        "--negative_prompt",
        "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
        "--image_size",
        "1024",
        "1024",
        "--infer_steps",
        "28",
        "--flow_shift",
        "1.0",
        "--sampler",
        "er_sde",
        "--guidance_scale",
        "4.0",
        "--seed",
        "40",
        "--save_path",
        "output/tests",
    ]


INFERENCE_BASE = build_inference_base()
