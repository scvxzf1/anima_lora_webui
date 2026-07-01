from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.tasks import _common


LEGACY_PROFILE_HELP = """
usage: nsys profile [<args>] [application] [<application args>]
    --capture-range=
    --capture-range-end=
    --cpuctxsw=
    --cuda-graph-trace=
    --cuda-memory-usage=
    --cudabacktrace=
    --gpu-metrics-device=
    --gpu-metrics-frequency=
    --gpu-metrics-set=
    --resolve-symbols=
    --sample=
    --stats=
    --trace=
    --wait=
"""


MODERN_PROFILE_HELP = LEGACY_PROFILE_HELP.replace(
    "--gpu-metrics-device=", "--gpu-metrics-devices="
) + """
    --python-sampling=
    --python-sampling-frequency=
"""


LEGACY_REPORT_HELP = """
  gpukernsum[:base|:mangled] -- CUDA GPU Kernel Summary
  gpumemtimesum -- GPU Memory Operations Summary (by Time)
  gpumemsizesum -- GPU Memory Operations Summary (by Size)
  cudaapisum -- CUDA API Summary
  kernexecsum[:base|:mangled] -- Summary of kernel launch and exec times
  nvtxkernsum[:base|:mangled] -- NVTX Range Kernel Summary
"""


def test_nsys_wrapper_downgrades_for_legacy_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("PROFILE_STEPS", "3-5")
    monkeypatch.setenv("NSYS_OUT", str(tmp_path / "legacy.nsys-rep"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(_common.shutil, "which", lambda name: "/usr/bin/nsys" if name == "nsys" else None)

    def fake_run(cmd, **kwargs):
        if cmd == ["/usr/bin/nsys", "profile", "--help"]:
            return SimpleNamespace(returncode=0, stdout=LEGACY_PROFILE_HELP, stderr="")
        if cmd == ["/usr/bin/nsys", "profile", "--gpu-metrics-device=help"]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Possible --gpu-metrics-device values are:\n"
                    "\t0: NVIDIA GeForce GTX 960 (not supported)\n"
                    "\t1: NVIDIA GeForce RTX 3080\n"
                    "\tnone: Disable GPU Metrics [Default]\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr(_common.subprocess, "run", fake_run)

    prefix, out_path = _common._nsys_wrapper()

    assert out_path == tmp_path / "legacy.nsys-rep"
    assert "--python-sampling=true" not in prefix
    assert "--gpu-metrics-device=1" in prefix
    assert "--gpu-metrics-devices=cuda-visible" not in prefix
    assert "--cuda-graph-trace=node" in prefix


def test_nsys_wrapper_uses_modern_python_sampling_and_cuda_visible(monkeypatch, tmp_path):
    monkeypatch.setenv("PROFILE_STEPS", "3-5")
    monkeypatch.setenv("NSYS_OUT", str(tmp_path / "modern.nsys-rep"))
    monkeypatch.setattr(_common.shutil, "which", lambda name: "/usr/bin/nsys" if name == "nsys" else None)

    def fake_run(cmd, **kwargs):
        if cmd == ["/usr/bin/nsys", "profile", "--help"]:
            return SimpleNamespace(returncode=0, stdout=MODERN_PROFILE_HELP, stderr="")
        if cmd == ["/usr/bin/nsys", "profile", "--gpu-metrics-devices=help"]:
            return SimpleNamespace(returncode=0, stdout="cuda-visible\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr(_common.subprocess, "run", fake_run)

    prefix, _ = _common._nsys_wrapper()

    assert "--python-sampling=true" in prefix
    assert "--python-sampling-frequency=1000" in prefix
    assert "--gpu-metrics-devices=cuda-visible" in prefix


def test_nsys_wrapper_allows_explicit_modern_gpu_metrics_device(monkeypatch, tmp_path):
    monkeypatch.setenv("PROFILE_STEPS", "3-5")
    monkeypatch.setenv("NSYS_OUT", str(tmp_path / "modern.nsys-rep"))
    monkeypatch.setenv("NSYS_GPU_METRICS_DEVICES", "1")
    monkeypatch.setenv("NSYS_GPU_METRICS_SET", "ga10x")
    monkeypatch.setenv("NSYS_GPU_METRICS_FREQUENCY", "1000")
    monkeypatch.setattr(_common.shutil, "which", lambda name: "/usr/bin/nsys" if name == "nsys" else None)

    def fake_run(cmd, **kwargs):
        if cmd == ["/usr/bin/nsys", "profile", "--help"]:
            return SimpleNamespace(returncode=0, stdout=MODERN_PROFILE_HELP, stderr="")
        if cmd == ["/usr/bin/nsys", "profile", "--gpu-metrics-devices=help"]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Possible --gpu-metrics-devices values are:\n"
                    "\t1: Ampere GA102 | NVIDIA GeForce RTX 3080\n"
                    "\tcuda-visible: Select GPUs that match CUDA_VISIBLE_DEVICES\n"
                    "Some GPUs are not supported:\n"
                    "\tMaxwell GM206 | NVIDIA GeForce GTX 960 - Unsupported architecture\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr(_common.subprocess, "run", fake_run)

    prefix, _ = _common._nsys_wrapper()

    assert "--gpu-metrics-devices=1" in prefix
    assert "--gpu-metrics-devices=cuda-visible" not in prefix
    assert "--gpu-metrics-set=ga10x" in prefix
    assert "--gpu-metrics-frequency=1000" in prefix


def test_nsys_wrapper_can_disable_python_sampling(monkeypatch, tmp_path):
    monkeypatch.setenv("PROFILE_STEPS", "3-5")
    monkeypatch.setenv("NSYS_OUT", str(tmp_path / "modern.nsys-rep"))
    monkeypatch.setenv("NSYS_PYTHON_SAMPLING", "0")
    monkeypatch.setattr(_common.shutil, "which", lambda name: "/usr/bin/nsys" if name == "nsys" else None)

    def fake_run(cmd, **kwargs):
        if cmd == ["/usr/bin/nsys", "profile", "--help"]:
            return SimpleNamespace(returncode=0, stdout=MODERN_PROFILE_HELP, stderr="")
        if cmd == ["/usr/bin/nsys", "profile", "--gpu-metrics-devices=help"]:
            return SimpleNamespace(returncode=0, stdout="cuda-visible\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr(_common.subprocess, "run", fake_run)

    prefix, _ = _common._nsys_wrapper()

    assert "--python-sampling=true" not in prefix
    assert "--python-sampling-frequency=1000" not in prefix
    assert "--gpu-metrics-devices=cuda-visible" in prefix


def test_nsys_stats_reports_legacy(monkeypatch):
    monkeypatch.setattr(_common, "_nsys_stats_help", lambda nsys: LEGACY_REPORT_HELP)

    assert _common._nsys_stats_reports("/usr/bin/nsys") == (
        "gpukernsum",
        "nvtxkernsum",
        "gpumemtimesum",
        "gpumemsizesum",
        "cudaapisum",
        "kernexecsum",
    )


def test_nsys_run_stats_reports_missing_importer(monkeypatch, tmp_path, capsys):
    rep = tmp_path / "profile.nsys-rep"
    qdstrm = tmp_path / "profile.qdstrm"
    qdstrm.write_bytes(b"trace")
    monkeypatch.setattr(_common.shutil, "which", lambda name: "/usr/bin/nsys" if name == "nsys" else None)
    monkeypatch.setattr(_common, "_nsys_qdstrm_importers", lambda: ())

    _common._nsys_run_stats(rep)

    err = capsys.readouterr().err
    assert "found" in err
    assert "qdstrm" in err
    assert "QdstrmImporter" in err


def test_nsys_run_stats_imports_qdstrm_before_stats(monkeypatch, tmp_path):
    rep = tmp_path / "profile.nsys-rep"
    qdstrm = tmp_path / "profile.qdstrm"
    qdstrm.write_bytes(b"trace")
    importer = tmp_path / "QdstrmImporter"
    importer.write_text("#!/bin/sh\n", encoding="utf-8")
    seen = {"stats": False}

    monkeypatch.setattr(_common.shutil, "which", lambda name: "/usr/bin/nsys" if name == "nsys" else None)
    monkeypatch.setattr(_common, "_nsys_qdstrm_importers", lambda: (str(importer),))
    monkeypatch.setattr(_common, "_nsys_stats_reports", lambda nsys: ("gpukernsum",))

    def fake_run(cmd, **kwargs):
        if cmd == [str(importer), "-i", str(qdstrm), "-o", str(rep), "-f"]:
            rep.write_bytes(b"report")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:2] == ["/usr/bin/nsys", "stats"]:
            seen["stats"] = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr(_common.subprocess, "run", fake_run)

    _common._nsys_run_stats(rep)

    assert seen["stats"] is True


def test_nsys_run_stats_accepts_importer_diagnostic_when_report_exists(monkeypatch, tmp_path, capsys):
    rep = tmp_path / "profile.nsys-rep"
    qdstrm = tmp_path / "profile.qdstrm"
    qdstrm.write_bytes(b"trace")
    importer = tmp_path / "QdstrmImporter"
    importer.write_text("#!/bin/sh\n", encoding="utf-8")
    seen = {"stats": False}

    monkeypatch.setattr(_common.shutil, "which", lambda name: "/usr/bin/nsys" if name == "nsys" else None)
    monkeypatch.setattr(_common, "_nsys_qdstrm_importers", lambda: (str(importer),))
    monkeypatch.setattr(_common, "_nsys_stats_reports", lambda nsys: ("gpukernsum",))

    def fake_run(cmd, **kwargs):
        if cmd == [str(importer), "-i", str(qdstrm), "-o", str(rep), "-f"]:
            rep.write_bytes(b"report")
            return SimpleNamespace(
                returncode=3,
                stdout="Processing 100%\n",
                stderr="Unknown runtime API function index: 461\n",
            )
        if cmd[:2] == ["/usr/bin/nsys", "stats"]:
            seen["stats"] = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr(_common.subprocess, "run", fake_run)

    _common._nsys_run_stats(rep)

    err = capsys.readouterr().err
    assert "exited with 3" in err
    assert "Unknown runtime API function index" in err
    assert seen["stats"] is True


def test_nsys_run_stats_retries_importer_candidates(monkeypatch, tmp_path):
    rep = tmp_path / "profile.nsys-rep"
    qdstrm = tmp_path / "profile.qdstrm"
    qdstrm.write_bytes(b"trace")
    first = tmp_path / "new" / "QdstrmImporter"
    second = tmp_path / "old" / "QdstrmImporter"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("#!/bin/sh\n", encoding="utf-8")
    second.write_text("#!/bin/sh\n", encoding="utf-8")
    seen = {"stats": False, "importers": []}

    monkeypatch.setattr(_common.shutil, "which", lambda name: "/usr/bin/nsys" if name == "nsys" else None)
    monkeypatch.setattr(_common, "_nsys_qdstrm_importers", lambda: (str(first), str(second)))
    monkeypatch.setattr(_common, "_nsys_stats_reports", lambda nsys: ("gpukernsum",))

    def fake_run(cmd, **kwargs):
        if cmd[0] in {str(first), str(second)}:
            seen["importers"].append(cmd[0])
            if cmd[0] == str(second):
                rep.write_bytes(b"report")
                return SimpleNamespace(returncode=3, stdout="", stderr="old importer diagnostic")
            return SimpleNamespace(returncode=1, stdout="", stderr="new importer failed")
        if cmd[:2] == ["/usr/bin/nsys", "stats"]:
            seen["stats"] = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd!r}")

    monkeypatch.setattr(_common.subprocess, "run", fake_run)

    _common._nsys_run_stats(rep)

    assert seen["importers"] == [str(first), str(second)]
    assert seen["stats"] is True


def test_accelerate_launch_does_not_insert_nsys_separator(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(_common, "build_launch_cmd", lambda *args: ["/py", "train.py", *args])
    monkeypatch.setattr(_common, "_nsys_wrapper", lambda: (["nsys", "profile"], tmp_path / "p.nsys-rep"))
    monkeypatch.setattr(_common, "_nsys_run_stats", lambda path: None)

    def fake_run(cmd):
        seen["cmd"] = cmd

    monkeypatch.setattr(_common, "run", fake_run)

    _common.accelerate_launch("--x")

    assert seen["cmd"] == ["nsys", "profile", "/py", "train.py", "--x"]
    assert "--" not in seen["cmd"]
