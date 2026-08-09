"""R-verify: 3080 1024×1024 NF4×block swap 显存边界测绘 (消融步进).

固定 1024×1024 + TE-CPU + 磁盘 NF4, 支持两种扫法:
  - sweep-up (K2_ABL_MODE=sweep, 默认): 从低 swap 往高扫完整曲线. OOM 时按
    K2_ABL_OOM_STEP (默认 4) 快速逼近临界, 过点后按 K2_ABL_FINE_STEP (默认 2)
    细扫到上限, 记 swap↔GPU peak/速度曲线. 到上限或过点后再 OOM 停.
  - probe-down (K2_ABL_MODE=probe, 旧逻辑): 从高 swap 往低找下界, 训练完成就停.

环境变量:
  K2_ABL_GPU=0/1        (默认 0=3080)
  K2_ABL_IMG=N          (默认 1024)
  K2_ABL_START_SWAP=N   (默认 8, sweep 模式跳过低 swap 必 OOM 段)
  K2_ABL_MAX_SWAP=N     (默认 28 = DiT 全块数, 全交换上限)
  K2_ABL_MODE=sweep|probe (默认 sweep)
  K2_ABL_OOM_STEP=N     (sweep 模式 OOM 步进, 默认 4)
  K2_ABL_FINE_STEP=N    (sweep 模式过点后细扫步进, 默认 2)
  K2_ABL_STEP=N         (probe 模式 OOM 步进, 默认 2; 兼容旧调用)
  K2_ABL_NF4_PATH=path  (磁盘 NF4)
  K2_ABL_STEPS=N        (每轮训练步数, 默认 30)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "scripts" / "krea2" / "probe_nf4_blockswap.py"
DEFAULT_NF4 = ROOT / "models" / "diffusion_models" / "krea2_raw_nf4.safetensors"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def parse_probe_output(stdout: str) -> dict:
    """从探针 stdout 抽 GPU peak / host RAM / loss 首末 / OOM / 通过."""
    out: dict = {}
    out["oom"] = "OutOfMemoryError" in stdout
    m = re.search(r"NF4 \+ block_swap GPU peak:\s*([\d.]+)GB", stdout)
    out["gpu_peak_gb"] = float(m.group(1)) if m else None
    m = re.search(r"host RAM RSS 峰值:\s*([\d.]+)GB", stdout)
    out["rss_peak_gb"] = float(m.group(1)) if m else None
    m = re.search(r"first5 avg=([\d.]+),\s*last5 avg=([\d.]+)", stdout)
    out["loss_first5"] = float(m.group(1)) if m else None
    out["loss_last5"] = float(m.group(2)) if m else None
    m = re.search(r"avg step:\s*([\d.]+)s", stdout)
    out["avg_step_s"] = float(m.group(1)) if m else None
    out["trained"] = "=== E. 验证 ===" in stdout
    m = re.search(r"方向 A 端到端通过:\s*(True|False)", stdout)
    out["probe_pass"] = (m.group(1) == "True") if m else False
    return out


def run_one(swap: int, env_over: dict) -> dict:
    """跑一轮指定 swap 的探针, 返回解析结果 + 耗时."""
    env = os.environ.copy()
    env.update(env_over)
    env["K2_NF4BS_SWAP"] = str(swap)
    t0 = time.time()
    # timeout 容忍: TE-CPU ~3min + 加载 ~2min + 30 步重搬运 ~6min, 给 20min 余量.
    proc = subprocess.run(
        [sys.executable, str(PROBE)],
        env=env, capture_output=True, text=True, timeout=1500,
    )
    elapsed = time.time() - t0
    parsed = parse_probe_output(proc.stdout + "\n" + proc.stderr)
    parsed["swap"] = swap
    parsed["exit_code"] = proc.returncode
    parsed["elapsed_s"] = elapsed
    parsed["stdout_tail"] = (proc.stdout + proc.stderr)[-800:]
    return parsed


def main() -> int:
    gpu = os.environ.get("K2_ABL_GPU", "0")
    img = _env_int("K2_ABL_IMG", 1024)
    start_swap = _env_int("K2_ABL_START_SWAP", 8)
    max_swap = _env_int("K2_ABL_MAX_SWAP", 28)
    mode = os.environ.get("K2_ABL_MODE", "sweep")
    oom_step = _env_int("K2_ABL_OOM_STEP", 4)
    fine_step = _env_int("K2_ABL_FINE_STEP", 2)
    step = _env_int("K2_ABL_STEP", 2)  # probe 模式用
    nf4_path = os.environ.get("K2_ABL_NF4_PATH", str(DEFAULT_NF4))
    n_steps = _env_int("K2_ABL_STEPS", 30)

    env_over = {
        "K2_NF4BS_GPU": gpu,
        "K2_NF4BS_IMG": str(img),
        "K2_NF4BS_STEPS": str(n_steps),
        "K2_NF4BS_TE_CPU": "1",
        "K2_NF4BS_NF4_PATH": nf4_path,
    }
    print(f"=== 3080 {img}×{img} NF4×block swap 显存边界测绘 (GPU={gpu}, mode={mode}) ===")
    if mode == "sweep":
        print(f"起步 swap={start_swap}, OOM +{oom_step} 逼近, 过点后 +{fine_step} 细扫, 上限 {max_swap}")
    else:
        print(f"起步 swap={start_swap}, OOM +{step}, 训练完成即停, 上限 {max_swap}")
    print(f"NF4={nf4_path}")
    print(f"{'swap':>4} | {'OOM':>5} | {'trained':>7} | {'pass':>5} | {'GPUpeak':>8} | {'RSSpeak':>8} | {'avgStep':>7} | {'lossF→L':>12} | {'耗时':>6}")
    print("-" * 100)

    results = []
    swap = start_swap
    passed_once = False  # sweep 模式: 是否已出现过训练完成点
    while swap <= max_swap:
        print(f"\n>>> 第 {len(results)+1} 轮: swap={swap} ...", flush=True)
        try:
            r = run_one(swap, env_over)
        except subprocess.TimeoutExpired:
            r = {"swap": swap, "oom": False, "trained": False, "probe_pass": False,
                 "gpu_peak_gb": None, "rss_peak_gb": None, "avg_step_s": None,
                 "loss_first5": None, "loss_last5": None, "exit_code": -1,
                 "elapsed_s": 1500.0, "stdout_tail": "TIMEOUT"}
        results.append(r)
        loss_str = (
            f"{r['loss_first5']:.4f}→{r['loss_last5']:.4f}"
            if r["loss_first5"] is not None and r["loss_last5"] is not None else "-"
        )
        print(
            f"{swap:>4} | {str(r['oom']):>5} | {str(r['trained']):>7} | "
            f"{str(r['probe_pass']):>5} | "
            f"{(r['gpu_peak_gb'] or 0):>7.2f}G | {(r['rss_peak_gb'] or 0):>7.2f}G | "
            f"{(r['avg_step_s'] or 0):>6.2f}s | {loss_str:>12} | {r['elapsed_s']:>5.0f}s"
        )
        if r["oom"] and r["stdout_tail"] != "TIMEOUT":
            print(f"    [OOM] stdout tail:\n{r['stdout_tail'][-300:]}")

        if mode == "probe":
            # 旧逻辑: 训练完成就停.
            if r["trained"]:
                print(f"\n>>> swap={swap} 训练完成, 显存边界找到, 停.")
                break
            if r["oom"] and swap + step <= max_swap:
                swap += step
                time.sleep(8)
                continue
            break

        # sweep 模式: 扫完整曲线.
        if r["oom"]:
            # OOM 后等 GPU 释放.
            time.sleep(8)
            if passed_once:
                # 过点后又 OOM (不该发生, swap 越高显存越省) → 异常, 停.
                print(f"\n>>> swap={swap} 过点后反 OOM (异常), 停.")
                break
            # 未过点: 大步逼近临界.
            swap += oom_step
            continue
        if r["trained"]:
            passed_once = True
            # 过点后细扫, 直到上限.
            if swap + fine_step <= max_swap:
                swap += fine_step
                continue
            break
        # 非 OOM 非训练完成 (超时/异常) → 停.
        break

    # === 边界报告 ===
    print("\n" + "=" * 100)
    print("=== 测绘边界报告 ===")
    trained = [r for r in results if r["trained"]]
    oom = [r for r in results if r["oom"]]
    if trained:
        print(f"  训练完成点: swap={[r['swap'] for r in trained]}")
        first_pass = trained[0]
        print(f"  最低通过 swap (临界): swap={first_pass['swap']}")
        print(f"    GPU peak: {first_pass['gpu_peak_gb']:.2f}GB (3080 可用 ~8.6GB)")
        print(f"    host RAM RSS: {first_pass['rss_peak_gb']:.2f}GB")
        print(f"    avg step: {first_pass['avg_step_s']:.2f}s")
        print(f"    loss: {first_pass['loss_first5']:.4f} -> {first_pass['loss_last5']:.4f}")
        if mode == "sweep" and len(trained) > 1:
            print(f"\n  swap↔GPU peak / 速度曲线 (过点后):")
            print(f"    {'swap':>4} | {'GPUpeak':>8} | {'RSSpeak':>8} | {'avgStep':>7}")
            for t in trained:
                print(f"    {t['swap']:>4} | {t['gpu_peak_gb']:>7.2f}G | {t['rss_peak_gb']:>7.2f}G | {t['avg_step_s']:>6.2f}s")
    else:
        print(f"  未找到训练完成点 (所有 {len(results)} 轮均未跑完 30 步).")
    if oom:
        print(f"  OOM 轮: swap={[r['swap'] for r in oom]} (真 CUDA 显存爆, 未到临界)")
    print(f"  全部轮次: {len(results)}, swap 序列: {[r['swap'] for r in results]}")
    return 0 if trained else 1


if __name__ == "__main__":
    sys.exit(main())
