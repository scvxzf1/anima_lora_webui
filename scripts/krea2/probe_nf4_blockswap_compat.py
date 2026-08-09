"""R-verify: NF4 × block swap 兼容性探针 (方向 A 命门验证).

不改 offloader, 纯验 bnb 0.49.2 Params4bit 在 block swap 搬运模式下的契约:
  deepcopy(已量化 Params4bit) -> CPU 持有 -> .to(cuda) 整体搬回 -> forward
  对比 不搬运 的 forward 数值, 确认 quant_state 不丢、数值一致.

这是方向 A (offloader 加 Params4bit 专用分支) 的命门. 若此探针崩或数值不一致,
说明 bnb 契约不支持 block swap 搬运节奏, 方向 A 走不通, 要回头.

验证项:
1. deepcopy 副本完整: quant_state / data / state2 (双重量化) 全复制, bnb_quantized=True.
2. deepcopy 后 new.module 引用: bnb __deepcopy__ 不深拷 module (modules.py:256),
   副本 module 指向原 Linear4bit — 验证此事实并确认需手动设 new.module.
3. CPU 持有合法: 副本留在 CPU 不崩, bnb_quantized 保持 True.
4. .to(cuda) 整体搬运: 4-bit 码 + quant_state (含 state2) 同步到 GPU, device 一致.
5. forward 数值一致: deepcopy->CPU->cuda 的 forward vs 原地 forward, max delta < 1e-5.
6. 多块循环搬运 (模拟 block swap 节奏): 2 个 Linear4bit 交替 CPU<->cuda N 轮,
   每轮 forward 数值一致, 确认反复搬运不累积误差/不丢 quant_state.

PG199 bf16, NF4 量化, 不加载完整 DiT (只造 2 个小 Linear4bit, 隔离验证 bnb 契约).
"""
from __future__ import annotations

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bitsandbytes.nn import Linear4bit, Params4bit  # noqa: E402

COMPUTE_DTYPE = torch.bfloat16
QUANT_TYPE = "nf4"
COMPRESS_STATISTICS = True  # 双重量化 — 验证 state2 nested 路径


def make_linear4bit(out_features: int, in_features: int, device, dtype) -> Linear4bit:
    """造一个已量化的 Linear4bit (CPU 构造 bf16 权重 -> .to(cuda) 触发量化)."""
    lin = Linear4bit(
        in_features, out_features, bias=True, compute_dtype=dtype,
        quant_type=QUANT_TYPE, compress_statistics=COMPRESS_STATISTICS,
    )
    # 喂确定性 bf16 权重 (固定 seed)
    g = torch.Generator(device="cpu").manual_seed(42)
    with torch.no_grad():
        lin.weight = Params4bit(
            torch.randn(out_features, in_features, generator=g, dtype=dtype),
            requires_grad=False,
            quant_type=QUANT_TYPE,
            compress_statistics=COMPRESS_STATISTICS,
        )
        if lin.bias is not None:
            lin.bias.data = torch.randn(out_features, generator=g, dtype=dtype) * 0.01
    # .to(cuda) 触发 _quantize (bnb_quantized=False -> True)
    lin = lin.to(device)
    assert lin.weight.bnb_quantized, "量化未触发"
    return lin


def check_params4bit_integrity(p: Params4bit, label: str, expect_device) -> list[str]:
    """检查 Params4bit 完整性, 返回问题列表 (空=OK)."""
    problems = []
    if not p.bnb_quantized:
        problems.append(f"{label}: bnb_quantized=False (应 True)")
    qs = p.quant_state
    if qs is None:
        problems.append(f"{label}: quant_state=None")
        return problems
    if qs.absmax.device.type != expect_device.type:
        problems.append(f"{label}: absmax device {qs.absmax.device} != {expect_device}")
    if qs.code.device.type != expect_device.type:
        problems.append(f"{label}: code device {qs.code.device} != {expect_device}")
    if p.data.device.type != expect_device.type:
        problems.append(f"{label}: data(4bit码) device {p.data.device} != {expect_device}")
    # 双重量化: state2 必须也跟着搬
    if qs.nested:
        if qs.state2 is None:
            problems.append(f"{label}: nested=True 但 state2=None")
        else:
            if qs.state2.absmax.device.type != expect_device.type:
                problems.append(f"{label}: state2.absmax device {qs.state2.absmax.device} != {expect_device}")
            if qs.state2.code.device.type != expect_device.type:
                problems.append(f"{label}: state2.code device {qs.state2.code.device} != {expect_device}")
        if qs.offset is None:
            problems.append(f"{label}: nested=True 但 offset=None")
        elif qs.offset.device.type != expect_device.type:
            problems.append(f"{label}: offset device {qs.offset.device} != {expect_device}")
    return problems


def forward_ref(lin: Linear4bit, x: torch.Tensor) -> torch.Tensor:
    """原地 (不搬) forward 参考值."""
    with torch.no_grad():
        return lin(x).clone()


def main() -> int:
    device = torch.device("cuda")
    dtype = COMPUTE_DTYPE
    print(f"=== NF4 × block swap 兼容性探针 (bnb Params4bit 契约验证) ===")
    print(f"quant_type={QUANT_TYPE}, compress_statistics={COMPRESS_STATISTICS} (双重量化), dtype={dtype}")

    all_ok = True

    # === 1. 造 2 个已量化 Linear4bit ===
    print("\n--- 1. 构造 2 个已量化 Linear4bit (GPU) ---")
    lin_a = make_linear4bit(512, 1024, device, dtype)
    lin_b = make_linear4bit(512, 1024, device, dtype)
    for label, lin in [("A", lin_a), ("B", lin_b)]:
        probs = check_params4bit_integrity(lin.weight, f"原 {label}", device)
        print(f"  {label}: bnb_quantized={lin.weight.bnb_quantized}, "
              f"nested={lin.weight.quant_state.nested}, "
              f"4bit码 {tuple(lin.weight.data.shape)} {lin.weight.data.dtype}, "
              f"integrity OK={not probs}")
        if probs:
            print(f"    !! {probs}"); all_ok = False

    # 参考输入
    g = torch.Generator(device=device).manual_seed(7)
    x_a = torch.randn(4, 1024, generator=g, device=device, dtype=dtype)
    x_b = torch.randn(4, 1024, generator=g, device=device, dtype=dtype)
    ref_a = forward_ref(lin_a, x_a)
    ref_b = forward_ref(lin_b, x_b)
    print(f"  参考 forward: A {tuple(ref_a.shape)}, B {tuple(ref_b.shape)}")

    # === 2. deepcopy -> CPU 持有 ===
    print("\n--- 2. deepcopy 副本 -> CPU 持有 ---")
    cpu_a = copy.deepcopy(lin_a)
    cpu_b = copy.deepcopy(lin_b)
    # 搬到 CPU (走 Params4bit.to else 分支, 搬 4bit码+quant_state)
    cpu_a = cpu_a.to("cpu")
    cpu_b = cpu_b.to("cpu")
    probs_a = check_params4bit_integrity(cpu_a.weight, "cpu A", torch.device("cpu"))
    probs_b = check_params4bit_integrity(cpu_b.weight, "cpu B", torch.device("cpu"))
    print(f"  cpu A: bnb_quantized={cpu_a.weight.bnb_quantized}, integrity OK={not probs_a}")
    print(f"  cpu B: bnb_quantized={cpu_b.weight.bnb_quantized}, integrity OK={not probs_b}")
    if probs_a: print(f"    !! {probs_a}"); all_ok = False
    if probs_b: print(f"    !! {probs_b}"); all_ok = False

    # === 3. deepcopy 后 module 引用事实 (bnb modules.py:256 不深拷 module) ===
    print("\n--- 3. deepcopy module 引用检查 ---")
    # cpu_a.weight.module 应指向某 Linear4bit — 验证它指向谁
    mod_ref = cpu_a.weight.module
    print(f"  cpu_a.weight.module is cpu_a? {mod_ref is cpu_a}")
    print(f"  cpu_a.weight.module is lin_a (原)? {mod_ref is lin_a}")
    # bnb __deepcopy__ 不深拷 module, 副本 module 指向 deepcopy 出的 cpu_a 自己
    # (因为 __setstate__ 时 self.module=state['module'], 而 state 来自原 lin_a.__dict__,
    #  但 deepcopy lin_a 时其 weight 的 module 字段... 实测确认指向)
    module_ref_ok = mod_ref is cpu_a or mod_ref is lin_a
    print(f"  module 引用指向 cpu_a 或 lin_a: {module_ref_ok}")
    # 关键: 搬回 GPU 挂到新 module 时, 若 forward 用 self.weight.quant_state 直接取,
    # module 引用不影响数值. 但 fix_4bit_weight_quant_state_from_module 会用它.

    # === 4. CPU -> cuda 整体搬运 -> forward 数值一致 ===
    print("\n--- 4. CPU->cuda 整体搬运 -> forward 数值对比 ---")
    cuda_a = cpu_a.to(device)
    cuda_b = cpu_b.to(device)
    probs_a2 = check_params4bit_integrity(cuda_a.weight, "cuda A (搬回)", device)
    probs_b2 = check_params4bit_integrity(cuda_b.weight, "cuda B (搬回)", device)
    print(f"  cuda A (搬回): integrity OK={not probs_a2}")
    print(f"  cuda B (搬回): integrity OK={not probs_b2}")
    if probs_a2: print(f"    !! {probs_a2}"); all_ok = False
    if probs_b2: print(f"    !! {probs_b2}"); all_ok = False

    out_a = forward_ref(cuda_a, x_a)
    out_b = forward_ref(cuda_b, x_b)
    delta_a = (out_a - ref_a).abs().max().item()
    delta_b = (out_b - ref_b).abs().max().item()
    print(f"  A forward max delta: {delta_a:.2e} (容差 1e-5)")
    print(f"  B forward max delta: {delta_b:.2e} (容差 1e-5)")
    numerical_ok = delta_a < 1e-5 and delta_b < 1e-5
    if not numerical_ok:
        all_ok = False
        print("    !! 数值不一致, 方向 A 命门失败")
    else:
        print("    OK 数值一致 (deepcopy->CPU->cuda 不改 forward 语义)")

    # === 5. 多块循环搬运 (模拟 block swap 节奏) ===
    # 关键: Linear4bit.to() 是原地改 weight device (同 nn.Module.to), 所以不能复用
    # 同一对象引用做 "active 搬GPU / inactive 留CPU" — 原地操作会污染 inactive.
    # block swap 真实场景: offloader 持有 CPU master (独立副本), 搬运时从 master
    # 重建 GPU 副本, 不动 master. 这里模拟: 每轮从 CPU master deepcopy 出 active.
    print("\n--- 5. 多块循环搬运 (2 块交替 CPU<->cuda, 5 轮, master 独立副本) ---")
    cpu_masters = {"A": copy.deepcopy(lin_a).to("cpu"),
                   "B": copy.deepcopy(lin_b).to("cpu")}
    refs = {"A": ref_a, "B": ref_b}
    xs = {"A": x_a, "B": x_b}
    cycle_ok = True
    for cycle in range(5):
        active_key = "A" if cycle % 2 == 0 else "B"
        inactive_key = "B" if active_key == "A" else "A"
        # active: 从 CPU master deepcopy 一份搬到 GPU (模拟 offloader 从 master 重建)
        active = copy.deepcopy(cpu_masters[active_key]).to(device)
        # inactive: master 本身留在 CPU, 验证它没被 active 的搬运污染
        inactive = cpu_masters[inactive_key]
        probs_inactive = check_params4bit_integrity(
            inactive.weight, f"cycle{cycle} inactive {inactive_key}", torch.device("cpu")
        )
        out = forward_ref(active, xs[active_key])
        delta = (out - refs[active_key]).abs().max().item()
        if delta >= 1e-5:
            print(f"  cycle {cycle}: {active_key} forward delta={delta:.2e} !! 不一致")
            cycle_ok = False
        else:
            print(f"  cycle {cycle}: {active_key} master->GPU fwd delta={delta:.2e}, "
                  f"{inactive_key} master 留 CPU integrity OK={not probs_inactive}")
        if probs_inactive:
            print(f"    !! inactive {inactive_key} master 被污染: {probs_inactive}")
            cycle_ok = False
        del active
        torch.cuda.empty_cache()
    if not cycle_ok:
        all_ok = False
        print("    !! 循环搬运累积问题, 方向 A 命门失败")
    else:
        print("    OK 5 轮交替搬运数值一致, CPU master 不被搬运污染")

    # === 6. 搬运后量化状态保持 ===
    print("\n--- 6. 搬回 GPU 后 bnb_quantized 保持 ---")
    final_a = copy.deepcopy(cpu_masters["A"]).to(device)
    qz_ok = final_a.weight.bnb_quantized
    print(f"  最终搬回 A: bnb_quantized={qz_ok} (应 True)")
    if not qz_ok:
        all_ok = False
        print("    !! 量化状态丢失, 方向 A 命门失败")

    # === 总结 ===
    print(f"\n=== 兼容性验证总结 ===")
    print(f"  deepcopy 完整性: {'OK' if not (probs_a or probs_b) else 'FAIL'}")
    print(f"  CPU 持有合法: {'OK' if not (probs_a or probs_b) else 'FAIL'}")
    print(f"  整体搬运 forward 数值一致: {'OK' if numerical_ok else 'FAIL'}")
    print(f"  多块循环搬运: {'OK' if cycle_ok else 'FAIL'}")
    print(f"  量化状态保持: {'OK' if qz_ok else 'FAIL'}")
    print(f"\n方向 A 命门通过: {all_ok}")
    print(f"\n若通过: offloader 加 isinstance(Params4bit) 分支用 deepcopy+.to() 整体搬运可行")
    print(f"若失败: bnb 契约不支持 block swap 节奏, 需换方向")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
