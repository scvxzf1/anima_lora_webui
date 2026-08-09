# Krea-2-Raw NF4 在线量化 (QLoRA 路径)
#
# 把 bf16 DiT 的所有 nn.Linear 在线替换为 bnb Linear4bit(nf4), .to(device)
# 时 Params4bit.to() 检测 bnb_quantized=False 触发 _quantize. forward 走
# bnb.matmul_4bit (fused 反量化+matmul, 不生成完整 bf16 权重).
#
# 三层验证 (probe_nf4 / probe_nf4_lora / probe_nf4_train) 已过:
#   层1 确定性: 同 bf16 + 同 bnb 版本两次量化逐层 bit 一致.
#   层2 round-trip: dequant vs 原 bf16 cosine>0.99 (方向保持; rel_l2~10% 是
#     NF4 16-bucket 物理下限, 非数值无损).
#   层3 端到端: NF4+LoRA+grad-ckpt 1024 训练 loss 单调下降, 显存 10.49GB
#     (vs bf16 27.9GB), step 3.41s (vs 3.47s).
#   LoRA 兼容: Linear4bit 继承 nn.Linear 被 isinstance 命中; apply_to 后
#     org_forward 仍指向 Linear4bit.forward (反量化保住); 默认 down_init=kaiming
#     不读 Params4bit (weight_svd 仅显式传才触发, 默认 preset 不碰).
#
# NF4 物理事实 (子代理 + 亲自读 bnb 0.49.2 modules.py 核实):
#   13B DiT 264 Linear -> 6.63GB (4x 压缩), 量化耗时 ~19s 一次性.
#   Linear4bit(nn.Linear) forward 调 bnb.matmul_4bit (底层 MatMul4Bit.forward
#   做 F.linear(A, F.dequantize_4bit(B, quant_state)) — 反量化在 matmul 边界,
#   不持久化完整 bf16).
#   Params4bit.to(device) (modules.py:341-361): if device is not None and
#   not self.bnb_quantized: return self._quantize(device) — 量化触发点.

from __future__ import annotations

import logging
import os
import time

import torch
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

# NF4 量化超参 (QLoRA 论文默认). compute_dtype 是反量化后的计算精度,
# 非 weight 存储 dtype (weight 存 4-bit uint8).
QUANT_TYPE = "nf4"
COMPRESS_STATISTICS = True  # double quantization, 省量化元数据显存
COMPUTE_DTYPE = torch.bfloat16


def _collect_linear_paths(
    module: torch.nn.Module, prefix: str = ""
) -> list[tuple[str, torch.nn.Linear]]:
    """递归收集所有 nn.Linear 的 (dotted_name, module), 供量化替换."""
    out: list[tuple[str, torch.nn.Linear]] = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            out.append((path, child))
        else:
            out.extend(_collect_linear_paths(child, path))
    return out


def _set_nested(module: torch.nn.Module, path: str, value: torch.nn.Module) -> None:
    """按 dotted path setattr (a.b.c -> module.a.b.c)."""
    obj = module
    for p in path.split(".")[:-1]:
        obj = getattr(obj, p)
    setattr(obj, path.split(".")[-1], value)


def _get_nested(module: torch.nn.Module, path: str) -> torch.nn.Module:
    obj = module
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def _replace_with_linear4bit(
    parent: torch.nn.Module, attr: str, orig: torch.nn.Linear
) -> Linear4bit:
    """把 parent.attr 处的 nn.Linear 换成 Linear4bit, 拷贝权重与 bias.

    先在 CPU 构造并喂 bf16 权重 (Params4bit 此时 bnb_quantized=False),
    后续 model.to(device) 触发 _quantize.
    """
    new = Linear4bit(
        orig.in_features,
        orig.out_features,
        bias=orig.bias is not None,
        compute_dtype=COMPUTE_DTYPE,
        compress_statistics=COMPRESS_STATISTICS,
        quant_type=QUANT_TYPE,
        device="cpu",
    )
    with torch.no_grad():
        new.weight = Params4bit(
            orig.weight.data.to(torch.bfloat16).contiguous(),
            requires_grad=False,
            compress_statistics=COMPRESS_STATISTICS,
            quant_type=QUANT_TYPE,
            module=new,
        )
        if orig.bias is not None:
            new.bias = torch.nn.Parameter(
                orig.bias.data.to(torch.bfloat16).contiguous()
            )
    setattr(parent, attr, new)
    return new


def quantize_dit_to_nf4(
    model: torch.nn.Module, device: torch.device, *, keep_orig_weights: bool = False
) -> dict[str, torch.Tensor]:
    """遍历 model 所有 nn.Linear 替换为 Linear4bit(nf4), .to(device) 触发量化.

    Args:
        keep_orig_weights: 保留 {path: orig_bf16_weight} 供 round-trip 对比
            (探针用). 生产路径默认 False — clone 264 个 bf16 权重 (~25GB)
            既占 CPU 内存又耗 ~140s, 生产无需 round-trip 应跳过.

    前置: model 的 Linear 权重已是 bf16 (load_krea2_dit 加载后). 量化在 .to(device)
    时由 Params4bit.to() 触发, 这里只做 Linear->Linear4bit 替换 + 最终 .to.

    Returns:
        keep_orig_weights=True 时返回 {path: orig_bf16_weight}; False 时返回空 dict.
    """
    t_start = time.time()
    paths = _collect_linear_paths(model)
    logger.info(f"NF4 量化: 发现 {len(paths)} 个 nn.Linear 待量化")

    orig_weights: dict[str, torch.Tensor] = {}
    for path, orig in paths:
        if keep_orig_weights:
            orig_weights[path] = orig.weight.data.to(torch.bfloat16).contiguous().clone()
        parent_path, attr = path.rsplit(".", 1) if "." in path else ("", path)
        parent = _get_nested(model, parent_path) if parent_path else model
        _replace_with_linear4bit(parent, attr, orig)
    t_replaced = time.time()
    logger.info(
        f"NF4 Linear->Linear4bit 替换完成: {len(paths)} 个, "
        f"耗时 {t_replaced - t_start:.1f}s"
    )

    # .to(device) 触发量化: Params4bit.to() 检测 bnb_quantized=False 调 _quantize.
    model.to(device)
    t_done = time.time()
    logger.info(
        f"NF4 量化完成: 总耗时 {t_done - t_start:.1f}s "
        f"(替换 {t_replaced - t_start:.1f}s + 量化&H2D {t_done - t_replaced:.1f}s)"
    )
    return orig_weights


# === NF4 量化权重磁盘落盘/加载 ===
#
# 动机: 在线量化 (quantize_dit_to_nf4) 的 .to(device) 要把整个 bf16 DiT (26GB)
# 一次性搬到 GPU 触发量化, 需要一张能放 26GB bf16 的卡 (PG199 32GB). 3080 等
# 8-12GB 卡无法在线量化. 落盘后小卡直接加载 6.6GB NF4 权重, 绕过硬约束.
#
# 契约 (bnb 0.49.2 官方 safetensors round-trip):
#   存: Params4bit.data (4-bit uint8) + QuantState.as_dict(packed=True) —— 后者把
#       absmax/quant_map/nested_* (都是 tensor) 和 quant_type/blocksize/dtype/
#       shape/nested_offset (非 tensor, pack 进 quant_state.bitsandbytes__nf4 tensor)
#       全打包成 dict[str, Tensor], 可直接 save_file.
#   读: QuantState.from_dict(qs_dict, device) 自动 unpack_tensor_to_dict 解 meta,
#       Params4bit.from_prequantized(data, qs_dict, ...) 重建 bnb_quantized=True 的
#       Params4bit, 挂回 Linear4bit.weight.
# 键命名: 每个 Linear4bit path P 存:
#   {P}.weight                      = 4-bit uint8 data
#   {P}.weight.quant_state.<qs_key> = QuantState.as_dict(packed=True) 的各项
#   {P}.bias                        = bias (若有)
# 另存一个 {__nf4_meta__} tensor (pack 元数据: Linear4bit path 列表 + 各自 in/out
# features + has_bias), 供加载时重建 Linear4bit 模块结构.


def _nf4_linear_paths(model: torch.nn.Module) -> list[tuple[str, Linear4bit]]:
    """收集所有 Linear4bit 的 (dotted_path, module), 供存盘/加载遍历."""
    out: list[tuple[str, Linear4bit]] = []
    for name, child in model.named_modules():
        if isinstance(child, Linear4bit):
            out.append((name, child))
    return out


def save_nf4_dit(model: torch.nn.Module, out_path: str) -> dict[str, int]:
    """把已 NF4 量化的 DiT 存成 safetensors (4-bit 码 + quant_state).

    前置: model 已过 quantize_dit_to_nf4 (Linear 全是 Linear4bit, bnb_quantized=True).
    产出: out_path 一个 safetensors 文件, 含 264 个 Linear4bit 的 4-bit 码 + 各自
    quant_state (含 state2 双重量化) + bias + 模块结构 meta.

    Returns:
        {"linear4bit_count": N, "bytes": 文件字节数}
    """
    from bitsandbytes.functional import QuantState

    out_path = str(out_path)
    l4_paths = _nf4_linear_paths(model)
    if not l4_paths:
        raise ValueError("save_nf4_dit: model 中没有 Linear4bit, 先调 quantize_dit_to_nf4")

    state_dict: dict[str, torch.Tensor] = {}
    meta_rows: list[tuple[int, int, int, int]] = []  # (path_idx, in_f, out_f, has_bias)
    path_list: list[str] = []

    for path, lin in l4_paths:
        w = lin.weight
        if not isinstance(w, Params4bit) or not w.bnb_quantized:
            raise ValueError(
                f"save_nf4_dit: {path}.weight 不是已量化 Params4bit "
                f"(bnb_quantized={getattr(w, 'bnb_quantized', None)})"
            )
        state_dict[f"{path}.weight"] = w.data.to("cpu").contiguous()
        # quant_state.as_dict(packed=True): 纯 dict[str, Tensor], 可 safetensors 存.
        qs = w.quant_state
        qs_dict = QuantState.as_dict(qs, packed=True)
        for qs_key, qs_val in qs_dict.items():
            state_dict[f"{path}.weight.quant_state.{qs_key}"] = qs_val.to("cpu")
        if lin.bias is not None:
            state_dict[f"{path}.bias"] = lin.bias.data.to("cpu").contiguous()
        has_bias = 1 if lin.bias is not None else 0
        path_list.append(path)
        meta_rows.append((len(path_list) - 1, lin.in_features, lin.out_features, has_bias))

    # meta: 路径列表 (字符串 pack 成 1D int tensor, 每字符一个 int) + 结构行.
    # 简单可靠: 把每个 path 编码成 utf-8 字节序列, pack 进一个 1D uint8 tensor,
    # 配一个 offsets tensor 标定每个 path 的起止. meta_rows pack 成 (N,4) int tensor.
    import torch as _torch

    all_bytes = b"\x00".join(p.encode("utf-8") for p in path_list)
    path_bytes = _torch.tensor(list(all_bytes), dtype=_torch.uint8)
    # offset[i] = 第 i 个 path 的起始字节; 末尾加总长标定最后一个.
    offsets = []
    cur = 0
    for p in path_list:
        offsets.append(cur)
        cur += len(p.encode("utf-8")) + 1  # +1 for \x00 separator
    path_offsets = _torch.tensor(offsets + [cur], dtype=_torch.int64)
    struct_rows = _torch.tensor(meta_rows, dtype=_torch.int64)
    state_dict["__nf4_meta__.path_bytes"] = path_bytes
    state_dict["__nf4_meta__.path_offsets"] = path_offsets
    state_dict["__nf4_meta__.struct_rows"] = struct_rows

    save_file(state_dict, out_path)
    nbytes = os.path.getsize(out_path)
    logger.info(
        f"save_nf4_dit: {len(l4_paths)} 个 Linear4bit -> {out_path} "
        f"({nbytes / 1e9:.2f}GB, {len(state_dict)} keys)"
    )
    return {"linear4bit_count": len(l4_paths), "bytes": nbytes}


def _decode_nf4_meta(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, int, int, bool]]:
    """从 state_dict 解出 [(path, in_features, out_features, has_bias)]."""
    path_bytes = state_dict["__nf4_meta__.path_bytes"].tolist()
    path_offsets = state_dict["__nf4_meta__.path_offsets"].tolist()
    struct_rows = state_dict["__nf4_meta__.struct_rows"].tolist()
    raw = bytes(path_bytes)
    paths = []
    for i in range(len(path_offsets) - 1):
        seg = raw[path_offsets[i]:path_offsets[i + 1]]
        # 去掉末尾 \x00 分隔符 (最后一个段没有, 容错).
        if seg.endswith(b"\x00"):
            seg = seg[:-1]
        paths.append(seg.decode("utf-8"))
    out = []
    for row in struct_rows:
        path_idx, in_f, out_f, has_bias = row
        out.append((paths[path_idx], int(in_f), int(out_f), bool(has_bias)))
    return out


def load_nf4_dit_into(
    model: torch.nn.Module,
    nf4_path: str,
    device: torch.device,
) -> int:
    """把磁盘上的 NF4 权重加载进已构造 (bf16 空结构) 的 DiT, 替换所有 Linear.

    前置: model 是 bf16 SingleStreamDiT (load_krea2_dit(nf4=False) 构造的空结构).
    流程: 读 safetensors -> 解 meta 得 Linear path 列表 -> 对每个 path 把 nn.Linear
    替换成 Linear4bit (空壳) -> 用 from_prequantized 重建 Params4bit 挂回 -> 偏置挂回.
    不重新量化 (bnb_quantized=True 直接来自存盘), 无需 26GB bf16 在 GPU.

    Returns: 加载的 Linear4bit 个数.
    """
    from bitsandbytes.functional import QuantState
    from safetensors.torch import load_file

    state_dict = load_file(nf4_path, device="cpu")
    entries = _decode_nf4_meta(state_dict)
    count = 0
    for path, in_f, out_f, has_bias in entries:
        parent_path, attr = path.rsplit(".", 1) if "." in path else ("", path)
        parent = _get_nested(model, parent_path) if parent_path else model
        # 构造空壳 Linear4bit (CPU), compute_dtype/quant_type 后面由 Params4bit 覆盖.
        new = Linear4bit(
            in_f,
            out_f,
            bias=has_bias,
            compute_dtype=COMPUTE_DTYPE,
            compress_statistics=COMPRESS_STATISTICS,
            quant_type=QUANT_TYPE,
            device="cpu",
        )
        # 收集本 path 的 quant_state 项, 重建 QuantState + Params4bit.
        qs_prefix = f"{path}.weight.quant_state."
        qs_dict = {
            k[len(qs_prefix):]: v for k, v in state_dict.items() if k.startswith(qs_prefix)
        }
        data = state_dict[f"{path}.weight"]
        p4 = Params4bit.from_prequantized(
            data,
            qs_dict,
            requires_grad=False,
            device="cpu",  # 先 CPU, 整体 .to(device) 由调用方控
            module=new,
        )
        new.weight = p4
        if has_bias:
            new.bias = torch.nn.Parameter(state_dict[f"{path}.bias"].to(torch.bfloat16))
        setattr(parent, attr, new)
        count += 1
    logger.info(f"load_nf4_dit_into: 从 {nf4_path} 加载 {count} 个 Linear4bit (CPU)")
    return count
