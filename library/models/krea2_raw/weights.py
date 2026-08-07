# Krea-2-Raw 权重加载
#
# 路径 B（裸移植 mmdit.py）：本地权重是 mmdit.py 原生命名
# （blocks.N.attn.wq 等），非 diffusers 命名。load_state_dict(strict=True,
# assign=True) 直接吃，无需 key 重映射、无需 qkv/kv/adaln fuse。
#
# 详见 docs/findings/krea2_raw_migration_stage0_findings.md §R4/R8。

from __future__ import annotations

import logging
import os
from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file

from library.env import resolve_under_home
from library.log import setup_logging
from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT

setup_logging()
logger = logging.getLogger(__name__)


def load_krea2_dit(
    dit_path: Union[str, os.PathLike],
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
    config: Optional[SingleMMDiTConfig] = None,
    eval: bool = True,
) -> SingleStreamDiT:
    """加载 Krea-2-Raw DiT 单文件权重（路径 B，原生命名，strict 加载）。

    Args:
        dit_path: diffusion_models/krea2_raw_bf16.safetensors 路径。
        device: 最终放置设备。权重在 CPU 上 strict 加载后再 to(device)。
        dtype: 目标 dtype；None 表示保持文件原 dtype。
        config: 可选 config 覆盖；默认用 SingleMMDiTConfig.krea2_raw()。
        eval: 是否调用 model.eval()。

    Returns:
        SingleStreamDiT 模型（权重已加载）。
    """
    dit_path = str(resolve_under_home(dit_path))
    if not os.path.exists(dit_path):
        raise FileNotFoundError(f"Krea-2 DiT 权重不存在: {dit_path}")

    if config is None:
        config = SingleMMDiTConfig.krea2_raw()

    # init_empty_weights 避免 __init__ 期间实例化全部权重，节省加载峰值内存。
    # 与 anima load_anima_model 同构（见 library/anima/weights.py:177）。
    with init_empty_weights():
        model = SingleStreamDiT(config)
        if dtype is not None:
            model = model.to(dtype)

    # Krea-2 权重是 mmdit.py 原生命名，无 prefix、无 fuse——直接 strict 加载。
    # assign=True 让加载器直接接管 tensor storage，避免一次额外 copy。
    logger.info(f"Loading Krea-2 DiT from {dit_path}, dtype={dtype}")
    state_dict = load_file(dit_path, device="cpu")

    # 归一化 mod.lin 这种单 Parameter key 的命名一致性：
    # state_dict 里是 "blocks.N.mod.lin"，model 里也是 "blocks.N.mod.lin"（因
    # DoubleSharedModulation.lin 直接是 nn.Parameter，其 name 在 ModuleList 里
    # 是 "mod.lin"）。strict 加载会直接对齐。
    missing, unexpected = model.load_state_dict(state_dict, strict=True, assign=True)
    # strict=True 下 missing/unexpected 必为空；此处只做防御性日志。
    if missing or unexpected:
        raise RuntimeError(
            f"Krea-2 DiT strict 加载失败: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    logger.info(
        f"Loaded Krea-2 DiT ({len(state_dict)} keys) from {dit_path}"
    )

    if dtype is not None:
        model = model.to(dtype)
    model = model.to(device)
    if eval:
        model.eval()

    return model


def load_vae(
    vae_path: Union[str, os.PathLike],
    input_channels: int = 3,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    spatial_chunk_size: Optional[int] = None,
    disable_cache: bool = False,
    dtype: Optional[torch.dtype] = None,
    eval: bool = True,
):
    """加载 Krea-2 VAE——直接复用 anima 的 AutoencoderKLQwenImage 加载器。

    R2 已验证：Krea-2 VAE 与 anima 是同一个 AutoencoderKLQwenImage，
    latents_mean/std 逐元素一致，encode/decode 严格互逆（PSNR 37-41 dB）。
    详见 docs/findings/krea2_raw_migration_stage0_findings.md §R2。
    """
    from library.models.qwen_vae import load_vae as _load_vae

    return _load_vae(
        vae_path=str(resolve_under_home(vae_path)),
        input_channels=input_channels,
        device=device,
        disable_mmap=disable_mmap,
        spatial_chunk_size=spatial_chunk_size,
        disable_cache=disable_cache,
        dtype=dtype,
        eval=eval,
    )


def inspect_dit_keys(dit_path: Union[str, os.PathLike]) -> dict:
    """读取 DiT 单文件的 key 清单和 shape（调试/核验用）。"""
    dit_path = str(resolve_under_home(dit_path))
    keys: list[str] = []
    shapes: dict[str, torch.Size] = {}
    with safe_open(dit_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            keys.append(k)
            shapes[k] = tuple(f.get_tensor(k).shape)
    return {"keys": sorted(keys), "shapes": shapes, "count": len(keys)}


def inspect_dit_config(
    dit_path: Union[str, os.PathLike],
) -> dict[str, int]:
    """从权重 shape 反推 Krea-2 DiT config（核验用）。

    返回的 dict 可与 SingleMMDiTConfig.krea2_raw() 对照。
    """
    info = inspect_dit_keys(dit_path)
    sh = info["shapes"]

    def _shape(k: str) -> tuple[int, ...]:
        return tuple(int(x) for x in sh[k])

    # first.weight: (features, channels*patch**2)
    fw = _shape("first.weight")
    features = fw[0]
    patch_sq_channels = fw[1]
    # tmlp.0.weight: (features, tdim)
    tdim = _shape("tmlp.0.weight")[1]
    # txtmlp.1.weight: (features, txtdim)
    txtdim = _shape("txtmlp.1.weight")[1]
    # blocks.0.attn.wq.weight: (features, features) -> heads = features/headdim
    # blocks.0.attn.wk.weight: (kvheads*headdim, features) -> kvheads
    wq = _shape("blocks.0.attn.wq.weight")
    wk = _shape("blocks.0.attn.wk.weight")
    headdim = wq[0] // (wq[1] // wq[0]) if False else None  # placeholder
    # 直接：wq 输出 = heads*headdim = features，headdim = wq[1]//heads
    # 但更稳：wq shape (features, features) 且 heads*headdim=features；
    # wk shape (kvheads*headdim, features)，所以 headdim = wk[0]//kvheads。
    # 已知 Krea-2 headdim=128，反推 heads/kvheads：
    headdim = 128
    heads = wq[0] // headdim
    kvheads = wk[0] // headdim
    # mlp.up.weight: (mlpdim, features) -> multiplier = mlpdim / (2*features/3)
    mlpdim = _shape("blocks.0.mlp.up.weight")[0]
    multiplier = round(mlpdim / (2 * features / 3))
    # blocks 层数
    block_ids = {
        int(k.split(".")[1])
        for k in info["keys"]
        if k.startswith("blocks.") and k.split(".")[1].isdigit()
    }
    layers = len(block_ids)
    # tproj.1.weight: (features*6, features)
    assert _shape("tproj.1.weight") == (features * 6, features)
    # txtfusion.projector.weight: (1, txtlayers)
    txtlayers = _shape("txtfusion.projector.weight")[1]
    # channels/patch
    # patch_sq_channels = channels * patch**2；已知 patch=2 -> channels = patch_sq_channels/4
    patch = 2
    channels = patch_sq_channels // (patch**2)
    return {
        "features": features,
        "tdim": tdim,
        "txtdim": txtdim,
        "heads": heads,
        "kvheads": kvheads,
        "headdim": headdim,
        "multiplier": multiplier,
        "layers": layers,
        "patch": patch,
        "channels": channels,
        "txtlayers": txtlayers,
    }
