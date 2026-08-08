"""Krea-2-Raw ModelFamily 实现 (阶段 4).

沿 docs/proposal/krea2_raw_migration.md §3 的 ModelFamily 边界落地. 本模块是
Krea-2 在训练侧的承重接口: forward_for_loss 把 (5D latent, 文本 hiddens/mask,
timestep) 转成 Krea-2 single-stream DiT 需要的 (patchified img, context, pos,
mask) 并跑 forward, 还原成与 latent 同形的 velocity (5D).

阶段 4 仅在自包含训练探针里验证; 正式串通 train.py / noise_target.py 是阶段 6
配置收口的事 (反上帝守则: 不在一轮里同时改架构和改行为).

关键不变量:
- latent 5D (B, C, T=1, H, W), 单例时间轴 dim 2 (同 anima, AGENTS.md 不可破坏).
- 进 DiT 前 squeeze(2) 到 4D (B, C, H, W), 出 DiT 后 unsqueeze(2) 回 5D.
- text_emb = (hiddens (B, L, 12, 2560), mask (B, L) bool) — R1 契约: mask 屏蔽
  padding, 不二次置零 (与 anima zero-sink 不同, 见 stage1 findings).
- timestep = σ ∈ [0, 1] float, DiT 内部 temb 做 t*tfactor(1e3) sinusoidal embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange, repeat
from torch import Tensor


@dataclass
class Krea2TextEmbedding:
    """Krea-2 文本编码输出 (forward_for_loss 的 text_emb 入参).

    hiddens: (B, L_txt, 12, 2560) — Qwen3-VL 选 12 层 stack dim=2, 切 prefix 34 后.
    mask: (B, L_txt) bool — True=有效 token, False=padding (R1: 屏蔽不置零).
    """

    hiddens: Tensor
    mask: Tensor


def prepare_img_tokens(
    img: Tensor,
    txtlen: int,
    patch: int,
    txtmask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """patchify latent + 构造 3D pos / key-padding mask.

    移植自 krea-ai/krea-2 sampling.prepare (同 scripts/krea2/probe_*.py).
    img: (B, C, H, W) 4D latent (已 squeeze(2) 离开 5D 时间轴).
    返回:
      img_tokens (B, L_img, patch*patch*C),
      pos (B, L_txt+L_img, 3),
      mask (B, L_txt+L_img) bool.
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def forward_for_loss(
    dit: torch.nn.Module,
    latents_5d: Tensor,
    text_emb: Krea2TextEmbedding | tuple[Tensor, Tensor],
    t: Tensor,
    **kw: Any,
) -> Tensor:
    """Krea-2 承重接口: 5D latent → DiT forward → 5D velocity.

    latents_5d: (B, C, T=1, H, W) — anima 5D 不变量.
    text_emb: Krea2TextEmbedding 或 (hiddens, mask) tuple.
    t: (B,) timestep = σ ∈ [0, 1].

    返回 velocity (B, C, T=1, H, W) — 与 latents_5d 同形, 用于 flow-matching loss.
    """
    if isinstance(text_emb, Krea2TextEmbedding):
        hiddens, txtmask = text_emb.hiddens, text_emb.mask
    else:
        hiddens, txtmask = text_emb

    # 5D → 4D (squeeze 单例时间轴 dim 2; 不用裸 squeeze 防误伤 batch=1).
    b, c, _t, h, w = latents_5d.shape
    latents_4d = latents_5d.squeeze(2)
    assert latents_4d.shape == (b, c, h, w)

    patch = dit.config.patch
    channels = dit.config.channels
    txtlen = hiddens.shape[1]

    img_tokens, pos, mask = prepare_img_tokens(
        latents_4d, txtlen, patch, txtmask
    )
    context = hiddens

    out = dit(img=img_tokens, context=context, t=t, pos=pos, mask=mask)
    # out: (B, L_img, patch*patch*C) → 还原 4D → 5D.
    latent_h, latent_w = h // patch, w // patch
    velocity_4d = rearrange(
        out,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=latent_h,
        w=latent_w,
        c=channels,
        ph=patch,
        pw=patch,
    )
    velocity_5d = velocity_4d.unsqueeze(2)
    return velocity_5d
