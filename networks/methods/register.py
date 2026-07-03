"""Register-token adapter for a frozen Anima DiT."""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.methods.base import AdapterNetworkBase
from networks.register_injection import RegisterInjector


def _parse_target_blocks(spec, n_blocks: int) -> list[int]:
    if spec is None or spec == "all":
        blocks = list(range(n_blocks))
    elif isinstance(spec, (list, tuple)):
        blocks = [int(x) for x in spec]
    else:
        text = str(spec).strip()
        if text.startswith("["):
            blocks = [int(x) for x in json.loads(text)]
        elif "-" in text and "," not in text:
            lo, hi = text.split("-", 1)
            blocks = list(range(int(lo), int(hi) + 1))
        else:
            blocks = [int(x) for x in text.split(",") if x != ""]

    if not blocks:
        raise ValueError("target_blocks must not be empty")
    invalid = [idx for idx in blocks if idx < 0 or idx >= n_blocks]
    if invalid:
        raise ValueError(
            f"target_blocks must be in [0, {n_blocks}), got {sorted(set(invalid))}"
        )
    return blocks


class _QKVSurface(nn.Module):
    """Per-block trained self-attention QKV surface."""

    def __init__(
        self,
        in_dim: int,
        inner_dim: int,
        qkv_mode: str,
        rank: int,
        *,
        down_init: str = "kaiming",
        base_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.qkv_mode = qkv_mode
        if qkv_mode == "lora":
            if down_init == "weight_svd":
                if base_weight is None:
                    raise ValueError("base_weight is required for down_init='weight_svd'")
                weight = base_weight.detach().float()
                q = min(max(4 * rank, rank + 8), min(weight.shape))
                _, _, v = torch.svd_lowrank(weight, q=q, niter=4)
                self.down = nn.Parameter(v[:, :rank].T.contiguous())
            elif down_init == "kaiming":
                self.down = nn.Parameter(torch.empty(rank, in_dim))
                nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
            else:
                raise ValueError(
                    f"down_init must be 'kaiming' or 'weight_svd', got {down_init!r}"
                )
            self.up_q = nn.Parameter(torch.zeros(inner_dim, rank))
            self.up_k = nn.Parameter(torch.zeros(inner_dim, rank))
            self.up_v = nn.Parameter(torch.zeros(inner_dim, rank))
        else:
            self.q = nn.Parameter(torch.zeros(inner_dim, in_dim))
            self.k = nn.Parameter(torch.zeros(inner_dim, in_dim))
            self.v = nn.Parameter(torch.zeros(inner_dim, in_dim))

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        if self.qkv_mode == "lora":
            delta = F.linear(x, self.down)
            dq = F.linear(delta, self.up_q)
            dk = F.linear(delta, self.up_k)
            dv = F.linear(delta, self.up_v)
            return torch.cat([dq, dk, dv], dim=-1) * scale
        return torch.cat(
            [F.linear(x, self.q), F.linear(x, self.k), F.linear(x, self.v)], dim=-1
        )


class RegisterNetwork(AdapterNetworkBase):
    network_module = "networks.methods.register"
    network_spec = "register"
    mergeable = False

    def __init__(
        self,
        unet,
        *,
        num_registers: int = 36,
        arm: str = "B",
        qkv_mode: str = "unfrozen",
        lora_rank: int = 8,
        lora_alpha: Optional[float] = None,
        target_blocks=None,
        insert_block: int = 8,
        down_init: str = "kaiming",
        init_std: float = 0.02,
        register_lr_scale: float = 100.0,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if arm not in ("A", "B"):
            raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
        if qkv_mode not in ("lora", "unfrozen"):
            raise ValueError(f"qkv_mode must be 'lora' or 'unfrozen', got {qkv_mode!r}")
        if down_init != "kaiming" and qkv_mode != "lora":
            raise ValueError(
                f"down_init={down_init!r} only applies to qkv_mode='lora' "
                f"(got qkv_mode={qkv_mode!r})"
            )

        object.__setattr__(self, "_dit", unet)
        self.multiplier = multiplier
        self.K = int(num_registers)
        self.arm = arm
        self.qkv_mode = qkv_mode
        self.D = int(unet.model_channels)
        self.register_lr_scale = float(register_lr_scale)

        if self.K > 0 and arm == "B":
            self.register = nn.Parameter(torch.randn(self.K, self.D) * init_std)
        else:
            self.register_buffer("register", torch.zeros(max(self.K, 0), self.D))

        n_blocks = len(unet.blocks)
        self.insert_block = int(insert_block)
        if not (0 <= self.insert_block < n_blocks):
            raise ValueError(
                f"insert_block must be in [0, {n_blocks}), got {self.insert_block}"
            )
        self.target_blocks = _parse_target_blocks(target_blocks, n_blocks)
        self.lora_rank = int(lora_rank)
        alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.scale = alpha / lora_rank if self.lora_rank else 1.0

        inner_dim = (
            unet.blocks[self.target_blocks[0]].self_attn.qkv_proj.out_features // 3
        )
        self.down_init = down_init
        self.qkv = nn.ModuleDict(
            {
                str(block_idx): _QKVSurface(
                    self.D,
                    inner_dim,
                    qkv_mode,
                    self.lora_rank,
                    down_init=down_init,
                    base_weight=unet.blocks[block_idx].self_attn.qkv_proj.weight,
                )
                for block_idx in self.target_blocks
            }
        )

        self.extra_seq_tokens = self.K
        self._injector = RegisterInjector(
            num_registers=self.K,
            insert_block=self.insert_block,
            get_scaled_tokens=lambda: self.register * self.multiplier,
        )
        self._applied = False
        self._orig_qkv_fwd: dict[int, object] = {}

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        qkv_params = [p for module in self.qkv.values() for p in module.parameters()]
        groups = [{"params": qkv_params, "lr": lr}]
        descriptions = ["register.qkv"]
        if self.arm == "B" and self.K > 0:
            groups.append(
                {"params": [self.register], "lr": lr * self.register_lr_scale}
            )
            descriptions.append("register.tokens")
        return groups, descriptions

    def metadata_fields(self) -> dict[str, str]:
        return {
            "ss_num_registers": str(self.K),
            "ss_arm": self.arm,
            "ss_qkv_mode": self.qkv_mode,
            "ss_lora_rank": str(self.lora_rank),
            "ss_scale": str(self.scale),
            "ss_insert_block": str(self.insert_block),
            "ss_down_init": self.down_init,
            "ss_target_blocks": json.dumps(self.target_blocks),
            "ss_model_channels": str(self.D),
            "ss_num_blocks": str(len(self._dit.blocks)),
        }

    @property
    def last_reg_ratio(self) -> Optional[float]:
        return self._injector.last_reg_ratio

    @property
    def last_patch_sink_ratio(self) -> Optional[float]:
        return self._injector.last_patch_sink_ratio

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        del text_encoders, apply_text_encoder
        if self._applied or not apply_unet:
            return

        anima = unet
        network = self
        self._injector.apply(anima)

        for block_idx in self.target_blocks:
            qkv = anima.blocks[block_idx].self_attn.qkv_proj
            self._orig_qkv_fwd[block_idx] = qkv.forward
            surface = self.qkv[str(block_idx)]

            def make_fwd(orig, surface_module):
                def fwd(x):
                    return orig(x) + surface_module(x, network.scale) * network.multiplier

                return fwd

            qkv.forward = make_fwd(qkv.forward, surface)

        self._applied = True

    def remove(self) -> None:
        if not self._applied:
            return
        anima = self._dit
        self._injector.remove()
        for block_idx, orig in self._orig_qkv_fwd.items():
            anima.blocks[block_idx].self_attn.qkv_proj.forward = orig
        self._orig_qkv_fwd = {}
        self._applied = False


def _build(unet, network_dim, network_alpha, kwargs) -> RegisterNetwork:
    return RegisterNetwork(
        unet,
        num_registers=int(kwargs.get("num_registers", 36)),
        arm=str(kwargs.get("arm", "B")),
        qkv_mode=str(kwargs.get("qkv_mode", "unfrozen")),
        lora_rank=int(network_dim) if network_dim else 8,
        lora_alpha=float(network_alpha) if network_alpha else None,
        target_blocks=kwargs.get("target_blocks", "all"),
        insert_block=int(kwargs.get("insert_block", 8)),
        down_init=str(kwargs.get("down_init", "kaiming")),
        init_std=float(kwargs.get("init_std", 0.02)),
        register_lr_scale=float(kwargs.get("register_lr_scale", 100.0)),
    )


def create_network(
    multiplier,
    network_dim,
    network_alpha,
    vae,
    text_encoders,
    unet,
    neuron_dropout=None,
    **kwargs,
):
    del vae, text_encoders, neuron_dropout
    network = _build(unet, network_dim, network_alpha, kwargs)
    network.set_multiplier(float(multiplier))
    return network


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    del ae, text_encoders, for_inference
    metadata = {}
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors import safe_open

            weights_sd = {}
            with safe_open(file, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                for key in handle.keys():
                    weights_sd[key] = handle.get_tensor(key)
        else:
            weights_sd = torch.load(file, map_location="cpu")
    elif file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}

    rank = int(metadata.get("ss_lora_rank", kwargs.get("lora_rank", 8)))
    alpha = float(metadata["ss_scale"]) * rank if "ss_scale" in metadata else None
    network = RegisterNetwork(
        unet,
        num_registers=int(
            metadata.get("ss_num_registers", kwargs.get("num_registers", 36))
        ),
        arm=str(metadata.get("ss_arm", kwargs.get("arm", "B"))),
        qkv_mode=str(
            metadata.get("ss_qkv_mode", kwargs.get("qkv_mode", "unfrozen"))
        ),
        lora_rank=rank,
        lora_alpha=alpha,
        target_blocks=json.loads(metadata["ss_target_blocks"])
        if "ss_target_blocks" in metadata
        else kwargs.get("target_blocks", "all"),
        insert_block=int(
            metadata.get("ss_insert_block", kwargs.get("insert_block", 0))
        ),
    )
    network.load_state_dict(weights_sd, strict=False)
    network.set_multiplier(float(multiplier))
    return network, weights_sd
