"""Turbo Anima — DP-DMD distillation harness.

Owns two plain ``LoRANetwork`` instances (student + fake) on one frozen Anima
DiT. Both call ``apply_to(unet)`` which chains them onto every targeted
Linear's forward — at runtime the chain order is::

    linear(x) -> fake.forward -> student.forward -> original_linear.forward

Each LoRA module short-circuits at ``not self.enabled`` (see
``lora_modules/lora.py::LoRAModule.forward``), so view-toggling is just
``set_enabled(bool)`` on each network — O(num_modules) Python loop, negligible
vs a DiT forward.

Used by ``scripts/distill_turbo/distill.py``. Plain turbo students load through
the normal LoRA path; per-step-expert students are kept live so inference can
select the active up-head by denoise step. CFG is baked in during distillation.

This harness is method-agnostic (two view-toggled LoRA stacks); the shipped
objective driving it is DP-DMD.

Docs: ``docs/structure/dpdmd.md`` (structure), ``docs/experimental/dpdmd.md`` (ops).
Paper: Wu, Li, Zhang, Ma, "Diversity-Preserved Distribution Matching
Distillation" (arXiv:2602.03139).
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

from networks.lora_anima.factory import create_network
from networks.lora_anima.network import LoRANetwork

logger = logging.getLogger(__name__)

View = Literal["teacher", "student", "fake"]


def load_step_expert_student(
    unet,
    weights_sd: dict[str, torch.Tensor],
    metadata: dict[str, str],
    *,
    multiplier: float = 1.0,
) -> LoRANetwork:
    """Rebuild a per-step-expert turbo student as a router-free kept-live net.

    Per-step-expert checkpoints cannot be merged into one static DiT weight.
    They keep K up-heads per adapted Linear and select the active head by the
    denoise step counter, so inference must attach the network dynamically.
    """
    K = int(metadata.get("ss_turbo_step_expert_K", "0") or "0")
    if K <= 1:
        raise RuntimeError(
            "load_step_expert_student called on a checkpoint without "
            "ss_turbo_step_expert_K > 1."
        )
    rank = int(metadata.get("ss_turbo_student_rank", "0") or "0")
    if rank <= 0:
        raise RuntimeError(
            "per-step-expert turbo checkpoint missing ss_turbo_student_rank "
            "metadata."
        )
    alpha = float(metadata.get("ss_turbo_student_alpha", str(rank)) or rank)

    network = create_network(
        multiplier=multiplier,
        network_dim=rank,
        network_alpha=alpha,
        vae=None,
        text_encoders=[],
        unet=unet,
        step_expert_K=K,
    )
    network.apply_to([], unet, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights_sd, strict=False)
    if info.unexpected_keys:
        logger.warning(
            f"step-expert turbo: unexpected keys in state dict: "
            f"{info.unexpected_keys[:5]}..."
        )
    if info.missing_keys:
        logger.warning(
            f"step-expert turbo: {len(info.missing_keys)} missing keys "
            f"(first: {info.missing_keys[:5]})"
        )
    network.set_step_index(0)
    logger.info(
        f"step-expert turbo: router-free kept-live attached "
        f"({len(network.unet_loras)} modules, K={K} heads, rank={rank})"
    )
    return network


class TurboDMDNetwork:
    """Two LoRA stacks on one frozen DiT, view-toggleable per forward.

    Not a ``nn.Module`` — it's a thin coordinator that holds two real
    ``LoRANetwork`` instances. The DiT itself is owned by the caller and
    stays frozen.
    """

    def __init__(
        self,
        unet,
        *,
        student_rank: int,
        fake_rank: int,
        student_alpha: float | None = None,
        fake_alpha: float | None = None,
        use_custom_down_autograd: bool = False,
        student_step_expert_K: int = 0,
    ) -> None:
        self.unet = unet
        self.student_rank = int(student_rank)
        self.fake_rank = int(fake_rank)
        self.student_step_expert_K = int(student_step_expert_K)

        # Plain LoRA on both — defaults from LoRANetworkCfg give us
        # use_moe_style=False / route_per_layer=False / router_source="none" /
        # use_ortho=False / use_timestep_mask=False / add_reft=False. No MoE,
        # no ortho, no T-LoRA, no ReFT — keep slice 1 KISS.
        # alpha = rank by default (scale = alpha/rank = 1.0) — matches the
        # project's LoRA-family convention. Halving alpha would silently halve
        # every student contribution per forward, making the 28→4 step trajectory
        # remap harder to bake without buying any stability we don't already
        # get from α-warmup + grad-clip + LR.
        # ``use_custom_down_autograd`` is forwarded as a ``**kwargs`` key because
        # ``create_network``'s positional surface doesn't include it — the factory
        # reads it out of ``kwargs`` and flips each module's flag post-construction.
        _student_kwargs: dict = {}
        if self.student_step_expert_K > 1:
            _student_kwargs["step_expert_K"] = self.student_step_expert_K
        self.student: LoRANetwork = create_network(
            multiplier=1.0,
            network_dim=self.student_rank,
            network_alpha=student_alpha if student_alpha is not None else self.student_rank,
            vae=None,
            text_encoders=[],
            unet=unet,
            use_custom_down_autograd=use_custom_down_autograd,
            **_student_kwargs,
        )
        self.fake: LoRANetwork = create_network(
            multiplier=1.0,
            network_dim=self.fake_rank,
            network_alpha=fake_alpha if fake_alpha is not None else self.fake_rank,
            vae=None,
            text_encoders=[],
            unet=unet,
            use_custom_down_autograd=use_custom_down_autograd,
        )

        # Apply order matters for the forward chain. We pick student-first so
        # the runtime chain is ``linear -> fake -> student -> original``. Both
        # are functionally symmetric (additive contributions) but having a
        # stable order makes debugging easier.
        self.student.apply_to(
            text_encoders=[],
            unet=unet,
            apply_text_encoder=False,
            apply_unet=True,
        )
        self.fake.apply_to(
            text_encoders=[],
            unet=unet,
            apply_text_encoder=False,
            apply_unet=True,
        )

        logger.info(
            f"TurboDMDNetwork: student rank={self.student_rank} "
            f"({len(self.student.unet_loras)} modules), "
            f"fake rank={self.fake_rank} "
            f"({len(self.fake.unet_loras)} modules)"
        )

        # Start in teacher view — both off, base DiT is exactly itself.
        # LoRA modules default enabled=True, so disable explicitly before
        # recording the current view.
        self.student.set_enabled(False)
        self.fake.set_enabled(False)
        self._view: View = "teacher"

    # ----------------- view toggle -----------------

    # Per-view (student_on, fake_on) target states. Lookup avoids the
    # if/elif ladder and makes the "flip only what changed" diff explicit.
    _VIEW_FLAGS: dict[str, tuple[bool, bool]] = {
        "teacher": (False, False),
        "student": (True, False),
        "fake": (False, True),
    }

    def set_view(self, view: View) -> None:
        """Flip per-network enabled flags so the next DiT forward acts as
        the named view.

        - ``teacher``: both LoRA stacks off, DiT delivers base velocity.
        - ``student``: student on, fake off — produces v_student for x_pred.
        - ``fake``: fake on, student off — fake's score estimate at τ_DM.

        Short-circuits when already in the target view (consecutive teacher
        forwards in the CA + DM branches don't repay the ~O(num_modules)
        attribute-write loop, and dynamo doesn't get a chance to invalidate
        guards it would have re-validated anyway).
        """
        if view == self._view:
            return
        try:
            want_student, want_fake = self._VIEW_FLAGS[view]
        except KeyError as e:
            raise ValueError(
                f"Unknown view {view!r}; expected teacher/student/fake"
            ) from e
        cur_student, cur_fake = self._VIEW_FLAGS[self._view]
        if want_student != cur_student:
            self.student.set_enabled(want_student)
        if want_fake != cur_fake:
            self.fake.set_enabled(want_fake)
        self._view = view

    @property
    def view(self) -> View:
        return self._view

    # ----------------- per-step expert head selection -----------------

    def set_student_step(self, i: int) -> None:
        """Select the student's step-``i`` up-head when per-step expert is on."""
        if self.student_step_expert_K > 1:
            self.student.set_step_index(i)

    # ----------------- param accessors -----------------

    def student_params(self):
        """Trainable params for the student optimizer."""
        return [p for p in self.student.parameters() if p.requires_grad]

    def fake_params(self):
        """Trainable params for the fake optimizer."""
        return [p for p in self.fake.parameters() if p.requires_grad]

    def freeze_dit(self) -> None:
        """Set ``requires_grad=False`` on every base DiT param.

        Must be called AFTER both ``apply_to``'s — the LoRA networks add
        sub-modules to ``unet`` via ``add_module(lora.lora_name, lora)``, so
        a wholesale ``unet.requires_grad_(False)`` BEFORE apply would still
        be undone by the LoRA modules' own requires_grad=True params (good),
        but a wholesale call AFTER would zero those too (bad). We selectively
        walk only ``unet`` params whose name doesn't start with a LoRA prefix.
        """
        lora_prefixes = tuple(
            set(m.lora_name for m in self.student.unet_loras)
            | set(m.lora_name for m in self.fake.unet_loras)
        )
        n_frozen = 0
        for name, param in self.unet.named_parameters():
            if name.startswith(lora_prefixes):
                continue
            param.requires_grad_(False)
            n_frozen += 1
        logger.info(f"freeze_dit: {n_frozen} base params frozen")

    # ----------------- save / load -----------------

    def save_student(
        self,
        file: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Serialize only the student LoRA in the standard plain-LoRA layout.

        Output is loadable by ``inference.py --lora_weight <file>`` — the
        fake network is training scaffolding and never shipped.
        """
        # Pull exactly the student's params, prefixed by LoRA-net key style
        # (this is what LoRANetwork.state_dict() returns — naturally so
        # because each LoRA was add_module'd onto the network).
        sd = self.student.state_dict()
        # Strip any non-LoRA keys defensively (router params, etc. — plain
        # LoRA shouldn't have any, but the LoRANetwork instance itself may
        # carry buffers that aren't load-bearing for inference).
        sd = {k: v for k, v in sd.items() if ".lora_" in k or ".alpha" in k}

        if self.student_step_expert_K > 1:
            self._save_student_step_expert(sd, file, dtype, metadata)
            return

        from networks.lora_save import save_network_weights

        save_network_weights(
            sd,
            file=file,
            dtype=dtype,
            metadata=metadata,
            save_variant="standard",
        )
        logger.info(f"saved student LoRA → {file}  ({len(sd)} keys)")

    def _save_student_step_expert(
        self,
        sd: dict[str, torch.Tensor],
        file: str,
        dtype: torch.dtype,
        metadata: dict[str, str] | None,
    ) -> None:
        """Write the per-step-expert student in its kept-live layout."""
        from safetensors.torch import save_file

        from library.training.hashing import precalculate_safetensors_hashes
        from networks.lora_modules.lora import bake_inv_scale

        # Fold any per-channel scaling into lora_down (no-op when absent) so the
        # on-disk delta acts on raw inputs.
        bake_inv_scale(sd)

        if dtype is not None:
            sd = {k: v.detach().clone().to("cpu").to(dtype) for k, v in sd.items()}

        meta = dict(metadata or {})
        model_hash, legacy_hash = precalculate_safetensors_hashes(sd, meta)
        meta["sshs_model_hash"] = model_hash
        meta["sshs_legacy_hash"] = legacy_hash

        save_file(sd, file, meta)
        logger.info(
            f"saved step-expert student LoRA → {file}  ({len(sd)} keys, "
            f"K={self.student_step_expert_K} up-heads/Linear; kept-live only)"
        )
