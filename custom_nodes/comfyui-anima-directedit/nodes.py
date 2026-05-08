"""Anima DirectEdit node.

One combined node: image in, edited image out. The tagger runs inside the
node to derive psi_src; the user's ``edit_text`` is appended to form
psi_tar. DiT / text encoder / VAE / tagger are loaded from disk per
invocation (no MODEL / CLIP / VAE wires) because the underlying
``library.inference.directedit`` primitives target the ``library/anima``
model, not ``comfy.ldm.anima``.

Pipeline mirrors ``scripts/edit.py``:

  IMAGE -> PIL                        (ComfyUI [B,H,W,C] in [0,1] -> PIL.RGB)
  AnimaTagger.predict_caption -> psi_src     (skipped if user supplies override)
  psi_tar = psi_src + ", " + edit_text
  bucket pick from source aspect ratio
  AnimaTokenize/TextEncoding strategies set
  load DiT
  load TE -> encode psi_src / psi_tar / psi_neg -> drop TE
  load VAE -> encode source -> z_clean -> VAE off-device
  invert(z_clean, psi_src) -> z_inv, delta_z
  edit_forward(z_inv[0], delta_z, psi_tar) -> z_edit
  drop DiT -> VAE on-device -> decode z_edit -> pixels
  pixels [3,H,W] in [-1,1] -> ComfyUI IMAGE [1,H,W,3] in [0,1]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import torch

# Make ``anima_lora/`` importable. This file lives at
# ``anima_lora/custom_nodes/comfyui-anima-directedit/nodes.py`` so parents[2]
# is ``anima_lora/``.
ANIMA_LORA = Path(__file__).resolve().parents[2]
if str(ANIMA_LORA) not in sys.path:
    sys.path.insert(0, str(ANIMA_LORA))

from PIL import Image  # noqa: E402

import folder_paths  # noqa: E402  ComfyUI module; only resolvable inside ComfyUI

from library.anima import strategy as strategy_anima, text_strategies  # noqa: E402
from library.captioning.anima_tagger import AnimaTagger  # noqa: E402
from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS  # noqa: E402
from library.inference import directedit, sampling as inference_utils  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.text import prepare_text_inputs  # noqa: E402
from library.models import qwen_vae as qwen_image_autoencoder_kl  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

logger = logging.getLogger(__name__)


# ComfyUI registers DiT files under either "diffusion_models" or "unet";
# text encoders under "text_encoders" or "clip"; VAEs under "vae". We look
# up against all of them so the dropdowns populate regardless of which key
# the user's install uses.
_DIT_KEYS = ("diffusion_models", "unet")
_TE_KEYS = ("text_encoders", "clip")
_VAE_KEYS = ("vae",)


def _list_files(keys: Sequence[str]) -> List[str]:
    """Union of filenames registered under any of the given folder_paths keys."""
    seen: set[str] = set()
    out: List[str] = []
    for k in keys:
        try:
            files = folder_paths.get_filename_list(k)
        except Exception:
            continue
        for f in files:
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


def _resolve_full_path(filename: str, keys: Sequence[str]) -> str:
    """Return the on-disk path for ``filename`` looked up under any of ``keys``."""
    for k in keys:
        try:
            p = folder_paths.get_full_path(k, filename)
        except Exception:
            continue
        if p:
            return p
    raise FileNotFoundError(
        f"{filename!r} not found in folder_paths under any of: {keys}"
    )


def _pick_bucket(pil_img: Image.Image) -> tuple[int, int]:
    """Closest CONSTANT_TOKEN_BUCKETS entry by aspect ratio. Returns (H, W)."""
    rw, rh = pil_img.size
    target = rw / rh
    best = min(CONSTANT_TOKEN_BUCKETS, key=lambda wh: abs(wh[0] / wh[1] - target))
    return best[1], best[0]  # bucket is (W, H); we return (H, W)


def _comfy_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE [B, H, W, C] in [0,1] -> PIL.RGB (first batch element)."""
    arr = image_tensor[0].clamp(0, 1).cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


def _pixels_to_comfy_image(pixels: torch.Tensor) -> torch.Tensor:
    """VAE-decoded [3, H, W] in [-1,1] -> ComfyUI IMAGE [1, H, W, 3] in [0,1]."""
    img = (pixels.float().clamp(-1, 1) + 1.0) / 2.0
    img = img.permute(1, 2, 0).unsqueeze(0).contiguous().cpu()
    return img


class AnimaDirectEdit:
    """Image + tag-to-add -> edited image, with the tagger bundled inside.

    Loads its own DiT / TE / VAE / tagger from disk per invocation. Cold-load
    is intentional for v0 to match ``scripts/edit.py`` semantics; caching
    across invocations is a clear v1 improvement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        dits = _list_files(_DIT_KEYS) or ["<none>"]
        tes = _list_files(_TE_KEYS) or ["<none>"]
        vaes = _list_files(_VAE_KEYS) or ["<none>"]
        return {
            "required": {
                "image": ("IMAGE",),
                "edit_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "double peace",
                        "tooltip": (
                            "Tag(s) to add to the source caption. "
                            "psi_tar = psi_src + ', ' + edit_text. Leave "
                            "empty to use psi_src as-is (sanity-check; "
                            "should reconstruct the source)."
                        ),
                    },
                ),
                "dit": (
                    dits,
                    {"tooltip": "Anima DiT checkpoint (e.g. anima-preview3-base.safetensors)."},
                ),
                "text_encoder": (
                    tes,
                    {"tooltip": "Anima text encoder (Qwen3 06B)."},
                ),
                "vae": (vaes, {"tooltip": "Qwen Image VAE."}),
                "tagger_dir": (
                    "STRING",
                    {
                        "default": "models/captioners/anima-tagger-v1",
                        "tooltip": (
                            "AnimaTagger checkpoint directory. Relative paths "
                            "are resolved against the anima_lora/ project root; "
                            "absolute paths used as-is. Must contain "
                            "model.safetensors + config.json + vocab.json + "
                            "thresholds.safetensors + rules.yaml."
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "worst quality"},
                ),
                "infer_steps": ("INT", {"default": 28, "min": 1, "max": 200}),
                "flow_shift": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "CFG scale for the edit (target) pass.",
                    },
                ),
                "invert_guidance": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": (
                            "CFG scale during inversion. Default 1.0 (no CFG); "
                            "raise only if the inverted noise must match a "
                            "high-CFG generation seed."
                        ),
                    },
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1}),
            },
            "optional": {
                "prompt_src_override": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional: replace the tagger's caption with your "
                            "own psi_src. Leave empty to use the tagger output. "
                            "Useful when the source is an Anima-generated image "
                            "and you already know the original prompt."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "prompt_src", "prompt_tar")
    FUNCTION = "edit"
    CATEGORY = "anima"
    DESCRIPTION = (
        "DirectEdit (Yang & Ye arXiv:2605.02417) editor for Anima. Runs the "
        "AnimaTagger on the source image to derive psi_src, appends edit_text "
        "to form psi_tar, then performs flow inversion + delta_z-anchored "
        "resampling. Self-contained: loads its own DiT / TE / VAE / tagger "
        "(does not reuse the canvas's MODEL / CLIP / VAE wires)."
    )

    def edit(
        self,
        image: torch.Tensor,
        edit_text: str,
        dit: str,
        text_encoder: str,
        vae: str,
        tagger_dir: str,
        negative_prompt: str,
        infer_steps: int,
        flow_shift: float,
        guidance_scale: float,
        invert_guidance: float,
        seed: int,
        prompt_src_override: str = "",
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dit_path = _resolve_full_path(dit, _DIT_KEYS)
        te_path = _resolve_full_path(text_encoder, _TE_KEYS)
        vae_path = _resolve_full_path(vae, _VAE_KEYS)

        tdir = Path(tagger_dir)
        if not tdir.is_absolute():
            tdir = ANIMA_LORA / tdir

        pil_src = _comfy_image_to_pil(image)

        if prompt_src_override.strip():
            psi_src = prompt_src_override.strip()
            logger.info("DirectEdit: using prompt_src_override: %r", psi_src)
        else:
            logger.info("DirectEdit: running AnimaTagger from %s", tdir)
            tagger = AnimaTagger(ckpt_dir=tdir, device=device)
            psi_src = tagger.predict_caption(pil_src)
            del tagger
            clean_memory_on_device(device)
            logger.info("DirectEdit: psi_src = %r", psi_src)

        edit = (edit_text or "").strip()
        psi_tar = f"{psi_src}, {edit}" if edit else psi_src
        logger.info("DirectEdit: psi_tar = %r", psi_tar)

        h_pix, w_pix = _pick_bucket(pil_src)
        pil_src_resized = pil_src.resize((w_pix, h_pix), Image.LANCZOS)
        logger.info(
            "DirectEdit: bucket %dx%d (HxW) for source aspect %.3f",
            h_pix, w_pix, pil_src.size[0] / pil_src.size[1],
        )

        # Args namespace mirroring scripts/edit.py's parsed args. Many fields
        # are passthrough plumbing read by load_dit_model / load_text_encoder
        # / prepare_text_inputs / save_images downstream.
        args = SimpleNamespace(
            dit=dit_path,
            text_encoder=te_path,
            vae=vae_path,
            attn_mode="flash",
            image=None,
            prompt_src=psi_src,
            prompt_tar=psi_tar,
            cached_embed=None,
            cached_embed_variants="all",
            negative_prompt=negative_prompt,
            mask=None,
            infer_steps=infer_steps,
            flow_shift=flow_shift,
            guidance_scale=guidance_scale,
            invert_guidance=invert_guidance,
            t_inj=0,
            image_size=[h_pix, w_pix],
            seed=seed,
            save_path=None,
            vae_chunk_size=64,
            vae_disable_cache=True,
            text_encoder_cpu=False,
            device=device,
            no_metadata=True,
            lora_weight=None,
            lora_multiplier=1.0,
            lycoris=False,
            compile_blocks=False,
            compile_inductor_mode=None,
            fp8=False,
            compile=False,
        )

        # Tokenize / encoding strategies (process-global set-once singletons in
        # library.anima.text_strategies). The CLI runs once per process so this
        # never trips, but ComfyUI keeps the process alive across invocations
        # — clear the cached strategies first so a re-run (or a TE swap between
        # runs) doesn't hit the "strategy is already set" guard.
        text_strategies.TokenizeStrategy._strategy = None
        text_strategies.TextEncodingStrategy._strategy = None
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=te_path,
            t5_tokenizer_path=None,
            qwen3_max_length=512,
            t5_max_length=512,
        )
        text_strategies.TokenizeStrategy.set_strategy(tokenize_strategy)
        text_strategies.TextEncodingStrategy.set_strategy(
            strategy_anima.AnimaTextEncodingStrategy()
        )

        logger.info("DirectEdit: loading DiT %s", dit_path)
        anima = load_dit_model(args, device, dit_weight_dtype=torch.bfloat16)

        logger.info("DirectEdit: loading TE and encoding prompts")
        args_src = SimpleNamespace(**vars(args))
        args_src.prompt = psi_src
        args_src.negative_prompt = negative_prompt
        args_tar = SimpleNamespace(**vars(args))
        args_tar.prompt = psi_tar
        args_tar.negative_prompt = negative_prompt

        text_enc = load_text_encoder(args, dtype=torch.bfloat16, device=device)
        shared = {"text_encoder": text_enc, "conds_cache": {}}
        ctx_src, ctx_neg = prepare_text_inputs(args_src, device, anima, shared)
        ctx_tar, _ = prepare_text_inputs(args_tar, device, anima, shared)

        text_enc.to("cpu")
        del text_enc, shared
        clean_memory_on_device(device)

        embed_src = ctx_src["embed"][0].to(device, dtype=torch.bfloat16)
        embed_tar = ctx_tar["embed"][0].to(device, dtype=torch.bfloat16)
        embed_neg = ctx_neg["embed"][0].to(device, dtype=torch.bfloat16)

        # VAE: encode source, then move off-device for the DiT loop.
        from torchvision import transforms

        logger.info("DirectEdit: loading VAE %s", vae_path)
        vae_obj = qwen_image_autoencoder_kl.load_vae(
            vae_path,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
        )
        vae_obj.to(torch.bfloat16).eval().to(device)

        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        # 5D [B, C, T=1, H, W]: qwen_vae preserves rank, DiT expects 5D.
        img_t = (
            tfm(pil_src_resized)
            .unsqueeze(0)
            .unsqueeze(2)
            .to(device, dtype=torch.bfloat16)
        )
        with torch.no_grad():
            z_clean = vae_obj.encode_pixels_to_latents(img_t)
        logger.info("DirectEdit: encoded source latent %s", tuple(z_clean.shape))

        vae_obj.to("cpu")
        clean_memory_on_device(device)

        timesteps, sigmas = inference_utils.get_timesteps_sigmas(
            args.infer_steps, args.flow_shift, device
        )
        sigmas = sigmas.to(device)

        logger.info(
            "DirectEdit: invert (T=%d, src_guidance=%.2f) -> edit (tar_guidance=%.2f)",
            args.infer_steps, args.invert_guidance, args.guidance_scale,
        )
        z_inv, delta_z = directedit.invert(
            anima=anima,
            z_clean=z_clean,
            embed_src=embed_src,
            embed_neg=embed_neg if args.invert_guidance != 1.0 else None,
            sigmas=sigmas,
            guidance_scale=args.invert_guidance,
        )
        z_edit = directedit.edit_forward(
            anima=anima,
            z_init=z_inv[0],
            delta_z=delta_z,
            embed_tar=embed_tar,
            embed_neg=embed_neg,
            sigmas=sigmas,
            guidance_scale=args.guidance_scale,
        )

        # Drop DiT, bring VAE back, decode.
        del anima
        clean_memory_on_device(device)

        vae_obj.to(device)
        with torch.no_grad():
            pixels = vae_obj.decode_to_pixels(z_edit.to(device, dtype=vae_obj.dtype))
        if pixels.ndim == 5:
            pixels = pixels.squeeze(2)
        pixels = pixels[0]  # [3, H, W]

        vae_obj.to("cpu")
        del vae_obj
        clean_memory_on_device(device)

        comfy_img = _pixels_to_comfy_image(pixels)
        return (comfy_img, psi_src, psi_tar)


NODE_CLASS_MAPPINGS = {
    "AnimaDirectEdit": AnimaDirectEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaDirectEdit": "Anima DirectEdit",
}
