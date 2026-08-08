# Krea-2-Raw 文本链路 (阶段 1)
#
# Qwen3-VL-4B-Instruct 文本编码 + 12 层 MFA + caching。
#
# 与 anima (library/anima/strategy.py) 的关键差异 (R1 定论见
# docs/findings/krea2_raw_migration_stage0_findings.md §R1):
#   - anima 用 Qwen3 纯文本 LLM + LLM Adapter 桥到 T5 space, padding 位
#     "zero 作 cross-attn sink" (strategy.py:137 [~mask]=0 + models.py:2736)。
#   - Krea-2 用 Qwen3-VL-4B 直接, hidden_dim=2560 = DiT txtdim, 无需 adapter。
#     padding 用 attention mask 屏蔽 (DiT 内部不二次置零)。
#   - anima max-pad 到 512; Krea-2 ChatML 模板 pad 到 541 (512+34-5) + cat suffix。
#
# encoder.py 核实 (krea-ai/krea-2 commit db3984f, 见阶段1 子代理核实):
#   - Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
#   - select_layers=(2,5,8,11,14,17,20,23,26,29,32,35), stack dim=2
#   - prompt_template_encode_start_idx=34 (切掉 system prompt 前 34 token)
#   - max_length=512, padding 长度 = 512+34-5 = 541 (suffix_start_idx=5)
#   - system prompt 固定 "Describe the image...", suffix "<|im_end|>\n<|im_start|>assistant\n"
#   - MFA projector (Linear(12,1)) 和 2560→6144 投影都在 DiT 的 txtfusion/txtmlp, encoder 零可训练参数
#
# 单文件 safetensors 加载 (同 anima load_qwen3_text_encoder 模式):
# bundled config 目录 library/models/krea2_raw/configs/qwen3vl_4b/ + 单 safetensors。

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from safetensors import safe_open

from library.anima.text_strategies import (
    TextEncodingStrategy,
    TextEncoderOutputsCachingStrategy,
    TokenizeStrategy,
)
from library.io.cache import resolve_cache_path
from library.log import setup_logging
from safetensors.torch import save_file as _save_safetensors

setup_logging()
logger = logging.getLogger(__name__)


# ---- 常量 (核实自 krea-ai/krea-2 encoder.py) ----

KREA2_TE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
KREA2_TE_BUNDLED_CONFIG_DIR = str(
    Path(__file__).resolve().parent / "configs" / "qwen3vl_4b"
)

KREA2_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_NUM_TXT_LAYERS = len(KREA2_SELECT_LAYERS)  # 12
KREA2_PREFIX_IDX = 34  # prompt_template_encode_start_idx: 切掉 system prompt 前 34 token
KREA2_SUFFIX_START_IDX = 5  # prompt_template_encode_suffix_start_idx (padding 公式用)
KREA2_MAX_LENGTH = 512  # user prompt 正文 max_length
# padding 长度 = max_length + prefix_idx - suffix_start_idx = 512+34-5 = 541
KREA2_PAD_LENGTH = KREA2_MAX_LENGTH + KREA2_PREFIX_IDX - KREA2_SUFFIX_START_IDX

KREA2_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n"
)
KREA2_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

KREA2_TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_krea2_te.safetensors"
KREA2_CACHE_DTYPE = torch.bfloat16


class Krea2TokenizeStrategy(TokenizeStrategy):
    """Qwen3-VL ChatML tokenize (prompt pad 到 541 + cat suffix, 不裁剪)."""

    def __init__(
        self,
        max_length: int = KREA2_MAX_LENGTH,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_length = max_length + KREA2_PREFIX_IDX - KREA2_SUFFIX_START_IDX
        config_dir = tokenizer_path or KREA2_TE_BUNDLED_CONFIG_DIR
        from transformers import AutoTokenizer

        logger.info(f"load Krea-2 tokenizer from {config_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config_dir, local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        if isinstance(text, str):
            text = [text]
        # ChatML: system prefix + user prompt, 一起 tokenize + pad 到 pad_length
        full = [KREA2_PROMPT_PREFIX + t for t in text]
        inputs = self.tokenizer(
            full,
            truncation=True,
            padding="max_length",
            max_length=self.pad_length,
            return_tensors="pt",
        )
        # suffix 单独 tokenize (不 pad), cat 到右侧
        suffix_inputs = self.tokenizer(
            [KREA2_PROMPT_SUFFIX] * len(text),
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = torch.cat(
            [inputs["input_ids"], suffix_inputs["input_ids"]], dim=1
        )
        attn_mask = torch.cat(
            [inputs["attention_mask"], suffix_inputs["attention_mask"]], dim=1
        )
        return [input_ids, attn_mask]

    def tokenize_with_weights(
        self, text: Union[str, List[str]]
    ) -> tuple:
        # Krea-2 首日不支持 weighted prompt (无 T5 weighted tokenize). 直走 tokenize.
        tokens = self.tokenize(text)
        weights = [torch.ones_like(tokens[0], dtype=torch.float32)]
        return tokens, weights


class Krea2TextEncodingStrategy(TextEncodingStrategy):
    """Qwen3-VL forward + 12 层 stack dim=2 + 切 prefix, 返回 (hiddens, mask)."""

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        qwen = models[0]  # Qwen3VLForConditionalGeneration, eval + frozen
        input_ids, attn_mask = tokens[0], tokens[1]
        device = next(qwen.parameters()).device
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            states = qwen(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        # hidden_states tuple 长 37 (embedding + 36 层); 选 12 层 stack dim=2
        hiddens = torch.stack(
            [states.hidden_states[i] for i in KREA2_SELECT_LAYERS], dim=2
        )
        # 切掉 system prompt 前 34 token
        hiddens = hiddens[:, KREA2_PREFIX_IDX:]
        mask = attn_mask[:, KREA2_PREFIX_IDX:].bool()
        # 注: 不二次置零 padding (R1 定论: Krea-2 用 mask 屏蔽, 非 anima zero-sink)
        return [hiddens, mask]


class Krea2TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """{stem}_krea2_te.safetensors caching (suffix 隔离, 不污染 _anima_te)."""

    def __init__(
        self,
        cache_to_disk: bool = True,
        batch_size: Optional[int] = None,
        skip_disk_cache_validity_check: bool = False,
    ) -> None:
        super().__init__(
            cache_to_disk=cache_to_disk,
            batch_size=batch_size,
            skip_disk_cache_validity_check=skip_disk_cache_validity_check,
        )

    def get_outputs_npz_path(
        self,
        image_abs_path: str,
        cache_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
    ) -> str:
        return resolve_cache_path(
            image_abs_path,
            KREA2_TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
            cache_dir=cache_dir,
            image_dir=image_dir,
        )

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        # Lazy per-tensor read via safe_open (same pattern as anima's
        # AnimaTextEncoderOutputsCachingStrategy.load_outputs_npz). Returns
        # torch tensors in the [hiddens, mask, caption_dropout_rate] order
        # that family.compute_noise_pred_and_target unpacks
        # (conds[0]=hiddens, conds[1]=mask; the trailing rate mirrors anima's
        # cache layout so split_cached_text_encoder_outputs can surface it as
        # batch["caption_dropout_rates"] instead of eating the mask).
        # bfloat16 hiddens have no numpy representation; the anima base class
        # signature is nominal — the real contract carries torch tensors.
        with safe_open(npz_path, framework="pt") as f:
            hiddens = f.get_tensor("hiddens")
            mask = f.get_tensor("mask")
            caption_dropout_rate = f.get_tensor("caption_dropout_rate")
        return [hiddens, mask, caption_dropout_rate]

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not os.path.exists(npz_path):
            return False
        try:
            with safe_open(npz_path, framework="pt") as f:
                keys = set(f.keys())
            # 最低要求: hiddens + mask
            return "hiddens" in keys and "mask" in keys
        except Exception:
            return False

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        batch: List,
    ) -> None:
        """缓存 Krea-2 文本编码 (hiddens + mask) 到 safetensors.

        与 anima 的 _cache_batch_outputs_single 同构: 批量 encode 一组 caption,
        拆回 per-sample 写盘 (或塞 info.text_encoder_outputs 供 in-memory 使用).
        不二次置零 padding (R1 契约: mask 屏蔽即可).
        """
        captions = [info.caption for info in batch]
        # Krea2TextEncodingStrategy.encode_tokens 返回 [hiddens (B,L,12,2560),
        # mask (B,L) bool]. 一次 encode 整批, CPU 上 no_grad.
        encoded = text_encoding_strategy.encode_tokens(
            tokenize_strategy, models, tokenize_strategy.tokenize(captions)
        )
        hiddens, mask = encoded[0], encoded[1]
        for i, info in enumerate(batch):
            h_i = hiddens[i].clone()
            m_i = mask[i].clone()
            caption_dropout_rate = torch.tensor(
                info.caption_dropout_rate, dtype=torch.float32
            )
            if self.cache_to_disk:
                save_dict = {
                    "hiddens": h_i.contiguous(),
                    "mask": m_i.contiguous(),
                    "caption_dropout_rate": caption_dropout_rate,
                }
                _save_safetensors(save_dict, info.text_encoder_outputs_npz)
            else:
                info.text_encoder_outputs = (h_i, m_i, caption_dropout_rate)


def load_krea2_text_encoder(
    te_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> tuple:
    """加载 Qwen3-VL-4B (单 safetensors + bundled config).

    返回 (model, tokenizer). model 是 Qwen3VLForConditionalGeneration,
    eval + frozen + no cache. visual 权重在但 text path 不激活.
    """
    import transformers

    te_path = str(te_path)
    logger.info(f"Loading Krea-2 Qwen3-VL text encoder from {te_path}")

    config = transformers.Qwen3VLConfig.from_pretrained(
        KREA2_TE_BUNDLED_CONFIG_DIR, local_files_only=True
    )
    model = transformers.Qwen3VLForConditionalGeneration(config)

    state_dict = load_safetensors(te_path, device="cpu")
    # 单文件权重 key 形如 "model.language_model.*", "model.visual.*";
    # Qwen3VLForConditionalGeneration 的 state_dict key 形如 "model.language_model.*"
    # ( transformers 会给顶层 model. 前缀 ). 去掉可能的 "model." 重复或保留.
    new_sd = {}
    for k, v in state_dict.items():
        # transformers Qwen3VLForConditionalGeneration 期望 "model." 前缀
        if k.startswith("model."):
            new_sd[k] = v
        else:
            new_sd[f"model.{k}"] = v
    info = model.load_state_dict(new_sd, strict=False)
    n_missing = len(info.missing_keys)
    n_unexpected = len(info.unexpected_keys)
    # visual 权重在, 但若 config 没 visual 字段会 missing; 阶段 1 只用 LM 部分,
    # 视觉 missing/unexpected 在容忍范围. 这里只 log 不硬断.
    logger.info(
        f"Qwen3-VL state dict: missing={n_missing}, unexpected={n_unexpected}"
    )

    model.config.use_cache = False
    model = model.to(device, dtype=dtype).eval().requires_grad_(False)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        KREA2_TE_BUNDLED_CONFIG_DIR, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        f"Loaded Krea-2 Qwen3-VL. Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    return model, tokenizer


def load_safetensors(path: str, device: str = "cpu") -> dict:
    """单文件 safetensors 加载 (CPU, 不占 GPU)."""
    from safetensors.torch import load_file

    return load_file(path, device=device)
