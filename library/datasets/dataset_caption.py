"""Caption processing and tokenizer helpers for BaseDataset.

Extracted so the core dataset class stays focused on construction, registration,
and orchestration of mixins.
"""

from __future__ import annotations

import math
import random
import re

import torch

from library.datasets.subsets import BaseSubset


class DatasetCaptionMixin:
    """Mixin providing caption dropout/wildcard processing and input-id packing."""

    def process_caption(self, subset: BaseSubset, caption):
        if subset.caption_prefix:
            caption = subset.caption_prefix + " " + caption
        if subset.caption_suffix:
            caption = caption + " " + subset.caption_suffix

        is_drop_out = (
            subset.caption_dropout_rate > 0
            and random.random() < subset.caption_dropout_rate
        )
        is_drop_out = (
            is_drop_out
            or subset.caption_dropout_every_n_epochs > 0
            and self.current_epoch % subset.caption_dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            # process wildcards
            if subset.enable_wildcard:
                # if caption is multiline, random choice one line
                if "\n" in caption:
                    caption = random.choice(caption.split("\n"))

                # wildcard is like '{aaa|bbb|ccc...}'
                # escape the curly braces like {{ or }}
                replacer1 = "⦅"
                replacer2 = "⦆"
                while replacer1 in caption or replacer2 in caption:
                    replacer1 += "⦅"
                    replacer2 += "⦆"

                caption = caption.replace("{{", replacer1).replace("}}", replacer2)

                # replace the wildcard
                def replace_wildcard(match):
                    return random.choice(match.group(1).split("|"))

                caption = re.sub(r"\{([^}]+)\}", replace_wildcard, caption)

                # unescape the curly braces
                caption = caption.replace(replacer1, "{").replace(replacer2, "}")
            else:
                # if caption is multiline, use the first line
                caption = caption.split("\n")[0]

            if subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0:
                fixed_tokens = []
                flex_tokens = []
                fixed_suffix_tokens = []
                if (
                    hasattr(subset, "keep_tokens_separator")
                    and subset.keep_tokens_separator
                    and subset.keep_tokens_separator in caption
                ):
                    fixed_part, flex_part = caption.split(
                        subset.keep_tokens_separator, 1
                    )
                    if subset.keep_tokens_separator in flex_part:
                        flex_part, fixed_suffix_part = flex_part.split(
                            subset.keep_tokens_separator, 1
                        )
                        fixed_suffix_tokens = [
                            t.strip()
                            for t in fixed_suffix_part.split(subset.caption_separator)
                            if t.strip()
                        ]

                    fixed_tokens = [
                        t.strip()
                        for t in fixed_part.split(subset.caption_separator)
                        if t.strip()
                    ]
                    flex_tokens = [
                        t.strip()
                        for t in flex_part.split(subset.caption_separator)
                        if t.strip()
                    ]
                else:
                    tokens = [
                        t.strip()
                        for t in caption.strip().split(subset.caption_separator)
                    ]
                    flex_tokens = tokens[:]
                    if subset.keep_tokens > 0:
                        fixed_tokens = flex_tokens[: subset.keep_tokens]
                        flex_tokens = tokens[subset.keep_tokens :]

                if subset.token_warmup_step < 1:
                    subset.token_warmup_step = math.floor(
                        subset.token_warmup_step * self.max_train_steps
                    )
                if (
                    subset.token_warmup_step
                    and self.current_step < subset.token_warmup_step
                ):
                    tokens_len = (
                        math.floor(
                            (self.current_step)
                            * (
                                (len(flex_tokens) - subset.token_warmup_min)
                                / (subset.token_warmup_step)
                            )
                        )
                        + subset.token_warmup_min
                    )
                    flex_tokens = flex_tokens[:tokens_len]

                def dropout_tags(tokens):
                    if subset.caption_tag_dropout_rate <= 0:
                        return tokens
                    filtered = []
                    for token in tokens:
                        if random.random() >= subset.caption_tag_dropout_rate:
                            filtered.append(token)
                    return filtered

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens + fixed_suffix_tokens)

            # process secondary separator
            if subset.secondary_separator:
                caption = caption.replace(
                    subset.secondary_separator, subset.caption_separator
                )

            for str_from, str_to in self.replacements.items():
                if str_from == "":
                    # replace all
                    if isinstance(str_to, list):
                        caption = random.choice(str_to)
                    else:
                        caption = str_to
                else:
                    caption = caption.replace(str_from, str_to)

        return caption

    def get_input_ids(self, caption, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizers[0]

        input_ids = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        ).input_ids

        if self.tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                for i in range(
                    1,
                    self.tokenizer_max_length - tokenizer.model_max_length + 2,
                    tokenizer.model_max_length - 2,
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                for i in range(
                    1,
                    self.tokenizer_max_length - tokenizer.model_max_length + 2,
                    tokenizer.model_max_length - 2,
                ):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    if (
                        ids_chunk[-2] != tokenizer.eos_token_id
                        and ids_chunk[-2] != tokenizer.pad_token_id
                    ):
                        ids_chunk[-1] = tokenizer.eos_token_id
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids
