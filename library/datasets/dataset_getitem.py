"""Batch assembly helpers for BaseDataset.__getitem__.

Keeps the long sample-loading / stacking path out of the core dataset class so
BaseDataset can stay focused on construction, captions, and registration.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional

import torch

from library.datasets.dataset_getitem_cache import get_item_for_caching as _get_item_for_caching_impl
from library.datasets.dataset_getitem_sample import (
    load_feature_sidecars,
    load_text_fields,
    load_visual_sample,
    resolve_ip_pair,
)
from library.datasets.dataset_getitem_stack import assemble_training_example


class DatasetGetItemMixin:
    """Mixin implementing training-time ``__getitem__`` assembly."""

    def __getitem__(self, index):
        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size

        if (
            self.caching_mode is not None
        ):  # return batch for latents/text encoder outputs caching
            return self.get_item_for_caching(bucket, bucket_batch_size, image_index)

        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        cond_latents_list: List[Optional[torch.Tensor]] = []
        alpha_mask_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []
        text_encoder_outputs_list = []
        custom_attributes = []
        inversion_runs_list: List[Optional[torch.Tensor]] = []
        ip_features_list: List[Optional[torch.Tensor]] = []
        ip_features_shuffled_list: List[Optional[torch.Tensor]] = []
        # Soft-tokens contrastive negatives: per-image (k, S, D) stack of cached
        # negative text embeddings, or None when no sampler is attached.
        neg_crossattn_list: List[Optional[torch.Tensor]] = []
        # Per-image (k,) tag-overlap weights for jaccard mode; None otherwise.
        neg_jaccard_list: List[Optional[torch.Tensor]] = []
        # BYG per-image edit-tuple dicts (role embeddings + masks), or None.
        byg_tuple_list: List[Optional[dict]] = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            custom_attributes.append(subset.custom_attributes)
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)
            flipped = subset.flip_aug and random.random() < 0.5

            visual = load_visual_sample(self, image_info, subset, flipped)
            images.append(visual["image"])
            latents_list.append(visual["latents"])
            cond_latents_list.append(visual["cond_latents"])
            alpha_mask_list.append(visual["alpha_mask"])
            original_sizes_hw.append(visual["original_size_hw"])
            crop_top_lefts.append(visual["crop_top_left"])
            target_sizes_hw.append(visual["target_size_hw"])
            flippeds.append(visual["flipped"])

            target_stem = os.path.splitext(os.path.basename(image_info.absolute_path))[0]
            ip_pair = resolve_ip_pair(self, image_info, subset, target_stem)
            text_fields = load_text_fields(
                self,
                image_info,
                subset,
                sampler=ip_pair["sampler"],
                target_stem=target_stem,
                strip_identity=ip_pair["strip_identity"],
            )
            text_encoder_outputs_list.append(text_fields["text_encoder_outputs"])
            input_ids_list.append(text_fields["input_ids"])
            captions.append(text_fields["caption"])

            sidecars = load_feature_sidecars(
                self,
                image_info,
                subset,
                target_stem=target_stem,
                sampler=ip_pair["sampler"],
                ip_ref_stem=ip_pair["ip_ref_stem"],
                ip_ref_subset=ip_pair["ip_ref_subset"],
                ip_ref_reldir=ip_pair["ip_ref_reldir"],
                ip_shuffled_stem=ip_pair["ip_shuffled_stem"],
            )
            byg_tuple_list.append(sidecars["byg_tuple"])
            inversion_runs_list.append(sidecars["inversion_runs"])
            ip_features_list.append(sidecars["ip_features"])
            ip_features_shuffled_list.append(sidecars["ip_features_shuffled"])
            neg_crossattn_list.append(sidecars["neg_crossattn"])
            neg_jaccard_list.append(sidecars["neg_jaccard"])

        return assemble_training_example(
            dataset=self,
            custom_attributes=custom_attributes,
            loss_weights=loss_weights,
            text_encoder_outputs_list=text_encoder_outputs_list,
            input_ids_list=input_ids_list,
            alpha_mask_list=alpha_mask_list,
            images=images,
            latents_list=latents_list,
            cond_latents_list=cond_latents_list,
            captions=captions,
            original_sizes_hw=original_sizes_hw,
            crop_top_lefts=crop_top_lefts,
            target_sizes_hw=target_sizes_hw,
            flippeds=flippeds,
            inversion_runs_list=inversion_runs_list,
            ip_features_list=ip_features_list,
            ip_features_shuffled_list=ip_features_shuffled_list,
            neg_crossattn_list=neg_crossattn_list,
            neg_jaccard_list=neg_jaccard_list,
            byg_tuple_list=byg_tuple_list,
            bucket=bucket,
            image_index=image_index,
        )

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        return _get_item_for_caching_impl(self, bucket, bucket_batch_size, image_index)
