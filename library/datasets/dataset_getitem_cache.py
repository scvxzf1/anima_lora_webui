"""Caching-mode batch helpers for BaseDataset.__getitem__."""

from __future__ import annotations

from library.datasets.image_utils import load_image


def get_item_for_caching(dataset, bucket, bucket_batch_size, image_index):
    captions = []
    images = []
    input_ids1_list = []
    input_ids2_list = []
    absolute_paths = []
    resized_sizes = []
    bucket_reso = None
    flip_aug = None
    alpha_mask = None
    random_crop = None

    for image_key in bucket[image_index : image_index + bucket_batch_size]:
        image_info = dataset.image_data[image_key]
        subset = dataset.image_to_subset[image_key]

        if flip_aug is None:
            flip_aug = subset.flip_aug
            alpha_mask = subset.alpha_mask
            random_crop = subset.random_crop
            bucket_reso = image_info.bucket_reso
        else:
            assert flip_aug == subset.flip_aug, "flip_aug must be same in a batch"
            assert alpha_mask == subset.alpha_mask, (
                "alpha_mask must be same in a batch"
            )
            assert random_crop == subset.random_crop, (
                "random_crop must be same in a batch"
            )
            assert bucket_reso == image_info.bucket_reso, (
                "bucket_reso must be same in a batch"
            )

        caption = image_info.caption

        if dataset.caching_mode == "latents":
            image = load_image(image_info.absolute_path)
        else:
            image = None

        if dataset.caching_mode == "text":
            input_ids1 = dataset.get_input_ids(caption, dataset.tokenizers[0])
            input_ids2 = dataset.get_input_ids(caption, dataset.tokenizers[1])
        else:
            input_ids1 = None
            input_ids2 = None

        captions.append(caption)
        images.append(image)
        input_ids1_list.append(input_ids1)
        input_ids2_list.append(input_ids2)
        absolute_paths.append(image_info.absolute_path)
        resized_sizes.append(image_info.resized_size)

    example = {}

    if images[0] is None:
        images = None
    example["images"] = images

    example["captions"] = captions
    example["input_ids1_list"] = input_ids1_list
    example["input_ids2_list"] = input_ids2_list
    example["absolute_paths"] = absolute_paths
    example["resized_sizes"] = resized_sizes
    example["flip_aug"] = flip_aug
    example["alpha_mask"] = alpha_mask
    example["random_crop"] = random_crop
    example["bucket_reso"] = bucket_reso
    return example
