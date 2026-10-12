"""Header census boundaries: no defaults, tensors, or GPU are needed."""
import json

import pytest

from web.services.model_block_inventory import inspect_model_blocks


def checkpoint(path, keys):
    header = json.dumps({key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
                         for key in keys}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
    return str(path)


@pytest.mark.parametrize("family,prefix", [("anima", "net.blocks"), ("krea2_raw", "blocks"),
    ("z_image", "layers"), ("qwen_image_2_1", "transformer_blocks")])
def test_counts_actual_header_and_preserves_two_resident(tmp_path, family, prefix):
    path = checkpoint(tmp_path / "model.safetensors", [f"{prefix}.{i}.weight" for i in range(7)])
    assert inspect_model_blocks(family, path)["block_count"] == 7
    assert inspect_model_blocks(family, path)["max_swap"] == 5


def test_replacement_invalidates_hot_cache(tmp_path):
    file = tmp_path / "model.safetensors"
    checkpoint(file, ["blocks.0.weight", "blocks.1.weight"])
    assert inspect_model_blocks("anima", str(file))["block_count"] == 2
    checkpoint(file, [f"blocks.{i}.weight" for i in range(4)])
    assert inspect_model_blocks("anima", str(file))["block_count"] == 4


@pytest.mark.parametrize("keys", [["blocks.1.weight"], ["blocks.0.weight", "blocks.2.weight"], ["layers.0.weight"]])
def test_missing_or_gapped_blocks_fail_closed(tmp_path, keys):
    path = checkpoint(tmp_path / "model.safetensors", keys)
    with pytest.raises(ValueError):
        inspect_model_blocks("anima", path)


def test_reads_actual_shard_headers_not_index_claims(tmp_path):
    directory = tmp_path / "transformer"
    directory.mkdir()
    checkpoint(directory / "a.safetensors", ["layers.0.weight"])
    checkpoint(directory / "b.safetensors", ["layers.1.weight", "layers.2.weight"])
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"layers.99.weight": "a.safetensors", "layers.2.weight": "b.safetensors"}}))
    assert inspect_model_blocks("z_image", str(tmp_path))["block_count"] == 3
    index.write_text(json.dumps({"weight_map": {"layers.0.weight": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="路径"):
        inspect_model_blocks("z_image", str(tmp_path))


def test_unknown_family_and_traversal_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        inspect_model_blocks("unknown", "model.safetensors")
    with pytest.raises(ValueError, match="不允许"):
        inspect_model_blocks("anima", str(tmp_path / ".." / "model.safetensors"))
