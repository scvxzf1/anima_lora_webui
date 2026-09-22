import pytest
from safetensors.torch import save_file
import torch

from bench.adaptive_runtime.inputs import load_krea_inputs, load_z_image_inputs


def fixture(path, *, schema="adaptive_z_image_inputs_v1", nonfinite=False):
    values = {"latents": torch.ones(1, 16, 1, 4, 4),
              "noise": torch.zeros(1, 16, 1, 4, 4), "prompt": torch.ones(8, 32)}
    if nonfinite:
        values["prompt"][0, 0] = float("nan")
    save_file(values, str(path), {"schema": schema})


def test_inputs_dtype_cast_and_hash(tmp_path):
    path = tmp_path / "inputs.safetensors"
    fixture(path)
    values, digest = load_z_image_inputs(path, resolution=32, prompt_dim=32,
                                        device="cpu", dtype=torch.float16)
    assert all(t.dtype == torch.float16 for t in values.values())
    assert len(digest) == 64


@pytest.mark.parametrize("schema,nonfinite,resolution,prompt_dim", [
    ("unknown", False, 32, 32), ("adaptive_z_image_inputs_v1", True, 32, 32),
    ("adaptive_z_image_inputs_v1", False, 64, 32), ("adaptive_z_image_inputs_v1", False, 32, 64),
])
def test_bad_inputs_fail_closed(tmp_path, schema, nonfinite, resolution, prompt_dim):
    path = tmp_path / "inputs.safetensors"
    fixture(path, schema=schema, nonfinite=nonfinite)
    with pytest.raises(ValueError):
        load_z_image_inputs(path, resolution=resolution, prompt_dim=prompt_dim,
                            device="cpu", dtype=torch.float32)


@pytest.mark.parametrize("invalid", [None, "mask", "nonfinite", "family"])
def test_krea_cache_contract(tmp_path, invalid):
    path = tmp_path / "krea.safetensors"
    values = {"latents": torch.ones(1, 16, 1, 4, 4), "noise": torch.zeros(1, 16, 1, 4, 4),
              "hidden": torch.zeros(1, 512, 12, 2560), "mask": torch.ones(1, 512, dtype=torch.bool)}
    if invalid == "mask":
        values["mask"].zero_()
    if invalid == "nonfinite":
        values["hidden"][0, 0, 0, 0] = float("inf")
    schema = "adaptive_z_image_inputs_v1" if invalid == "family" else "adaptive_krea_inputs_v1"
    save_file(values, str(path), {"schema": schema})
    kwargs = {"resolution": 32, "device": "cpu", "dtype": torch.float16}
    if invalid:
        with pytest.raises(ValueError):
            load_krea_inputs(path, **kwargs)
    else:
        cached, digest = load_krea_inputs(path, **kwargs)
        assert cached["hidden"].dtype == torch.float16
        assert cached["mask"].dtype == torch.bool
        assert len(digest) == 64
