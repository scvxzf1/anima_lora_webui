import pytest
import torch
from bench.convrot_90hx.numerics import compare_named, vector_metrics, tensor_hash


def test_vectors_detect_rotation_not_just_norm():
    result = vector_metrics(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert result["reference_norm"] == result["candidate_norm"] == 1
    assert result["cosine"] == 0
    assert result["relative_l2"] == pytest.approx(2**0.5)


def test_zero_is_explicit_and_undefined_metrics_are_null():
    result = vector_metrics(torch.zeros(3), torch.ones(3))
    assert result["reference_zero"] and not result["candidate_zero"]
    assert result["cosine"] is None and result["relative_l2"] is None
    assert result["max_abs"] == 1


def test_named_missing_gradient_is_retained_in_complete_vector():
    result = compare_named(
        {"a": torch.ones(2), "b": torch.ones(2)}, {"a": torch.ones(2)}
    )
    assert result["missing_candidate"] == ["b"]
    assert result["overall"]["relative_l2"] == pytest.approx(2**-0.5)
    assert result["layers"]["b"]["candidate_zero"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_is_reported_without_nan_json(bad):
    result = vector_metrics(torch.ones(1), torch.tensor([bad]))
    assert not result["finite"] and result["relative_l2"] is None


def test_shape_mismatch_fails_and_hash_preserves_bf16():
    with pytest.raises(ValueError):
        vector_metrics(torch.ones(2), torch.ones(3))
    assert tensor_hash(torch.ones(2, dtype=torch.bfloat16)) != tensor_hash(
        torch.ones(2)
    )
