"""Real Inductor, all-linear NF4, production LoRA and swapped Qwen Edit."""
import pytest
import torch

from library.quantization import save_mixed_nf4
from scripts.quantization.qwen_nf4_compile_acceptance import (
    compare_runs, run_case, tiny_loader, tiny_model,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Inductor NF4 acceptance")


@pytest.mark.parametrize("scope", ["all", "resident"])
@pytest.mark.parametrize("restore_mode", ["foreach", "slab"])
@pytest.mark.parametrize("count", [1, 2])
def test_qwen_nf4_real_inductor_edit_training(tmp_path, count, restore_mode, scope):
    torch.manual_seed(37)
    source = tiny_model()
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, .03)
            elif name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()
    path = tmp_path / "qwen-all-linear.safetensors"
    save_mixed_nf4(source, path, family="qwen_image_2_1",
        source_sha256="a" * 64, policy="all-linear-nf4-v1")
    del source
    def loader():
        return tiny_loader(path)
    eager = run_case(loader, compiled=False, count=count, restore_mode=restore_mode, scope=scope)
    compiled = run_case(loader, compiled=True, count=count, restore_mode=restore_mode, scope=scope)
    metrics = compare_runs(eager, compiled, report_path=tmp_path / "comparison.json")
    print({"count": count, "restore_mode": restore_mode, "scope": scope,
        "unique_graphs": compiled["unique_graphs"], "inductor": compiled["inductor_counters"],
        "max_relative_l2": max(value["relative_l2"] for value in metrics.values())})
