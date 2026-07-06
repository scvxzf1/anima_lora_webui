from types import SimpleNamespace

import pytest
import torch

from networks.lora_anima import optimizer_groups


class FakeLora(torch.nn.Module):
    def __init__(self, original_name: str, lora_name: str) -> None:
        super().__init__()
        self.original_name = original_name
        self.lora_name = lora_name
        self.lora_down = torch.nn.Linear(2, 1, bias=False)
        self.lora_up = torch.nn.Linear(1, 2, bias=False)
        self.router = torch.nn.Linear(2, 2, bias=False)


class FakeNetwork(torch.nn.Module):
    LORA_PREFIX_TEXT_ENCODER = "lora_te1"

    def __init__(
        self,
        *,
        unet_loras=None,
        global_router=None,
        freq_router=None,
        content_router=None,
        cfg=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SimpleNamespace(
            reg_lrs=None,
            router_lr_scale=3.0,
            content_router_lr_scale=7.0,
            freq_router_lr_scale=5.0,
            register_lr_scale=11.0,
            use_chimera_hydra=False,
        )
        self.text_encoder_loras = torch.nn.ModuleList()
        self.unet_loras = torch.nn.ModuleList(unet_loras or [])
        self.text_encoder_refts = torch.nn.ModuleList()
        self.unet_refts = torch.nn.ModuleList()
        self.global_router = global_router
        self.freq_router = freq_router
        self.content_router = content_router
        self.register_injector = None
        self.register_tokens = None
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None


def _lr_by_description(params, descriptions):
    return {desc: group["lr"] for group, desc in zip(params, descriptions)}


def test_optimizer_groups_keep_reg_lrs_loraplus_and_router_order() -> None:
    cfg = SimpleNamespace(
        reg_lrs={r"blocks\.0\.q_proj": 3e-4},
        router_lr_scale=5.0,
        content_router_lr_scale=7.0,
        freq_router_lr_scale=1.0,
        register_lr_scale=1.0,
        use_chimera_hydra=True,
    )
    net = FakeNetwork(
        cfg=cfg,
        unet_loras=[
            FakeLora("blocks.0.q_proj", "lora_unet_blocks_0_q_proj"),
            FakeLora("blocks.1.q_proj", "lora_unet_blocks_1_q_proj"),
        ],
    )
    optimizer_groups.set_loraplus_lr_ratio(net, 2.0, None, None)

    params, descriptions = optimizer_groups.prepare_lora_optimizer_params(
        net, None, 1e-4, 9e-5
    )

    assert descriptions == [
        "unet reg_lr_0",
        "unet reg_lr_0 plus",
        "unet reg_lr_0 router",
        "unet",
        "unet plus",
        "unet router",
    ]
    lrs = _lr_by_description(params, descriptions)
    assert lrs["unet reg_lr_0"] == pytest.approx(3e-4)
    assert lrs["unet reg_lr_0 plus"] == pytest.approx(6e-4)
    assert lrs["unet reg_lr_0 router"] == pytest.approx(3e-4 * 5.0 * 7.0)
    assert lrs["unet"] == pytest.approx(1e-4)
    assert lrs["unet plus"] == pytest.approx(2e-4)
    assert lrs["unet router"] == pytest.approx(1e-4 * 5.0 * 7.0)


def test_optimizer_groups_keep_network_router_lr_scales() -> None:
    net = FakeNetwork(
        global_router=torch.nn.Linear(2, 2, bias=False),
        freq_router=torch.nn.Linear(2, 2, bias=False),
        content_router=torch.nn.Linear(2, 2, bias=False),
    )

    params, descriptions = optimizer_groups.prepare_lora_optimizer_params(
        net, None, 1e-4, 9e-5
    )

    assert descriptions == [
        "global router",
        "chimera freq router",
        "chimera content router",
    ]
    lrs = _lr_by_description(params, descriptions)
    assert lrs["global router"] == pytest.approx(1e-4 * 3.0)
    assert lrs["chimera freq router"] == pytest.approx(1e-4 * 3.0 * 5.0)
    assert lrs["chimera content router"] == pytest.approx(1e-4 * 3.0 * 7.0)
