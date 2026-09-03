from __future__ import annotations

import os
import socket
from multiprocessing import get_context

import pytest
import torch
import torch.distributed as dist


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_tiny(seed: int, dtype: torch.dtype = torch.float32):
    from library.anima.models import Anima
    from scripts.experiments.anima_parallel.common import create_mlp_lora

    torch.manual_seed(seed)
    model = Anima(
        max_img_h=4,
        max_img_w=4,
        max_frames=1,
        in_channels=4,
        out_channels=4,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=24,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        crossattn_emb_channels=24,
        pos_emb_learnable=True,
        use_adaln_lora=True,
        adaln_lora_dim=8,
        use_llm_adapter=False,
        attn_mode="torch",
    ).to(dtype=dtype)
    network = create_mlp_lora(model, seed=seed + 1, rank_dim=4, alpha=4.0)
    network.to(dtype=dtype)
    return model, network


def _tp_worker(rank: int, world: int, port: int, mode: str, queue) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from scripts.experiments.anima_parallel.collectives import (
            CommunicationStats,
            configure_collectives,
        )
        from scripts.experiments.anima_parallel.tensor_parallel import (
            parallelize_anima_blocks,
            synchronize_replicated_lora_gradients,
        )

        generator = torch.Generator().manual_seed(991)
        dtype = torch.bfloat16
        x = torch.randn((1, 4, 1, 4, 4), generator=generator).to(dtype)
        timestep = torch.tensor([0.4], dtype=dtype)
        context = torch.randn((1, 3, 24), generator=generator).to(dtype)

        baseline, _baseline_network = _build_tiny(123, dtype)
        expected = baseline(x, timestep, context).detach()

        model, network = _build_tiny(123, dtype)
        specs = parallelize_anima_blocks(model, network, rank=rank, world=world)
        stats = CommunicationStats()
        configure_collectives(mode, stats)
        output = model(x, timestep, context)
        output.square().mean().backward()
        synchronize_replicated_lora_gradients(network, specs)
        max_abs = float((output.detach() - expected).abs().max())
        finite_grads = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in network.parameters()
        )
        queue.put((rank, max_abs, finite_grads, stats.wire_bytes, stats.collective_calls))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("mode", "tolerance"), [("bf16", 5e-2), ("int8", 1e-1)])
def test_tensor_parallel_tiny_anima_matches_full_model(mode: str, tolerance: float):
    ctx = get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    processes = [
        ctx.Process(target=_tp_worker, args=(rank, 2, port, mode, queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=90) for _ in processes]
    for process in processes:
        process.join(timeout=90)
        assert process.exitcode == 0
    assert {rank for rank, *_ in results} == {0, 1}
    assert max(max_abs for _, max_abs, *_ in results) <= tolerance
    assert all(finite for _, _, finite, _, _ in results)
    assert all(wire_bytes > 0 and calls > 0 for _, _, _, wire_bytes, calls in results)


def test_consolidate_tp_state_reassembles_shards():
    from scripts.experiments.anima_parallel.tensor_parallel import (
        ShardSpec,
        consolidate_tp_state,
    )

    states = [
        {"up": torch.tensor([[1.0, 2.0]]), "replica": torch.tensor([7.0])},
        {"up": torch.tensor([[3.0, 4.0]]), "replica": torch.tensor([7.0])},
    ]
    merged = consolidate_tp_state(states, {"up": ShardSpec(0), "replica": ShardSpec(None)})
    assert torch.equal(merged["up"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(merged["replica"], torch.tensor([7.0]))
