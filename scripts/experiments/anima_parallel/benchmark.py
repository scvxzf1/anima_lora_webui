"""Matched-sample PP/single/TP execution for the pipeline optimization probe."""

from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .checkpoint import load_stage_checkpoint, save_stage_checkpoint
from .common import create_mlp_lora, load_model, to_device_batch, write_json
from .stages import AnimaOwnedStage, build_schedule, stage_examples


def configure_checkpoint(model, policy: str):
    if policy not in {"full", "off", "every_other"}:
        raise ValueError(f"unknown checkpoint policy {policy}")
    for index, block in enumerate(model.blocks):
        block.gradient_checkpointing = policy == "full" or (
            policy == "every_other" and index % 2 == 0
        )


def pack_batches(batches, device):
    moved = [to_device_batch(batch, device, torch.bfloat16) for batch in batches]
    inputs = tuple(torch.cat([batch[0][i] for batch in moved]) for i in range(4))
    target = torch.cat([batch[1] for batch in moved])
    return inputs, target


class Experiment:
    def __init__(self, args, rank, device, example):
        self.args, self.rank, self.device = args, rank, device
        self.schedule = None
        self.tp_specs = None
        self.tp_stats = None
        model = load_model(args.dit_path, torch.device("cpu"), args.attn_mode)
        network = create_mlp_lora(
            model, seed=args.seed + 101, rank_dim=args.rank_dim, alpha=args.alpha
        )
        configure_checkpoint(model, args.checkpoint)
        if args.mode == "pp":
            inputs, _ = pack_batches([example], torch.device("cpu"))
            boundary, output, rope = stage_examples(model, inputs)
            # RoPE caches are ordinary dictionaries, not registered buffers.
            # Rebuild on the compute device to match the unsharded forward.
            model.pos_embedder.to(device=device, dtype=torch.bfloat16)
            model.pos_embedder._cos_sin_cache.clear()
            with torch.no_grad():
                rope = tuple(
                    value.to(torch.bfloat16)
                    for value in model.pos_embedder.generate_embeddings(
                        boundary[0].shape
                    )
                )
            self.module = AnimaOwnedStage(
                model, network, rank=rank, split=args.split, rope=rope
            )
            self.module.to(device=device, dtype=torch.bfloat16).train()
            examples = tuple(value.to(device) for value in inputs)
            boundary = tuple(
                value.to(device).detach().requires_grad_(value.requires_grad)
                for value in boundary
            )
            output = output.to(device).detach().requires_grad_()
            self.schedule = build_schedule(
                self.module,
                examples,
                boundary,
                output,
                microbatches=args.microbatches,
                schedule=args.schedule,
            )
            self.adapters = self.module.adapters
        else:
            if args.mode == "tp":
                from .collectives import CommunicationStats, configure_collectives
                from .tensor_parallel import parallelize_anima_blocks

                self.tp_stats = CommunicationStats()
                configure_collectives("bf16", self.tp_stats)
                self.tp_specs = parallelize_anima_blocks(
                    model, network, rank=rank, world=2
                )
            self.module = model.to(device=device, dtype=torch.bfloat16).train()
            self.adapters = network.to(device=device, dtype=torch.bfloat16).train()
        self.optimizer = torch.optim.AdamW(self.adapters.parameters(), lr=args.lr)
        del model, network
        gc.collect()
        torch.cuda.empty_cache()

    def step(self, inputs, target, *, update=True, capture=False):
        self.optimizer.zero_grad(set_to_none=True)
        if self.schedule is not None:
            losses = []
            if self.rank == 0:
                self.schedule.step(*inputs, losses=losses, return_outputs=False)
            else:
                output = self.schedule.step(target=target, losses=losses, return_outputs=capture)
                if capture:
                    self.captured_output = output.detach()
            loss = (
                torch.stack(losses).mean()
                if losses
                else torch.zeros((), device=self.device)
            )
        else:
            losses = []
            captured_outputs = []
            for index in range(len(target)):
                x, timestep, context, mask = (
                    value[index : index + 1] for value in inputs
                )
                output = self.module(x, timestep, context, padding_mask=mask)
                loss = F.mse_loss(output.float(), target[index : index + 1].float())
                (loss / len(target)).backward()
                losses.append(loss.detach())
                if capture:
                    captured_outputs.append(output.detach())
            if self.tp_specs is not None:
                from .tensor_parallel import synchronize_replicated_lora_gradients

                synchronize_replicated_lora_gradients(self.adapters, self.tp_specs)
            loss = torch.stack(losses).mean()
            if capture:
                self.captured_output = torch.cat(captured_outputs)
        if update:
            self.optimizer.step()
        return loss.detach()

    def communication(self):
        if self.schedule is not None:
            return dict(self.schedule._stage.communication)
        if self.tp_stats is not None:
            return {
                "collective_calls": self.tp_stats.collective_calls,
                "logical_tensor_bytes": self.tp_stats.payload_bytes,
            }
        return {}

    def adapter_state(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.adapters.state_dict().items()
        }

    def verify_resume(self, directory: Path, inputs, target, contract):
        if self.schedule is None:
            return None
        path = directory / f"resume-check-rank{self.rank}.pt"
        torch.cuda.synchronize(self.device)
        save_stage_checkpoint(
            path, self.module, self.optimizer, step=0, contract=contract
        )
        loss_a = self.step(inputs, target)
        state_a = self.adapter_state()
        gradients_a = {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in self.adapters.named_parameters()
            if parameter.grad is not None
        }
        load_stage_checkpoint(path, self.module, self.optimizer, contract=contract)
        loss_b = self.step(inputs, target)
        state_b = self.adapter_state()
        max_abs = max(
            float((state_a[key].float() - value.float()).abs().max())
            for key, value in state_b.items()
        )
        loss_delta = float((loss_a - loss_b).abs())
        gradient_details = []
        for name, parameter in self.adapters.named_parameters():
            if parameter.grad is None:
                continue
            first, second = gradients_a[name], parameter.grad.detach().float().cpu()
            difference = first - second
            gradient_details.append(
                {
                    "name": name,
                    "max_abs": float(difference.abs().max()),
                    "rel_l2": float(difference.norm() / first.norm().clamp_min(1e-30)),
                    "sign_flips": int(((first * second) < 0).sum()),
                    "numel": first.numel(),
                }
            )
        report = {
            "adapter_max_abs": max_abs,
            "loss_abs": loss_delta,
            "gradients": gradient_details,
        }
        write_json(directory / f"resume-diagnostics-rank{self.rank}.json", report)
        load_stage_checkpoint(path, self.module, self.optimizer, contract=contract)
        if max_abs != 0 or loss_delta != 0:
            raise RuntimeError(
                f"PP resume mismatch: adapter={max_abs}, loss={loss_delta}"
            )
        return report


def timed_step(experiment, inputs, target, *, distributed, update=True, capture=False):
    if distributed:
        dist.barrier()
    torch.cuda.synchronize(experiment.device)
    started = perf_counter()
    loss = experiment.step(inputs, target, update=update, capture=capture)
    torch.cuda.synchronize(experiment.device)
    elapsed = perf_counter() - started
    times = torch.tensor([elapsed], dtype=torch.float64, device=experiment.device)
    if distributed:
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
        if experiment.args.mode == "pp":
            dist.broadcast(loss, src=1)
    return {
        "seconds": float(times.item()),
        "rank_seconds": elapsed,
        "loss": float(loss),
    }
