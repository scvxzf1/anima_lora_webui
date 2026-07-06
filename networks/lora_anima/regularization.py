"""Regularization helpers for the LoRA-family network facade."""

import torch


def apply_max_norm_regularization(network, max_norm_value, device):
    if getattr(network.cfg, "use_dora", False):
        return 0, 0, 0

    downkeys = []
    upkeys = []
    alphakeys = []
    norms = []
    keys_scaled = 0

    state_dict = network.state_dict()
    for key in state_dict.keys():
        if "lora_down" in key and "weight" in key:
            downkeys.append(key)
            upkeys.append(key.replace("lora_down", "lora_up"))
            alphakeys.append(key.replace("lora_down.weight", "alpha"))

    for i in range(len(downkeys)):
        down = state_dict[downkeys[i]].to(device)
        up = state_dict[upkeys[i]].to(device)
        alpha = state_dict[alphakeys[i]].to(device)
        dim = down.shape[0]
        scale = alpha / dim

        if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
            updown = (
                (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
            )
        elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
            updown = torch.nn.functional.conv2d(
                down.permute(1, 0, 2, 3), up
            ).permute(1, 0, 2, 3)
        else:
            updown = up @ down

        updown *= scale

        norm = updown.norm().clamp(min=max_norm_value / 2)
        desired = torch.clamp(norm, max=max_norm_value)
        ratio = desired.cpu() / norm.cpu()
        sqrt_ratio = ratio**0.5
        if ratio != 1:
            keys_scaled += 1
            state_dict[upkeys[i]] *= sqrt_ratio
            state_dict[downkeys[i]] *= sqrt_ratio
        scalednorm = updown.norm() * ratio
        norms.append(scalednorm.item())

    return keys_scaled, sum(norms) / len(norms), max(norms)
