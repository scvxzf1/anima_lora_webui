# LoRA module building blocks. Public API re-exported here so
# `from networks.lora_modules import LoRAModule, ...` works unchanged.

from .base import BaseLoRAModule, _absorb_channel_scale
from .chimera import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
)
from .dora import DoRALoRAModule
from .hydra import HydraLoRAModule, _sigma_sinusoidal_features
from .lora import LoRAModule
from .ortho import (
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
)
from .reft import ReFTModule
from .stacked_experts import StackedExpertsLoRAModule
from .step_expert_lora import StepExpertLoRAModule

__all__ = [
    "BaseLoRAModule",
    "ChimeraHydraInferenceModule",
    "ChimeraHydraLoRAModule",
    "DoRALoRAModule",
    "HydraLoRAModule",
    "LoRAModule",
    "OrthoHydraLoRAModule",
    "OrthoLoRAModule",
    "ReFTModule",
    "StackedExpertsLoRAModule",
    "StepExpertLoRAModule",
    "_absorb_channel_scale",
    "_sigma_sinusoidal_features",
]
