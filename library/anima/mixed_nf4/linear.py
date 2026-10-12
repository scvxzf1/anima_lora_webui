"""Compatibility import for the shared BF16 autocast packed Linear."""
from library.quantization.nf4 import BF16AutocastLinear4bit

AnimaMixedNF4Linear = BF16AutocastLinear4bit
