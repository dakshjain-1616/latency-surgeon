"""
LatencySurgeon - Tucker-decomposed attention for HuggingFace transformers
"""

from .tucker import TuckerDecomposition, decompose_tensor, reconstruct_tensor, validate_decomposition
from .attention_replace import TuckerAttention, replace_attention
from .hf_integration import load_model, optimize_model, benchmark_model

__version__ = "0.1.0"
__all__ = [
    "TuckerDecomposition",
    "TuckerAttention", 
    "decompose_tensor",
    "reconstruct_tensor",
    "validate_decomposition",
    "replace_attention",
    "load_model",
    "optimize_model",
    "benchmark_model",
]