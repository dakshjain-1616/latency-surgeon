"""
HuggingFace Integration Layer
"""

import torch
import torch.nn as nn
import time
from typing import Optional, Dict, Any, List, Tuple

from .attention_replace import replace_attention


def load_model(model_name: str, task: str = "base", **kwargs) -> nn.Module:
    """Load a HuggingFace transformer model."""
    try:
        from transformers import AutoModel, AutoModelForSequenceClassification
        if task == "classification":
            return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        return AutoModel.from_pretrained(model_name, **kwargs)
    except ImportError:
        raise ImportError("transformers library required. Install with: pip install transformers")


def save_model(model: nn.Module, output_path: str, config: Optional[Any] = None):
    """Save a modified model."""
    model.save_pretrained(output_path)
    if config is not None:
        config.save_pretrained(output_path)


def detect_attention_layers(model: nn.Module) -> List[str]:
    """Detect all attention layers in a model."""
    attention_layers = []
    
    for name, module in model.named_modules():
        if any(pattern in name.lower() for pattern in ['attention', 'self_attn', 'attn']):
            if hasattr(module, 'q_proj') or hasattr(module, 'query') or hasattr(module, 'qkv'):
                attention_layers.append(name)
    
    return attention_layers


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get model architecture information."""
    info = {
        'type': model.__class__.__name__,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_attention_layers': len(detect_attention_layers(model)),
        'attention_layers': detect_attention_layers(model),
    }
    
    for module in model.modules():
        if hasattr(module, 'hidden_size'):
            info['hidden_size'] = module.hidden_size
            break
        if hasattr(module, 'embed_dim'):
            info['hidden_size'] = module.embed_dim
            break
    
    for module in model.modules():
        if hasattr(module, 'num_heads'):
            info['num_heads'] = module.num_heads
            break
        if hasattr(module, 'n_heads'):
            info['num_heads'] = module.n_heads
            break
    
    return info


def optimize_model(model: nn.Module, rank: int = 64, verbose: bool = True) -> nn.Module:
    """Optimize model with Tucker decomposition."""
    if verbose:
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"Optimizing with Tucker decomposition (rank={rank})")
        print(f"Original parameters: {orig_params}")
    
    optimized = replace_attention(model, rank=rank, verbose=verbose)
    
    if verbose:
        opt_params = sum(p.numel() for p in optimized.parameters())
        print(f"Optimized parameters: {opt_params}")
        compression = 1 - (opt_params / orig_params) if orig_params > 0 else 0
        print(f"Compression: {compression*100:.1f}%")
    
    return optimized


def benchmark_model(model: nn.Module, input_shape: Tuple[int, int, int] = (1, 128, 768),
                    num_runs: int = 10, device: str = "cpu") -> Dict[str, float]:
    """Benchmark model inference latency."""
    model = model.to(device)
    model.eval()
    
    batch_size, seq_len, hidden = input_shape
    dummy_input = torch.randn(batch_size, seq_len, hidden).to(device)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    
    return {
        'mean_latency': sum(times) / len(times),
        'min_latency': min(times),
        'max_latency': max(times),
    }


def compare_models(original_model: nn.Module, optimized_model: nn.Module,
                   test_input: torch.Tensor) -> Dict[str, Any]:
    """Compare original and optimized models."""
    original_model.eval()
    optimized_model.eval()
    
    with torch.no_grad():
        original_output = original_model(test_input)
        optimized_output = optimized_model(test_input)
    
    def get_tensor(output):
        if hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state
        elif hasattr(output, '__getitem__'):
            return output[0]
        return output
    
    orig_tensor = get_tensor(original_output)
    opt_tensor = get_tensor(optimized_output)
    
    cosine_sim = torch.nn.functional.cosine_similarity(orig_tensor.flatten(), opt_tensor.flatten())
    mse = torch.mean((orig_tensor - opt_tensor) ** 2)
    
    return {
        'cosine_similarity': cosine_sim.item(),
        'mse': mse.item(),
        'output_shape': orig_tensor.shape,
    }