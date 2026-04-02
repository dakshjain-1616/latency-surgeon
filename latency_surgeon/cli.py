"""
CLI Interface for LatencySurgeon
"""

import click
import torch
from typing import Optional

from .hf_integration import (
    load_model, save_model, optimize_model, 
    benchmark_model, detect_attention_layers, get_model_info
)
from .tucker import validate_decomposition


@click.group()
def main():
    """LatencySurgeon - Tucker-decomposed attention for faster inference"""
    pass


@main.command()
@click.option('--model', required=True, help='HuggingFace model name')
@click.option('--rank', default=64, help='Tucker decomposition rank')
@click.option('--output', default='./optimized_model', help='Output directory')
@click.option('--task', default='base', help='Task type: base or classification')
def decompose(model: str, rank: int, output: str, task: str):
    """Decompose a HuggingFace model with Tucker attention."""
    print(f"Loading model: {model}")
    original = load_model(model, task=task)
    
    print(f"Detecting attention layers...")
    layers = detect_attention_layers(original)
    print(f"Found {len(layers)} attention layers")
    
    print(f"Optimizing with rank={rank}...")
    optimized = optimize_model(original, rank=rank, verbose=True)
    
    print(f"Saving to {output}...")
    save_model(optimized, output)
    print("Done!")


@main.command()
@click.option('--model', required=True, help='HuggingFace model name')
def analyze(model: str):
    """Analyze model architecture."""
    print(f"Loading model: {model}")
    model_obj = load_model(model)
    
    info = get_model_info(model_obj)
    print("\nModel Information:")
    print(f"  Type: {info['type']}")
    print(f"  Parameters: {info['num_parameters']:,}")
    print(f"  Hidden size: {info.get('hidden_size', 'N/A')}")
    print(f"  Num heads: {info.get('num_heads', 'N/A')}")
    print(f"  Attention layers: {info['num_attention_layers']}")
    
    if info['attention_layers']:
        print("\nAttention layer paths:")
        for layer in info['attention_layers'][:10]:
            print(f"  - {layer}")


@main.command()
@click.option('--model', required=True, help='HuggingFace model name')
@click.option('--rank', default=64, help='Tucker rank')
@click.option('--runs', default=10, help='Number of benchmark runs')
@click.option('--decomposed', is_flag=True, help='Benchmark decomposed model')
def benchmark(model: str, rank: int, runs: int, decomposed: bool):
    """Benchmark model latency."""
    print(f"Loading model: {model}")
    original = load_model(model)
    
    if decomposed:
        print(f"Creating decomposed version (rank={rank})...")
        model_to_test = optimize_model(original, rank=rank, verbose=False)
    else:
        model_to_test = original
    
    print(f"Running benchmark ({runs} runs)...")
    results = benchmark_model(model_to_test, num_runs=runs)
    
    print("\nBenchmark Results:")
    print(f"  Mean latency: {results['mean_latency']*1000:.2f} ms")
    print(f"  Min latency: {results['min_latency']*1000:.2f} ms")
    print(f"  Max latency: {results['max_latency']*1000:.2f} ms")


@main.command()
@click.option('--tensor-shape', default='10,10,10', help='Test tensor shape')
@click.option('--rank', default=5, help='Test rank')
def validate(tensor_shape: str, rank: int):
    """Validate Tucker decomposition."""
    shape = tuple(int(x) for x in tensor_shape.split(','))
    print(f"Testing Tucker decomposition with shape={shape}, rank={rank}")
    
    mse = validate_decomposition(tensor_shape=shape, rank=rank)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    if mse < 0.1:
        print("✓ Decomposition validated successfully")
    else:
        print("✗ Decomposition may need adjustment")


@main.command()
@click.option('--model', required=True, help='HuggingFace model name')
@click.option('--rank', default=64, help='Tucker rank')
def compare(model: str, rank: int):
    """Compare original vs decomposed model outputs."""
    from .hf_integration import compare_models
    
    print(f"Loading model: {model}")
    original = load_model(model)
    
    print(f"Creating decomposed version...")
    optimized = optimize_model(original, rank=rank, verbose=False)
    
    dummy_input = torch.randn(1, 128, 768)
    
    print("Comparing outputs...")
    comparison = compare_models(original, optimized, dummy_input)
    
    print("\nComparison Results:")
    print(f"  Cosine similarity: {comparison['cosine_similarity']:.6f}")
    print(f"  MSE: {comparison['mse']:.6f}")
    print(f"  Output shape: {comparison['output_shape']}")


if __name__ == '__main__':
    main()