"""
Tests for Tucker decomposition module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from latency_surgeon.tucker import TuckerDecomposition, decompose_tensor, reconstruct_tensor, validate_decomposition


def test_tucker_3d_decomposition():
    """Test Tucker decomposition on 3D tensor."""
    tensor = torch.randn(10, 10, 10)
    tucker = TuckerDecomposition(rank=(5, 5, 5))
    
    core, factors = tucker.decompose(tensor)
    reconstructed = tucker.reconstruct(core, factors)
    
    mse = torch.mean((tensor - reconstructed) ** 2).item()
    assert mse < 1.0, f"Reconstruction MSE too high: {mse}"
    print(f"✓ 3D decomposition test passed (MSE={mse:.4f})")


def test_tucker_2d_decomposition():
    """Test Tucker decomposition on 2D tensor."""
    tensor = torch.randn(10, 10)
    tucker = TuckerDecomposition(rank=(5, 5))
    
    core, factors = tucker.decompose(tensor)
    reconstructed = tucker.reconstruct(core, factors)
    
    mse = torch.mean((tensor - reconstructed) ** 2).item()
    assert mse < 1.0, f"Reconstruction MSE too high: {mse}"
    print(f"✓ 2D decomposition test passed (MSE={mse:.4f})")


def test_decompose_tensor_function():
    """Test convenience function."""
    tensor = torch.randn(6, 6, 6)
    core, factors = decompose_tensor(tensor, rank=3)
    
    assert core is not None
    assert len(factors) == 3
    print("✓ decompose_tensor function test passed")


def test_reconstruct_tensor_function():
    """Test reconstruction convenience function."""
    tensor = torch.randn(5, 5, 5)
    core, factors = decompose_tensor(tensor, rank=3)
    reconstructed = reconstruct_tensor(core, factors)
    
    assert reconstructed.shape == tensor.shape
    print("✓ reconstruct_tensor function test passed")


def test_validate_decomposition():
    """Test validation function."""
    mse = validate_decomposition(tensor_shape=(10, 10, 10), rank=5)
    assert mse < 0.5, f"Validation MSE too high: {mse}"
    print(f"✓ validate_decomposition test passed (MSE={mse:.4f})")


if __name__ == '__main__':
    print("Running Tucker decomposition tests...\n")
    
    test_tucker_3d_decomposition()
    test_tucker_2d_decomposition()
    test_decompose_tensor_function()
    test_reconstruct_tensor_function()
    test_validate_decomposition()
    
    print("\n✓ All Tucker tests passed!")