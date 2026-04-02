"""
Tucker Decomposition Core Module
"""

import numpy as np
import torch
from typing import Tuple, List
from sklearn.decomposition import TruncatedSVD


class TuckerDecomposition:
    """Tucker decomposition for tensors."""
    
    def __init__(self, rank: Tuple[int, ...] = (64, 64, 64)):
        self.rank = rank
        self.core_tensor = None
        self.factor_matrices = None
    
    def decompose(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Decompose tensor using Tucker decomposition."""
        dims = tensor.dim()
        if dims == 2:
            return self._decompose_2d(tensor)
        elif dims == 3:
            return self._decompose_3d(tensor)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {dims}D")
    
    def _decompose_2d(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """2D decomposition using SVD."""
        U, S, V = torch.svd(tensor)
        r = min(self.rank[0], len(S))
        core = torch.diag(S[:r])
        factors = [U[:, :r], V[:, :r]]
        self.core_tensor = core
        self.factor_matrices = factors
        return core, factors
    
    def _decompose_3d(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """3D Tucker decomposition."""
        I, J, K = tensor.shape
        R1, R2, R3 = min(self.rank[0], I), min(self.rank[1], J), min(self.rank[2], K)
        
        factors = []
        for mode, dim, rank in [(0, I, R1), (1, J, R2), (2, K, R3)]:
            perm = [mode] + [i for i in range(3) if i != mode]
            tensor_flat = tensor.permute(*perm).reshape(dim, -1)
            U, S, V = torch.svd(tensor_flat)
            factor = U[:, :rank]
            factors.append(factor)
        
        core = tensor
        for i, factor in enumerate(factors):
            core = torch.tensordot(core, factor, dims=[[i], [0]])
        
        self.core_tensor = core
        self.factor_matrices = factors
        return core, factors
    
    def reconstruct(self, core_tensor=None, factor_matrices=None) -> torch.Tensor:
        """Reconstruct tensor from decomposition."""
        if core_tensor is None:
            core_tensor = self.core_tensor
        if factor_matrices is None:
            factor_matrices = self.factor_matrices
        
        if core_tensor is None or factor_matrices is None:
            raise ValueError("No decomposition data")
        
        reconstructed = core_tensor
        for i, factor in enumerate(factor_matrices):
            reconstructed = torch.tensordot(reconstructed, factor, dims=[[i], [1]])
        
        return reconstructed


def decompose_tensor(tensor: torch.Tensor, rank: int = 64) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Convenience function for Tucker decomposition."""
    tucker = TuckerDecomposition(rank=(rank, rank, rank) if tensor.dim() == 3 else (rank, rank))
    return tucker.decompose(tensor)


def reconstruct_tensor(core_tensor: torch.Tensor, factor_matrices: List[torch.Tensor]) -> torch.Tensor:
    """Convenience function for reconstruction."""
    tucker = TuckerDecomposition()
    return tucker.reconstruct(core_tensor, factor_matrices)


def validate_decomposition(tensor_shape: Tuple[int, int, int] = (10, 10, 10), rank: int = 5) -> float:
    """Validate Tucker decomposition."""
    test_tensor = torch.randn(tensor_shape)
    tucker = TuckerDecomposition(rank=(rank, rank, rank))
    core, factors = tucker.decompose(test_tensor)
    reconstructed = tucker.reconstruct(core, factors)
    mse = torch.mean((test_tensor - reconstructed) ** 2)
    return mse.item()