"""
Attention Extraction and Replacement Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .tucker import TuckerDecomposition, reconstruct_tensor


class TuckerAttention(nn.Module):
    """Tucker-decomposed attention layer."""
    
    def __init__(self, hidden_size: int, num_heads: int, rank: int = 64, 
                 dropout: float = 0.1, original_attn: Optional[nn.Module] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.rank = rank
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        
        self.tucker_q = TuckerDecomposition(rank=(rank, rank, rank))
        self.tucker_k = TuckerDecomposition(rank=(rank, rank, rank))
        self.tucker_v = TuckerDecomposition(rank=(rank, rank, rank))
        
        self.core_q = None
        self.core_k = None
        self.core_v = None
        self.factors_q = None
        self.factors_k = None
        self.factors_v = None
        
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        if original_attn is not None:
            self._initialize_from_original(original_attn)
    
    def _initialize_from_original(self, original_attn: nn.Module):
        """Initialize from original attention."""
        q_weight, k_weight, v_weight = self._extract_weights(original_attn)
        self.core_q, self.factors_q = self.tucker_q.decompose(q_weight)
        self.core_k, self.factors_k = self.tucker_k.decompose(k_weight)
        self.core_v, self.factors_v = self.tucker_v.decompose(v_weight)
        
        if hasattr(original_attn, 'out_proj'):
            self.out_proj.weight = original_attn.out_proj.weight
            if hasattr(original_attn.out_proj, 'bias'):
                self.out_proj.bias = original_attn.out_proj.bias
    
    def _extract_weights(self, attn: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract Q, K, V weights."""
        q_weight = None
        k_weight = None
        v_weight = None
        
        if hasattr(attn, 'q_proj'):
            q_weight = attn.q_proj.weight.detach()
        if hasattr(attn, 'k_proj'):
            k_weight = attn.k_proj.weight.detach()
        if hasattr(attn, 'v_proj'):
            v_weight = attn.v_proj.weight.detach()
        
        if q_weight is None and hasattr(attn, 'query'):
            q_weight = attn.query.weight.detach()
        if k_weight is None and hasattr(attn, 'key'):
            k_weight = attn.key.weight.detach()
        if v_weight is None and hasattr(attn, 'value'):
            v_weight = attn.value.weight.detach()
        
        if q_weight is None or k_weight is None or v_weight is None:
            raise ValueError("Could not extract Q/K/V weights")
        
        return q_weight, k_weight, v_weight
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, _ = x.shape
        
        q_weight = reconstruct_tensor(self.core_q, self.factors_q)
        k_weight = reconstruct_tensor(self.core_k, self.factors_k)
        v_weight = reconstruct_tensor(self.core_v, self.factors_v)
        
        q = torch.matmul(x, q_weight.T)
        k = torch.matmul(x, k_weight.T)
        v = torch.matmul(x, v_weight.T)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(context)


def replace_attention(model: nn.Module, rank: int = 64, verbose: bool = True) -> nn.Module:
    """Replace attention modules with Tucker attention."""
    replaced_count = 0
    
    for name, module in model.named_modules():
        if any(pattern in name.lower() for pattern in ['attention', 'self_attn', 'attn']):
            if hasattr(module, 'q_proj') or hasattr(module, 'query') or hasattr(module, 'qkv'):
                hidden_size = getattr(module, 'hidden_size', getattr(module, 'embed_dim', 768))
                num_heads = getattr(module, 'num_heads', getattr(module, 'n_heads', 12))
                
                tucker_attn = TuckerAttention(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    rank=rank,
                    original_attn=module
                )
                
                parts = name.split('.')
                parent_name = '.'.join(parts[:-1])
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, parts[-1], tucker_attn)
                    replaced_count += 1
                    if verbose:
                        print(f"Replaced {name}")
    
    if verbose:
        print(f"Total replaced: {replaced_count}")
    
    return model