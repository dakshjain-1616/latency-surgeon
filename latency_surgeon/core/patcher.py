"""
Auto-detect and surgically replace attention projections in HF transformers.

This module walks through model named_modules(), pattern-matches q_proj/k_proj/v_proj/o_proj,
and produces a SurgeryManifest for the attention replacement procedure.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import torch.nn as nn


@dataclass
class SurgicalTarget:
    """Represents a single attention projection layer to be replaced."""
    module_name: str
    module_path: str  # e.g., "encoder.layer.0.attention.self.query"
    in_features: int
    out_features: int
    layer_type: str  # "q_proj", "k_proj", "v_proj", "o_proj"
    parent_module: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "module_path": self.module_path,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "layer_type": self.layer_type,
            "parent_module": self.parent_module
        }


@dataclass
class SurgeryManifest:
    """Complete surgical plan for attention replacement."""
    model_name: str
    total_layers: int
    attention_targets: List[SurgicalTarget] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_target(self, target: SurgicalTarget) -> None:
        self.attention_targets.append(target)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_layers": self.total_layers,
            "attention_targets": [t.to_dict() for t in self.attention_targets],
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(self.to_json())


# Common attention layer patterns across HF models
ATTENTION_PATTERNS = {
    # BERT-style
    "bert": ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"],
    # GPT-style
    "gpt": ["attn.c_attn", "attn.c_proj", "c_attn", "c_proj"],
    # Llama-style
    "llama": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    # T5-style
    "t5": ["SelfAttention.q", "SelfAttention.k", "SelfAttention.v", "SelfAttention.o"],
    # Generic patterns
    "generic": ["q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value", "out_proj"]
}

LAYER_TYPE_MAP = {
    "q_proj": "q_proj", "query": "q_proj", "q": "q_proj",
    "k_proj": "k_proj", "key": "k_proj", "k": "k_proj",
    "v_proj": "v_proj", "value": "v_proj", "v": "v_proj",
    "o_proj": "o_proj", "out_proj": "o_proj", "dense": "o_proj", "c_proj": "o_proj"
}


def detect_model_family(model_name: str) -> str:
    """Detect model family from model name or config."""
    model_name_lower = model_name.lower()
    if "bert" in model_name_lower:
        return "bert"
    elif "gpt" in model_name_lower or "gpt2" in model_name_lower:
        return "gpt"
    elif "llama" in model_name_lower:
        return "llama"
    elif "t5" in model_name_lower:
        return "t5"
    return "generic"


def is_attention_layer(module_name: str, module: nn.Module, model_family: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a module is an attention projection layer.
    
    Returns:
        Tuple of (is_attention, layer_type) where layer_type is "q_proj"/"k_proj"/"v_proj"/"o_proj" or None
    """
    # Check if it's a Linear layer (attention projections are Linear)
    if not isinstance(module, nn.Linear):
        return False, None
    
    patterns = ATTENTION_PATTERNS.get(model_family, ATTENTION_PATTERNS["generic"])
    
    for pattern in patterns:
        if pattern in module_name.lower():
            # Determine layer type
            for key, layer_type in LAYER_TYPE_MAP.items():
                if key in module_name.lower():
                    return True, layer_type
    
    return False, None


def walk_modules(model: nn.Module, prefix: str = "") -> List[Tuple[str, nn.Module]]:
    """
    Recursively walk all named modules in a model.
    
    Returns:
        List of (full_path, module) tuples
    """
    results = []
    for name, module in model.named_modules():
        full_path = f"{prefix}.{name}" if prefix else name
        results.append((full_path, module))
    return results


def create_surgery_manifest(
    model: nn.Module,
    model_name: str,
    model_family: Optional[str] = None
) -> SurgeryManifest:
    """
    Create a complete surgery manifest for attention replacement.
    
    Args:
        model: The HF transformer model
        model_name: Model name/id (e.g., "bert-base-uncased")
        model_family: Optional override for model family detection
        
    Returns:
        SurgeryManifest with all attention targets identified
    """
    if model_family is None:
        model_family = detect_model_family(model_name)
    
    # Count total layers (heuristic: count "layer" in module paths)
    total_layers = 0
    for name, _ in model.named_modules():
        if "layer" in name.lower():
            try:
                layer_num = int(name.split(".")[-1])
                total_layers = max(total_layers, layer_num + 1)
            except (ValueError, IndexError):
                pass
    
    manifest = SurgeryManifest(
        model_name=model_name,
        total_layers=total_layers,
        metadata={"model_family": model_family, "detected_at": str(torch.cuda.is_available())}
    )
    
    # Walk all modules and identify attention targets
    for module_path, module in walk_modules(model):
        is_attn, layer_type = is_attention_layer(module_path.split(".")[-1], module, model_family)
        
        if is_attn and layer_type:
            target = SurgicalTarget(
                module_name=module_path.split(".")[-1],
                module_path=module_path,
                in_features=module.in_features,
                out_features=module.out_features,
                layer_type=layer_type,
                parent_module=module_path.rsplit(".", 1)[0] if "." in module_path else None
            )
            manifest.add_target(target)
    
    return manifest


def apply_surgery(
    model: nn.Module,
    manifest: SurgeryManifest,
    tucker_linear_class: Any,  # TuckerLinear from tucker.py
    rank: int
) -> nn.Module:
    """
    Apply the surgical replacement of attention layers.
    
    Args:
        model: The model to modify (modified in-place)
        manifest: SurgeryManifest from create_surgery_manifest
        tucker_linear_class: TuckerLinear class to use for replacement
        rank: Tucker decomposition rank
        
    Returns:
        The modified model (same object, modified in-place)
    """
    for target in manifest.attention_targets:
        # Get parent module
        parent_path = target.parent_module
        if parent_path:
            parent = model.get_submodule(parent_path)
        else:
            parent = model
        
        # Get the original layer
        original_layer = model.get_submodule(target.module_path)
        
        # Create TuckerLinear replacement
        tucker_layer = tucker_linear_class.from_linear(original_layer, rank)
        
        # Replace in parent
        layer_name = target.module_name
        setattr(parent, layer_name, tucker_layer)
    
    return model


def get_surgery_stats(manifest: SurgeryManifest) -> Dict[str, Any]:
    """Get statistics about the planned surgery."""
    layer_counts = {"q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 0}
    total_params = 0
    
    for target in manifest.attention_targets:
        layer_counts[target.layer_type] = layer_counts.get(target.layer_type, 0) + 1
        total_params += target.in_features * target.out_features
    
    return {
        "total_targets": len(manifest.attention_targets),
        "layer_counts": layer_counts,
        "total_attention_params": total_params,
        "model_layers": manifest.total_layers
    }