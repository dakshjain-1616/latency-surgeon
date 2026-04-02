"""Tests for the patcher module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch.nn as nn
from latency_surgeon.core.patcher import (
    SurgicalTarget, SurgeryManifest, create_surgery_manifest,
    is_attention_layer, detect_model_family, get_surgery_stats
)


class SimpleAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)
        self.ff = nn.Linear(64, 256)


def test_detect_model_family_bert():
    assert detect_model_family("bert-base-uncased") == "bert"


def test_detect_model_family_gpt():
    assert detect_model_family("gpt2") == "gpt"


def test_detect_model_family_llama():
    assert detect_model_family("llama-7b") == "llama"


def test_detect_model_family_generic():
    assert detect_model_family("unknown-model") == "generic"


def test_is_attention_layer_linear():
    layer = nn.Linear(64, 64)
    is_attn, layer_type = is_attention_layer("q_proj", layer, "generic")
    assert is_attn is True
    assert layer_type == "q_proj"


def test_is_attention_layer_non_linear():
    layer = nn.ReLU()
    is_attn, layer_type = is_attention_layer("q_proj", layer, "generic")
    assert is_attn is False
    assert layer_type is None


def test_is_attention_layer_ff():
    layer = nn.Linear(64, 256)
    is_attn, _ = is_attention_layer("fc1", layer, "generic")
    assert is_attn is False


def test_surgical_target_to_dict():
    target = SurgicalTarget(
        module_name="q_proj",
        module_path="layer.0.q_proj",
        in_features=64,
        out_features=64,
        layer_type="q_proj",
    )
    d = target.to_dict()
    assert d["module_name"] == "q_proj"
    assert d["in_features"] == 64
    assert d["layer_type"] == "q_proj"


def test_surgery_manifest_add_target():
    manifest = SurgeryManifest(model_name="test", total_layers=2)
    target = SurgicalTarget("q_proj", "layer.0.q_proj", 64, 64, "q_proj")
    manifest.add_target(target)
    assert len(manifest.attention_targets) == 1


def test_surgery_manifest_to_json():
    manifest = SurgeryManifest(model_name="test", total_layers=1)
    json_str = manifest.to_json()
    assert "test" in json_str


def test_create_surgery_manifest():
    model = SimpleAttentionModel()
    manifest = create_surgery_manifest(model, "test-model", model_family="generic")
    assert isinstance(manifest, SurgeryManifest)
    assert manifest.model_name == "test-model"
    assert len(manifest.attention_targets) > 0


def test_get_surgery_stats():
    model = SimpleAttentionModel()
    manifest = create_surgery_manifest(model, "test-model", model_family="generic")
    stats = get_surgery_stats(manifest)
    assert "total_targets" in stats
    assert stats["total_targets"] > 0
