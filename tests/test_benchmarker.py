"""Tests for the benchmarker module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
from latency_surgeon.core.benchmarker import (
    benchmark_model, compare_benchmarks, BenchmarkResult, get_gpu_memory_mb
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def tiny_model():
    return TinyModel()


@pytest.fixture
def sample_input():
    return torch.randn(1, 8, 16)


def test_benchmark_model_runs(tiny_model, sample_input):
    result = benchmark_model(tiny_model, sample_input, num_runs=5, use_rich=False)
    assert isinstance(result, BenchmarkResult)
    assert result.total_runs == 5
    assert result.tokens_per_sec > 0


def test_benchmark_model_latencies(tiny_model, sample_input):
    result = benchmark_model(tiny_model, sample_input, num_runs=10, use_rich=False)
    assert result.latency_p50 > 0
    assert result.latency_p95 >= result.latency_p50
    assert result.latency_p99 >= result.latency_p95


def test_benchmark_model_cpu_only(tiny_model, sample_input):
    result = benchmark_model(tiny_model, sample_input, num_runs=5, device="cpu", use_rich=False)
    assert result.cpu_only is True
    assert result.peak_vram_mb is None


def test_benchmark_result_to_dict(tiny_model, sample_input):
    result = benchmark_model(tiny_model, sample_input, num_runs=3, use_rich=False)
    d = result.to_dict()
    assert "tokens_per_sec" in d
    assert "latency_p50_ms" in d
    assert "cpu_only" in d


def test_benchmark_result_summary(tiny_model, sample_input):
    result = benchmark_model(tiny_model, sample_input, num_runs=3, use_rich=False)
    summary = result.summary()
    assert isinstance(summary, str)
    assert "Tokens/sec" in summary


def test_compare_benchmarks(tiny_model, sample_input):
    before = benchmark_model(tiny_model, sample_input, num_runs=5, use_rich=False)
    after = benchmark_model(tiny_model, sample_input, num_runs=5, use_rich=False)
    comparison = compare_benchmarks(before, after)
    assert "speedup_factor" in comparison
    assert "latency_reduction_percent" in comparison
    assert isinstance(comparison["speedup_factor"], float)


def test_get_gpu_memory_mb():
    mem = get_gpu_memory_mb()
    # On CPU, should return None
    if not torch.cuda.is_available():
        assert mem is None
