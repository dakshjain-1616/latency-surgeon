"""
Performance benchmarking with Rich Live panels.

Measures tokens/sec, p50/p95/p99 latency, peak VRAM, with CPU fallback support.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import time
import statistics
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from contextlib import contextmanager

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_name: str
    input_shape: Tuple[int, int]
    total_runs: int
    tokens_per_sec: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    latency_std: float
    peak_vram_mb: Optional[float] = None
    cpu_only: bool = False
    timestamps: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_shape": self.input_shape,
            "total_runs": self.total_runs,
            "tokens_per_sec": self.tokens_per_sec,
            "latency_p50_ms": self.latency_p50,
            "latency_p95_ms": self.latency_p95,
            "latency_p99_ms": self.latency_p99,
            "latency_mean_ms": self.latency_mean,
            "latency_std_ms": self.latency_std,
            "peak_vram_mb": self.peak_vram_mb,
            "cpu_only": self.cpu_only
        }

    def summary(self) -> str:
        vram_str = f"{self.peak_vram_mb:.1f} MB" if self.peak_vram_mb else "N/A"
        return (
            f"Model: {self.model_name}\n"
            f"Input: {self.input_shape[0]}x{self.input_shape[1]} | Runs: {self.total_runs}\n"
            f"Tokens/sec: {self.tokens_per_sec:.2f} | Latency P50: {self.latency_p50:.2f}ms | P95: {self.latency_p95:.2f}ms | P99: {self.latency_p99:.2f}ms\n"
            f"Peak VRAM: {vram_str} | CPU-only: {self.cpu_only}"
        )


def get_gpu_memory_mb() -> Optional[float]:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return None


def reset_gpu_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_memory_mb() -> Optional[float]:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None


def run_inference(model: nn.Module, input_tensor: torch.Tensor, device: str = "cpu") -> Tuple[torch.Tensor, float]:
    model.eval()
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.perf_counter()
    
    latency_sec = end_time - start_time
    if isinstance(output, tuple):
        output = output[0]
    output = output.cpu()
    
    return output, latency_sec


def benchmark_model(model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100, device: Optional[str] = None, use_rich: bool = True) -> BenchmarkResult:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cpu_only = device == "cpu"
    _ = run_inference(model, input_tensor, device)
    
    if not cpu_only:
        reset_gpu_memory_stats()
    
    latencies: List[float] = []
    timestamps: List[float] = []
    
    console = Console()
    live = None
    if use_rich and RICH_AVAILABLE:
        live = Live(get_refresh_table(model, latencies, cpu_only), console=console, refresh_per_second=4)
        live.start()
    
    try:
        for i in range(num_runs):
            _, latency_sec = run_inference(model, input_tensor, device)
            latencies.append(latency_sec)
            timestamps.append(time.time())
            
            if live and RICH_AVAILABLE:
                live.update(get_refresh_table(model, latencies, cpu_only))
            if i < num_runs - 1 and use_rich:
                time.sleep(0.01)
    finally:
        if live and RICH_AVAILABLE:
            live.stop()
    
    latencies_ms = [l * 1000 for l in latencies]
    sorted_latencies = sorted(latencies_ms)
    
    p50 = sorted_latencies[len(sorted_latencies) // 2]
    p95_idx = int(len(sorted_latencies) * 0.95)
    p99_idx = int(len(sorted_latencies) * 0.99)
    
    latency_p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
    latency_p99 = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
    
    latency_mean = statistics.mean(latencies_ms)
    latency_std = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    
    batch_size, seq_length = input_tensor.shape[:2]
    total_tokens = batch_size * seq_length * num_runs
    total_time = sum(latencies)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    
    peak_vram = get_peak_gpu_memory_mb() if not cpu_only else None
    
    return BenchmarkResult(
        model_name=model.__class__.__name__,
        input_shape=(batch_size, seq_length),
        total_runs=num_runs,
        tokens_per_sec=tokens_per_sec,
        latency_p50=p50,
        latency_p95=latency_p95,
        latency_p99=latency_p99,
        latency_mean=latency_mean,
        latency_std=latency_std,
        peak_vram_mb=peak_vram,
        cpu_only=cpu_only,
        timestamps=timestamps
    )


def get_refresh_table(model: nn.Module, latencies: List[float], cpu_only: bool) -> Panel:
    if not latencies:
        return Panel("Starting benchmark...", title="🏥 Operating Theatre")
    
    current_latency = latencies[-1] * 1000
    avg_latency = statistics.mean(latencies) * 1000
    runs_completed = len(latencies)
    
    table = Table(title="🏥 Operating Theatre - Patient Vitals")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model", model.__class__.__name__)
    table.add_row("Device", "CPU" if cpu_only else "CUDA")
    table.add_row("Runs Completed", str(runs_completed))
    table.add_row("Current Latency", f"{current_latency:.2f} ms")
    table.add_row("Average Latency", f"{avg_latency:.2f} ms")
    
    if not cpu_only and torch.cuda.is_available():
        current_vram = get_gpu_memory_mb()
        if current_vram:
            table.add_row("Current VRAM", f"{current_vram:.1f} MB")
    
    return Panel(table, title="🏥 Operating Theatre")


def compare_benchmarks(before: BenchmarkResult, after: BenchmarkResult) -> Dict[str, Any]:
    speedup = after.tokens_per_sec / before.tokens_per_sec if before.tokens_per_sec > 0 else 1.0
    latency_reduction = (before.latency_p50 - after.latency_p50) / before.latency_p50 if before.latency_p50 > 0 else 0.0
    
    return {
        "speedup_factor": speedup,
        "speedup_percent": (speedup - 1) * 100,
        "latency_reduction_percent": latency_reduction * 100,
        "vram_change_mb": (after.peak_vram_mb or 0) - (before.peak_vram_mb or 0)
    }