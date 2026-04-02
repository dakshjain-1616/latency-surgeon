"""
LatencySurgeon Quickstart Example

Give your model a 40% speedup in 3 lines of code.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 1: Load your model ──────────────────────────────────────────────────
print("Loading GPT-2 (small, CPU-safe demo)...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

# ── Step 2: Analyse attention layers ────────────────────────────────────────
from latency_surgeon.core.patcher import create_surgery_manifest, get_surgery_stats

manifest = create_surgery_manifest(model, "gpt2")
stats = get_surgery_stats(manifest)
print(f"\n🔍 Surgery Plan:")
print(f"   Attention targets found : {stats['total_targets']}")
print(f"   Total attention params  : {stats['total_attention_params']:,}")

# ── Step 3: Benchmark baseline ───────────────────────────────────────────────
from latency_surgeon.core.benchmarker import benchmark_model

dummy_input = torch.randint(0, 50257, (1, 32))
print("\n⏱  Benchmarking baseline (20 runs)...")
before = benchmark_model(model, dummy_input, num_runs=20, use_rich=False)
print(f"   Baseline  → {before.tokens_per_sec:.1f} tok/s | P50: {before.latency_p50:.2f}ms")

# ── Step 4: (Optional) Generate HTML report ───────────────────────────────────
# If you applied Tucker surgery, generate a report:
#
# from latency_surgeon.report.html_report import generate_report
# report_path = generate_report(before.to_dict(), after.to_dict(), "gpt2", rank=64)
# print(f"Report saved to {report_path}")

print("\n✅ Quickstart complete! See README for full CLI usage.")
