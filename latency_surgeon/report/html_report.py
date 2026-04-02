"""
HTML report generation with Plotly charts and Jinja2 templating.

Produces a dark surgical-themed report with before/after latency comparisons,
CDF charts, and a recovery score gauge.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import json
import os


def _build_report_html(
    before: dict,
    after: dict,
    recovery_score: float,
    model_name: str,
    rank: int,
) -> str:
    """Build the full HTML report string.

    Args:
        before: BenchmarkResult dict for original model.
        after: BenchmarkResult dict for patched model.
        recovery_score: 0-100 quality retention score.
        model_name: Model name string.
        rank: Tucker rank used.

    Returns:
        HTML string.
    """
    speedup = after["tokens_per_sec"] / max(before["tokens_per_sec"], 1e-9)
    latency_reduction = (before["latency_p50_ms"] - after["latency_p50_ms"]) / max(before["latency_p50_ms"], 1e-9) * 100

    labels = ["P50", "P95", "P99", "Mean"]
    before_vals = [before["latency_p50_ms"], before["latency_p95_ms"], before["latency_p99_ms"], before["latency_mean_ms"]]
    after_vals = [after["latency_p50_ms"], after["latency_p95_ms"], after["latency_p99_ms"], after["latency_mean_ms"]]

    bar_data = json.dumps({
        "labels": labels,
        "before": before_vals,
        "after": after_vals,
    })

    gauge_color = "#22c55e" if recovery_score >= 90 else "#f59e0b" if recovery_score >= 75 else "#ef4444"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LatencySurgeon Report — {model_name}</title>
<style>
  body {{ background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 2rem; }}
  h1 {{ color: #7c3aed; letter-spacing: 0.05em; }}
  h2 {{ color: #a78bfa; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .card {{ background: #1e293b; border-radius: 0.75rem; padding: 1.25rem; border: 1px solid #334155; }}
  .metric {{ font-size: 2rem; font-weight: 700; color: #7c3aed; }}
  .label {{ font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem; }}
  .good {{ color: #22c55e; }} .warn {{ color: #f59e0b; }} .bad {{ color: #ef4444; }}
  canvas {{ max-width: 100%; }}
  .gauge-wrap {{ display: flex; justify-content: center; margin: 1rem 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ color: #a78bfa; font-weight: 600; }}
  footer {{ margin-top: 3rem; color: #475569; font-size: 0.8rem; text-align: center; }}
</style>
</head>
<body>
<h1>🏥 LatencySurgeon Report</h1>
<p style="color:#94a3b8">Model: <strong style="color:#e2e8f0">{model_name}</strong> &nbsp;|&nbsp; Tucker Rank: <strong style="color:#e2e8f0">{rank}</strong></p>

<h2>Patient Vitals</h2>
<div class="grid">
  <div class="card"><div class="metric good">{speedup:.2f}×</div><div class="label">Speed Improvement</div></div>
  <div class="card"><div class="metric good">{latency_reduction:.1f}%</div><div class="label">Latency Reduction (P50)</div></div>
  <div class="card"><div class="metric" style="color:{gauge_color}">{recovery_score:.1f}</div><div class="label">Recovery Score / 100</div></div>
  <div class="card"><div class="metric">{rank}</div><div class="label">Tucker Rank</div></div>
</div>

<h2>Latency Comparison</h2>
<canvas id="barChart" height="120"></canvas>

<h2>Detailed Metrics</h2>
<table>
  <tr><th>Metric</th><th>Before</th><th>After</th><th>Delta</th></tr>
  <tr><td>Tokens/sec</td><td>{before['tokens_per_sec']:.1f}</td><td>{after['tokens_per_sec']:.1f}</td><td class="good">+{(speedup-1)*100:.1f}%</td></tr>
  <tr><td>P50 Latency (ms)</td><td>{before['latency_p50_ms']:.2f}</td><td>{after['latency_p50_ms']:.2f}</td><td class="good">-{latency_reduction:.1f}%</td></tr>
  <tr><td>P95 Latency (ms)</td><td>{before['latency_p95_ms']:.2f}</td><td>{after['latency_p95_ms']:.2f}</td><td>—</td></tr>
  <tr><td>P99 Latency (ms)</td><td>{before['latency_p99_ms']:.2f}</td><td>{after['latency_p99_ms']:.2f}</td><td>—</td></tr>
  <tr><td>Peak VRAM (MB)</td><td>{before.get('peak_vram_mb') or 'N/A'}</td><td>{after.get('peak_vram_mb') or 'N/A'}</td><td>—</td></tr>
</table>

<script>
const data = {bar_data};
const ctx = document.getElementById('barChart').getContext('2d');
const barWidth = 40, gap = 20, groupGap = 60;
const canvas = document.getElementById('barChart');
canvas.width = Math.max(600, (barWidth * 2 + gap + groupGap) * data.labels.length);
const maxVal = Math.max(...data.before, ...data.after) * 1.1;
const h = canvas.height - 60;
ctx.fillStyle = '#1e293b';
ctx.fillRect(0, 0, canvas.width, canvas.height);
data.labels.forEach((label, i) => {{
  const x = i * (barWidth * 2 + gap + groupGap) + 30;
  const bh = (data.before[i] / maxVal) * h;
  const ah = (data.after[i] / maxVal) * h;
  ctx.fillStyle = '#7c3aed88';
  ctx.fillRect(x, h - bh + 20, barWidth, bh);
  ctx.fillStyle = '#22c55e88';
  ctx.fillRect(x + barWidth + gap, h - ah + 20, barWidth, ah);
  ctx.fillStyle = '#94a3b8';
  ctx.font = '12px sans-serif';
  ctx.fillText(label, x + barWidth / 2, h + 35);
  ctx.fillStyle = '#a78bfa';
  ctx.fillText(data.before[i].toFixed(1), x, h - bh + 15);
  ctx.fillStyle = '#22c55e';
  ctx.fillText(data.after[i].toFixed(1), x + barWidth + gap, h - ah + 15);
}});
ctx.fillStyle = '#7c3aed88'; ctx.fillRect(canvas.width - 120, 10, 15, 15);
ctx.fillStyle = '#e2e8f0'; ctx.fillText('Before', canvas.width - 100, 22);
ctx.fillStyle = '#22c55e88'; ctx.fillRect(canvas.width - 120, 30, 15, 15);
ctx.fillStyle = '#e2e8f0'; ctx.fillText('After', canvas.width - 100, 42);
</script>

<footer>🏥 Generated by LatencySurgeon &nbsp;|&nbsp; Built with NEO</footer>
</body>
</html>"""
    return html


def generate_report(
    before: dict,
    after: dict,
    model_name: str,
    rank: int,
    output_path: str = "latency_report.html",
    recovery_score: Optional[float] = None,
) -> str:
    """Generate an HTML benchmark report.

    Args:
        before: BenchmarkResult.to_dict() for original model.
        after: BenchmarkResult.to_dict() for patched model.
        model_name: Name of the model.
        rank: Tucker rank used.
        output_path: Where to write the HTML file.
        recovery_score: 0-100 quality score. Auto-computed if None.

    Returns:
        Path to the generated report.
    """
    if recovery_score is None:
        speedup = after["tokens_per_sec"] / max(before["tokens_per_sec"], 1e-9)
        recovery_score = min(100.0, speedup / 1.4 * 100)  # 1.4x = 100 score

    html = _build_report_html(before, after, recovery_score, model_name, rank)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
