"""
Rank tuner: binary search over Tucker rank ratios using perplexity delta threshold.

Calibrates on wikitext-2 to find the highest compression that keeps
perplexity degradation within the specified threshold.
"""

from __future__ import annotations
from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
import math


def compute_perplexity(model: nn.Module, tokenizer, texts: list[str], device: str = "cpu") -> float:
    """Compute perplexity of model on a list of texts.

    Args:
        model: The language model.
        tokenizer: HuggingFace tokenizer.
        texts: List of text strings to evaluate on.
        device: Compute device.

    Returns:
        Perplexity score (lower is better).
    """
    model.eval()
    model = model.to(device)
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]

            outputs = model(**inputs, labels=input_ids)
            loss = outputs.loss
            n_tokens = input_ids.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def load_wikitext2_sample(n_samples: int = 50) -> list[str]:
    """Load a small sample from wikitext-2 for calibration.

    Args:
        n_samples: Number of samples to load.

    Returns:
        List of text strings.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [row["text"] for row in dataset if len(row["text"].strip()) > 100]
        return texts[:n_samples]
    except Exception:
        # Fallback: synthetic calibration text
        return [
            "The transformer architecture has revolutionized natural language processing. "
            "Self-attention mechanisms allow models to weigh the importance of different tokens. "
        ] * n_samples


class RankTuner:
    """Binary search over Tucker rank ratios to find optimal compression.

    Finds the highest compression rank that keeps perplexity delta
    below the specified threshold.

    Args:
        ppl_delta_threshold: Maximum allowed perplexity increase (default 0.05 = 5%).
        min_rank_ratio: Minimum rank ratio to consider (default 0.1).
        max_rank_ratio: Maximum rank ratio to consider (default 0.9).
        n_iterations: Number of binary search iterations (default 8).
        device: Compute device.
    """

    def __init__(
        self,
        ppl_delta_threshold: float = 0.05,
        min_rank_ratio: float = 0.1,
        max_rank_ratio: float = 0.9,
        n_iterations: int = 8,
        device: str = "cpu",
    ):
        self.ppl_delta_threshold = ppl_delta_threshold
        self.min_rank_ratio = min_rank_ratio
        self.max_rank_ratio = max_rank_ratio
        self.n_iterations = n_iterations
        self.device = device
        self.search_history: list[dict] = []

    def tune(
        self,
        model_factory: Callable[[float], nn.Module],
        tokenizer,
        calibration_texts: Optional[list[str]] = None,
    ) -> Tuple[float, float, float]:
        """Find optimal rank ratio via binary search.

        Args:
            model_factory: Callable that takes rank_ratio (0-1) and returns a patched model.
            tokenizer: HuggingFace tokenizer.
            calibration_texts: Texts to evaluate on. Defaults to wikitext-2 sample.

        Returns:
            Tuple of (best_rank_ratio, baseline_ppl, best_ppl).
        """
        if calibration_texts is None:
            calibration_texts = load_wikitext2_sample()

        # Baseline perplexity (rank_ratio = 1.0 means no compression)
        baseline_model = model_factory(1.0)
        baseline_ppl = compute_perplexity(baseline_model, tokenizer, calibration_texts, self.device)
        del baseline_model

        low = self.min_rank_ratio
        high = self.max_rank_ratio
        best_ratio = high
        best_ppl = baseline_ppl

        for i in range(self.n_iterations):
            mid = (low + high) / 2.0
            candidate_model = model_factory(mid)
            candidate_ppl = compute_perplexity(candidate_model, tokenizer, calibration_texts, self.device)
            ppl_delta = (candidate_ppl - baseline_ppl) / baseline_ppl
            del candidate_model

            self.search_history.append({
                "iteration": i,
                "rank_ratio": mid,
                "ppl": candidate_ppl,
                "ppl_delta": ppl_delta,
                "within_threshold": ppl_delta <= self.ppl_delta_threshold,
            })

            if ppl_delta <= self.ppl_delta_threshold:
                # Compression is acceptable — try going lower (more compression)
                best_ratio = mid
                best_ppl = candidate_ppl
                high = mid
            else:
                # Too much degradation — reduce compression
                low = mid

        return best_ratio, baseline_ppl, best_ppl

    def get_report(self) -> str:
        """Get a human-readable report of the search history."""
        lines = ["Rank Tuning Search History", "=" * 40]
        for entry in self.search_history:
            status = "✅" if entry["within_threshold"] else "❌"
            lines.append(
                f"{status} Iter {entry['iteration']:2d} | rank_ratio={entry['rank_ratio']:.3f} "
                f"| ppl={entry['ppl']:.4f} | delta={entry['ppl_delta']*100:.2f}%"
            )
        return "\n".join(lines)


def auto_tune_rank(
    model: nn.Module,
    tokenizer,
    patch_fn: Callable[[nn.Module, int], nn.Module],
    base_rank: int = 256,
    ppl_delta_threshold: float = 0.05,
    device: str = "cpu",
) -> int:
    """Convenience function to auto-tune Tucker rank for a given model.

    Args:
        model: Base model to optimize.
        tokenizer: HuggingFace tokenizer.
        patch_fn: Function(model, rank) -> patched model.
        base_rank: Maximum rank to use (will search below this).
        ppl_delta_threshold: Maximum allowed perplexity delta.
        device: Compute device.

    Returns:
        Optimal rank (integer).
    """
    import copy

    def model_factory(rank_ratio: float) -> nn.Module:
        rank = max(1, int(base_rank * rank_ratio))
        cloned = copy.deepcopy(model)
        return patch_fn(cloned, rank)

    tuner = RankTuner(ppl_delta_threshold=ppl_delta_threshold, device=device)
    best_ratio, baseline_ppl, best_ppl = tuner.tune(model_factory, tokenizer)
    optimal_rank = max(1, int(base_rank * best_ratio))

    print(tuner.get_report())
    print(f"\nBaseline PPL: {baseline_ppl:.4f} → Best PPL: {best_ppl:.4f} at rank {optimal_rank}")
    return optimal_rank
