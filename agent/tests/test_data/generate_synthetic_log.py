"""Generate synthetic training metrics for end-to-end demo.

Simulates a policy_collapse scenario:
- reward rises steadily, then suddenly drops and stays low
- KL divergence spikes at the collapse point
- entropy collapses to near-zero
- grad_norm briefly spikes

Usage:
    python tests/test_data/generate_synthetic_log.py --scenario policy_collapse
    python tests/test_data/generate_synthetic_log.py --scenario healthy
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def _policy_collapse() -> Dict[str, List[Tuple[int, float]]]:
    """Reward rises, then collapses near end; KL spikes; entropy collapses.

    Note: log_analyzer is snapshot-based (analyzes the last 50 points).
    The collapse is timed so it falls WITHIN the analysis window when
    analyze() is called at the end of the series.
    """
    metrics: Dict[str, List[Tuple[int, float]]] = {}
    reward, kl, entropy, grad = [], [], [], []
    collapse_step = 1800
    for step in range(0, 2000, 10):
        if step < collapse_step:
            r = 0.5 * step / 50 + math.sin(step / 50) * 0.3
            k = 0.005 + abs(math.sin(step / 100)) * 0.003
            e = 1.5 - step * 0.0002  # slow decline
            g = 1.0 + abs(math.sin(step / 30)) * 0.5
        else:
            t = (step - collapse_step) / 10
            r = max(0.0, 10.0 * math.exp(-t * 0.5) - 1.0)
            k = 0.08 + abs(math.sin(t)) * 0.04
            e = max(0.05, 1.1 - t * 0.15)  # collapse to <0.1 of initial
            g = 5.0 + abs(math.sin(t * 2)) * 3.0
        reward.append((step, r))
        kl.append((step, k))
        entropy.append((step, e))
        grad.append((step, g))
    metrics["Train/mean_reward"] = reward
    metrics["Train/kl_divergence"] = kl
    metrics["Train/entropy"] = entropy
    metrics["Loss/grad_norm"] = grad
    return metrics


def _healthy() -> Dict[str, List[Tuple[int, float]]]:
    metrics: Dict[str, List[Tuple[int, float]]] = {}
    reward, kl, entropy, grad = [], [], [], []
    for step in range(0, 2000, 10):
        r = 0.5 * step / 50 * (1 - math.exp(-step / 200)) + math.sin(step / 50) * 0.2
        k = 0.005 + abs(math.sin(step / 100)) * 0.002
        e = 1.5 - step * 0.0005
        g = 1.0 + abs(math.sin(step / 30)) * 0.3
        reward.append((step, r))
        kl.append((step, k))
        entropy.append((step, e))
        grad.append((step, g))
    metrics["Train/mean_reward"] = reward
    metrics["Train/kl_divergence"] = kl
    metrics["Train/entropy"] = entropy
    metrics["Loss/grad_norm"] = grad
    return metrics


def _gradient_explosion() -> Dict[str, List[Tuple[int, float]]]:
    metrics: Dict[str, List[Tuple[int, float]]] = {}
    grad, vloss, reward = [], [], []
    for step in range(0, 1500, 10):
        if step < 800:
            g = 1.0 + abs(math.sin(step / 30)) * 0.3
            vl = 0.5 + math.sin(step / 50) * 0.1
            r = 0.3 * step / 50
        else:
            t = (step - 800) / 50
            g = 1.0 * (1 + t ** 3)
            vl = 0.5 + t ** 2 * 10
            if step > 1200:
                vl = float("nan")
            r = max(0, 4.8 - t * 0.5)
        grad.append((step, g))
        vloss.append((step, vl))
        reward.append((step, r))
    metrics["Loss/grad_norm"] = grad
    metrics["Loss/value_loss"] = vloss
    metrics["Train/mean_reward"] = reward
    return metrics


SCENARIOS = {
    "policy_collapse": _policy_collapse,
    "healthy": _healthy,
    "gradient_explosion": _gradient_explosion,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="policy_collapse")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metrics = SCENARIOS[args.scenario]()
    out_dir = args.output or Path(f"tests/test_data/synthetic_{args.scenario}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"

    serializable = {k: [{"step": s, "value": v} for s, v in series]
                    for k, series in metrics.items()}
    out_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Wrote {sum(len(s) for s in metrics.values())} points across "
          f"{len(metrics)} metrics to {out_file}")


if __name__ == "__main__":
    main()
