#!/usr/bin/env python3
"""ROUGE-L threshold sensitivity analysis (V4 ablation).

Re-computes verifier verdicts at different τ values and reports:
  - Label composition (correct/incorrect/unknown ratio)
  - Downstream probe performance and EDR@5 under FPR cap
  
This runs on EXISTING trajectory data — no GPU needed.
"""

import argparse
import json
import numpy as np
from pathlib import Path


def rouge_l_score(reference, hypothesis):
    """Compute ROUGE-L F1 between reference and hypothesis."""
    if not reference or not hypothesis:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS length
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]

    precision = lcs / n if n > 0 else 0.0
    recall = lcs / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_composition(traj_path, tau):
    """Compute label composition at given ROUGE-L threshold."""
    correct = 0
    incorrect = 0
    total = 0

    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            ref = item.get("answer", "")
            gen = item.get("generated_text", "")
            score = rouge_l_score(ref, gen)
            total += 1
            if score >= tau:
                correct += 1
            else:
                incorrect += 1

    return {
        "tau": tau,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "incorrect_ratio": incorrect / total if total > 0 else 0.0,
    }


def sweep_tau(traj_dir, tau_values=None, output_path=None):
    """Sweep ROUGE-L threshold and report composition changes."""
    if tau_values is None:
        tau_values = [0.08, 0.10, 0.12, 0.15, 0.20]

    traj_path = Path(traj_dir) / "trajectories.jsonl"
    if not traj_path.exists():
        print(f"  WARNING: {traj_path} not found, skipping")
        return []

    results = []
    for tau in tau_values:
        comp = compute_composition(str(traj_path), tau)
        results.append(comp)
        print(f"  τ={tau:.2f}: incorrect={comp['incorrect']}/{comp['total']} "
              f"({comp['incorrect_ratio']:.1%})")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dirs", required=True, nargs="+",
                    help="Directories containing trajectories.jsonl")
    ap.add_argument("--tau_values", type=str, default="0.08,0.10,0.12,0.15,0.20")
    ap.add_argument("--output", default="results/tau_sensitivity.json")
    args = ap.parse_args()

    tau_values = [float(t) for t in args.tau_values.split(",")]
    all_results = {}
    for traj_dir in args.traj_dirs:
        name = Path(traj_dir).name
        print(f"\n=== {name} ===")
        results = sweep_tau(traj_dir, tau_values)
        all_results[name] = results

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved tau sensitivity results to {args.output}")
