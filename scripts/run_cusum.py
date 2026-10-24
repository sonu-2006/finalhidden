#!/usr/bin/env python3
"""CUSUM (Cumulative Sum Control Chart) trigger for hallucination detection.

Implements the one-sided upper CUSUM:
    S_0 = 0
    S_t = max(0, S_{t-1} + (r_t - mu_0 - k))
    alarm when S_t > h

Parameters:
    mu_0: in-control mean risk (estimated from Safe trajectories)
    k:    allowance (sensitivity parameter)
    h:    decision interval (alarm threshold)
"""

import argparse
import json
import numpy as np
from pathlib import Path


class CUSUMTrigger:
    def __init__(self, mu_0, k, h, min_tokens=10):
        self.mu_0 = mu_0
        self.k = k
        self.h = h
        self.min_tokens = min_tokens

    def detect(self, risk_sequence):
        """Return trigger time t* or None if no trigger."""
        S = 0.0
        for t, r_t in enumerate(risk_sequence):
            if t < self.min_tokens:
                continue
            S = max(0.0, S + (r_t - self.mu_0 - self.k))
            if S > self.h:
                return t
        return None

    def detect_with_trace(self, risk_sequence):
        """Return (trigger_time, cusum_trace) for visualization."""
        S = 0.0
        trace = []
        trigger = None
        for t, r_t in enumerate(risk_sequence):
            if t < self.min_tokens:
                trace.append(0.0)
                continue
            S = max(0.0, S + (r_t - self.mu_0 - self.k))
            trace.append(S)
            if trigger is None and S > self.h:
                trigger = t
        return trigger, trace


def estimate_mu_0(records):
    """Estimate in-control mean risk from no-error trajectories."""
    safe_risks = []
    for rec in records:
        if rec["t_error"] is None:  # no-error trajectory
            safe_risks.extend(rec["risk"].tolist())
    if not safe_risks:
        return 0.3  # fallback
    return float(np.mean(safe_risks))


def evaluate_cusum(records, k, h, min_tokens=10, fpr_caps=None):
    """Evaluate CUSUM trigger on records, report EDR@K and FPR.
    
    records: list of dicts with keys 'risk', 't_error', 'length'
    """
    if fpr_caps is None:
        fpr_caps = [0.05, 0.10, 0.20, 0.30]

    mu_0 = estimate_mu_0(records)
    trigger = CUSUMTrigger(mu_0=mu_0, k=k, h=h, min_tokens=min_tokens)

    total_with_error = 0
    total_no_error = 0
    false_positives = 0
    detections = {}  # k_val -> count
    lead_times = []

    for K_val in [5, 10, 20, 50]:
        detections[K_val] = 0

    for rec in records:
        t_star = trigger.detect(rec["risk"])
        t_error = rec["t_error"]

        if t_error is None:
            total_no_error += 1
            if t_star is not None:
                false_positives += 1
        else:
            total_with_error += 1
            if t_star is not None:
                lt = t_error - t_star
                lead_times.append(lt)
                for K_val in detections:
                    if t_star < t_error + K_val:
                        detections[K_val] += 1

    fpr = false_positives / total_no_error if total_no_error > 0 else 0.0
    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0

    results = {
        "mu_0": mu_0,
        "k": k,
        "h": h,
        "min_tokens": min_tokens,
        "fpr": fpr,
        "mean_lead_time": mean_lead,
        "total_with_error": total_with_error,
        "total_no_error": total_no_error,
        "false_positives": false_positives,
    }
    for K_val in [5, 10, 20, 50]:
        edr = detections[K_val] / total_with_error if total_with_error > 0 else 0.0
        results[f"edr_at_{K_val}"] = edr

    return results


def sweep_cusum(records, k_values=None, h_values=None, min_tokens=10):
    """Grid search over CUSUM parameters."""
    if k_values is None:
        k_values = [0.05, 0.10, 0.15]
    if h_values is None:
        h_values = [0.5, 1.0, 2.0, 3.0]

    all_results = []
    for k in k_values:
        for h in h_values:
            result = evaluate_cusum(records, k=k, h=h, min_tokens=min_tokens)
            all_results.append(result)
            print(f"  CUSUM k={k} h={h}: FPR={result['fpr']:.3f} EDR@5={result['edr_at_5']:.3f} "
                  f"LeadTime={result['mean_lead_time']:.1f}")
    return all_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_json", required=True,
                    help="JSON file with list of {risk:[], t_error:int|null, length:int}")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min_tokens", type=int, default=10)
    args = ap.parse_args()

    with open(args.records_json) as f:
        records = json.load(f)
    for rec in records:
        rec["risk"] = np.array(rec["risk"], dtype=np.float32)

    results = sweep_cusum(records, min_tokens=args.min_tokens)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} CUSUM results to {args.output}")
