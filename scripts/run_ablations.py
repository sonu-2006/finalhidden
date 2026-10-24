#!/usr/bin/env python3
"""V10 Comprehensive Ablation: Binary vs Ternary + History Pooling.

Self-contained script (no external imports from src/) for SCP independence.
Fixes critical ROUGE-L bug: uses word-level (not char-level) ROUGE-L for labeling.

Ablations:
  1. Binary vs Ternary: 3-state vs 2-state probe (conservative + aggressive)
  2. History pooling: H={1, 4, 8}

Rule-gated detection sweep matching V7 paper:
  theta x persistence x ma_window x min_tokens = 1800 configs

Datasets: Endoscopy, PubMedQA, MedHallu (Qwen only)
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

# ============================================================
# ROUGE-L (WORD-LEVEL) — matches tau_sweep, NOT char-level
# ============================================================

def rouge_l_word(reference, hypothesis):
    """Word-level ROUGE-L F1. Matches ablation_tau_sweep.py implementation."""
    if not reference or not hypothesis:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        ri = ref_tokens[i - 1]
        for j in range(1, n + 1):
            if ri == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    lcs = prev[-1]
    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================
# State annotation (window mode)
# ============================================================

def annotate_trajectory(gen_text, ref_text, tau=0.12, window=10, tail=3, n_tokens=256):
    """Annotate one trajectory with word-level ROUGE-L verdict + window states."""
    score = rouge_l_word(ref_text, gen_text)
    verdict = "correct" if score >= tau else "incorrect"

    if verdict == "correct":
        states = ["Safe"] * n_tokens
    else:
        states = ["Safe"] * n_tokens
        # Hallucinating tail
        for j in range(max(0, n_tokens - tail), n_tokens):
            states[j] = "Hallucinating"
        # AtRisk window before tail
        for j in range(max(0, n_tokens - tail - window), max(0, n_tokens - tail)):
            states[j] = "AtRisk"

    return verdict, states, score


# ============================================================
# Data loading
# ============================================================

def load_trajectories_and_annotate(traj_jsonl, hidden_base_dir, layer_idx=20, tau=0.12):
    """Load trajectories, re-annotate with word-level ROUGE-L, load hidden states."""
    trajectories = []

    with open(traj_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            item = json.loads(line)
            gen_text = item.get("generated_text", "")
            ref_text = item.get("answer", "")

            # Resolve hidden path
            hpath = item["hidden_path"]
            if not os.path.isabs(hpath):
                hpath = os.path.join(hidden_base_dir, hpath)
            if not os.path.exists(hpath):
                # Try stripping prefix
                for prefix in ["results/", "../results/"]:
                    if prefix in item["hidden_path"]:
                        rel = item["hidden_path"][item["hidden_path"].find("results/"):]
                        candidate = os.path.join(hidden_base_dir, rel)
                        if os.path.exists(candidate):
                            hpath = candidate
                            break

            if not os.path.exists(hpath):
                print(f"  WARNING: missing {hpath}, skipping line {line_idx}")
                continue

            data = np.load(hpath)

            # Select layer
            if "hidden_layers" in data and layer_idx is not None:
                layer_ids = data["layer_ids"]
                idx_pos = None
                for k, lid in enumerate(layer_ids):
                    if lid == layer_idx:
                        idx_pos = k
                        break
                if idx_pos is not None:
                    hidden = data["hidden_layers"][:, idx_pos, :].astype(np.float32)
                else:
                    hidden = data["hidden"].astype(np.float32)
            else:
                hidden = data["hidden"].astype(np.float32)

            n_tokens = len(hidden)
            if n_tokens == 0:
                continue

            verdict, states, rouge_score = annotate_trajectory(
                gen_text, ref_text, tau=tau, window=10, tail=3, n_tokens=n_tokens
            )

            # Find t_error
            t_error = None
            for t, s in enumerate(states):
                if s == "Hallucinating":
                    t_error = t
                    break

            trajectories.append({
                "hidden": hidden,
                "states": states,
                "verdict": verdict,
                "t_error": t_error,
                "n_tokens": n_tokens,
                "rouge_score": rouge_score,
            })

    return trajectories


def apply_history_pooling(hidden, history):
    """Apply mean pooling over last H steps."""
    if history <= 1:
        return hidden
    T, D = hidden.shape
    pooled = np.zeros_like(hidden)
    for t in range(T):
        start = max(0, t - history + 1)
        pooled[t] = hidden[start:t+1].mean(axis=0)
    return pooled


# ============================================================
# Probe model
# ============================================================

class MLPProbe(nn.Module):
    def __init__(self, hidden_dim=4096, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.net(x)


STATE2ID_3 = {"Safe": 0, "AtRisk": 1, "Hallucinating": 2}
STATE2ID_2_CONSV = {"Safe": 0, "Hallucinating": 1}  # AtRisk→Safe
STATE2ID_2_AGGR = {"Safe": 0, "Hallucinating": 1}   # AtRisk→Hallucinating


def remap_labels(states, mode="conservative"):
    """Remap 3-state labels to 2-state."""
    remapped = []
    for s in states:
        if s == "AtRisk":
            remapped.append("Safe" if mode == "conservative" else "Hallucinating")
        else:
            remapped.append(s)
    return remapped


# ============================================================
# Training
# ============================================================

def train_probe(trajectories, state2id, n_classes, history=8, epochs=3, lr=1e-3, batch_size=256):
    """Train probe and return per-trajectory risk scores."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare data with history pooling
    X_all = []
    y_all = []
    for traj in trajectories:
        hidden = apply_history_pooling(traj["hidden"], history)
        X_all.append(hidden)
        y_all.append([state2id[s] for s in traj["states"]])

    # Flatten for training
    X_flat = np.concatenate(X_all, axis=0)
    y_flat = np.concatenate([np.array(y, dtype=np.int64) for y in y_all])

    hidden_dim = X_flat.shape[1]

    # Class weights (inverse frequency)
    counts = Counter(y_flat.tolist())
    total = len(y_flat)
    weights = torch.zeros(n_classes)
    for c in range(n_classes):
        weights[c] = total / (n_classes * counts.get(c, 1))
    weights = weights.to(device)

    model = MLPProbe(hidden_dim=hidden_dim, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weights)

    X_t = torch.tensor(X_flat, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_flat, dtype=torch.long).to(device)

    # Train
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_t))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_t), batch_size):
            idx = perm[start:start + batch_size]
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    # Evaluate: compute risk scores per trajectory
    model.eval()
    safe_idx = state2id["Safe"]
    risk_sequences = []

    for X_seq in X_all:
        X_chunk = torch.tensor(X_seq, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(X_chunk), dim=-1).cpu().numpy()
        risk = 1.0 - probs[:, safe_idx]
        risk_sequences.append(risk)

    # Classification metrics
    model.eval()
    with torch.no_grad():
        all_preds = torch.argmax(model(X_t), dim=-1).cpu().numpy()

    from sklearn.metrics import f1_score, accuracy_score
    try:
        macro_f1 = f1_score(y_flat, all_preds, average="macro", zero_division=0)
        per_class_f1 = f1_score(y_flat, all_preds, average=None, zero_division=0)
        acc = accuracy_score(y_flat, all_preds)
    except Exception:
        macro_f1 = 0.0
        per_class_f1 = [0.0] * n_classes
        acc = 0.0

    clf_metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(x) for x in per_class_f1],
        "class_counts": {str(k): v for k, v in counts.items()},
    }

    return risk_sequences, clf_metrics


# ============================================================
# Rule-gated detection sweep (matching V7 paper)
# ============================================================

def rule_gated_detect(risk, theta, persistence, ma_window, min_tokens):
    """Apply rule-gated detection: MA smoothing + threshold + persistence + min_tokens."""
    T = len(risk)

    # Moving average
    if ma_window > 1 and T >= ma_window:
        cumsum = np.cumsum(np.insert(risk, 0, 0.0))
        smoothed = (cumsum[ma_window:] - cumsum[:-ma_window]) / float(ma_window)
        pad = np.full(ma_window - 1, smoothed[0] if len(smoothed) > 0 else 0.0)
        risk_smooth = np.concatenate([pad, smoothed])
    else:
        risk_smooth = risk

    # Threshold + persistence + min_tokens
    cnt = 0
    for t in range(T):
        if t < min_tokens:
            cnt = 0
            continue
        if risk_smooth[t] > theta:
            cnt += 1
            if cnt >= persistence:
                return t - persistence + 1  # trigger time
        else:
            cnt = 0

    return None  # no trigger


def evaluate_detection_sweep(risk_sequences, trajectories,
                              theta_list, persistence_list, ma_window_list, min_tokens_list,
                              k=5):
    """Run full rule-gated sweep and return best EDR@5 under FPR caps."""

    fpr_caps = [0.05, 0.10, 0.20, 0.30]

    # Pre-compute error/no-error indices
    error_indices = [i for i, t in enumerate(trajectories) if t["t_error"] is not None]
    noerror_indices = [i for i, t in enumerate(trajectories) if t["t_error"] is None]
    n_error = len(error_indices)
    n_noerror = len(noerror_indices)

    best_per_cap = {str(cap): {"edr_at_k": 0.0, "fpr": 0.0, "lead_time": 0.0,
                                 "theta": 0.0, "p": 0, "w": 0, "m": 0,
                                 "note": "no valid point"} for cap in fpr_caps}

    total_configs = len(theta_list) * len(persistence_list) * len(ma_window_list) * len(min_tokens_list)
    config_idx = 0

    for theta in theta_list:
        for p in persistence_list:
            for w in ma_window_list:
                for m in min_tokens_list:
                    config_idx += 1

                    # Evaluate this config
                    detected = 0
                    lead_times = []
                    fp = 0

                    for i in error_indices:
                        t_star = rule_gated_detect(risk_sequences[i], theta, p, w, m)
                        t_error = trajectories[i]["t_error"]
                        if t_star is not None and t_star < t_error + k:
                            detected += 1
                            lead_times.append(t_error - t_star)

                    for i in noerror_indices:
                        t_star = rule_gated_detect(risk_sequences[i], theta, p, w, m)
                        if t_star is not None:
                            fp += 1

                    edr = detected / n_error if n_error > 0 else 0.0
                    fpr = fp / n_noerror if n_noerror > 0 else 0.0
                    lt = float(np.mean(lead_times)) if lead_times else 0.0

                    # Update best per cap
                    for cap in fpr_caps:
                        if fpr <= cap:
                            cap_key = str(cap)
                            if edr > best_per_cap[cap_key]["edr_at_k"] or \
                               (edr == best_per_cap[cap_key]["edr_at_k"] and fpr < best_per_cap[cap_key]["fpr"]):
                                best_per_cap[cap_key] = {
                                    "edr_at_k": edr, "fpr": fpr, "lead_time": lt,
                                    "theta": theta, "p": p, "w": w, "m": m,
                                    "k": k,
                                }

    return best_per_cap


# ============================================================
# Main ablation
# ============================================================

def run_ablation(dataset_name, traj_jsonl, hidden_base_dir, output_dir,
                 layer_idx=20, tau=0.12):
    """Run full ablation for one dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Trajectory: {traj_jsonl}")
    print(f"Layer: {layer_idx}, Tau: {tau}")
    print(f"{'='*60}")

    # Load and annotate
    t0 = time.time()
    trajectories = load_trajectories_and_annotate(
        traj_jsonl, hidden_base_dir, layer_idx=layer_idx, tau=tau
    )
    print(f"Loaded {len(trajectories)} trajectories in {time.time()-t0:.1f}s")

    # Print verdict distribution
    verdicts = Counter(t["verdict"] for t in trajectories)
    print(f"Verdict distribution (word-level ROUGE-L τ={tau}): {dict(verdicts)}")

    n_incorrect = verdicts.get("incorrect", 0)
    n_correct = verdicts.get("correct", 0)
    print(f"Incorrect rate: {n_incorrect}/{len(trajectories)} = {n_incorrect/len(trajectories)*100:.1f}%")

    if n_incorrect == 0:
        print("WARNING: No incorrect trajectories! Cannot evaluate detection.")
        return {"error": "no incorrect trajectories"}

    # Detection sweep parameters (matching V7 paper: 50*4*3*3=1800 configs)
    theta_list = [round(x, 2) for x in np.arange(0.50, 1.00, 0.01)]  # 50 values
    persistence_list = [1, 2, 3, 5]
    ma_window_list = [1, 3, 5]
    min_tokens_list = [0, 32, 64]

    results = {}

    # ── Ablation 1: Binary vs Ternary ──
    print(f"\n--- Ablation 1: Binary vs Ternary ---")

    configs = [
        ("ternary",            STATE2ID_3,      3, None),
        ("binary_conservative", STATE2ID_2_CONSV, 2, "conservative"),
        ("binary_aggressive",   STATE2ID_2_AGGR,  2, "aggressive"),
    ]

    for name, state2id, n_classes, remap_mode in configs:
        print(f"\n  Config: {name}")

        # Remap labels if binary
        if remap_mode is not None:
            for traj in trajectories:
                traj["_states_remapped"] = remap_labels(traj["states"], remap_mode)
        else:
            for traj in trajectories:
                traj["_states_remapped"] = traj["states"]

        # Swap states for training
        orig_states = [traj["states"] for traj in trajectories]
        for traj in trajectories:
            traj["states"] = traj["_states_remapped"]

        # Flat label distribution
        flat_labels = [s for traj in trajectories for s in traj["states"]]
        print(f"  Label distribution: {Counter(flat_labels)}")

        t0 = time.time()
        risk_seqs, clf_metrics = train_probe(trajectories, state2id, n_classes, history=8)
        print(f"  Training: {time.time()-t0:.1f}s, Acc={clf_metrics['accuracy']:.3f}, MacroF1={clf_metrics['macro_f1']:.3f}")
        print(f"  Per-class F1: {clf_metrics['per_class_f1']}")

        t0 = time.time()
        best_per_cap = evaluate_detection_sweep(
            risk_seqs, trajectories,
            theta_list, persistence_list, ma_window_list, min_tokens_list
        )
        print(f"  Detection sweep: {time.time()-t0:.1f}s")

        for cap in ["0.05", "0.1", "0.2", "0.3"]:
            b = best_per_cap[cap]
            if b.get("note") == "no valid point":
                print(f"  FPR≤{cap}: no valid point")
            else:
                print(f"  FPR≤{cap}: EDR@5={b['edr_at_k']:.3f} FPR={b['fpr']:.3f} LT={b['lead_time']:.1f} θ={b['theta']:.2f} p={b['p']} w={b['w']} m={b['m']}")

        results[f"binary_ternary_{name}"] = {
            "classification": clf_metrics,
            "detection": best_per_cap,
        }

        # Restore original states
        for traj, orig in zip(trajectories, orig_states):
            traj["states"] = orig

    # ── Ablation 2: History Pooling H={1, 4, 8} ──
    print(f"\n--- Ablation 2: History Pooling ---")

    for H in [1, 4, 8]:
        print(f"\n  H={H}")

        t0 = time.time()
        risk_seqs, clf_metrics = train_probe(trajectories, STATE2ID_3, 3, history=H)
        print(f"  Training: {time.time()-t0:.1f}s, Acc={clf_metrics['accuracy']:.3f}, MacroF1={clf_metrics['macro_f1']:.3f}")
        print(f"  Per-class F1: {clf_metrics['per_class_f1']}")

        t0 = time.time()
        best_per_cap = evaluate_detection_sweep(
            risk_seqs, trajectories,
            theta_list, persistence_list, ma_window_list, min_tokens_list
        )
        print(f"  Detection sweep: {time.time()-t0:.1f}s")

        for cap in ["0.05", "0.1", "0.2", "0.3"]:
            b = best_per_cap[cap]
            if b.get("note") == "no valid point":
                print(f"  FPR≤{cap}: no valid point")
            else:
                print(f"  FPR≤{cap}: EDR@5={b['edr_at_k']:.3f} FPR={b['fpr']:.3f} LT={b['lead_time']:.1f}")

        results[f"history_H{H}"] = {
            "classification": clf_metrics,
            "detection": best_per_cap,
        }

    # ── Save results ──
    results["metadata"] = {
        "dataset": dataset_name,
        "n_trajectories": len(trajectories),
        "n_incorrect": n_incorrect,
        "n_correct": n_correct,
        "incorrect_rate": n_incorrect / len(trajectories),
        "tau": tau,
        "layer_idx": layer_idx,
        "rouge_type": "word_level",
        "sweep_configs": len(theta_list) * len(persistence_list) * len(ma_window_list) * len(min_tokens_list),
    }

    out_path = Path(output_dir) / f"v10_ablation_{dataset_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="V10 Comprehensive Ablation")
    parser.add_argument("--base_dir", default="/home/yzq/paper/Probe_HalluDynamics_B",
                        help="Project base directory")
    parser.add_argument("--output_dir", default="results/v10_ablations",
                        help="Output directory for results")
    parser.add_argument("--datasets", nargs="+",
                        default=["endoscopy", "pubmedqa", "medhallu"],
                        help="Datasets to run")
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.12)
    args = parser.parse_args()

    base = Path(args.base_dir)

    # Dataset configuration
    dataset_configs = {
        "endoscopy": {
            "traj_jsonl": str(base / "results/v9_endo_qwen/trajectories.jsonl"),
            "hidden_base": str(base),
        },
        "pubmedqa": {
            "traj_jsonl": str(base / "results/v9_pubmedqa_qwen_traj/trajectories.jsonl"),
            "hidden_base": str(base),
        },
        "medhallu": {
            "traj_jsonl": str(base / "results/v9_medhallu_qwen_traj/trajectories.jsonl"),
            "hidden_base": str(base),
        },
    }

    print("V10 Comprehensive Ablation")
    print(f"Base: {args.base_dir}")
    print(f"Datasets: {args.datasets}")
    print(f"Layer: {args.layer_idx}, Tau: {args.tau}")
    print(f"ROUGE-L: WORD-LEVEL (fixed)")
    print(f"Output: {args.output_dir}")

    all_results = {}
    for ds_name in args.datasets:
        if ds_name not in dataset_configs:
            print(f"WARNING: unknown dataset {ds_name}, skipping")
            continue

        cfg = dataset_configs[ds_name]
        if not os.path.exists(cfg["traj_jsonl"]):
            print(f"WARNING: {cfg['traj_jsonl']} not found, skipping {ds_name}")
            continue

        result = run_ablation(
            ds_name, cfg["traj_jsonl"], cfg["hidden_base"],
            str(base / args.output_dir),
            layer_idx=args.layer_idx, tau=args.tau,
        )
        all_results[ds_name] = result

    # Save combined results
    combined_path = Path(args.base_dir) / args.output_dir / "v10_ablation_combined.json"
    with combined_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
