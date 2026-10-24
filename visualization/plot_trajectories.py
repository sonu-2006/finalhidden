#!/usr/bin/env python3
"""Generate per-token risk trajectory figures (Fig 2) with per-model optimal θ."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.direction_b.state_classifier import MLP, STATE2ID
from src.common.hidden_utils import load_hidden_from_npz

# Per-model optimal operating points from rule-sweep stats
CONFIGS = {
    "endo_qwen": {
        "label": "Endoscopy × Qwen",
        "states_jsonl": ROOT / "results/b_suite_v8_lenfix300_qwen_rouge012_fast/states_risk_delta.jsonl",
        "clf_pt": ROOT / "results/b_suite_v8_lenfix300_qwen_rouge012_fast/clf_risk_delta_layerNone_h1.pt",
        "theta": 0.92,
        "persistence": 3,
        "ma_window": 5,
        "min_tokens": 0,
        "history": 1,
    },
    "endo_ministral": {
        "label": "Endoscopy × Llama-3.1",
        "states_jsonl": ROOT / "results/b_suite_v8_lenfix300_ministral_rouge012_fast/states_window10.jsonl",
        "clf_pt": ROOT / "results/b_suite_v8_lenfix300_ministral_rouge012_fast/clf_window10_layerNone_h8.pt",
        "theta": 0.99,
        "persistence": 5,
        "ma_window": 5,
        "min_tokens": 0,
        "history": 8,
    },
    "endo_biomistral": {
        "label": "Endoscopy × BioMistral",
        "states_jsonl": ROOT / "results/b_suite_v8_lenfix300_biomistral_rouge012_fast/states_window10.jsonl",
        "clf_pt": ROOT / "results/b_suite_v8_lenfix300_biomistral_rouge012_fast/clf_window10_layerNone_h8.pt",
        "theta": 0.93,
        "persistence": 5,
        "ma_window": 5,
        "min_tokens": 0,
        "history": 8,
    },
}

K = 5


def moving_average(arr, w):
    if len(arr) < w or w <= 1:
        return arr.copy()
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad = np.full(w - 1, out[0])
    return np.concatenate([pad, out])


def simulate_detection(risk, states, theta, persistence, min_tokens, ma_window, k):
    t_error = None
    for i, s in enumerate(states):
        if s == "Hallucinating":
            t_error = i
            break

    risk_ma = moving_average(risk, ma_window)
    consecutive = 0
    t_star = None
    for i in range(min_tokens, len(risk_ma)):
        if risk_ma[i] > theta:
            consecutive += 1
            if consecutive >= persistence:
                t_star = i
                break
        else:
            consecutive = 0

    if t_error is not None:
        if t_star is not None and t_star <= t_error + k:
            verdict = "TP"
        else:
            verdict = "FN"
    else:
        if t_star is not None:
            verdict = "FP"
        else:
            verdict = "TN"

    return verdict, t_error, t_star


def extract_risks(cfg, device):
    states_jsonl = str(cfg["states_jsonl"])
    clf_pt = str(cfg["clf_pt"])
    history = cfg["history"]
    theta = cfg["theta"]
    persistence = cfg["persistence"]
    ma_window = cfg["ma_window"]
    min_tokens = cfg["min_tokens"]

    # Determine hidden_dim from history
    hidden_dim = 4096
    clf = MLP(hidden_dim=hidden_dim).to(device)
    state_dict = torch.load(clf_pt, map_location=device, weights_only=True)
    clf.load_state_dict(state_dict)
    clf.eval()

    records = []
    with open(states_jsonl, "r") as f:
        for line in f:
            item = json.loads(line)
            states = item["states"]
            hp = item["hidden_path"]
            try:
                hidden = load_hidden_from_npz(hp, layer_idx=None, layer_pool="mean", history=history)
            except Exception as e:
                continue
            if len(hidden) == 0:
                continue

            X = torch.tensor(hidden, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = clf(X)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            risk = 1.0 - probs[:, STATE2ID["Safe"]]
            n = min(len(risk), len(states))
            risk = risk[:n]
            states_trimmed = states[:n]

            verdict, t_error, t_star = simulate_detection(
                risk, states_trimmed, theta, persistence, min_tokens, ma_window, K)

            records.append({
                "id": item["id"],
                "risk": risk,
                "states": states_trimmed,
                "verdict": verdict,
                "t_error": t_error,
                "t_star": t_star,
                "length": n,
            })
    return records


def pick_representative(records, verdict):
    candidates = [r for r in records if r["verdict"] == verdict]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r["length"], reverse=True)
    mid = len(candidates) // 2
    return candidates[mid]


def plot_trajectory_panel(ax, record, title, theta):
    risk = record["risk"]
    states = record["states"]
    t = np.arange(len(risk))

    for i in range(len(states)):
        if states[i] == "AtRisk":
            ax.axvspan(i - 0.5, i + 0.5, color="#FFFFCC", alpha=0.5, zorder=0)
        elif states[i] == "Hallucinating":
            ax.axvspan(i - 0.5, i + 0.5, color="#FFCCCC", alpha=0.5, zorder=0)

    ax.plot(t, risk, color="#2176AE", linewidth=1.0, alpha=0.9, label="Risk Score ($r_t$)")
    ax.axhline(theta, color="gold", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"Threshold $\\theta$={theta}")

    if record["t_error"] is not None:
        ax.axvline(record["t_error"], color="red", linestyle=":", linewidth=1.5,
                   label="Critical Point ($t_{err}$)")

    if record["t_star"] is not None:
        ax.axvline(record["t_star"], color="black", linewidth=1.5,
                   label="Detection Trigger")

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Time step ($t$)", fontsize=9)
    ax.set_ylabel("Risk Prob", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-2, len(risk) + 2)
    ax.grid(True, linestyle=":", alpha=0.3)


def generate_figure(config_key, cfg, device, outdir):
    label = cfg["label"]
    theta = cfg["theta"]

    if not cfg["states_jsonl"].exists():
        print(f"  Skipping {label}: states not found at {cfg['states_jsonl']}")
        return
    if not cfg["clf_pt"].exists():
        print(f"  Skipping {label}: clf not found at {cfg['clf_pt']}")
        return

    print(f"  Processing {label} (theta={theta})...")
    records = extract_risks(cfg, device)
    verdict_counts = {v: sum(1 for r in records if r["verdict"] == v) for v in ["TP", "FN", "FP", "TN"]}
    print(f"    {len(records)} trajectories, verdicts: {verdict_counts}")

    tp = pick_representative(records, "TP")
    fn = pick_representative(records, "FN")
    fp = pick_representative(records, "FP")

    if not all([tp, fn, fp]):
        missing = []
        if tp is None: missing.append("TP")
        if fn is None: missing.append("FN")
        if fp is None: missing.append("FP")
        print(f"    WARNING: Missing verdict types: {missing}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)

    panels = [
        (axes[0], tp, "Correct Detection (TP)"),
        (axes[1], fn, "Missed Detection (FN)"),
        (axes[2], fp, "False Alarm (FP)"),
    ]

    for ax, rec, title in panels:
        if rec is not None:
            plot_trajectory_panel(ax, rec, title, theta)
        else:
            ax.set_title(f"{title}\n(no cases)", fontsize=11)
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha='center', fontsize=14)

    if tp is not None:
        axes[0].legend(fontsize=7, loc="upper left", framealpha=0.9)
    elif fn is not None:
        axes[1].legend(fontsize=7, loc="upper left", framealpha=0.9)

    fig.suptitle(f"Risk Trajectories — {label} ($\\theta$={theta})", fontsize=13, y=1.02)
    plt.tight_layout()

    fname = f"fig2_trajectories_{config_key}"
    fig.savefig(outdir / f"{fname}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {fname}.pdf/.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--configs", nargs="*", default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    outdir = ROOT / "results" / "trajectory_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    configs_to_run = args.configs or list(CONFIGS.keys())
    for key in configs_to_run:
        if key not in CONFIGS:
            print(f"Unknown config: {key}")
            continue
        generate_figure(key, CONFIGS[key], device, outdir)

    print("Done!")


if __name__ == "__main__":
    main()
