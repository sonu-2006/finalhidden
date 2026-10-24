"""Rule-gated critical-point detection and internal baselines.

Evaluates early detection rate (EDR@K) and false positive rate (FPR)
using probe risk scores with threshold-based and persistence-based triggering.
"""

import json
import numpy as np
import torch

from src.hidden_utils import load_hidden_from_npz
from src.state_classifier import MLP, STATE2ID


def load_model(model_path, hidden_dim=4096):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def _metrics_from_counts(detected, with_error, lead_times, fp, no_error, k, theta):
    edr = detected / with_error if with_error else 0.0
    lead_time = float(np.mean(lead_times)) if lead_times else 0.0
    fpr = fp / no_error if no_error else 0.0
    return {
        "edr_at_k": edr,
        "lead_time": lead_time,
        "fpr": fpr,
        "k": k,
        "theta": theta,
    }


def _pick_t_star(risk, theta=0.5, mode="threshold"):
    if len(risk) == 0:
        return None
    if mode == "gradient":
        deltas = np.diff(risk, prepend=risk[0])
        return int(np.argmax(deltas))
    for t, r in enumerate(risk):
        if r > theta:
            return t
    return None


def evaluate_critical_points(
    labels_jsonl,
    model_path,
    theta=0.5,
    k=20,
    layer_idx=None,
    layer_pool="mean",
    history=1,
    critical_mode="threshold",
    cases_out=None,
):
    model, device = load_model(model_path)

    total_with_error = 0
    detected_in_k = 0
    lead_times = []
    false_positives = 0
    total_no_error = 0

    cases_f = open(cases_out, "w", encoding="utf-8") if cases_out else None
    sample_idx = 0

    with open(labels_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            states = item["states"]
            hidden = load_hidden_from_npz(
                item["hidden_path"],
                layer_idx=layer_idx,
                layer_pool=layer_pool,
                history=history,
            )
            if len(hidden) == 0:
                continue

            X = torch.tensor(hidden, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            risk = 1.0 - probs[:, STATE2ID["Safe"]]
            t_star = _pick_t_star(risk, theta=theta, mode=critical_mode)

            t_error = None
            for t, s in enumerate(states):
                if s == "Hallucinating":
                    t_error = t
                    break

            if t_error is None:
                total_no_error += 1
                if t_star is not None:
                    false_positives += 1
            else:
                total_with_error += 1
                if t_star is not None and t_star < t_error + k:
                    detected_in_k += 1
                    lead_times.append(t_error - t_star)

            if cases_f is not None:
                lead_time = None
                if t_error is not None and t_star is not None:
                    lead_time = int(t_error - t_star)
                record = {
                    "id": item.get("id", sample_idx),
                    "length": len(states),
                    "verdict": item.get("verdict", None),
                    "t_error": t_error,
                    "t_star": t_star,
                    "lead_time": lead_time,
                    "triggered": bool(t_error is not None and t_star is not None and t_star < t_error + k),
                    "false_positive": bool(t_error is None and t_star is not None),
                    "k": k,
                    "theta": theta,
                    "max_risk": float(np.max(risk)) if len(risk) > 0 else None,
                    "mean_risk": float(np.mean(risk)) if len(risk) > 0 else None,
                    "risk_at_error": float(risk[t_error]) if t_error is not None and t_error < len(risk) else None,
                    "risk_at_t_star": float(risk[t_star]) if t_star is not None and t_star < len(risk) else None,
                }
                cases_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            sample_idx += 1

    if cases_f is not None:
        cases_f.close()

    metrics = _metrics_from_counts(
        detected_in_k,
        total_with_error,
        lead_times,
        false_positives,
        total_no_error,
        k,
        theta,
    )
    print("metrics", metrics)
    return metrics


def evaluate_critical_points_stratified(
    labels_jsonl,
    model_path,
    theta=0.5,
    k=20,
    short_len=80,
    layer_idx=None,
    layer_pool="mean",
    history=1,
    critical_mode="threshold",
):
    model, device = load_model(model_path)

    buckets = {
        "short": {"with_error": 0, "detected": 0, "no_error": 0, "fp": 0, "lead_times": []},
        "long": {"with_error": 0, "detected": 0, "no_error": 0, "fp": 0, "lead_times": []},
    }

    with open(labels_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            states = item["states"]
            hidden = load_hidden_from_npz(
                item["hidden_path"],
                layer_idx=layer_idx,
                layer_pool=layer_pool,
                history=history,
            )
            if len(hidden) == 0:
                continue

            X = torch.tensor(hidden, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(X)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            risk = 1.0 - probs[:, STATE2ID["Safe"]]
            t_star = _pick_t_star(risk, theta=theta, mode=critical_mode)

            t_error = None
            for t, s in enumerate(states):
                if s == "Hallucinating":
                    t_error = t
                    break

            bucket = "short" if len(states) <= short_len else "long"
            b = buckets[bucket]
            if t_error is None:
                b["no_error"] += 1
                if t_star is not None:
                    b["fp"] += 1
            else:
                b["with_error"] += 1
                if t_star is not None and t_star < t_error + k:
                    b["detected"] += 1
                    b["lead_times"].append(t_error - t_star)

    out = {
        "short": _metrics_from_counts(
            buckets["short"]["detected"],
            buckets["short"]["with_error"],
            buckets["short"]["lead_times"],
            buckets["short"]["fp"],
            buckets["short"]["no_error"],
            k,
            theta,
        ),
        "long": _metrics_from_counts(
            buckets["long"]["detected"],
            buckets["long"]["with_error"],
            buckets["long"]["lead_times"],
            buckets["long"]["fp"],
            buckets["long"]["no_error"],
            k,
            theta,
        ),
    }
    print("stratified", out)
    return out


def evaluate_tokenprob_baseline(labels_jsonl, theta=0.5, k=20, short_len=80):
    total_with_error = 0
    detected_in_k = 0
    lead_times = []
    false_positives = 0
    total_no_error = 0


    buckets = {
        "short": {"with_error": 0, "detected": 0, "no_error": 0, "fp": 0, "lead_times": []},
        "long": {"with_error": 0, "detected": 0, "no_error": 0, "fp": 0, "lead_times": []},
    }

    with open(labels_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            states = item["states"]
            data = np.load(item["hidden_path"])
            max_probs = data["max_probs"]
            if len(max_probs) == 0:
                continue
            risk = 1.0 - max_probs

            t_star = None
            for t, r in enumerate(risk):
                if r > theta:
                    t_star = t
                    break

            t_error = None
            for t, s in enumerate(states):
                if s == "Hallucinating":
                    t_error = t
                    break

            bucket = "short" if len(states) <= short_len else "long"
            b = buckets[bucket]

            if t_error is None:
                total_no_error += 1
                b["no_error"] += 1
                if t_star is not None:
                    false_positives += 1
                    b["fp"] += 1
            else:
                total_with_error += 1
                b["with_error"] += 1
                if t_star is not None and t_star < t_error + k:
                    detected_in_k += 1
                    lead_times.append(t_error - t_star)
                    b["detected"] += 1
                    b["lead_times"].append(t_error - t_star)

    overall = _metrics_from_counts(
        detected_in_k, total_with_error, lead_times, false_positives, total_no_error, k, theta
    )
    stratified = {
        "short": _metrics_from_counts(
            buckets["short"]["detected"],
            buckets["short"]["with_error"],
            buckets["short"]["lead_times"],
            buckets["short"]["fp"],
            buckets["short"]["no_error"],
            k,
            theta,
        ),
        "long": _metrics_from_counts(
            buckets["long"]["detected"],
            buckets["long"]["with_error"],
            buckets["long"]["lead_times"],
            buckets["long"]["fp"],
            buckets["long"]["no_error"],
            k,
            theta,
        ),
    }

    print("tokenprob_overall", overall)
    print("tokenprob_stratified", stratified)
    return overall, stratified


def evaluate_staticprobe_baseline(labels_jsonl, theta=0.5, k=20, short_len=80):
    total_with_error = 0
    detected_in_k = 0
    lead_times = []
    false_positives = 0
    total_no_error = 0


    buckets = {
        "short": {"with_error": 0, "detected": 0, "no_error": 0, "fp": 0, "lead_times": []},
        "long": {"with_error": 0, "detected": 0, "no_error": 0, "fp": 0, "lead_times": []},
    }

    with open(labels_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            states = item["states"]
            data = np.load(item["hidden_path"])
            max_probs = data["max_probs"]
            if len(max_probs) == 0:
                continue
            risk = 1.0 - max_probs
            static_risk = float(np.mean(risk))

            t_star = 0 if static_risk > theta else None

            t_error = None
            for t, s in enumerate(states):
                if s == "Hallucinating":
                    t_error = t
                    break

            bucket = "short" if len(states) <= short_len else "long"
            b = buckets[bucket]

            if t_error is None:
                total_no_error += 1
                b["no_error"] += 1
                if t_star is not None:
                    false_positives += 1
                    b["fp"] += 1
            else:
                total_with_error += 1
                b["with_error"] += 1
                if t_star is not None and t_star < t_error + k:
                    detected_in_k += 1
                    lead_times.append(t_error - t_star)
                    b["detected"] += 1
                    b["lead_times"].append(t_error - t_star)

    overall = _metrics_from_counts(
        detected_in_k, total_with_error, lead_times, false_positives, total_no_error, k, theta
    )
    stratified = {
        "short": _metrics_from_counts(
            buckets["short"]["detected"],
            buckets["short"]["with_error"],
            buckets["short"]["lead_times"],
            buckets["short"]["fp"],
            buckets["short"]["no_error"],
            k,
            theta,
        ),
        "long": _metrics_from_counts(
            buckets["long"]["detected"],
            buckets["long"]["with_error"],
            buckets["long"]["lead_times"],
            buckets["long"]["fp"],
            buckets["long"]["no_error"],
            k,
            theta,
        ),
    }

    print("staticprobe_overall", overall)
    print("staticprobe_stratified", stratified)
    return overall, stratified


def evaluate_posthoc_baseline(labels_jsonl, k=20, short_len=80):
    total_with_error = 0
    detected_in_k = 0
    lead_times = []
    false_positives = 0
    total_no_error = 0


    buckets = {
        "short": {"with_error": 0, "detected": 0, "no_error": 0, "fp": 0, "lead_times": []},
        "long": {"with_error": 0, "detected": 0, "no_error": 0, "fp": 0, "lead_times": []},
    }

    with open(labels_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            states = item["states"]

            t_error = None
            for t, s in enumerate(states):
                if s == "Hallucinating":
                    t_error = t
                    break

            bucket = "short" if len(states) <= short_len else "long"
            b = buckets[bucket]

            if t_error is None:
                total_no_error += 1
                b["no_error"] += 1
            else:
                total_with_error += 1
                b["with_error"] += 1
                t_star = len(states)
                if t_star < t_error + k:
                    detected_in_k += 1
                    lead_times.append(t_error - t_star)
                    b["detected"] += 1
                    b["lead_times"].append(t_error - t_star)

    overall = _metrics_from_counts(
        detected_in_k, total_with_error, lead_times, false_positives, total_no_error, k, theta=1.0
    )
    stratified = {
        "short": _metrics_from_counts(
            buckets["short"]["detected"],
            buckets["short"]["with_error"],
            buckets["short"]["lead_times"],
            buckets["short"]["fp"],
            buckets["short"]["no_error"],
            k,
            theta=1.0,
        ),
        "long": _metrics_from_counts(
            buckets["long"]["detected"],
            buckets["long"]["with_error"],
            buckets["long"]["lead_times"],
            buckets["long"]["fp"],
            buckets["long"]["no_error"],
            k,
            theta=1.0,
        ),
    }

    print("posthoc_overall", overall)
    print("posthoc_stratified", stratified)
    return overall, stratified


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model", required=False)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--baseline", choices=["tokenprob", "staticprobe", "posthoc"], default=None)
    parser.add_argument("--layer_idx", type=int, default=None)
    parser.add_argument("--layer_pool", type=str, default="mean", choices=["mean", "concat"])
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--critical_mode", choices=["threshold", "gradient"], default="threshold")
    parser.add_argument("--cases_out", default=None)
    args = parser.parse_args()

    if args.baseline == "tokenprob":
        evaluate_tokenprob_baseline(args.labels, theta=args.theta, k=args.k)
    elif args.baseline == "staticprobe":
        evaluate_staticprobe_baseline(args.labels, theta=args.theta, k=args.k)
    elif args.baseline == "posthoc":
        evaluate_posthoc_baseline(args.labels, k=args.k)
    elif args.stratified:
        evaluate_critical_points_stratified(
            args.labels,
            args.model,
            theta=args.theta,
            k=args.k,
            layer_idx=args.layer_idx,
            layer_pool=args.layer_pool,
            history=args.history,
            critical_mode=args.critical_mode,
        )
    else:
        evaluate_critical_points(
            args.labels,
            args.model,
            theta=args.theta,
            k=args.k,
            layer_idx=args.layer_idx,
            layer_pool=args.layer_pool,
            history=args.history,
            critical_mode=args.critical_mode,
            cases_out=args.cases_out,
        )
