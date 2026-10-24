"""Intervention policy evaluation.

Evaluates early-stop, persistent-stop, and caution intervention policies
by simulating probe-triggered truncation of hallucinated trajectories.
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


def moving_average(arr, window=5):
    if len(arr) == 0:
        return arr
    if window <= 1:
        return arr
    if len(arr) < window:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out = (cumsum[window:] - cumsum[:-window]) / float(window)
    pad = np.full(window - 1, out[0])
    return np.concatenate([pad, out])


def evaluate_intervention(
    labels_jsonl,
    model_path,
    stop_threshold=0.85,
    min_tokens=64,
    ma_window=5,
    layer_idx=None,
    layer_pool="mean",
    history=1,
):
    model, device = load_model(model_path)

    total = 0
    hallucinated = 0
    hallucinated_after = 0
    length_before = 0
    length_after = 0

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
            risk = moving_average(risk, window=ma_window)

            t_stop = None
            for t, r in enumerate(risk):
                if t < min_tokens:
                    continue
                if r > stop_threshold:
                    t_stop = t
                    break

            total += 1
            length_before += len(states)
            if "Hallucinating" in states:
                hallucinated += 1

            if t_stop is None:
                t_stop = len(states)
            truncated_states = states[:t_stop]
            length_after += len(truncated_states)
            if "Hallucinating" in truncated_states:
                hallucinated_after += 1

    if total == 0:
        return {
            "hallucination_reduction": 0.0,
            "completeness": 0.0,
            "stop_threshold": stop_threshold,
            "min_tokens": min_tokens,
            "ma_window": ma_window,
        }

    base_rate = hallucinated / total
    after_rate = hallucinated_after / total
    reduction = (base_rate - after_rate) / base_rate if base_rate > 0 else 0.0
    completeness = length_after / length_before if length_before > 0 else 0.0

    metrics = {
        "hallucination_reduction": reduction,
        "completeness": completeness,
        "stop_threshold": stop_threshold,
        "min_tokens": min_tokens,
        "ma_window": ma_window,
        "layer_idx": layer_idx,
        "layer_pool": layer_pool,
        "history": history,
    }
    print("metrics", metrics)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stop_threshold", type=float, default=0.85)
    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--ma_window", type=int, default=5)
    parser.add_argument("--layer_idx", type=int, default=None)
    parser.add_argument("--layer_pool", type=str, default="mean", choices=["mean", "concat"])
    parser.add_argument("--history", type=int, default=1)
    args = parser.parse_args()

    evaluate_intervention(
        args.labels,
        args.model,
        stop_threshold=args.stop_threshold,
        min_tokens=args.min_tokens,
        ma_window=args.ma_window,
        layer_idx=args.layer_idx,
        layer_pool=args.layer_pool,
        history=args.history,
    )
