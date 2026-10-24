"""Token-state annotation for trajectory supervision.

Assigns Safe/AtRisk/Hallucinating labels to each token position
based on verifier outcomes and temporal heuristics (window, risk-delta, hybrid).
"""

import json
from pathlib import Path

import numpy as np

from src.claim_verifier import verify_claim_jaccard, verify_claim_rouge


def _moving_average(arr, window=1):
    if window <= 1:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out = (cumsum[window:] - cumsum[:-window]) / float(window)
    pad = np.full(window - 1, out[0] if len(out) > 0 else 0.0)
    return np.concatenate([pad, out])


def _states_risk_delta(risk, theta_low, theta_high, delta, hallucination_tail):
    states = []
    for i, r in enumerate(risk):
        rising = i > 0 and (r - risk[i - 1] > delta)
        if r > theta_high:
            states.append("Hallucinating")
        elif (theta_low < r < theta_high) or rising:
            states.append("AtRisk")
        else:
            states.append("Safe")

    if "Hallucinating" not in states and len(states) > 0:
        for j in range(max(0, len(states) - hallucination_tail), len(states)):
            states[j] = "Hallucinating"
    return states


def _states_window(n_tokens, window=10, hallucination_tail=3):
    states = ["Safe"] * n_tokens
    if n_tokens == 0:
        return states
    for j in range(max(0, n_tokens - hallucination_tail), n_tokens):
        states[j] = "Hallucinating"
    for j in range(max(0, n_tokens - hallucination_tail - window), max(0, n_tokens - hallucination_tail)):
        states[j] = "AtRisk"
    return states


def annotate_states(
    trajectories_jsonl,
    output_jsonl,
    theta_low=0.3,
    theta_high=0.6,
    delta=0.1,
    atrisk_mode="risk_delta",
    window=10,
    hallucination_tail=3,
    ma_window=1,
    verifier="jaccard",
    verifier_threshold=0.5,
):
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(trajectories_jsonl, "r", encoding="utf-8") as f_in, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            item = json.loads(line)
            hidden_path = item["hidden_path"]
            data = np.load(hidden_path)
            max_probs = data["max_probs"]
            token_ids = data["token_ids"]

            risk = 1.0 - max_probs
            risk = _moving_average(risk, window=ma_window)

            gen_text = item.get("generated_text", "")
            ref_text = item.get("answer", "")
            if verifier == "rouge":
                verdict = verify_claim_rouge(gen_text, ref_text, threshold=verifier_threshold)
            else:
                verdict = verify_claim_jaccard(gen_text, ref_text, threshold=verifier_threshold)

            if verdict == "correct":
                states = ["Safe"] * len(risk)
            elif verdict == "incorrect":
                if atrisk_mode == "window":
                    states = _states_window(len(risk), window=window, hallucination_tail=hallucination_tail)
                elif atrisk_mode == "hybrid":
                    states = _states_risk_delta(risk, theta_low, theta_high, delta, hallucination_tail)
                    for j in range(max(0, len(risk) - hallucination_tail - window), max(0, len(risk) - hallucination_tail)):
                        if states[j] != "Hallucinating":
                            states[j] = "AtRisk"
                    for j in range(max(0, len(risk) - hallucination_tail), len(risk)):
                        states[j] = "Hallucinating"
                else:
                    states = _states_risk_delta(risk, theta_low, theta_high, delta, hallucination_tail)
            else:
                states = ["Safe"] * len(risk)

            record = {
                "id": item["id"],
                "verdict": verdict,
                "risk": risk.tolist(),
                "states": states,
                "token_ids": token_ids.tolist(),
                "hidden_path": hidden_path,
                "atrisk_mode": atrisk_mode,
                "window": window,
                "hallucination_tail": hallucination_tail,
                "ma_window": ma_window,
                "theta_low": theta_low,
                "theta_high": theta_high,
                "delta": delta,
                "verifier": verifier,
                "verifier_threshold": verifier_threshold,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("wrote", out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--theta_low", type=float, default=0.3)
    parser.add_argument("--theta_high", type=float, default=0.6)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--atrisk_mode", choices=["risk_delta", "window", "hybrid"], default="risk_delta")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--hallucination_tail", type=int, default=3)
    parser.add_argument("--ma_window", type=int, default=1)
    parser.add_argument("--verifier", choices=["jaccard", "rouge"], default="jaccard")
    parser.add_argument("--verifier_threshold", type=float, default=0.5)
    args = parser.parse_args()

    annotate_states(
        args.trajectories,
        args.output,
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        delta=args.delta,
        atrisk_mode=args.atrisk_mode,
        window=args.window,
        hallucination_tail=args.hallucination_tail,
        ma_window=args.ma_window,
        verifier=args.verifier,
        verifier_threshold=args.verifier_threshold,
    )
