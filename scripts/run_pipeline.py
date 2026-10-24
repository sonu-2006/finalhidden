#!/usr/bin/env python3
"""V11b Full Pipeline: Corrected ROUGE-L + proper trajectory-level 80/20 split.

Self-contained (no external src/ imports). Produces all results needed for paper.

KEY FIX over V11: Trajectory-level stratified 80/20 split.
  - Probe trained on 80% of trajectories
  - ALL evaluation (classification, detection, baselines, intervention) on 20% held-out

Runs for: PubMedQA x {Qwen, Ministral, BioMistral}
          MedHallu x {Qwen, Ministral, BioMistral}

Per dataset-model:
  1. Binary aggressive probe (PRIMARY) - classification + detection
  2. Ternary probe (ABLATION) - for binary-vs-ternary comparison
  3. Internal baselines: TokenProb, Entropy, Margin, SmoothLogProb, StaticProbe(H=1)
  4. Intervention evaluation at best operating points
  5. Multi-K detection: K=5, 10, 20, 50

Output: one JSON per dataset-model + combined JSON with all results.
"""

import argparse
import json
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# ROUGE-L (WORD-LEVEL) — fixes char-level bug in text_utils.py
# ============================================================

def rouge_l_word(reference, hypothesis):
    """Word-level ROUGE-L F1."""
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
# State annotation
# ============================================================

def annotate_trajectory(gen_text, ref_text, tau=0.12, window=10, tail=3, n_tokens=256):
    """Word-level ROUGE-L verdict + window-based state assignment."""
    score = rouge_l_word(ref_text, gen_text)
    verdict = "correct" if score >= tau else "incorrect"

    if verdict == "correct":
        states = ["Safe"] * n_tokens
    else:
        states = ["Safe"] * n_tokens
        for j in range(max(0, n_tokens - tail), n_tokens):
            states[j] = "Hallucinating"
        for j in range(max(0, n_tokens - tail - window), max(0, n_tokens - tail)):
            states[j] = "AtRisk"

    return verdict, states, score


# ============================================================
# Data loading
# ============================================================

def load_dataset(traj_jsonl, hidden_base_dir, layer_idx=20, tau=0.12):
    """Load trajectories, annotate with word-level ROUGE-L, load hidden states + features."""
    trajectories = []

    with open(traj_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            item = json.loads(line)
            gen_text = item.get("generated_text", "")
            ref_text = item.get("answer", "")
            traj_id = item.get("id", str(line_idx))

            # Resolve hidden path
            hpath = item["hidden_path"]
            if not os.path.isabs(hpath):
                hpath = os.path.join(hidden_base_dir, hpath)
            if not os.path.exists(hpath):
                for prefix in ["results/", "../results/"]:
                    if prefix in item["hidden_path"]:
                        rel = item["hidden_path"][item["hidden_path"].find("results/"):]
                        candidate = os.path.join(hidden_base_dir, rel)
                        if os.path.exists(candidate):
                            hpath = candidate
                            break

            if not os.path.exists(hpath):
                continue

            data = np.load(hpath)

            # Select layer from multi-layer data
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

            # Uncertainty features
            max_probs = data["max_probs"].astype(np.float32) if "max_probs" in data else np.ones(n_tokens, dtype=np.float32)
            entropies = data["entropies"].astype(np.float32) if "entropies" in data else np.zeros(n_tokens, dtype=np.float32)
            margins = data["margins"].astype(np.float32) if "margins" in data else np.ones(n_tokens, dtype=np.float32)

            verdict, states, rouge_score = annotate_trajectory(
                gen_text, ref_text, tau=tau, window=10, tail=3, n_tokens=n_tokens
            )

            t_error = None
            for t, s in enumerate(states):
                if s == "Hallucinating":
                    t_error = t
                    break

            trajectories.append({
                "id": traj_id,
                "hidden": hidden,
                "max_probs": max_probs,
                "entropies": entropies,
                "margins": margins,
                "states": states,
                "verdict": verdict,
                "t_error": t_error,
                "n_tokens": n_tokens,
                "rouge_score": rouge_score,
            })

    return trajectories


def apply_history_pooling(hidden, history):
    """Mean pooling over last H steps."""
    if history <= 1:
        return hidden
    T, D = hidden.shape
    pooled = np.zeros_like(hidden)
    for t in range(T):
        start = max(0, t - history + 1)
        pooled[t] = hidden[start:t+1].mean(axis=0)
    return pooled


# ============================================================
# Trajectory-level stratified split
# ============================================================

def split_trajectories(trajectories, test_size=0.2, random_state=42):
    """Split trajectories 80/20 stratified by verdict.

    Ensures ALL tokens from a trajectory stay in the same partition.
    This matches the paper's claimed protocol.
    """
    rng = np.random.RandomState(random_state)

    incorrect_idx = [i for i, t in enumerate(trajectories) if t["verdict"] == "incorrect"]
    correct_idx = [i for i, t in enumerate(trajectories) if t["verdict"] == "correct"]

    rng.shuffle(incorrect_idx)
    rng.shuffle(correct_idx)

    n_inc_test = max(1, int(len(incorrect_idx) * test_size))
    n_cor_test = max(1, int(len(correct_idx) * test_size))

    test_idx = set(incorrect_idx[:n_inc_test] + correct_idx[:n_cor_test])
    train_idx = set(incorrect_idx[n_inc_test:] + correct_idx[n_cor_test:])

    train_trajs = [trajectories[i] for i in sorted(train_idx)]
    test_trajs = [trajectories[i] for i in sorted(test_idx)]

    return train_trajs, test_trajs


# ============================================================
# Probe model
# ============================================================

import torch
import torch.nn as nn


class MLPProbe(nn.Module):
    def __init__(self, hidden_dim=4096, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


STATE2ID_TERNARY = {"Safe": 0, "AtRisk": 1, "Hallucinating": 2}
STATE2ID_BINARY = {"Safe": 0, "Hallucinating": 1}


def remap_labels_aggressive(states):
    """Binary aggressive: AtRisk -> Hallucinating (i.e., NotSafe)."""
    return ["Hallucinating" if s == "AtRisk" else s for s in states]


def remap_labels_conservative(states):
    """Binary conservative: AtRisk -> Safe."""
    return ["Safe" if s == "AtRisk" else s for s in states]


# ============================================================
# Training
# ============================================================

def train_probe(train_trajs, test_trajs, state2id, n_classes, history=8, epochs=3, lr=1e-3, batch_size=256):
    """Train probe on train_trajs, evaluate risk scores + classification on test_trajs."""
    from collections import Counter
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare TRAINING data
    X_train_seqs = []
    y_train_seqs = []
    for traj in train_trajs:
        hidden = apply_history_pooling(traj["hidden"], history)
        X_train_seqs.append(hidden)
        y_train_seqs.append(np.array([state2id[s] for s in traj["_labels"]], dtype=np.int64))

    X_train_flat = np.concatenate(X_train_seqs, axis=0)
    y_train_flat = np.concatenate(y_train_seqs)
    hidden_dim = X_train_flat.shape[1]

    counts = Counter(y_train_flat.tolist())
    total = len(y_train_flat)
    weights = torch.zeros(n_classes)
    for c in range(n_classes):
        weights[c] = total / (n_classes * counts.get(c, 1))
    weights = weights.to(device)

    model = MLPProbe(hidden_dim=hidden_dim, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weights)

    X_t = torch.tensor(X_train_flat, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train_flat, dtype=torch.long).to(device)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_t))
        for start in range(0, len(X_t), batch_size):
            idx = perm[start:start + batch_size]
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ── Evaluate on TEST trajectories only ──
    model.eval()
    safe_idx = state2id["Safe"]

    # Risk scores on test set
    X_test_seqs = []
    y_test_seqs = []
    for traj in test_trajs:
        hidden = apply_history_pooling(traj["hidden"], history)
        X_test_seqs.append(hidden)
        y_test_seqs.append(np.array([state2id[s] for s in traj["_labels"]], dtype=np.int64))

    risk_sequences = []
    for X_seq in X_test_seqs:
        X_chunk = torch.tensor(X_seq, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(X_chunk), dim=-1).cpu().numpy()
        risk_sequences.append(1.0 - probs[:, safe_idx])

    # Classification metrics on test set
    X_test_flat = np.concatenate(X_test_seqs, axis=0)
    y_test_flat = np.concatenate(y_test_seqs)
    X_test_t = torch.tensor(X_test_flat, dtype=torch.float32).to(device)

    with torch.no_grad():
        all_preds = torch.argmax(model(X_test_t), dim=-1).cpu().numpy()

    from sklearn.metrics import f1_score, accuracy_score
    macro_f1 = f1_score(y_test_flat, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_test_flat, all_preds, average=None, zero_division=0)
    acc = accuracy_score(y_test_flat, all_preds)

    test_counts = Counter(y_test_flat.tolist())
    clf_metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(x) for x in per_class_f1],
        "train_class_counts": {str(k): v for k, v in counts.items()},
        "test_class_counts": {str(k): v for k, v in test_counts.items()},
    }

    # Free GPU memory
    del X_t, y_t, X_test_t, model
    torch.cuda.empty_cache()

    return risk_sequences, clf_metrics


# ============================================================
# Baseline risk computation
# ============================================================

def compute_baseline_risks(trajectories):
    """Compute risk sequences for all internal baselines."""
    baselines = {}

    # 1. TokenProb: risk = 1 - max_prob
    tokprob_risks = [1.0 - traj["max_probs"] for traj in trajectories]
    baselines["tokprob"] = tokprob_risks

    # 2. Entropy: normalize to [0,1] per dataset
    all_ent = np.concatenate([traj["entropies"] for traj in trajectories])
    ent_max = np.percentile(all_ent, 99.5) if len(all_ent) > 0 else 1.0
    ent_max = max(ent_max, 1e-6)
    entropy_risks = [np.clip(traj["entropies"] / ent_max, 0, 1) for traj in trajectories]
    baselines["entropy"] = entropy_risks

    # 3. Margin: risk = 1 - margin
    margin_risks = [1.0 - traj["margins"] for traj in trajectories]
    baselines["margin"] = margin_risks

    # 4. SmoothLogProb: exponential smoothed neg-log-prob, normalized
    alpha = 0.3
    smooth_risks_raw = []
    for traj in trajectories:
        nlp = -np.log(np.clip(traj["max_probs"], 1e-10, 1.0))
        smooth = np.zeros_like(nlp)
        smooth[0] = nlp[0]
        for t in range(1, len(nlp)):
            smooth[t] = alpha * nlp[t] + (1 - alpha) * smooth[t - 1]
        smooth_risks_raw.append(smooth)
    # Normalize
    all_smooth = np.concatenate(smooth_risks_raw)
    sm_max = np.percentile(all_smooth, 99.5) if len(all_smooth) > 0 else 1.0
    sm_max = max(sm_max, 1e-6)
    baselines["smoothlogprob"] = [np.clip(s / sm_max, 0, 1) for s in smooth_risks_raw]

    return baselines


# ============================================================
# Rule-gated detection
# ============================================================

def rule_gated_detect(risk_smooth, theta, persistence, min_tokens):
    """Apply threshold + persistence + min_tokens on pre-smoothed risk."""
    T = len(risk_smooth)
    cnt = 0
    for t in range(T):
        if t < min_tokens:
            cnt = 0
            continue
        if risk_smooth[t] > theta:
            cnt += 1
            if cnt >= persistence:
                return t - persistence + 1
        else:
            cnt = 0
    return None


def smooth_ma(risk, window):
    """Moving average smoothing."""
    if window <= 1:
        return risk
    T = len(risk)
    if T < window:
        return risk
    cumsum = np.cumsum(np.insert(risk, 0, 0.0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    pad = np.full(window - 1, smoothed[0] if len(smoothed) > 0 else 0.0)
    return np.concatenate([pad, smoothed])


def detection_sweep(risk_sequences, trajectories,
                    theta_list, persistence_list, ma_window_list, min_tokens_list,
                    k_list=(5, 10, 20, 50)):
    """Full rule-gated sweep. Returns best EDR@K per FPR cap per K."""

    fpr_caps = [0.05, 0.10, 0.20, 0.30]
    error_indices = [i for i, t in enumerate(trajectories) if t["t_error"] is not None]
    noerror_indices = [i for i, t in enumerate(trajectories) if t["t_error"] is None]
    n_error = len(error_indices)
    n_noerror = len(noerror_indices)

    if n_error == 0 or n_noerror == 0:
        empty = {f"K{k}": {str(c): {"edr_at_k": 0.0, "fpr": 0.0, "lead_time": 0.0,
                                      "note": "degenerate"} for c in fpr_caps} for k in k_list}
        return empty

    # Pre-compute MA-smoothed sequences
    smoothed_cache = {}
    for w in ma_window_list:
        smoothed_cache[w] = [smooth_ma(risk_sequences[i], w) for i in range(len(risk_sequences))]

    # Initialize best results
    best = {}
    for k in k_list:
        best[f"K{k}"] = {}
        for cap in fpr_caps:
            best[f"K{k}"][str(cap)] = {
                "edr_at_k": 0.0, "fpr": 0.0, "lead_time": 0.0,
                "theta": 0.0, "p": 0, "w": 0, "m": 0, "note": "no valid point"
            }

    for w in ma_window_list:
        seqs = smoothed_cache[w]
        for theta in theta_list:
            for p in persistence_list:
                for m in min_tokens_list:
                    # Compute triggers
                    triggers = [rule_gated_detect(seqs[i], theta, p, m) for i in range(len(seqs))]

                    # FPR
                    fp = sum(1 for i in noerror_indices if triggers[i] is not None)
                    fpr = fp / n_noerror

                    if fpr > max(fpr_caps):
                        continue

                    # EDR and lead time per K
                    for k in k_list:
                        detected = 0
                        lead_times = []
                        for i in error_indices:
                            t_err = trajectories[i]["t_error"]
                            if triggers[i] is not None and triggers[i] < t_err + k:
                                detected += 1
                                lead_times.append(t_err - triggers[i])
                        edr = detected / n_error
                        lt = float(np.mean(lead_times)) if lead_times else 0.0

                        for cap in fpr_caps:
                            if fpr <= cap:
                                cap_key = str(cap)
                                k_key = f"K{k}"
                                prev = best[k_key][cap_key]
                                if edr > prev["edr_at_k"] or \
                                   (edr == prev["edr_at_k"] and fpr < prev.get("fpr", 1.0)):
                                    best[k_key][cap_key] = {
                                        "edr_at_k": round(edr, 4),
                                        "fpr": round(fpr, 4),
                                        "lead_time": round(lt, 2),
                                        "theta": theta, "p": p, "w": w, "m": m,
                                    }

    return best


# ============================================================
# Intervention evaluation
# ============================================================

def evaluate_intervention(risk_sequences, trajectories, operating_point):
    """Evaluate early-stop intervention at a given operating point."""
    theta = operating_point["theta"]
    p = operating_point["p"]
    w = operating_point["w"]
    m = operating_point["m"]

    n_incorrect = 0
    n_caught = 0       # trigger exists on incorrect trajectory
    n_prevented = 0    # trigger BEFORE t_error
    completeness_vals = []
    fp_interventions = 0
    n_correct = 0

    for i, traj in enumerate(trajectories):
        risk_smooth = smooth_ma(risk_sequences[i], w)
        t_star = rule_gated_detect(risk_smooth, theta, p, m)

        if traj["verdict"] == "incorrect":
            n_incorrect += 1
            if t_star is not None:
                n_caught += 1
                if t_star < traj["t_error"]:
                    n_prevented += 1
                completeness_vals.append(t_star / traj["n_tokens"])
            else:
                completeness_vals.append(1.0)
        else:
            n_correct += 1
            if t_star is not None:
                fp_interventions += 1
                completeness_vals.append(t_star / traj["n_tokens"])
            else:
                completeness_vals.append(1.0)

    return {
        "hallucination_reduction": round(n_caught / max(n_incorrect, 1), 4),
        "hallucination_prevention": round(n_prevented / max(n_incorrect, 1), 4),
        "completeness": round(float(np.mean(completeness_vals)), 4),
        "fp_intervention_rate": round(fp_interventions / max(n_correct, 1), 4),
        "n_incorrect": n_incorrect,
        "n_caught": n_caught,
        "n_prevented": n_prevented,
        "n_correct": n_correct,
        "fp_interventions": fp_interventions,
    }


# ============================================================
# Main pipeline for one dataset-model
# ============================================================

def run_pipeline(dataset_name, model_name, traj_jsonl, hidden_base_dir,
                 output_dir, layer_idx=20, tau=0.12):
    """Run full pipeline for one dataset-model combination."""
    combo_name = f"{dataset_name}_{model_name}"
    print(f"\n{'='*70}")
    print(f"  {combo_name.upper()}")
    print(f"  Trajectory: {traj_jsonl}")
    print(f"  Layer: {layer_idx}, Tau: {tau}, ROUGE: word-level")
    print(f"{'='*70}")
    sys.stdout.flush()

    # ── Load data ──
    t0 = time.time()
    trajectories = load_dataset(traj_jsonl, hidden_base_dir, layer_idx=layer_idx, tau=tau)
    load_time = time.time() - t0
    print(f"Loaded {len(trajectories)} trajectories in {load_time:.1f}s")

    from collections import Counter
    verdicts = Counter(t["verdict"] for t in trajectories)
    n_incorrect = verdicts.get("incorrect", 0)
    n_correct = verdicts.get("correct", 0)
    print(f"Verdicts: {dict(verdicts)} | Incorrect rate: {n_incorrect/len(trajectories)*100:.1f}%")
    sys.stdout.flush()

    if n_incorrect == 0:
        print("ERROR: No incorrect trajectories. Skipping.")
        return {"error": "no_incorrect_trajectories", "metadata": {"dataset": dataset_name, "model": model_name}}

    # ── Trajectory-level 80/20 stratified split ──
    train_trajs, test_trajs = split_trajectories(trajectories, test_size=0.2, random_state=42)
    from collections import Counter as Ctr
    train_v = Ctr(t["verdict"] for t in train_trajs)
    test_v = Ctr(t["verdict"] for t in test_trajs)
    print(f"Split: train={len(train_trajs)} ({dict(train_v)}), test={len(test_trajs)} ({dict(test_v)})")
    sys.stdout.flush()

    results = {}

    # Sweep parameters: 50 * 4 * 3 * 3 = 1800 configs
    theta_list = [round(x, 2) for x in np.arange(0.50, 1.00, 0.01)]
    persistence_list = [1, 2, 3, 5]
    ma_window_list = [1, 3, 5]
    min_tokens_list = [0, 32, 64]
    k_list = [5, 10, 20, 50]

    # ── Phase 1: Binary Aggressive Probe (PRIMARY) ──
    print(f"\n[1/6] Binary Aggressive Probe (H=8)...")
    sys.stdout.flush()
    for traj in train_trajs + test_trajs:
        traj["_labels"] = remap_labels_aggressive(traj["states"])

    t0 = time.time()
    risk_binary, clf_binary = train_probe(train_trajs, test_trajs, STATE2ID_BINARY, 2, history=8)
    print(f"  Train: {time.time()-t0:.1f}s | TestAcc={clf_binary['accuracy']:.3f} TestMacroF1={clf_binary['macro_f1']:.3f}")
    sys.stdout.flush()

    t0 = time.time()
    det_binary = detection_sweep(risk_binary, test_trajs,
                                  theta_list, persistence_list, ma_window_list, min_tokens_list,
                                  k_list=k_list)
    print(f"  Sweep: {time.time()-t0:.1f}s")
    for cap in ["0.05", "0.1", "0.2", "0.3"]:
        b = det_binary["K5"][cap]
        edr = b.get("edr_at_k", 0)
        print(f"  EDR@5 FPR<={cap}: {edr:.3f}")
    sys.stdout.flush()

    results["probe_binary_aggressive"] = {
        "classification": clf_binary,
        "detection": det_binary,
    }

    # ── Phase 2: Ternary Probe (ABLATION) ──
    print(f"\n[2/6] Ternary Probe (H=8)...")
    sys.stdout.flush()
    for traj in train_trajs + test_trajs:
        traj["_labels"] = traj["states"]

    t0 = time.time()
    risk_ternary, clf_ternary = train_probe(train_trajs, test_trajs, STATE2ID_TERNARY, 3, history=8)
    print(f"  Train: {time.time()-t0:.1f}s | TestAcc={clf_ternary['accuracy']:.3f} TestMacroF1={clf_ternary['macro_f1']:.3f}")
    sys.stdout.flush()

    t0 = time.time()
    det_ternary = detection_sweep(risk_ternary, test_trajs,
                                   theta_list, persistence_list, ma_window_list, min_tokens_list,
                                   k_list=k_list)
    print(f"  Sweep: {time.time()-t0:.1f}s")
    for cap in ["0.05", "0.1", "0.2", "0.3"]:
        b = det_ternary["K5"][cap]
        edr = b.get("edr_at_k", 0)
        print(f"  EDR@5 FPR<={cap}: {edr:.3f}")
    sys.stdout.flush()

    results["probe_ternary"] = {
        "classification": clf_ternary,
        "detection": det_ternary,
    }

    # ── Phase 3: StaticProbe (H=1, binary aggressive) ──
    print(f"\n[3/6] Static Probe (H=1, binary aggressive)...")
    sys.stdout.flush()
    for traj in train_trajs + test_trajs:
        traj["_labels"] = remap_labels_aggressive(traj["states"])

    t0 = time.time()
    risk_static, clf_static = train_probe(train_trajs, test_trajs, STATE2ID_BINARY, 2, history=1)
    print(f"  Train: {time.time()-t0:.1f}s | TestAcc={clf_static['accuracy']:.3f} TestMacroF1={clf_static['macro_f1']:.3f}")
    sys.stdout.flush()

    t0 = time.time()
    det_static = detection_sweep(risk_static, test_trajs,
                                  theta_list, persistence_list, ma_window_list, min_tokens_list,
                                  k_list=k_list)
    print(f"  Sweep: {time.time()-t0:.1f}s")
    for cap in ["0.1", "0.2"]:
        b = det_static["K5"][cap]
        edr = b.get("edr_at_k", 0)
        print(f"  EDR@5 FPR<={cap}: {edr:.3f}")
    sys.stdout.flush()

    results["baseline_staticprobe"] = {
        "classification": clf_static,
        "detection": det_static,
    }

    # ── Phase 4: Internal Baselines (on TEST set only) ──
    print(f"\n[4/6] Internal Baselines (TokenProb, Entropy, Margin, SmoothLogProb)...")
    sys.stdout.flush()

    t0 = time.time()
    baseline_risks = compute_baseline_risks(test_trajs)
    print(f"  Risk computation: {time.time()-t0:.1f}s")
    sys.stdout.flush()

    for bl_name, bl_risks in baseline_risks.items():
        t0 = time.time()
        det_bl = detection_sweep(bl_risks, test_trajs,
                                  theta_list, persistence_list, ma_window_list, min_tokens_list,
                                  k_list=k_list)
        elapsed = time.time() - t0
        edr5_020 = det_bl["K5"]["0.2"].get("edr_at_k", 0)
        print(f"  {bl_name}: sweep {elapsed:.1f}s | EDR@5 FPR<=0.2: {edr5_020:.3f}")
        sys.stdout.flush()

        results[f"baseline_{bl_name}"] = {"detection": det_bl}

    # ── Phase 5: Intervention (on TEST set) ──
    print(f"\n[5/6] Intervention evaluation...")
    sys.stdout.flush()

    intervention_results = {}

    # At best binary_agg operating point for each FPR cap
    for cap in ["0.1", "0.2", "0.3"]:
        op = det_binary["K5"][cap]
        if op.get("note") == "no valid point":
            intervention_results[f"binary_agg_fpr_{cap}"] = {"note": "no valid operating point"}
            continue
        iv = evaluate_intervention(risk_binary, test_trajs, op)
        iv["config"] = {k: op[k] for k in ["theta", "p", "w", "m"]}
        intervention_results[f"binary_agg_fpr_{cap}"] = iv
        print(f"  FPR<={cap}: reduction={iv['hallucination_reduction']:.3f} completeness={iv['completeness']:.3f}")

    # Matched policy: theta=0.70, m=32, w=1, p=3
    matched = {"theta": 0.70, "p": 3, "w": 1, "m": 32}
    iv_matched = evaluate_intervention(risk_binary, test_trajs, matched)
    iv_matched["config"] = matched
    intervention_results["matched_policy"] = iv_matched
    print(f"  Matched(θ=0.70,p=3): reduction={iv_matched['hallucination_reduction']:.3f} completeness={iv_matched['completeness']:.3f}")
    sys.stdout.flush()

    results["intervention"] = intervention_results

    # ── Phase 6: Post-hoc Oracle (on TEST set) ──
    print(f"\n[6/6] Post-hoc oracle baseline...")
    sys.stdout.flush()

    oracle_risks = []
    for traj in test_trajs:
        if traj["verdict"] == "incorrect":
            oracle_risks.append(np.ones(traj["n_tokens"], dtype=np.float32))
        else:
            oracle_risks.append(np.zeros(traj["n_tokens"], dtype=np.float32))

    det_oracle = detection_sweep(oracle_risks, test_trajs,
                                  [0.5], [1], [1], [0], k_list=k_list)
    results["baseline_posthoc"] = {"detection": det_oracle}
    print(f"  Oracle EDR@5: {det_oracle['K5']['0.05'].get('edr_at_k', 0):.3f}")
    sys.stdout.flush()

    # ── Metadata ──
    n_train_inc = sum(1 for t in train_trajs if t["verdict"] == "incorrect")
    n_test_inc = sum(1 for t in test_trajs if t["verdict"] == "incorrect")
    results["metadata"] = {
        "dataset": dataset_name,
        "model": model_name,
        "n_trajectories": len(trajectories),
        "n_incorrect": n_incorrect,
        "n_correct": n_correct,
        "incorrect_rate": round(n_incorrect / len(trajectories), 4),
        "n_train": len(train_trajs),
        "n_test": len(test_trajs),
        "n_train_incorrect": n_train_inc,
        "n_test_incorrect": n_test_inc,
        "split": "trajectory_level_80_20_stratified",
        "tau": tau,
        "layer_idx": layer_idx,
        "rouge_type": "word_level",
        "sweep_configs": len(theta_list) * len(persistence_list) * len(ma_window_list) * len(min_tokens_list),
        "load_time_s": round(load_time, 1),
    }

    # ── Save ──
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"v11b_{combo_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    sys.stdout.flush()

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="V11 Full Pipeline")
    parser.add_argument("--base_dir", default="/home/yzq/paper/Probe_HalluDynamics_B")
    parser.add_argument("--output_dir", default="results/v11b_pipeline")
    parser.add_argument("--datasets", nargs="+", default=["pubmedqa", "medhallu"])
    parser.add_argument("--models", nargs="+", default=["qwen", "ministral", "biomistral"])
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.12)
    args = parser.parse_args()

    base = args.base_dir

    # Dataset-model path mapping
    traj_paths = {
        ("pubmedqa", "qwen"): "results/v9_pubmedqa_qwen_traj/trajectories.jsonl",
        ("pubmedqa", "ministral"): "results/v9_pubmedqa_ministral_traj/trajectories.jsonl",
        ("pubmedqa", "biomistral"): "results/v9_pubmedqa_biomistral_traj/trajectories.jsonl",
        ("medhallu", "qwen"): "results/v9_medhallu_qwen_traj/trajectories.jsonl",
        ("medhallu", "ministral"): "results/v9_medhallu_ministral_traj/trajectories.jsonl",
        ("medhallu", "biomistral"): "results/v9_medhallu_biomistral_traj/trajectories.jsonl",
    }

    print("=" * 70)
    print("V11b FULL PIPELINE — Corrected ROUGE-L + Trajectory-Level 80/20 Split")
    print("=" * 70)
    print(f"Base: {base}")
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Layer: {args.layer_idx}, Tau: {args.tau}")
    print(f"Output: {args.output_dir}")
    print(f"Combinations: {len(args.datasets) * len(args.models)}")
    sys.stdout.flush()

    all_results = {}
    total_start = time.time()

    for ds in args.datasets:
        for mdl in args.models:
            key = (ds, mdl)
            if key not in traj_paths:
                print(f"\nWARNING: No path for {key}, skipping")
                continue

            traj_jsonl = os.path.join(base, traj_paths[key])
            if not os.path.exists(traj_jsonl):
                print(f"\nWARNING: {traj_jsonl} not found, skipping")
                continue

            output_dir = os.path.join(base, args.output_dir)

            try:
                result = run_pipeline(
                    ds, mdl, traj_jsonl, base, output_dir,
                    layer_idx=args.layer_idx, tau=args.tau
                )
                all_results[f"{ds}_{mdl}"] = result
            except Exception as e:
                print(f"\nERROR on {ds}_{mdl}: {e}")
                import traceback
                traceback.print_exc()
                all_results[f"{ds}_{mdl}"] = {"error": str(e)}

            sys.stdout.flush()

    # Save combined
    combined_path = os.path.join(base, args.output_dir, "v11b_combined.json")
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL DONE. Total time: {total_elapsed/60:.1f} minutes")
    print(f"Combined results: {combined_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
