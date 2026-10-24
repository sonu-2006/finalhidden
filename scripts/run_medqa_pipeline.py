#!/usr/bin/env python3
"""
V12: MedQA-USMLE Trajectory Generation + V11b-style Evaluation Pipeline
- Generates 5 trajectories per prompt with hidden state extraction
- Keyword-based verification (correct option matching)
- Binary aggressive probe + baselines + intervention
- Trajectory-level 80/20 stratified split
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import gc

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ========== CONFIG ==========
MODEL_PATHS = {
    "qwen": "/home/yzq/models/Qwen/Qwen3-8B",
    "llama": "/home/yzq/models/meta-llama/Llama-3.1-8B-Instruct",
    "biomistral": "/home/yzq/models/BioMistral/BioMistral-7B",
}
SAVE_LAYERS = [4, 20, 32]  # early, middle, final
DEFAULT_LAYER = 20
N_GEN = 5
MAX_NEW_TOKENS = 256
MIN_NEW_TOKENS = 32
TEMPERATURE = 0.7
TOP_P = 0.9


# ========== PHASE 1: TRAJECTORY GENERATION ==========

def load_model_for_gen(model_name):
    """Load model with all necessary patches."""
    model_path = MODEL_PATHS[model_name]

    # BioMistral needs pickle workaround
    if model_name == "biomistral":
        os.environ["ALLOW_UNSAFE_TORCH_LOAD"] = "1"
        _orig_torch_load = torch.load
        torch.load = lambda *a, **kw: _orig_torch_load(
            *a, **{k: v for k, v in kw.items() if k != "weights_only"}
        )
        try:
            import transformers.utils.import_utils as iu
            iu.check_torch_load_is_safe = lambda: None
        except Exception:
            pass
        try:
            import transformers.modeling_utils as mu
            mu.check_torch_load_is_safe = lambda: None
        except Exception:
            pass

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedTokenizerFast

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True, local_files_only=True,
        )
    except Exception:
        # Fallback tokenizer
        from pathlib import Path
        tok_file = str(Path(model_path) / "tokenizer.json")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_file,
            bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else None,
        device_map=None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def build_prompt_text(sample):
    """Build plain text prompt for reference."""
    return (
        f"Reference:\n{sample['context']}\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Please select the correct answer and explain your reasoning briefly."
    )


def build_chat_prompt(sample, tokenizer):
    """Build chat-formatted prompt using model's chat template."""
    user_msg = (
        f"Read the following clinical scenario and answer the question.\n\n"
        f"{sample['context']}\n\n"
        f"{sample['question']}\n\n"
        f"Select the correct answer and explain your reasoning briefly."
    )
    messages = [{"role": "user", "content": user_msg}]
    try:
        # Disable thinking mode for Qwen3 (otherwise it uses <think> tags
        # that consume all tokens on chain-of-thought without final answer)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older models don't support enable_thinking kwarg
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = build_prompt_text(sample)
    except Exception:
        # Fallback to plain text
        prompt = build_prompt_text(sample)
    return prompt


def generate_trajectories(data_path, model_name, output_dir, max_samples=None):
    """Generate trajectories with hidden state extraction."""
    output_dir = Path(output_dir)
    hidden_dir = output_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line))
    if max_samples:
        samples = samples[:max_samples]

    print(f"Generating trajectories: {len(samples)} prompts × {N_GEN} gens = {len(samples) * N_GEN} total")
    print(f"Model: {model_name} ({MODEL_PATHS[model_name]})")

    model, tokenizer, device = load_model_for_gen(model_name)
    print(f"Model loaded on {device}")

    layer_ids = np.array(SAVE_LAYERS, dtype=np.int64)

    out_path = output_dir / "trajectories.jsonl"
    t0 = time.time()

    with open(out_path, "w") as f_out:
        for si, sample in enumerate(tqdm(samples, desc=f"gen_{model_name}")):
            prompt = build_chat_prompt(sample, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]

            for g in range(N_GEN):
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        min_new_tokens=MIN_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        output_scores=True,
                    )

                gen_ids = gen.sequences[0, prompt_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                n_tokens = len(gen.scores)

                # Extract hidden states, uncertainties
                hidden_list = []
                hidden_layers_list = []
                max_probs = []
                entropies = []
                margins = []

                for t_idx in range(n_tokens):
                    step_h = gen.hidden_states[t_idx]

                    # Last layer hidden state
                    h_last = step_h[-1][0, -1, :].detach().to(torch.float16).cpu().numpy()
                    hidden_list.append(h_last)

                    # Multi-layer hidden states
                    layers = []
                    for lid in SAVE_LAYERS:
                        if lid < len(step_h):
                            layers.append(step_h[lid][0, -1, :].detach().to(torch.float16).cpu().numpy())
                        else:
                            layers.append(step_h[-1][0, -1, :].detach().to(torch.float16).cpu().numpy())
                    hidden_layers_list.append(layers)

                    # Uncertainty features from logits
                    probs = torch.softmax(gen.scores[t_idx][0], dim=-1)
                    sorted_p, _ = torch.sort(probs, descending=True)
                    max_probs.append(float(sorted_p[0].cpu()))
                    margins.append(float((sorted_p[0] - sorted_p[1]).cpu()))
                    ent = -torch.sum(probs * torch.log(probs + 1e-12)).cpu().item()
                    entropies.append(ent)

                # Stack and pad to 256
                pad_len = MAX_NEW_TOKENS
                d_model = hidden_list[0].shape[0] if hidden_list else 4096

                hidden_arr = np.zeros((pad_len, d_model), dtype=np.float16)
                hidden_layers_arr = np.zeros((pad_len, len(SAVE_LAYERS), d_model), dtype=np.float16)
                max_probs_arr = np.zeros(pad_len, dtype=np.float32)
                entropies_arr = np.zeros(pad_len, dtype=np.float32)
                margins_arr = np.zeros(pad_len, dtype=np.float32)
                token_ids_arr = np.zeros(pad_len, dtype=np.int64)

                actual_len = min(n_tokens, pad_len)
                if actual_len > 0:
                    hidden_arr[:actual_len] = np.stack(hidden_list[:actual_len])
                    hidden_layers_arr[:actual_len] = np.stack(hidden_layers_list[:actual_len])
                    max_probs_arr[:actual_len] = np.array(max_probs[:actual_len], dtype=np.float32)
                    entropies_arr[:actual_len] = np.array(entropies[:actual_len], dtype=np.float32)
                    margins_arr[:actual_len] = np.array(margins[:actual_len], dtype=np.float32)
                    token_ids_arr[:actual_len] = gen_ids[:actual_len].cpu().numpy()

                # Save NPZ
                sample_id = f"{sample['id']}_{g}"
                npz_path = hidden_dir / f"{sample_id}.npz"
                np.savez_compressed(
                    npz_path,
                    hidden=hidden_arr,
                    hidden_layers=hidden_layers_arr,
                    layer_ids=layer_ids,
                    max_probs=max_probs_arr,
                    entropies=entropies_arr,
                    margins=margins_arr,
                    token_ids=token_ids_arr,
                )

                record = {
                    "id": sample_id,
                    "prompt": prompt,
                    "generated_text": gen_text,
                    "answer": sample["answer"],
                    "correct_letter": sample["correct_letter"],
                    "correct_text": sample["correct_text"],
                    "hidden_path": str(npz_path),
                    "n_tokens": actual_len,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (si + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (si + 1) / elapsed * 60
                print(f"  [{si+1}/{len(samples)}] {elapsed/60:.1f}min, {rate:.0f} prompts/min")

    elapsed = time.time() - t0
    print(f"Done: {len(samples) * N_GEN} trajectories in {elapsed/60:.1f} min")
    print(f"Saved: {out_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


# ========== PHASE 2: VERIFICATION ==========

def verify_medqa_keyword(generated_text, correct_letter, correct_text):
    """
    Verify MedQA answer by checking if the correct option appears in generation.
    Returns 'correct' or 'incorrect'.
    """
    text_lower = generated_text.lower().strip()
    correct_text_lower = correct_text.lower().strip()

    # Strategy 1: Check if the correct option letter appears early (first 50 chars)
    # e.g., "B.", "B)", "B ", "Answer: B"
    first_part = text_lower[:80]
    letter_patterns = [
        f"{correct_letter.lower()}.",
        f"{correct_letter.lower()})",
        f"{correct_letter.lower()} ",
        f"answer is {correct_letter.lower()}",
        f"answer: {correct_letter.lower()}",
        f"option {correct_letter.lower()}",
    ]
    letter_match = any(p in first_part for p in letter_patterns)

    # Strategy 2: Check if the correct option TEXT appears anywhere
    # Handle multi-word options: at least the core text should appear
    text_match = correct_text_lower in text_lower

    # Strategy 3: For short options (1-3 words), check word-level presence
    if not text_match and len(correct_text.split()) <= 3:
        # Check each word (skip very common words)
        stopwords = {"the", "a", "an", "of", "in", "to", "and", "or", "is", "are", "with", "for"}
        key_words = [w.lower() for w in correct_text.split() if w.lower() not in stopwords and len(w) > 2]
        if key_words:
            text_match = all(w in text_lower for w in key_words)

    # Also check wrong letters — if model explicitly chooses a wrong letter
    wrong_letters = [l for l in "ABCD" if l != correct_letter]
    wrong_letter_match = False
    for wl in wrong_letters:
        wrong_patterns = [
            f"{wl.lower()}.",
            f"answer is {wl.lower()}",
            f"answer: {wl.lower()}",
        ]
        if any(p in first_part for p in wrong_patterns):
            wrong_letter_match = True
            break

    if letter_match and not wrong_letter_match:
        return "correct"
    if wrong_letter_match and not letter_match:
        return "incorrect"
    if text_match:
        return "correct"

    return "incorrect"


# ========== PHASE 3: PROBE TRAINING & EVALUATION (from V11b) ==========

def load_trajectories_with_verification(traj_dir):
    """Load trajectories and verify using keyword matching."""
    traj_path = Path(traj_dir) / "trajectories.jsonl"
    trajectories = []
    with open(traj_path) as f:
        for line in f:
            d = json.loads(line)
            d["verdict"] = verify_medqa_keyword(
                d["generated_text"],
                d["correct_letter"],
                d["correct_text"],
            )
            trajectories.append(d)
    return trajectories


def split_trajectories(trajectories, test_size=0.2, random_state=42):
    """Trajectory-level stratified 80/20 split."""
    rng = np.random.RandomState(random_state)
    incorrect_idx = [i for i, t in enumerate(trajectories) if t["verdict"] == "incorrect"]
    correct_idx = [i for i, t in enumerate(trajectories) if t["verdict"] == "correct"]
    rng.shuffle(incorrect_idx)
    rng.shuffle(correct_idx)
    n_inc_test = max(1, int(len(incorrect_idx) * test_size))
    n_cor_test = max(1, int(len(correct_idx) * test_size))
    test_idx = set(incorrect_idx[:n_inc_test] + correct_idx[:n_cor_test])
    train_idx = set(range(len(trajectories))) - test_idx
    train_trajs = [trajectories[i] for i in sorted(train_idx)]
    test_trajs = [trajectories[i] for i in sorted(test_idx)]
    return train_trajs, test_trajs


def assign_token_labels_binary_agg(n_tokens, is_incorrect, window=10, tail=3):
    """Binary aggressive: Safe(0) vs NotSafe(1). Last `tail` tokens = NotSafe if incorrect, preceding `window` tokens also NotSafe."""
    labels = np.zeros(n_tokens, dtype=np.int64)
    if is_incorrect and n_tokens > 0:
        err_start = max(0, n_tokens - tail)
        at_risk_start = max(0, err_start - window)
        labels[at_risk_start:] = 1  # everything from at_risk_start onwards = NotSafe
    return labels


def assign_token_labels_ternary(n_tokens, is_incorrect, window=10, tail=3):
    """Ternary: Safe(0), AtRisk(1), Hallucinating(2)."""
    labels = np.zeros(n_tokens, dtype=np.int64)
    if is_incorrect and n_tokens > 0:
        err_start = max(0, n_tokens - tail)
        at_risk_start = max(0, err_start - window)
        labels[at_risk_start:err_start] = 1
        labels[err_start:] = 2
    return labels


def extract_features(traj, layer_idx=1, history=8):
    """Extract hidden state features for a trajectory. layer_idx is index into SAVE_LAYERS."""
    hidden_path = traj["hidden_path"]
    if not os.path.exists(hidden_path):
        return None, None

    npz = np.load(hidden_path)
    n_tokens = traj.get("n_tokens", 0)
    if n_tokens == 0:
        # Infer from non-zero tokens
        token_ids = npz["token_ids"]
        n_tokens = int(np.sum(token_ids != 0))
        if n_tokens == 0:
            n_tokens = len(token_ids)

    if n_tokens < history:
        return None, None

    # Get hidden states from specified layer
    if "hidden_layers" in npz.files:
        h = npz["hidden_layers"][:n_tokens, layer_idx, :]  # (T, D)
    else:
        h = npz["hidden"][:n_tokens, :]  # fallback to last layer

    # Build features with history pooling (float16 to save memory)
    T, D = h.shape
    features = np.zeros((T, D * history), dtype=np.float16)
    for t in range(T):
        start = max(0, t - history + 1)
        chunk = h[start:t+1]  # (<=history, D)
        # Pad if not enough history
        if chunk.shape[0] < history:
            pad = np.zeros((history - chunk.shape[0], D), dtype=h.dtype)
            chunk = np.concatenate([pad, chunk], axis=0)
        features[t] = chunk.reshape(-1).astype(np.float16)

    return features, n_tokens


class SimpleProbe(nn.Module):
    def __init__(self, input_dim, n_classes=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(train_trajs, test_trajs, label_fn, n_classes, layer_idx=1, history=8,
                epochs=3, batch_size=256, lr=1e-3, device="cpu",
                train_tokens_per_traj=32):
    """Train probe on train_trajs, evaluate on test_trajs.
    train_tokens_per_traj: subsample tokens per trajectory for training to save memory.
    Test uses all tokens (needed for detection sweep)."""

    rng = np.random.RandomState(42)

    def collect(trajs, subsample=0):
        all_feats, all_labels, traj_indices = [], [], []
        for ti, traj in enumerate(trajs):
            feats, n_tokens = extract_features(traj, layer_idx=layer_idx, history=history)
            if feats is None:
                continue
            is_inc = traj["verdict"] == "incorrect"
            labels = label_fn(n_tokens, is_inc)
            if subsample > 0 and n_tokens > subsample:
                # Random subsample tokens for training
                idx = rng.choice(n_tokens, subsample, replace=False)
                idx.sort()
                feats = feats[idx]
                labels = labels[idx]
                traj_indices.extend([ti] * subsample)
            else:
                traj_indices.extend([ti] * n_tokens)
            all_feats.append(feats)
            all_labels.append(labels)
        if not all_feats:
            return None, None, []
        return np.concatenate(all_feats), np.concatenate(all_labels), traj_indices

    X_train, y_train, _ = collect(train_trajs, subsample=train_tokens_per_traj)
    gc.collect()
    X_test, y_test, test_traj_idx = collect(test_trajs, subsample=0)

    if X_train is None or X_test is None:
        return None, {}, {}

    # Class weights
    class_counts = np.bincount(y_train, minlength=n_classes)
    total = len(y_train)
    weights = torch.tensor([total / (n_classes * max(c, 1)) for c in class_counts], dtype=torch.float32).to(device)

    # Train
    input_dim = X_train.shape[1]
    probe = SimpleProbe(input_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Convert float16 numpy → float32 tensor in chunks to avoid 3x memory peak
    def np16_to_tensor(arr):
        t = torch.empty(arr.shape, dtype=torch.float32)
        cs = 50000
        for i in range(0, len(arr), cs):
            t[i:i+cs] = torch.from_numpy(arr[i:i+cs].astype(np.float32))
        return t

    train_X_t = np16_to_tensor(X_train)
    del X_train; gc.collect()
    train_y_t = torch.tensor(y_train, dtype=torch.long)
    del y_train; gc.collect()

    dataset = TensorDataset(train_X_t, train_y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    probe.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

    del train_X_t, train_y_t, dataset, loader; gc.collect()

    # Evaluate on test in chunks (never load full test features at once)
    probe.eval()
    test_X_t = np16_to_tensor(X_test)
    del X_test; gc.collect()

    with torch.no_grad():
        chunk_size = 10000
        all_logits = []
        for i in range(0, len(test_X_t), chunk_size):
            chunk = test_X_t[i:i+chunk_size].to(device)
            all_logits.append(probe(chunk).cpu())
        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy()

    del test_X_t; gc.collect()

    # Classification metrics
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_test, preds, average=None, zero_division=0).tolist()

    cls_metrics = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": [round(f, 4) for f in per_class_f1],
        "train_class_counts": class_counts.tolist(),
        "test_class_counts": np.bincount(y_test, minlength=n_classes).tolist(),
    }

    # Compute per-trajectory risk scores for detection
    # risk_t = 1 - P(Safe) = P(NotSafe) for binary, or 1 - P(Safe) for ternary
    risk_scores = 1.0 - probs[:, 0]  # P(not safe)

    # Map back to trajectories
    traj_risks = {}
    for idx, ti in enumerate(test_traj_idx):
        if ti not in traj_risks:
            traj_risks[ti] = []
        traj_risks[ti].append(risk_scores[idx])

    return probe, cls_metrics, traj_risks


# ========== PHASE 4: DETECTION SWEEP ==========

def detection_sweep(traj_risks, test_trajs, K_list=[5, 10, 20, 50],
                    fpr_caps=[0.05, 0.1, 0.2, 0.3]):
    """Rule-gated detection sweep."""
    thetas = np.arange(0.50, 1.00, 0.01)
    persistences = [1, 2, 3, 5]
    ma_windows = [1, 3, 5]
    min_tokens_list = [0, 32, 64]

    results = {}

    for K in K_list:
        best_by_cap = {}
        for gamma in fpr_caps:
            best_by_cap[str(gamma)] = {
                "edr_at_k": 0.0, "fpr": 0.0, "lead_time": 0.0,
                "theta": 0.0, "p": 0, "w": 0, "m": 0,
                "note": "no valid point"
            }

        for theta in thetas:
            for p in persistences:
                for w in ma_windows:
                    for m in min_tokens_list:
                        # Evaluate this config
                        triggers = {}
                        for ti, risk_list in traj_risks.items():
                            risks = np.array(risk_list)
                            T = len(risks)

                            # Moving average smoothing
                            if w > 1:
                                smoothed = np.convolve(risks, np.ones(w)/w, mode='same')
                            else:
                                smoothed = risks

                            # Find trigger point
                            trigger_t = None
                            consecutive = 0
                            for t in range(m, T):
                                if smoothed[t] >= theta:
                                    consecutive += 1
                                    if consecutive >= p:
                                        trigger_t = t - p + 1
                                        break
                                else:
                                    consecutive = 0
                            triggers[ti] = trigger_t

                        # Compute EDR and FPR
                        n_incorrect = 0
                        n_correct = 0
                        n_early_detect = 0
                        n_false_trigger = 0
                        lead_times = []

                        for ti, traj in enumerate(test_trajs):
                            if ti not in traj_risks:
                                continue
                            trigger_t = triggers.get(ti)
                            is_inc = traj["verdict"] == "incorrect"
                            n_tokens = len(traj_risks[ti])

                            if is_inc:
                                n_incorrect += 1
                                if trigger_t is not None:
                                    err_start = max(0, n_tokens - 3)
                                    lead = err_start - trigger_t
                                    if lead >= K:
                                        n_early_detect += 1
                                        lead_times.append(lead)
                            else:
                                n_correct += 1
                                if trigger_t is not None:
                                    n_false_trigger += 1

                        if n_incorrect == 0 or n_correct == 0:
                            continue

                        edr = n_early_detect / n_incorrect
                        fpr = n_false_trigger / n_correct
                        avg_lt = np.mean(lead_times) if lead_times else 0.0

                        # Update best per cap
                        for gamma in fpr_caps:
                            if fpr <= gamma:
                                current = best_by_cap[str(gamma)]
                                if edr > current["edr_at_k"] or (edr == current["edr_at_k"] and fpr < current["fpr"]):
                                    best_by_cap[str(gamma)] = {
                                        "edr_at_k": round(edr, 4),
                                        "fpr": round(fpr, 4),
                                        "lead_time": round(avg_lt, 2),
                                        "theta": round(float(theta), 2),
                                        "p": p, "w": w, "m": m,
                                    }

        results[f"K{K}"] = best_by_cap

    return results


def baseline_detection(test_trajs, feature_name, K_list=[5, 10, 20, 50],
                       fpr_caps=[0.05, 0.1, 0.2, 0.3]):
    """Run detection sweep using uncertainty baselines."""
    # Extract per-trajectory risk from uncertainty features
    traj_risks = {}
    for ti, traj in enumerate(test_trajs):
        hidden_path = traj["hidden_path"]
        if not os.path.exists(hidden_path):
            continue
        npz = np.load(hidden_path)
        n_tokens = traj.get("n_tokens", 0)
        if n_tokens == 0:
            token_ids = npz["token_ids"]
            n_tokens = int(np.sum(token_ids != 0))

        if n_tokens < 5:
            continue

        if feature_name == "tokprob":
            raw = npz["max_probs"][:n_tokens]
            risk = 1.0 - raw
        elif feature_name == "entropy":
            raw = npz["entropies"][:n_tokens]
            risk = raw / (np.max(raw) + 1e-8)
        elif feature_name == "margin":
            raw = npz["margins"][:n_tokens]
            risk = 1.0 - raw
        elif feature_name == "smoothlogprob":
            raw = npz["max_probs"][:n_tokens]
            neg_log = -np.log(raw + 1e-8)
            alpha = 0.3
            smoothed = np.zeros_like(neg_log)
            smoothed[0] = neg_log[0]
            for t in range(1, len(neg_log)):
                smoothed[t] = alpha * neg_log[t] + (1 - alpha) * smoothed[t-1]
            risk = smoothed / (np.max(smoothed) + 1e-8)
        else:
            continue

        traj_risks[ti] = risk.tolist()

    return detection_sweep(traj_risks, test_trajs, K_list, fpr_caps)


def intervention_eval(traj_risks, test_trajs, fpr_caps=[0.1, 0.2, 0.3]):
    """Evaluate intervention: if we stop at trigger point, how many hallucinations prevented?"""
    results = {}

    # Get best detection config per FPR cap from K=5
    det_results = detection_sweep(traj_risks, test_trajs, K_list=[5], fpr_caps=fpr_caps)
    k5 = det_results["K5"]

    for gamma in fpr_caps:
        cap_key = str(gamma)
        config = k5[cap_key]
        theta = config["theta"]
        p_val = config["p"]
        w_val = config["w"]
        m_val = config["m"]

        if config.get("note") == "no valid point":
            results[f"binary_agg_fpr_{gamma}"] = {
                "hallucination_reduction": 0.0, "completeness": 0.0,
                "n_incorrect": 0, "n_caught": 0, "n_correct": 0,
            }
            continue

        n_incorrect = 0
        n_caught = 0
        completeness_vals = []
        n_correct = 0
        n_fp = 0

        for ti, traj in enumerate(test_trajs):
            if ti not in traj_risks:
                continue
            risks = np.array(traj_risks[ti])
            T = len(risks)
            is_inc = traj["verdict"] == "incorrect"

            # Find trigger
            if w_val > 1:
                smoothed = np.convolve(risks, np.ones(w_val)/w_val, mode='same')
            else:
                smoothed = risks

            trigger_t = None
            consecutive = 0
            for t in range(m_val, T):
                if smoothed[t] >= theta:
                    consecutive += 1
                    if consecutive >= p_val:
                        trigger_t = t - p_val + 1
                        break
                else:
                    consecutive = 0

            if is_inc:
                n_incorrect += 1
                if trigger_t is not None:
                    n_caught += 1
                    completeness_vals.append(trigger_t / T)
                else:
                    completeness_vals.append(1.0)
            else:
                n_correct += 1
                if trigger_t is not None:
                    n_fp += 1
                    completeness_vals.append(trigger_t / T)
                else:
                    completeness_vals.append(1.0)

        results[f"binary_agg_fpr_{gamma}"] = {
            "hallucination_reduction": round(n_caught / max(n_incorrect, 1), 4),
            "completeness": round(np.mean(completeness_vals), 4) if completeness_vals else 0.0,
            "fp_intervention_rate": round(n_fp / max(n_correct, 1), 4),
            "n_incorrect": n_incorrect,
            "n_caught": n_caught,
            "n_correct": n_correct,
        }

    # Matched policy: fixed theta=0.70, p=3, w=1, m=32
    matched_theta, matched_p, matched_w, matched_m = 0.70, 3, 1, 32
    n_incorrect = n_caught = 0
    completeness_vals = []
    n_correct = n_fp = 0

    for ti, traj in enumerate(test_trajs):
        if ti not in traj_risks:
            continue
        risks = np.array(traj_risks[ti])
        T = len(risks)
        is_inc = traj["verdict"] == "incorrect"

        trigger_t = None
        consecutive = 0
        for t in range(matched_m, T):
            if risks[t] >= matched_theta:
                consecutive += 1
                if consecutive >= matched_p:
                    trigger_t = t - matched_p + 1
                    break
            else:
                consecutive = 0

        if is_inc:
            n_incorrect += 1
            if trigger_t is not None:
                n_caught += 1
                completeness_vals.append(trigger_t / T)
            else:
                completeness_vals.append(1.0)
        else:
            n_correct += 1
            if trigger_t is not None:
                n_fp += 1
                completeness_vals.append(trigger_t / T)
            else:
                completeness_vals.append(1.0)

    results["matched_policy"] = {
        "hallucination_reduction": round(n_caught / max(n_incorrect, 1), 4),
        "completeness": round(np.mean(completeness_vals), 4) if completeness_vals else 0.0,
        "n_incorrect": n_incorrect,
        "n_caught": n_caught,
        "n_correct": n_correct,
    }

    return results


# ========== PHASE 5: MAIN PIPELINE ==========

def run_pipeline(traj_dir, output_path, layer_idx=1, history=8):
    """Run full evaluation pipeline on generated trajectories."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION: {traj_dir}")
    print(f"  Layer: {SAVE_LAYERS[layer_idx]}, History: {history}")
    print(f"{'='*70}")

    # Load and verify
    trajs = load_trajectories_with_verification(traj_dir)
    verdicts = {}
    for t in trajs:
        v = t["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    n_total = len(trajs)
    n_incorrect = verdicts.get("incorrect", 0)
    n_correct = verdicts.get("correct", 0)
    inc_rate = n_incorrect / n_total if n_total > 0 else 0

    print(f"Loaded {n_total} trajectories")
    print(f"Verdicts: {verdicts} | Incorrect rate: {inc_rate:.1%}")

    # Split
    train_trajs, test_trajs = split_trajectories(trajs)
    train_v = {}
    test_v = {}
    for t in train_trajs:
        train_v[t["verdict"]] = train_v.get(t["verdict"], 0) + 1
    for t in test_trajs:
        test_v[t["verdict"]] = test_v.get(t["verdict"], 0) + 1
    print(f"Split: train={len(train_trajs)} ({train_v}), test={len(test_trajs)} ({test_v})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"metadata": {
        "n_trajectories": n_total, "n_incorrect": n_incorrect, "n_correct": n_correct,
        "incorrect_rate": round(inc_rate, 4),
        "n_train": len(train_trajs), "n_test": len(test_trajs),
        "n_train_incorrect": train_v.get("incorrect", 0),
        "n_test_incorrect": test_v.get("incorrect", 0),
    }}

    # [1/6] Binary Aggressive Probe
    print(f"\n[1/6] Binary Aggressive Probe (H={history})...")
    t0 = time.time()
    _, cls_ba, risks_ba = train_probe(
        train_trajs, test_trajs,
        lambda n, inc: assign_token_labels_binary_agg(n, inc),
        n_classes=2, layer_idx=layer_idx, history=history, device=device,
    )
    t1 = time.time()
    det_ba = detection_sweep(risks_ba, test_trajs) if risks_ba else {}
    t2 = time.time()
    print(f"  Train: {t1-t0:.1f}s | TestAcc={cls_ba.get('accuracy',0):.3f} TestMacroF1={cls_ba.get('macro_f1',0):.3f}")
    print(f"  Sweep: {t2-t1:.1f}s")
    for gamma in ["0.05", "0.1", "0.2", "0.3"]:
        edr = det_ba.get("K5", {}).get(gamma, {}).get("edr_at_k", 0)
        print(f"  EDR@5 FPR<={gamma}: {edr:.3f}")
    results["probe_binary_aggressive"] = {"classification": cls_ba, "detection": det_ba}
    gc.collect()

    # [2/6] Ternary Probe
    print(f"\n[2/6] Ternary Probe (H={history})...")
    t0 = time.time()
    _, cls_ter, risks_ter = train_probe(
        train_trajs, test_trajs,
        lambda n, inc: assign_token_labels_ternary(n, inc),
        n_classes=3, layer_idx=layer_idx, history=history, device=device,
    )
    t1 = time.time()
    det_ter = detection_sweep(risks_ter, test_trajs) if risks_ter else {}
    t2 = time.time()
    print(f"  Train: {t1-t0:.1f}s | TestAcc={cls_ter.get('accuracy',0):.3f} TestMacroF1={cls_ter.get('macro_f1',0):.3f}")
    print(f"  Sweep: {t2-t1:.1f}s")
    for gamma in ["0.1", "0.2"]:
        edr = det_ter.get("K5", {}).get(gamma, {}).get("edr_at_k", 0)
        print(f"  EDR@5 FPR<={gamma}: {edr:.3f}")
    results["probe_ternary"] = {"classification": cls_ter, "detection": det_ter}
    del risks_ter; gc.collect()

    # [3/6] Static Probe (H=1)
    print(f"\n[3/6] Static Probe (H=1, binary aggressive)...")
    t0 = time.time()
    _, cls_sp, risks_sp = train_probe(
        train_trajs, test_trajs,
        lambda n, inc: assign_token_labels_binary_agg(n, inc),
        n_classes=2, layer_idx=layer_idx, history=1, device=device,
    )
    t1 = time.time()
    det_sp = detection_sweep(risks_sp, test_trajs) if risks_sp else {}
    t2 = time.time()
    print(f"  Train: {t1-t0:.1f}s | TestAcc={cls_sp.get('accuracy',0):.3f} TestMacroF1={cls_sp.get('macro_f1',0):.3f}")
    print(f"  Sweep: {t2-t1:.1f}s")
    for gamma in ["0.1", "0.2"]:
        edr = det_sp.get("K5", {}).get(gamma, {}).get("edr_at_k", 0)
        print(f"  EDR@5 FPR<={gamma}: {edr:.3f}")
    results["baseline_staticprobe"] = {"classification": cls_sp, "detection": det_sp}
    del risks_sp; gc.collect()

    # [4/6] Uncertainty baselines
    print(f"\n[4/6] Internal Baselines (TokenProb, Entropy, Margin, SmoothLogProb)...")
    t0 = time.time()
    for bl_name in ["tokprob", "entropy", "margin", "smoothlogprob"]:
        det_bl = baseline_detection(test_trajs, bl_name)
        edr_02 = det_bl.get("K5", {}).get("0.2", {}).get("edr_at_k", 0)
        print(f"  {bl_name}: EDR@5 FPR<=0.2: {edr_02:.3f}")
        results[f"baseline_{bl_name}"] = {"detection": det_bl}
    print(f"  Baselines: {time.time()-t0:.1f}s")

    # [5/6] Intervention
    print(f"\n[5/6] Intervention evaluation...")
    intv = intervention_eval(risks_ba, test_trajs)
    for gamma in [0.1, 0.2, 0.3]:
        iv = intv.get(f"binary_agg_fpr_{gamma}", {})
        print(f"  FPR<={gamma}: reduction={iv.get('hallucination_reduction',0):.3f} completeness={iv.get('completeness',0):.3f}")
    mp = intv.get("matched_policy", {})
    print(f"  Matched(θ=0.70,p=3): reduction={mp.get('hallucination_reduction',0):.3f} completeness={mp.get('completeness',0):.3f}")
    results["intervention"] = intv

    # [6/6] Post-hoc oracle
    print(f"\n[6/6] Post-hoc oracle baseline...")
    oracle_risks = {}
    for ti, traj in enumerate(test_trajs):
        if ti not in risks_ba:
            continue
        n_tokens = len(risks_ba[ti])
        is_inc = traj["verdict"] == "incorrect"
        oracle_risks[ti] = [1.0] * n_tokens if is_inc else [0.0] * n_tokens
    det_oracle = detection_sweep(oracle_risks, test_trajs, K_list=[5])
    oracle_edr = det_oracle.get("K5", {}).get("0.05", {}).get("edr_at_k", 0)
    print(f"  Oracle EDR@5: {oracle_edr:.3f}")
    results["baseline_posthoc"] = {"detection": det_oracle}

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    return results


# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(description="V12: MedQA Pipeline")
    parser.add_argument("--mode", choices=["generate", "evaluate", "both"], default="both")
    parser.add_argument("--data", default="data/medqa.jsonl")
    parser.add_argument("--models", nargs="+", default=["qwen", "llama", "biomistral"])
    parser.add_argument("--traj_base", default="results")
    parser.add_argument("--output_dir", default="results/v12_medqa_pipeline")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for model_name in args.models:
        traj_dir = os.path.join(args.traj_base, f"v9_medqa_{model_name}_traj")
        output_path = os.path.join(args.output_dir, f"v12_medqa_{model_name}.json")

        # Phase 1: Generate trajectories
        if args.mode in ["generate", "both"]:
            print(f"\n{'='*70}")
            print(f"  GENERATING: medqa × {model_name}")
            print(f"{'='*70}")
            generate_trajectories(args.data, model_name, traj_dir, max_samples=args.max_samples)

        # Phase 2: Evaluate
        if args.mode in ["evaluate", "both"]:
            results = run_pipeline(traj_dir, output_path)
            all_results[f"medqa_{model_name}"] = results

    # Save combined
    if args.mode in ["evaluate", "both"] and all_results:
        combined_path = os.path.join(args.output_dir, "v12_combined.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"ALL DONE. Combined: {combined_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
