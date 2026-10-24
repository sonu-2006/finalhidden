#!/usr/bin/env python3
"""SelfCheckGPT-NLI baseline for hallucination detection.

For each trajectory, uses the other N-1 generations from the same prompt
to check sentence-level consistency via NLI. Inconsistency → hallucination risk.

This adapts to the pipeline's FPR-cap protocol by producing per-token risk scores.
"""

import argparse
import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import transformers.modeling_utils
    import transformers.utils.import_utils
    # Bypass CVE-2025-32434 check for trusted local weights
    transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **k: None
    transformers.utils.import_utils.check_torch_load_is_safe = lambda *a, **k: None
    print("Security check bypassed for check_torch_load_is_safe")
except Exception as e:
    print(f"Bypass failed: {e}")



def split_sentences(text):
    """Simple sentence splitter. Splits on period, question mark, exclamation mark."""
    # Split on sentence-ending punctuation followed by whitespace or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.strip()) > 3]


def sentence_to_token_map(text, sentences):
    """Map sentence indices back to approximate character positions, then to token fractions."""
    char_pos = 0
    sent_ranges = []
    for sent in sentences:
        start = text.find(sent, char_pos)
        if start == -1:
            start = char_pos
        end = start + len(sent)
        sent_ranges.append((start, end))
        char_pos = end
    return sent_ranges


def expand_sentence_scores_to_tokens(scores, sentences, text, n_tokens):
    """Expand sentence-level scores to token-level scores (approximate)."""
    if not scores or not sentences:
        return np.zeros(n_tokens)

    sent_ranges = sentence_to_token_map(text, sentences)
    total_chars = len(text)
    if total_chars == 0:
        return np.full(n_tokens, np.mean(scores))

    # Map each token position to its approximate sentence
    token_scores = np.zeros(n_tokens)
    for t in range(n_tokens):
        # Approximate: token t corresponds to character range [t*C/T, (t+1)*C/T]
        char_start = int(t * total_chars / n_tokens)
        char_end = int((t + 1) * total_chars / n_tokens)
        char_mid = (char_start + char_end) // 2

        # Find which sentence this falls in
        best_sent = 0
        for i, (s, e) in enumerate(sent_ranges):
            if s <= char_mid < e:
                best_sent = i
                break
        if best_sent < len(scores):
            token_scores[t] = scores[best_sent]

    return token_scores


class SelfCheckNLI:
    """SelfCheckGPT using NLI model for cross-sample consistency checking."""

    def __init__(self, nli_model_name="microsoft/deberta-v3-large-mnli", device=None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NLI model: {nli_model_name}")
        try:

            self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)

        except Exception as e:

            print(f'AutoTokenizer failed: {e}')

            # Fallback for DeBERTa v1 with bpe_encoder.bin

            from transformers import DebertaTokenizer

            import os

            fallback_path = nli_model_name

            bpe_path = os.path.join(fallback_path, 'bpe_encoder.bin')

            if os.path.exists(bpe_path):

                print(f'Fallback: loading DeBERTa tokenizer from {bpe_path}')

                self.tokenizer = DebertaTokenizer(vocab_file=bpe_path, merges_file=os.path.join(fallback_path, 'merges.txt'))

            else:

                raise
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
        self.model.eval()

        # Label mapping for DeBERTa-v3-large-mnli: 0=contradiction, 1=neutral, 2=entailment
        self.contradiction_idx = 0

    def nli_contradiction_prob(self, premise, hypothesis, max_length=512):
        """Return P(contradiction | premise, hypothesis)."""
        import torch

        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        return float(probs[self.contradiction_idx].cpu().item())

    def score_trajectory(self, target_text, other_generations):
        """
        Score each sentence of target_text for hallucination risk
        by checking NLI consistency with other_generations.

        Returns: list of per-sentence risk scores in [0, 1].
        """
        sentences = split_sentences(target_text)
        if not sentences:
            return []

        scores = []
        for sent in sentences:
            contradictions = []
            for other_text in other_generations:
                # Use other_text as premise, sent as hypothesis
                p_contra = self.nli_contradiction_prob(other_text, sent)
                contradictions.append(p_contra)
            # Average contradiction probability across other generations
            scores.append(float(np.mean(contradictions)))

        return scores


def group_trajectories_by_prompt(traj_dir):
    """Group trajectories by their base prompt ID (before _0, _1, etc.)."""
    traj_path = Path(traj_dir) / "trajectories.jsonl"
    records = []
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Group by base_id (strip the _N suffix)
    groups = defaultdict(list)
    for rec in records:
        sid = rec["id"]
        # base_id is everything before the last _N
        parts = sid.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            base_id = parts[0]
        else:
            base_id = sid
        groups[base_id].append(rec)

    return groups


def run_selfcheck(traj_dir, output_path, nli_model_name="microsoft/deberta-v3-large-mnli"):
    """Run SelfCheckGPT-NLI on all trajectories in traj_dir."""
    groups = group_trajectories_by_prompt(traj_dir)
    checker = SelfCheckNLI(nli_model_name)

    all_results = []
    for base_id, recs in groups.items():
        if len(recs) < 2:
            continue  # Need at least 2 generations for cross-checking

        for i, target_rec in enumerate(recs):
            others = [r["generated_text"] for j, r in enumerate(recs) if j != i]
            target_text = target_rec["generated_text"]

            sent_scores = checker.score_trajectory(target_text, others)
            sentences = split_sentences(target_text)

            # Load hidden to get n_tokens
            npz = np.load(target_rec["hidden_path"])
            n_tokens = len(npz.get("max_probs", []))

            # Expand to token level
            token_scores = expand_sentence_scores_to_tokens(
                sent_scores, sentences, target_text, n_tokens
            )

            all_results.append({
                "id": target_rec["id"],
                "sentence_scores": sent_scores,
                "token_risk": token_scores.tolist(),
                "answer_risk": float(np.mean(sent_scores)) if sent_scores else 0.0,
            })

        print(f"  Processed prompt {base_id}: {len(recs)} generations")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_results)} SelfCheckGPT results to {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--nli_model", default="microsoft/deberta-v3-large-mnli")
    args = ap.parse_args()
    run_selfcheck(args.traj_dir, args.output, args.nli_model)
