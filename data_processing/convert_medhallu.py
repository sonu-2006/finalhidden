#!/usr/bin/env python3
"""Convert MedHallu (pqa_labeled) to the pipeline's standard format.

Input:  medhallu_pqa_labeled/train.jsonl
Output: data/medhallu.jsonl
Format: {"id": ..., "context": ..., "question": ..., "answer": ..., ...}

Stratified sampling: balanced by difficulty level, default 500 samples.
"""

import argparse
import json
import random
from pathlib import Path


def convert(input_path: str, output_path: str, max_samples: int = 500, seed: int = 42):
    random.seed(seed)

    raw = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line))

    # Stratified sampling by difficulty level
    by_difficulty = {}
    for item in raw:
        d = item.get("Difficulty Level", "unknown")
        by_difficulty.setdefault(d, []).append(item)

    if max_samples > 0 and max_samples < len(raw):
        selected = []
        per_level = max_samples // len(by_difficulty)
        remainder = max_samples - per_level * len(by_difficulty)
        for level, items in sorted(by_difficulty.items()):
            n = min(per_level, len(items))
            selected.extend(random.sample(items, n))
        # distribute remainder
        for level, items in sorted(by_difficulty.items()):
            if len(selected) >= max_samples:
                break
            already = [i for i in selected if i.get("Difficulty Level") == level]
            remaining = [i for i in items if i not in already]
            take = min(remainder, len(remaining))
            selected.extend(random.sample(remaining, take))
            remainder -= take
        selected = selected[:max_samples]
    else:
        selected = raw

    records = []
    for i, item in enumerate(selected):
        knowledge = item.get("Knowledge", [])
        if isinstance(knowledge, list):
            context = "\n".join(knowledge)
        else:
            context = str(knowledge)

        record = {
            "id": f"medhallu_{i}",
            "context": context,
            "question": item.get("Question", ""),
            "answer": item.get("Ground Truth", ""),
            # Extra metadata for verifier validation & analysis
            "hallucinated_answer": item.get("Hallucinated Answer", ""),
            "difficulty": item.get("Difficulty Level", ""),
            "hallucination_category": item.get("Category of Hallucination", ""),
        }
        records.append(record)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report distribution
    dist = {}
    for r in records:
        d = r.get("difficulty", "unknown")
        dist[d] = dist.get(d, 0) + 1
    print(f"Converted {len(records)} MedHallu samples -> {output_path}")
    print(f"  Difficulty distribution: {dist}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to medhallu_pqa_labeled/train.jsonl")
    ap.add_argument("--output", default="data/medhallu.jsonl")
    ap.add_argument("--max_samples", type=int, default=500, help="0 = use all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    convert(args.input, args.output, args.max_samples, args.seed)
