#!/usr/bin/env python3
"""Convert PubMedQA (pqa_labeled) to the pipeline's standard format.

Input:  pubmed_qa_pqa_labeled/train.jsonl
Output: data/pubmedqa.jsonl
Format: {"id": ..., "context": ..., "question": ..., "answer": ...}
"""

import argparse
import json
from pathlib import Path


def convert(input_path: str, output_path: str, max_samples: int = 0):
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # PubMedQA format:
            #   context.contexts[] -> join as context
            #   question -> question
            #   long_answer -> answer
            ctx = item.get("context", {})
            if isinstance(ctx, dict):
                paragraphs = ctx.get("contexts", [])
            elif isinstance(ctx, list):
                paragraphs = ctx
            else:
                paragraphs = [str(ctx)]

            record = {
                "id": str(item.get("pubid", len(records))),
                "context": "\n".join(paragraphs),
                "question": item["question"],
                "answer": item.get("long_answer", ""),
                # metadata (not used by pipeline but useful for analysis)
                "final_decision": item.get("final_decision", ""),
            }
            records.append(record)

    if max_samples > 0:
        records = records[:max_samples]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Converted {len(records)} PubMedQA samples -> {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to pubmed_qa_pqa_labeled/train.jsonl")
    ap.add_argument("--output", default="data/pubmedqa.jsonl")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = use all")
    args = ap.parse_args()
    convert(args.input, args.output, args.max_samples)
