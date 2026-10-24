"""Trajectory collection with hidden-state extraction.

Generates multiple answer trajectories per prompt using a causal LM,
saving hidden states, token probabilities, and generated text.
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.model_loader import load_model


def default_prompt_fn(sample: dict) -> str:
    """Default prompt template for medical QA."""
    return "Reference: {}\nQuestion: {}\nPlease answer based on the reference.".format(
        sample["context"], sample["question"]
    )


def _parse_layers(spec):
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def collect_trajectories(
    data_path,
    model_name,
    output_dir,
    max_samples=50,
    n_gen=1,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    save_layers="",
):
    output_dir = Path(output_dir)
    hidden_dir = output_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model(model_name)

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    layer_ids = _parse_layers(save_layers)

    out_jsonl = output_dir / "trajectories.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f_out:
        for sample in tqdm(samples, desc="collect"):
            prompt = default_prompt_fn(sample)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            for g in range(n_gen):
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        top_p=top_p,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        output_scores=True,
                    )

                generated_ids = gen.sequences[0, inputs["input_ids"].shape[1] :]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                hidden_steps = []
                hidden_layers_steps = []
                for step_h in gen.hidden_states:
                    last_layer = (
                        step_h[-1][0, -1, :].detach().to(torch.float16).cpu().numpy()
                    )
                    hidden_steps.append(last_layer)
                    if layer_ids:
                        layers = []
                        for lid in layer_ids:
                            layers.append(
                                step_h[lid][0, -1, :]
                                .detach()
                                .to(torch.float16)
                                .cpu()
                                .numpy()
                            )
                        hidden_layers_steps.append(layers)
                hidden_steps = np.stack(hidden_steps, axis=0) if hidden_steps else np.zeros((0,))

                max_probs = []
                for score in gen.scores:
                    probs = torch.softmax(score[0], dim=-1)
                    max_probs.append(float(torch.max(probs).detach().cpu().item()))
                max_probs = np.array(max_probs, dtype=np.float32)

                base_id = str(sample.get("id"))
                sample_id = f"{base_id}_{g}" if n_gen > 1 else base_id
                save_path = hidden_dir / (sample_id + ".npz")
                if layer_ids and hidden_layers_steps:
                    hidden_layers = np.stack(hidden_layers_steps, axis=0).astype(np.float16)
                    np.savez_compressed(
                        save_path,
                        hidden=hidden_steps.astype(np.float16),
                        hidden_layers=hidden_layers,
                        layer_ids=np.array(layer_ids, dtype=np.int64),
                        max_probs=max_probs,
                        token_ids=generated_ids.detach().cpu().numpy(),
                    )
                else:
                    np.savez_compressed(
                        save_path,
                        hidden=hidden_steps.astype(np.float16),
                        max_probs=max_probs,
                        token_ids=generated_ids.detach().cpu().numpy(),
                    )

                record = {
                    "id": sample_id,
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "answer": sample.get("answer", ""),
                    "hidden_path": str(save_path),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("wrote", out_jsonl)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--n_gen", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--save_layers", type=str, default="")
    args = parser.parse_args()

    collect_trajectories(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        n_gen=args.n_gen,
        top_p=args.top_p,
        save_layers=args.save_layers,
    )
