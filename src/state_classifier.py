"""Probe MLP training for three-state token classification.

Trains a 2-layer MLP on hidden-state features to classify tokens
as Safe, AtRisk, or Hallucinating.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.hidden_utils import load_hidden_from_npz


STATE2ID = {"Safe": 0, "AtRisk": 1, "Hallucinating": 2}


class MLP(torch.nn.Module):
    def __init__(self, hidden_dim=4096, num_classes=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_dataset(label_jsonl, layer_idx=None, layer_pool="mean", history=1):
    xs = []
    ys = []
    with open(label_jsonl, "r", encoding="utf-8") as f:
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
            n = min(len(hidden), len(states))
            for i in range(n):
                xs.append(hidden[i])
                ys.append(STATE2ID.get(states[i], 0))
    X = np.stack(xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    return X, y


def train_classifier(
    label_jsonl,
    out_path,
    epochs=5,
    lr=1e-3,
    batch_size=256,
    layer_idx=None,
    layer_pool="mean",
    history=1,
):
    X, y = load_dataset(label_jsonl, layer_idx=layer_idx, layer_pool=layer_pool, history=history)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(hidden_dim=X.shape[1]).to(device)

    counts = np.bincount(y_train)
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.sum()
    weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

    model.train()
    for _ in range(epochs):
        idx = torch.randperm(X_train_t.shape[0])
        for start in range(0, X_train_t.shape[0], batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_test_t).cpu().numpy()
        preds = np.argmax(logits, axis=1)

    acc = accuracy_score(y_test, preds)
    f1s = f1_score(y_test, preds, average=None, labels=[0, 1, 2])

    out = {
        "accuracy": float(acc),
        "f1_safe": float(f1s[0]),
        "f1_atrisk": float(f1s[1]),
        "f1_hallucinating": float(f1s[2]),
        "layer_idx": layer_idx,
        "layer_pool": layer_pool,
        "history": history,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    print("metrics", out)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--layer_idx", type=int, default=None)
    parser.add_argument("--layer_pool", type=str, default="mean", choices=["mean", "concat"])
    parser.add_argument("--history", type=int, default=1)
    args = parser.parse_args()

    train_classifier(
        args.labels,
        args.model_out,
        epochs=args.epochs,
        layer_idx=args.layer_idx,
        layer_pool=args.layer_pool,
        history=args.history,
    )
