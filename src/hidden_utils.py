"""Hidden state I/O and history pooling utilities.

Loads hidden states from .npz files with optional layer selection,
multi-layer fusion, and temporal history pooling.
"""

import numpy as np


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Apply causal rolling mean along the time axis."""
    if window <= 1:
        return x
    out = np.zeros_like(x)
    cumsum = np.cumsum(x, axis=0)
    for i in range(x.shape[0]):
        start = max(0, i - window + 1)
        if start == 0:
            out[i] = cumsum[i] / (i + 1)
        else:
            out[i] = (cumsum[i] - cumsum[start - 1]) / (i - start + 1)
    return out


def _resolve_layer_position(layer_idx: int, data: dict) -> int:
    """Map a model layer number to its position in the hidden_layers axis.

    If the .npz contains a 'layer_ids' array (e.g. [4, 20, 32]),
    looks up the position of layer_idx in that array.
    Falls back to using layer_idx as a direct index.
    """
    if "layer_ids" not in data:
        return int(layer_idx)

    layer_ids = data["layer_ids"]
    num_layers = data["hidden_layers"].shape[1]

    matches = np.where(layer_ids == int(layer_idx))[0]
    if matches.size > 0:
        return int(matches[0])

    if 0 <= int(layer_idx) < num_layers:
        return int(layer_idx)

    raise IndexError(
        f"layer_idx={layer_idx} not found in layer_ids={layer_ids.tolist()} "
        f"and out of bounds for hidden_layers with {num_layers} layers"
    )


def load_hidden_from_npz(
    npz_path: str,
    layer_idx: int | list[int] | None = None,
    layer_pool: str = "mean",
    history: int = 1,
) -> np.ndarray:
    """Load hidden states from a .npz file.

    Args:
        npz_path: Path to the .npz file.
        layer_idx: Layer(s) to extract. None uses the default 'hidden' key.
        layer_pool: How to combine multiple layers ('concat' or 'mean').
        history: Temporal pooling window size (1 = no pooling).

    Returns:
        Array of shape (num_tokens, hidden_dim).
    """
    data = np.load(npz_path)

    if layer_idx is None or "hidden_layers" not in data:
        hidden = data["hidden"]
        if hidden.ndim != 2:
            hidden = hidden.reshape(hidden.shape[0], -1)
    else:
        hidden_layers = data["hidden_layers"]
        if isinstance(layer_idx, (list, tuple)):
            positions = [_resolve_layer_position(li, data) for li in layer_idx]
            layers = hidden_layers[:, positions, :]
            if layer_pool == "concat":
                hidden = layers.reshape(layers.shape[0], -1)
            else:
                hidden = layers.mean(axis=1)
        else:
            pos = _resolve_layer_position(layer_idx, data)
            hidden = hidden_layers[:, pos, :]

    if history and history > 1:
        hidden = _rolling_mean(hidden, history)

    return hidden
