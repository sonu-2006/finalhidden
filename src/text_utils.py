"""Text similarity utilities for verifier-based supervision.

Provides character-level and word-level ROUGE-L, Jaccard similarity,
and sentence splitting for Chinese/English medical text.
"""

import re


SENT_SPLIT_RE = re.compile(r"[。！？!?]\s*")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using Chinese/English punctuation."""
    if not text:
        return []
    return [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]


def jaccard_char(a: str, b: str) -> float:
    """Character-level Jaccard similarity."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _lcs_len(a, b) -> int:
    """Longest common subsequence length (DP with rolling rows)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[-1]


def rouge_l_f1_char(a: str, b: str) -> float:
    """Character-level ROUGE-L F1 score."""
    if not a or not b:
        return 0.0
    lcs = _lcs_len(a, b)
    if lcs == 0:
        return 0.0
    prec = lcs / len(a)
    rec = lcs / len(b)
    return (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0


def rouge_l_f1_word(reference: str, hypothesis: str) -> float:
    """Word-level ROUGE-L F1 score (default verifier metric)."""
    if not reference or not hypothesis:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_len(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
