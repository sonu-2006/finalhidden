"""Answer-level claim verification using text similarity and NLI.

Provides ROUGE-L and Jaccard-based verification for determining
whether a generated answer matches the reference answer.
"""

from src.text_utils import jaccard_char, rouge_l_f1_char, rouge_l_f1_word


def verify_claim_jaccard(
    generated_text: str, reference: str, threshold: float = 0.5, min_len: int = 4
) -> str:
    """Verify answer correctness using character-level Jaccard similarity."""
    if not generated_text or not reference:
        return "unknown"
    if len(generated_text) < min_len or len(reference) < min_len:
        return "unknown"
    score = jaccard_char(generated_text, reference)
    return "correct" if score >= threshold else "incorrect"


def verify_claim_rouge(
    generated_text: str, reference: str, threshold: float = 0.12, min_len: int = 4
) -> str:
    """Verify answer correctness using word-level ROUGE-L F1 (default)."""
    if not generated_text or not reference:
        return "unknown"
    if len(generated_text) < min_len or len(reference) < min_len:
        return "unknown"
    score = rouge_l_f1_word(generated_text, reference)
    return "correct" if score >= threshold else "incorrect"


def verify_claim_rouge_char(
    generated_text: str, reference: str, threshold: float = 0.3, min_len: int = 4
) -> str:
    """Verify answer correctness using character-level ROUGE-L F1."""
    if not generated_text or not reference:
        return "unknown"
    if len(generated_text) < min_len or len(reference) < min_len:
        return "unknown"
    score = rouge_l_f1_char(generated_text, reference)
    return "correct" if score >= threshold else "incorrect"
