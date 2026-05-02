"""
Generation quality metrics: ROUGE, BLEU, and answer completeness.

Used for offline eval against reference answers. Note that ROUGE/BLEU are
weak proxies for LLM output quality — use LLM-judge (faithfulness, relevance)
as the primary signals. ROUGE is retained for fast CI checks that don't
incur LLM API costs.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class GenerationMetrics:
    rouge_l: float          # ROUGE-L F1: longest common subsequence
    bleu_1: float           # BLEU-1: unigram precision with brevity penalty
    answer_completeness: float  # Fraction of required keywords present
    contains_refusal: bool  # True if answer is a refusal pattern


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 based on longest common subsequence of tokens."""
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs_len / len(hyp_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu_1(hypothesis: str, reference: str) -> float:
    """BLEU-1: unigram precision with brevity penalty."""
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)

    clipped = sum(min(count, ref_counts[token]) for token, count in hyp_counts.items())
    precision = clipped / len(hyp_tokens)

    # Brevity penalty
    if len(hyp_tokens) < len(ref_tokens):
        bp = (len(hyp_tokens) / len(ref_tokens)) ** 0.5
    else:
        bp = 1.0

    return bp * precision


def compute_answer_completeness(answer: str, required_keywords: list[str]) -> float:
    """Fraction of required keywords present in the answer (case-insensitive)."""
    if not required_keywords:
        return 1.0
    answer_lower = answer.lower()
    present = sum(1 for kw in required_keywords if kw.lower() in answer_lower)
    return present / len(required_keywords)


def compute_generation_metrics(
    answer: str,
    reference: str = "",
    required_keywords: list[str] | None = None,
) -> GenerationMetrics:
    refusal_patterns = [
        r"i('m| am) unable to",
        r"i cannot (help|answer)",
        r"that('s| is) outside (my|the)",
        r"i don'?t have (access|information)",
    ]
    is_refusal = any(re.search(p, answer.lower()) for p in refusal_patterns)

    return GenerationMetrics(
        rouge_l=compute_rouge_l(answer, reference) if reference else 0.0,
        bleu_1=compute_bleu_1(answer, reference) if reference else 0.0,
        answer_completeness=compute_answer_completeness(answer, required_keywords or []),
        contains_refusal=is_refusal,
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence (space-optimized DP)."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]
