"""
Retrieval evaluation metrics: Recall@k, MRR, nDCG, Precision@k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    precision_at_5: float


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in relevant if r in set(retrieved[:k]))
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for r in retrieved[:k] if r in relevant_set) / k


def mean_reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    relevant_set = set(relevant)
    top_k = retrieved[:k]

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant_set
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def compute_all(retrieved: list[str], relevant: list[str]) -> RetrievalMetrics:
    """Compute all standard retrieval metrics for a single query."""
    return RetrievalMetrics(
        recall_at_1=recall_at_k(retrieved, relevant, k=1),
        recall_at_3=recall_at_k(retrieved, relevant, k=3),
        recall_at_5=recall_at_k(retrieved, relevant, k=5),
        recall_at_10=recall_at_k(retrieved, relevant, k=10),
        mrr=mean_reciprocal_rank(retrieved, relevant),
        ndcg_at_5=ndcg_at_k(retrieved, relevant, k=5),
        ndcg_at_10=ndcg_at_k(retrieved, relevant, k=10),
        precision_at_5=precision_at_k(retrieved, relevant, k=5),
    )


def aggregate(metrics_list: list[RetrievalMetrics]) -> dict[str, float]:
    """Average metrics across a list of per-query results."""
    n = len(metrics_list)
    if n == 0:
        return {}

    keys = ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
            "mrr", "ndcg_at_5", "ndcg_at_10", "precision_at_5"]
    return {k: sum(getattr(m, k) for m in metrics_list) / n for k in keys}
