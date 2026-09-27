"""Объединение ранжированных результатов BM25 и dense"""

from nlp_ranker.domain.search import SearchHit
from nlp_ranker.ml.config import RetrievalConfig


def fuse_results(
    lexical: list[SearchHit],
    dense: list[SearchHit],
    config: RetrievalConfig,
    limit: int = 100,
) -> list[SearchHit]:
    """Объединяет уникальные документы по RRF или нормированным оценкам"""
    if limit < 1 or config.fusion is None:
        raise ValueError("Укажите способ объединения и положительный лимит")
    scores: dict[str, float] = {}
    for hits, weight in [
        (lexical, 1 - config.dense_weight),
        (dense, config.dense_weight),
    ]:
        if config.fusion == "weighted_sum" and weight == 0:
            continue
        unique = []
        seen = set()
        for hit in hits:
            if hit.document_id not in seen:
                unique.append(hit)
                seen.add(hit.document_id)
        values = [hit.score for hit in unique]
        low, high = (min(values), max(values)) if values else (0.0, 0.0)
        for rank, hit in enumerate(unique, start=1):
            if config.fusion == "rrf":
                contribution = 1 / (config.rrf_k + rank)
            else:
                normalized = (hit.score - low) / (high - low) if high > low else 1.0
                contribution = weight * normalized
            scores[hit.document_id] = scores.get(hit.document_id, 0.0) + contribution
    ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [SearchHit(doc_id, scores[doc_id], "") for doc_id in ids[:limit]]
