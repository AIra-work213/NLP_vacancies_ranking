"""Пакетное ранжирование готовым многоязычным cross-encoder на CPU"""

import asyncio
from typing import Any

from nlp_ranker.domain.search import SearchHit
from nlp_ranker.ml.config import RerankingConfig


class Reranker:
    """Оценивает фрагменты вакансий и оставляет лучшую оценку каждого документа"""

    def __init__(self, config: RerankingConfig) -> None:
        """Загружает фиксированную модель без исполнения кода репозитория"""
        import torch
        from sentence_transformers import CrossEncoder

        if not config.model_name or not config.revision:
            raise ValueError("Задайте модель reranker и фиксированную revision")
        torch.set_num_threads(config.threads)
        self.config = config
        self.model = CrossEncoder(
            config.model_name,
            revision=config.revision,
            device="cpu",
            max_length=config.max_length,
            trust_remote_code=False,
        )
        self.model.model.eval()
        self.lock = asyncio.Lock()

    def rank(
        self,
        query: str,
        hits: list[SearchHit],
        chunks: dict[str, list[str]],
        limit: int = 10,
    ) -> list[SearchHit]:
        """Считает logits, при равенстве сохраняет порядок кандидатов"""
        import math

        import torch

        if limit < 1:
            raise ValueError("Лимит должен быть положительным")
        if not query.strip() or not hits:
            return []
        ids = list(dict.fromkeys(hit.document_id for hit in hits))
        pairs = []
        owners = []
        tokenizer = self.model.tokenizer
        query_ids = tokenizer.encode(query, add_special_tokens=False)[
            : min(96, self.config.max_length // 4)
        ]
        short_query = tokenizer.decode(query_ids, skip_special_tokens=True)
        for doc_id in ids:
            passages = chunks.get(doc_id)
            if not passages:
                raise LookupError("Нет фрагментов кандидата для reranker")
            for passage in passages:
                title, _, body = passage.partition("\n")
                title_ids = tokenizer.encode(title, add_special_tokens=False)[
                    : min(48, self.config.max_length // 4)
                ]
                body_ids = tokenizer.encode(body, add_special_tokens=False)
                budget = self.config.max_length - len(query_ids) - len(title_ids) - 8
                text = tokenizer.decode(title_ids, skip_special_tokens=True) + "\n"
                text += tokenizer.decode(
                    body_ids[: max(0, budget)], skip_special_tokens=True
                )
                pairs.append((short_query, text))
                owners.append(doc_id)
        with torch.inference_mode():
            scores: Any = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                activation_fn=torch.nn.Identity(),
            )
        best: dict[str, float] = {}
        for doc_id, value in zip(owners, scores, strict=True):
            score = float(value)
            if not math.isfinite(score):
                raise ValueError("Reranker вернул неконечную оценку")
            best[doc_id] = max(score, best.get(doc_id, -math.inf))
        positions = {doc_id: pos for pos, doc_id in enumerate(ids)}
        ordered = sorted(best, key=lambda doc_id: (-best[doc_id], positions[doc_id]))
        return [SearchHit(doc_id, best[doc_id], "") for doc_id in ordered[:limit]]

    async def search(
        self,
        query: str,
        hits: list[SearchHit],
        chunks: dict[str, list[str]],
        limit: int,
    ) -> list[SearchHit]:
        """Выполняет один инференс и дожидается его при отмене"""
        async with self.lock:
            task = asyncio.create_task(
                asyncio.to_thread(self.rank, query, hits, chunks, limit)
            )
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                raise
