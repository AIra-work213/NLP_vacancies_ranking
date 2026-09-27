"""Инференс энкодера и токенное разбиение документов"""

import asyncio

from nlp_ranker.ml.config import DenseConfig


class Encoder:
    """Кодирует документы и запросы с обязательными префиксами E5"""

    def __init__(self, config: DenseConfig) -> None:
        """Загружает фиксированную модель и ограничивает CPU-потоки"""
        import torch
        from sentence_transformers import SentenceTransformer

        if not config.model_name or not config.revision:
            raise ValueError("Задайте модель эмбеддингов и revision")
        torch.set_num_threads(config.threads)
        self.config = config
        self.model = SentenceTransformer(
            config.model_name,
            revision=config.revision,
            device=config.device,
            trust_remote_code=False,
        )
        self.model.max_seq_length = config.max_length
        self.model.eval()
        if self.model.get_embedding_dimension() != config.dimensions:
            raise ValueError("Размерность модели не совпадает с Vector(384)")
        self.lock = asyncio.Lock()

    def encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        """Выполняет пакетный инференс без градиентов и нормирует векторы"""
        import torch

        if not texts:
            return []
        with torch.inference_mode():
            vectors = self.model.encode(
                [prefix + text for text in texts],
                batch_size=self.config.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [[float(value) for value in vector] for vector in vectors]

    async def query(self, text: str) -> list[float]:
        """Удерживает слот до реального завершения инференса вне цикла событий"""
        async with self.lock:
            task = asyncio.create_task(
                asyncio.to_thread(self.encode, [text], "query: ")
            )
            try:
                vectors = await asyncio.shield(task)
            except asyncio.CancelledError:
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                raise
            return vectors[0]

    def chunks(self, title: str, text: str) -> list[str]:
        """Повторяет заголовок в токенных фрагментах с небольшим перекрытием"""
        tokenizer = self.model.tokenizer
        title_ids = tokenizer.encode(title, add_special_tokens=False)[:48]
        heading = tokenizer.decode(title_ids, skip_special_tokens=True)
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        step = self.config.chunk_tokens - self.config.overlap_tokens
        chunks = []
        for start in range(0, max(1, len(tokens)), step):
            part = tokens[start : start + self.config.chunk_tokens]
            chunks.append(
                heading + "\n" + tokenizer.decode(part, skip_special_tokens=True)
            )
            if start + self.config.chunk_tokens >= len(tokens):
                break
        return chunks
