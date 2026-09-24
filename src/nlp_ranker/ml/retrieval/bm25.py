"""Построение, сохранение и поиск по лексическому индексу"""

import hashlib
import json
import re
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from nlp_ranker.domain.search import SearchHit
from nlp_ranker.ml.config import BM25Config

if TYPE_CHECKING:
    import bm25s


TOKEN_PATTERN = r"(?ui)c\+\+|c#|\.net|\b\w+\b"


class BM25Index:
    """Ищет по сохранённым строкам индекса, не обращаясь к HTTP или базе"""

    def __init__(
        self,
        retriever: "bm25s.BM25",
        documents: list[dict[str, object]],
        metadata: dict[str, object],
    ) -> None:
        """Сохраняет индекс и соответствие его строк документам"""
        self.archive_sha256 = ""
        self.retriever = retriever
        self.documents = documents
        self.metadata = metadata
        ids = [doc["document_id"] for doc in documents]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("В индексе нужны уникальные ID документов")
        if retriever.scores["num_docs"] != len(documents):
            raise ValueError("Размер индекса не совпадает со списком ID")

    @classmethod
    def build(
        cls,
        documents: list[dict[str, object]],
        config: BM25Config,
        data_version: str | None,
        corpus_hash: str,
    ) -> "BM25Index":
        """Индексирует заголовки и текст с единой токенизацией"""
        import bm25s

        if not documents:
            raise ValueError("Корпус пуст")
        texts = [
            (str(doc["title"]) + " ") * config.title_repeat + str(doc["text"])
            for doc in documents
        ]
        tokens = bm25s.tokenize(
            texts, stopwords=[], token_pattern=TOKEN_PATTERN, show_progress=False
        )
        retriever = bm25s.BM25(k1=config.k1, b=config.b, method="lucene")
        retriever.index(tokens, show_progress=False)
        columns = ["document_id", "city", "work_format", "experience"]
        metadata: dict[str, object] = {
            "schema_version": 1,
            "data_version": data_version,
            "corpus_sha256": corpus_hash,
            "token_pattern": TOKEN_PATTERN,
            "bm25": config.model_dump(mode="json"),
        }
        rows = [{name: doc.get(name) for name in columns} for doc in documents]
        return cls(retriever, rows, metadata)

    def search(
        self,
        query: str,
        limit: int = 10,
        allowed_ids: set[str] | None = None,
        city: str | None = None,
        work_format: str | None = None,
        experience: str | None = None,
    ) -> list[SearchHit]:
        """Фильтрует кандидатов до top-k, при равенстве сохраняет порядок корпуса"""
        if limit < 1:
            raise ValueError("Лимит должен быть положительным")
        tokens = re.findall(TOKEN_PATTERN, query.lower())
        tokens = [token for token in tokens if token in self.retriever.vocab_dict]
        if not tokens:
            return []
        scores = self.retriever.get_scores(tokens)
        filters = {"city": city, "work_format": work_format, "experience": experience}
        candidates = []
        for position, doc in enumerate(self.documents):
            if allowed_ids is not None and str(doc["document_id"]) not in allowed_ids:
                continue
            if any(
                value is not None and doc.get(name) != value
                for name, value in filters.items()
            ):
                continue
            if float(scores[position]) > 0:
                candidates.append(position)
        candidates.sort(key=lambda position: (-float(scores[position]), position))
        return [
            SearchHit(str(self.documents[pos]["document_id"]), float(scores[pos]), "")
            for pos in candidates[:limit]
        ]

    def save(self, path: Path) -> None:
        """Сохраняет индекс и его метаданные одним архивом без pickle"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as directory:
            folder = Path(directory)
            self.retriever.save(str(folder), show_progress=False)
            (folder / "documents.json").write_text(
                json.dumps(self.documents, ensure_ascii=False), encoding="utf-8"
            )
            (folder / "metadata.json").write_text(
                json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8"
            )
            archive_path = folder / "index.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(folder.iterdir()):
                    if file != archive_path:
                        info = zipfile.ZipInfo(
                            file.name, date_time=(1980, 1, 1, 0, 0, 0)
                        )
                        info.compress_type = zipfile.ZIP_DEFLATED
                        archive.writestr(info, file.read_bytes())
            archive_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Загружает индекс в память и проверяет версию формата"""
        import bm25s

        with TemporaryDirectory() as directory, path.open("rb") as stream:
            archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                for name in archive.namelist():
                    if Path(name).name != name:
                        raise ValueError("Архив индекса должен содержать только файлы")
                archive.extractall(directory)
            folder = Path(directory)
            metadata = json.loads(
                (folder / "metadata.json").read_text(encoding="utf-8")
            )
            if (
                metadata["schema_version"] != 1
                or metadata["token_pattern"] != TOKEN_PATTERN
            ):
                raise ValueError("Несовместимая версия индекса или токенизации")
            retriever = bm25s.BM25.load(
                directory, allow_pickle=False, show_progress=False
            )
            documents = json.loads(
                (folder / "documents.json").read_text(encoding="utf-8")
            )
        index = cls(retriever, documents, metadata)
        index.archive_sha256 = archive_hash
        return index
