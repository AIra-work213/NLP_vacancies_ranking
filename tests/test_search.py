"""Проверки лексического индекса и его восстановления"""

import json
import subprocess
import sys

import pytest

pytest.importorskip("bm25s")

from nlp_ranker.ml.config import BM25Config
from nlp_ranker.ml.retrieval.bm25 import BM25Index


def documents():
    """Создаёт две вакансии с разными фильтрами"""
    return [
        {
            "document_id": "first",
            "title": "Python Python разработчик",
            "text": "SQL C++",
            "city": "Москва",
            "work_format": None,
            "experience": "1_to_3_years",
        },
        {
            "document_id": "second",
            "title": "Python разработчик",
            "text": "SQL C# .NET",
            "city": "Казань",
            "work_format": "remote",
            "experience": "no_experience",
        },
    ]


def index():
    """Строит маленький индекс без внешних сервисов"""
    return BM25Index.build(documents(), BM25Config(), "test-v1", "corpus-hash")


def test_filters_before_top_k():
    """Находит подходящий документ даже когда общий top-1 не проходит фильтр"""
    search = index()
    assert search.search("Python", 1)[0].document_id == "first"
    hits = search.search(
        "Python", 1, city="Казань", work_format="remote", experience="no_experience"
    )
    assert [hit.document_id for hit in hits] == ["second"]
    assert search.search("Python", allowed_ids=set()) == []
    assert search.search("Python", allowed_ids={"second"})[0].document_id == "second"
    assert search.search("Python", city="Сочи") == []


def test_technology_tokens_and_empty_query():
    """Различает C++ и C# и не выдаёт произвольные документы на неизвестный запрос"""
    search = index()
    assert search.search("C++")[0].document_id == "first"
    assert search.search("c#")[0].document_id == "second"
    assert search.search(".NET")[0].document_id == "second"
    assert search.search("несуществующийтокен") == []
    assert search.search("?!") == []
    assert search.search(" ") == []
    with pytest.raises(ValueError):
        search.search("Python", 0)


def test_save_is_repeatable_and_loads_in_new_process(tmp_path):
    """Сохраняет одинаковые байты и восстанавливает выдачу в другом процессе"""
    search = index()
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    search.save(first)
    search.save(second)
    assert first.read_bytes() == second.read_bytes()
    code = """
from pathlib import Path
import json
import sys
from nlp_ranker.ml.retrieval.bm25 import BM25Index
index = BM25Index.load(Path(sys.argv[1]))
print(json.dumps([(hit.document_id, hit.score) for hit in index.search('Python')]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(first)],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == [
        [hit.document_id, hit.score] for hit in search.search("Python")
    ]


def test_equal_scores_and_duplicate_ids():
    """Сохраняет порядок корпуса при одинаковых оценках и запрещает повтор ID"""
    rows = documents()
    rows[1]["title"] = rows[0]["title"]
    rows[1]["text"] = rows[0]["text"]
    search = BM25Index.build(rows, BM25Config(), "v1", "hash")
    assert [hit.document_id for hit in search.search("Python")] == ["first", "second"]
    rows[1]["document_id"] = rows[0]["document_id"]
    with pytest.raises(ValueError, match="уникальные"):
        BM25Index.build(rows, BM25Config(), "v1", "hash")


def test_loaded_index_keeps_its_version_after_replacement(tmp_path):
    """Загруженный индекс сохраняет версию после атомарной замены файла"""
    from nlp_ranker.ml.evaluation.metrics import checksum

    path = tmp_path / "index.zip"
    index().save(path)
    loaded = BM25Index.load(path)
    original_hash = checksum(path)
    rows = documents()
    rows[0]["title"] = "Другая вакансия"
    BM25Index.build(rows, BM25Config(), "v2", "other-hash").save(path)
    assert loaded.archive_sha256 == original_hash
    assert loaded.archive_sha256 != checksum(path)
    assert loaded.search("Python")[0].document_id == "first"
