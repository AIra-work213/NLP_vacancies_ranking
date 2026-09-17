"""Основные проверки анализа вакансий"""

import pytest

from nlp_ranker.ml.evaluation.metrics import require_judged_results
from nlp_ranker.pipelines.data import (
    clean_text,
    inspect_source,
    prepare,
    prepare_labels,
)


def test_source_statistics(tmp_path):
    """Учитывает пропуски, повторные описания и ID разных источников"""
    path = tmp_path / "vacancies.csv"
    path.write_text(
        "data_source;id;name;description\n"
        "hh;1;Разработчик;Python\n"
        "rvr;1;Разработчик;Python\n"
        "hh;1;Разработчик;Java\n"
        "hh;2;;\n",
        encoding="utf-8",
    )

    report = inspect_source(path)

    assert report["rows"] == 4
    assert report["usable_required_fields"] == 3
    assert report["missing_values"] == {
        "data_source": 0,
        "id": 0,
        "name": 1,
        "description": 1,
    }
    assert report["duplicate_source_ids"] == 1
    assert report["duplicate_descriptions"] == 1
    assert report["description_characters"] == {"min": 4, "median": 6, "max": 6}


def test_missing_columns(tmp_path):
    """Сообщает понятную ошибку при выборе CSV другого формата"""
    path = tmp_path / "invalid.csv"
    path.write_text("text\nВакансия\n", encoding="utf-8")

    with pytest.raises(ValueError, match="обязательные поля"):
        inspect_source(path)


def test_clean_text():
    """Сохраняет абзацы и названия технологий, удаляя разметку и скрипты"""
    assert (
        clean_text(
            "<p> Python &amp; C++ </p><ul><li> .NET   SQL </li></ul>"
            "<script>private()</script><p>Удалённая работа</p>"
        )
        == "Python & C++\n.NET SQL\nУдалённая работа"
    )


def test_prepare_parquet(tmp_path):
    """Проверяет очистку, пропуски, группы дублей и устойчивые ID в Parquet"""
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("pandas")
    source = tmp_path / "vacancies.csv"
    output = tmp_path / "documents.parquet"
    source.write_text(
        "data_source;id;name;description;experience;schedule;region_name\n"
        'hh;001;Python;"<p>Python &amp; SQL</p>";1-3;remote;Москва\n'
        'rvr;001;Python;"python   &amp; SQL";unknown;full_day;Москва\n'
        "hh;001;Python;Повтор ID;0;shift;Москва\n"
        "hh;002;Python;<p> </p>;0;remote;Москва\n",
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "seed": 42,
        "input_path": str(source),
        "output_path": str(output),
        "split_group_field": "group_id",
    }

    prepare(config)
    documents = pq.read_table(output).to_pylist()

    assert len(documents) == 2
    assert documents[0]["source_id"] == "001"
    assert documents[0]["document_id"] != documents[1]["document_id"]
    assert documents[0]["group_id"] == documents[1]["group_id"]
    assert documents[0]["text"] == "Python & SQL"
    assert documents[0]["experience"] == "1_to_3_years"
    assert documents[1]["experience"] is None
    assert documents[0]["work_format"] == "remote"
    assert documents[1]["work_format"] is None
    assert documents[0]["city"] is None
    assert documents[0]["region"] == "Москва"
    assert (
        pq.read_table(output).schema.field("city").type
        == pq.read_table(output).schema.field("title").type
    )

    prepare(config)
    assert pq.read_table(output).to_pylist() == documents


def test_label_splits(tmp_path):
    """Сохраняет оценки и запрещает утечку группы документов между разбиениями"""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("pandas")
    corpus = tmp_path / "documents.parquet"
    queries = tmp_path / "queries.csv"
    labels = tmp_path / "qrels.csv"
    output = tmp_path / "processed"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"document_id": "doc-1", "group_id": "duplicate-group"},
                {"document_id": "doc-2", "group_id": "duplicate-group"},
            ]
        ),
        corpus,
    )
    queries.write_text(
        "query_id,text,group_id,split,category\n"
        "q1,Повар,cook,train,role\n"
        "q2,Работа поваром,cook,train,role\n",
        encoding="utf-8",
    )
    labels.write_text(
        "query_id,document_id,relevance,reason\n"
        "q1,doc-1,2,Подходит\n"
        "q2,doc-2,1,Не все требования подтверждены\n",
        encoding="utf-8",
    )

    prepare_labels(corpus, queries, labels, output)

    assert pq.read_table(output / "queries.parquet").num_rows == 2
    assert pq.read_table(output / "qrels.parquet")["relevance"].to_pylist() == [2, 1]

    queries.write_text(
        "query_id,text,group_id,split,category\n"
        "q1,Повар,cook,train,role\n"
        "q2,Пекарь,baker,test,role\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Связанные документы"):
        prepare_labels(corpus, queries, labels, output)


def test_unjudged_results_are_rejected():
    """Не позволяет считать неизвестный результат нерелевантным документом"""
    labels = {"query": {"good": 2, "bad": 0}}
    require_judged_results({"query": ["good", "bad"]}, labels, limit=2)

    with pytest.raises(ValueError, match="Дополните разметку"):
        require_judged_results({"query": ["good", "unknown"]}, labels, limit=2)
