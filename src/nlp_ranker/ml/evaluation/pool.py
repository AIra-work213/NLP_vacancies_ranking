"""Построение пула документов для разметки релевантности"""


def build_candidate_pool(
    documents: list[dict[str, str]], queries: dict[str, str], limit: int = 10
) -> dict[str, dict[str, list[str]]]:
    """Возвращает кандидатов BM25 по тексту и TF-IDF по символам названия"""
    import bm25s
    from sklearn.feature_extraction.text import TfidfVectorizer

    if limit < 1 or len(documents) < limit:
        raise ValueError("Недостаточно документов для выбранной глубины пула")

    texts = [(doc["title"] + " ") * 3 + doc["text"] for doc in documents]
    tokens = bm25s.tokenize(texts, stopwords=[], show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    query_tokens = bm25s.tokenize(
        list(queries.values()), stopwords=[], show_progress=False
    )
    indexes, _ = retriever.retrieve(query_tokens, k=limit, show_progress=False)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    titles = vectorizer.fit_transform([doc["title"] for doc in documents])
    similarities = (vectorizer.transform(list(queries.values())) @ titles.T).toarray()
    bm25_results = {}
    title_results = {}

    for position, query_id in enumerate(queries):
        bm25_results[query_id] = [
            documents[int(index)]["document_id"] for index in indexes[position]
        ]
        ordered = sorted(
            range(len(documents)),
            key=lambda index: (-float(similarities[position, index]), index),
        )
        title_results[query_id] = [
            documents[index]["document_id"] for index in ordered[:limit]
        ]

    return {"bm25": bm25_results, "title": title_results}
