def hybrid_search(query, vector_db, bm25, k=3):
    vector_results = vector_db.similarity_search_with_score(query, k=k)
    bm25_results = bm25.search(query, k=k)

    combined = []

    for doc, score in vector_results:
        combined.append((doc, score, "vector"))

    for doc, score in bm25_results:
        combined.append((doc, score, "bm25"))

    return combined
