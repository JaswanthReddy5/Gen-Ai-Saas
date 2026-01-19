def hybridsearch(query, vectordb, bm25, k=3):
    vectorresults = vectordb.similarity_search_with_score(query, k=k)
    bm25results = bm25.search(query, k=k)

    combined = []

    for doc, score in vectorresults:
        combined.append((doc, score, "vector"))

    for doc, score in bm25results:
        combined.append((doc, score, "bm25"))

    return combined
