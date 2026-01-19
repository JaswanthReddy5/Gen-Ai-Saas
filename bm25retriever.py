from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, docs):
        self.docs = docs
        corpus = [doc.page_content.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query, k=3):
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:k]
