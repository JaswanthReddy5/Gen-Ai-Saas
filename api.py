from fastapi import FastAPI

from loaddocs import loaddocs
from vectorstore import createvectorstore
from bm25retriever import BM25Retriever
from hybridretrieval import hybridsearch
from validation import validateresults
from answergenerator import generateanswer

app = FastAPI()

docs = loaddocs()
vectordb = createvectorstore(docs)
bm25 = BM25Retriever(docs)

@app.get("/ask")
def ask(query: str):
    results = hybridsearch(query, vectordb, bm25)
    isvalid, response = validateresults(results)

    if not isvalid:
        return {"error": response}

    return {
        "query": query,
        "answer": generateanswer(query, response),
        "sources": [
            {"text": d.page_content, "score": s, "type": src}
            for d, s, src in response
        ]
    }
