import streamlit as st

from loaddocs import loaddocs
from vectorstore import createvectorstore
from bm25retriever import BM25Retriever
from hybridretrieval import hybridsearch
from validation import validateresults
from answergenerator import generateanswer

st.title("📘 Support Assistant (RAG)")

docs = loaddocs()
vectordb = createvectorstore(docs)
bm25 = BM25Retriever(docs)

query = st.text_input("Ask your question")

if query:
    results = hybridsearch(query, vectordb, bm25)

    isvalid, response = validateresults(results)

    st.subheader("Query")
    st.write(query)

    if not isvalid:
        st.error(response)
    else:
        answer = generateanswer(query, response)

        st.subheader("Final Grounded Answer")
        st.write(answer)

        st.subheader("Retrieved Chunks")
        for doc, score, source in response:
            st.write(f"Source: {source} | Score: {score}")
            st.write(doc.page_content)
