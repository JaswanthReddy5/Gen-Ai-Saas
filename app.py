import streamlit as st

from loaddocs import loaddocs
from vectorstore import createvectorstore
from bm25retriever import BM25Retriever
from hybridretrieval import hybridsearch
from validation import validateresults
from answergenerator import generateanswer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Support Assistant (RAG)",
    layout="wide"
)

# ---------------- HEADER ----------------
st.title("Support Assistant (RAG)")
st.caption("A Retrieval-Augmented Generation system with hallucination control")

st.divider()

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_backend():
    docs = loaddocs()
    vectordb = createvectorstore(docs)
    bm25 = BM25Retriever(docs)
    return docs, vectordb, bm25

docs, vectordb, bm25 = load_backend()

# ---------------- USER INPUT ----------------
query = st.text_input(
    "Ask your question",
    placeholder="e.g. How do I reset my password?"
)

# ---------------- PROCESS QUERY ----------------
if query:
    with st.spinner("Searching documentation..."):
        results = hybridsearch(query, vectordb, bm25)
        isvalid, response = validateresults(results)

    st.divider()

    # ---------------- QUERY DISPLAY ----------------
    st.subheader("User Query")
    st.write(query)

    # ---------------- VALIDATION ----------------
    if not isvalid:
        st.error(response)

    else:
        # ---------------- ANSWER ----------------
        answer = generateanswer(query, response)

        st.subheader("Final Grounded Answer")
        st.success(answer)

        # ---------------- SOURCES ----------------
        st.subheader("Retrieved Evidence")

        for i, (doc, score, source) in enumerate(response, start=1):
            with st.expander(f"Source {i} | {source.upper()} | Score: {round(score, 3)}"):
                st.write(doc.page_content)

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "This system answers strictly from retrieved documentation. "
    "Low-relevance queries are rejected to prevent hallucinations."
)
