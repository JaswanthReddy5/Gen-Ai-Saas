from langchain.schema import Document
import json

def load_docs():
    with open("data/faqs.json") as f:
        faqs = json.load(f)
    docs = []
    for item in faqs:
        text = f"Q: {item['question']}\nA: {item['answer']}"
        docs.append(Document(page_content=text, metadata={"id": item["id"]}))
    return docs
