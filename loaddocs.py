from langchain_core.documents import Document
import json

def loaddocs():
    with open("data/faqs.json", "r") as f:
        faqs = json.load(f)

    docs = []
    for item in faqs:
        text = f"Q: {item['question']}\nA: {item['answer']}"
        docs.append(Document(page_content=text, metadata={"id": item["id"]}))
    return docs
