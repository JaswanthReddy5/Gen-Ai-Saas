def generateanswer(query, validateddocs):
    context = ""

    for doc, score, source in validateddocs:
        context += doc.page_content + "\n\n"

    answer = f"""
Answer based strictly on documentation:

{context}
"""
    return answer.strip()
