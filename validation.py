MINRELEVANCESCORE = 0.3

def validateresults(results):
    validresults = []

    for doc, score, source in results:
        if score >= MINRELEVANCESCORE:
            validresults.append((doc, score, source))

    if len(validresults) == 0:
        return False, "Not enough relevant information. Please rephrase your question."

    return True, validresults
