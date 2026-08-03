from collections import defaultdict

def rank_evidence(evidences):
    final_evidence = {}
    for category, evidence_list in evidences.items():
        if not evidence_list:
            final_evidence[category] = None
            continue

        final_evidence[category] = select_best(evidence_list)

    return final_evidence

def select_best(evidence_list):
    scores = defaultdict(int)

    for evidence in evidence_list:
        scores[evidence["value"]] += evidence["weight"]

    winner = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = (scores[winner] / total_score) * 100

    # reduce confidence if only one weak evidence exists
    if len(evidence_list) == 1:
        confidence *= 0.7

    return {
        "value": winner,
        "score": scores[winner],
        "confidence": round(confidence, 2)
    }
