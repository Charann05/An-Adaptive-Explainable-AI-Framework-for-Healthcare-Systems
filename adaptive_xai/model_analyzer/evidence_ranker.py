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
        val = evidence["value"]

        key = tuple(val) if isinstance(val, list) else val
        scores[key] += evidence["weight"]

    winner_key = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = (scores[winner_key] / total_score) * 100 if total_score > 0 else 0

    if len(evidence_list) == 1:
        confidence *= 0.7

    winner = list(winner_key) if isinstance(winner_key, tuple) else winner_key

    return {
        "value": winner,
        "score": scores[winner_key],
        "confidence": round(confidence, 2),
    }