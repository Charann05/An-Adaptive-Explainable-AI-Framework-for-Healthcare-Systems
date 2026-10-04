from collections import Counter, defaultdict


def reduce_evidence(evidences):
    if not evidences:
        return {}

    reduced_evidences = {}
    for category, evidence_list in evidences.items():
        reduced_evidences[category] = reduce_category(evidence_list)

    return reduced_evidences


def reduce_category(evidence_list):
    if not evidence_list:
        return []

    grouped = defaultdict(list)
    for evidence in evidence_list:
        grouped[evidence.get("source")].append(evidence)

    reduced = []
    for source, items in grouped.items():
        if not items:
            continue

        if isinstance(items[0].get("value"), list):
            best_item = max(items, key=lambda item: item.get("weight", 0))
            winner = best_item["value"]
        else:
            weighted_scores = defaultdict(float)
            for item in items:
                weighted_scores[item["value"]] += item.get("weight", 1.0)

            counter = Counter(item["value"] for item in items)
            winner = max(
                weighted_scores.keys(),
                key=lambda val: (weighted_scores[val], counter[val], str(val)),
            )

        matching_items = [item for item in items if item["value"] == winner]
        best_matching_item = max(
            matching_items, key=lambda item: item.get("weight", 0)
        )

        reduced.append(
            {
                "value": best_matching_item["value"],
                "source": best_matching_item["source"],
                "weight": best_matching_item.get("weight", 1.0),
            }
        )

    return reduced