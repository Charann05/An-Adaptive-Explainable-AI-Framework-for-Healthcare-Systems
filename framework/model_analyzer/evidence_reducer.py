from collections import defaultdict, Counter


def reduce_evidence(evidences):

    reduced_evidences = {}

    for category, evidence_list in evidences.items():
        reduced_evidences[category] = reduce_category(evidence_list)

    return reduced_evidences


def reduce_category(evidence_list):

    if not evidence_list:
        return []

    # Group evidences by source
    grouped = defaultdict(list)

    for evidence in evidence_list:
        grouped[evidence["source"]].append(evidence)

    reduced = []

    # Process each source separately
    for source, items in grouped.items():

        # Count values from this source
        counter = Counter(
            item["value"] for item in items
        )

        # Select highest frequency value
        winner = counter.most_common(1)[0][0]

        # Keep one evidence object
        for item in items:
            if item["value"] == winner:
                reduced.append({
                    "value": item["value"],
                    "source": item["source"],
                    "weight": item["weight"]
                })
                break

    return reduced