from .knowledge import CATEGORY_PRIORITY


def rank_techniques(candidates, category):

    candidate_dict = dict(candidates)

    ordered = []

    if category in CATEGORY_PRIORITY:

        for technique in CATEGORY_PRIORITY[category]:

            if technique in candidate_dict:

                ordered.append(
                    (technique, candidate_dict[technique])
                )

    for technique, meta in candidates:

        if technique not in [x[0] for x in ordered]:

            ordered.append(
                (technique, meta)
            )

    return ordered