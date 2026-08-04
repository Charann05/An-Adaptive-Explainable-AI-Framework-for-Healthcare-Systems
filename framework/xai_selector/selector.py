from .matcher import find_candidates
from .recommender import rank_techniques


def build_reason(model_info, selected, meta):

    return [

        f"Framework : {model_info['framework']}",

        f"Architecture : {model_info['architecture']}",

        f"Detected Category : {model_info['category']}",

        f"Input Type : {model_info['input_type']}",

        f"Selected Technique : {selected}",

        meta["description"]

    ]


def select_xai(model_info):

    candidates = find_candidates(

        model_info["category"],

        model_info["input_type"]

    )

    ranked = rank_techniques(

        candidates,

        model_info["category"]

    )

    selected_name, selected_meta = ranked[0]

    alternatives = [

        name for name, _ in ranked[1:]

    ]

    return {

        "selected": selected_name,

        "alternatives": alternatives,

        "reason": build_reason(

            model_info,

            selected_name,

            selected_meta

        )

    }