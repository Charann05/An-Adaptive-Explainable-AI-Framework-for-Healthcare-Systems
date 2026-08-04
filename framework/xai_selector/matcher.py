from .knowledge import XAI_TECHNIQUES


def find_candidates(category, input_type):

    candidates = []

    for name, info in XAI_TECHNIQUES.items():

        model_ok = (
            category in info["models"]
            or "Any" in info["models"]
        )

        input_ok = input_type in info["inputs"]

        if model_ok and input_ok:

            candidates.append(
    (
        name,
        {
            "description": info["description"]
        }
    )
)

    return candidates