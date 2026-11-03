from .. import cli_display


def format_result(model_info: dict, result: dict) -> str:
    model_items = [
        (k.replace("_", " ").title(), str(v))
        for k, v in model_info.items()
        if v is not None
    ]

    rows = list(model_items)
    section_breaks = []
    if model_items:
        section_breaks.append(len(rows))

    rows.extend([
        ("Selected Technique", str(result.get("selected", ""))),
        ("Alternatives", ", ".join(result.get("alternatives", []))),
    ])
    section_breaks.append(len(rows))
    reasons = result.get("reason", [])
    if reasons:
        rows.extend(
            ("Reason(s)" if index == 0 else "", str(reason))
            for index, reason in enumerate(reasons)
        )
        section_breaks.append(len(rows))

    return (
        "\nXAI TECHNIQUE SELECTION REPORT\n"
        + cli_display.format_table(rows, section_breaks=section_breaks)
    )