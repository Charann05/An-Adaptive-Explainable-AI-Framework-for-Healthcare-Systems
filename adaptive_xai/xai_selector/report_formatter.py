def format_result(model_info: dict, result: dict) -> str:
    lines = []

    model_items = [
        (k.replace("_", " ").title(), str(v))
        for k, v in model_info.items()
        if v is not None
    ]

    all_keys = [k for k, _ in model_items] + ["Selected Technique", "Reason", "Alternative Techniques", "Parameter"]
    all_vals = [v for _, v in model_items] + [str(result.get("selected", "")), ", ".join(result.get("alternatives", []))]
    
    col1_w = max(len(k) for k in all_keys)
    col2_w = max(len(v) for v in all_vals)
    
    col1_w = max(col1_w, 20)
    col2_w = max(col2_w, 30)

    sep = f"+{'-' * (col1_w + 2)}+{'-' * (col2_w + 2)}+"

    lines.append("\nXAI TECHNIQUE SELECTION REPORT")
    lines.append(sep)
    lines.append(f"| {'Parameter':<{col1_w}} | {'Value':<{col2_w}} |")
    lines.append(sep)

    for label, val in model_items:
        lines.append(f"| {label:<{col1_w}} | {val:<{col2_w}} |")

    if model_items:
        lines.append(sep)

    selected = str(result.get("selected", ""))
    lines.append(f"| {'Selected Technique':<{col1_w}} | {selected:<{col2_w}} |")
    
    alternatives = ", ".join(result.get("alternatives", []))
    lines.append(f"| {'Alternatives':<{col1_w}} | {alternatives:<{col2_w}} |")
    lines.append(sep)

    reasons = result.get("reason", [])
    if reasons:
        lines.append(f"| {'Reason(s)':<{col1_w}} | {reasons[0]:<{col2_w}} |")
        for r in reasons[1:]:
            lines.append(f"| {'':<{col1_w}} | {r:<{col2_w}} |")
        lines.append(sep)

    return "\n".join(lines)