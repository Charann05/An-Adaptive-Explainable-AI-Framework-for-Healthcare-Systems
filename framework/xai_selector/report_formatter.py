def format_result(model_info: dict, result: dict) -> str:
    lines = []
    lines.append("XAI TECHNIQUE SELECTION REPORT")
    lines.append("=" * 40)
    lines.append(f"Framework       : {model_info.get('framework')}")
    lines.append(f"Architecture    : {model_info.get('architecture')}")
    lines.append(f"Detected Category: {model_info.get('category')}")
    lines.append(f"Input Type      : {model_info.get('input_type')}")
    lines.append(f"Detection Confidence: {model_info.get('confidence')}%")
    if model_info.get("low_confidence"):
        lines.append("⚠  Low confidence detection — verify manually if possible")
    lines.append("-" * 40)
    lines.append(f"✅ Selected Technique: {result['selected']}")
    lines.append("Reason:")
    for r in result["reason"]:
        lines.append(f"   - {r}")
    lines.append(f"Alternative Techniques: {', '.join(result['alternatives'])}")
    lines.append("=" * 40)
    return "\n".join(lines)