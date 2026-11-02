from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .template_engine import TemplateEngine


_TEMPLATE_ENGINE = TemplateEngine(Path(__file__).with_name("templates"))
_TEMPLATE_BY_TECHNIQUE = {
    "SHAP": "shap.json",
    "LIME": "lime.json",
    "FEATUREIMPORTANCE": "feature_importance.json",
    "GRADCAM": "gradcam.json",
    "TREEINTERPRETER": "tree_interpreter.json",
}


def _relative_percentages(scores: list[float]) -> list[float]:
    magnitudes = [abs(score) for score in scores]
    largest = max(magnitudes, default=0.0)
    if largest == 0.0:
        return [0.0] * len(scores)

    scaled = [magnitude / largest for magnitude in magnitudes]
    total = sum(scaled)
    percentages = [round(100.0 * magnitude / total, 1) for magnitude in scaled]
    percentages[max(range(len(scores)), key=magnitudes.__getitem__)] += round(
        100.0 - sum(percentages),
        1,
    )
    return percentages


def _format_result_table(rows: list[tuple[str, str]], title: str) -> str:
    parameter_width = max(
        [len("Parameter"), *(len(label) for label, _ in rows)]
    )
    value_width = max([len("Value"), *(len(value) for _, value in rows)])
    separator = (
        f"+{'-' * (parameter_width + 2)}"
        f"+{'-' * (value_width + 2)}+"
    )
    lines = [
        "",
        title,
        separator,
        f"| {'Parameter':<{parameter_width}} | {'Value':<{value_width}} |",
        separator,
    ]
    for label, value in rows:
        lines.append(
            f"| {label:<{parameter_width}} | {value:<{value_width}} |"
        )
    lines.append(separator)
    return "\n".join(lines)


def format_xai_output(
    result: dict[str, Any],
    input_type: str,
    visualization_path: str | None = None,
) -> str:
    """Render the technique and its structured outputs without narrative text."""
    common = _TEMPLATE_ENGINE.load_template("common.json")
    technique = str(result.get("technique_used") or "Unknown")
    rows = [
        ("Technique", technique),
        (
            "Prediction",
            _TEMPLATE_ENGINE.render(
                common["prediction_line"],
                prediction=result.get("prediction"),
            ).removeprefix("Prediction: "),
        ),
    ]

    confidence = result.get("confidence_score")
    if confidence is not None:
        rows.append(
            (
                "Model score",
                _TEMPLATE_ENGINE.render(
                    common["confidence_line"],
                    confidence=confidence,
                ).removeprefix("Model score: "),
            )
        )

    if str(input_type or "").strip().lower() == "image":
        if visualization_path and technique.strip().upper() == "GRADCAM":
            gradcam = _TEMPLATE_ENGINE.load_template("gradcam.json")
            rows.append(
                (
                    "Grad-CAM visualization",
                    _TEMPLATE_ENGINE.render(
                        gradcam["heatmap"],
                        visualization_path=visualization_path,
                    ).removeprefix("Grad-CAM visualization: "),
                )
            )
        elif visualization_path and technique.strip().upper() == "LIME":
            rows.append(("LIME visualization", visualization_path))
        elif visualization_path and technique.strip().upper() == "SHAP":
            rows.append(("SHAP visualization", visualization_path))
        elif result.get("raw_output") is not None:
            rows.append(("Attributions", common["image_attribution_available"]))
        else:
            rows.append(("Attributions", common["no_attributions"]))
        return _format_result_table(rows, common["xai_output_heading"])

    ranking = result.get("feature_ranking") or []
    if not ranking:
        rows.append(("Attributions", common["no_attributions"]))
        if visualization_path:
            rows.append(("Visualization", visualization_path))
        return _format_result_table(rows, common["xai_output_heading"])

    scores = [float(item["importance"]) for item in ranking]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Feature contribution scores must be finite numbers.")
    percentages = _relative_percentages(scores)
    technique_key = technique.strip().upper()
    template_filename = _TEMPLATE_BY_TECHNIQUE.get(
        technique_key,
        "common.json",
    )
    template = _TEMPLATE_ENGINE.load_template(template_filename)
    rows.append(
        (
            "Attributions",
            template.get("output_heading", common["attributions_heading"])
            .removesuffix(":"),
        )
    )
    is_global_importance = technique_key == "FEATUREIMPORTANCE"
    for item, score, percentage in zip(ranking, scores, percentages):
        factor = _TEMPLATE_ENGINE.render(
            template.get("output_factor", common["output_factor"]),
            feature=item["feature"],
            score=abs(score) if is_global_importance else score,
            sign="+" if score > 0 and not is_global_importance else "",
            percentage=percentage,
        ).strip().removeprefix("- ")
        rows.append(("Feature", factor))
    rows.append(("Note", common["relative_attribution_note"]))
    if visualization_path:
        rows.append(("Visualization", visualization_path))
    return _format_result_table(rows, common["xai_output_heading"])


def format_explanation(
    result: dict[str, Any],
    input_type: str,
    visualization_path: str | None = None,
) -> str:
    """Render a separate plain-language explanation of the XAI result."""
    common = _TEMPLATE_ENGINE.load_template("common.json")
    technique = str(result.get("technique_used") or "Unknown").strip().upper()
    template_filename = _TEMPLATE_BY_TECHNIQUE.get(
        technique,
        "common.json",
    )
    template = _TEMPLATE_ENGINE.load_template(template_filename)
    heading = common["human_explanation_heading"]
    lines = [
        heading,
        "-" * len(heading),
        _TEMPLATE_ENGINE.render(
            common["prediction_summary"],
            prediction=result.get("prediction"),
        ),
    ]

    confidence = result.get("confidence_score")
    if confidence is not None:
        lines.append(
            _TEMPLATE_ENGINE.render(
                common["confidence_explanation"],
                confidence=confidence,
            )
        )

    if str(input_type or "").strip().lower() == "image":
        lines.append("")
        if visualization_path and technique == "GRADCAM":
            lines.append(template["explanation"])
        elif result.get("raw_output") is not None:
            lines.append(
                template.get(
                    "mask_explanation",
                    common["generic_image_explanation"],
                )
            )
        else:
            lines.append(common["no_explanation"])
        lines.append("")
        lines.append(common["medical_disclaimer"])
        return "\n".join(lines)

    ranking = result.get("feature_ranking") or []
    if not ranking:
        lines.append("")
        lines.append(common["no_explanation"])
        lines.append("")
        lines.append(common["medical_disclaimer"])
        return "\n".join(lines)

    scores = [float(item["importance"]) for item in ranking]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Feature contribution scores must be finite numbers.")
    percentages = _relative_percentages(scores)

    if technique == "FEATUREIMPORTANCE":
        lines.append("")
        lines.append(
            _TEMPLATE_ENGINE.render(
                template["explanation"],
                features=", ".join(
                    str(item["feature"])
                    for item in ranking[:3]
                ),
            )
        )
        lines.append(template["scope"])
    elif max((abs(score) for score in scores), default=0.0) == 0.0:
        lines.append("")
        lines.append(common["no_explanation"])
    else:
        lines.append("")
        lines.append(template["lead"])
        ranked_factors = sorted(
            zip(ranking, scores, percentages),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        for index, (item, score, percentage) in enumerate(ranked_factors[:3]):
            if score > 0:
                sentence = template["supporting_factor"]
            elif score < 0:
                sentence = template["opposing_factor"]
            else:
                sentence = template["neutral_factor"]
            lines.append(
                _TEMPLATE_ENGINE.render(
                    sentence,
                    feature=item["feature"],
                    strength=(
                        "the biggest clue"
                        if index == 0
                        else "another important clue"
                        if percentage >= 20.0
                        else "a smaller clue"
                    ),
                    prediction=result.get("prediction"),
                )
            )
        lines.append(template["scope"])

    lines.append("")
    lines.append(common["medical_disclaimer"])
    return "\n".join(lines)
