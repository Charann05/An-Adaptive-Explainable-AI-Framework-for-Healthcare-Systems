from pathlib import Path
import re
import constants

def scan_readme(MODEL_FOLDER, evidences, file_info):
    print("Scanning README files...")

    for file in file_info:
        if file["extension"] not in [".md", ".txt", ".rst"]:
            continue

        if Path(file["name"]).stem.lower() != "readme":
            continue

        readme_path = Path(MODEL_FOLDER) / file["name"]
        try:

            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().lower()

        except Exception:
            continue

        for keyword, framework in constants.FRAMEWORK_MAP.items():
            if keyword.lower() in text:
                evidences["framework"].append({
                    "value": framework,
                    "source": "readme",
                    "weight": constants.SOURCE_WEIGHTS["readme"]
                })

        for keyword, architecture in constants.ARCHITECTURE_MAP.items():
            if keyword.lower() in text:
                evidences["architecture"].append({
                    "value": architecture,
                    "source": "readme",
                    "weight": constants.SOURCE_WEIGHTS["readme"]
                })

        for keyword, optimizer in constants.OPTIMIZER_MAP.items():
            if keyword.lower() in text:
                evidences["optimizer"].append({
                    "value": optimizer,
                    "source": "readme",
                    "weight": constants.SOURCE_WEIGHTS["readme"]
                })

        for keyword, loss in constants.LOSS_FUNCTION_MAP.items():
            if keyword.lower() in text:
                evidences["loss_function"].append({
                    "value": loss,
                    "source": "readme",
                    "weight": constants.SOURCE_WEIGHTS["readme"]
                })

        for keyword, input_type in constants.INPUT_README_MAP.items():
            if keyword in text:
                evidences["input_type"].append({
                    "value": input_type,
                    "source": "readme",
                    "weight": constants.SOURCE_WEIGHTS["readme"]
                })

        for pattern in constants.CLASS_PATTERNS:
            match = re.search(pattern, text)

            if match:
                evidences["num_classes"].append({
                    "value": int(match.group(1)),
                    "source": "readme",
                    "weight": constants.SOURCE_WEIGHTS["readme"]
                })
                break

        if "binary classification" in text:
            evidences["num_classes"].append({
                "value": 2,
                "source": "readme",
                "weight": constants.SOURCE_WEIGHTS["readme"]
            })