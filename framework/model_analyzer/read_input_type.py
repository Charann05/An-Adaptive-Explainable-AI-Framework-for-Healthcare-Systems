import ast
from pathlib import Path
from . import constants
from .read_notebook import get_notebook_trees


def detect_input_type(MODEL_FOLDER, evidences, file_info):

    detected_types = set()

    for file in file_info:
        variables = {}
        
        if file["extension"] not in [".py", ".ipynb"]:
            continue
        file_path = Path(MODEL_FOLDER) / file["name"]

        try:
            if file_path.suffix == ".py":
                with open(file_path, "r", encoding="utf-8") as f:
                    trees = [ast.parse(f.read(), filename=str(file_path))]

            else:
                trees = get_notebook_trees(file_path)

        except Exception as e:
            print(f"Error while parsing {file_path}: {e}")
            continue

        # ---------------------------------
        # Pass 1 : Collect string variables
        # ---------------------------------

        for tree in trees:
            for node in ast.walk(tree):

                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):

                    for target in node.targets:

                        if isinstance(target, ast.Name):
                            variables[target.id] = node.value.value

        # ---------------------------------
        # Pass 2 : Detect from API calls
        # ---------------------------------

        for tree in trees:
            for node in ast.walk(tree):

                if not isinstance(node, ast.Call):
                    continue

                call_name = ""

                if isinstance(node.func, ast.Name):
                    call_name = node.func.id.lower()

                elif isinstance(node.func, ast.Attribute):

                    parts = []
                    current = node.func

                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value

                    if isinstance(current, ast.Name):
                        parts.append(current.id)

                    call_name = ".".join(reversed(parts)).lower()

                for key, value in constants.INPUT_PATTERNS.items():

                    if key in call_name and value not in detected_types:

                        evidences["input_type"].append({
                            "value": value,
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                        detected_types.add(value)
                        break

        # ---------------------------------
        # Pass 3 : Detect from file paths
        # ---------------------------------

        for value in variables.values():

            if not isinstance(value, str):
                continue

            value = value.lower()

            # Image
            if any(ext in value for ext in constants.IMAGE_EXTENSIONS):

                if "Image" not in detected_types:
                    evidences["input_type"].append({
                        "value": "Image",
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })
                    detected_types.add("Image")

            # Medical Images
            elif any(ext in value for ext in constants.MEDICAL_EXTENSIONS):

                if "3D Medical Image" not in detected_types:
                    evidences["input_type"].append({
                        "value": "3D Medical Image",
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })
                    detected_types.add("3D Medical Image")

            # Text
            elif any(ext in value for ext in constants.TEXT_EXTENSIONS):

                if "Text" not in detected_types:
                    evidences["input_type"].append({
                        "value": "Text",
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })
                    detected_types.add("Text")

            # CSV → only if no stronger input type has been found
            elif (
                any(ext in value for ext in constants.TABULAR_EXTENSIONS)
                and not detected_types.intersection({"Image", "Text", "3D Medical Image"})
            ):

                if "Tabular" not in detected_types:
                    evidences["input_type"].append({
                        "value": "Tabular",
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })
                    detected_types.add("Tabular")