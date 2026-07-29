import ast
from pathlib import Path
from load_model_files import MODEL_FOLDER
from analyzer import evidences
import constants

files = list(Path(MODEL_FOLDER).rglob("*.py"))


def scan_file(file, evidences):

    try:
        with open(file, 'r', encoding="UTF-8") as content:
            tree = ast.parse(content.read(), filename=file)

    except Exception:
        return

    for node in ast.walk(tree):
        # print(type(node))
        if isinstance(node, ast.Import):
            for module in node.names:
                if module.name in constants.FRAMEWORK_MAP:
                    evidences["framework"].append({
                        "value" : constants.FRAMEWORK_MAP[module.name],
                        "source" : "python",
                        "weight" : constants.SOURCE_WEIGHTS["python"]
                    })

    # for node in ast.walk(tree):
        elif isinstance(node, ast.ImportFrom):

            # Detect framework from the module being imported from
            if node.module in constants.FRAMEWORK_MAP:
                evidences["framework"].append({
                    "value": constants.FRAMEWORK_MAP[node.module],
                    "source": "python",
                    "weight": constants.SOURCE_WEIGHTS["python"]
                })

            # Detect framework/architecture from imported names
            for module in node.names:
                if module.name in constants.FRAMEWORK_MAP:
                    evidences["framework"].append({
                        "value": constants.FRAMEWORK_MAP[module.name],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

                if module.name in constants.ARCHITECTURE_MAP:
                    evidences["architecture"].append({
                        "value": constants.ARCHITECTURE_MAP[module.name],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

    # print("Functions Calls")
    # for node in ast.walk(tree):
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in constants.ARCHITECTURE_MAP:
                    evidences["architecture"].append({
                        "value" : constants.ARCHITECTURE_MAP[node.func.id],
                        "source" : "python",
                        "weight" : constants.SOURCE_WEIGHTS["python"]                       
                    })

    # print("Attribute Function Call")
    # for node in ast.walk(tree):
        # elif isinstance(node, ast.Call):
            elif isinstance(node.func, ast.Attribute):
                # print(node.func.attr)
                if node.func.attr in constants.ARCHITECTURE_MAP:
                    evidences["architecture"].append({
                        "value" : constants.ARCHITECTURE_MAP[node.func.attr],
                        "source" : "python",
                        "weight" : constants.SOURCE_WEIGHTS["python"]                       
                    })               

for file in files:
    scan_file(file, evidences)

