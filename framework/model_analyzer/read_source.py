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
                        "value" : module.name,
                        "source" : "python",
                        "weight" : constants.SOURCE_WEIGHTS["python"]
                    })

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for module in node.names:
              if module.name in constants.FRAMEWORK_MAP:
                    evidences["framework"].append({
                        "value" : module.name,
                        "source" : "python",
                        "weight" : constants.SOURCE_WEIGHTS["python"]
                    })

    print("Functions Calls")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                print(node.func.id)

    print("Attribute Function Call")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                print(node.func.attr)

for file in files:
    scan_file(file, evidences)

