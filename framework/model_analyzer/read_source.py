import ast
from pathlib import Path
import constants

# files = list(Path(MODEL_FOLDER).rglob("*.py"))


def scan_file(MODEL_FOLDER, evidences, file_info):
    print("Scannig source files...")
    for file in file_info:
        if file["extension"] != ".py":
            continue

        file_path = Path(MODEL_FOLDER) / file["name"]
        print("Scanning file:", file_path)
        scan_single_file(file_path, evidences)

def scan_single_file(file,evidences):
    try:
        with open(file, 'r', encoding="UTF-8") as content:
            tree = ast.parse(content.read(), filename=file)

    except Exception as e:
        print("Error while parsing", file, ":", e)
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

# for file in files:
#     scan_file(file, evidences)

