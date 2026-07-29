import ast
from pathlib import Path
import constants


def scan_file(MODEL_FOLDER, evidences, file_info):
    print("Scanning source files...")

    for file in file_info:

        if file["extension"] != ".py":
            continue

        file_path = Path(MODEL_FOLDER) / file["name"]
        print("Scanning file:", file_path)

        scan_single_file(file_path, evidences)


def scan_single_file(file, evidences):

    try:
        with open(file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file))

    except Exception as e:
        print(f"Error while parsing {file}: {e}")
        return

    # --------------------------------------------------
    # Pass 1 : Collect variables
    # --------------------------------------------------

    variables = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Constant):

                for target in node.targets:

                    if isinstance(target, ast.Name):
                        variables[target.id] = node.value.value

        elif isinstance(node, ast.FunctionDef):
            args = node.args.args
            defaults = node.args.defaults
            offset = len(args) - len(defaults)

            for arg, default in zip(args[offset:], defaults):

                if isinstance(default, ast.Constant):
                    variables[arg.arg] = default.value

    print("Variables:", variables)

    # --------------------------------------------------
    # Pass 2 : Detect evidences
    # --------------------------------------------------

    for node in ast.walk(tree):

        # ---------------- IMPORT ----------------

        if isinstance(node, ast.Import):

            for module in node.names:
                framework = module.name.split(".")[0].lower()
                if framework in constants.FRAMEWORK_MAP:

                    evidences["framework"].append({
                        "value": constants.FRAMEWORK_MAP[framework],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

        # ------------- IMPORT FROM -------------

        elif isinstance(node, ast.ImportFrom):

            if node.module:
                framework = node.module.split(".")[0].lower()
                if framework in constants.FRAMEWORK_MAP:

                    evidences["framework"].append({
                        "value": constants.FRAMEWORK_MAP[framework],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

            for module in node.names:
                name = module.name.lower()
                if name in constants.FRAMEWORK_MAP:

                    evidences["framework"].append({
                        "value": constants.FRAMEWORK_MAP[name],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

                if name in constants.ARCHITECTURE_MAP:

                    evidences["architecture"].append({
                        "value": constants.ARCHITECTURE_MAP[name],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

        # ---------------- CALL ----------------

        elif isinstance(node, ast.Call):

            # ------------------------------------
            # Detect architecture by constructor
            # ------------------------------------

            if isinstance(node.func, ast.Name):

                name = node.func.id.lower()
                if name in constants.ARCHITECTURE_MAP:
                    evidences["architecture"].append({
                        "value": constants.ARCHITECTURE_MAP[name],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

                # <<< NEW >>> Optimizer
                if name in constants.OPTIMIZER_MAP:
                    evidences["optimizer"].append({
                        "value": constants.OPTIMIZER_MAP[name],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

                # <<< NEW >>> Loss Function
                if name in constants.LOSS_FUNCTION_MAP:
                    evidences["loss_function"].append({
                        "value": constants.LOSS_FUNCTION_MAP[name],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr.lower()
                if name in constants.ARCHITECTURE_MAP:

                    evidences["architecture"].append({
                        "value": constants.ARCHITECTURE_MAP[name],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

                # <<< NEW >>> Optimizer
                if name in constants.OPTIMIZER_MAP:

                    evidences["optimizer"].append({
                        "value": constants.OPTIMIZER_MAP[name],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

                # <<< NEW >>> Loss Function
                if name in constants.LOSS_FUNCTION_MAP:

                    evidences["loss_function"].append({
                        "value": constants.LOSS_FUNCTION_MAP[name],
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

            # ------------------------------------
            # Detect architecture from arguments
            # ------------------------------------

            for arg in node.args:
                model_name = None

                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                ):
                    model_name = arg.value

                elif isinstance(arg, ast.Name):
                    value = variables.get(arg.id)

                    if isinstance(value, str):
                        model_name = value

                if model_name:
                    model_name = model_name.lower()
                    for architecture in sorted(
                        constants.ARCHITECTURE_MAP.keys(),
                        key=len,
                        reverse=True
                    ):

                        if architecture in model_name:
                            evidences["architecture"].append({
                                "value": constants.ARCHITECTURE_MAP[architecture],
                                "source": "python",
                                "weight": constants.SOURCE_WEIGHTS["python"]
                            })

                            break

            # ------------------------------------
            # Detect num_classes
            # ------------------------------------

            for keyword in node.keywords:
                if keyword.arg not in constants.CLASS_PARAMETER_NAMES:
                    continue
                num_classes = None

                if (
                    isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, int)
                ):
                    num_classes = keyword.value.value

                elif isinstance(keyword.value, ast.Name):
                    value = variables.get(keyword.value.id)

                    if isinstance(value, int):
                        num_classes = value

                if num_classes is not None:
                    evidences["num_classes"].append({
                        "value": num_classes,
                        "source": "python",
                        "weight": constants.SOURCE_WEIGHTS["python"]
                    })

             # ------------------------------------
            # Detect optimizer and loss
            # ------------------------------------

            for keyword in node.keywords:

                # -------- Optimizer --------

                if keyword.arg in constants.OPTIMIZER_PARAMETER_NAMES:
                    optimizer = None
                    # optimizer="adam"
                    if (
                        isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        optimizer = keyword.value.value

                    # optimizer=optimizer_name
                    elif isinstance(keyword.value, ast.Name):
                        value = variables.get(keyword.value.id)
                        if isinstance(value, str):
                            optimizer = value

                    if optimizer:
                        optimizer = optimizer.lower()
                        if optimizer in constants.OPTIMIZER_MAP:
                            evidences["optimizer"].append({
                                "value": constants.OPTIMIZER_MAP[optimizer],
                                "source": "python",
                                "weight": constants.SOURCE_WEIGHTS["python"]
                            })

                # -------- Loss Function --------

                elif keyword.arg in constants.LOSS_PARAMETER_NAMES:
                    loss = None
                    # loss="cross_entropy"
                    if (
                        isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        loss = keyword.value.value

                    # loss=loss_name
                    elif isinstance(keyword.value, ast.Name):
                        value = variables.get(keyword.value.id)
                        if isinstance(value, str):
                            loss = value

                    if loss:
                        loss = loss.lower()
                        if loss in constants.LOSS_FUNCTION_MAP:
                            evidences["loss_function"].append({
                                "value": constants.LOSS_FUNCTION_MAP[loss],
                                "source": "python",
                                "weight": constants.SOURCE_WEIGHTS["python"]
                            })