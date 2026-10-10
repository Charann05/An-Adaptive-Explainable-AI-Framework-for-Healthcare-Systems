import ast
from pathlib import Path
from . import constants
from collections import Counter
from .read_notebook import get_notebook_trees
from .. import cli_display

CLASSIFIER_HEAD_NAMES = {
    "classifier", "classification_head", "fc", "head", "output", "logits",
    "classification_layer", "class_head",
}

def _target_name(target):

    if isinstance(target, ast.Name):
        return target.id.lower()
    
    if isinstance(target, ast.Attribute):
        return target.attr.lower()
    return None

def _build_call_target_map(tree):

    call_targets = {}
    for parent in ast.walk(tree):
        if isinstance(parent, ast.Assign) and isinstance(parent.value, ast.Call):

            names = {
                name for target in parent.targets
                if (name := _target_name(target)) is not None
            }
            call_targets[id(parent.value)] = names

        elif isinstance(parent, ast.AnnAssign) and isinstance(parent.value, ast.Call):
            name = _target_name(parent.target)

            if name:
                call_targets.setdefault(id(parent.value), set()).add(name)
    return call_targets

def _append_evidence(evidences, field, value, weight):

    evidence_list = evidences.setdefault(field, [])

    if not any(
        item.get("value") == value and item.get("source") == "python"
        for item in evidence_list
    ):
        evidence_list.append({
            "value": value,
            "source": "python",
            "weight": weight,
        })

def scan_file(MODEL_FOLDER, evidences, file_info):

    cli_display.print_status("Scanning source (.py/.ipynb) files")
    for file in file_info:
        if file["extension"] not in [".py", ".ipynb"]:
            continue

        file_path = Path(MODEL_FOLDER) / file["name"]
        cli_display.print_step(f"Scanning {file_path}")
        scan_single_file(file_path, evidences)

def scan_single_file(file, evidences):

    family_counter = Counter()
    try:
        if file.suffix == ".py":
            with open(file, "r", encoding="utf-8") as f:
                trees = [ast.parse(f.read(), filename=str(file))]

        elif file.suffix == ".ipynb":
            trees = get_notebook_trees(file)

        else:
            return
    except Exception as e:
        print(f"Error while parsing {file}: {e}")
        return

    # Pass 1 : Collect variables

    variables = {}
    # Step 1: Collect direct assignments first
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):

                if isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables[target.id] = node.value.value

            elif isinstance(node, ast.AnnAssign):
                if (
                    isinstance(node.target, ast.Name)
                    and isinstance(node.value, ast.Constant)
                ):
                    variables[node.target.id] = node.value.value

    # Step 2: Resolve function parameter defaults
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                
                continue
            args = node.args.args
            defaults = node.args.defaults
            offset = len(args) - len(defaults)

            for arg, default in zip(args[offset:], defaults):
                if isinstance(default, ast.Constant):
                    variables[arg.arg] = default.value

                elif isinstance(default, ast.Name):
                    value = variables.get(default.id)

                    if isinstance(value, int) and not isinstance(value, bool):
                        variables[arg.arg] = value

    # Pass 2 : Detect evidences

    class_parameter_names = {
        name.lower() for name in constants.CLASS_PARAMETER_NAMES
    }
    for tree in trees:
        call_target_map = _build_call_target_map(tree)

        for node in ast.walk(tree):
            # Detect assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    variable_name = target.id.lower()

                    # Detect classification target variables
                    if variable_name in constants.CLASSIFICATION_TARGET_NAMES:
                        evidences["task"].append({
                            "value": "Classification",
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"] - 2
                        })

                    # Detect feature names and number of inputs
                    if variable_name in {
                        "x",
                        "features",
                        "inputs",
                        "x_train",
                        "x_test",
                        "train_x",
                        "test_x"
                    }:
                        
                        feature_names = extract_feature_names(node.value)
                        if feature_names:
                            evidences["feature_names"].append({
                                "value": feature_names,
                                "source": "python",
                                "weight": constants.SOURCE_WEIGHTS["python"]
                            })

                            evidences["num_inputs"].append({
                                "value": len(feature_names),
                                "source": "python",
                                "weight": constants.SOURCE_WEIGHTS["python"]
                            })


            if isinstance(node, ast.Call):
                for keyword in node.keywords:

                    if keyword.arg and keyword.arg.lower() in class_parameter_names:
                        value = resolve_integer_value(keyword.value, variables, tree)
                        if value is not None and value > 0:
                            _append_evidence(
                                evidences, "num_classes", value,
                                constants.SOURCE_WEIGHTS["python"]
                            )

                            evidences["task"].append({
                                "value": "Classification",
                                "source": "python",
                                "weight": constants.SOURCE_WEIGHTS["python"] - 2,
                            })

                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr.lower() == "linear"
                    and len(node.args) >= 2
                ):
                    
                    output_size_expr = node.args[1]
                    output_size = resolve_integer_value(
                        output_size_expr, variables, tree
                    )

                    assigned_names = call_target_map.get(id(node), set())
                    named_as_class_count = (
                        isinstance(output_size_expr, ast.Name)
                        and output_size_expr.id.lower() in class_parameter_names
                    )

                    is_likely_head = bool(assigned_names & CLASSIFIER_HEAD_NAMES)
                    if (
                        output_size is not None
                        and output_size > 0
                        and (named_as_class_count or is_likely_head)
                    ):
                        _append_evidence(
                            evidences, "num_classes", output_size,
                            constants.SOURCE_WEIGHTS["python"]
                        )

                        evidences["task"].append({
                            "value": "Classification",
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"] - (1 if named_as_class_count else 3),
                        })

            if isinstance(node, ast.ClassDef):
                class_name = node.name.lower()
                if "cnn" in class_name:
                    family_counter["Custom CNN"] += 3

                elif (
                    "rnn" in class_name
                    or "lstm" in class_name
                    or "gru" in class_name
                ):
                    family_counter["Custom RNN"] += 3

                elif "transformer" in class_name:
                    family_counter["Custom Transformer"] += 3

                elif "gan" in class_name:
                    family_counter["GAN"] += 3

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

                    module = node.module.lower()
                    if "torch_geometric" in module:
                        family_counter["Graph Neural Network"] += 2
                    elif "transformers" in module:
                        family_counter["Custom Transformer"] += 2

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
                if isinstance(node.func, ast.Name):
                    name = node.func.id.lower()

                    if name in constants.CLASSIFIER_MODELS:
                        evidences["task"].append({
                            "value": "Classification",
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    elif name in constants.REGRESSOR_MODELS:
                        evidences["task"].append({
                            "value": "Regression",
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    if name in constants.GENERIC_ARCHITECTURE_MAP:
                        family = constants.GENERIC_ARCHITECTURE_MAP[name]
                        family_counter[family] += 1

                    if name in constants.ARCHITECTURE_MAP:
                        evidences["architecture"].append({
                            "value": constants.ARCHITECTURE_MAP[name],
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    # Optimizer
                    if name in constants.OPTIMIZER_MAP:
                        evidences["optimizer"].append({
                            "value": constants.OPTIMIZER_MAP[name],
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    # Loss Function
                    if name in constants.LOSS_FUNCTION_MAP:
                        evidences["loss_function"].append({
                            "value": constants.LOSS_FUNCTION_MAP[name],
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    if name in constants.CLASSIFICATION_LOSSES:
                        evidences["task"].append({
                            "value": "Classification",
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    elif name in constants.REGRESSION_LOSSES:
                        evidences["task"].append({
                            "value": "Regression",
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr.lower()
                    if name in constants.CLASSIFIER_MODELS:
                        evidences["task"].append({
                            "value": "Classification",
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    elif name in constants.REGRESSOR_MODELS:
                        evidences["task"].append({
                            "value": "Regression",
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    if name in constants.GENERIC_ARCHITECTURE_MAP:
                        family = constants.GENERIC_ARCHITECTURE_MAP[name]
                        family_counter[family] += 1

                    if name in constants.ARCHITECTURE_MAP:
                        evidences["architecture"].append({
                            "value": constants.ARCHITECTURE_MAP[name],
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    # Optimizer
                    if name in constants.OPTIMIZER_MAP:
                        evidences["optimizer"].append({
                            "value": constants.OPTIMIZER_MAP[name],
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    # Loss Function
                    if name in constants.LOSS_FUNCTION_MAP:
                        evidences["loss_function"].append({
                            "value": constants.LOSS_FUNCTION_MAP[name],
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    if name in constants.CLASSIFICATION_LOSSES:
                        evidences["task"].append({
                            "value": "Classification",
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                    elif name in constants.REGRESSION_LOSSES:
                        evidences["task"].append({
                            "value": "Regression",
                            "source": "python",
                            "weight": constants.SOURCE_WEIGHTS["python"]
                        })

                # Detect architecture from arguments
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

                for keyword in node.keywords:
                    # -------- Optimizer --------
                    if keyword.arg in constants.OPTIMIZER_PARAMETER_NAMES:
                        optimizer = None
       
                        if (
                            isinstance(keyword.value, ast.Constant)
                            and isinstance(keyword.value.value, str)
                        ):
                            optimizer = keyword.value.value

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
                        if (
                            isinstance(keyword.value, ast.Constant)
                            and isinstance(keyword.value.value, str)
                        ):
                            loss = keyword.value.value

                        elif isinstance(keyword.value, ast.Name):
                            value = variables.get(keyword.value.id)
                            if isinstance(value, str):
                                loss = value

                        elif isinstance(keyword.value, ast.Call):
                            if isinstance(keyword.value.func, ast.Attribute):
                                loss = keyword.value.func.attr.lower()
                            elif isinstance(keyword.value.func, ast.Name):
                                loss = keyword.value.func.id.lower()

                        if loss:
                            loss = loss.lower()
                            if loss in constants.LOSS_FUNCTION_MAP:
                                evidences["loss_function"].append({
                                    "value": constants.LOSS_FUNCTION_MAP[loss],
                                    "source": "python",
                                    "weight": constants.SOURCE_WEIGHTS["python"]
                                })

                            if loss in constants.CLASSIFICATION_LOSSES:
                                evidences["task"].append({
                                    "value": "Classification",
                                    "source": "python",
                                    "weight": constants.SOURCE_WEIGHTS["python"]
                                })

                            elif loss in constants.REGRESSION_LOSSES:
                                evidences["task"].append({
                                    "value": "Regression",
                                    "source": "python",
                                    "weight": constants.SOURCE_WEIGHTS["python"]
                                })

    if not evidences["architecture"] and family_counter:
        family = family_counter.most_common(1)[0][0]
        evidences["architecture"].append({
            "value": family,
            "source": "python",
            "weight": constants.SOURCE_WEIGHTS["python"]
        })

def resolve_integer_value(expression, variables, tree, depth=0):

    if depth > 10:
        return None
    if (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
    ):
        return expression.value
    
    if not isinstance(expression, ast.Name):
        return None
    
    name = expression.id
    value = variables.get(name)

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    for item in ast.walk(tree):
        if isinstance(item, ast.Assign):
            targets = item.targets

        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        else:
            continue

        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in targets
        ):
            value = resolve_integer_value(
                item.value, variables, tree, depth + 1
            )

            if value is not None:
                return value
            
    for item in ast.walk(tree):
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        args = item.args
        positional_args = args.posonlyargs + args.args
        defaults = args.defaults

        if not defaults:
            continue

        default_start = len(positional_args) - len(defaults)
        for index, parameter in enumerate(positional_args):
            if parameter.arg != name or index < default_start:
                continue

            default = defaults[index - default_start]
            value = resolve_integer_value(
                default, variables, tree, depth + 1
            )

            if value is not None:
                return value
    return None

def extract_feature_names(node):

    if not isinstance(node, ast.Subscript):
        return []
    
    slice_node = node.slice

    if isinstance(slice_node, ast.List):
        return extract_string_list(slice_node)
    
    if isinstance(slice_node, ast.Tuple):
        for element in slice_node.elts:
            if isinstance(element, ast.List):
                return extract_string_list(element)
    return []

def extract_string_list(node):

    feature_names = []
    for element in node.elts:
        if isinstance(element, ast.Constant):
            if isinstance(element.value, str):
                feature_names.append(element.value)
    return feature_names
