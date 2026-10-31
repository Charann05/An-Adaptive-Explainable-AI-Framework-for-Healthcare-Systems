def print_banner():
    print(r"""
    ────────────────────────────────────────────────────────────────────────────────────────────────────
        ▄▖ ▌    ▗ ▘      ▖▖▄▖▄▖  ▄▖                ▌   ▐▘      ▖▖    ▜ ▗ ▌           ▄▖    ▗        
        ▌▌▛▌▀▌▛▌▜▘▌▌▌█▌  ▚▘▌▌▐   ▙▖▛▘▀▌▛▛▌█▌▌▌▌▛▌▛▘▙▘  ▜▘▛▌▛▘  ▙▌█▌▀▌▐ ▜▘▛▌▛▘▀▌▛▘█▌  ▚ ▌▌▛▘▜▘█▌▛▛▌▛▘
        ▛▌▙▌█▌▙▌▐▖▌▚▘▙▖  ▌▌▛▌▟▖  ▌ ▌ █▌▌▌▌▙▖▚▚▘▙▌▌ ▛▖  ▐ ▙▌▌   ▌▌▙▖█▌▐▖▐▖▌▌▙▖█▌▌ ▙▖  ▄▌▙▌▄▌▐▖▙▖▌▌▌▄▌
            ▌                                                                          ▄▌    
    ────────────────────────────────────────────────────────────────────────────────────────────────────       
    """)


def print_status(message):
    print(f"[✓] {message}")

def print_error(message):
    print(f"[✗] {message}")

def print_analysis_validation_error(reasons):
    print()
    print(
        "\033[1;37;41m The selected folder may not be the correct "
        "model project. \033[0m"
    )
    print("\033[1;31mSetup stopped because the analyzer could not identify:\033[0m")
    for reason in reasons:
        print(f"  - {reason}")
    print(
        "Check that you selected the project folder containing the model "
        "source/configuration files, then rerun the setup."
    )


def print_model_compatibility_error(message):
    print()
    print("\033[1;37;41m MODEL VERSION INCOMPATIBILITY \033[0m")
    print(f"\033[1;31m{message}\033[0m")
    print(
        "Retrain and save the model with the installed scikit-learn version, "
        "or use a separate environment with a compatible Python and "
        "scikit-learn version. The active environment was not downgraded."
    )


def print_model_load_error(error):
    from .xai_integration.exceptions import ModelCompatibilityError

    root_cause = error
    seen = set()
    while root_cause.__cause__ is not None and id(root_cause) not in seen:
        seen.add(id(root_cause))
        root_cause = root_cause.__cause__

    detail = str(root_cause)
    normalized_detail = detail.lower()
    if (
        isinstance(error, ModelCompatibilityError)
        and error.package == "scikit-learn"
        and error.saved_versions
    ):
        saved_versions = ", ".join(error.saved_versions)
        suggested_version = error.saved_versions[0]
        title = "SCIKIT-LEARN VERSION MISMATCH"
        message = (
            f"The model metadata says it was saved with scikit-learn "
            f"{saved_versions}; this environment has scikit-learn "
            f"{error.installed_version}. To try using it, create a separate "
            "environment with a Python version supported by the saved "
            f"scikit-learn release and install "
            f"`scikit-learn=={suggested_version}`. A compatible NumPy version "
            "may also be required. This is a best starting point, not a "
            "guarantee; retraining and re-saving with the current environment "
            "is the most reliable option."
        )
    elif isinstance(error, ModelCompatibilityError) or (
        "node array from the pickle has an incompatible dtype"
        in normalized_detail
    ):
        title = "MODEL VERSION INCOMPATIBILITY"
        message = (
            "The model was saved with a scikit-learn tree format that is "
            "incompatible with the installed version. Retrain/re-save it "
            "with the installed version or load it in a separate environment "
            "using a compatible Python and scikit-learn version."
        )
    elif (
        "sklearn.compose._column_transformer" in normalized_detail
        and "_remaindercolslist" in normalized_detail
    ):
        title = "SCIKIT-LEARN PIPELINE VERSION MISMATCH"
        message = (
            "This pipeline uses an internal scikit-learn component that "
            "differs from the version installed here. This is a model/library "
            "compatibility issue, not a problem with adap_xai.py, and the "
            "framework has not modified your model file. To continue, "
            "preferably retrain and re-save the pipeline with the installed "
            "scikit-learn version. If you cannot retrain, use a separate "
            "environment with a compatible Python and the scikit-learn "
            "version used to save the model; avoid downgrading a shared "
            "environment."
        )
    elif (
        "xgboost" in normalized_detail
        and (
            "estimator" in normalized_detail
            or "xgboost.sklearn" in normalized_detail
        )
    ):
        title = "XGBOOST MODEL INTERFACE UNAVAILABLE"
        message = (
            "XGBoost is required, but its scikit-learn estimator interface "
            "is unavailable in the active Python environment. Verify the "
            "installed XGBoost version and import path, then use an "
            "environment that provides the interface/version used to save "
            "this pickle. If possible, re-export the model in XGBoost's "
            "native JSON or UBJ format."
        )
    elif isinstance(root_cause, ModuleNotFoundError):
        title = "MODEL DEPENDENCY UNAVAILABLE"
        module_name = root_cause.name or "a required module"
        message = (
            f"The model requires Python module '{module_name}', which could "
            "not be imported. Install a compatible dependency in the active "
            "Python environment or use the environment that created the model."
        )
    elif isinstance(root_cause, FileNotFoundError):
        title = "MODEL FILE NOT FOUND"
        message = "Check that the configured model path exists and is accessible."
    else:
        title = "MODEL LOAD FAILED"
        message = (
            "The model could not be loaded. Check that the selected file is "
            "valid and compatible with the installed framework and dependency "
            "versions."
        )

    print()
    print(f"\033[1;37;41m {title} \033[0m")
    print(f"\033[1;31m{message}\033[0m")
    if detail:
        print(f"Details: {detail}")

def print_explanation_error(error):
    detail = str(error)
    if "background_data" in detail.lower():
        title = "EXPLANATION BACKGROUND DATA REQUIRED"
        message = (
            "The selected tabular SHAP or LIME explanation needs representative "
            "background rows. Add a matching CSV/Excel dataset to the project, "
            "or edit BACKGROUND_DATA in adap_xai.py with rows in the same "
            "feature order the model expects. The framework will not invent "
            "or modify model inputs."
        )
    else:
        title = "EXPLANATION FAILED"
        message = (
            "The model loaded, but the selected XAI technique could not produce "
            "an explanation. Check the technique's input requirements and the "
            "details below."
        )

    print()
    print(f"\033[1;37;41m {title} \033[0m")
    print(f"\033[1;31m{message}\033[0m")
    if detail:
        print(f"Details: {detail}")


def print_section(title):
    print("\n" + "─" * 62)
    print(f"{title:^62}")
    print("─" * 62)


def print_step(message):
    print(f"  → {message}")
