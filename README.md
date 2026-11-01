# Adaptive XAI Framework for Healthcare Systems

An adaptive explainable-AI framework for helping developers analyze existing
machine learning projects and select explainable-AI (XAI) techniques, especially
for healthcare and clinical modeling workflows.

The framework analyzes project files, infers a model profile, recommends a
suitable XAI technique, and prepares its dependencies. The repository also
contains XAI plugins and integration utilities for creating a standalone runner
that runs the user's project with explanations. After setup, use this runner
instead of repeating the full analysis and dependency-preparation workflow.

## Overview

Adaptive XAI currently supports this analysis and selection workflow:

1. Analyze a project directory to infer the model type, architecture, input
   modality, and available feature metadata.
2. Select a suitable XAI technique from the inferred model profile.
3. Prepare any required dependencies for the chosen technique.
4. Report the selected technique, alternatives, and rationale.

The analyzer writes a `model_report.json` profile that can be used by the
selector without repeating the analysis. Integration and explanation components
are available in the repository, but automatic instrumentation of the target
project is not yet part of the end-to-end CLI workflow.

## Package & installation

The Python package is published on [PyPI](https://pypi.org/project/adaptive-xai/)
as:

```bash
adaptive-xai
```

The source code is published in the
[GitHub repository](https://github.com/Charann05/An-Adaptive-Explainable-AI-Framework-for-Healthcare-Systems).

### Installation prerequisites

- Python 3.14 or newer. This is the minimum version declared by the package.
- `pip`, to install the package from PyPI.
- Access to the target project directory and its source/configuration files when
  running the analyzer.
- The target project's own ML framework and model dependencies if you intend to
  load or run that model. These are separate from Adaptive XAI's core
  dependencies.

The package's core dependencies (`pandas`, `numpy`, and `nbformat`) are installed
by pip. Dependencies for individual explanation techniques may also be needed;
the framework's dependency manager can prepare technique-specific packages.

Install the package in the Python environment that you intend to use:

```bash
pip install adaptive-xai
```

This installs the package and registers the command-line tools defined in
`pyproject.toml` in the target Python environment.

## CLI commands

### adaptive-xai

Run the analyzer, technique selector, and dependency-preparation workflow from
the target project root:

```bash
adaptive-xai
```

This starts the current framework orchestration flow. It analyzes the project,
selects a technique, and prepares dependencies. It does not automatically
instrument the target project's prediction code.

### analyze

Run the analyzer directly inside a target project directory:

```bash
analyze
```

This scans the project and generates or updates `model_report.json` in the
current working directory.

### xai-select

Use an existing analysis report without rerunning the full analyzer:

```bash
xai-select
```

By default, it reads `model_report.json` from the current directory. It can also
accept a path to a report file.

### convert

Convert one or more Jupyter notebooks into runnable Python scripts:

```bash
convert my_notebook.ipynb
convert notebooks/
```

The generated `.py` file keeps the same base name as the notebook and preserves
cell contents in plain Python while commenting out notebook magics and shell
commands that are not valid in a normal script.

## Framework architecture

The repository is organized as follows:

```text
adaptive_xai/
├── __init__.py
├── run.py                     # orchestration entry point
├── cli_display.py             # terminal output and UI helpers
├── model_analyzer/           # project/model profiling
├── xai_selector/             # technique selection logic
├── dependency_manager/       # dependency preparation and validation
├── xai_integration/          # model loading, plugin integration, execution
├── explanation_generator/    # template-based explanation rendering
├── notebook_converter.py     # notebook-to-Python conversion tool
└── ...
```

### Module Responsibilities

| Module | Description |
|---|---|
| **Model Analyzer** | Infers project and model characteristics, including ML framework, architecture, task type, input modality, feature metadata, and dataset/configuration signals. Generates `model_report.json`. |
| **XAI Selector** | Selects a compatible XAI method using the model profile and other signals. Includes mapping rules, strategy logic, recommendation/ranking support, and CLI tooling. |
| **Dependency Manager** | Prepares dependencies required by the selected explanation method and checks whether the required libraries are available. |
| **XAI Integration** | Handles model loading, explanation execution, plugin generation and integration, output schema definition, and persistent integration generation for project use. |
| **Explanation Generator** | Contains the template engine, generator entry point, and technique-specific JSON templates for explanation rendering. The generator is currently a placeholder. |

### Supported XAI Techniques

- SHAP
- LIME
- Feature Importance
- Grad-CAM
- Tree Interpreter

## Typical workflow

From a project directory:

```bash
analyze
xai-select
```

or:

```bash
adaptive-xai
```

The workflow produces or updates `model_report.json`, recommends an XAI
approach, and prepares dependencies. Running explanations from the target
project's normal prediction flow still requires integration that the current
CLI does not generate automatically.

## Output artifacts

The framework's current workflow produces or uses the following artifacts:

- `model_report.json` — inferred project/model profile
- selected technique, alternatives, and rationale reported by the selector
- XAI integration files and explanation output when the integration utilities
  are used separately
- technique-specific explanation templates included in the package

## Limitations

- **Project analysis is heuristic.** The analyzer infers project and model
  details from source code, notebooks, configuration, requirements, and dataset
  evidence. It may miss dynamically constructed models or misclassify projects
  when evidence is incomplete or ambiguous. Review `model_report.json` and the
  selector's recommendation before relying on them.
- **Technique selection is rule-based.** Recommendations come from the
  framework's current technique knowledge and mapping rules; they are not a
  guarantee that a technique is appropriate for every model, dataset, or
  prediction task.
- **Supported techniques and model types are limited.** The selector and plugin
  implementations cover SHAP, LIME, feature importance, Grad-CAM, and Tree
  Interpreter for selected model and input categories. Compatibility depends on
  the model loader, adapter, technique-specific dependencies, and available
  input/background data.
- **Healthcare use is not clinical validation.** Explanations describe aspects
  of model behavior; they are not medical advice, a diagnosis, evidence of
  causation, or a substitute for clinical validation and expert review.

## License

This project is licensed under the terms of the LICENSE file included in this
repository.
