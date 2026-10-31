# Adaptive XAI Framework for Healthcare Systems

An adaptive explainable-AI framework for helping developers integrate XAI into
existing machine learning projects, especially healthcare and clinical modeling
workflows.

The framework analyzes a project, infers a model profile, selects a suitable XAI
technique, prepares dependencies, generates an integration, and produces
human-readable explanations for model predictions.

## Overview

Adaptive XAI is designed to support the following workflow:

1. Analyze a project directory to infer the model type, architecture, input
   modality, and available feature metadata.
2. Select a suitable XAI technique from the inferred model profile.
3. Prepare any required dependencies for the chosen technique.
4. Generate a model integration that can run explanations against the project.
5. Produce readable explanation output in a format suitable for downstream use,
   reporting, and debugging.

The framework is intended to work with existing ML projects without requiring the
user to rebuild a separate example model or re-run the analysis repeatedly.

## Package name and installation

The Python package is published as:

```bash
adaptive-xai
```

Install it from PyPI with:

```bash
pip install adaptive-xai
```

This installs the package and registers the command-line tools defined in
`pyproject.toml` in the target Python environment.

## CLI commands

### `adaptive-xai`

Run the full framework workflow from the project root:

```bash
adaptive-xai
```

This starts the framework orchestration flow, including model analysis,
selection, dependency preparation, and integration generation.

### `analyze`

Run the analyzer directly inside a target project directory:

```bash
analyze
```

This scans the project and generates or updates `model_report.json` in the
current working directory.

### `xai-select`

Use an existing analysis report without rerunning the full analyzer:

```bash
xai-select
```

By default, it reads `model_report.json` from the current directory. It can also
accept a path to a report file.

### `convert`

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

## Module responsibilities

### `adaptive_xai/model_analyzer/`

Responsible for inferring project and model characteristics including:

- ML framework
- model architecture or category
- task type
- input modality
- feature metadata
- dataset and config signals

It writes a `model_report.json` that summarizes the inferred profile.

### `adaptive_xai/xai_selector/`

Selects a compatible XAI method based on the model profile and other signals.
It includes:

- mapping rules and strategy logic
- recommendation/ranking support
- CLI tooling for continuing from a saved report

### `adaptive_xai/dependency_manager/`

Prepares dependencies required by the selected explanation method and checks
whether required libraries are available.

### `adaptive_xai/xai_integration/`

Handles the execution side of XAI integration, including:

- model loading
- explanation execution
- plugin generation and integration
- output schema definition
- persistent integration generation for project use

### `adaptive_xai/explanation_generator/`

Transforms structured XAI outputs into readable and presentation-friendly text.

This module contains:

- `template_engine.py` for JSON template loading and formatting
- `generator.py` for rendering explanation output
- `templates/*.json` for technique-specific output templates

Supported techniques include:

- SHAP
- LIME
- Feature Importance
- Grad-CAM
- Tree Interpreter

This is the module added to the framework for explanation formatting and result
presentation.

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

The workflow produces a `model_report.json`, identifies the best XAI approach,
prepares dependencies, and generates the integration needed to explain model
predictions.

## Output artifacts

The framework generates and relies on the following artifacts:

- `model_report.json` — inferred project/model profile
- generated XAI integration files
- selected technique metadata and explanation output
- template-based explanation reports

## Example usage

```bash
cd /path/to/your/project
analyze
xai-select
```

Or directly:

```bash
cd /path/to/your/project
adaptive-xai
```

## License

This project is licensed under the terms of the LICENSE file included in this
repository.
