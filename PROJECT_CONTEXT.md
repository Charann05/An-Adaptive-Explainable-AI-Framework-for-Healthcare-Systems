# Project Context

## Product vision

Adaptive XAI is a developer tool/framework for adding explainable-AI
capabilities to an existing machine-learning project, including healthcare
projects.

The developer initially supplies the path to their project directory. The
tool analyzes that project, builds a profile of the model and its inputs,
selects a suitable XAI technique, and integrates that technique into the
project. After integration, the developer should be able to run their project
in its normal way and receive explanations alongside its normal model results.
They should not need to run Adaptive XAI again each time they want an
explanation.

The integration should live in the user's working directory and be tailored
to that project. It should connect to the model and runtime values used by
the project, rather than require a separate, unrelated example model or
user-supplied prediction input.

## Intended workflow

1. **Analyze the project directory.** Inspect project source, notebooks,
   configuration, requirements, and datasets to infer a model profile.
2. **Select an XAI technique.** Choose a technique based on the profile, with
   the current selector using a knowledge/rule-based mapping. Keep suitable
   alternatives available when practical.
3. **Prepare dependencies.** Check whether the selected technique's packages
   are available and install missing dependencies when needed.
4. **Integrate into the user's project.** Add a plugin file for the selected
   technique and automatically connect the user's project to it. The
   integration locates the model artifact when available and fills in
   project-specific values. If the project has a live model in memory but no
   saved model file, use that live model for the explanation and save a copy
   to a file.
5. **Run with the user's project.** On each prediction with new input, pass
   that same input to the plugin and generate an explanation. The user should
   see the explanation in the terminal when running the project normally;
   they should not need to rerun Adaptive XAI.
6. **Generate a human-readable explanation.** Convert the structured XAI
   output into readable text using templates. The explanation is an account
   of model behavior, not a medical diagnosis.

## Intended module responsibilities

- **Model analyzer (`framework/model_analyzer/`)**: infer a model profile from
  static project evidence. The profile includes framework, architecture or
  category, task, input modality, and available feature/schema information.
  Model-artifact discovery belongs to the integrator, not the analyzer.
- **XAI selector (`framework/xai_selector/`)**: recommend an XAI method and
  alternatives from the profile. The current approach is knowledge-based.
- **Dependency manager (`framework/dependency_manager/`)**: prepare the
  selected method's dependencies.
- **XAI integration (`framework/xai_integration/`)**: provide adapters,
  plugins, model loading, and explanation execution, and integrate the
  selected method with the user's project. The intended integration adds a
  plugin file and connects it to the project's prediction flow.
- **Explanation generator (`framework/explanation_generator/`)**: turn
  structured explanation results into user-facing text through templates.
- **CLI/orchestration (`framework/run.py`)**: coordinate the end-to-end
  workflow, beginning with the project directory and ending with a project
  that runs with XAI integrated.

## Current repository state (observed)

- `pyproject.toml` exposes `adaptive-xai` as `framework.run:main`.
- `framework/run.py` currently calls the analyzer and selector, displays the
  selection, and prepares dependencies. It does not call the XAI integration,
  locate/load a model artifact, or modify/generate integration code for the
  user's project.
- `framework/model_analyzer/` scans source and notebooks, requirements,
  configuration, and datasets; it reduces and ranks evidence and writes
  `model_report.json`.
- The analyzer does not discover a serialized model artifact. The separate
  `find_model_file()` helper is in
  `framework/xai_integration/persistent.py`.
- `framework/xai_selector/` contains technique knowledge and ranking logic.
  `select_xai()` returns a selected technique, alternatives, and reasons.
- `framework/xai_integration/` has an explanation engine exposed through
  `run_explanation()`, model adapters, plugins for GradCAM, SHAP, LIME,
  FeatureImportance, and TreeInterpreter, a model loader, and dependency
  handling.
- `PersistentIntegration` can generate a standalone `adap_xai.py` runner.
  That runner loads a model and accepts input through a CLI or prompt, then
  calls `run_explanation()`. It is not currently invoked by `framework/run.py`
  and is not connected to the user's existing project execution flow.
- `framework/explanation_generator/` has a template engine and templates,
  but `generator.py` and `rank_features.py` are empty and no current
  end-to-end flow consumes these templates.
- `framework/frontend_generator/` appears incomplete and disconnected from
  the CLI flow.
- `evaluation/` benchmarks analyzer fields; it does not validate the complete
  analyze-select-integrate-run-explain workflow.

## Confirmed integration requirements

- Initial supported project types should be Python scripts and notebooks.
- Integration should add a plugin file and automatically connect the user's
  project to it; the user's normal project run should trigger explanations.
- Generate an explanation after each prediction on new input, passing that
  prediction's input to the plugin.
- Print the resulting human-readable explanation in the terminal.
- The integrator, not the analyzer, is responsible for finding the saved model
  file. The analyzer should profile model and project characteristics, not
  search for model artifacts.
- When a live model is available in memory, use it to explain predictions. If
  no saved model file exists, save a copy of that live model to a file.
- The tool should be able to apply integration changes to the user's project
  automatically.

## Open design questions

These details were not yet specified and should be resolved before designing
the integrator:

- How to reliably identify the model object, prediction call, and input value
  across different scripts and notebooks.
- What should happen when the tool cannot safely locate the model or
  prediction point: stop with instructions, or ask the user to identify the
  relevant code.
- Where to save a model copy, which serialization format to use, and how to
  save framework-specific models safely.
- Whether integration should be reversible, backed up, safe to run more than
  once, and straightforward to update or remove.
- Whether the generated plugin requires Adaptive XAI to remain installed or
  should be self-contained in the user's project.
