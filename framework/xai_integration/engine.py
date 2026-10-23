from . import adapters
from .adapters import resolve_adapter
from . import plugins

from .dependency_manager import prepare_dependencies
from .exceptions import (
    ExplanationExecutionError,
    PluginNotFoundError,
)
from .output_schema import ExplanationResult
from .plugin_registry import get_plugin


class XAIIntegrationEngine:

    def __init__(self, auto_install_dependencies: bool = True):
        self.auto_install_dependencies = auto_install_dependencies

    def explain(self, model, input_data, model_info: dict, selector_result: dict) -> ExplanationResult:
        model_info = self._enrich_model_info(model, model_info)
        techniques = self._build_technique_order(selector_result)
        failures = []

        try:
            adapter = resolve_adapter(model, model_info)
        except ValueError as exc:
            raise ExplanationExecutionError(str(exc)) from exc

        for technique in techniques:
            try:
                plugin = get_plugin(technique)
            except PluginNotFoundError as exc:
                failures.append({"technique": technique, "stage": "plugin_resolution", "error": str(exc)})
                continue

            if not plugin.is_compatible(model_info):
                failures.append({
                    "technique": technique, "stage": "compatibility",
                    "error": f"{technique} is not compatible with model_info={model_info}",
                })
                continue

            dependency_result = prepare_dependencies(technique, auto_install=self.auto_install_dependencies)
            if not dependency_result.available:
                failures.append({"technique": technique, "stage": "dependency", "error": dependency_result.error})
                continue

            try:
                result = plugin.explain(
                    model=model,
                    input_data=input_data,
                    model_info=model_info,
                    adapter=adapter,
                )
                result.metadata.setdefault("attempted_techniques", techniques)
                result.metadata.setdefault("fallback_count", len(failures))
                # Capture the exact reasons why prior choices failed
                result.metadata.setdefault("fallback_details", failures) 
                return result
            except Exception as exc:
                failures.append({"technique": technique, "stage": "execution", "error": str(exc)})

        raise ExplanationExecutionError(self._build_failure_message(failures))

    @staticmethod
    def _enrich_model_info(model, model_info: dict) -> dict:
        """Add capability metadata that static project analysis cannot know."""
        enriched = dict(model_info)
        architecture = str(enriched.get("architecture", "")).lower()
        category = str(enriched.get("category", "")).lower()

        if "randomforest" in architecture or hasattr(model, "estimators_"):
            enriched["category"] = "RandomForest"
        elif "decisiontree" in architecture or hasattr(model, "tree_"):
            enriched["category"] = "DecisionTree"
        elif "xgb" in architecture or "xgboost" in architecture:
            enriched["category"] = "XGBoost"
        elif category in {"classification", "classifier"} and hasattr(
            model, "feature_importances_"
        ):
            enriched["category"] = "Tree"

        return enriched

    @staticmethod
    def _build_technique_order(selector_result: dict) -> list[str]:
        selected = selector_result.get("selected")
        if not selected:
            raise ValueError("Selector result does not contain 'selected'.")

        alternatives = selector_result.get("alternatives", [])
        techniques = [selected]
        for technique in alternatives:
            if technique and technique not in techniques:
                techniques.append(technique)
        return techniques

    @staticmethod
    def _build_failure_message(failures: list[dict]) -> str:
        if not failures:
            return "No XAI technique could be executed."
        details = "\n".join(f"- {item['technique']} [{item['stage']}]: {item['error']}" for item in failures)
        return f"All candidate XAI techniques failed:\n{details}"


_engine = XAIIntegrationEngine()


def run_explanation(model, input_data, model_info: dict, selector_result: dict) -> ExplanationResult:
    return _engine.explain(
        model=model,
        input_data=input_data,
        model_info=model_info,
        selector_result=selector_result,
    )