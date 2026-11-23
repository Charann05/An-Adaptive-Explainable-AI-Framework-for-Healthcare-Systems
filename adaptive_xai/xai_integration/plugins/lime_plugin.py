# framework/xai_integration/plugins/lime_plugin.py
import re

from ..base_plugin import XAIPlugin
from ..output_schema import ExplanationResult
from ..plugin_registry import register_plugin
from ..adapters import resolve_adapter
from ..prediction_utils import predicted_value, probability_matrix, to_numpy


_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _lime_feature_name(condition: str) -> str:
    condition = str(condition)
    chained = re.fullmatch(
        rf"\s*(?P<lower>{_NUMBER_PATTERN})\s*(?P<lower_op><=|<)\s*"
        rf"(?P<feature>.+?)\s*(?P<upper_op><=|<)\s*"
        rf"(?P<upper>{_NUMBER_PATTERN})\s*",
        condition,
    )
    if chained:
        return chained.group("feature").strip()

    direct = re.fullmatch(
        r"\s*(?P<feature>.+?)\s*(?P<operator>>=|>|<=|<|==|=)\s*.+\s*",
        condition,
    )
    if direct:
        return direct.group("feature").strip()

    reverse = re.fullmatch(
        rf"\s*(?P<value>{_NUMBER_PATTERN})\s*(?P<operator>>=|>|<=|<)\s*"
        rf"(?P<feature>.+?)\s*",
        condition,
    )
    if reverse:
        return reverse.group("feature").strip()
    return condition


@register_plugin("LIME")
class LIMEPlugin(XAIPlugin):

    name = "LIME"

    def is_compatible(self, model_info: dict) -> bool:
        return model_info.get("input_type") in ("Tabular", "Image", "Text")

    def explain(self, model, input_data, model_info: dict, adapter=None) -> ExplanationResult:
        if adapter is None:
            adapter = resolve_adapter(model, model_info)

        input_type = model_info.get("input_type")

        if input_type == "Tabular":
            return self._explain_tabular(model, input_data, model_info, adapter)
        elif input_type == "Image":
            return self._explain_image(model, input_data, model_info, adapter)
        elif input_type == "Text":
            return self._explain_text(model, input_data, model_info, adapter)

        raise ValueError(f"LIME does not support input_type={input_type}")

    def _explain_tabular(self, model, input_data, model_info, adapter):
        import numpy as np
        import pandas as pd
        from lime.lime_tabular import LimeTabularExplainer

        background_data = model_info.get("background_data")
        if background_data is None:
            raise ValueError("LIME requires background_data for tabular input.")

        model_feature_names = getattr(model, "feature_names_in_", None)
        input_is_dataframe = isinstance(input_data, pd.DataFrame)
        if input_is_dataframe:
            input_frame = input_data
            if model_feature_names is not None:
                missing = [
                    name
                    for name in model_feature_names
                    if name not in input_frame.columns
                ]
                if missing:
                    raise ValueError(
                        "LIME input DataFrame is missing model columns: "
                        + ", ".join(map(str, missing))
                    )
                input_frame = input_frame.loc[:, model_feature_names]
            dataframe_columns = list(input_frame.columns)
            input_array = input_frame.to_numpy()
        else:
            input_array = np.asarray(input_data)
            dataframe_columns = (
                list(model_feature_names)
                if model_feature_names is not None
                else None
            )
            if (
                dataframe_columns is not None
                and len(dataframe_columns) != input_array.shape[-1]
            ):
                raise ValueError(
                    f"LIME input has {input_array.shape[-1]} features, but "
                    f"the model expects {len(dataframe_columns)} named features."
                )

        if input_array.ndim != 2 or input_array.shape[0] != 1:
            raise ValueError(
                "LIME tabular explanations require exactly one input row."
            )

        if isinstance(background_data, pd.DataFrame):
            if dataframe_columns is not None:
                missing = [
                    name
                    for name in dataframe_columns
                    if name not in background_data.columns
                ]
                if missing:
                    raise ValueError(
                        "LIME background DataFrame is missing model columns: "
                        + ", ".join(map(str, missing))
                    )
                background_data = background_data.loc[:, dataframe_columns]
            training_data = background_data.to_numpy()
        else:
            training_data = np.asarray(background_data)
            if dataframe_columns is not None:
                if training_data.shape[-1] != len(dataframe_columns):
                    raise ValueError(
                        "LIME background data has "
                        f"{training_data.shape[-1]} features, but the model "
                        f"expects {len(dataframe_columns)} named features."
                    )
                background_data = pd.DataFrame(
                    training_data,
                    columns=dataframe_columns,
                )

        if training_data.ndim != 2 or (
            training_data.shape[1] != input_array.shape[1]
        ):
            raise ValueError(
                "LIME background data and input must have the same number "
                "of features."
            )

        def model_input(values):
            if dataframe_columns is None:
                return values
            return pd.DataFrame(values, columns=dataframe_columns)

        instance = input_array[0]
        feature_names = model_info.get("feature_names")
        if dataframe_columns is not None:
            feature_names = dataframe_columns
        is_regression = model_info.get("task") == "Regression" or model_info.get(
            "category"
        ) in {"Linear", "Regression"}

        def predict_fn(values):
            if is_regression:
                return to_numpy(
                    adapter.predict(model, model_input(values))
                ).reshape(-1)

            try:
                predictions = adapter.predict_proba(model, model_input(values))
            except NotImplementedError:
                predictions = to_numpy(
                    adapter.predict(model, model_input(values))
                )
                if predictions.ndim == 1 or (
                    predictions.ndim == 2 and predictions.shape[1] == 1
                ):
                    classes = getattr(model, "classes_", None)
                    if classes is None:
                        training_predictions = to_numpy(
                            adapter.predict(
                                model,
                                model_input(training_data),
                            )
                        ).reshape(-1)
                        classes = np.unique(training_predictions)
                    classes = np.asarray(classes).reshape(-1)
                    if classes.size < 2:
                        raise ValueError(
                            "LIME classification needs predictions for at "
                            "least two classes."
                        )
                    labels = predictions.reshape(-1)
                    return np.column_stack(
                        [labels == label for label in classes]
                    ).astype(float)

            return probability_matrix(predictions)

        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode="regression" if is_regression else "classification",
            discretize_continuous=True,
        )
        explanation = explainer.explain_instance(
            instance,
            predict_fn,
            num_features=len(instance),
        )

        lime_features = [
            (_lime_feature_name(name), float(score))
            for name, score in explanation.as_list()
        ]
        ranking = [
            {"feature": name, "importance": score}
            for name, score in lime_features
        ]
        prediction = predicted_value(
            adapter.predict(model, model_input(input_array)),
            "Regression" if is_regression else "Classification",
            class_scores=(
                not is_regression
                and adapter.framework in {"pytorch", "tensorflow"}
            ),
        )

        return ExplanationResult(
            technique_used=self.name, prediction=prediction,
            feature_ranking=ranking, raw_output=lime_features,
        )

    def _explain_image(self, model, input_data, model_info, adapter):
        import numpy as np
        from lime.lime_image import LimeImageExplainer

        def predict_fn(images_batch):
            batch_input = np.asarray(images_batch)
            channel_order = model_info.get("channel_order")
            if channel_order is None:
                channel_order = (
                    "channels_first"
                    if adapter.framework == "pytorch"
                    else "channels_last"
                )
            if channel_order == "channels_first":
                batch_input = np.transpose(batch_input, (0, 3, 1, 2))
            output = adapter.predict_proba(model, batch_input)
            return (
                output.detach().cpu().numpy()
                if hasattr(output, "detach")
                else np.asarray(output)
            )

        image = to_numpy(input_data)
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        explainer = LimeImageExplainer()
        explanation = explainer.explain_instance(image, predict_fn, top_labels=1, num_samples=100)

        prediction = explanation.top_labels[0]
        _, mask = explanation.get_image_and_mask(prediction, positive_only=True, num_features=5)

        return ExplanationResult(
            technique_used=self.name, prediction=int(prediction),
            raw_output=mask.tolist(),
            metadata={"note": "raw_output is a segmentation mask highlighting influential regions"},
        )

    def _explain_text(self, model, input_data, model_info, adapter):
        from lime.lime_text import LimeTextExplainer

        texts = input_data if isinstance(input_data, list) else [input_data]

        def predict_fn(values):
            output = adapter.predict_proba(model, values)
            return output.detach().cpu().numpy() if hasattr(output, "detach") else output

        explanation = LimeTextExplainer(
            class_names=model_info.get("class_names"),
        ).explain_instance(texts[0], predict_fn)
        prediction = predicted_value(
            adapter.predict(model, texts),
            "Classification",
            class_scores=adapter.framework in {"pytorch", "tensorflow"},
        )
        return ExplanationResult(
            technique_used=self.name,
            prediction=prediction,
            feature_ranking=[
                {"feature": word, "importance": float(score)}
                for word, score in explanation.as_list()
            ],
            raw_output=explanation.as_list(),
        )