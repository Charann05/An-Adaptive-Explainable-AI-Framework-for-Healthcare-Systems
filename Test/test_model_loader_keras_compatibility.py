import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.xai_integration.model_loader import (
    ModelLoader,
    _keras_legacy_custom_objects,
)


class FakeSeparableConv2D:
    @classmethod
    def from_config(cls, config):
        return config


class KerasModelLoaderCompatibilityTests(unittest.TestCase):
    def test_loads_without_compilation_and_supplies_compatibility_layer(self):
        load_arguments = {}

        def load_model(path, **kwargs):
            load_arguments.update(path=path, **kwargs)
            return "loaded model"

        fake_tensorflow = SimpleNamespace(
            keras=SimpleNamespace(
                layers=SimpleNamespace(
                    SeparableConv2D=FakeSeparableConv2D,
                ),
                models=SimpleNamespace(load_model=load_model),
            )
        )

        with patch.dict(sys.modules, {"tensorflow": fake_tensorflow}):
            result = ModelLoader._load_tensorflow(Path("model.h5"))

        self.assertEqual(result, "loaded model")
        self.assertFalse(load_arguments["compile"])
        compatible_layer = load_arguments["custom_objects"]["SeparableConv2D"]
        loaded_config = compatible_layer.from_config({
            "groups": 1,
            "kernel_initializer": "legacy_initializer",
            "kernel_regularizer": None,
            "kernel_constraint": None,
            "depthwise_initializer": "depthwise_initializer",
            "pointwise_initializer": "pointwise_initializer",
            "depthwise_regularizer": None,
            "pointwise_regularizer": None,
        })
        self.assertNotIn("groups", loaded_config)
        self.assertNotIn("kernel_initializer", loaded_config)
        self.assertEqual(
            loaded_config["depthwise_initializer"],
            "depthwise_initializer",
        )
        self.assertEqual(
            loaded_config["pointwise_initializer"],
            "pointwise_initializer",
        )

    def test_rejects_unsupported_grouped_separable_convolution(self):
        fake_tensorflow = SimpleNamespace(
            keras=SimpleNamespace(
                layers=SimpleNamespace(
                    SeparableConv2D=FakeSeparableConv2D,
                )
            )
        )

        with patch.dict(sys.modules, {"tensorflow": fake_tensorflow}):
            compatible_layer = _keras_legacy_custom_objects(
                fake_tensorflow
            )["SeparableConv2D"]
            with self.assertRaisesRegex(ValueError, "groups other than 1"):
                compatible_layer.from_config({"groups": 2})


if __name__ == "__main__":
    unittest.main()
