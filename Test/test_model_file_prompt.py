import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from framework.run import prompt_for_model_file


class PromptForModelFileTests(unittest.TestCase):
    def test_accepts_a_path_relative_to_the_project_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "models" / "classifier.h5"
            model_file.parent.mkdir()
            model_file.touch()

            with patch("builtins.input", return_value="models/classifier.h5"):
                selected = prompt_for_model_file(directory, "Keras")

            self.assertEqual(selected, model_file.resolve())

    def test_reprompts_when_the_selected_path_is_not_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            model_file = Path(directory) / "classifier.pkl"
            model_file.touch()

            with patch(
                "builtins.input",
                side_effect=["missing.pkl", "classifier.pkl"],
            ) as prompt:
                selected = prompt_for_model_file(directory, "sklearn")

            self.assertEqual(selected, model_file.resolve())
            self.assertEqual(prompt.call_count, 2)

    def test_blank_input_cancels_the_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("builtins.input", return_value="  "):
                selected = prompt_for_model_file(directory, "Keras")

            self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
