import tempfile
import unittest
from pathlib import Path

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.model_analyzer.read_dataset import scan_dataset


class ReadDatasetTests(unittest.TestCase):
    def test_prefers_widest_feature_dataset_and_matching_separate_target(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            data = project / "data"
            data.mkdir()
            (data / "clean_y.csv").write_text(
                "target\n0\n1\n0\n",
                encoding="utf-8",
            )
            (data / "heart_disease.csv").write_text(
                "age,chol,target\n40,200,0\n50,220,1\n60,180,0\n",
                encoding="utf-8",
            )
            (data / "cleaned_X.csv").write_text(
                "age,chol,thalach,oldpeak\n"
                "40,200,150,1.0\n"
                "50,220,140,2.0\n"
                "60,180,130,0.5\n",
                encoding="utf-8",
            )
            evidences = {
                "dataset_columns": [],
                "feature_names": [],
                "num_inputs": [],
                "target_column": [],
            }
            file_info = [
                {"name": "data/clean_y.csv", "extension": ".csv"},
                {"name": "data/heart_disease.csv", "extension": ".csv"},
                {"name": "data/cleaned_X.csv", "extension": ".csv"},
            ]

            scan_dataset(project, evidences, file_info)

        self.assertEqual(
            evidences["dataset_columns"][0]["value"],
            ["age", "chol", "thalach", "oldpeak", "target"],
        )
        self.assertEqual(
            evidences["feature_names"][0]["value"],
            ["age", "chol", "thalach", "oldpeak"],
        )
        self.assertEqual(evidences["num_inputs"][0]["value"], 4)
        self.assertEqual(evidences["target_column"][0]["value"], "target")

    def test_does_not_borrow_target_from_different_sized_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "features.csv").write_text(
                "age,chol\n40,200\n50,220\n",
                encoding="utf-8",
            )
            (project / "target.csv").write_text(
                "target\n0\n1\n0\n",
                encoding="utf-8",
            )
            evidences = {
                "dataset_columns": [],
                "feature_names": [],
                "num_inputs": [],
                "target_column": [],
            }

            scan_dataset(
                project,
                evidences,
                [
                    {"name": "target.csv", "extension": ".csv"},
                    {"name": "features.csv", "extension": ".csv"},
                ],
            )

        self.assertEqual(evidences["dataset_columns"][0]["value"], ["age", "chol"])
        self.assertEqual(evidences["num_inputs"][0]["value"], 2)
        self.assertEqual(evidences["target_column"], [])


if __name__ == "__main__":
    unittest.main()
