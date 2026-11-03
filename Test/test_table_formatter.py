import unittest

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from adaptive_xai.cli_display import format_table
from adaptive_xai.xai_selector.report_formatter import format_result


class TableFormatterTests(unittest.TestCase):
    def test_wraps_long_values_without_widening_table(self):
        names = [f"feature_{index:02}" for index in range(25)]
        table = format_table([("Feature Names", str(names))])
        lines = table.splitlines()

        self.assertGreater(len(lines), 5)
        self.assertLessEqual(max(map(len, lines)), 79)
        self.assertIn("feature_24", table)
        self.assertEqual(sum("| Feature Names " in line for line in lines), 1)

    def test_wraps_long_values_and_reasons_in_selector_report(self):
        report = format_result(
            {"framework": "Scikit-Learn", "architecture": "Logistic Regression"},
            {
                "selected": "SHAP",
                "alternatives": ["LIME"],
                "reason": [
                    "This is a longer explanation that should wrap over multiple "
                    "lines without making the report excessively wide."
                ],
            },
        )

        self.assertLessEqual(max(map(len, report.splitlines())), 79)
        self.assertIn("Selected Technique", report)
        self.assertIn("excessively", report)
        self.assertIn("wide.", report)
        self.assertTrue(
            any(
                line.startswith("|") and line.split("|")[1].strip() == ""
                for line in report.splitlines()
            )
        )


if __name__ == "__main__":
    unittest.main()
