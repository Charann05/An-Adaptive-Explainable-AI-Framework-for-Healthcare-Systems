import json
from pathlib import Path
from typing import Any


class TemplateEngine:

    def __init__(self, template_directory: str | Path):
        self.template_directory = Path(template_directory)
        self._cache: dict[str, dict[str, str]] = {}

    def load_template(self, filename: str) -> dict[str, str]:
        if filename not in self._cache:
            template_path = self.template_directory / filename

            if not template_path.is_file():
                raise FileNotFoundError(
                    f"Template file not found: {template_path}"
                )

            with template_path.open("r", encoding="utf-8") as file:
                template = json.load(file)
            if not isinstance(template, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in template.items()
            ):
                raise ValueError(
                    f"Template must contain a JSON object of string values: "
                    f"{template_path}"
                )
            self._cache[filename] = template

        return self._cache[filename]

    @staticmethod
    def render(template: str, **values: Any) -> str:
        return template.format_map(values)
