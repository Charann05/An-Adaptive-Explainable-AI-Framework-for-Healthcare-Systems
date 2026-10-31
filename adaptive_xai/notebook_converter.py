from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat


def _as_python_comments(markdown: str) -> str:
    """Convert markdown cells into Python comments so they stay readable."""
    text = str(markdown or "").rstrip()
    if not text:
        return ""

    lines = text.splitlines()
    comments = []
    for line in lines:
        comments.append(f"# {line}" if line else "#")
    return "\n".join(comments)


def _sanitize_code_cell(source: str) -> str:
    """Keep code cells runnable in plain Python scripts."""
    lines = []
    for raw_line in str(source or "").splitlines():
        stripped = raw_line.lstrip()
        if not stripped:
            lines.append("")
        elif stripped.startswith(("!", "%", "%%")):
            lines.append(f"# Skipped notebook magic/command: {raw_line}")
        else:
            lines.append(raw_line)
    return "\n".join(lines)


def convert_notebook_to_python(notebook_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convert a single .ipynb file into a runnable .py script."""
    notebook_file = Path(notebook_path).expanduser().resolve()
    if notebook_file.suffix.lower() != ".ipynb":
        raise ValueError(f"Expected a .ipynb file, got: {notebook_file}")
    if not notebook_file.is_file():
        raise FileNotFoundError(f"Notebook not found: {notebook_file}")

    notebook = nbformat.read(notebook_file, as_version=4)

    script_lines = [
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        f"# Converted from notebook: {notebook_file.name}",
        "# This file preserves the notebook logic as plain Python.",
        "# Notebook magics and shell commands were kept as comments to avoid runtime errors.",
        "",
    ]

    for index, cell in enumerate(notebook.cells, start=1):
        cell_source = cell.get("source", "")
        if cell.cell_type == "markdown":
            rendered = _as_python_comments(cell_source)
            if rendered:
                script_lines.append(f"# --- Markdown cell {index} ---")
                script_lines.append(rendered)
                script_lines.append("")
        elif cell.cell_type == "code":
            code = _sanitize_code_cell(cell_source)
            if code.strip():
                script_lines.append(f"# --- Code cell {index} ---")
                script_lines.append(code)
                script_lines.append("")

    output_file = Path(output_path).expanduser().resolve() if output_path else notebook_file.with_suffix(".py")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(script_lines).rstrip() + "\n", encoding="utf-8")
    return output_file


def _expand_targets(paths: list[str]) -> list[Path]:
    if not paths:
        return [Path.cwd()]

    targets: list[Path] = []
    for raw_path in paths:
        candidate = Path(raw_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Path does not exist: {candidate}")
        targets.append(candidate)
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks (.ipynb) into runnable Python scripts.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="One or more .ipynb files or directories to convert. Defaults to the current directory.",
    )
    args = parser.parse_args(argv)

    try:
        targets = _expand_targets(args.paths)
        converted_files: list[Path] = []

        for target in targets:
            if target.is_dir():
                notebook_files = sorted(target.rglob("*.ipynb"))
                if not notebook_files:
                    print(f"No .ipynb files found in: {target}", file=sys.stderr)
                    continue
                for notebook in notebook_files:
                    converted_files.append(convert_notebook_to_python(notebook))
            elif target.is_file():
                if target.suffix.lower() != ".ipynb":
                    raise ValueError(f"Only .ipynb files are supported: {target}")
                converted_files.append(convert_notebook_to_python(target))
            else:
                raise ValueError(f"Unsupported target: {target}")

        if not converted_files:
            print("No files were converted.", file=sys.stderr)
            return 1

        for file_path in converted_files:
            print(f"Converted: {file_path}")
        return 0

    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
