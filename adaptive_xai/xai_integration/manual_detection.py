import ast
from pathlib import Path

from ..model_analyzer.load_model_files import collect_files
from ..model_analyzer.read_notebook import get_notebook_trees


XAI_IMPORTS = {
    "shap": "SHAP",
    "lime": "LIME",
    "pytorch_grad_cam": "GradCAM",
    "treeinterpreter": "TreeInterpreter",
}


def detect_manual_xai_usage(
    project_root: str | Path,
) -> dict[str, list[str]]:
    root = Path(project_root).expanduser().resolve()
    detected: dict[str, set[str]] = {}

    for relative_path in collect_files(root):
        source_path = Path(relative_path)
        if source_path.name == "adap_xai.py":
            continue
        if source_path.suffix.lower() not in {".py", ".ipynb"}:
            continue

        full_path = root / source_path
        if source_path.suffix.lower() == ".ipynb":
            trees = get_notebook_trees(full_path)
        else:
            try:
                source = full_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            try:
                trees = [ast.parse(source, filename=str(full_path))]
            except SyntaxError:
                continue

        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules = [node.module]
                else:
                    continue

                for module_name in imported_modules:
                    package_name = module_name.split(".", maxsplit=1)[0]
                    technique = XAI_IMPORTS.get(package_name)
                    if technique is not None:
                        detected.setdefault(technique, set()).add(relative_path)

    return {
        technique: sorted(paths)
        for technique, paths in sorted(detected.items())
    }
