import ast
import nbformat


def get_notebook_trees(file):
    trees = []
    with open(file, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        code = cell.source
        if not code.strip():
            continue

        try:
            tree = ast.parse(code)
            trees.append(tree)

        except SyntaxError:
            continue

    return trees