from pathlib import Path
import ast
import json

EXPECTED_COLUMNS = [
    "season",
    "date",
    "event",
    "patch",
    "blue_team",
    "red_team",
    "winner",
    "blue_team_win",
]


def _load_notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "".join(
        "\n".join(cell["source"])
        + "\n"
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _extract_returned_columns(function_node: ast.FunctionDef) -> list[str]:
    for node in ast.walk(function_node):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Subscript):
            slice_node = node.value.slice
            if isinstance(slice_node, ast.List):
                columns = []
                for element in slice_node.elts:
                    if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                        raise AssertionError("artifact column selection must be a string list")
                    columns.append(element.value)
                return columns
    raise AssertionError("could not find artifact column selection in notebook helper")


def test_clean_matches_artifact_has_required_columns():
    notebook_path = Path("notebooks/01_data_pull_and_audit.ipynb")
    source = _load_notebook_source(notebook_path)
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_matches_csv"
    )

    assert "from mlops.data import load_matches" not in source
    assert "import sys" not in source
    assert 'df.to_csv(ARTIFACTS_DIR / "clean_matches.csv", index=False)' in source
    assert _extract_returned_columns(helper) == EXPECTED_COLUMNS
    assert all(f'"{column}"' in source for column in EXPECTED_COLUMNS)
