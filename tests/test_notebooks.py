"""Unit tests for demonstration Jupyter notebooks."""

from pathlib import Path
import pytest
import nbformat


NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
EXPECTED_NOTEBOOKS = [
    "01_synthetic_model_exploration.ipynb",
    "02_monte_carlo_and_optimization.ipynb",
    "03_sensitivity_and_decision_maps.ipynb",
    "04_historical_data_validation.ipynb",
    "05_scenario_analysis_and_what_if.ipynb",
]


def test_all_notebooks_exist():
    """Verify that all five curriculum demonstration notebooks are present."""
    assert NOTEBOOK_DIR.exists(), f"Notebooks directory {NOTEBOOK_DIR} does not exist"
    for nb_name in EXPECTED_NOTEBOOKS:
        nb_path = NOTEBOOK_DIR / nb_name
        assert nb_path.exists(), f"Expected notebook {nb_name} was not found in {NOTEBOOK_DIR}"


@pytest.mark.parametrize("nb_name", EXPECTED_NOTEBOOKS)
def test_notebook_format_and_cells(nb_name: str):
    """Verify that each notebook is valid nbformat v4 and contains required cell structures."""
    nb_path = NOTEBOOK_DIR / nb_name
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Check nbformat version
    assert nb.nbformat == 4
    assert len(nb.cells) >= 5, f"{nb_name} should contain at least 5 cells"

    cell_types = {cell.cell_type for cell in nb.cells}
    assert "markdown" in cell_types, f"{nb_name} must contain markdown cells"
    assert "code" in cell_types, f"{nb_name} must contain code cells"

    # Verify that all code cells have non-empty source
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    for idx, cell in enumerate(code_cells):
        assert len(cell.source.strip()) > 0, f"Code cell {idx} in {nb_name} has empty source"
