"""Unit tests for Monte Carlo visualization tools."""

import matplotlib
matplotlib.use("Agg")  # Headless backend for automated testing

from pathlib import Path
import pytest

from src.config import create_race_model, get_default_compounds
from src.montecarlo import simulate_monte_carlo
from src.stochastic import StochasticParameters
from src.strategies import Stint, Strategy
from src.visualization import (
    plot_1d_sensitivity,
    plot_2d_phase_diagram,
    plot_monte_carlo_dashboard,
    plot_sensitivity_dashboard,
    plot_strategy_boxplots,
    plot_strategy_cdf,
    plot_strategy_distributions,
    plot_strategy_histograms,
    plot_win_probability_matrix,
)


@pytest.fixture
def mc_results():
    compounds = get_default_compounds()
    model = create_race_model("bahrain")
    params = StochasticParameters()

    strat1 = Strategy(
        stints=[
            Stint(compound=compounds["Medium"], laps=26),
            Stint(compound=compounds["Hard"], laps=31),
        ],
        name="1-Stop (M-H)",
    )
    strat2 = Strategy(
        stints=[
            Stint(compound=compounds["Soft"], laps=15),
            Stint(compound=compounds["Medium"], laps=21),
            Stint(compound=compounds["Soft"], laps=21),
        ],
        name="2-Stop (S-M-S)",
    )

    res1 = simulate_monte_carlo(strat1, model, params, n_iterations=100, seed=42)
    res2 = simulate_monte_carlo(strat2, model, params, n_iterations=100, seed=42)
    return [res1, res2]


def test_plot_strategy_distributions(mc_results, tmp_path: Path) -> None:
    out_file = tmp_path / "distributions.png"
    plot_strategy_distributions(mc_results, save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_strategy_histograms(mc_results, tmp_path: Path) -> None:
    out_file = tmp_path / "histograms.png"
    plot_strategy_histograms(mc_results, bins=20, save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_strategy_cdf(mc_results, tmp_path: Path) -> None:
    out_file = tmp_path / "cdf.png"
    plot_strategy_cdf(mc_results, save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_strategy_boxplots(mc_results, tmp_path: Path) -> None:
    out_file = tmp_path / "boxplots.png"
    plot_strategy_boxplots(mc_results, save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_monte_carlo_dashboard(mc_results, tmp_path: Path) -> None:
    out_file = tmp_path / "dashboard.png"
    plot_monte_carlo_dashboard(mc_results, save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_win_probability_matrix(mc_results, tmp_path: Path) -> None:
    out_file = tmp_path / "win_matrix.png"
    plot_win_probability_matrix(mc_results, save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_1d_sensitivity(tmp_path: Path) -> None:
    import numpy as np
    from src.config import create_race_model, get_circuit_compounds
    from src.sensitivity import ParameterSweep1D, SensitivityParameter
    model = create_race_model("bahrain")
    compounds = get_circuit_compounds("bahrain")
    sweeper = ParameterSweep1D(model, compounds)
    res = sweeper.sweep_reoptimization(SensitivityParameter.TYRE_DEG_MULTIPLIER, np.linspace(1.0, 1.4, 3))

    out_file = tmp_path / "sensitivity_1d.png"
    plot_1d_sensitivity(res, save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_2d_phase_diagram(tmp_path: Path) -> None:
    from src.config import create_race_model, get_circuit_compounds
    from src.sensitivity import PhaseDiagram2D
    model = create_race_model("bahrain")
    compounds = get_circuit_compounds("bahrain")
    p2d = PhaseDiagram2D(model, compounds)
    res = p2d.compute_phase_map(resolution=(4, 4))

    out_file = tmp_path / "phase_map_2d.png"
    plot_2d_phase_diagram(res, save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_sensitivity_dashboard(tmp_path: Path) -> None:
    import numpy as np
    from src.config import create_race_model, get_circuit_compounds
    from src.sensitivity import ParameterSweep1D, PhaseDiagram2D, SensitivityParameter
    model = create_race_model("bahrain")
    compounds = get_circuit_compounds("bahrain")
    sweeper = ParameterSweep1D(model, compounds)
    sweep_deg = sweeper.sweep_reoptimization(SensitivityParameter.TYRE_DEG_MULTIPLIER, np.linspace(1.0, 1.4, 3))
    sweep_pit = sweeper.sweep_reoptimization(SensitivityParameter.PIT_LOSS, np.linspace(20.0, 24.0, 3))
    p2d = PhaseDiagram2D(model, compounds)
    phase_res = p2d.compute_phase_map(resolution=(3, 3))

    out_file = tmp_path / "sens_dashboard.png"
    plot_sensitivity_dashboard(sweep_deg, sweep_pit, phase_res, save_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0

