"""Unit tests for Phase 4: Sensitivity Analysis & Decision Phase Maps."""

import numpy as np
import pytest

from src.config import create_race_model, get_circuit_compounds
from src.sensitivity import (
    CrossoverFinder,
    ParameterSweep1D,
    PhaseDiagram2D,
    RobustnessAnalyzer,
    SensitivityParameter,
    clone_model_with_parameter,
    get_nominal_parameter_value,
)
from src.strategies import Stint, Strategy


@pytest.fixture
def bahrain_setup():
    model = create_race_model("bahrain")
    compounds = get_circuit_compounds("bahrain")
    return model, compounds


@pytest.fixture
def monza_setup():
    model = create_race_model("monza")
    compounds = get_circuit_compounds("monza")
    return model, compounds


def test_1d_sweep_monotonicity(bahrain_setup):
    """Verify that increasing pit loss and degradation monotonically increases total race time."""
    model, compounds = bahrain_setup
    s = compounds["Soft"]
    m = compounds["Medium"]
    strat = Strategy([Stint(m, 26), Stint(s, 31)], name="M26-S31")

    sweeper = ParameterSweep1D(model, compounds)

    # Pit loss sweep
    pit_vals = np.array([18.0, 22.0, 26.0, 30.0])
    res_pit = sweeper.sweep_strategies([strat], SensitivityParameter.PIT_LOSS, pit_vals)
    times_pit = res_pit.strategy_times[strat.description]
    assert np.all(np.diff(times_pit) > 0), "Race time must increase strictly with pit loss"

    # Degradation sweep
    deg_vals = np.array([0.8, 1.0, 1.3, 1.6])
    res_deg = sweeper.sweep_strategies([strat], SensitivityParameter.TYRE_DEG_MULTIPLIER, deg_vals)
    times_deg = res_deg.strategy_times[strat.description]
    assert np.all(np.diff(times_deg) > 0), "Race time must increase strictly with tyre degradation"


def test_brent_crossover_precision(bahrain_setup):
    """Verify that Brent's method discovers the exact tipping point where Delta T = 0."""
    model, compounds = bahrain_setup
    cf = CrossoverFinder(model, compounds)

    # Crossover on tyre degradation for Bahrain
    c_pt = cf.find_1_vs_2_stop_crossover(
        SensitivityParameter.TYRE_DEG_MULTIPLIER,
        search_range=(0.8, 1.5),
        xtol=1e-4,
    )

    assert c_pt.found is True
    assert 1.10 <= c_pt.crossover_value <= 1.25, f"Unexpected crossover value {c_pt.crossover_value}"
    assert c_pt.strategy_a_name == "1-Stop"
    assert c_pt.strategy_b_name == "2-Stop"
    assert c_pt.sensitivity_derivative > 0, "d(Delta T)/d(deg) must be positive"


def test_circuit_nominal_regimes(bahrain_setup, monza_setup):
    """Assert that Bahrain baseline is in 2-stop regime and Monza is in 1-stop regime."""
    b_model, b_compounds = bahrain_setup
    m_model, m_compounds = monza_setup

    cf_b = CrossoverFinder(b_model, b_compounds)
    c_b = cf_b.find_1_vs_2_stop_crossover(
        SensitivityParameter.TYRE_DEG_MULTIPLIER,
        search_range=(0.8, 1.6),
    )
    # Bahrain nominal (1.30) is greater than crossover (~1.16), so 2-stop wins
    assert c_b.nominal_value > c_b.crossover_value
    assert c_b.distance_from_nominal < 0  # crossover - nominal is negative

    cf_m = CrossoverFinder(m_model, m_compounds)
    c_m = cf_m.find_1_vs_2_stop_crossover(
        SensitivityParameter.TYRE_DEG_MULTIPLIER,
        search_range=(0.8, 1.5),
    )
    # Monza nominal (0.80) is less than crossover (~1.08), so 1-stop wins
    assert c_m.nominal_value < c_m.crossover_value
    assert c_m.distance_from_nominal > 0  # crossover - nominal is positive


def test_crossover_not_found_handling(monza_setup):
    """Verify graceful handling when no root exists in search interval."""
    m_model, m_compounds = monza_setup
    cf = CrossoverFinder(m_model, m_compounds)

    # In [0.4, 0.7], 1-stop strictly wins throughout (crossover is at 1.08)
    c_pt = cf.find_1_vs_2_stop_crossover(
        SensitivityParameter.TYRE_DEG_MULTIPLIER,
        search_range=(0.4, 0.7),
    )
    assert c_pt.found is False
    assert np.isnan(c_pt.crossover_value)
    assert "No crossover detected" in c_pt.message


def test_2d_phase_diagram_shape_and_boundaries(bahrain_setup):
    """Verify 2D phase map dimensions, grid shapes, and nominal point identification."""
    model, compounds = bahrain_setup
    p2d = PhaseDiagram2D(model, compounds)

    res = p2d.compute_phase_map(
        param_x=SensitivityParameter.PIT_LOSS,
        x_range=(18.0, 28.0),
        param_y=SensitivityParameter.TYRE_DEG_MULTIPLIER,
        y_range=(0.8, 1.6),
        resolution=(6, 5),
    )

    assert res.x_grid.shape == (5, 6)
    assert res.y_grid.shape == (5, 6)
    assert res.optimal_stops_grid.shape == (5, 6)
    assert res.delta_grid.shape == (5, 6)
    assert len(res.winning_strategies) == 5
    assert len(res.winning_strategies[0]) == 6

    # Bahrain nominal point should be 2-stop
    assert res.nominal_stops == 2
    assert res.nominal_point == (22.5, 1.3)


def test_elasticity_computation(bahrain_setup):
    """Verify dimensionless elasticity calculation."""
    model, compounds = bahrain_setup
    s = compounds["Soft"]
    m = compounds["Medium"]
    strat_1 = Strategy([Stint(m, 26), Stint(s, 31)], name="1-Stop (M-S)")
    strat_2 = Strategy([Stint(s, 15), Stint(m, 21), Stint(s, 21)], name="2-Stop (S-M-S)")

    sweeper = ParameterSweep1D(model, compounds)

    # 1. Pit loss elasticity: both positive, 2-stop greater than 1-stop
    vals_pit = np.array([20.0, 22.5, 25.0])
    res_pit = sweeper.sweep_strategies([strat_1, strat_2], SensitivityParameter.PIT_LOSS, vals_pit)
    assert res_pit.elasticities[strat_1.description] > 0
    assert res_pit.elasticities[strat_2.description] > 0
    assert res_pit.elasticities[strat_2.description] > res_pit.elasticities[strat_1.description]

    # 2. Tyre degradation elasticity: 1-stop has longer stint, so deg elasticity is higher
    vals_deg = np.array([1.0, 1.3, 1.6])
    res_deg = sweeper.sweep_strategies([strat_1, strat_2], SensitivityParameter.TYRE_DEG_MULTIPLIER, vals_deg)
    assert res_deg.elasticities[strat_1.description] > 0
    assert res_deg.elasticities[strat_2.description] > 0
    assert res_deg.elasticities[strat_1.description] > res_deg.elasticities[strat_2.description]


def test_robustness_analyzer(bahrain_setup):
    """Verify minimax regret and expected regret analysis."""
    model, compounds = bahrain_setup
    s = compounds["Soft"]
    m = compounds["Medium"]
    strat_1 = Strategy([Stint(m, 26), Stint(s, 31)], name="1-Stop")
    strat_2 = Strategy([Stint(s, 15), Stint(m, 21), Stint(s, 21)], name="2-Stop")

    analyzer = RobustnessAnalyzer(model, compounds)
    vals = np.linspace(0.8, 1.6, 9)
    regret_res = analyzer.compute_regret([strat_1, strat_2], SensitivityParameter.TYRE_DEG_MULTIPLIER, vals)

    assert len(regret_res.max_regret) == 2
    for r in regret_res.max_regret.values():
        assert r >= 0.0, "Regret must be non-negative"
    assert regret_res.minimax_strategy in regret_res.strategy_names


def test_soft_compound_delta_sensitivity(bahrain_setup):
    """Verify that sweeping COMPOUND_DELTA_SOFT impacts race times and discovers a crossover."""
    model, compounds = bahrain_setup
    cf = CrossoverFinder(model, compounds)

    pt = cf.find_1_vs_2_stop_crossover(
        SensitivityParameter.COMPOUND_DELTA_SOFT,
        search_range=(-0.5, 0.5),
    )
    assert pt.found is True
    assert -0.2 <= pt.crossover_value <= 0.3
    assert pt.strategy_a_name == "2-Stop"
    assert pt.strategy_b_name == "1-Stop"


def test_reoptimization_elasticity_populated(bahrain_setup):
    """Verify that sweep_reoptimization computes dimensionless elasticity for optimal strategies."""
    model, compounds = bahrain_setup
    sweeper = ParameterSweep1D(model, compounds)
    vals = np.linspace(0.8, 1.6, 5)
    res = sweeper.sweep_reoptimization(SensitivityParameter.TYRE_DEG_MULTIPLIER, vals)

    assert "Optimal 1-Stop" in res.elasticities
    assert "Optimal 2-Stop" in res.elasticities
    assert res.elasticities["Optimal 1-Stop"] > 0
    assert res.elasticities["Optimal 2-Stop"] > 0


def test_pit_loss_crossover_derivative_smoothness(monza_setup):
    """Verify that Monza pit loss crossover derivative does not suffer from discrete traffic spikes."""
    model, compounds = monza_setup
    cf = CrossoverFinder(model, compounds)
    c_pt = cf.find_1_vs_2_stop_crossover(
        SensitivityParameter.PIT_LOSS,
        search_range=(16.0, 26.0),
    )
    assert c_pt.found is True
    # Physical derivative is around -1.0 to -2.0 s/unit, definitely not spiked to -20+ s/unit
    assert -5.0 <= c_pt.sensitivity_derivative <= -0.5

