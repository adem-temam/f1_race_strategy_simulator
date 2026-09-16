"""Automated test suite for Phase 3: Strategy Optimization Engine.
Verifies Dynamic Programming, Combinatorial Grid Search, Pit Windows, and Pareto Risk Analysis.
"""

import pytest

from src.config import create_race_model, get_circuit_compounds, get_default_compounds
from src.optimization import (
    DynamicProgrammingSolver,
    OptimizationObjective,
    OptimizationResult,
    ParetoPoint,
    PitWindowAnalyzer,
    StrategyOptimizer,
    compute_pareto_frontier,
    simulate_race_time_fast,
)
from src.simulation import simulate_race
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


def test_dynamic_programming_solver(bahrain_setup):
    """Verify Dynamic Programming solver finds a valid, FIA-compliant strategy."""
    model, compounds = bahrain_setup
    dp = DynamicProgrammingSolver(model, compounds)
    optimal_strat, race_time = dp.solve()

    assert optimal_strat.total_laps == model.circuit.total_laps
    assert len(optimal_strat.compounds_used) >= 2
    assert optimal_strat.num_stops >= 1
    assert race_time > 0

    # Sanity check: deterministic simulation matches or is within 1s of DP value
    sim_res = simulate_race(optimal_strat, model)
    assert abs(sim_res.total_time - race_time) < 1.0


def test_combinatorial_search_generates_valid_strategies(bahrain_setup):
    """Verify that all generated candidate strategies conform to FIA regulations."""
    model, compounds = bahrain_setup
    optimizer = StrategyOptimizer(model, compounds)

    candidates = optimizer.grid_solver.generate_candidate_strategies(max_stops=2, step=3)
    assert len(candidates) > 100

    for strat in candidates:
        is_valid, msg = strat.validate(model.circuit.total_laps)
        assert is_valid, f"Invalid strategy generated: {msg}"
        assert len(strat.compounds_used) >= 2
        assert strat.total_laps == model.circuit.total_laps
        assert all(stint.laps >= 5 for stint in strat.stints)


def test_low_deg_monza_prefers_one_stop(monza_setup):
    """Verify that low-degradation, high-pit-loss Monza prefers a 1-stop strategy."""
    model, compounds = monza_setup
    optimizer = StrategyOptimizer(model, compounds)

    res = optimizer.optimize(max_stops=2, step=2)
    assert isinstance(res, OptimizationResult)
    # The optimal strategy at Monza should be a 1-stop strategy
    assert res.optimal_strategy.num_stops == 1
    assert len(res.optimal_strategy.compounds_used) >= 2


def test_high_deg_bahrain_prefers_two_stop(bahrain_setup):
    """Verify that high-abrasion Bahrain prefers a 2-stop strategy."""
    model, compounds = bahrain_setup
    optimizer = StrategyOptimizer(model, compounds)

    res = optimizer.optimize(max_stops=2, step=2)
    assert isinstance(res, OptimizationResult)
    # High degradation and cliff at Sakhir heavily favors 2-stop
    assert res.optimal_strategy.num_stops == 2
    assert len(res.optimal_strategy.compounds_used) >= 2


def test_pit_window_analyzer(bahrain_setup):
    """Verify that tactical pit windows are accurately bounded by tolerance."""
    model, compounds = bahrain_setup
    optimizer = StrategyOptimizer(model, compounds)
    res = optimizer.optimize(max_stops=2, step=2)

    windows = res.pit_windows
    assert len(windows) == res.optimal_strategy.num_stops
    assert len(windows) >= 1

    for pw in windows:
        assert pw.window_open_lap <= pw.optimal_lap <= pw.window_close_lap
        assert pw.window_size >= 1
        assert pw.delta_open <= pw.tolerance_seconds + 0.05
        assert pw.delta_close <= pw.tolerance_seconds + 0.05


def test_max_stops_constraint_enforcement(bahrain_setup):
    """Verify that max_stops parameter strictly constrains the strategy search."""
    model, compounds = bahrain_setup
    optimizer = StrategyOptimizer(model, compounds)

    res_1stop = optimizer.optimize(max_stops=1, step=2)
    assert res_1stop.optimal_strategy.num_stops == 1

    res_2stop = optimizer.optimize(max_stops=2, step=2)
    assert res_2stop.optimal_strategy.num_stops <= 2


def test_multi_objective_risk_optimization(monza_setup):
    """Verify optimization with MIN_RISK_P95 objective and Pareto frontier generation."""
    model, compounds = monza_setup
    optimizer = StrategyOptimizer(model, compounds)

    res = optimizer.optimize(
        max_stops=2,
        objective=OptimizationObjective.MIN_RISK_P95,
        mc_iterations=100,
        step=2,
    )

    assert res.objective == OptimizationObjective.MIN_RISK_P95
    assert len(res.pareto_frontier) > 0
    # Verify ParetoPoint fields
    first_point = res.pareto_frontier[0]
    assert first_point.expected_time > 0
    assert first_point.p95_time >= first_point.expected_time
    assert first_point.risk_penalty >= 0


def test_dynamic_programming_max_stops_enforced(bahrain_setup):
    """Verify that DynamicProgrammingSolver strictly respects the max_stops argument."""
    model, compounds = bahrain_setup
    dp = DynamicProgrammingSolver(model, compounds)

    strat_1stop, _ = dp.solve(max_stops=1)
    assert strat_1stop.num_stops == 1
    assert len(strat_1stop.compounds_used) >= 2

    strat_2stop, _ = dp.solve(max_stops=2)
    assert strat_2stop.num_stops <= 2
    assert len(strat_2stop.compounds_used) >= 2


def test_fast_sim_matches_standard_simulate_race(bahrain_setup, monza_setup):
    """Verify that simulate_race_time_fast accurately replicates simulate_race."""
    for model, compounds in [bahrain_setup, monza_setup]:
        strats = [
            Strategy([
                Stint(compounds[list(compounds.keys())[0]], 20),
                Stint(compounds[list(compounds.keys())[1]], model.circuit.total_laps - 20),
            ]),
            Strategy([
                Stint(compounds[list(compounds.keys())[0]], 15),
                Stint(compounds[list(compounds.keys())[1]], 20),
                Stint(compounds[list(compounds.keys())[0]], model.circuit.total_laps - 35),
            ]),
        ]
        for s in strats:
            std_time = simulate_race(s, model, validate=False).total_time
            fast_time = simulate_race_time_fast(s, model)
            assert fast_time == pytest.approx(std_time, abs=1e-5)


def test_strict_pareto_non_domination():
    """Verify compute_pareto_frontier filters out strictly dominated strategies."""
    compounds = get_default_compounds()
    s = compounds["Soft"]

    m = compounds["Medium"]
    dummy_strat1 = Strategy([Stint(s, 20), Stint(m, 37)], name="Strat A")
    dummy_strat2 = Strategy([Stint(m, 25), Stint(s, 32)], name="Strat B")
    dummy_strat3 = Strategy([Stint(s, 15), Stint(m, 42)], name="Strat C (Dominated)")

    # Strat A: fastest expected, moderate risk
    pt_a = ParetoPoint(dummy_strat1, expected_time=5400.0, p95_time=5415.0, std_dev=3.0, risk_penalty=15.0)
    # Strat B: slightly slower expected, but lower risk
    pt_b = ParetoPoint(dummy_strat2, expected_time=5405.0, p95_time=5410.0, std_dev=2.0, risk_penalty=5.0)
    # Strat C: worse in BOTH expected and P95 compared to Strat A (strictly dominated)
    pt_c = ParetoPoint(dummy_strat3, expected_time=5410.0, p95_time=5420.0, std_dev=4.0, risk_penalty=10.0)

    frontier = compute_pareto_frontier([pt_a, pt_b, pt_c])
    assert len(frontier) == 2
    assert pt_a in frontier
    assert pt_b in frontier
    assert pt_c not in frontier

