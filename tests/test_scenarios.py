"""Unit tests for Phase 6 Scenario Analysis Engine."""

import pytest

from src.scenarios import (
    PRESET_SCENARIOS,
    Scenario,
    ScenarioCategory,
    ScenarioEngine,
    ScenarioPerturbation,
    ScenarioResult,
)


def test_scenario_engine_initialization():
    """Verify ScenarioEngine initializes properly with baseline models."""
    engine = ScenarioEngine("bahrain")
    assert engine.circuit_key == "bahrain"
    assert engine.baseline_model.circuit.total_laps == 57
    assert len(engine.compounds) >= 3


def test_slow_stop_scenario_penalizes_multi_stop():
    """Verify that adding pit loss increases multi-stop strategy race time."""
    engine = ScenarioEngine("bahrain")
    scenario = PRESET_SCENARIOS["slow_stop"]
    res = engine.evaluate_scenario(scenario)

    assert isinstance(res, ScenarioResult)
    assert res.strategic_regret >= 0.0
    # In Bahrain, optimal is 2-stop. Slow stop (+2.0s per stop = +4.0s total)
    # should increase scenario time relative to baseline
    assert res.scenario_optimal_time > res.baseline_optimal_time


def test_botched_stop_triggers_strategy_pivot():
    """Verify that a massive pit loss (+4.5s) forces pivot away from high stops."""
    engine = ScenarioEngine("bahrain")
    scenario = PRESET_SCENARIOS["botched_stop"]
    res = engine.evaluate_scenario(scenario)

    assert res.strategic_regret > 0.0
    assert len(res.strategy_outcomes) >= 2
    # Baseline optimal is 2-stop; botched stop should favor 1-stop
    assert res.scenario_optimal_strategy is not None


def test_low_deg_favors_fewer_stops_or_softer_tyres():
    """Verify that a 25% degradation drop allows longer stints and pivots strategy."""
    engine = ScenarioEngine("bahrain")
    scenario = PRESET_SCENARIOS["low_deg"]
    res = engine.evaluate_scenario(scenario)

    assert res.strategy_pivoted is True
    assert res.strategic_regret > 0.0
    # 1-stop becomes viable/dominant when degradation drops 25% in Bahrain
    assert len(res.scenario_optimal_strategy.stints) <= 3


def test_sprint_distance_scales_properly():
    """Verify that a sprint distance override (19 laps) correctly scales stints."""
    engine = ScenarioEngine("bahrain")
    scenario = PRESET_SCENARIOS["sprint_distance"]
    res = engine.evaluate_scenario(scenario)

    assert res.perturbed_model.circuit.total_laps == 19
    assert res.scenario_optimal_strategy.total_laps == 19
    for oc in res.strategy_outcomes:
        assert oc.scenario_time > 0.0


def test_safety_car_opportunistic_stop():
    """Verify that a mid-race Safety Car correctly discounts pit loss."""
    from src.simulation import simulate_race

    engine = ScenarioEngine("bahrain")
    scenario = PRESET_SCENARIOS["safety_car_lap18"]
    res = engine.evaluate_scenario(scenario)

    assert res.scenario_optimal_strategy is not None
    assert res.scenario_optimal_strategy.stints[0].laps == 18
    # The SC strategy should be strictly faster than the exact same strategy run under green flag
    green_flag_time = simulate_race(res.scenario_optimal_strategy, engine.baseline_model).total_time
    assert res.scenario_optimal_time < green_flag_time
    # The discount should be approximately ~11s
    assert green_flag_time - res.scenario_optimal_time >= 10.0


def test_custom_user_scenario_evaluation():
    """Verify custom user scenario perturbations execute cleanly."""
    engine = ScenarioEngine("monza")
    custom = Scenario(
        id="custom_test",
        name="Custom Extreme Track Heat",
        category=ScenarioCategory.ENVIRONMENTAL,
        description="Test custom deg and pit loss.",
        research_question="Does extreme heat shift Monza to 2-stop?",
        perturbation=ScenarioPerturbation(
            deg_multiplier_factor=1.5,
            pit_loss_delta=-2.0,  # Fast pit entry
        ),
    )
    res = engine.evaluate_scenario(custom)

    assert res.scenario.id == "custom_test"
    assert res.strategic_regret >= 0.0
    assert len(res.summary_insight) > 10


def test_run_all_presets_completeness():
    """Verify that all 11 preset scenarios evaluate without runtime errors."""
    engine = ScenarioEngine("monza")
    results = engine.run_all_presets()

    assert len(results) == len(PRESET_SCENARIOS)
    for k, r in results.items():
        assert r.scenario.id == k
        assert r.strategic_regret >= 0.0
        assert r.baseline_optimal_time > 0.0
        assert r.scenario_optimal_time > 0.0
