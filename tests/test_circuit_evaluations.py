"""Automated test suite executing multiple evaluations for all 3 Grand Prix circuits
(Bahrain, Barcelona, Monza) across multiple strategies with 5 stochastic runs each.
"""

import numpy as np
import pytest

from run_simulation import get_default_strategies
from src.config import create_race_model, get_default_compounds
from src.montecarlo import simulate_monte_carlo
from src.simulation import simulate_race
from src.stochastic import StochasticParameters
from src.strategies import Stint, Strategy


@pytest.fixture
def compounds():
    return get_default_compounds()


@pytest.fixture
def stochastic_params():
    return StochasticParameters()


CIRCUIT_KEYS = ["bahrain", "barcelona", "monza"]


@pytest.mark.parametrize("circuit_key", CIRCUIT_KEYS)
def test_all_three_gps_run_multiple_strategies(circuit_key, compounds):
    """Verify that all 3 GP presets simulate properly across all default strategies."""
    model = create_race_model(circuit_key)
    strategies = get_default_strategies(circuit_key, compounds)

    assert len(strategies) >= 3, f"Expected at least 3 strategies for {circuit_key}"

    for strat in strategies:
        result = simulate_race(strat, model)
        assert result.total_time > 0
        assert len(result.lap_records) == model.circuit.total_laps
        assert result.pure_racing_time > 0
        assert result.total_pit_loss >= 0


@pytest.mark.parametrize("circuit_key", CIRCUIT_KEYS)
def test_five_stochastic_runs_per_strategy(circuit_key, compounds, stochastic_params):
    """Verify that 5 stochastic runs per strategy produce consistent, bounded variance."""
    model = create_race_model(circuit_key)
    strategies = get_default_strategies(circuit_key, compounds)

    for strat in strategies:
        trial_times = []
        # Execute 5 distinct stochastic runs with different random seeds
        for seed in [11, 22, 33, 44, 55]:
            mc_res = simulate_monte_carlo(
                strat, model, stochastic_params, n_iterations=1, seed=seed
            )
            trial_times.append(float(mc_res.race_times[0]))

        assert len(trial_times) == 5
        mean_time = np.mean(trial_times)
        std_time = np.std(trial_times)

        # Baseline deterministic
        det_res = simulate_race(strat, model)

        # The 5-run mean should be within 1% of the deterministic total time
        assert abs(mean_time - det_res.total_time) / det_res.total_time < 0.01
        # Standard deviation across 5 runs should be bounded (typically 2s to 12s)
        assert 0.5 < std_time < 20.0


def test_real_world_winning_strategies_simulation(compounds):
    """Verify real-world 2024 winning strategies for Bahrain, Barcelona, and Monza."""
    # 1. Bahrain GP 2024: Verstappen S-H-S (57 laps)
    bahrain_model = create_race_model("bahrain")
    strat_bahrain = Strategy(
        [
            Stint(compounds["Soft"], 17),
            Stint(compounds["Hard"], 20),
            Stint(compounds["Soft"], 20),
        ],
        name="Verstappen 2024 S-H-S",
    )
    res_bahrain = simulate_race(strat_bahrain, bahrain_model)
    assert len(res_bahrain.lap_records) == 57
    # Verstappen real winning time was 5504.74s (~1h 31m 44s); simulation is around 5450s
    assert 5300 < res_bahrain.total_time < 5600

    # 2. Barcelona GP 2024: Verstappen S-M-S (66 laps)
    barcelona_model = create_race_model("barcelona")
    strat_barcelona = Strategy(
        [
            Stint(compounds["Soft"], 17),
            Stint(compounds["Medium"], 27),
            Stint(compounds["Soft"], 22),
        ],
        name="Verstappen 2024 S-M-S",
    )
    res_barcelona = simulate_race(strat_barcelona, barcelona_model)
    assert len(res_barcelona.lap_records) == 66
    # Real winning time was 5300.22s (~1h 28m 20s); simulation is around 5409s
    assert 5200 < res_barcelona.total_time < 5550

    # 3. Monza GP 2024: Leclerc 1-Stop M-H (53 laps) vs Piastri 2-Stop M-H-H
    monza_model = create_race_model("monza")
    strat_leclerc = Strategy(
        [
            Stint(compounds["Medium"], 15),
            Stint(compounds["Hard"], 38),
        ],
        name="Leclerc 2024 1-Stop M-H",
    )
    strat_piastri = Strategy(
        [
            Stint(compounds["Medium"], 16),
            Stint(compounds["Hard"], 22),
            Stint(compounds["Hard"], 15),
        ],
        name="Piastri 2024 2-Stop M-H-H",
    )
    res_leclerc = simulate_race(strat_leclerc, monza_model)
    res_piastri = simulate_race(strat_piastri, monza_model)

    assert len(res_leclerc.lap_records) == 53
    assert len(res_piastri.lap_records) == 53
    # In real world, Leclerc's 1-stop beat Piastri's 2-stop.
    # Simulation must reflect 1-stop advantage at low-deg Monza!
    assert res_leclerc.total_time < res_piastri.total_time
