"""Tests for the Monte Carlo simulation engine."""

import numpy as np
import pytest

from src.config import create_race_model, get_default_compounds
from src.montecarlo import calculate_win_probability, compute_win_probability_matrix, simulate_monte_carlo
from src.stochastic import StochasticParameters
from src.strategies import Stint, Strategy


@pytest.fixture
def compounds():
    return get_default_compounds()


@pytest.fixture
def model():
    return create_race_model("bahrain")


def test_montecarlo_deterministic_convergence(compounds, model) -> None:
    """Test that with zero noise, Monte Carlo matches deterministic exactly."""
    strat = Strategy(
        stints=[
            Stint(compound=compounds["Medium"], laps=26),
            Stint(compound=compounds["Hard"], laps=31),
        ],
        name="1-Stop",
    )
    
    # Zero out all noise
    params = StochasticParameters(lap_time_std=0.0, pit_stop_sigma=0.0, tyre_deg_std=0.0)
    
    # Hack pit_stop_mu to be a huge negative number so log-normal approaches 0 delay
    # Or just test with pit_stop_sigma = 0 and mu = -100
    params = StochasticParameters(lap_time_std=0.0, pit_stop_mu=-100.0, pit_stop_sigma=0.0, tyre_deg_std=0.0)

    mc_result = simulate_monte_carlo(strat, model, params, n_iterations=100, seed=42)
    det_time = mc_result.deterministic_result.total_time

    assert mc_result.n_iterations == 100
    assert len(mc_result.race_times) == 100
    assert mc_result.mean_time == pytest.approx(det_time, abs=0.01)
    assert mc_result.std_dev == pytest.approx(0.0, abs=0.01)


def test_montecarlo_variance(compounds, model) -> None:
    """Test that lap variance correctly scales into total race variance."""
    strat = Strategy(
        stints=[
            Stint(compound=compounds["Medium"], laps=28),
            Stint(compound=compounds["Hard"], laps=29),
        ],
        name="1-Stop M-H",
    )
    
    # Pure lap noise, no pit stops, no tyre noise
    lap_std = 0.5
    params = StochasticParameters(lap_time_std=lap_std, pit_stop_sigma=0.0, tyre_deg_std=0.0)
    
    mc_result = simulate_monte_carlo(strat, model, params, n_iterations=10000, seed=42)
    
    # Variance of sum of N independent normals is N * sigma^2
    # Standard deviation is sqrt(N) * sigma
    expected_std = np.sqrt(57) * lap_std
    
    assert mc_result.std_dev == pytest.approx(expected_std, rel=0.05)


def test_win_probability(compounds, model) -> None:
    strat1 = Strategy(
        stints=[
            Stint(compound=compounds["Medium"], laps=26),
            Stint(compound=compounds["Hard"], laps=31),
        ],
        name="1-Stop",
    )
    
    # Create an artificially terrible strategy (Soft - Hard, but Soft goes way too long)
    strat2 = Strategy(
        stints=[
            Stint(compound=compounds["Soft"], laps=40),
            Stint(compound=compounds["Hard"], laps=17),
        ],
        name="Terrible 1-Stop",
    )
    
    params = StochasticParameters()
    
    res1 = simulate_monte_carlo(strat1, model, params, n_iterations=500, seed=42)
    res2 = simulate_monte_carlo(strat2, model, params, n_iterations=500, seed=42)
    
    # strat1 should easily beat the terrible strat2 which pushes softs to 40 laps
    p_win = calculate_win_probability(res1, res2)
    assert p_win == 1.0  # 100% win rate


def test_compute_win_probability_matrix(compounds, model) -> None:
    strat1 = Strategy(
        stints=[
            Stint(compound=compounds["Medium"], laps=26),
            Stint(compound=compounds["Hard"], laps=31),
        ],
        name="Strategy 1",
    )
    strat2 = Strategy(
        stints=[
            Stint(compound=compounds["Soft"], laps=15),
            Stint(compound=compounds["Medium"], laps=21),
            Stint(compound=compounds["Soft"], laps=21),
        ],
        name="Strategy 2",
    )
    strat3 = Strategy(
        stints=[
            Stint(compound=compounds["Soft"], laps=40),
            Stint(compound=compounds["Hard"], laps=17),
        ],
        name="Strategy 3",
    )

    params = StochasticParameters()
    res1 = simulate_monte_carlo(strat1, model, params, n_iterations=200, seed=42)
    res2 = simulate_monte_carlo(strat2, model, params, n_iterations=200, seed=42)
    res3 = simulate_monte_carlo(strat3, model, params, n_iterations=200, seed=42)

    matrix, names = compute_win_probability_matrix([res1, res2, res3])

    assert matrix.shape == (3, 3)
    assert names == ["Strategy 1", "Strategy 2", "Strategy 3"]
    # Check diagonal is 0.5
    for i in range(3):
        assert matrix[i, i] == pytest.approx(0.5)
    # Check complement probability P(i < j) + P(j < i) == 1.0
    for i in range(3):
        for j in range(3):
            if i != j:
                assert matrix[i, j] + matrix[j, i] == pytest.approx(1.0, abs=1e-5)

