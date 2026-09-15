"""Monte Carlo simulation engine for stochastic race analysis."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.model import RaceModel
from src.simulation import RaceResult, simulate_race
from src.stochastic import StochasticParameters
from src.strategies import Strategy


@dataclass
class MonteCarloResult:
    """Encapsulates the statistical results of a Monte Carlo strategy simulation.

    Attributes:
        deterministic_result: The baseline Phase 1 RaceResult without noise.
        stochastic_params: The noise parameters used.
        n_iterations: Number of simulated races (M).
        race_times: NumPy array of shape (M,) containing total race time for each iteration.
    """

    deterministic_result: RaceResult
    stochastic_params: StochasticParameters
    n_iterations: int
    race_times: np.ndarray

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.race_times))

    @property
    def median_time(self) -> float:
        return float(np.median(self.race_times))

    @property
    def std_dev(self) -> float:
        return float(np.std(self.race_times))

    @property
    def p05_time(self) -> float:
        """5th percentile (Best-case scenario)."""
        return float(np.percentile(self.race_times, 5))

    @property
    def p95_time(self) -> float:
        """95th percentile (Worst-case scenario / Value at Risk)."""
        return float(np.percentile(self.race_times, 95))

    def format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        rem = seconds % 3600
        minutes = int(rem // 60)
        secs = rem % 60
        if hours > 0:
            return f"{hours}h {minutes:02d}m {secs:06.3f}s"
        return f"{minutes:02d}m {secs:06.3f}s"

    def summary(self) -> dict[str, Any]:
        """Return a dictionary of statistical summary metrics."""
        return {
            "strategy": self.deterministic_result.strategy.name or self.deterministic_result.strategy.description,
            "stops": self.deterministic_result.strategy.num_stops,
            "iterations": self.n_iterations,
            "mean": self.mean_time,
            "median": self.median_time,
            "std_dev": self.std_dev,
            "p05": self.p05_time,
            "p95": self.p95_time,
            "formatted_mean": self.format_time(self.mean_time),
            "formatted_p95": self.format_time(self.p95_time),
        }


def simulate_monte_carlo(
    strategy: Strategy,
    race_model: RaceModel,
    params: StochasticParameters,
    n_iterations: int = 10000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Execute a vectorized Monte Carlo race simulation.

    Args:
        strategy: The pit-stop strategy to simulate.
        race_model: The deterministic race physics model.
        params: The stochastic noise parameters.
        n_iterations: Number of parallel simulations to run (M).
        seed: Optional random seed for reproducibility.

    Returns:
        MonteCarloResult containing the simulated race times and statistics.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Obtain deterministic baseline
    det_result = simulate_race(strategy, race_model, validate=True)
    lap_records = det_result.lap_records

    n_laps = len(lap_records)
    n_stops = strategy.num_stops

    # 2. Extract deterministic components
    base_components = np.array([
        r.base_time + r.compound_delta + r.fuel_penalty - r.track_evolution
        for r in lap_records
    ])
    deg_components = np.array([r.tyre_degradation for r in lap_records])
    pit_losses = np.array([r.pit_loss for r in lap_records if r.pit_loss > 0])

    # 3. Generate Vectorized Noise
    # Lap pace noise: shape (M, N)
    lap_noise = np.random.normal(0.0, params.lap_time_std, size=(n_iterations, n_laps))
    
    # Tyre degradation noise (global multiplier per race): shape (M, 1)
    tyre_noise = np.random.normal(0.0, params.tyre_deg_std, size=(n_iterations, 1))
    
    # Pit stop delay noise (log-normal): shape (M, K) where K is num stops
    if n_stops > 0:
        pit_delay_noise = np.random.lognormal(
            mean=params.pit_stop_mu, sigma=params.pit_stop_sigma, size=(n_iterations, n_stops)
        )
    else:
        pit_delay_noise = np.zeros((n_iterations, 0))

    # 4. Vectorized Summation
    # Broadcasting base components and noise
    # (M, N) lap matrix
    simulated_laps = base_components + (deg_components * (1.0 + tyre_noise)) + lap_noise
    
    # Sum across laps (axis=1) -> shape (M,)
    total_on_track_time = np.sum(simulated_laps, axis=1)
    
    # Add pit stops (deterministic + delay noise) -> shape (M,)
    if n_stops > 0:
        total_pit_time = np.sum(pit_losses + pit_delay_noise, axis=1)
    else:
        total_pit_time = np.zeros(n_iterations)

    # Total race time
    race_times = total_on_track_time + total_pit_time

    return MonteCarloResult(
        deterministic_result=det_result,
        stochastic_params=params,
        n_iterations=n_iterations,
        race_times=race_times,
    )


def calculate_win_probability(res_a: MonteCarloResult, res_b: MonteCarloResult) -> float:
    """Calculate the probability that Strategy A beats Strategy B.

    Assumes res_a and res_b were run with the same number of iterations and ideally the same seed 
    to properly correlate track conditions, though independent distributions also work.

    Returns:
        Probability in range [0, 1].
    """
    if res_a.n_iterations != res_b.n_iterations:
        raise ValueError("Must compare results with the same number of iterations.")
    
    wins = np.sum(res_a.race_times < res_b.race_times)
    return float(wins / res_a.n_iterations)
