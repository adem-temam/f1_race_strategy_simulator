"""Stochastic parameters for modeling race uncertainty."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StochasticParameters:
    """Parameters governing the Monte Carlo noise injection.

    Attributes:
        lap_time_std: Standard deviation of driver pace per lap (seconds).
        pit_stop_mu: Mean of the underlying normal distribution for log-normal pit delay.
        pit_stop_sigma: Standard deviation of the underlying normal distribution for log-normal pit delay.
        tyre_deg_std: Standard deviation of the tyre degradation multiplier (e.g., 0.10 for 10% variance).
    """

    lap_time_std: float = 0.4
    pit_stop_mu: float = -1.0     # exp(-1) ~ 0.36s median delay
    pit_stop_sigma: float = 0.8   # Heavy tail for botched pit stops
    tyre_deg_std: float = 0.05    # 5% uncertainty on wear rates

    def __post_init__(self) -> None:
        if self.lap_time_std < 0:
            raise ValueError("lap_time_std must be non-negative.")
        if self.pit_stop_sigma < 0:
            raise ValueError("pit_stop_sigma must be non-negative.")
        if self.tyre_deg_std < 0:
            raise ValueError("tyre_deg_std must be non-negative.")
