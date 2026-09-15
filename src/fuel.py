"""Fuel load and fuel effect modeling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FuelModel:
    """Represents fuel consumption dynamics and the associated lap-time weight penalty.

    Attributes:
        initial_fuel_kg: Fuel mass on board at race start (lap 1) in kg.
        fuel_penalty_per_kg: Time penalty in seconds per kg of fuel (gamma_fuel).
                             Typically ~0.030 - 0.035 s/kg (~0.33s per 10 kg).
        reserve_fuel_kg: Minimum residual fuel remaining at checkered flag
                         (e.g., FIA technical inspection requirement).
    """

    initial_fuel_kg: float = 105.0
    fuel_penalty_per_kg: float = 0.033
    reserve_fuel_kg: float = 2.0

    def __post_init__(self) -> None:
        if self.initial_fuel_kg <= 0:
            raise ValueError(f"initial_fuel_kg must be positive, got {self.initial_fuel_kg}")
        if self.reserve_fuel_kg < 0:
            raise ValueError(f"reserve_fuel_kg cannot be negative, got {self.reserve_fuel_kg}")
        if self.reserve_fuel_kg >= self.initial_fuel_kg:
            raise ValueError("reserve_fuel_kg must be strictly less than initial_fuel_kg")
        if self.fuel_penalty_per_kg < 0:
            raise ValueError(f"fuel_penalty_per_kg must be non-negative, got {self.fuel_penalty_per_kg}")

    def fuel_burned_per_lap(self, total_laps: int) -> float:
        """Calculate the average fuel burned per lap over the race distance.

        Args:
            total_laps: Total race distance in laps.

        Returns:
            Burn rate in kg/lap.
        """
        if total_laps < 1:
            raise ValueError(f"total_laps must be >= 1, got {total_laps}")
        return (self.initial_fuel_kg - self.reserve_fuel_kg) / total_laps

    def fuel_at_lap(self, lap: int, total_laps: int) -> float:
        """Return the estimated fuel mass (kg) on board at the start of lap `lap`.

        Formula:
            m_f(n) = initial_fuel_kg - (n - 1) * burn_rate

        Args:
            lap: 1-indexed lap number (1 <= lap <= total_laps).
            total_laps: Total race distance in laps.

        Returns:
            Fuel mass in kg on board.
        """
        if lap < 1 or lap > total_laps:
            raise ValueError(f"lap must be between 1 and {total_laps}, got {lap}")
        burn_rate = self.fuel_burned_per_lap(total_laps)
        return self.initial_fuel_kg - (lap - 1) * burn_rate

    def penalty_at_lap(self, lap: int, total_laps: int) -> float:
        """Calculate the lap-time penalty (in seconds) caused by fuel weight at lap `lap`.

        Formula:
            F(m_f(n)) = gamma_fuel * m_f(n)

        Args:
            lap: 1-indexed lap number.
            total_laps: Total race distance in laps.

        Returns:
            Lap time penalty in seconds.
        """
        fuel_mass = self.fuel_at_lap(lap, total_laps)
        return fuel_mass * self.fuel_penalty_per_kg
