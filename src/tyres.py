"""Tyre compound modeling and degradation functions."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TyreCompound:
    """Represents a Formula 1 tyre compound specification and its wear characteristics.

    Attributes:
        name: Compound identifier (e.g. 'Soft', 'Medium', 'Hard', 'C1', 'C2', 'C3').
        base_delta: Inherent pace offset in seconds relative to the baseline compound (Medium = 0.0s).
                    Negative values indicate faster compounds (e.g. -0.65s for Soft).
        alpha: Linear degradation rate in seconds per lap (alpha >= 0).
        beta: Quadratic degradation rate in seconds per lap squared (beta >= 0).
        cliff_lap: Tyre age (in laps) where severe thermal / structural degradation begins.
        cliff_coefficient: Quadratic penalty multiplier applied beyond cliff_lap.
    """

    name: str
    base_delta: float
    alpha: float
    beta: float = 0.0
    cliff_lap: Optional[int] = None
    cliff_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError(f"Linear degradation alpha must be non-negative, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"Quadratic degradation beta must be non-negative, got {self.beta}")
        if self.cliff_coefficient < 0:
            raise ValueError(f"cliff_coefficient must be non-negative, got {self.cliff_coefficient}")
        if self.cliff_lap is not None and self.cliff_lap < 1:
            raise ValueError(f"cliff_lap must be at least 1, got {self.cliff_lap}")

    def degradation(self, age: float, circuit_multiplier: float = 1.0) -> float:
        """Calculate the tyre degradation penalty in seconds at tyre age `age`.

        The degradation formula is:
            D(a) = (alpha * a + beta * a^2) * circuit_multiplier + cliff_penalty(a)
        where the onset of the performance cliff accelerates on higher wear circuits:
            a_cliff,eff = a_cliff / circuit_multiplier

        Args:
            age: Tyre age in laps (1 for a fresh tyre on its first lap).
            circuit_multiplier: Circuit-specific degradation and abrasiveness multiplier (> 0).

        Returns:
            Degradation time penalty in seconds.
        """
        if age < 1:
            raise ValueError(f"Tyre age must be >= 1, got {age}")
        if circuit_multiplier <= 0:
            raise ValueError(f"circuit_multiplier must be positive, got {circuit_multiplier}")

        deg = (self.alpha * age + self.beta * (age**2)) * circuit_multiplier
        if self.cliff_lap is not None:
            effective_cliff = self.cliff_lap / circuit_multiplier
            if age > effective_cliff:
                deg += self.cliff_coefficient * circuit_multiplier * ((age - effective_cliff) ** 2)
        return deg

    def pace_penalty(self, age: float, circuit_multiplier: float = 1.0) -> float:
        """Calculate the total tyre effect on lap time: inherent delta + degradation.

        Args:
            age: Tyre age in laps.
            circuit_multiplier: Circuit-specific degradation and abrasiveness multiplier.

        Returns:
            Combined delta in seconds: base_delta + degradation(age, circuit_multiplier).
        """
        return self.base_delta + self.degradation(age, circuit_multiplier=circuit_multiplier)


PIRELLI_COMPOUNDS: dict[str, TyreCompound] = {
    "C1": TyreCompound(
        name="C1",
        base_delta=0.90,
        alpha=0.030,
        beta=0.0003,
        cliff_lap=42,
        cliff_coefficient=0.030,
    ),
    "C2": TyreCompound(
        name="C2",
        base_delta=0.45,
        alpha=0.045,
        beta=0.0005,
        cliff_lap=34,
        cliff_coefficient=0.035,
    ),
    "C3": TyreCompound(
        name="C3",
        base_delta=0.00,
        alpha=0.065,
        beta=0.0008,
        cliff_lap=28,
        cliff_coefficient=0.040,
    ),
    "C4": TyreCompound(
        name="C4",
        base_delta=-0.50,
        alpha=0.095,
        beta=0.0012,
        cliff_lap=20,
        cliff_coefficient=0.045,
    ),
    "C5": TyreCompound(
        name="C5",
        base_delta=-0.90,
        alpha=0.140,
        beta=0.0020,
        cliff_lap=14,
        cliff_coefficient=0.050,
    ),
    "Intermediate": TyreCompound(
        name="Intermediate",
        base_delta=6.50,
        alpha=0.075,
        beta=0.0009,
        cliff_lap=30,
        cliff_coefficient=0.040,
    ),
    "Wet": TyreCompound(
        name="Wet",
        base_delta=14.00,
        alpha=0.055,
        beta=0.0006,
        cliff_lap=35,
        cliff_coefficient=0.035,
    ),
}

PIRELLI_COMPOUNDS["I"] = PIRELLI_COMPOUNDS["Intermediate"]
PIRELLI_COMPOUNDS["W"] = PIRELLI_COMPOUNDS["Wet"]

