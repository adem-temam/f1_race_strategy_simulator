"""Pit stop time loss modeling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PitStopModel:
    """Represents pit stop execution time and net race time loss.

    Attributes:
        pit_loss: Net lap-time delta (seconds) incurred by executing a pit stop.
                  This represents (transit time at speed limit + stationary time - track speed equivalent).
                  Typically ~20.0s - 25.0s on most F1 circuits.
        stationary_time: Stationary wheel change duration in seconds (typically ~2.2s - 2.8s).
    """

    pit_loss: float = 22.0
    stationary_time: float = 2.5

    def __post_init__(self) -> None:
        if self.pit_loss < 0:
            raise ValueError(f"pit_loss cannot be negative, got {self.pit_loss}")
        if self.stationary_time < 0:
            raise ValueError(f"stationary_time cannot be negative, got {self.stationary_time}")
        if self.stationary_time > self.pit_loss:
            raise ValueError(
                f"stationary_time ({self.stationary_time}s) cannot exceed total pit_loss ({self.pit_loss}s)"
            )

    @property
    def pit_lane_transit_loss(self) -> float:
        """Net loss attributable strictly to driving through the pit lane at the speed limit."""
        return self.pit_loss - self.stationary_time
