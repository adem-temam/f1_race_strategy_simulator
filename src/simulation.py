"""Deterministic race simulation engine and results reporting."""

from dataclasses import dataclass
from typing import Any
import pandas as pd

from src.model import LapRecord, RaceModel
from src.strategies import Strategy


@dataclass
class RaceResult:
    """Encapsulates the complete results and telemetry of a simulated race.

    Attributes:
        strategy: Strategy executed in the race.
        circuit_name: Name of the circuit simulated.
        total_time: Total race completion time in seconds.
        lap_records: Lap-by-lap breakdown records.
        pit_laps: List of lap numbers where pit stops occurred.
    """

    strategy: Strategy
    circuit_name: str
    total_time: float
    lap_records: list[LapRecord]
    pit_laps: list[int]

    @property
    def formatted_total_time(self) -> str:
        """Format total race time as HH:MM:SS.mmm or MM:SS.mmm."""
        total_seconds = self.total_time
        hours = int(total_seconds // 3600)
        rem = total_seconds % 3600
        minutes = int(rem // 60)
        seconds = rem % 60
        if hours > 0:
            return f"{hours}h {minutes:02d}m {seconds:06.3f}s"
        return f"{minutes:02d}m {seconds:06.3f}s"

    @property
    def average_lap_time(self) -> float:
        """Average effective lap time over all laps (including pit stops)."""
        if not self.lap_records:
            return 0.0
        return self.total_time / len(self.lap_records)

    @property
    def pure_racing_time(self) -> float:
        """Total time excluding pit stop losses."""
        return sum(rec.lap_time for rec in self.lap_records)

    @property
    def total_pit_loss(self) -> float:
        """Total time spent on pit stop losses."""
        return sum(rec.pit_loss for rec in self.lap_records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert lap-by-lap records into a pandas DataFrame."""
        records = [
            {
                "lap": r.lap_number,
                "stint": r.stint_index + 1,
                "compound": r.compound,
                "tyre_age": r.tyre_age,
                "fuel_kg": round(r.fuel_kg, 2),
                "base_time": round(r.base_time, 3),
                "compound_delta": round(r.compound_delta, 3),
                "tyre_deg": round(r.tyre_degradation, 3),
                "fuel_penalty": round(r.fuel_penalty, 3),
                "track_evo": round(r.track_evolution, 3),
                "pit_loss": round(r.pit_loss, 3),
                "lap_time": round(r.lap_time, 3),
                "effective_time": round(r.effective_lap_time, 3),
                "cumulative_time": round(r.cumulative_time, 3),
            }
            for r in self.lap_records
        ]
        return pd.DataFrame(records)

    def summary(self) -> dict[str, Any]:
        """Return a structured summary dictionary of the race outcome."""
        return {
            "strategy": self.strategy.name or self.strategy.description,
            "stints": self.strategy.description,
            "stops": self.strategy.num_stops,
            "pit_laps": self.pit_laps,
            "total_time_seconds": round(self.total_time, 3),
            "formatted_time": self.formatted_total_time,
            "pure_racing_time": round(self.pure_racing_time, 3),
            "total_pit_loss": round(self.total_pit_loss, 3),
            "avg_lap_time": round(self.average_lap_time, 3),
        }


def simulate_race(
    strategy: Strategy,
    model: RaceModel,
    validate: bool = True,
    enforce_two_compounds: bool = True,
) -> RaceResult:
    """Execute a deterministic race simulation for a given strategy and physical model.

    Args:
        strategy: Race pit-stop strategy to evaluate.
        model: Physical race model combining circuit, fuel, tyres, and pit stops.
        validate: Whether to validate strategy length and rules before running.
        enforce_two_compounds: Whether to require at least two distinct tyre compounds.

    Returns:
        RaceResult object containing full lap telemetry and performance metrics.

    Raises:
        ValueError: If validation fails.
    """
    if validate:
        is_valid, msg = strategy.validate(
            required_laps=model.circuit.total_laps,
            enforce_two_compounds=enforce_two_compounds,
        )
        if not is_valid:
            raise ValueError(f"Invalid strategy: {msg}")

    pit_lap_set = set(strategy.pit_laps)
    lap_records: list[LapRecord] = []
    cumulative_time = 0.0

    current_lap = 1
    for stint_idx, stint in enumerate(strategy.stints):
        for tyre_age in range(1, stint.laps + 1):
            is_pit = current_lap in pit_lap_set
            record = model.compute_lap(
                lap=current_lap,
                stint_index=stint_idx,
                compound=stint.compound,
                tyre_age=tyre_age,
                is_pit_lap=is_pit,
                previous_cumulative_time=cumulative_time,
            )
            lap_records.append(record)
            cumulative_time = record.cumulative_time
            current_lap += 1

    return RaceResult(
        strategy=strategy,
        circuit_name=model.circuit.name,
        total_time=cumulative_time,
        lap_records=lap_records,
        pit_laps=strategy.pit_laps,
    )
