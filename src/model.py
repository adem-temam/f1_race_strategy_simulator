"""Core lap-time physics and synthesis model."""

from dataclasses import dataclass
from typing import Optional

from src.fuel import FuelModel
from src.pitstop import PitStopModel
from src.tyres import TyreCompound


@dataclass(frozen=True)
class CircuitTrafficConfig:
    """Traffic and Dirty Air configuration for a circuit.
    
    Attributes:
        field_spread_rate: Seconds the field spreads out per lap.
        dirty_air_time_penalty: Constant lap time penalty (seconds) when in dirty air/DRS trains.
        dirty_air_deg_multiplier: Multiplier for tyre aging when in dirty air.
        overtake_difficulty: Number of laps spent in traffic per second of pit-loss deficit.
    """
    field_spread_rate: float = 0.5
    dirty_air_time_penalty: float = 0.8
    dirty_air_deg_multiplier: float = 1.2
    overtake_difficulty: float = 0.5

@dataclass(frozen=True)
class CircuitConfig:
    """Represents circuit-specific characteristics.

    Attributes:
        name: Name of the circuit / Grand Prix (e.g. 'Sakhir - Bahrain GP').
        total_laps: Standard scheduled Grand Prix distance in laps.
        base_lap_time: Reference lap time (seconds) at zero fuel with baseline tyre.
        track_evolution_total: Total lap-time improvement (seconds) due to rubber laying down
                               over the duration of the race (defaults to 0.0 for pure synthetic models).
        tyre_degradation_multiplier: Circuit-specific abrasiveness and energy (1.0 = standard).
        traffic_config: Traffic parameters for this specific circuit.
    """

    name: str
    total_laps: int
    base_lap_time: float
    track_evolution_total: float = 0.0
    tyre_degradation_multiplier: float = 1.0
    traffic_config: CircuitTrafficConfig = CircuitTrafficConfig()
    elevation_change_m: float = 0.0
    circuit_type: str = "Standard"

    def __post_init__(self) -> None:
        if self.total_laps < 1:
            raise ValueError(f"total_laps must be >= 1, got {self.total_laps}")
        if self.base_lap_time <= 0:
            raise ValueError(f"base_lap_time must be positive, got {self.base_lap_time}")
        if self.track_evolution_total < 0:
            raise ValueError(f"track_evolution_total must be non-negative, got {self.track_evolution_total}")
        if self.tyre_degradation_multiplier <= 0:
            raise ValueError(f"tyre_degradation_multiplier must be positive, got {self.tyre_degradation_multiplier}")

    def track_evolution_at_lap(self, lap: int) -> float:
        """Calculate the grip improvement (in seconds) on lap `lap`."""
        if self.total_laps <= 1:
            return 0.0
        return self.track_evolution_total * ((lap - 1) / (self.total_laps - 1))


@dataclass(frozen=True)
class LapRecord:
    """Detailed mathematical breakdown of a single lap in the race.

    Attributes:
        lap_number: Lap index (1 to N).
        stint_index: 0-indexed stint number.
        compound: Name of the active tyre compound.
        tyre_age: Number of laps run on this tyre set (starts at 1).
        effective_tyre_age: Tyre age accounting for extra wear from dirty air.
        fuel_kg: Fuel mass remaining on board at start of lap (kg).
        base_time: Circuit baseline lap time (s).
        compound_delta: Inherent compound offset (s).
        tyre_degradation: Time lost to tyre degradation (s).
        fuel_penalty: Time lost to fuel mass (s).
        track_evolution: Time gained from track rubbering in (s).
        in_traffic: True if the lap was affected by dirty air/traffic.
        traffic_penalty: Time lost due to dirty air (s).
        pit_loss: Time lost if a pit stop occurred on this lap (s).
        lap_time: Pure lap time excluding pit loss (s).
        effective_lap_time: Total elapsed lap time including pit loss (s).
        cumulative_time: Total race time elapsed from start through end of this lap (s).
    """

    lap_number: int
    stint_index: int
    compound: str
    tyre_age: int
    effective_tyre_age: float
    fuel_kg: float
    base_time: float
    compound_delta: float
    tyre_degradation: float
    fuel_penalty: float
    track_evolution: float
    in_traffic: bool
    traffic_penalty: float
    pit_loss: float
    lap_time: float
    effective_lap_time: float
    cumulative_time: float


class RaceModel:
    """Unified physical model synthesizing car, tyre, fuel, circuit, and pit stop effects."""

    def __init__(
        self,
        circuit: CircuitConfig,
        fuel_model: Optional[FuelModel] = None,
        pitstop_model: Optional[PitStopModel] = None,
    ) -> None:
        self.circuit = circuit
        self.fuel_model = fuel_model or FuelModel()
        self.pitstop_model = pitstop_model or PitStopModel()

    def compute_lap(
        self,
        lap: int,
        stint_index: int,
        compound: TyreCompound,
        tyre_age: int,
        effective_tyre_age: float,
        is_pit_lap: bool,
        in_traffic: bool = False,
        previous_cumulative_time: float = 0.0,
        pit_condition: str = "normal",
    ) -> LapRecord:
        """Compute the deterministic physical breakdown of a single lap.

        Mathematical formula:
            T_lap = T_base + Delta_compound + D_tyre(a_eff, mult) + F_fuel(m_f) - E_track(n) + T_traffic
            T_effective = T_lap + (PitLoss if is_pit_lap else 0)

        Args:
            lap: Current 1-indexed lap number.
            stint_index: Current stint index.
            compound: Active tyre compound.
            tyre_age: Nominal number of laps completed on current tyre set.
            effective_tyre_age: Accelerated age due to traffic wear.
            is_pit_lap: True if a pit stop occurs at the end of this lap.
            in_traffic: True if the car is currently in dirty air.
            previous_cumulative_time: Total elapsed race time up to the start of this lap.
            pit_condition: Track state during pit stop ('normal', 'vsc', 'safety_car').

        Returns:
            LapRecord containing all individual physical terms.
        """
        base_time = self.circuit.base_lap_time
        compound_delta = compound.base_delta
        tyre_degradation = compound.degradation(
            effective_tyre_age, circuit_multiplier=self.circuit.tyre_degradation_multiplier
        )
        fuel_kg = self.fuel_model.fuel_at_lap(lap, self.circuit.total_laps)
        fuel_penalty = self.fuel_model.penalty_at_lap(lap, self.circuit.total_laps)
        track_evolution = self.circuit.track_evolution_at_lap(lap)
        
        traffic_penalty = self.circuit.traffic_config.dirty_air_time_penalty if in_traffic else 0.0

        # Pure on-track lap time
        lap_time = base_time + compound_delta + tyre_degradation + fuel_penalty - track_evolution + traffic_penalty

        # Pit stop delta
        pit_loss = self.pitstop_model.effective_loss(pit_condition) if is_pit_lap else 0.0
        effective_lap_time = lap_time + pit_loss
        cumulative_time = previous_cumulative_time + effective_lap_time

        return LapRecord(
            lap_number=lap,
            stint_index=stint_index,
            compound=compound.name,
            tyre_age=tyre_age,
            effective_tyre_age=effective_tyre_age,
            fuel_kg=fuel_kg,
            base_time=base_time,
            compound_delta=compound_delta,
            tyre_degradation=tyre_degradation,
            fuel_penalty=fuel_penalty,
            track_evolution=track_evolution,
            in_traffic=in_traffic,
            traffic_penalty=traffic_penalty,
            pit_loss=pit_loss,
            lap_time=lap_time,
            effective_lap_time=effective_lap_time,
            cumulative_time=cumulative_time,
        )
