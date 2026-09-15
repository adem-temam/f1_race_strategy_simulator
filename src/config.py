"""Circuit and compound configuration presets."""

from src.fuel import FuelModel
from src.model import CircuitConfig, RaceModel
from src.pitstop import PitStopModel
from src.tyres import TyreCompound


def get_default_compounds() -> dict[str, TyreCompound]:
    """Return standard dry tyre compounds with realistic baseline F1 parameters."""
    return {
        "Soft": TyreCompound(
            name="Soft",
            base_delta=-0.65,
            alpha=0.110,
            beta=0.0015,
            cliff_lap=18,
            cliff_coefficient=0.05,
        ),
        "Medium": TyreCompound(
            name="Medium",
            base_delta=0.00,
            alpha=0.065,
            beta=0.0008,
            cliff_lap=28,
            cliff_coefficient=0.04,
        ),
        "Hard": TyreCompound(
            name="Hard",
            base_delta=0.70,
            alpha=0.035,
            beta=0.0003,
            cliff_lap=40,
            cliff_coefficient=0.03,
        ),
    }


CIRCUIT_PRESETS: dict[str, dict] = {
    "bahrain": {
        "name": "Sakhir (Bahrain GP)",
        "total_laps": 57,
        "base_lap_time": 92.0,
        "track_evolution_total": 0.4,
        "pit_loss": 22.5,
        "stationary_time": 2.5,
        "initial_fuel_kg": 105.0,
        "fuel_penalty_per_kg": 0.033,
    },
    "barcelona": {
        "name": "Circuit de Barcelona-Catalunya (Spanish GP)",
        "total_laps": 66,
        "base_lap_time": 78.0,
        "track_evolution_total": 0.5,
        "pit_loss": 22.0,
        "stationary_time": 2.5,
        "initial_fuel_kg": 105.0,
        "fuel_penalty_per_kg": 0.032,
    },
    "monza": {
        "name": "Autodromo Nazionale Monza (Italian GP)",
        "total_laps": 53,
        "base_lap_time": 82.5,
        "track_evolution_total": 0.3,
        "pit_loss": 24.0,
        "stationary_time": 2.4,
        "initial_fuel_kg": 100.0,
        "fuel_penalty_per_kg": 0.030,
    },
}


def get_circuit_config(circuit_key: str) -> CircuitConfig:
    """Retrieve CircuitConfig for a known circuit key."""
    key = circuit_key.lower()
    if key not in CIRCUIT_PRESETS:
        raise ValueError(
            f"Unknown circuit '{circuit_key}'. Available: {list(CIRCUIT_PRESETS.keys())}"
        )
    preset = CIRCUIT_PRESETS[key]
    return CircuitConfig(
        name=preset["name"],
        total_laps=preset["total_laps"],
        base_lap_time=preset["base_lap_time"],
        track_evolution_total=preset["track_evolution_total"],
    )


def create_race_model(circuit_key: str = "bahrain") -> RaceModel:
    """Create a fully configured RaceModel for a given circuit preset."""
    key = circuit_key.lower()
    if key not in CIRCUIT_PRESETS:
        raise ValueError(
            f"Unknown circuit '{circuit_key}'. Available: {list(CIRCUIT_PRESETS.keys())}"
        )
    preset = CIRCUIT_PRESETS[key]

    circuit = CircuitConfig(
        name=preset["name"],
        total_laps=preset["total_laps"],
        base_lap_time=preset["base_lap_time"],
        track_evolution_total=preset["track_evolution_total"],
    )
    fuel = FuelModel(
        initial_fuel_kg=preset["initial_fuel_kg"],
        fuel_penalty_per_kg=preset["fuel_penalty_per_kg"],
    )
    pitstop = PitStopModel(
        pit_loss=preset["pit_loss"],
        stationary_time=preset["stationary_time"],
    )
    return RaceModel(circuit=circuit, fuel_model=fuel, pitstop_model=pitstop)
