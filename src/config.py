"""Circuit and compound configuration presets."""

from src.fuel import FuelModel
from src.model import CircuitConfig, RaceModel, CircuitTrafficConfig
from src.pitstop import PitStopModel
from src.tyres import TyreCompound, PIRELLI_COMPOUNDS


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
        "tyre_degradation_multiplier": 1.3,
        "compounds": {"Hard": "C1", "Medium": "C2", "Soft": "C3"},
        "traffic": {
            "field_spread_rate": 0.5,
            "dirty_air_time_penalty": 0.6,
            "dirty_air_deg_multiplier": 1.2,
            "overtake_difficulty": 0.6,
        },
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
        "tyre_degradation_multiplier": 1.5,
        "compounds": {"Hard": "C1", "Medium": "C2", "Soft": "C3"},
        "traffic": {
            "field_spread_rate": 0.4,
            "dirty_air_time_penalty": 0.8,
            "dirty_air_deg_multiplier": 1.15,
            "overtake_difficulty": 0.7,
        },
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
        "tyre_degradation_multiplier": 0.8,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "traffic": {
            "field_spread_rate": 0.6,
            "dirty_air_time_penalty": 1.5,
            "dirty_air_deg_multiplier": 1.1,
            "overtake_difficulty": 0.9,
        },
    },
}


def get_circuit_compounds(circuit_key: str) -> dict[str, TyreCompound]:
    """Return the nominated Pirelli compounds for a circuit (keyed by role and Cx)."""
    key = circuit_key.lower()
    if key not in CIRCUIT_PRESETS:
        raise ValueError(
            f"Unknown circuit '{circuit_key}'. Available: {list(CIRCUIT_PRESETS.keys())}"
        )
    nom = CIRCUIT_PRESETS[key].get("compounds")
    if not nom:
        return get_default_compounds()

    res = {}
    for role, c_code in nom.items():
        comp = PIRELLI_COMPOUNDS[c_code]
        res[role] = comp
        res[c_code] = comp
    return res


def _dict_to_traffic_config(data: dict, era: str = "2024") -> CircuitTrafficConfig:
    if era == "2026":
        # 2026 Active aerodynamics (X-mode/Z-mode) significantly reduces dirty air penalty
        return CircuitTrafficConfig(
            field_spread_rate=data["field_spread_rate"],
            dirty_air_time_penalty=data["dirty_air_time_penalty"] * 0.50,
            dirty_air_deg_multiplier=1.08,
            overtake_difficulty=data["overtake_difficulty"] * 0.75,
        )
    return CircuitTrafficConfig(
        field_spread_rate=data["field_spread_rate"],
        dirty_air_time_penalty=data["dirty_air_time_penalty"],
        dirty_air_deg_multiplier=data["dirty_air_deg_multiplier"],
        overtake_difficulty=data["overtake_difficulty"],
    )


def get_circuit_config(circuit_key: str, era: str = "2024") -> CircuitConfig:
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
        tyre_degradation_multiplier=preset.get("tyre_degradation_multiplier", 1.0),
        traffic_config=_dict_to_traffic_config(preset["traffic"], era=era),
    )


def create_race_model(circuit_key: str = "bahrain", era: str = "2024") -> RaceModel:
    """Create a fully configured RaceModel for a given circuit preset and regulation era."""
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
        tyre_degradation_multiplier=preset.get("tyre_degradation_multiplier", 1.0),
        traffic_config=_dict_to_traffic_config(preset["traffic"], era=era),
    )

    if era == "2026":
        # 2026: 768 kg minimum weight (down from 798 kg) and ~75 kg fuel capacity (50% electric MGU-K)
        initial_fuel_kg = 75.0
        fuel_penalty_per_kg = preset["fuel_penalty_per_kg"] * 0.85
    else:
        initial_fuel_kg = preset["initial_fuel_kg"]
        fuel_penalty_per_kg = preset["fuel_penalty_per_kg"]

    fuel = FuelModel(
        initial_fuel_kg=initial_fuel_kg,
        fuel_penalty_per_kg=fuel_penalty_per_kg,
    )
    pitstop = PitStopModel(
        pit_loss=preset["pit_loss"],
        stationary_time=preset["stationary_time"],
    )
    return RaceModel(circuit=circuit, fuel_model=fuel, pitstop_model=pitstop)

