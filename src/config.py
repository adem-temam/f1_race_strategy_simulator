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
        "default_strategies": ["S15-M21-S21", "M26-H31", "S14-H22-M21"],
        "traffic": {
            "field_spread_rate": 0.5,
            "dirty_air_time_penalty": 0.6,
            "dirty_air_deg_multiplier": 1.2,
            "overtake_difficulty": 0.6,
        },
    },
    "jeddah": {
        "name": "Jeddah Corniche (Saudi Arabian GP)",
        "total_laps": 50,
        "base_lap_time": 88.5,
        "track_evolution_total": 0.45,
        "pit_loss": 20.5,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 0.85,
        "compounds": {"Hard": "C2", "Medium": "C3", "Soft": "C4"},
        "default_strategies": ["M22-H28", "S16-H34", "S14-M18-M18"],
        "traffic": {
            "field_spread_rate": 0.55,
            "dirty_air_time_penalty": 0.30,
            "dirty_air_deg_multiplier": 1.06,
            "overtake_difficulty": 0.50,
        },
    },
    "melbourne": {
        "name": "Albert Park (Australian GP)",
        "total_laps": 58,
        "base_lap_time": 78.2,
        "track_evolution_total": 0.5,
        "pit_loss": 20.0,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 0.95,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["M24-H34", "S18-H40", "S16-M21-M21"],
        "traffic": {
            "field_spread_rate": 0.5,
            "dirty_air_time_penalty": 0.35,
            "dirty_air_deg_multiplier": 1.07,
            "overtake_difficulty": 0.55,
        },
    },
    "suzuka": {
        "name": "Suzuka International Racing Course (Japanese GP)",
        "total_laps": 53,
        "base_lap_time": 91.0,
        "track_evolution_total": 0.35,
        "pit_loss": 22.0,
        "stationary_time": 2.5,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.4,
        "compounds": {"Hard": "C1", "Medium": "C2", "Soft": "C3"},
        "default_strategies": ["S14-H20-H19", "M22-H31", "S15-M20-S18"],
        "traffic": {
            "field_spread_rate": 0.45,
            "dirty_air_time_penalty": 0.40,
            "dirty_air_deg_multiplier": 1.10,
            "overtake_difficulty": 0.65,
        },
    },
    "shanghai": {
        "name": "Shanghai International Circuit (Chinese GP)",
        "total_laps": 56,
        "base_lap_time": 95.0,
        "track_evolution_total": 0.4,
        "pit_loss": 23.0,
        "stationary_time": 2.5,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.25,
        "compounds": {"Hard": "C2", "Medium": "C3", "Soft": "C4"},
        "default_strategies": ["M24-H32", "S16-H22-M18", "S15-M21-S20"],
        "traffic": {
            "field_spread_rate": 0.48,
            "dirty_air_time_penalty": 0.38,
            "dirty_air_deg_multiplier": 1.09,
            "overtake_difficulty": 0.52,
        },
    },
    "miami": {
        "name": "Miami International Autodrome (Miami GP)",
        "total_laps": 57,
        "base_lap_time": 88.0,
        "track_evolution_total": 0.45,
        "pit_loss": 20.5,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.1,
        "compounds": {"Hard": "C2", "Medium": "C3", "Soft": "C4"},
        "default_strategies": ["M25-H32", "S16-H41", "S15-M21-S21"],
        "traffic": {
            "field_spread_rate": 0.5,
            "dirty_air_time_penalty": 0.36,
            "dirty_air_deg_multiplier": 1.08,
            "overtake_difficulty": 0.54,
        },
    },
    "imola": {
        "name": "Autodromo Enzo e Dino Ferrari (Emilia Romagna GP)",
        "total_laps": 63,
        "base_lap_time": 76.5,
        "track_evolution_total": 0.4,
        "pit_loss": 28.0,
        "stationary_time": 2.5,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.0,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["M28-H35", "S20-H43", "S18-M23-M22"],
        "traffic": {
            "field_spread_rate": 0.42,
            "dirty_air_time_penalty": 0.45,
            "dirty_air_deg_multiplier": 1.10,
            "overtake_difficulty": 0.75,
        },
    },
    "monaco": {
        "name": "Circuit de Monaco (Monaco GP)",
        "total_laps": 78,
        "base_lap_time": 72.8,
        "track_evolution_total": 0.6,
        "pit_loss": 21.0,
        "stationary_time": 2.4,
        "initial_fuel_kg": 70.0,
        "fuel_penalty_per_kg": 0.026,
        "tyre_degradation_multiplier": 0.45,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["S32-H46", "M38-H40", "S25-M27-M26"],
        "traffic": {
            "field_spread_rate": 0.35,
            "dirty_air_time_penalty": 0.55,
            "dirty_air_deg_multiplier": 1.05,
            "overtake_difficulty": 0.95,
        },
    },
    "montreal": {
        "name": "Circuit Gilles Villeneuve (Canadian GP)",
        "total_laps": 70,
        "base_lap_time": 72.2,
        "track_evolution_total": 0.55,
        "pit_loss": 18.5,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 0.9,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["M30-H40", "S22-H48", "S18-M26-M26"],
        "traffic": {
            "field_spread_rate": 0.52,
            "dirty_air_time_penalty": 0.34,
            "dirty_air_deg_multiplier": 1.07,
            "overtake_difficulty": 0.52,
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
        "default_strategies": ["S18-M24-S24", "S20-H46", "M28-H38"],
        "traffic": {
            "field_spread_rate": 0.4,
            "dirty_air_time_penalty": 0.8,
            "dirty_air_deg_multiplier": 1.15,
            "overtake_difficulty": 0.7,
        },
    },
    "redbullring": {
        "name": "Red Bull Ring (Austrian GP)",
        "total_laps": 71,
        "base_lap_time": 65.2,
        "track_evolution_total": 0.35,
        "pit_loss": 20.5,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.2,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["M28-H43", "S22-M25-M24", "S20-H51"],
        "traffic": {
            "field_spread_rate": 0.55,
            "dirty_air_time_penalty": 0.32,
            "dirty_air_deg_multiplier": 1.07,
            "overtake_difficulty": 0.48,
        },
    },
    "silverstone": {
        "name": "Silverstone Circuit (British GP)",
        "total_laps": 52,
        "base_lap_time": 86.5,
        "track_evolution_total": 0.38,
        "pit_loss": 20.0,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.35,
        "compounds": {"Hard": "C1", "Medium": "C2", "Soft": "C3"},
        "default_strategies": ["M22-H30", "S16-H36", "S14-M19-M19"],
        "traffic": {
            "field_spread_rate": 0.50,
            "dirty_air_time_penalty": 0.38,
            "dirty_air_deg_multiplier": 1.09,
            "overtake_difficulty": 0.52,
        },
    },
    "hungaroring": {
        "name": "Hungaroring (Hungarian GP)",
        "total_laps": 70,
        "base_lap_time": 75.8,
        "track_evolution_total": 0.5,
        "pit_loss": 21.5,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.3,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["M28-H42", "S20-M25-M25", "S22-H48"],
        "traffic": {
            "field_spread_rate": 0.40,
            "dirty_air_time_penalty": 0.48,
            "dirty_air_deg_multiplier": 1.10,
            "overtake_difficulty": 0.72,
        },
    },
    "spa": {
        "name": "Circuit de Spa-Francorchamps (Belgian GP)",
        "total_laps": 44,
        "base_lap_time": 103.5,
        "track_evolution_total": 0.4,
        "pit_loss": 19.5,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.15,
        "compounds": {"Hard": "C2", "Medium": "C3", "Soft": "C4"},
        "default_strategies": ["M18-H26", "S12-H32", "S11-M17-M16"],
        "traffic": {
            "field_spread_rate": 0.54,
            "dirty_air_time_penalty": 0.35,
            "dirty_air_deg_multiplier": 1.08,
            "overtake_difficulty": 0.48,
        },
    },
    "zandvoort": {
        "name": "Circuit Zandvoort (Dutch GP)",
        "total_laps": 72,
        "base_lap_time": 70.2,
        "track_evolution_total": 0.5,
        "pit_loss": 21.0,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.25,
        "compounds": {"Hard": "C1", "Medium": "C2", "Soft": "C3"},
        "default_strategies": ["S24-H48", "M30-H42", "S18-M27-M27"],
        "traffic": {
            "field_spread_rate": 0.42,
            "dirty_air_time_penalty": 0.45,
            "dirty_air_deg_multiplier": 1.10,
            "overtake_difficulty": 0.68,
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
        "default_strategies": ["M22-H31", "S15-H38", "S14-M20-S19"],
        "traffic": {
            "field_spread_rate": 0.6,
            "dirty_air_time_penalty": 1.5,
            "dirty_air_deg_multiplier": 1.1,
            "overtake_difficulty": 0.9,
        },
    },
    "baku": {
        "name": "Baku City Circuit (Azerbaijan GP)",
        "total_laps": 51,
        "base_lap_time": 101.5,
        "track_evolution_total": 0.55,
        "pit_loss": 21.0,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 0.85,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["M20-H31", "S14-H37", "S13-M19-M19"],
        "traffic": {
            "field_spread_rate": 0.52,
            "dirty_air_time_penalty": 0.36,
            "dirty_air_deg_multiplier": 1.07,
            "overtake_difficulty": 0.50,
        },
    },
    "singapore": {
        "name": "Marina Bay Street Circuit (Singapore GP)",
        "total_laps": 62,
        "base_lap_time": 94.5,
        "track_evolution_total": 0.6,
        "pit_loss": 28.5,
        "stationary_time": 2.5,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.15,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["M26-H36", "S18-H44", "S16-M23-M23"],
        "traffic": {
            "field_spread_rate": 0.38,
            "dirty_air_time_penalty": 0.52,
            "dirty_air_deg_multiplier": 1.10,
            "overtake_difficulty": 0.82,
        },
    },
    "austin": {
        "name": "Circuit of the Americas (United States GP)",
        "total_laps": 56,
        "base_lap_time": 95.0,
        "track_evolution_total": 0.45,
        "pit_loss": 21.5,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.35,
        "compounds": {"Hard": "C2", "Medium": "C3", "Soft": "C4"},
        "default_strategies": ["M22-H34", "S15-H22-M19", "S14-M21-S21"],
        "traffic": {
            "field_spread_rate": 0.48,
            "dirty_air_time_penalty": 0.38,
            "dirty_air_deg_multiplier": 1.09,
            "overtake_difficulty": 0.54,
        },
    },
    "mexico": {
        "name": "Autódromo Hermanos Rodríguez (Mexico City GP)",
        "total_laps": 71,
        "base_lap_time": 78.0,
        "track_evolution_total": 0.4,
        "pit_loss": 22.5,
        "stationary_time": 2.4,
        "initial_fuel_kg": 70.0,
        "fuel_penalty_per_kg": 0.026,
        "tyre_degradation_multiplier": 0.9,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["M30-H41", "S22-H49", "S18-M27-M26"],
        "traffic": {
            "field_spread_rate": 0.46,
            "dirty_air_time_penalty": 0.42,
            "dirty_air_deg_multiplier": 1.08,
            "overtake_difficulty": 0.62,
        },
    },
    "interlagos": {
        "name": "Autódromo José Carlos Pace (São Paulo GP)",
        "total_laps": 71,
        "base_lap_time": 70.2,
        "track_evolution_total": 0.45,
        "pit_loss": 21.0,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.3,
        "compounds": {"Hard": "C2", "Medium": "C3", "Soft": "C4"},
        "default_strategies": ["S20-M26-M25", "M28-H43", "S22-H49"],
        "traffic": {
            "field_spread_rate": 0.52,
            "dirty_air_time_penalty": 0.36,
            "dirty_air_deg_multiplier": 1.08,
            "overtake_difficulty": 0.50,
        },
    },
    "lasvegas": {
        "name": "Las Vegas Strip Circuit (Las Vegas GP)",
        "total_laps": 50,
        "base_lap_time": 92.5,
        "track_evolution_total": 0.6,
        "pit_loss": 21.0,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 0.8,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["M22-H28", "S15-H35", "S14-M18-M18"],
        "traffic": {
            "field_spread_rate": 0.56,
            "dirty_air_time_penalty": 0.32,
            "dirty_air_deg_multiplier": 1.06,
            "overtake_difficulty": 0.46,
        },
    },
    "qatar": {
        "name": "Lusail International Circuit (Qatar GP)",
        "total_laps": 57,
        "base_lap_time": 83.2,
        "track_evolution_total": 0.35,
        "pit_loss": 24.5,
        "stationary_time": 2.5,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.6,
        "compounds": {"Hard": "C1", "Medium": "C2", "Soft": "C3"},
        "default_strategies": ["M19-M19-H19", "S15-M21-M21", "S14-H22-M21"],
        "traffic": {
            "field_spread_rate": 0.46,
            "dirty_air_time_penalty": 0.40,
            "dirty_air_deg_multiplier": 1.10,
            "overtake_difficulty": 0.58,
        },
    },
    "abudhabi": {
        "name": "Yas Marina Circuit (Abu Dhabi GP)",
        "total_laps": 58,
        "base_lap_time": 85.0,
        "track_evolution_total": 0.45,
        "pit_loss": 22.0,
        "stationary_time": 2.4,
        "initial_fuel_kg": 75.0,
        "fuel_penalty_per_kg": 0.028,
        "tyre_degradation_multiplier": 1.05,
        "compounds": {"Hard": "C3", "Medium": "C4", "Soft": "C5"},
        "default_strategies": ["M24-H34", "S17-H41", "S15-M22-S21"],
        "traffic": {
            "field_spread_rate": 0.50,
            "dirty_air_time_penalty": 0.36,
            "dirty_air_deg_multiplier": 1.08,
            "overtake_difficulty": 0.54,
        },
    },
}

CIRCUIT_METADATA: dict[str, dict] = {
    "bahrain": {
        "elevation_change_m": 17.5,
        "circuit_type": "High-Degradation Desert Permanent (Turn 4 & 10 Compression)",
        "avg_speed_kmh": 210.0,
        "typical_race_duration": "90–92 min",
    },
    "jeddah": {
        "elevation_change_m": 12.0,
        "circuit_type": "Ultra-High-Speed Coastal Street",
        "avg_speed_kmh": 252.0,
        "typical_race_duration": "75–80 min",
    },
    "melbourne": {
        "elevation_change_m": 2.6,
        "circuit_type": "Semi-Permanent Lakeside Street",
        "avg_speed_kmh": 235.0,
        "typical_race_duration": "78–82 min",
    },
    "suzuka": {
        "elevation_change_m": 40.4,
        "circuit_type": "High-Downforce Undulating Figure-8",
        "avg_speed_kmh": 228.0,
        "typical_race_duration": "82–85 min",
    },
    "shanghai": {
        "elevation_change_m": 7.4,
        "circuit_type": "Front-Limited Long-Straight Permanent",
        "avg_speed_kmh": 205.0,
        "typical_race_duration": "90–93 min",
    },
    "miami": {
        "elevation_change_m": 5.0,
        "circuit_type": "Semi-Street Autodrome",
        "avg_speed_kmh": 220.0,
        "typical_race_duration": "85–88 min",
    },
    "imola": {
        "elevation_change_m": 34.0,
        "circuit_type": "Historic Undulating Parkland",
        "avg_speed_kmh": 230.0,
        "typical_race_duration": "83–86 min",
    },
    "monaco": {
        "elevation_change_m": 42.0,
        "circuit_type": "Ultra-High-Downforce Street (Beau Rivage Climb)",
        "avg_speed_kmh": 155.0,
        "typical_race_duration": "98–105 min (Slowest F1 Speed)",
    },
    "montreal": {
        "elevation_change_m": 5.2,
        "circuit_type": "Stop-and-Go Island Circuit",
        "avg_speed_kmh": 215.0,
        "typical_race_duration": "85–88 min",
    },
    "barcelona": {
        "elevation_change_m": 29.6,
        "circuit_type": "High-Downforce Aerodynamic Benchmark",
        "avg_speed_kmh": 218.0,
        "typical_race_duration": "88–92 min",
    },
    "redbullring": {
        "elevation_change_m": 63.5,
        "circuit_type": "Alpine High-Elevation Power (Turn 3 Climb)",
        "avg_speed_kmh": 240.0,
        "typical_race_duration": "80–83 min",
    },
    "silverstone": {
        "elevation_change_m": 11.3,
        "circuit_type": "Flat Former-Airfield High-Speed Flow",
        "avg_speed_kmh": 238.0,
        "typical_race_duration": "77–80 min",
    },
    "hungaroring": {
        "elevation_change_m": 34.7,
        "circuit_type": "Natural Amphitheater Low-Speed Technical",
        "avg_speed_kmh": 195.0,
        "typical_race_duration": "92–96 min",
    },
    "spa": {
        "elevation_change_m": 102.2,
        "circuit_type": "Ardennes Mountain (Eau Rouge 17% Climb)",
        "avg_speed_kmh": 235.0,
        "typical_race_duration": "77–80 min",
    },
    "zandvoort": {
        "elevation_change_m": 15.0,
        "circuit_type": "Coastal Dunes & 19° Banked Corners",
        "avg_speed_kmh": 218.0,
        "typical_race_duration": "88–92 min",
    },
    "monza": {
        "elevation_change_m": 12.8,
        "circuit_type": "Low-Drag Temple of Speed (Highest Speed)",
        "avg_speed_kmh": 255.0,
        "typical_race_duration": "72–75 min (Fastest F1 Grand Prix)",
    },
    "baku": {
        "elevation_change_m": 24.0,
        "circuit_type": "Long-Straight & Medieval Castle Street",
        "avg_speed_kmh": 210.0,
        "typical_race_duration": "88–92 min",
    },
    "singapore": {
        "elevation_change_m": 6.0,
        "circuit_type": "High-Humidity High-Deg Street (Max Stamina)",
        "avg_speed_kmh": 175.0,
        "typical_race_duration": "100–110 min (Longest F1 Grand Prix)",
    },
    "austin": {
        "elevation_change_m": 40.9,
        "circuit_type": "Undulating Modern Circuit (Turn 1 Climb)",
        "avg_speed_kmh": 210.0,
        "typical_race_duration": "90–94 min",
    },
    "mexico": {
        "elevation_change_m": 2.8,
        "circuit_type": "High-Altitude Low-Air-Density (2,285m)",
        "avg_speed_kmh": 205.0,
        "typical_race_duration": "93–96 min",
    },
    "interlagos": {
        "elevation_change_m": 43.0,
        "circuit_type": "Natural Bowl (Senna S Drop & Junção Climb)",
        "avg_speed_kmh": 220.0,
        "typical_race_duration": "85–88 min",
    },
    "lasvegas": {
        "elevation_change_m": 6.5,
        "circuit_type": "Cold-Night Low-Grip Street Strip",
        "avg_speed_kmh": 240.0,
        "typical_race_duration": "78–82 min",
    },
    "qatar": {
        "elevation_change_m": 7.0,
        "circuit_type": "Extreme Lateral Energy Moto-Style",
        "avg_speed_kmh": 230.0,
        "typical_race_duration": "82–85 min",
    },
    "abudhabi": {
        "elevation_change_m": 10.7,
        "circuit_type": "Modern Twilight Harbor Marina",
        "avg_speed_kmh": 222.0,
        "typical_race_duration": "84–87 min",
    },
}

# Attach metadata to all circuit presets
for _k, _meta in CIRCUIT_METADATA.items():
    if _k in CIRCUIT_PRESETS:
        CIRCUIT_PRESETS[_k].update(_meta)


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
        elevation_change_m=preset.get("elevation_change_m", 0.0),
        circuit_type=preset.get("circuit_type", "Standard"),
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
        elevation_change_m=preset.get("elevation_change_m", 0.0),
        circuit_type=preset.get("circuit_type", "Standard"),
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


