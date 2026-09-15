"""Tests for physical lap-time modeling and decomposition."""

import pytest
from src.fuel import FuelModel
from src.model import CircuitConfig, RaceModel
from src.pitstop import PitStopModel
from src.tyres import TyreCompound


def test_circuit_config_track_evolution() -> None:
    circuit = CircuitConfig(
        name="Test GP",
        total_laps=51,
        base_lap_time=90.0,
        track_evolution_total=0.5,
    )
    # Lap 1: 0.0s evolution
    assert circuit.track_evolution_at_lap(1) == pytest.approx(0.0)
    # Lap 26 (halfway): 0.5 * 25 / 50 = 0.25s
    assert circuit.track_evolution_at_lap(26) == pytest.approx(0.25)
    # Lap 51: full 0.5s
    assert circuit.track_evolution_at_lap(51) == pytest.approx(0.5)


def test_race_model_lap_computation() -> None:
    circuit = CircuitConfig(
        name="Test GP",
        total_laps=50,
        base_lap_time=90.0,
        track_evolution_total=0.0,
    )
    fuel = FuelModel(initial_fuel_kg=100.0, reserve_fuel_kg=0.0, fuel_penalty_per_kg=0.03)
    pitstop = PitStopModel(pit_loss=20.0, stationary_time=2.5)
    soft = TyreCompound(name="Soft", base_delta=-0.5, alpha=0.1, beta=0.0)

    model = RaceModel(circuit=circuit, fuel_model=fuel, pitstop_model=pitstop)

    # Lap 1 on fresh tyre (age 1), non-pit lap:
    # Fuel: 100 kg * 0.03 = 3.0s
    # Tyre: -0.5 + 0.1 * 1 = -0.4s
    # Base: 90.0s
    # Expected pure lap time: 90.0 - 0.4 + 3.0 = 92.6s
    rec = model.compute_lap(
        lap=1,
        stint_index=0,
        compound=soft,
        tyre_age=1,
        is_pit_lap=False,
        previous_cumulative_time=0.0,
    )

    assert rec.lap_number == 1
    assert rec.base_time == 90.0
    assert rec.compound_delta == -0.5
    assert rec.tyre_degradation == pytest.approx(0.1)
    assert rec.fuel_penalty == pytest.approx(3.0)
    assert rec.pit_loss == 0.0
    assert rec.lap_time == pytest.approx(92.6)
    assert rec.effective_lap_time == pytest.approx(92.6)
    assert rec.cumulative_time == pytest.approx(92.6)

    # Pit lap on lap 15 (tyre age 15):
    # Tyre deg: 0.1 * 15 = 1.5s
    # Fuel at lap 15: 100 - 14 * 2 = 72 kg -> penalty: 72 * 0.03 = 2.16s
    # Pure lap time: 90.0 - 0.5 + 1.5 + 2.16 = 93.16s
    # Effective time: 93.16 + 20.0 (pit loss) = 113.16s
    rec_pit = model.compute_lap(
        lap=15,
        stint_index=0,
        compound=soft,
        tyre_age=15,
        is_pit_lap=True,
        previous_cumulative_time=1300.0,
    )

    assert rec_pit.pit_loss == 20.0
    assert rec_pit.lap_time == pytest.approx(93.16)
    assert rec_pit.effective_lap_time == pytest.approx(113.16)
    assert rec_pit.cumulative_time == pytest.approx(1413.16)
