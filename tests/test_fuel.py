"""Tests for fuel consumption and weight penalty modeling."""

import pytest
from src.fuel import FuelModel


def test_fuel_consumption() -> None:
    fuel = FuelModel(initial_fuel_kg=102.0, reserve_fuel_kg=2.0, fuel_penalty_per_kg=0.03)
    # Burn rate over 50 laps: (102 - 2) / 50 = 2.0 kg/lap
    assert fuel.fuel_burned_per_lap(50) == pytest.approx(2.0)

    # Lap 1: full 102 kg
    assert fuel.fuel_at_lap(1, 50) == pytest.approx(102.0)
    # Lap 2: 102 - 2 = 100 kg
    assert fuel.fuel_at_lap(2, 50) == pytest.approx(100.0)
    # Lap 50: 102 - 49 * 2 = 4 kg
    assert fuel.fuel_at_lap(50, 50) == pytest.approx(4.0)


def test_fuel_penalty() -> None:
    fuel = FuelModel(initial_fuel_kg=100.0, reserve_fuel_kg=0.0, fuel_penalty_per_kg=0.035)
    # Lap 1 (100 kg): 100 * 0.035 = 3.5s
    assert fuel.penalty_at_lap(1, 50) == pytest.approx(3.5)


def test_fuel_validation() -> None:
    with pytest.raises(ValueError, match="reserve_fuel_kg must be strictly less than initial_fuel_kg"):
        FuelModel(initial_fuel_kg=50.0, reserve_fuel_kg=50.0)

    with pytest.raises(ValueError, match="initial_fuel_kg must be positive"):
        FuelModel(initial_fuel_kg=-10.0)

    fuel = FuelModel()
    with pytest.raises(ValueError, match="lap must be between 1 and 50"):
        fuel.fuel_at_lap(51, 50)
