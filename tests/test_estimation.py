"""Unit tests for Empirical Parameter Estimation."""

import numpy as np
import pytest

from src.data_pipeline import HistoricalLap, load_historical_data
from src.estimation import (
    EmpiricalCircuitParameters,
    FuelCorrectionEstimator,
    TyreDegradationFitter,
    estimate_empirical_parameters,
)
from src.model import RaceModel


def test_fuel_correction_estimator():
    """Verify fuel mass burn and correction properties."""
    fce = FuelCorrectionEstimator(initial_fuel_kg=105.0, fuel_penalty_per_kg=0.033, total_laps=57)

    # Mass at lap 1 should be initial fuel
    assert fce.fuel_mass_at_lap(1) == 105.0
    # Mass should strictly decrease with lap number
    assert fce.fuel_mass_at_lap(30) < fce.fuel_mass_at_lap(10)
    assert fce.fuel_mass_at_lap(57) >= 5.0

    # Fuel-corrected lap time removes the fuel penalty
    raw_time = 95.0
    corr_time_lap1 = fce.correct_lap_time(raw_time, lap_number=1)
    corr_time_lap50 = fce.correct_lap_time(raw_time, lap_number=50)

    # Lap 1 has ~105kg * 0.033 = ~3.46s subtracted
    assert corr_time_lap1 < raw_time
    # Lap 50 has much less fuel subtracted, so it is relatively higher
    assert corr_time_lap50 > corr_time_lap1


def test_tyre_degradation_fitter_synthetic_recovery():
    """Verify that curve fitting recovers known synthetic degradation coefficients."""
    fitter = TyreDegradationFitter()

    # Generate synthetic clean laps for 2 drivers on 'SOFT'
    clean_laps = []
    np.random.seed(42)
    true_alpha = 0.08
    true_beta = 0.001

    for d_num in [1, 2]:
        base_dur = 90.0 if d_num == 1 else 91.5
        for age in range(1, 20):
            wear = true_alpha * age + true_beta * (age**2)
            noise = np.random.normal(0, 0.05)
            # Add fuel penalty to test end-to-end correction
            fuel_p = (105.0 - (age - 1) * 1.8) * 0.033
            dur = base_dur + wear + fuel_p + noise
            clean_laps.append(
                HistoricalLap(
                    lap_number=age,
                    driver_number=d_num,
                    lap_duration=dur,
                    compound="SOFT",
                    tyre_age=age,
                    stint_number=1,
                    is_clean=True,
                )
            )

    fit_res = fitter.fit_compound_degradation(clean_laps, "SOFT")
    assert fit_res.sample_count == 38
    assert fit_res.alpha > 0.01
    assert fit_res.beta >= 0.0
    assert fit_res.r_squared > 0.85


def test_tyre_degradation_non_negative_enforcement():
    """Verify that negative degradation slopes are clamped to >= 0."""
    fitter = TyreDegradationFitter()
    # Decreasing lap times (simulating noise or extreme driver pace improvement)
    clean_laps = [
        HistoricalLap(
            lap_number=age,
            driver_number=1,
            lap_duration=100.0 - 0.2 * age,
            compound="HARD",
            tyre_age=age,
            stint_number=1,
            is_clean=True,
        )
        for age in range(1, 15)
    ]
    res = fitter.fit_compound_degradation(clean_laps, "HARD")
    assert res.alpha >= 0.0
    assert res.beta >= 0.0


def test_estimate_empirical_parameters_bahrain():
    """Verify empirical calibration on Bahrain 2024 race data."""
    data = load_historical_data("bahrain")
    params = estimate_empirical_parameters(data)

    assert isinstance(params, EmpiricalCircuitParameters)
    assert params.circuit_key == "bahrain"
    assert "Hard" in params.compounds
    assert "Soft" in params.compounds
    assert params.compounds["Hard"].alpha > 0
    assert params.compounds["Soft"].alpha > 0
    assert params.pit_loss_mean > 20.0
    assert params.lap_noise_std > 0

    # Test conversion to simulator objects
    compounds_dict = params.to_tyre_compounds()
    assert "Hard" in compounds_dict
    assert "Soft" in compounds_dict
    assert compounds_dict["Hard"].alpha == params.compounds["Hard"].alpha

    model = params.to_race_model(data.total_laps)
    assert isinstance(model, RaceModel)
    assert model.circuit.total_laps == 57
    assert model.pitstop_model.pit_loss >= 20.0
