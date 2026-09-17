"""Unit tests for Phase 5 Historical Data Ingestion and Telemetry Cleaning."""

import pytest
from src.data_pipeline import (
    HistoricalLap,
    HistoricalPitStop,
    HistoricalStint,
    filter_clean_laps,
    load_historical_data,
)


def test_load_all_benchmark_circuits_from_cache():
    """Verify all three benchmark circuits load successfully from local cache."""
    for c_key in ["bahrain", "barcelona", "monza"]:
        data = load_historical_data(c_key)
        assert data.circuit_key == c_key
        assert data.year == 2024
        assert data.total_laps > 50
        assert len(data.stints) >= 40
        assert len(data.pit_stops) >= 20
        assert len(data.laps) >= 900
        assert len(data.drivers) >= 15


def test_clean_lap_filtering_logic():
    """Verify that non-representative laps are correctly filtered out."""
    stints = [
        HistoricalStint(driver_number=1, stint_number=1, compound="SOFT", lap_start=1, lap_end=15),
        HistoricalStint(driver_number=1, stint_number=2, compound="HARD", lap_start=16, lap_end=30),
    ]
    pit_stops = [
        HistoricalPitStop(driver_number=1, lap_number=15, lane_duration=24.5)
    ]

    raw_laps = [
        # Lap 1: Standing start
        {"driver_number": 1, "lap_number": 1, "lap_duration": 105.0, "is_pit_out_lap": False},
        # Lap 2-14: Normal racing laps
        {"driver_number": 1, "lap_number": 2, "lap_duration": 94.0, "is_pit_out_lap": False},
        {"driver_number": 1, "lap_number": 3, "lap_duration": 94.2, "is_pit_out_lap": False},
        {"driver_number": 1, "lap_number": 4, "lap_duration": 94.3, "is_pit_out_lap": False},
        # Lap 5: Spin / traffic outlier (> 1.07 * 94.2 = 100.8s)
        {"driver_number": 1, "lap_number": 5, "lap_duration": 108.5, "is_pit_out_lap": False},
        # Lap 15: In-lap (pit stop)
        {"driver_number": 1, "lap_number": 15, "lap_duration": 98.0, "is_pit_out_lap": False},
        # Lap 16: Out-lap
        {"driver_number": 1, "lap_number": 16, "lap_duration": 115.0, "is_pit_out_lap": True},
        # Lap 17: Normal racing lap
        {"driver_number": 1, "lap_number": 17, "lap_duration": 93.5, "is_pit_out_lap": False},
    ]

    clean_laps = filter_clean_laps(raw_laps, stints, pit_stops)
    lap_map = {l.lap_number: l for l in clean_laps}

    # Lap 1 must be marked not clean (standing start)
    assert not lap_map[1].is_clean
    # Laps 2, 3, 4 must be clean
    assert lap_map[2].is_clean
    assert lap_map[3].is_clean
    assert lap_map[4].is_clean
    # Lap 5 is an outlier (>107% of median)
    assert not lap_map[5].is_clean
    # Lap 15 is in-lap
    assert not lap_map[15].is_clean
    # Lap 16 is out-lap
    assert not lap_map[16].is_clean
    # Lap 17 is clean
    assert lap_map[17].is_clean


def test_driver_strategy_summary():
    """Verify strategy extraction per driver."""
    data = load_historical_data("bahrain")
    # Driver 1 (Verstappen) ran Soft -> Hard -> Soft (2 stops)
    seq, num_stops = data.get_driver_strategy_summary(1)
    assert num_stops == 2
    assert len(seq) == 3
    assert seq[0][0].upper() == "SOFT"
    assert seq[1][0].upper() == "HARD"
    assert seq[2][0].upper() == "SOFT"


def test_unknown_circuit_raises():
    """Verify invalid circuit key raises ValueError."""
    with pytest.raises(ValueError, match="Unknown circuit key"):
        load_historical_data("silverstone_fake")
