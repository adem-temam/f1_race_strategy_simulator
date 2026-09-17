"""Unit tests for Strategy Backtesting, Metrics, and Discrepancy Analysis."""

import pytest

from src.data_pipeline import load_historical_data
from src.validation import (
    DiscrepancyAnalyzer,
    DiscrepancyDiagnostic,
    RaceBacktester,
    ValidationMetrics,
)


def test_race_backtester_bahrain():
    """Verify strategy backtest against Max Verstappen's winning race in Bahrain 2024."""
    data = load_historical_data("bahrain")
    backtester = RaceBacktester(data)

    metrics = backtester.backtest_driver(1, "Max Verstappen", max_stops=2)

    assert isinstance(metrics, ValidationMetrics)
    assert metrics.circuit_key == "bahrain"
    assert metrics.driver_name == "Max Verstappen"
    assert metrics.actual_stops == 2
    assert metrics.optimal_stops == 2
    assert metrics.stop_count_matches  # Model agrees on 2 stops!
    assert metrics.stint_length_mae < 6.0  # Stints within 6 laps of actual
    assert metrics.clean_lap_rmse < 1.0  # Pace error < 1s across 57 laps
    assert metrics.relative_error_pct < 2.0  # Race duration within 2% of actual


def test_race_backtester_monza():
    """Verify backtesting against Charles Leclerc's winning 1-stop race in Monza 2024."""
    data = load_historical_data("monza")
    backtester = RaceBacktester(data)

    metrics = backtester.backtest_driver(16, "Charles Leclerc", max_stops=2)

    assert metrics.circuit_key == "monza"
    assert metrics.driver_name == "Charles Leclerc"
    assert metrics.actual_stops == 1
    assert metrics.optimal_stops == 1
    assert metrics.stop_count_matches  # Model agrees on 1 stop!
    assert metrics.stint_length_mae <= 20.0  # Captures the 17-lap discrepancy from Leclerc's 38L Hard stint


def test_race_backtester_invalid_driver():
    """Verify ValueError is raised for non-existent driver."""
    data = load_historical_data("bahrain")
    backtester = RaceBacktester(data)

    with pytest.raises(ValueError, match="No stint records found"):
        backtester.backtest_driver(999, "Nonexistent Driver")


def test_discrepancy_analyzer_all_circuits():
    """Verify causal discrepancy diagnostics for all 3 circuits."""
    for c_key in ["monza", "bahrain", "barcelona"]:
        diag = DiscrepancyAnalyzer.analyze_circuit(c_key)
        assert isinstance(diag, DiscrepancyDiagnostic)
        assert diag.circuit_key == c_key
        assert len(diag.case_title) > 10
        assert len(diag.observed_phenomenon) > 10
        assert len(diag.theoretical_prediction) > 10
        assert len(diag.root_cause_factors) >= 3
        assert len(diag.engineering_solution) > 10
