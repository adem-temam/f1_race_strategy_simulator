"""Integration tests for Phase 6 Interactive Strategy Dashboard FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.dashboard.app import app

client = TestClient(app)


def test_get_circuits():
    """Verify GET /api/circuits returns preset Grand Prix configurations."""
    response = client.get("/api/circuits")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3
    circuit_ids = [c["id"] for c in data]
    assert "bahrain" in circuit_ids
    assert "barcelona" in circuit_ids
    assert "monza" in circuit_ids


def test_post_simulate():
    """Verify POST /api/simulate runs race simulation and returns telemetry."""
    payload = {
        "circuit": "bahrain",
        "strategies": ["S15-M21-S21", "M26-H31"],
        "deg_multiplier": 1.1,
        "pit_loss": 22.5,
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["circuit"] is not None
    assert len(data["results"]) == 2
    r0 = data["results"][0]
    assert "total_time" in r0
    assert "formatted_time" in r0
    assert len(r0["laps"]) == 57


def test_post_optimize():
    """Verify POST /api/optimize runs DP solver and returns leaderboard and pit windows."""
    payload = {
        "circuit": "bahrain",
        "max_stops": 2,
        "objective": "time",
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimal_strategy" in data
    assert "optimal_race_time" in data
    assert len(data["leaderboard"]) >= 2
    assert len(data["pit_windows"]) >= 1


def test_post_optimize_risk_objective():
    """Verify POST /api/optimize runs with objective='risk' without error."""
    payload = {
        "circuit": "bahrain",
        "max_stops": 2,
        "objective": "risk",
        "mc_iterations": 50,
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimal_strategy" in data
    assert len(data["leaderboard"]) >= 1


def test_post_optimize_risk_default_and_zero_mc():
    """Verify POST /api/optimize runs with objective='risk' even if mc_iterations is 0 or omitted."""
    payload = {
        "circuit": "bahrain",
        "max_stops": 2,
        "objective": "risk",
        "mc_iterations": 0,
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimal_strategy" in data


def test_post_monte_carlo():
    """Verify POST /api/montecarlo runs stochastic iterations and computes win matrix."""
    payload = {
        "circuit": "monza",
        "strategies": ["M24-H29", "S18-H35"],
        "iterations": 200,
        "lap_noise_std": 0.3,
        "pit_stop_sigma": 0.7,
        "tyre_deg_std": 0.04,
    }
    response = client.post("/api/montecarlo", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["distributions"]) == 2
    assert "win_matrix" in data
    d0 = data["distributions"][0]
    assert d0["mean_time"] > 0
    assert len(d0["hist_bins"]) == len(d0["hist_density"])


def test_post_sensitivity_1d():
    """Verify POST /api/sensitivity/1d returns parameter sweep curve and crossovers."""
    payload = {
        "circuit": "bahrain",
        "parameter": "deg_multiplier",
        "range_min": 0.8,
        "range_max": 1.6,
        "num_points": 10,
    }
    response = client.post("/api/sensitivity/1d", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["parameter_name"] == "deg_multiplier"
    assert len(data["parameter_values"]) == 10
    assert len(data["optimal_envelope"]) == 10


def test_post_sensitivity_2d():
    """Verify POST /api/sensitivity/2d returns 2D phase boundary grid."""
    payload = {
        "circuit": "bahrain",
        "x_min": 18.0,
        "x_max": 26.0,
        "y_min": 0.8,
        "y_max": 1.6,
        "grid_resolution": 6,
    }
    response = client.post("/api/sensitivity/2d", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["optimal_stops_grid"]) == 6
    assert data["nominal_stops"] in (1, 2, 3)


def test_get_scenarios_presets():
    """Verify GET /api/scenarios/presets returns full scenario catalogue."""
    response = client.get("/api/scenarios/presets")
    assert response.status_code == 200
    presets = response.json()
    assert len(presets) >= 10
    ids = [p["id"] for p in presets]
    assert "slow_stop" in ids
    assert "high_deg" in ids
    assert "safety_car_lap18" in ids


def test_post_scenarios_run():
    """Verify POST /api/scenarios/run evaluates counterfactual scenarios."""
    payload = {
        "circuit": "bahrain",
        "scenario_id": "slow_stop",
    }
    response = client.post("/api/scenarios/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "strategic_regret" in data
    assert "summary_insight" in data
    assert len(data["strategy_outcomes"]) >= 2


def test_get_historical_validation():
    """Verify GET /api/historical/{circuit} returns empirical regression & backtest."""
    response = client.get("/api/historical/bahrain")
    assert response.status_code == 200
    data = response.json()
    assert data["circuit_name"] is not None
    assert len(data["fitted_compounds"]) >= 2
    assert "physics_duration_error_pct" in data
    assert "actual_strategy" in data


def test_serve_index_html():
    """Verify root / serves HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
