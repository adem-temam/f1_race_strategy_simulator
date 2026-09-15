"""Tests for deterministic race simulation and results aggregation."""

import pytest
from src.config import create_race_model, get_default_compounds
from src.simulation import simulate_race
from src.strategies import Stint, Strategy


@pytest.fixture
def compounds():
    return get_default_compounds()


@pytest.fixture
def model():
    return create_race_model("bahrain")


def test_simulation_lap_conservation(compounds, model) -> None:
    # 57 laps: Medium (25) -> Hard (32)
    strat = Strategy(
        stints=[
            Stint(compound=compounds["Medium"], laps=25),
            Stint(compound=compounds["Hard"], laps=32),
        ],
        name="1-Stop Medium-Hard",
    )

    result = simulate_race(strat, model)

    assert len(result.lap_records) == 57
    assert result.pit_laps == [25]
    assert result.circuit_name == "Sakhir (Bahrain GP)"

    # Total time equals sum of effective lap times
    total_effective = sum(r.effective_lap_time for r in result.lap_records)
    assert result.total_time == pytest.approx(total_effective)

    # Pit stop loss matches exactly 1 stop
    assert result.total_pit_loss == pytest.approx(model.pitstop_model.pit_loss)


def test_simulation_dataframe_and_summary(compounds, model) -> None:
    strat = Strategy(
        stints=[
            Stint(compound=compounds["Soft"], laps=15),
            Stint(compound=compounds["Medium"], laps=21),
            Stint(compound=compounds["Soft"], laps=21),
        ],
        name="2-Stop S-M-S",
    )

    result = simulate_race(strat, model)
    df = result.to_dataframe()

    assert len(df) == 57
    assert "effective_time" in df.columns
    assert "fuel_kg" in df.columns
    assert "tyre_deg" in df.columns

    summary = result.summary()
    assert summary["stops"] == 2
    assert summary["pit_laps"] == [15, 36]
    assert summary["total_pit_loss"] == pytest.approx(2 * model.pitstop_model.pit_loss)


def test_strategy_comparison(compounds, model) -> None:
    # 1-stop M-H
    strat_1stop = Strategy(
        stints=[
            Stint(compound=compounds["Medium"], laps=26),
            Stint(compound=compounds["Hard"], laps=31),
        ],
        name="1-Stop M-H",
    )
    # 2-stop S-M-S
    strat_2stop = Strategy(
        stints=[
            Stint(compound=compounds["Soft"], laps=15),
            Stint(compound=compounds["Medium"], laps=21),
            Stint(compound=compounds["Soft"], laps=21),
        ],
        name="2-Stop S-M-S",
    )

    res_1 = simulate_race(strat_1stop, model)
    res_2 = simulate_race(strat_2stop, model)

    # Both should finish all 57 laps with realistic times (~5400s = 1h 30m)
    assert 5000.0 < res_1.total_time < 6000.0
    assert 5000.0 < res_2.total_time < 6000.0

    delta = res_1.total_time - res_2.total_time
    # Both are within plausible range of each other
    assert abs(delta) < 60.0
