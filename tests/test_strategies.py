"""Tests for Stint and Strategy modeling and validation."""

import pytest
from src.strategies import Stint, Strategy
from src.tyres import TyreCompound


@pytest.fixture
def compounds() -> dict[str, TyreCompound]:
    return {
        "Soft": TyreCompound(name="Soft", base_delta=-0.6, alpha=0.1),
        "Medium": TyreCompound(name="Medium", base_delta=0.0, alpha=0.06),
        "Hard": TyreCompound(name="Hard", base_delta=0.7, alpha=0.03),
    }


def test_stint_and_strategy_properties(compounds: dict[str, TyreCompound]) -> None:
    stint1 = Stint(compound=compounds["Soft"], laps=15)
    stint2 = Stint(compound=compounds["Medium"], laps=25)
    stint3 = Stint(compound=compounds["Hard"], laps=17)

    strat = Strategy(stints=[stint1, stint2, stint3], name="2-Stop S-M-H")

    assert strat.total_laps == 57
    assert strat.num_stops == 2
    assert strat.pit_laps == [15, 40]
    assert strat.compounds_used == {"Soft", "Medium", "Hard"}


def test_stint_lookup(compounds: dict[str, TyreCompound]) -> None:
    strat = Strategy(
        stints=[
            Stint(compound=compounds["Soft"], laps=15),
            Stint(compound=compounds["Hard"], laps=42),
        ]
    )

    # Lap 1: Stint 0, tyre age 1
    idx, stint, age = strat.get_stint_for_lap(1)
    assert idx == 0
    assert stint.compound.name == "Soft"
    assert age == 1

    # Lap 15: Stint 0, tyre age 15
    idx, stint, age = strat.get_stint_for_lap(15)
    assert idx == 0
    assert age == 15

    # Lap 16: Stint 1, tyre age 1
    idx, stint, age = strat.get_stint_for_lap(16)
    assert idx == 1
    assert stint.compound.name == "Hard"
    assert age == 1

    # Lap 57: Stint 1, tyre age 42
    idx, stint, age = strat.get_stint_for_lap(57)
    assert idx == 1
    assert age == 42


def test_strategy_validation(compounds: dict[str, TyreCompound]) -> None:
    strat_single_comp = Strategy(
        stints=[
            Stint(compound=compounds["Medium"], laps=30),
            Stint(compound=compounds["Medium"], laps=27),
        ]
    )

    # Fails FIA two-compound rule
    is_valid, msg = strat_single_comp.validate(required_laps=57, enforce_two_compounds=True)
    assert not is_valid
    assert "FIA rule violation" in msg

    # Passes if two compounds not enforced
    is_valid, _ = strat_single_comp.validate(required_laps=57, enforce_two_compounds=False)
    assert is_valid

    # Fails lap count mismatch
    is_valid, msg = strat_single_comp.validate(required_laps=60)
    assert not is_valid
    assert "does not match race distance" in msg
