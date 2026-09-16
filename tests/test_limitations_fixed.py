"""Tests verifying fixes for key limitations and discrepancies:
1. Dynamic circuit-scaled tyre cliffs.
2. Official Pirelli C1-C5 compound hierarchy & circuit nominations.
3. VSC and Safety Car pit loss delta mechanics.
4. 2026 Technical Regulations era parameters.
"""

import pytest

from src.config import create_race_model, get_circuit_compounds, get_default_compounds
from src.pitstop import PitStopModel
from src.simulation import simulate_race
from src.strategies import Stint, Strategy
from src.tyres import PIRELLI_COMPOUNDS, TyreCompound


def test_circuit_multiplier_accelerates_cliff():
    """Verify that a higher circuit degradation multiplier accelerates cliff onset."""
    compound = TyreCompound(
        name="TestComp",
        base_delta=0.0,
        alpha=0.05,
        beta=0.001,
        cliff_lap=20,
        cliff_coefficient=0.1,
    )

    # Lap 18 at standard multiplier (1.0) -> before cliff (20)
    deg_std = compound.degradation(18, circuit_multiplier=1.0)
    # At multiplier 1.3 (Bahrain), effective cliff is 20 / 1.3 = 15.38 laps!
    # So lap 18 is well past the cliff
    deg_bahrain = compound.degradation(18, circuit_multiplier=1.3)

    assert deg_bahrain > deg_std * 1.3  # Extra penalty from accelerated cliff


def test_bahrain_one_stop_penalized_by_cliff():
    """Verify that Bahrain 1-stop is penalized by the accelerated cliff."""
    bahrain_model = create_race_model("bahrain")
    compounds = get_circuit_compounds("bahrain")

    strat_1stop = Strategy(
        [Stint(compounds["Medium"], 26), Stint(compounds["Hard"], 31)],
        name="1-Stop (M26-H31)",
    )
    strat_2stop = Strategy(
        [
            Stint(compounds["Soft"], 15),
            Stint(compounds["Medium"], 21),
            Stint(compounds["Soft"], 21),
        ],
        name="2-Stop (S-M-S)",
    )

    res_1stop = simulate_race(strat_1stop, bahrain_model)
    res_2stop = simulate_race(strat_2stop, bahrain_model)

    # 1-stop should be significantly slower than 2-stop due to tyre cliff
    assert res_1stop.total_time > res_2stop.total_time


def test_pirelli_compounds_hierarchy():
    """Verify C1 to C5 pace delta and wear progression."""
    c_codes = ["C1", "C2", "C3", "C4", "C5"]
    for code in c_codes:
        assert code in PIRELLI_COMPOUNDS

    # Base delta should decrease (faster) from C1 to C5
    deltas = [PIRELLI_COMPOUNDS[c].base_delta for c in c_codes]
    assert deltas == sorted(deltas, reverse=True)

    # Alpha degradation should increase (wear faster) from C1 to C5
    alphas = [PIRELLI_COMPOUNDS[c].alpha for c in c_codes]
    assert alphas == sorted(alphas)

    # Cliff lap should decrease (reach cliff earlier) from C1 to C5
    cliffs = [PIRELLI_COMPOUNDS[c].cliff_lap for c in c_codes]
    assert cliffs == sorted(cliffs, reverse=True)


def test_circuit_nominated_compounds():
    """Verify that Bahrain, Barcelona, and Monza receive authentic Pirelli nominations."""
    bahrain_comps = get_circuit_compounds("bahrain")
    barcelona_comps = get_circuit_compounds("barcelona")
    monza_comps = get_circuit_compounds("monza")

    # Bahrain & Barcelona use C1 (Hard), C2 (Medium), C3 (Soft)
    assert bahrain_comps["Hard"].name == "C1"
    assert bahrain_comps["Medium"].name == "C2"
    assert bahrain_comps["Soft"].name == "C3"

    assert barcelona_comps["Hard"].name == "C1"
    assert barcelona_comps["Medium"].name == "C2"
    assert barcelona_comps["Soft"].name == "C3"

    # Monza uses C3 (Hard), C4 (Medium), C5 (Soft)
    assert monza_comps["Hard"].name == "C3"
    assert monza_comps["Medium"].name == "C4"
    assert monza_comps["Soft"].name == "C5"


def test_safety_car_pit_loss_reduction():
    """Verify that taking a pit stop under Safety Car or VSC reduces pit loss."""
    pitstop = PitStopModel(pit_loss=22.0, stationary_time=2.5, vsc_pit_loss=14.0, sc_pit_loss=11.5)

    assert pitstop.effective_loss("normal") == 22.0
    assert pitstop.effective_loss("vsc") == 14.0
    assert pitstop.effective_loss("safety_car") == 11.5
    assert pitstop.effective_loss("sc") == 11.5

    # Run a race with and without SC on pit lap
    model = create_race_model("bahrain")
    compounds = get_default_compounds()
    strat = Strategy(
        [Stint(compounds["Medium"], 26), Stint(compounds["Hard"], 31)],
        name="1-Stop",
    )

    normal_res = simulate_race(strat, model)
    # Pit occurs at lap 26; simulate SC on lap 26
    sc_res = simulate_race(strat, model, safety_car_laps={26: "safety_car"})

    # SC pit stop should save exactly pit_loss - sc_pit_loss (22.5 - 11.5 = 11.0s) in direct pit loss
    expected_pit_savings = model.pitstop_model.pit_loss - model.pitstop_model.sc_pit_loss
    actual_pit_savings = normal_res.total_pit_loss - sc_res.total_pit_loss
    assert pytest.approx(actual_pit_savings, abs=0.01) == expected_pit_savings

    # Total race time savings is even greater because pitting under SC avoids dirty air traffic
    assert (normal_res.total_time - sc_res.total_time) >= expected_pit_savings


def test_2026_technical_regulations_model():
    """Verify 2026 technical regulations differences vs 2024."""
    model_2024 = create_race_model("bahrain", era="2024")
    model_2026 = create_race_model("bahrain", era="2026")

    # 2026 has lower initial fuel mass (75kg vs 105kg)
    assert model_2026.fuel_model.initial_fuel_kg == 75.0
    assert model_2024.fuel_model.initial_fuel_kg == 105.0

    # 2026 has lower fuel weight penalty per kg
    assert model_2026.fuel_model.fuel_penalty_per_kg < model_2024.fuel_model.fuel_penalty_per_kg

    # 2026 has active aero reducing dirty air penalty by 50%
    assert (
        model_2026.circuit.traffic_config.dirty_air_time_penalty
        < model_2024.circuit.traffic_config.dirty_air_time_penalty
    )
    assert (
        model_2026.circuit.traffic_config.dirty_air_time_penalty
        == model_2024.circuit.traffic_config.dirty_air_time_penalty * 0.50
    )
