"""Tests for tyre compound and degradation modeling."""

import pytest
from src.tyres import TyreCompound


def test_linear_degradation() -> None:
    soft = TyreCompound(name="Soft", base_delta=-0.6, alpha=0.1, beta=0.0)
    assert soft.degradation(1) == pytest.approx(0.1)
    assert soft.degradation(10) == pytest.approx(1.0)


def test_quadratic_degradation() -> None:
    medium = TyreCompound(name="Medium", base_delta=0.0, alpha=0.05, beta=0.002)
    # At age 10: 0.05 * 10 + 0.002 * 100 = 0.5 + 0.2 = 0.7
    assert medium.degradation(10) == pytest.approx(0.7)


def test_cliff_degradation() -> None:
    compound = TyreCompound(
        name="Soft",
        base_delta=-0.6,
        alpha=0.1,
        beta=0.0,
        cliff_lap=15,
        cliff_coefficient=0.05,
    )
    # Before cliff (lap 15): 0.1 * 15 = 1.5
    assert compound.degradation(15) == pytest.approx(1.5)

    # After cliff (lap 17): 0.1 * 17 + 0.05 * (17 - 15)^2 = 1.7 + 0.05 * 4 = 1.9
    assert compound.degradation(17) == pytest.approx(1.9)


def test_pace_penalty() -> None:
    soft = TyreCompound(name="Soft", base_delta=-0.6, alpha=0.1)
    # On lap 1: -0.6 + 0.1 = -0.5
    assert soft.pace_penalty(1) == pytest.approx(-0.5)


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        TyreCompound(name="Soft", base_delta=-0.5, alpha=-0.1)

    with pytest.raises(ValueError, match="beta must be non-negative"):
        TyreCompound(name="Soft", base_delta=-0.5, alpha=0.1, beta=-0.01)

    with pytest.raises(ValueError, match="cliff_lap must be at least 1"):
        TyreCompound(name="Soft", base_delta=-0.5, alpha=0.1, cliff_lap=0)

    soft = TyreCompound(name="Soft", base_delta=-0.5, alpha=0.1)
    with pytest.raises(ValueError, match="Tyre age must be >= 1"):
        soft.degradation(0)
