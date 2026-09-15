"""Tests for pit stop loss modeling."""

import pytest
from src.pitstop import PitStopModel


def test_pit_loss_computation() -> None:
    pit = PitStopModel(pit_loss=22.5, stationary_time=2.5)
    assert pit.pit_loss == 22.5
    assert pit.stationary_time == 2.5
    assert pit.pit_lane_transit_loss == pytest.approx(20.0)


def test_pit_validation() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        PitStopModel(pit_loss=-5.0)

    with pytest.raises(ValueError, match="cannot exceed total pit_loss"):
        PitStopModel(pit_loss=20.0, stationary_time=25.0)
