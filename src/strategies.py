"""Stint and Strategy representation and validation."""

from dataclasses import dataclass
from typing import Optional

from src.tyres import TyreCompound


@dataclass(frozen=True)
class Stint:
    """Represents a continuous racing stint on a single set of tyres.

    Attributes:
        compound: The tyre compound used.
        laps: Number of completed laps in this stint (laps >= 1).
    """

    compound: TyreCompound
    laps: int

    def __post_init__(self) -> None:
        if self.laps < 1:
            raise ValueError(f"Stint length must be at least 1 lap, got {self.laps}")

    def __repr__(self) -> str:
        return f"{self.compound.name}({self.laps})"


@dataclass
class Strategy:
    """Represents an overall race pit-stop strategy consisting of sequential stints.

    Attributes:
        stints: Ordered list of Stint objects making up the race.
        name: Optional user-friendly name for this strategy (e.g. '1-Stop M-H').
    """

    stints: list[Stint]
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.stints:
            raise ValueError("Strategy must contain at least one stint.")

    @property
    def total_laps(self) -> int:
        """Total race distance covered by this strategy."""
        return sum(s.laps for s in self.stints)

    @property
    def num_stops(self) -> int:
        """Number of pit stops required (stints - 1)."""
        return max(0, len(self.stints) - 1)

    @property
    def pit_laps(self) -> list[int]:
        """Lap numbers at the end of which pit stops occur (1-indexed).

        Example:
            For stints of lengths [15, 25, 17], pit stops occur after lap 15 and lap 40.
            Returns [15, 40].
        """
        laps: list[int] = []
        cumulative = 0
        for stint in self.stints[:-1]:
            cumulative += stint.laps
            laps.append(cumulative)
        return laps

    @property
    def compounds_used(self) -> set[str]:
        """Set of unique compound names used in this strategy."""
        return {stint.compound.name for stint in self.stints}

    @property
    def description(self) -> str:
        """Human-readable sequence of stints, e.g. 'Soft (15) -> Medium (25) -> Hard (17)'."""
        return " -> ".join(f"{s.compound.name} ({s.laps}L)" for s in self.stints)

    def validate(self, required_laps: int, enforce_two_compounds: bool = True) -> tuple[bool, str]:
        """Validate if the strategy conforms to race length and sporting regulations.

        Args:
            required_laps: Expected total race laps for the circuit.
            enforce_two_compounds: Whether to enforce FIA rule requiring at least 2 distinct compounds.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if self.total_laps != required_laps:
            return False, f"Total strategy laps ({self.total_laps}) does not match race distance ({required_laps})."

        if enforce_two_compounds and len(self.compounds_used) < 2:
            return False, (
                f"FIA rule violation: Strategy uses only {len(self.compounds_used)} compound "
                f"({next(iter(self.compounds_used))}). At least 2 distinct compounds required for dry race."
            )

        return True, "Valid"

    def get_stint_for_lap(self, lap: int) -> tuple[int, Stint, int]:
        """Find the active stint and tyre age for a given race lap.

        Args:
            lap: 1-indexed lap number (1 <= lap <= total_laps).

        Returns:
            Tuple of (stint_index, active_stint, tyre_age).
        """
        if lap < 1 or lap > self.total_laps:
            raise ValueError(f"Lap {lap} is out of bounds for strategy with {self.total_laps} laps.")

        accumulated = 0
        for idx, stint in enumerate(self.stints):
            if accumulated < lap <= accumulated + stint.laps:
                tyre_age = lap - accumulated
                return idx, stint, tyre_age
            accumulated += stint.laps

        # Fallback (should not be reached given bounds check)
        raise RuntimeError(f"Could not find stint for lap {lap}")
