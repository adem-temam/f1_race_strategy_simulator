"""Historical Formula 1 telemetry ingestion, caching, and data cleaning engine."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Optional
import urllib.request


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

CIRCUIT_SESSION_MAP: dict[str, dict] = {
    "bahrain": {"session_key": 9472, "name": "Sakhir (Bahrain GP)", "laps": 57},
    "barcelona": {"session_key": 9539, "name": "Circuit de Barcelona-Catalunya (Spanish GP)", "laps": 66},
    "monza": {"session_key": 9590, "name": "Autodromo Nazionale Monza (Italian GP)", "laps": 53},
}


@dataclass(frozen=True)
class HistoricalLap:
    """Individual lap record from timing feeds.

    Attributes:
        lap_number: Lap index (1 to N).
        driver_number: Driver car number (e.g. 1 for Verstappen, 16 for Leclerc).
        lap_duration: Lap duration in seconds.
        duration_sector_1: First sector duration in seconds.
        duration_sector_2: Second sector duration in seconds.
        duration_sector_3: Third sector duration in seconds.
        is_pit_out_lap: True if out-lap exiting pit lane.
        compound: Compound name (e.g. 'SOFT', 'MEDIUM', 'HARD').
        tyre_age: Effective tyre age in laps (including previous qualifying/race laps).
        stint_number: 1-indexed stint index.
        is_clean: True if lap passed representative racing filter.
    """

    lap_number: int
    driver_number: int
    lap_duration: float
    duration_sector_1: Optional[float] = None
    duration_sector_2: Optional[float] = None
    duration_sector_3: Optional[float] = None
    is_pit_out_lap: bool = False
    compound: Optional[str] = None
    tyre_age: Optional[int] = None
    stint_number: Optional[int] = None
    is_clean: bool = True


@dataclass(frozen=True)
class HistoricalStint:
    """Stint record representing a tyre sequence.

    Attributes:
        driver_number: Driver car number.
        stint_number: 1-indexed stint number.
        compound: Tyre compound name ('SOFT', 'MEDIUM', 'HARD').
        lap_start: Stint starting lap number.
        lap_end: Stint ending lap number.
        tyre_age_at_start: Laps already run on this tyre set prior to stint start.
    """

    driver_number: int
    stint_number: int
    compound: str
    lap_start: int
    lap_end: int
    tyre_age_at_start: int = 0

    @property
    def stint_length(self) -> int:
        """Total laps run during this stint."""
        return max(1, self.lap_end - self.lap_start + 1)


@dataclass(frozen=True)
class HistoricalPitStop:
    """Recorded pit lane transit.

    Attributes:
        driver_number: Driver car number.
        lap_number: Lap on which pit lane transit occurred.
        lane_duration: Total pit lane transit duration in seconds.
        stop_duration: Stationary service duration in seconds (if recorded).
    """

    driver_number: int
    lap_number: int
    lane_duration: float
    stop_duration: Optional[float] = None


@dataclass
class HistoricalRaceData:
    """Container for a complete Grand Prix weekend timing dataset."""

    circuit_key: str
    circuit_name: str
    year: int
    session_key: int
    total_laps: int
    drivers: dict[int, str]
    stints: list[HistoricalStint]
    pit_stops: list[HistoricalPitStop]
    laps: list[HistoricalLap]
    race_control: list[dict] = field(default_factory=list)

    def get_driver_stints(self, driver_number: int) -> list[HistoricalStint]:
        """Return chronological stints for a driver."""
        stints = [s for s in self.stints if s.driver_number == driver_number]
        return sorted(stints, key=lambda s: s.stint_number)

    def get_driver_laps(
        self, driver_number: int, clean_only: bool = False
    ) -> list[HistoricalLap]:
        """Return chronological laps for a driver."""
        laps = [
            l
            for l in self.laps
            if l.driver_number == driver_number and (not clean_only or l.is_clean)
        ]
        return sorted(laps, key=lambda l: l.lap_number)

    def get_clean_compound_laps(
        self, compound: Optional[str] = None, min_stint_laps: int = 4
    ) -> list[HistoricalLap]:
        """Return clean laps across all drivers, optionally filtered by compound."""
        target_compound = compound.upper() if compound else None
        clean_laps = []

        # Find stints meeting minimum length
        stint_keys = {
            (s.driver_number, s.stint_number)
            for s in self.stints
            if s.stint_length >= min_stint_laps
            and (not target_compound or s.compound.upper() == target_compound)
        }

        for lap in self.laps:
            if not lap.is_clean or lap.stint_number is None or not lap.compound:
                continue
            if (lap.driver_number, lap.stint_number) in stint_keys:
                if not target_compound or lap.compound.upper() == target_compound:
                    clean_laps.append(lap)

        return clean_laps

    def get_driver_strategy_summary(
        self, driver_number: int
    ) -> tuple[list[tuple[str, int]], int]:
        """Return ([(compound, stint_laps), ...], num_stops) for a driver."""
        stints = self.get_driver_stints(driver_number)
        seq = [(s.compound.capitalize(), s.stint_length) for s in stints]
        num_stops = max(0, len(stints) - 1)
        return seq, num_stops


def _fetch_openf1_endpoint(endpoint: str, retries: int = 3, backoff: float = 1.0) -> list[dict]:
    """Fetch JSON from OpenF1 API with exponential backoff for rate limits."""
    url = f"https://api.openf1.org/v1/{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "RaceStrategySim/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(backoff)
                continue
            raise
    return []


def filter_clean_laps(
    raw_laps: list[dict],
    stints: list[HistoricalStint],
    pit_stops: list[HistoricalPitStop],
    outlier_ratio: float = 1.07,
) -> list[HistoricalLap]:
    """Apply representative racing filters to raw lap timing records.

    Excludes:
    - Lap 1 (standing start physics).
    - Out-laps (pit lane exit).
    - In-laps (lap immediately preceding pit transit).
    - Neutralized laps (SC / VSC pace drop > 125% stint median).
    - Severe traffic / lockup / spin outliers (> 107% stint median).
    """
    # Build pit lap lookup: (driver_number, lap_number) -> in-lap
    in_laps_set = {(p.driver_number, p.lap_number) for p in pit_stops}

    # Group laps by (driver_number, stint_number)
    stint_map: dict[tuple[int, int], HistoricalStint] = {}
    for s in stints:
        for lap_idx in range(s.lap_start, s.lap_end + 1):
            stint_map[(s.driver_number, lap_idx)] = s

    # Pre-parse valid duration laps
    parsed_candidates: list[dict] = []
    durations_by_stint: dict[tuple[int, int], list[float]] = {}

    for r in raw_laps:
        d_num = r.get("driver_number")
        l_num = r.get("lap_number")
        dur = r.get("lap_duration")
        is_out = bool(r.get("is_pit_out_lap"))

        if d_num is None or l_num is None or dur is None or dur <= 0:
            continue

        stint = stint_map.get((d_num, l_num))
        stint_idx = stint.stint_number if stint else None
        compound = stint.compound if stint else None
        tyre_age = (
            (l_num - stint.lap_start + 1 + stint.tyre_age_at_start) if stint else l_num
        )

        entry = {
            "lap_number": l_num,
            "driver_number": d_num,
            "lap_duration": dur,
            "duration_sector_1": r.get("duration_sector_1"),
            "duration_sector_2": r.get("duration_sector_2"),
            "duration_sector_3": r.get("duration_sector_3"),
            "is_pit_out_lap": is_out,
            "compound": compound,
            "tyre_age": tyre_age,
            "stint_number": stint_idx,
            "is_in_lap": (d_num, l_num) in in_laps_set,
        }
        parsed_candidates.append(entry)

        if l_num > 1 and not is_out and not entry["is_in_lap"]:
            if stint_idx is not None:
                key = (d_num, stint_idx)
                durations_by_stint.setdefault(key, []).append(dur)

    # Compute median pace per stint
    stint_medians: dict[tuple[int, int], float] = {}
    for key, durs in durations_by_stint.items():
        if durs:
            sorted_durs = sorted(durs)
            mid = len(sorted_durs) // 2
            med = (
                sorted_durs[mid]
                if len(sorted_durs) % 2 != 0
                else (sorted_durs[mid - 1] + sorted_durs[mid]) / 2.0
            )
            stint_medians[key] = med

    # Classify clean laps
    clean_laps: list[HistoricalLap] = []
    for c in parsed_candidates:
        d_num = c["driver_number"]
        l_num = c["lap_number"]
        dur = c["lap_duration"]
        is_out = c["is_pit_out_lap"]
        is_in = c["is_in_lap"]
        stint_idx = c["stint_number"]

        is_clean = True
        # Exclude Lap 1
        if l_num == 1:
            is_clean = False
        # Exclude in-laps and out-laps
        elif is_out or is_in:
            is_clean = False
        # Exclude extreme outliers
        elif stint_idx is not None and (d_num, stint_idx) in stint_medians:
            med = stint_medians[(d_num, stint_idx)]
            if dur > med * outlier_ratio:
                is_clean = False
            # Also filter out unrealistically fast sensor glitches (< 50s)
            elif dur < 50.0:
                is_clean = False

        lap_obj = HistoricalLap(
            lap_number=l_num,
            driver_number=d_num,
            lap_duration=dur,
            duration_sector_1=c["duration_sector_1"],
            duration_sector_2=c["duration_sector_2"],
            duration_sector_3=c["duration_sector_3"],
            is_pit_out_lap=is_out,
            compound=c["compound"],
            tyre_age=c["tyre_age"],
            stint_number=stint_idx,
            is_clean=is_clean,
        )
        clean_laps.append(lap_obj)

    return clean_laps


def load_historical_data(
    circuit_key: str,
    year: int = 2024,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> HistoricalRaceData:
    """Load or fetch historical timing records for a Grand Prix.

    Args:
        circuit_key: Canonical circuit identifier ('bahrain', 'barcelona', 'monza').
        year: Championship season year (default: 2024).
        cache_dir: Custom directory for raw JSON caching.
        force_download: If True, bypass local cache and query OpenF1 API.

    Returns:
        HistoricalRaceData object populated with clean laps, stints, and pit stops.
    """
    key = circuit_key.lower()
    if key not in CIRCUIT_SESSION_MAP:
        raise ValueError(
            f"Unknown circuit key '{circuit_key}'. Supported: {list(CIRCUIT_SESSION_MAP.keys())}"
        )

    info = CIRCUIT_SESSION_MAP[key]
    directory = Path(cache_dir) if cache_dir else DEFAULT_DATA_DIR
    cache_file = directory / f"{key}_{year}.json"

    raw_payload: Optional[dict] = None

    # 1. Try local cache
    if cache_file.exists() and not force_download:
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_payload = json.load(f)
    elif force_download or not cache_file.exists():
        # 2. Fetch from OpenF1 REST API
        sk = info["session_key"]
        stints_raw = _fetch_openf1_endpoint(f"stints?session_key={sk}")
        pit_raw = _fetch_openf1_endpoint(f"pit?session_key={sk}")
        laps_raw = _fetch_openf1_endpoint(f"laps?session_key={sk}")
        rc_raw = _fetch_openf1_endpoint(f"race_control?session_key={sk}")
        drivers_raw = _fetch_openf1_endpoint(f"drivers?session_key={sk}")

        driver_map = {
            d["driver_number"]: d.get("name_acronym", str(d["driver_number"]))
            for d in drivers_raw
        }

        raw_payload = {
            "circuit_key": key,
            "circuit_name": info["name"],
            "year": year,
            "session_key": sk,
            "total_laps": info["laps"],
            "drivers": driver_map,
            "stints": stints_raw,
            "pit_stops": pit_raw,
            "race_control": rc_raw,
            "laps": laps_raw,
        }

        # Write to cache
        os.makedirs(directory, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_payload, f)

    if not raw_payload:
        raise RuntimeError(f"Failed to load historical data for {key} {year}")

    # Parse stints
    stints: list[HistoricalStint] = []
    for s in raw_payload.get("stints", []):
        stints.append(
            HistoricalStint(
                driver_number=s["driver_number"],
                stint_number=s["stint_number"],
                compound=str(s["compound"]).upper(),
                lap_start=s["lap_start"],
                lap_end=s["lap_end"],
                tyre_age_at_start=s.get("tyre_age_at_start") or 0,
            )
        )

    # Parse pit stops
    pit_stops: list[HistoricalPitStop] = []
    for p in raw_payload.get("pit_stops", []):
        dur = p.get("lane_duration") or p.get("pit_duration")
        if dur and dur > 0:
            pit_stops.append(
                HistoricalPitStop(
                    driver_number=p["driver_number"],
                    lap_number=p["lap_number"],
                    lane_duration=float(dur),
                    stop_duration=(
                        float(p["stop_duration"]) if p.get("stop_duration") else None
                    ),
                )
            )

    # Parse & filter clean laps
    clean_laps = filter_clean_laps(raw_payload.get("laps", []), stints, pit_stops)

    # Driver dictionary
    drivers_dict = {
        int(k): v for k, v in raw_payload.get("drivers", {}).items()
    } if raw_payload.get("drivers") else {}

    return HistoricalRaceData(
        circuit_key=key,
        circuit_name=raw_payload.get("circuit_name", info["name"]),
        year=year,
        session_key=raw_payload.get("session_key", info["session_key"]),
        total_laps=raw_payload.get("total_laps", info["laps"]),
        drivers=drivers_dict,
        stints=stints,
        pit_stops=pit_stops,
        laps=clean_laps,
        race_control=raw_payload.get("race_control", []),
    )
