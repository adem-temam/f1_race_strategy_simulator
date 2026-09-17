"""Historical race strategy backtesting, accuracy evaluation, and discrepancy diagnostics."""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

from src.data_pipeline import HistoricalRaceData, load_historical_data
from src.estimation import (
    EmpiricalCircuitParameters,
    estimate_empirical_parameters,
)
from src.model import RaceModel
from src.optimization import OptimizationObjective, StrategyOptimizer
from src.simulation import simulate_race
from src.strategies import Stint, Strategy
from src.tyres import TyreCompound


@dataclass(frozen=True)
class ValidationMetrics:
    """Quantitative validation score measuring simulator accuracy against real Grand Prix outcomes."""

    circuit_key: str
    circuit_name: str
    driver_name: str
    driver_number: int
    actual_strategy_desc: str
    actual_stops: int
    optimal_strategy_desc: str
    optimal_stops: int
    stop_count_matches: bool
    stint_length_mae: float
    clean_lap_rmse: float
    pit_windows_covered: int
    total_pit_windows: int
    pit_window_hit_rate: float
    predicted_time: float
    actual_time: float
    time_delta: float
    relative_error_pct: float
    sim_actual_strategy_time: float = 0.0
    sim_actual_delta: float = 0.0
    sim_actual_error_pct: float = 0.0
    optimal_strategy_time: float = 0.0
    strategic_gain_delta: float = 0.0

    @property
    def formatted_predicted_time(self) -> str:
        hours = int(self.predicted_time // 3600)
        rem = self.predicted_time % 3600
        mins = int(rem // 60)
        secs = rem % 60
        return f"{hours}h {mins:02d}m {secs:06.3f}s" if hours > 0 else f"{mins:02d}m {secs:06.3f}s"

    @property
    def formatted_actual_time(self) -> str:
        hours = int(self.actual_time // 3600)
        rem = self.actual_time % 3600
        mins = int(rem // 60)
        secs = rem % 60
        return f"{hours}h {mins:02d}m {secs:06.3f}s" if hours > 0 else f"{mins:02d}m {secs:06.3f}s"

    @property
    def formatted_sim_actual_time(self) -> str:
        t = self.sim_actual_strategy_time if self.sim_actual_strategy_time > 0 else self.predicted_time
        hours = int(t // 3600)
        rem = t % 3600
        mins = int(rem // 60)
        secs = rem % 60
        return f"{hours}h {mins:02d}m {secs:06.3f}s" if hours > 0 else f"{mins:02d}m {secs:06.3f}s"


@dataclass(frozen=True)
class DiscrepancyDiagnostic:
    """Detailed causal analysis investigating why model recommendations diverge from real decisions."""

    circuit_key: str
    case_title: str
    observed_phenomenon: str
    theoretical_prediction: str
    root_cause_factors: list[str]
    engineering_solution: str


class DiscrepancyAnalyzer:
    """Automated diagnostic engine examining real-world Formula 1 tactical phenomena."""

    @staticmethod
    def analyze_circuit(circuit_key: str) -> DiscrepancyDiagnostic:
        """Return diagnostic case study for a circuit."""
        key = circuit_key.lower()

        if key == "monza":
            return DiscrepancyDiagnostic(
                circuit_key="monza",
                case_title="The Monza 2024 Leclerc Paradox: 1-Stop Graining Plateau & Track Position",
                observed_phenomenon=(
                    "Charles Leclerc won the Italian GP executing a 1-stop strategy (M15-H38), "
                    "defeating Oscar Piastri's 2-stop (M16-H22-H15) by 2.66s."
                ),
                theoretical_prediction=(
                    "Pure clean-air analytical models predict the 2-stop strategy is ~3.5s faster "
                    "due to quadratic tyre fatigue on a 38-lap Hard stint."
                ),
                root_cause_factors=[
                    "Non-Linear Tyre Graining & Thermal Recovery ('Second Life'): Around laps 8-16 on the Hard tyre, "
                    "Leclerc managed surface graining until it wore smooth, entering an ultra-low wear stabilization plateau.",
                    "Track Position & DRS Train Turbulence: Piastri exited his 2nd stop 18s behind Leclerc. "
                    "Although faster in clear air, closing within dirty air range added wake turbulence and front tyre sliding, "
                    "preventing an on-track overtake before race finish.",
                    "Pit Lane Delta Penalty: Monza's pit lane loss is long (~25.3s transit), imposing a massive hurdle for a second stop.",
                ],
                engineering_solution=(
                    "Incorporate non-monotonic graining recovery plateau D_grain(a) and dynamic track-position overtake difficulty."
                ),
            )

        elif key == "bahrain":
            return DiscrepancyDiagnostic(
                circuit_key="bahrain",
                case_title="The Bahrain 2024 Red Bull Strategy: FIA Weekend Tyre Allocation Limits",
                observed_phenomenon=(
                    "Max Verstappen and Sergio Perez secured a 1-2 finish running Soft-Hard-Soft (S17-H20-S20). "
                    "Ferrari (Carlos Sainz P3) ran Soft-Hard-Hard (S14-H21-H22)."
                ),
                theoretical_prediction=(
                    "On high-abrasion asphalt (abrasiveness multiplier 1.3), running two Hard stints (S-H-H) "
                    "is mathematically superior to a second Soft stint with alpha_soft = 0.11 s/lap."
                ),
                root_cause_factors=[
                    "FIA Article 30.2 Allocation Constraints: Teams receive only 13 dry tyre sets per weekend (typically 2 Hard, 3 Medium, 8 Soft). "
                    "Red Bull utilized one set of Hards in Friday practice, leaving exactly ONE fresh Hard set for the Grand Prix.",
                    "High Asphalt Abrasiveness Overheating Mediums: Sakhir's coarse granite asphalt severely degraded C2 Medium carcasses, "
                    "leading teams to avoid Mediums entirely during the race.",
                    "Clean Air Start Gap: Running fresh Softs on Stint 3 gave Verstappen rapid out-lap delta to maintain race lead in clear air.",
                ],
                engineering_solution=(
                    "Incorporate weekend tyre inventory constraints into candidate strategy enumeration."
                ),
            )

        elif key == "barcelona":
            return DiscrepancyDiagnostic(
                circuit_key="barcelona",
                case_title="The Barcelona 2024 Norris Clean-Air Paradox: Track Position vs Lap Pace",
                observed_phenomenon=(
                    "Max Verstappen (S17-M27-S22) won the Spanish GP ahead of Lando Norris (S23-M24-S19) by 2.2s."
                ),
                theoretical_prediction=(
                    "Norris's extended 23-lap opening stint gave him a substantial tyre offset in Stints 2 and 3, "
                    "yielding an analytical race time that was theoretically ~7.9s faster than Verstappen's."
                ),
                root_cause_factors=[
                    "Early Race Traffic Obstruction: Norris lost the lead into Turn 1 and was trapped behind George Russell "
                    "for 15 consecutive laps. Running in Russell's wake cost Norris ~0.8s/lap in aerodynamic downforce and accelerated tyre degradation.",
                    "Clean Air Advantage: Verstappen passed Russell early (Lap 3) and executed his strategy in completely clean air.",
                    "Undercut Power: Pitting early to undercut into clean air proved tactically superior to extending stints into dirty air.",
                ],
                engineering_solution=(
                    "Model Virtual Field Spread and DRS wake degradation penalties for cars following within 1.5s."
                ),
            )

        return DiscrepancyDiagnostic(
            circuit_key=key,
            case_title=f"General Telemetry Discrepancy Analysis: {key.capitalize()}",
            observed_phenomenon="Variations between clean-air simulations and real-world race telemetry.",
            theoretical_prediction="Theoretical optimum identifies minimum time in an unobstructed environment.",
            root_cause_factors=[
                "Driver tyre management and delta-to-target lifting.",
                "Tactical traffic and DRS train impediments.",
                "Pre-race tyre inventory constraints.",
            ],
            engineering_solution="Include traffic and allocation modifiers in strategy evaluation.",
        )


class RaceBacktester:
    """Executes end-to-end backtesting comparing model optimization against historical Grand Prix outcomes."""

    def __init__(
        self,
        race_data: HistoricalRaceData,
        empirical_params: Optional[EmpiricalCircuitParameters] = None,
    ) -> None:
        self.race_data = race_data
        self.params = empirical_params or estimate_empirical_parameters(race_data)
        self.compounds = self.params.to_tyre_compounds()
        self.model = self.params.to_race_model(race_data.total_laps)
        self.optimizer = StrategyOptimizer(self.model, self.compounds)

    def backtest_driver(
        self,
        driver_number: int,
        driver_name: Optional[str] = None,
        max_stops: int = 2,
    ) -> ValidationMetrics:
        """Run backtest comparing optimizer recommendations against a specific driver's race."""
        d_name = driver_name or self.race_data.drivers.get(driver_number, f"Driver #{driver_number}")
        actual_stints = self.race_data.get_driver_stints(driver_number)
        if not actual_stints:
            raise ValueError(f"No stint records found for driver {driver_number} in {self.race_data.circuit_key}")

        actual_laps_seq = [(s.compound.capitalize(), s.stint_length) for s in actual_stints]
        actual_stops = max(0, len(actual_stints) - 1)
        actual_desc = " -> ".join(f"{c} ({l}L)" for c, l in actual_laps_seq)

        # 1. Run Strategy Optimizer
        opt_res = self.optimizer.optimize(
            max_stops=max_stops,
            objective=OptimizationObjective.DETERMINISTIC_TIME,
        )
        opt_strat = opt_res.optimal_strategy
        opt_stops = opt_strat.num_stops
        opt_desc = opt_strat.description

        # 2. Stop count agreement
        stop_count_matches = (opt_stops == actual_stops)

        # 3. Stint length MAE
        # Compare stint lengths up to common count
        mae_errors = []
        for idx in range(min(len(opt_strat.stints), len(actual_stints))):
            mae_errors.append(abs(opt_strat.stints[idx].laps - actual_stints[idx].stint_length))
        stint_mae = float(np.mean(mae_errors)) if mae_errors else 0.0

        # 4. Clean Lap Pace RMSE
        # Reconstruct actual driver strategy in simulator to evaluate physics prediction error
        actual_sim_stints = []
        for s in actual_stints:
            comp_obj = self.compounds.get(s.compound.capitalize()) or self.compounds.get("Medium") or list(self.compounds.values())[0]
            actual_sim_stints.append(Stint(compound=comp_obj, laps=s.stint_length))

        sim_actual_res = simulate_race(Strategy(actual_sim_stints, name=f"Actual {d_name}"), self.model)
        sim_lap_map = {rec.lap_number: rec.lap_time for rec in sim_actual_res.lap_records}

        driver_clean_laps = self.race_data.get_driver_laps(driver_number, clean_only=True)
        sq_errors = []
        actual_time_sum = sum(l.lap_duration for l in self.race_data.get_driver_laps(driver_number, clean_only=False))

        for l in driver_clean_laps:
            if l.lap_number in sim_lap_map:
                diff = sim_lap_map[l.lap_number] - l.lap_duration
                sq_errors.append(diff**2)

        lap_rmse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else 0.0

        # 5. Pit window coverage (with 1-lap tactical undercut/overcut margin)
        pit_stops = [p for p in self.race_data.pit_stops if p.driver_number == driver_number]
        pit_windows = opt_res.pit_windows
        covered_count = 0
        total_windows = len(pit_stops)

        for p_idx, p in enumerate(pit_stops):
            if p_idx < len(pit_windows):
                w = pit_windows[p_idx]
                # Include a 1-lap margin to reflect tactical undercut/overcut positioning
                if (w.window_open_lap - 1) <= p.lap_number <= (w.window_close_lap + 1):
                    covered_count += 1

        hit_rate = (covered_count / total_windows) if total_windows > 0 else 1.0

        # 6. Race duration comparison: Physics Fidelity vs Strategic Gain
        opt_time = opt_res.optimal_race_time
        sim_actual_time = sim_actual_res.total_time
        actual_time = actual_time_sum if actual_time_sum > 0 else sim_actual_time

        # Pure physics error: Simulated actual strategy vs recorded clock time
        sim_actual_delta = sim_actual_time - actual_time
        sim_actual_error = abs(sim_actual_delta) / actual_time * 100.0

        # Strategic advantage: Difference between optimal strategy and actual strategy
        strategic_gain = sim_actual_time - opt_time

        # Overall optimization delta vs real outcome
        opt_delta = opt_time - actual_time
        opt_rel_error = abs(opt_delta) / actual_time * 100.0

        return ValidationMetrics(
            circuit_key=self.race_data.circuit_key,
            circuit_name=self.race_data.circuit_name,
            driver_name=d_name,
            driver_number=driver_number,
            actual_strategy_desc=actual_desc,
            actual_stops=actual_stops,
            optimal_strategy_desc=opt_desc,
            optimal_stops=opt_stops,
            stop_count_matches=stop_count_matches,
            stint_length_mae=round(stint_mae, 2),
            clean_lap_rmse=round(lap_rmse, 3),
            pit_windows_covered=covered_count,
            total_pit_windows=total_windows,
            pit_window_hit_rate=round(hit_rate, 2),
            predicted_time=round(opt_time, 3),
            actual_time=round(actual_time, 3),
            time_delta=round(opt_delta, 3),
            relative_error_pct=round(opt_rel_error, 2),
            sim_actual_strategy_time=round(sim_actual_time, 3),
            sim_actual_delta=round(sim_actual_delta, 3),
            sim_actual_error_pct=round(sim_actual_error, 2),
            optimal_strategy_time=round(opt_time, 3),
            strategic_gain_delta=round(strategic_gain, 3),
        )
