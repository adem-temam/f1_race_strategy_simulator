"""Strategy Optimization Engine: Automated pit-stop discovery, Dynamic Programming,
combinatorial search, tactical pit windows, and multi-objective Pareto analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Optional

import numpy as np

from src.model import RaceModel
from src.montecarlo import simulate_monte_carlo
from src.simulation import simulate_race
from src.stochastic import StochasticParameters
from src.strategies import Stint, Strategy
from src.tyres import TyreCompound


class OptimizationObjective(str, Enum):
    """Supported strategic optimization objectives."""

    DETERMINISTIC_TIME = "time"
    EXPECTED_TIME = "expected"
    MIN_RISK_P95 = "risk"


@dataclass
class PitWindow:
    """Tactical pit window for a specific pit stop in a strategy.

    Attributes:
        pit_index: 1-indexed stop number (e.g. 1 for Stop 1).
        optimal_lap: Lap on which the stop occurs in the optimal strategy.
        window_open_lap: Earliest acceptable lap to pit with delta <= tolerance.
        window_close_lap: Latest acceptable lap to pit with delta <= tolerance.
        tolerance_seconds: Allowable total race time penalty (seconds).
        delta_open: Time delta (seconds) lost if pitting on window_open_lap.
        delta_close: Time delta (seconds) lost if pitting on window_close_lap.
    """

    pit_index: int
    optimal_lap: int
    window_open_lap: int
    window_close_lap: int
    tolerance_seconds: float
    delta_open: float = 0.0
    delta_close: float = 0.0

    @property
    def window_size(self) -> int:
        """Total width of the pit window in laps."""
        return self.window_close_lap - self.window_open_lap + 1

    def __repr__(self) -> str:
        return (
            f"Stop {self.pit_index}: Lap {self.optimal_lap} "
            f"(Window: Laps {self.window_open_lap}–{self.window_close_lap}, "
            f"Tol: +{self.tolerance_seconds:.1f}s)"
        )


@dataclass
class ParetoPoint:
    """A candidate strategy plotted on the Pace vs. Risk Pareto frontier."""

    strategy: Strategy
    expected_time: float
    p95_time: float
    std_dev: float
    risk_penalty: float  # p95 - expected_time

    @property
    def formatted_expected(self) -> str:
        return _format_seconds(self.expected_time)

    @property
    def formatted_p95(self) -> str:
        return _format_seconds(self.p95_time)


@dataclass
class OptimizationResult:
    """Encapsulates the complete results of strategy optimization.

    Attributes:
        circuit_name: Name of the circuit.
        total_laps: Total laps in the Grand Prix.
        objective: The objective function optimized.
        optimal_strategy: The globally optimal strategy.
        optimal_race_time: Total race time (s) of the optimal strategy.
        formatted_optimal_time: Human-readable optimal time string.
        ranked_strategies: List of (Strategy, race_time) tuples sorted best to worst.
        pit_windows: List of PitWindow objects for the optimal strategy.
        evaluations_count: Total candidate strategies evaluated.
        solve_time_seconds: Elapsed time in seconds to perform the optimization.
        pareto_frontier: List of non-dominated ParetoPoint strategies (if computed).
    """

    circuit_name: str
    total_laps: int
    objective: OptimizationObjective
    optimal_strategy: Strategy
    optimal_race_time: float
    formatted_optimal_time: str
    ranked_strategies: list[tuple[Strategy, float]] = field(default_factory=list)
    pit_windows: list[PitWindow] = field(default_factory=list)
    evaluations_count: int = 0
    solve_time_seconds: float = 0.0
    pareto_frontier: list[ParetoPoint] = field(default_factory=list)


def _format_seconds(seconds: float) -> str:
    hours = int(seconds // 3600)
    rem = seconds % 3600
    minutes = int(rem // 60)
    secs = rem % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:06.3f}s"
    return f"{minutes:02d}m {secs:06.3f}s"


def simulate_race_time_fast(strategy: Strategy, model: RaceModel) -> float:
    """High-throughput race simulation returning total time without dataclass allocation.

    Evaluates the exact same physical model as `simulate_race()` (tyre degradation,
    fuel burn penalty, track evolution, pit loss, and dirty air / traffic deficit),
    but avoids creating millions of LapRecord instances during combinatorial grid search.
    """
    circuit = model.circuit
    fuel_model = model.fuel_model
    total_laps = circuit.total_laps
    base_lap_time = circuit.base_lap_time
    deg_multiplier = circuit.tyre_degradation_multiplier
    pit_loss = model.pitstop_model.effective_loss("normal")

    field_spread = circuit.traffic_config.field_spread_rate
    overtake_diff = circuit.traffic_config.overtake_difficulty
    dirty_air_penalty = circuit.traffic_config.dirty_air_time_penalty
    dirty_air_deg_mult = circuit.traffic_config.dirty_air_deg_multiplier

    pit_laps = set(strategy.pit_laps)
    cumulative_time = 0.0
    current_lap = 1
    laps_in_traffic = 0

    for stint in strategy.stints:
        comp = stint.compound
        compound_delta = comp.base_delta
        effective_age = 0.0

        for _ in range(stint.laps):
            is_pit = current_lap in pit_laps
            in_traffic = laps_in_traffic > 0

            if in_traffic:
                laps_in_traffic -= 1
                effective_age += dirty_air_deg_mult
                traffic_pen = dirty_air_penalty
            else:
                effective_age += 1.0
                traffic_pen = 0.0

            deg = comp.degradation(effective_age, circuit_multiplier=deg_multiplier)
            fuel_pen = fuel_model.penalty_at_lap(current_lap, total_laps)
            track_evo = circuit.track_evolution_at_lap(current_lap)

            lap_time = base_lap_time + compound_delta + deg + fuel_pen - track_evo + traffic_pen
            cumulative_time += lap_time + (pit_loss if is_pit else 0.0)

            if is_pit:
                traffic_deficit = max(0.0, pit_loss - (current_lap * field_spread))
                laps_in_traffic = int(traffic_deficit * overtake_diff)

            current_lap += 1

    return cumulative_time


def compute_pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Filter candidate points to return strictly non-dominated Pareto-optimal strategies.

    A point A dominates point B if:
        expected_time(A) <= expected_time(B) and p95_time(A) <= p95_time(B)
        with at least one strict inequality.
    Only points not dominated by any other point are kept.
    """
    frontier: list[ParetoPoint] = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            if (
                other.expected_time <= candidate.expected_time
                and other.p95_time <= candidate.p95_time
                and (
                    other.expected_time < candidate.expected_time
                    or other.p95_time < candidate.p95_time
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    frontier.sort(key=lambda p: p.expected_time)
    return frontier


class DynamicProgrammingSolver:
    """Exact global optimizer using Bellman backward recursion on a DAG."""

    def __init__(self, model: RaceModel, compounds: dict[str, TyreCompound]):
        self.model = model
        # Use unique compounds sorted from softest to hardest
        seen = set()
        self.compounds: list[TyreCompound] = []
        for c in compounds.values():
            if c.name not in seen:
                seen.add(c.name)
                self.compounds.append(c)

        self.num_compounds = len(self.compounds)
        self.total_laps = model.circuit.total_laps
        self.pit_loss = model.pitstop_model.effective_loss("normal")

        # Determine reasonable maximum tyre age per compound
        self.max_tyre_age = min(
            self.total_laps,
            max(int(1.6 * c.cliff_lap / model.circuit.tyre_degradation_multiplier) for c in self.compounds),
        )

    def _precompute_lap_table(self) -> np.ndarray:
        """Precompute lap times: table[lap, compound_idx, tyre_age]."""
        table = np.zeros((self.total_laps + 1, self.num_compounds, self.max_tyre_age + 1))
        circuit = self.model.circuit
        fuel_model = self.model.fuel_model

        for lap in range(1, self.total_laps + 1):
            fuel_penalty = fuel_model.penalty_at_lap(lap, self.total_laps)
            track_evolution = circuit.track_evolution_at_lap(lap)

            for c_idx, comp in enumerate(self.compounds):
                base_time = circuit.base_lap_time + comp.base_delta
                for age in range(1, self.max_tyre_age + 1):
                    deg = comp.degradation(age, circuit_multiplier=circuit.tyre_degradation_multiplier)
                    table[lap, c_idx, age] = base_time + deg + fuel_penalty - track_evolution
        return table

    def solve(self, max_stops: int = 2) -> tuple[Strategy, float]:
        """Execute Bellman backward induction to find the global optimal Strategy.

        Args:
            max_stops: Maximum number of pit stops allowed (strictly enforced).

        Returns:
            Tuple of (optimal Strategy, total race time in seconds).
        """
        lap_table = self._precompute_lap_table()

        # V[c_idx, age, u, stops] holds minimum time from current lap to race end
        # u = 1 if >=2 distinct compounds have been used, 0 otherwise
        # stops in {0, ..., max_stops}
        INF = 1e9
        V = np.full((self.num_compounds, self.max_tyre_age + 1, 2, max_stops + 1), INF)
        policy_action = {}  # (lap, c, a, u, stops) -> ("STAY" or ("PIT", next_c))

        # Terminal conditions at lap N
        for c_idx in range(self.num_compounds):
            for age in range(1, self.max_tyre_age + 1):
                lap_t = lap_table[self.total_laps, c_idx, age]
                for stops in range(max_stops + 1):
                    # At lap N, only u=1 (FIA compliance) has finite cost, requiring at least 1 stop
                    if stops >= 1:
                        V[c_idx, age, 1, stops] = lap_t
                    V[c_idx, age, 0, stops] = INF

        # Backward recursion from N-1 down to 1
        for lap in range(self.total_laps - 1, 0, -1):
            V_next = np.full((self.num_compounds, self.max_tyre_age + 1, 2, max_stops + 1), INF)

            for c_idx in range(self.num_compounds):
                for age in range(1, self.max_tyre_age + 1):
                    lap_t = lap_table[lap, c_idx, age]

                    for u in (0, 1):
                        for stops in range(max_stops + 1):
                            best_val = INF
                            best_act = "STAY"

                            # Option 1: STAY
                            if age < self.max_tyre_age:
                                val_stay = lap_t + V[c_idx, age + 1, u, stops]
                                if val_stay < best_val:
                                    best_val = val_stay
                                    best_act = "STAY"

                            # Option 2: PIT to c_next (only if stops < max_stops)
                            if stops < max_stops:
                                for c_next in range(self.num_compounds):
                                    if c_next == c_idx:
                                        continue  # Distinct compound
                                    # Pitting to a different compound satisfies FIA rule (u=1)
                                    val_pit = lap_t + self.pit_loss + V[c_next, 1, 1, stops + 1]
                                    if val_pit < best_val:
                                        best_val = val_pit
                                        best_act = ("PIT", c_next)

                            V_next[c_idx, age, u, stops] = best_val
                            policy_action[(lap, c_idx, age, u, stops)] = best_act

            V = V_next

        # Find best starting compound at lap 1, age 1, u=0, stops=0
        best_total = INF
        best_start_c = 0
        for c_idx in range(self.num_compounds):
            if V[c_idx, 1, 0, 0] < best_total:
                best_total = V[c_idx, 1, 0, 0]
                best_start_c = c_idx

        # Forward reconstruction of stints
        stints: list[Stint] = []
        curr_c = best_start_c
        curr_age = 1
        curr_u = 0
        curr_stops = 0
        stint_start_lap = 1

        for lap in range(1, self.total_laps):
            act = policy_action.get((lap, curr_c, curr_age, curr_u, curr_stops), "STAY")
            if act == "STAY":
                curr_age += 1
            else:
                # act is ("PIT", next_c)
                stint_len = lap - stint_start_lap + 1
                stints.append(Stint(self.compounds[curr_c], stint_len))
                curr_c = act[1]
                curr_age = 1
                curr_u = 1
                curr_stops += 1
                stint_start_lap = lap + 1

        # Final stint
        final_len = self.total_laps - stint_start_lap + 1
        stints.append(Stint(self.compounds[curr_c], final_len))

        strategy = Strategy(stints, name=f"DP Optimal ({len(stints)-1}-Stop)")
        sim_res = simulate_race(strategy, self.model)
        return strategy, sim_res.total_time




class CombinatorialSearchSolver:
    """Exhaustive grid search across compound sequences and integer stint partitions."""

    def __init__(self, model: RaceModel, compounds: dict[str, TyreCompound]):
        self.model = model
        seen = set()
        self.compounds: list[TyreCompound] = []
        for c in compounds.values():
            if c.name not in seen:
                seen.add(c.name)
                self.compounds.append(c)

        self.total_laps = model.circuit.total_laps

    def _generate_partitions(
        self,
        remaining_laps: int,
        stints_remaining: int,
        compounds_seq: list[TyreCompound],
        current_stint_idx: int = 0,
        min_stint: int = 5,
        step: int = 1,
    ):
        """Recursively yield valid integer stint length partitions."""
        if stints_remaining == 1:
            comp = compounds_seq[current_stint_idx]
            max_stint = int(1.5 * comp.cliff_lap / self.model.circuit.tyre_degradation_multiplier)
            if min_stint <= remaining_laps <= max_stint:
                yield (remaining_laps,)
            return

        comp = compounds_seq[current_stint_idx]
        max_stint = min(
            remaining_laps - (stints_remaining - 1) * min_stint,
            int(1.5 * comp.cliff_lap / self.model.circuit.tyre_degradation_multiplier),
        )

        for l1 in range(min_stint, max_stint + 1, step):
            for rest in self._generate_partitions(
                remaining_laps - l1,
                stints_remaining - 1,
                compounds_seq,
                current_stint_idx + 1,
                min_stint,
                step,
            ):
                yield (l1,) + rest

    def generate_candidate_strategies(
        self, max_stops: int = 2, min_stint_length: int = 5, step: int = 1
    ) -> list[Strategy]:
        """Generate all FIA-compliant candidate strategies within constraints."""
        candidates = []

        # 1-Stop sequences (K=2)
        if max_stops >= 1:
            for c1 in self.compounds:
                for c2 in self.compounds:
                    if c1.name == c2.name:
                        continue  # Must use at least 2 distinct compounds
                    seq = [c1, c2]
                    for part in self._generate_partitions(
                        self.total_laps, 2, seq, min_stint=min_stint_length, step=step
                    ):
                        stints = [Stint(c, l) for c, l in zip(seq, part)]
                        strat_name = f"1-Stop ({c1.name[0]}-{c2.name[0]} {part[0]}-{part[1]})"
                        candidates.append(Strategy(stints, name=strat_name))

        # 2-Stop sequences (K=3)
        if max_stops >= 2:
            for c1 in self.compounds:
                for c2 in self.compounds:
                    for c3 in self.compounds:
                        if len({c1.name, c2.name, c3.name}) < 2:
                            continue  # Must use at least 2 distinct compounds
                        seq = [c1, c2, c3]
                        for part in self._generate_partitions(
                            self.total_laps, 3, seq, min_stint=min_stint_length, step=step
                        ):
                            stints = [Stint(c, l) for c, l in zip(seq, part)]
                            strat_name = f"2-Stop ({c1.name[0]}-{c2.name[0]}-{c3.name[0]} {part[0]}-{part[1]}-{part[2]})"
                            candidates.append(Strategy(stints, name=strat_name))

        # 3-Stop sequences (K=4) - only if explicitly requested, with step >= 2
        if max_stops >= 3:
            s3_step = max(2, step)
            for c1 in self.compounds:
                for c2 in self.compounds:
                    for c3 in self.compounds:
                        for c4 in self.compounds:
                            if len({c1.name, c2.name, c3.name, c4.name}) < 2:
                                continue
                            seq = [c1, c2, c3, c4]
                            for part in self._generate_partitions(
                                self.total_laps, 4, seq, min_stint=min_stint_length, step=s3_step
                            ):
                                stints = [Stint(c, l) for c, l in zip(seq, part)]
                                strat_name = f"3-Stop ({c1.name[0]}-{c2.name[0]}-{c3.name[0]}-{c4.name[0]})"
                                candidates.append(Strategy(stints, name=strat_name))

        return candidates

    def search(
        self, max_stops: int = 2, min_stint_length: int = 5, step: int = 1
    ) -> list[tuple[Strategy, float]]:
        """Evaluate and rank all candidates by deterministic total race time."""
        candidates = self.generate_candidate_strategies(
            max_stops=max_stops, min_stint_length=min_stint_length, step=step
        )

        results = []
        for strat in candidates:
            total_time = simulate_race_time_fast(strat, self.model)
            results.append((strat, total_time))

        results.sort(key=lambda x: x[1])
        return results


class PitWindowAnalyzer:
    """Calculates tactical pit windows based on allowable lap-time sensitivity."""

    @staticmethod
    def compute_pit_windows(
        strategy: Strategy,
        model: RaceModel,
        tolerance_seconds: float = 1.5,
        max_delta_laps: int = 8,
    ) -> list[PitWindow]:
        """Compute the allowable pit window for each stop of the strategy."""
        baseline_res = simulate_race(strategy, model)
        baseline_time = baseline_res.total_time
        num_stops = strategy.num_stops

        if num_stops == 0:
            return []

        pit_windows: list[PitWindow] = []
        compounds = [s.compound for s in strategy.stints]
        stint_lengths = [s.laps for s in strategy.stints]

        accumulated = 0
        for stop_idx in range(num_stops):
            accumulated += stint_lengths[stop_idx]
            optimal_lap = accumulated

            # Perturb stop lap earlier and later by shifting laps between stint stop_idx and stop_idx+1
            valid_laps = [optimal_lap]
            open_lap = optimal_lap
            close_lap = optimal_lap
            delta_open = 0.0
            delta_close = 0.0

            # 1. Search earlier (undercut window)
            for delta in range(1, max_delta_laps + 1):
                new_l1 = stint_lengths[stop_idx] - delta
                new_l2 = stint_lengths[stop_idx + 1] + delta
                if new_l1 < 3:
                    break

                test_stints = list(strategy.stints)
                test_stints[stop_idx] = Stint(compounds[stop_idx], new_l1)
                test_stints[stop_idx + 1] = Stint(compounds[stop_idx + 1], new_l2)
                test_strat = Strategy(test_stints)

                test_time = simulate_race(test_strat, model).total_time
                loss = test_time - baseline_time

                if loss <= tolerance_seconds:
                    open_lap = optimal_lap - delta
                    delta_open = loss
                else:
                    break

            # 2. Search later (overcut window)
            for delta in range(1, max_delta_laps + 1):
                new_l1 = stint_lengths[stop_idx] + delta
                new_l2 = stint_lengths[stop_idx + 1] - delta
                if new_l2 < 3:
                    break

                test_stints = list(strategy.stints)
                test_stints[stop_idx] = Stint(compounds[stop_idx], new_l1)
                test_stints[stop_idx + 1] = Stint(compounds[stop_idx + 1], new_l2)
                test_strat = Strategy(test_stints)

                test_time = simulate_race(test_strat, model).total_time
                loss = test_time - baseline_time

                if loss <= tolerance_seconds:
                    close_lap = optimal_lap + delta
                    delta_close = loss
                else:
                    break

            pit_windows.append(
                PitWindow(
                    pit_index=stop_idx + 1,
                    optimal_lap=optimal_lap,
                    window_open_lap=open_lap,
                    window_close_lap=close_lap,
                    tolerance_seconds=tolerance_seconds,
                    delta_open=delta_open,
                    delta_close=delta_close,
                )
            )

        return pit_windows


class StrategyOptimizer:
    """High-level facade coordinating strategy search, pit window analysis, and Pareto risk modeling."""

    def __init__(self, model: RaceModel, compounds: dict[str, TyreCompound]):
        self.model = model
        self.compounds = compounds
        self.dp_solver = DynamicProgrammingSolver(model, compounds)
        self.grid_solver = CombinatorialSearchSolver(model, compounds)

    def optimize(
        self,
        max_stops: int = 2,
        objective: OptimizationObjective = OptimizationObjective.DETERMINISTIC_TIME,
        mc_iterations: int = 500,
        pit_window_tolerance: float = 1.5,
        step: int = 1,
    ) -> OptimizationResult:
        """Discover the optimal strategy according to the specified objective."""
        start_time = time.perf_counter()

        # Step 1: Run Combinatorial Grid Search to rank candidates
        ranked_candidates = self.grid_solver.search(max_stops=max_stops, step=step)
        evaluations_count = len(ranked_candidates)

        if not ranked_candidates:
            # Fallback to Dynamic Programming if grid search returned empty
            dp_strat, dp_time = self.dp_solver.solve(max_stops=max_stops)
            ranked_candidates = [(dp_strat, dp_time)]

        # Step 2: Handle Objective Selection
        pareto_points: list[ParetoPoint] = []

        if objective == OptimizationObjective.DETERMINISTIC_TIME:
            best_strat, best_time = ranked_candidates[0]

        else:
            # For stochastic objectives (EXPECTED_TIME or MIN_RISK_P95):
            # Evaluate top candidates (up to 40) under Monte Carlo simulation
            stochastic_params = StochasticParameters()
            pool_size = min(40, len(ranked_candidates))
            top_subset = ranked_candidates[:pool_size]
            all_points: list[ParetoPoint] = []
            mc_evaluated = []

            for strat, det_time in top_subset:
                mc_res = simulate_monte_carlo(
                    strat, self.model, stochastic_params, n_iterations=mc_iterations, seed=42
                )
                point = ParetoPoint(
                    strategy=strat,
                    expected_time=mc_res.mean_time,
                    p95_time=mc_res.p95_time,
                    std_dev=mc_res.std_dev,
                    risk_penalty=mc_res.p95_time - mc_res.mean_time,
                )
                all_points.append(point)
                mc_evaluated.append((strat, point))

            # Strictly filter for non-dominated Pareto frontier
            pareto_points = compute_pareto_frontier(all_points)

            if objective == OptimizationObjective.EXPECTED_TIME:
                mc_evaluated.sort(key=lambda x: x[1].expected_time)
                best_strat = mc_evaluated[0][0]
                best_time = mc_evaluated[0][1].expected_time
            elif objective == OptimizationObjective.MIN_RISK_P95:
                mc_evaluated.sort(key=lambda x: x[1].p95_time)
                best_strat = mc_evaluated[0][0]
                best_time = mc_evaluated[0][1].p95_time
            else:
                best_strat, best_time = ranked_candidates[0]


        # Step 3: Compute Pit Windows for the winning strategy
        pit_windows = PitWindowAnalyzer.compute_pit_windows(
            best_strat, self.model, tolerance_seconds=pit_window_tolerance
        )

        solve_duration = time.perf_counter() - start_time

        return OptimizationResult(
            circuit_name=self.model.circuit.name,
            total_laps=self.model.circuit.total_laps,
            objective=objective,
            optimal_strategy=best_strat,
            optimal_race_time=best_time,
            formatted_optimal_time=_format_seconds(best_time),
            ranked_strategies=ranked_candidates,
            pit_windows=pit_windows,
            evaluations_count=evaluations_count,
            solve_time_seconds=solve_duration,
            pareto_frontier=pareto_points,
        )
