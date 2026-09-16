"""Phase 4: Sensitivity Analysis & Decision Phase Maps.

Provides parameter sweeps, exact tipping point root-finding (Brent's method),
2D decision phase boundaries, elasticity analysis, and minimax regret modeling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
from scipy.optimize import brentq

from src.model import CircuitConfig, RaceModel
from src.fuel import FuelModel
from src.optimization import CombinatorialSearchSolver, simulate_race_time_fast
from src.pitstop import PitStopModel
from src.strategies import Strategy
from src.tyres import TyreCompound


class SensitivityParameter(str, Enum):
    """Parameters that can be varied during sensitivity analysis."""

    TYRE_DEG_MULTIPLIER = "deg_multiplier"
    PIT_LOSS = "pit_loss"
    COMPOUND_DELTA_SOFT = "soft_delta"
    TRACK_EVOLUTION = "track_evolution"
    FUEL_PENALTY = "fuel_penalty"


@dataclass
class CrossoverPoint:
    """Represents a critical tipping point where two strategies or stop counts cross.

    Attributes:
        parameter_name: Name of the parameter swept.
        crossover_value: Critical value theta* where Delta T = 0.
        strategy_a_name: Strategy dominant on the left of theta*.
        strategy_b_name: Strategy dominant on the right of theta*.
        time_at_crossover: Total race time (s) at theta*.
        sensitivity_derivative: d(Delta T)/d(theta) at crossover (sharpness of transition).
        nominal_value: Baseline nominal parameter value for this circuit.
        distance_from_nominal: crossover_value - nominal_value.
        found: True if an exact root was discovered, False if one regime dominates throughout.
        message: Informative summary of the crossover status.
    """

    parameter_name: str
    crossover_value: float
    strategy_a_name: str
    strategy_b_name: str
    time_at_crossover: float = 0.0
    sensitivity_derivative: float = 0.0
    nominal_value: float = 0.0
    distance_from_nominal: float = 0.0
    found: bool = True
    message: str = ""

    @property
    def formatted_crossover(self) -> str:
        return f"{self.crossover_value:.4f}"


@dataclass
class SweepResult1D:
    """Results from a 1-dimensional parameter sweep.

    Attributes:
        parameter_name: Parameter identifier.
        parameter_display_name: Formatted display label (e.g. 'Tyre Degradation Multiplier').
        parameter_values: 1D array of sampled parameter values.
        strategy_times: Dictionary mapping strategy name to race time array.
        optimal_times: Lower envelope of minimum race time at each parameter value.
        optimal_strategies: Name of winning strategy at each parameter value.
        optimal_stops: Optimal number of pit stops at each parameter value.
        crossovers: List of discovered critical crossover tipping points.
        elasticities: Normalized elasticity (dimensionless sensitivity) per strategy at nominal.
        nominal_value: Circuit baseline reference value.
    """

    parameter_name: str
    parameter_display_name: str
    parameter_values: np.ndarray
    strategy_times: dict[str, np.ndarray]
    optimal_times: np.ndarray
    optimal_strategies: list[str]
    optimal_stops: np.ndarray
    crossovers: list[CrossoverPoint] = field(default_factory=list)
    elasticities: dict[str, float] = field(default_factory=dict)
    nominal_value: float = 1.0


@dataclass
class PhaseDiagramResult2D:
    """Results of a 2D decision boundary phase mapping.

    Attributes:
        circuit_name: Name of the Grand Prix / circuit.
        param_x_name: Identifier for parameter on X-axis.
        param_x_label: Display label for X-axis (e.g. 'Pit Loss Time (s)').
        param_y_name: Identifier for parameter on Y-axis.
        param_y_label: Display label for Y-axis (e.g. 'Tyre Degradation Multiplier').
        x_values: 1D array of X coordinates.
        y_values: 1D array of Y coordinates.
        x_grid: 2D meshgrid X coordinates.
        y_grid: 2D meshgrid Y coordinates.
        optimal_stops_grid: 2D array of winning stop count (1, 2, or 3) of shape (Ny, Nx).
        delta_grid: 2D array of (T_1stop* - T_2stop*) in seconds (positive => 2-stop is faster).
        winning_strategies: 2D array of strings containing best strategy descriptions.
        nominal_point: Tuple of (x_nominal, y_nominal).
        nominal_stops: Optimal stop count at the circuit's nominal operating point.
    """

    circuit_name: str
    param_x_name: str
    param_x_label: str
    param_y_name: str
    param_y_label: str
    x_values: np.ndarray
    y_values: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    optimal_stops_grid: np.ndarray
    delta_grid: np.ndarray
    winning_strategies: list[list[str]]
    nominal_point: tuple[float, float]
    nominal_stops: int


@dataclass
class RegretResult:
    """Evaluation of strategy regret and minimax robustness under parameter uncertainty."""

    strategy_names: list[str]
    max_regret: dict[str, float]
    expected_regret: dict[str, float]
    minimax_strategy: str


def clone_model_with_parameter(
    model: RaceModel,
    compounds: dict[str, TyreCompound],
    param: SensitivityParameter,
    value: float,
) -> tuple[RaceModel, dict[str, TyreCompound]]:
    """Return a new RaceModel and compound dict modified by the specified parameter value."""
    circuit = model.circuit
    fuel_model = model.fuel_model
    pitstop_model = model.pitstop_model
    new_compounds = dict(compounds)

    if param == SensitivityParameter.TYRE_DEG_MULTIPLIER:
        new_circuit = CircuitConfig(
            name=circuit.name,
            total_laps=circuit.total_laps,
            base_lap_time=circuit.base_lap_time,
            track_evolution_total=circuit.track_evolution_total,
            tyre_degradation_multiplier=float(value),
            traffic_config=circuit.traffic_config,
        )
        return RaceModel(circuit=new_circuit, fuel_model=fuel_model, pitstop_model=pitstop_model), new_compounds

    elif param == SensitivityParameter.PIT_LOSS:
        new_pit = PitStopModel(
            pit_loss=float(value),
            stationary_time=pitstop_model.stationary_time,
        )
        return RaceModel(circuit=circuit, fuel_model=fuel_model, pitstop_model=new_pit), new_compounds

    elif param == SensitivityParameter.TRACK_EVOLUTION:
        new_circuit = CircuitConfig(
            name=circuit.name,
            total_laps=circuit.total_laps,
            base_lap_time=circuit.base_lap_time,
            track_evolution_total=float(value),
            tyre_degradation_multiplier=circuit.tyre_degradation_multiplier,
            traffic_config=circuit.traffic_config,
        )
        return RaceModel(circuit=new_circuit, fuel_model=fuel_model, pitstop_model=pitstop_model), new_compounds

    elif param == SensitivityParameter.FUEL_PENALTY:
        new_fuel = FuelModel(
            initial_fuel_kg=fuel_model.initial_fuel_kg,
            fuel_penalty_per_kg=float(value),
        )
        return RaceModel(circuit=circuit, fuel_model=new_fuel, pitstop_model=pitstop_model), new_compounds

    elif param == SensitivityParameter.COMPOUND_DELTA_SOFT:
        # Modify the base pace offset of the Soft tyre (or softest compound present)
        # Find softest compound by lowest base_delta
        softest_key = min(compounds.keys(), key=lambda k: compounds[k].base_delta)
        orig_soft = compounds[softest_key]
        modified_soft = TyreCompound(
            name=orig_soft.name,
            base_delta=float(value),
            alpha=orig_soft.alpha,
            beta=orig_soft.beta,
            cliff_lap=orig_soft.cliff_lap,
            cliff_coefficient=orig_soft.cliff_coefficient,
        )
        for k, comp in compounds.items():
            if comp.name == orig_soft.name:
                new_compounds[k] = modified_soft
        return model, new_compounds

    else:
        raise ValueError(f"Unsupported sensitivity parameter: {param}")


def get_nominal_parameter_value(
    model: RaceModel, compounds: dict[str, TyreCompound], param: SensitivityParameter
) -> float:
    """Retrieve the circuit baseline value for a parameter."""
    if param == SensitivityParameter.TYRE_DEG_MULTIPLIER:
        return model.circuit.tyre_degradation_multiplier
    elif param == SensitivityParameter.PIT_LOSS:
        return model.pitstop_model.pit_loss
    elif param == SensitivityParameter.TRACK_EVOLUTION:
        return model.circuit.track_evolution_total
    elif param == SensitivityParameter.FUEL_PENALTY:
        return model.fuel_model.fuel_penalty_per_kg
    elif param == SensitivityParameter.COMPOUND_DELTA_SOFT:
        softest_key = min(compounds.keys(), key=lambda k: compounds[k].base_delta)
        return compounds[softest_key].base_delta
    else:
        raise ValueError(f"Unsupported sensitivity parameter: {param}")


def get_parameter_display_name(param: SensitivityParameter) -> str:
    """Human-friendly display name for a parameter."""
    mapping = {
        SensitivityParameter.TYRE_DEG_MULTIPLIER: "Tyre Degradation Multiplier",
        SensitivityParameter.PIT_LOSS: "Pit Stop Time Loss (s)",
        SensitivityParameter.COMPOUND_DELTA_SOFT: "Soft Compound Pace Delta (s/lap)",
        SensitivityParameter.TRACK_EVOLUTION: "Track Evolution Total (s)",
        SensitivityParameter.FUEL_PENALTY: "Fuel Penalty (s/kg)",
    }
    return mapping.get(param, param.value)


class ParameterSweep1D:
    """Executes 1-dimensional parameter sweeps across strategies or re-optimization."""

    def __init__(self, model: RaceModel, compounds: dict[str, TyreCompound]):
        self.model = model
        self.compounds = compounds

    def sweep_strategies(
        self,
        strategies: list[Strategy],
        param: SensitivityParameter,
        values: np.ndarray,
    ) -> SweepResult1D:
        """Evaluate a fixed set of strategies across a 1D parameter array."""
        values = np.asarray(values, dtype=float)
        strategy_times: dict[str, np.ndarray] = {
            strat.description: np.zeros(len(values)) for strat in strategies
        }

        nominal_val = get_nominal_parameter_value(self.model, self.compounds, param)

        for i, val in enumerate(values):
            test_model, test_compounds = clone_model_with_parameter(
                self.model, self.compounds, param, val
            )
            for strat in strategies:
                time_s = simulate_race_time_fast(strat, test_model, compounds=test_compounds)
                strategy_times[strat.description][i] = time_s

        # Compute lower envelope
        all_times = np.vstack(list(strategy_times.values()))
        optimal_indices = np.argmin(all_times, axis=0)
        optimal_times = np.min(all_times, axis=0)

        strat_keys = list(strategy_times.keys())
        optimal_strategies = [strat_keys[idx] for idx in optimal_indices]
        optimal_stops = np.array([strategies[idx].num_stops for idx in optimal_indices])

        # Discover crossovers between adjacent strategies in envelope
        crossovers = []
        for i in range(len(values) - 1):
            s_left = optimal_indices[i]
            s_right = optimal_indices[i + 1]
            if s_left != s_right:
                # Interpolate linear crossover value
                t_diff_left = all_times[s_left, i] - all_times[s_right, i]
                t_diff_right = all_times[s_left, i + 1] - all_times[s_right, i + 1]
                denom = t_diff_left - t_diff_right
                frac = t_diff_left / denom if denom != 0 else 0.5
                c_val = values[i] + frac * (values[i + 1] - values[i])
                time_at_c = optimal_times[i] + frac * (optimal_times[i + 1] - optimal_times[i])

                d_val = values[i + 1] - values[i]
                deriv = (all_times[s_right, i + 1] - all_times[s_left, i + 1] - (all_times[s_right, i] - all_times[s_left, i])) / d_val if d_val > 0 else 0.0

                crossovers.append(
                    CrossoverPoint(
                        parameter_name=param.value,
                        crossover_value=float(c_val),
                        strategy_a_name=strat_keys[s_left],
                        strategy_b_name=strat_keys[s_right],
                        time_at_crossover=float(time_at_c),
                        sensitivity_derivative=float(deriv),
                        nominal_value=nominal_val,
                        distance_from_nominal=float(c_val - nominal_val),
                    )
                )

        # Compute dimensionless elasticity at nominal baseline
        # E_theta = (theta0 / T0) * (dT / dtheta)
        elasticities = {}
        if param == SensitivityParameter.PIT_LOSS:
            eps = 0.5  # 0.5s step prevents integer truncation spikes from traffic model
        elif param == SensitivityParameter.TYRE_DEG_MULTIPLIER:
            eps = 0.05
        else:
            eps = 1e-2 * (abs(nominal_val) if abs(nominal_val) > 1e-4 else 1.0)

        m_pos, c_pos = clone_model_with_parameter(self.model, self.compounds, param, nominal_val + eps)
        m_neg, c_neg = clone_model_with_parameter(self.model, self.compounds, param, nominal_val - eps)

        for strat in strategies:
            t_nom = simulate_race_time_fast(strat, self.model, compounds=self.compounds)
            t_pos = simulate_race_time_fast(strat, m_pos, compounds=c_pos)
            t_neg = simulate_race_time_fast(strat, m_neg, compounds=c_neg)
            dt_dparam = (t_pos - t_neg) / (2.0 * eps)
            elasticity = (nominal_val / t_nom) * dt_dparam if t_nom > 0 else 0.0
            elasticities[strat.description] = float(elasticity)

        return SweepResult1D(
            parameter_name=param.value,
            parameter_display_name=get_parameter_display_name(param),
            parameter_values=values,
            strategy_times=strategy_times,
            optimal_times=optimal_times,
            optimal_strategies=optimal_strategies,
            optimal_stops=optimal_stops,
            crossovers=crossovers,
            elasticities=elasticities,
            nominal_value=nominal_val,
        )

    def sweep_reoptimization(
        self,
        param: SensitivityParameter,
        values: np.ndarray,
        max_stops: int = 2,
        step: int = 2,
    ) -> SweepResult1D:
        """At each parameter value, discover the best 1-stop, best 2-stop, and global optimum."""
        values = np.asarray(values, dtype=float)
        grid_solver = CombinatorialSearchSolver(self.model, self.compounds)
        c1_candidates = grid_solver.generate_candidate_strategies(max_stops=1, step=1)
        c2_candidates = [
            s for s in grid_solver.generate_candidate_strategies(max_stops=2, step=step)
            if s.num_stops == 2
        ]

        strategy_times: dict[str, np.ndarray] = {
            "Optimal 1-Stop": np.zeros(len(values)),
            "Optimal 2-Stop": np.zeros(len(values)),
        }
        if max_stops >= 3:
            c3_candidates = [
                s for s in grid_solver.generate_candidate_strategies(max_stops=3, step=max(3, step))
                if s.num_stops == 3
            ]
            strategy_times["Optimal 3-Stop"] = np.zeros(len(values))
        else:
            c3_candidates = []

        nominal_val = get_nominal_parameter_value(self.model, self.compounds, param)
        winning_names = []
        optimal_stops_arr = np.zeros(len(values), dtype=int)
        optimal_times_arr = np.zeros(len(values))

        for i, val in enumerate(values):
            test_model, test_compounds = clone_model_with_parameter(
                self.model, self.compounds, param, val
            )

            # Evaluate best 1-stop
            t1 = min(simulate_race_time_fast(s, test_model, compounds=test_compounds) for s in c1_candidates)
            strategy_times["Optimal 1-Stop"][i] = t1

            # Evaluate best 2-stop
            t2 = min(simulate_race_time_fast(s, test_model, compounds=test_compounds) for s in c2_candidates)
            strategy_times["Optimal 2-Stop"][i] = t2

            if c3_candidates:
                t3 = min(simulate_race_time_fast(s, test_model, compounds=test_compounds) for s in c3_candidates)
                strategy_times["Optimal 3-Stop"][i] = t3
                scores = [(1, t1), (2, t2), (3, t3)]
            else:
                scores = [(1, t1), (2, t2)]

            scores.sort(key=lambda x: x[1])
            best_stops, best_time = scores[0]
            optimal_stops_arr[i] = best_stops
            optimal_times_arr[i] = best_time
            winning_names.append(f"Optimal {best_stops}-Stop")

        # Discover crossovers between 1-stop and 2-stop
        crossovers = []
        t1_arr = strategy_times["Optimal 1-Stop"]
        t2_arr = strategy_times["Optimal 2-Stop"]
        for i in range(len(values) - 1):
            diff_left = t1_arr[i] - t2_arr[i]
            diff_right = t1_arr[i + 1] - t2_arr[i + 1]
            if (diff_left < 0 and diff_right > 0) or (diff_left > 0 and diff_right < 0):
                denom = diff_left - diff_right
                frac = diff_left / denom if denom != 0 else 0.5
                c_val = values[i] + frac * (values[i + 1] - values[i])
                time_at_c = 0.5 * (t1_arr[i] + t2_arr[i]) + frac * (
                    0.5 * (t1_arr[i + 1] + t2_arr[i + 1]) - 0.5 * (t1_arr[i] + t2_arr[i])
                )
                left_winner = "Optimal 1-Stop" if diff_left < 0 else "Optimal 2-Stop"
                right_winner = "Optimal 2-Stop" if diff_left < 0 else "Optimal 1-Stop"
                d_val = values[i + 1] - values[i]
                deriv = (diff_right - diff_left) / d_val if d_val > 0 else 0.0

                crossovers.append(
                    CrossoverPoint(
                        parameter_name=param.value,
                        crossover_value=float(c_val),
                        strategy_a_name=left_winner,
                        strategy_b_name=right_winner,
                        time_at_crossover=float(time_at_c),
                        sensitivity_derivative=float(deriv),
                        nominal_value=nominal_val,
                        distance_from_nominal=float(c_val - nominal_val),
                    )
                )

        # Compute dimensionless elasticity at nominal baseline for optimal 1-stop and 2-stop
        elasticities = {}
        if param == SensitivityParameter.PIT_LOSS:
            eps = 0.5
        elif param == SensitivityParameter.TYRE_DEG_MULTIPLIER:
            eps = 0.05
        else:
            eps = 1e-2 * (abs(nominal_val) if abs(nominal_val) > 1e-4 else 1.0)

        m_pos, c_pos = clone_model_with_parameter(self.model, self.compounds, param, nominal_val + eps)
        m_neg, c_neg = clone_model_with_parameter(self.model, self.compounds, param, nominal_val - eps)

        t1_nom = min(simulate_race_time_fast(s, self.model, compounds=self.compounds) for s in c1_candidates)
        t1_pos = min(simulate_race_time_fast(s, m_pos, compounds=c_pos) for s in c1_candidates)
        t1_neg = min(simulate_race_time_fast(s, m_neg, compounds=c_neg) for s in c1_candidates)
        e1 = (nominal_val / t1_nom) * ((t1_pos - t1_neg) / (2.0 * eps)) if t1_nom > 0 else 0.0
        elasticities["Optimal 1-Stop"] = float(e1)

        t2_nom = min(simulate_race_time_fast(s, self.model, compounds=self.compounds) for s in c2_candidates)
        t2_pos = min(simulate_race_time_fast(s, m_pos, compounds=c_pos) for s in c2_candidates)
        t2_neg = min(simulate_race_time_fast(s, m_neg, compounds=c_neg) for s in c2_candidates)
        e2 = (nominal_val / t2_nom) * ((t2_pos - t2_neg) / (2.0 * eps)) if t2_nom > 0 else 0.0
        elasticities["Optimal 2-Stop"] = float(e2)

        return SweepResult1D(
            parameter_name=param.value,
            parameter_display_name=get_parameter_display_name(param),
            parameter_values=values,
            strategy_times=strategy_times,
            optimal_times=optimal_times_arr,
            optimal_strategies=winning_names,
            optimal_stops=optimal_stops_arr,
            crossovers=crossovers,
            elasticities=elasticities,
            nominal_value=nominal_val,
        )


class CrossoverFinder:
    """Computes exact critical tipping points using Brent's method."""

    def __init__(self, model: RaceModel, compounds: dict[str, TyreCompound]):
        self.model = model
        self.compounds = compounds
        grid_solver = CombinatorialSearchSolver(model, compounds)
        self.c1_pool = grid_solver.generate_candidate_strategies(max_stops=1, step=1)
        self.c2_pool = [
            s for s in grid_solver.generate_candidate_strategies(max_stops=2, step=2)
            if s.num_stops == 2
        ]

    def find_1_vs_2_stop_crossover(
        self,
        param: SensitivityParameter,
        search_range: tuple[float, float],
        xtol: float = 1e-4,
    ) -> CrossoverPoint:
        """Find exact tipping point theta* where T*(1-stop) == T*(2-stop)."""
        val_min, val_max = search_range
        nominal_val = get_nominal_parameter_value(self.model, self.compounds, param)

        def delta_objective(val: float) -> float:
            test_model, test_compounds = clone_model_with_parameter(self.model, self.compounds, param, val)
            t1 = min(simulate_race_time_fast(s, test_model, compounds=test_compounds) for s in self.c1_pool)
            t2 = min(simulate_race_time_fast(s, test_model, compounds=test_compounds) for s in self.c2_pool)
            return t1 - t2

        f_min = delta_objective(val_min)
        f_max = delta_objective(val_max)

        if f_min * f_max > 0:
            # No root in [val_min, val_max]
            dominant = "2-Stop" if f_min > 0 else "1-Stop"
            return CrossoverPoint(
                parameter_name=param.value,
                crossover_value=float("nan"),
                strategy_a_name=dominant,
                strategy_b_name=dominant,
                nominal_value=nominal_val,
                found=False,
                message=(
                    f"No crossover detected in [{val_min:.2f}, {val_max:.2f}]. "
                    f"{dominant} strictly dominates across entire range."
                ),
            )

        root = brentq(delta_objective, val_min, val_max, xtol=xtol)

        # Compute derivative d(Delta T)/d(val) at root using traffic-safe step size
        if param == SensitivityParameter.PIT_LOSS:
            eps = 0.5
        elif param == SensitivityParameter.TYRE_DEG_MULTIPLIER:
            eps = 0.05
        else:
            eps = max(1e-3, 1e-2 * (abs(root) if abs(root) > 1e-4 else 1.0))
        d_val = (delta_objective(root + eps) - delta_objective(root - eps)) / (2 * eps)

        test_model_root, test_compounds_root = clone_model_with_parameter(self.model, self.compounds, param, root)
        time_at_root = min(simulate_race_time_fast(s, test_model_root, compounds=test_compounds_root) for s in self.c1_pool)

        left_regime = "1-Stop" if f_min < 0 else "2-Stop"
        right_regime = "2-Stop" if f_min < 0 else "1-Stop"

        return CrossoverPoint(
            parameter_name=param.value,
            crossover_value=float(root),
            strategy_a_name=left_regime,
            strategy_b_name=right_regime,
            time_at_crossover=float(time_at_root),
            sensitivity_derivative=float(d_val),
            nominal_value=nominal_val,
            distance_from_nominal=float(root - nominal_val),
            found=True,
            message=f"Exact crossover discovered at {root:.4f} (baseline: {nominal_val:.2f})",
        )


class PhaseDiagram2D:
    """Computes high-throughput 2D decision boundary grids."""

    def __init__(self, model: RaceModel, compounds: dict[str, TyreCompound]):
        self.model = model
        self.compounds = compounds
        grid_solver = CombinatorialSearchSolver(model, compounds)
        self.c1_pool = grid_solver.generate_candidate_strategies(max_stops=1, step=1)
        self.c2_pool = [
            s for s in grid_solver.generate_candidate_strategies(max_stops=2, step=2)
            if s.num_stops == 2
        ]

    def compute_phase_map(
        self,
        param_x: SensitivityParameter = SensitivityParameter.PIT_LOSS,
        x_range: tuple[float, float] = (16.0, 32.0),
        param_y: SensitivityParameter = SensitivityParameter.TYRE_DEG_MULTIPLIER,
        y_range: tuple[float, float] = (0.6, 2.2),
        resolution: tuple[int, int] = (25, 25),
    ) -> PhaseDiagramResult2D:
        """Generate a 2D matrix of optimal stop counts and delta surfaces across two parameters."""
        nx, ny = resolution
        x_vals = np.linspace(x_range[0], x_range[1], nx)
        y_vals = np.linspace(y_range[0], y_range[1], ny)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)

        optimal_stops_grid = np.zeros((ny, nx), dtype=int)
        delta_grid = np.zeros((ny, nx), dtype=float)
        winning_strats: list[list[str]] = [["" for _ in range(nx)] for _ in range(ny)]

        nom_x = get_nominal_parameter_value(self.model, self.compounds, param_x)
        nom_y = get_nominal_parameter_value(self.model, self.compounds, param_y)

        for j in range(ny):
            y_val = y_vals[j]
            for i in range(nx):
                x_val = x_vals[i]

                # Clone model applying both parameters sequentially
                m_temp, c_temp = clone_model_with_parameter(
                    self.model, self.compounds, param_x, x_val
                )
                m_eval, c_eval = clone_model_with_parameter(
                    m_temp, c_temp, param_y, y_val
                )

                # Evaluate best 1-stop
                best_1_strat = min(self.c1_pool, key=lambda s: simulate_race_time_fast(s, m_eval, compounds=c_eval))
                t1 = simulate_race_time_fast(best_1_strat, m_eval, compounds=c_eval)

                # Evaluate best 2-stop
                best_2_strat = min(self.c2_pool, key=lambda s: simulate_race_time_fast(s, m_eval, compounds=c_eval))
                t2 = simulate_race_time_fast(best_2_strat, m_eval, compounds=c_eval)

                delta = t1 - t2  # Positive => 2-stop is faster
                delta_grid[j, i] = delta

                if delta > 0:
                    optimal_stops_grid[j, i] = 2
                    winning_strats[j][i] = best_2_strat.description
                else:
                    optimal_stops_grid[j, i] = 1
                    winning_strats[j][i] = best_1_strat.description

        # Classify nominal point
        m_nom_x, c_nom_x = clone_model_with_parameter(self.model, self.compounds, param_x, nom_x)
        m_nom, c_nom = clone_model_with_parameter(m_nom_x, c_nom_x, param_y, nom_y)
        nom_t1 = min(simulate_race_time_fast(s, m_nom, compounds=c_nom) for s in self.c1_pool)
        nom_t2 = min(simulate_race_time_fast(s, m_nom, compounds=c_nom) for s in self.c2_pool)
        nominal_stops = 2 if nom_t1 > nom_t2 else 1

        return PhaseDiagramResult2D(
            circuit_name=self.model.circuit.name,
            param_x_name=param_x.value,
            param_x_label=get_parameter_display_name(param_x),
            param_y_name=param_y.value,
            param_y_label=get_parameter_display_name(param_y),
            x_values=x_vals,
            y_values=y_vals,
            x_grid=x_grid,
            y_grid=y_grid,
            optimal_stops_grid=optimal_stops_grid,
            delta_grid=delta_grid,
            winning_strategies=winning_strats,
            nominal_point=(nom_x, nom_y),
            nominal_stops=nominal_stops,
        )


class RobustnessAnalyzer:
    """Calculates minimax regret and expected regret under parameter uncertainty."""

    def __init__(self, model: RaceModel, compounds: dict[str, TyreCompound]):
        self.model = model
        self.compounds = compounds

    def compute_regret(
        self,
        strategies: list[Strategy],
        param: SensitivityParameter,
        values: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> RegretResult:
        """Compute maximum regret and expected regret for candidate strategies."""
        values = np.asarray(values, dtype=float)
        if weights is None:
            weights = np.ones(len(values)) / len(values)
        else:
            weights = np.asarray(weights, dtype=float) / np.sum(weights)

        times_matrix = np.zeros((len(strategies), len(values)))
        for j, val in enumerate(values):
            test_model, test_compounds = clone_model_with_parameter(self.model, self.compounds, param, val)
            for i, strat in enumerate(strategies):
                times_matrix[i, j] = simulate_race_time_fast(strat, test_model, compounds=test_compounds)

        lower_envelope = np.min(times_matrix, axis=0)  # best possible time at each point
        regret_matrix = times_matrix - lower_envelope  # shape: (n_strategies, n_values)

        max_regrets = {}
        exp_regrets = {}
        names = [s.description for s in strategies]

        for i, name in enumerate(names):
            max_regrets[name] = float(np.max(regret_matrix[i, :]))
            exp_regrets[name] = float(np.sum(regret_matrix[i, :] * weights))

        minimax_strat = min(max_regrets.keys(), key=lambda k: max_regrets[k])

        return RegretResult(
            strategy_names=names,
            max_regret=max_regrets,
            expected_regret=exp_regrets,
            minimax_strategy=minimax_strat,
        )
