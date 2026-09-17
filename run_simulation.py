import argparse
import sys
from typing import Optional

import numpy as np

from src.config import CIRCUIT_PRESETS, create_race_model, get_default_compounds
from src.simulation import simulate_race
from src.strategies import Stint, Strategy


def parse_strategy_string(s_str: str, compounds: dict) -> Strategy:
    """Parse shorthand strategy like 'S15-M21-S21'."""
    stints = []
    parts = s_str.split("-")
    for part in parts:
        part = part.strip().upper()
        if not part:
            continue
        comp = part[0]
        try:
            laps = int(part[1:])
        except ValueError:
            raise ValueError(f"Invalid stint format '{part}'. Expected e.g. 'S15'.")

        if comp == "S":
            stints.append(Stint(compounds["Soft"], laps))
        elif comp == "M":
            stints.append(Stint(compounds["Medium"], laps))
        elif comp == "H":
            stints.append(Stint(compounds["Hard"], laps))
        else:
            raise ValueError(f"Unknown compound '{comp}'. Use S, M, or H.")
    return Strategy(stints, name=s_str)


def get_default_strategies(circuit_key: str, compounds: dict) -> list[Strategy]:
    """Return common strategies based on the circuit."""
    s = compounds["Soft"]
    m = compounds["Medium"]
    h = compounds["Hard"]

    if circuit_key == "bahrain":
        # 57 laps
        return [
            Strategy([Stint(s, 15), Stint(m, 21), Stint(s, 21)], name="2-Stop (S-M-S)"),
            Strategy([Stint(s, 14), Stint(h, 22), Stint(m, 21)], name="2-Stop (S-H-M)"),
            Strategy([Stint(m, 26), Stint(h, 31)], name="1-Stop (M-H)"),
            Strategy([Stint(m, 20), Stint(h, 20), Stint(s, 17)], name="2-Stop (M-H-S)"),
        ]
    elif circuit_key == "barcelona":
        # 66 laps
        return [
            Strategy([Stint(s, 18), Stint(m, 26), Stint(h, 22)], name="2-Stop (S-M-H)"),
            Strategy([Stint(m, 30), Stint(h, 36)], name="1-Stop (M-H)"),
            Strategy([Stint(s, 16), Stint(m, 25), Stint(s, 25)], name="2-Stop (S-M-S)"),
        ]
    elif circuit_key == "monza":
        # 53 laps (low degradation track)
        return [
            Strategy([Stint(m, 24), Stint(h, 29)], name="1-Stop (M-H)"),
            Strategy([Stint(s, 18), Stint(h, 35)], name="1-Stop (S-H)"),
            Strategy([Stint(s, 15), Stint(m, 20), Stint(s, 18)], name="2-Stop (S-M-S)"),
        ]
    else:
        raise ValueError(f"Unknown circuit: {circuit_key}")


def print_comparison_table(results: list, circuit_name: str, total_laps: int) -> None:
    """Print an ASCII comparison table sorted by total race time."""
    results.sort(key=lambda r: r.total_time)
    winning_time = results[0].total_time

    print("\n" + "=" * 92)
    print(f"  F1 RACE STRATEGY SIMULATION RESULTS: {circuit_name} ({total_laps} Laps)")
    print("=" * 92)
    print(f"{'Pos':<4} {'Strategy Name':<18} {'Stints':<30} {'Stops':<6} {'Total Time':<15} {'Delta':<10}")
    print("-" * 92)

    for pos, res in enumerate(results, start=1):
        s = res.summary()
        delta_str = "LEADER" if pos == 1 else f"+{res.total_time - winning_time:.3f}s"
        print(
            f"{pos:<4} {s['strategy']:<18} {s['stints']:<30} {s['stops']:<6} {s['formatted_time']:<15} {delta_str:<10}"
        )
    print("=" * 92 + "\n")


def print_lap_telemetry(result, max_laps: Optional[int] = None) -> None:
    """Print lap-by-lap breakdown."""
    df = result.to_dataframe()
    if max_laps:
        df = df.head(max_laps)

    print("\n" + "-" * 105)
    print(f"  LAP-BY-LAP BREAKDOWN: {result.strategy.name or result.strategy.description}")
    print("-" * 105)
    print(
        f"{'Lap':<5} {'Stint':<6} {'Tire':<7} {'Age':<5} {'Fuel(kg)':<9} "
        f"{'Base(s)':<8} {'Deg(s)':<8} {'Fuel(s)':<8} {'Pit(s)':<7} {'LapTime':<10} {'CumTime':<12}"
    )
    print("-" * 105)
    for _, row in df.iterrows():
        pit_marker = f"{row['pit_loss']:.1f}" if row["pit_loss"] > 0 else "-"
        print(
            f"{int(row['lap']):<5} {int(row['stint']):<6} {row['compound']:<7} {int(row['tyre_age']):<5} "
            f"{row['fuel_kg']:<9.2f} {row['base_time']:<8.2f} {row['tyre_deg']:<8.3f} "
            f"{row['fuel_penalty']:<8.3f} {pit_marker:<7} {row['effective_time']:<10.3f} {row['cumulative_time']:<12.3f}"
        )
    print("-" * 105 + "\n")


def print_comparison_table_mc(results: list, circuit_name: str, total_laps: int) -> None:
    """Print an ASCII comparison table sorted by expected race time."""
    results.sort(key=lambda r: r.mean_time)
    winning_time = results[0].mean_time

    print("\n" + "=" * 105)
    print(f"  F1 MONTE CARLO RESULTS: {circuit_name} ({total_laps} Laps, {results[0].n_iterations} Iterations)")
    print("=" * 105)
    print(f"{'Pos':<4} {'Strategy Name':<18} {'Stops':<6} {'Expected Time':<15} {'Delta':<10} {'P95 (Risk)':<15} {'Win% vs Field':<12}")
    print("-" * 105)

    for pos, res in enumerate(results, start=1):
        s = res.summary()
        delta_str = "LEADER" if pos == 1 else f"+{res.mean_time - winning_time:.3f}s"
        
        # Calculate win probability against the field (simplification: vs the leader, or just overall)
        wins = 0
        if pos == 1:
            win_prob_str = "-"
        else:
            wins_against_leader = np.sum(res.race_times < results[0].race_times)
            win_prob = wins_against_leader / res.n_iterations
            win_prob_str = f"{win_prob:.1%}"
            
        print(
            f"{pos:<4} {s['strategy']:<18} {s['stops']:<6} {s['formatted_mean']:<15} {delta_str:<10} {s['formatted_p95']:<15} {win_prob_str:<12}"
        )
    print("=" * 105 + "\n")


def print_optimization_table(res, show_pit_windows: bool = True) -> None:
    """Print structured ASCII table for optimization results."""
    print("\n" + "=" * 105)
    print(f"  F1 STRATEGY OPTIMIZATION ENGINE: {res.circuit_name} ({res.total_laps} Laps)")
    print("=" * 105)
    print(f"  Optimization Objective:  {res.objective.value.upper()}")
    print(f"  Optimal Strategy:        {res.optimal_strategy.name or res.optimal_strategy.description}")
    print(f"  Stint Breakdown:         {res.optimal_strategy.description}")
    print(f"  Total Race Time:         {res.formatted_optimal_time} ({res.optimal_race_time:.3f}s)")
    print(f"  Evaluations & Runtime:   {res.evaluations_count:,} strategies evaluated in {res.solve_time_seconds:.3f}s")

    if show_pit_windows and res.pit_windows:
        print("-" * 105)
        print("  TACTICAL PIT WINDOWS:")
        for pw in res.pit_windows:
            print(
                f"    * Stop {pw.pit_index}: Optimal Lap {pw.optimal_lap:<2} "
                f"[Open: Lap {pw.window_open_lap} (+{pw.delta_open:.2f}s) | "
                f"Close: Lap {pw.window_close_lap} (+{pw.delta_close:.2f}s)] "
                f"-> Window Size: {pw.window_size} Laps (Tol: +{pw.tolerance_seconds:.1f}s)"
            )

    if res.ranked_strategies:
        print("-" * 105)
        print("  TOP-5 STRATEGY LEADERBOARD:")
        print(f"  {'Pos':<4} {'Strategy Name':<28} {'Stops':<6} {'Stint Breakdown':<34} {'Race Time':<16} {'Delta':<10}")
        print("  " + "-" * 101)
        best_time = res.ranked_strategies[0][1]
        for pos, (strat, s_time) in enumerate(res.ranked_strategies[:5], start=1):
            delta_str = "OPTIMAL" if pos == 1 else f"+{s_time - best_time:.3f}s"
            hours = int(s_time // 3600)
            rem = s_time % 3600
            m_str = f"{hours}h {int(rem // 60):02d}m {rem % 60:06.3f}s" if hours > 0 else f"{int(rem // 60):02d}m {rem % 60:06.3f}s"
            print(
                f"  {pos:<4} {strat.name or 'Strategy':<28} {strat.num_stops:<6} "
                f"{strat.description:<34} {m_str:<16} {delta_str:<10}"
            )

    if res.pareto_frontier:
        print("-" * 105)
        print("  PARETO OPTIMAL FRONTIER (Pace vs. Risk):")
        print(f"  {'Rank':<5} {'Strategy':<30} {'Expected Time':<18} {'P95 (Risk)':<18} {'Risk Delta':<12}")
        print("  " + "-" * 101)
        for idx, pt in enumerate(res.pareto_frontier[:5], start=1):
            print(
                f"  {idx:<5} {pt.strategy.description:<30} {pt.formatted_expected:<18} "
                f"{pt.formatted_p95:<18} +{pt.risk_penalty:.2f}s"
            )

    print("=" * 105 + "\n")


def print_crossover_table(c, circuit_name: str) -> None:
    """Print ASCII table summarizing critical tipping points."""
    print("\n" + "=" * 95)
    print(f"  CRITICAL TIPPING POINT (CROSSOVER) ANALYSIS: {circuit_name}")
    print("=" * 95)
    print(f"  Parameter Evaluated:     {c.parameter_name}")
    print(f"  Circuit Nominal Baseline:{c.nominal_value:.4f}")
    if c.found:
        print(f"  Exact Crossover Root:    {c.crossover_value:.4f}")
        print(f"  Distance from Nominal:   {c.distance_from_nominal:+.4f} (Parameter delta required to flip optimum)")
        print(f"  Strategy on Left:        {c.strategy_a_name}")
        print(f"  Strategy on Right:       {c.strategy_b_name}")
        print(f"  Transition Sharpness:    {c.sensitivity_derivative:+.3f} s / unit (d(Delta T)/d(param))")
        if c.time_at_crossover > 0:
            from src.optimization import _format_seconds
            print(f"  Race Time at Crossover:  {_format_seconds(c.time_at_crossover)} ({c.time_at_crossover:.3f}s)")
    else:
        print(f"  Crossover Status:        {c.message}")
    print("=" * 95 + "\n")


def print_sensitivity_sweep_table(res, circuit_name: str) -> None:
    """Print ASCII table showing 1D parameter sweep and elasticity."""
    print("\n" + "=" * 105)
    print(f"  1D SENSITIVITY SWEEP: {circuit_name} - {res.parameter_display_name}")
    print("=" * 105)
    strat_names = list(res.strategy_times.keys())
    val_header = f"{'Param Val':<12}"
    strat_headers = "".join([f"{name[:20]:<22}" for name in strat_names])
    print(val_header + strat_headers + f"{'Winning Strategy':<20}")
    print("-" * 105)

    step_skip = max(1, len(res.parameter_values) // 14)
    for i in range(0, len(res.parameter_values), step_skip):
        v = res.parameter_values[i]
        nom_marker = " *" if abs(v - res.nominal_value) < 1e-4 else "  "
        v_str = f"{v:.3f}{nom_marker}"
        times_strs = "".join(
            [f"{res.strategy_times[name][i]/60.0:6.2f}m ({res.strategy_times[name][i]:.1f}s)    " for name in strat_names]
        )
        print(f"{v_str:<12}" + times_strs + f"{res.optimal_strategies[i]:<20}")
    print("-" * 105)

    if res.elasticities:
        print("  NORMALIZED DIMENSIONLESS ELASTICITY AT NOMINAL BASELINE (% Time / % Param):")
        for s_name, elast in res.elasticities.items():
            print(f"    * {s_name:<32}: E = {elast:+.4f} ({elast*100:+.2f}% race time shift per 1% change)")

    if res.crossovers:
        print("  DISCOVERED TIPPING POINTS:")
        for c in res.crossovers:
            print(
                f"    * Tipping at {c.crossover_value:.4f}: {c.strategy_a_name} ↔ {c.strategy_b_name} "
                f"(Nominal distance: {c.distance_from_nominal:+.4f})"
            )
    print("=" * 105 + "\n")


def print_phase_map_summary(phase_res) -> None:
    """Print summary metrics for 2D decision boundary phase mapping."""
    print("\n" + "=" * 95)
    print(f"  2D DECISION BOUNDARY PHASE MAP: {phase_res.circuit_name}")
    print("=" * 95)
    print(f"  X-Axis Parameter:        {phase_res.param_x_label} [{phase_res.x_values[0]:.1f} to {phase_res.x_values[-1]:.1f}]")
    print(f"  Y-Axis Parameter:        {phase_res.param_y_label} [{phase_res.y_values[0]:.2f} to {phase_res.y_values[-1]:.2f}]")
    nom_x, nom_y = phase_res.nominal_point
    print(f"  Nominal Operating Point: ({nom_x:.1f}s, {nom_y:.2f})")
    print(f"  Nominal Winning Regime:  {phase_res.nominal_stops}-Stop Strategy")

    n_1stop = int(np.sum(phase_res.optimal_stops_grid == 1))
    n_2stop = int(np.sum(phase_res.optimal_stops_grid == 2))
    total_pts = phase_res.optimal_stops_grid.size
    print(f"  Phase Partitioning:      1-Stop: {n_1stop}/{total_pts} ({n_1stop/total_pts:.1%}) | "
          f"2-Stop: {n_2stop}/{total_pts} ({n_2stop/total_pts:.1%})")

    has_crossover = np.min(phase_res.delta_grid) < 0 < np.max(phase_res.delta_grid)
    status = "Active Decision Boundary exists in parameter space." if has_crossover else "Single strategy dominates entire grid."
    print(f"  Boundary Status:         {status}")
    print("=" * 95 + "\n")


def print_scenario_table(result) -> None:
    """Print detailed impact table for a single counterfactual scenario."""
    print("\n" + "=" * 98)
    print(f"  WHAT-IF SCENARIO EVALUATION: {result.scenario.name.upper()} ({result.circuit_name})")
    print("=" * 98)
    print(f"  Description:       {result.scenario.description}")
    print(f"  Research Question: \"{result.scenario.research_question}\"")
    print("-" * 98)
    print(f"  Baseline Optimal:  {result.baseline_optimal_strategy.description:<35} Total Time: {result.formatted_baseline_time}")
    print(f"  Scenario Optimal:  {result.scenario_optimal_strategy.description:<35} Total Time: {result.formatted_scenario_time}")
    pivot_str = "YES (PIVOT RECOMMENDED)" if result.strategy_pivoted else "NO (STRATEGY FULLY ROBUST)"
    print(f"  Strategy Pivoted:  {pivot_str}")
    print(f"  Strategic Regret:  {result.formatted_regret} (penalty if baseline strategy is kept)")
    print("-" * 98)
    print(f"  Engineering Insight: {result.summary_insight}")
    print("=" * 98)
    print(f"{'Strategy Name':<32} {'Stops':<6} {'Base Time':<12} {'Scenario Time':<16} {'Impact':<10} {'Gap':<10}")
    print("-" * 98)
    for oc in result.strategy_outcomes:
        gap_str = "WINNER" if oc.gap_to_winner == 0 else f"+{oc.gap_to_winner:.2f}s"
        print(f"{oc.strategy_name:<32} {oc.stops:<6} {oc.baseline_time:<12.2f} {oc.formatted_scenario_time:<16} {oc.formatted_impact:<10} {gap_str:<10}")
    print("=" * 98 + "\n")


def print_all_scenarios_matrix(results: dict, circuit_name: str) -> None:
    """Print comparative matrix of all preset scenarios."""
    print("\n" + "=" * 105)
    print(f"  PHASE 6 SCENARIO ANALYSIS MATRIX: {circuit_name.upper()}")
    print("=" * 105)
    print(f"{'Scenario ID':<20} {'Scenario Name':<32} {'Pivot?':<8} {'Regret':<10} {'Scenario Winner':<32}")
    print("-" * 105)
    for scen_id, r in results.items():
        pivot_tag = "YES" if r.strategy_pivoted else "NO"
        print(f"{scen_id:<20} {r.scenario.name[:30]:<32} {pivot_tag:<8} {r.formatted_regret:<10} {r.scenario_optimal_strategy.description:<32}")
    print("=" * 105 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F1 Race Strategy Simulator - Evaluate and compare pit stop strategies."
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="bahrain",
        choices=list(CIRCUIT_PRESETS.keys()),
        help="Circuit preset to simulate (default: bahrain). Choices: bahrain, barcelona, monza.",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        action="append",
        help="Custom strategy in shorthand format (e.g. -s 'S15-M21-S21' -s 'M26-H31').",
    )
    parser.add_argument(
        "--era",
        type=str,
        default="2024",
        choices=["2024", "2026"],
        help="Formula 1 technical regulations era (default: 2024). Choices: 2024, 2026.",
    )
    parser.add_argument(
        "--sc-lap",
        type=int,
        action="append",
        metavar="LAP",
        help="Simulate a Safety Car deployment on specified lap(s) (e.g. --sc-lap 15 --sc-lap 16).",
    )
    parser.add_argument(
        "--vsc-lap",
        type=int,
        action="append",
        metavar="LAP",
        help="Simulate a Virtual Safety Car deployment on specified lap(s).",
    )
    parser.add_argument(
        "--laps",
        action="store_true",
        help="Print detailed lap-by-lap breakdown for the winning strategy (deterministic only).",
    )
    parser.add_argument(
        "--laps-all",
        action="store_true",
        help="Print detailed lap-by-lap breakdown for all simulated strategies.",
    )
    parser.add_argument(
        "--mc",
        type=int,
        metavar="ITERATIONS",
        help="Run Monte Carlo simulation with the specified number of iterations.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots for Monte Carlo results (KDE distribution and Win Probability).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Strategy Optimization Engine to automatically find the mathematically optimal strategy.",
    )
    parser.add_argument(
        "--max-stops",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Maximum number of pit stops allowed during strategy optimization (default: 2).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="time",
        choices=["time", "expected", "risk"],
        help="Optimization objective function: 'time' (deterministic), 'expected' (Monte Carlo mean), or 'risk' (P95).",
    )
    parser.add_argument(
        "--pit-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display allowable tactical pit windows for the winning strategy (default: True).",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run parameter sensitivity analysis to study strategy response curves.",
    )
    parser.add_argument(
        "--param",
        type=str,
        default="deg",
        choices=["deg", "pit-loss", "soft-delta", "track-evo", "fuel-penalty"],
        help="Parameter to sweep in sensitivity analysis (default: deg).",
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs="+",
        metavar=("MIN", "MAX"),
        help="Range of parameter sweep: MIN MAX [STEPS] (e.g. --range 0.6 2.0 15).",
    )
    parser.add_argument(
        "--crossover",
        action="store_true",
        help="Compute exact critical tipping point (root finding) where optimal strategy flips.",
    )
    parser.add_argument(
        "--phase-map",
        action="store_true",
        help="Generate a 2D decision boundary phase diagram across Pit Loss and Tyre Degradation.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run Phase 5 historical validation and backtesting against real Grand Prix outcomes.",
    )
    parser.add_argument(
        "--fit-params",
        action="store_true",
        help="Estimate empirical tyre degradation and fuel parameters from real telemetry.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Phase 6 interactive web dashboard in your browser.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run counterfactual what-if scenario (e.g. slow_stop, botched_stop, high_deg, low_deg, etc.).",
    )
    parser.add_argument(
        "--run-all-scenarios",
        action="store_true",
        help="Execute full catalog of counterfactual scenarios and output comparative impact matrix.",
    )

    args = parser.parse_args()

    # Launch Phase 6 Interactive Web Dashboard
    if args.dashboard:
        import run_dashboard
        run_dashboard.main()
        return

    # Check for Phase 5 Validation / Empirical Estimation request
    if args.validate or args.fit_params:
        from src.data_pipeline import load_historical_data
        from src.estimation import estimate_empirical_parameters
        from src.validation import RaceBacktester, DiscrepancyAnalyzer
        from run_validation import (
            print_empirical_params_table,
            print_backtest_table,
            print_discrepancy_diagnostic,
            DEFAULT_WINNERS,
        )

        data = load_historical_data(args.circuit)
        params = estimate_empirical_parameters(data)

        if args.fit_params or not args.validate:
            print_empirical_params_table(params, data.circuit_name)

        if args.validate:
            tester = RaceBacktester(data, empirical_params=params)
            default_d_num, default_d_name = DEFAULT_WINNERS[args.circuit]
            metrics = tester.backtest_driver(default_d_num, driver_name=default_d_name)
            print_backtest_table(metrics)
            diag = DiscrepancyAnalyzer.analyze_circuit(args.circuit)
            print_discrepancy_diagnostic(diag)
        return

    from src.config import get_circuit_compounds
    compounds = get_circuit_compounds(args.circuit)
    model = create_race_model(args.circuit, era=args.era)
    circuit_name = model.circuit.name
    total_laps = model.circuit.total_laps

    # Check for Phase 6 Scenario Analysis request
    if args.scenario or args.run_all_scenarios:
        from src.scenarios import ScenarioEngine, PRESET_SCENARIOS

        engine = ScenarioEngine(args.circuit)
        if args.run_all_scenarios:
            print(f"\nEvaluating complete Phase 6 scenario matrix for {circuit_name}...")
            all_res = engine.run_all_presets()
            print_all_scenarios_matrix(all_res, circuit_name)
            return

        scen_key = args.scenario.lower()
        if scen_key not in PRESET_SCENARIOS:
            print(f"Error: Scenario '{args.scenario}' not found in presets.", file=sys.stderr)
            print(f"Available presets: {', '.join(PRESET_SCENARIOS.keys())}", file=sys.stderr)
            sys.exit(1)

        print(f"\nEvaluating scenario '{scen_key}' for {circuit_name}...")
        scen_res = engine.evaluate_scenario(PRESET_SCENARIOS[scen_key])
        print_scenario_table(scen_res)
        return

    # Build safety car lap map if provided
    safety_car_laps = {}
    if args.sc_lap:
        for lap in args.sc_lap:
            safety_car_laps[lap] = "safety_car"
    if args.vsc_lap:
        for lap in args.vsc_lap:
            safety_car_laps[lap] = "vsc"

    # Check for 2D Phase Map request
    if args.phase_map:
        from src.sensitivity import PhaseDiagram2D, SensitivityParameter
        from src.visualization import plot_2d_phase_diagram

        print(f"\nComputing 2D Decision Phase Map for {circuit_name} (Pit Loss vs Tyre Degradation)...")
        p2d = PhaseDiagram2D(model, compounds)
        phase_res = p2d.compute_phase_map(
            param_x=SensitivityParameter.PIT_LOSS,
            x_range=(16.0, 32.0),
            param_y=SensitivityParameter.TYRE_DEG_MULTIPLIER,
            y_range=(0.6, 2.2),
            resolution=(25, 25),
        )
        print_phase_map_summary(phase_res)
        if args.plot:
            save_file = f"phase_map_{args.circuit}.png"
            plot_2d_phase_diagram(phase_res, save_path=save_file)
        return

    # Check for 1D Sensitivity Analysis / Crossover request
    if args.sensitivity or args.crossover:
        from src.sensitivity import (
            ParameterSweep1D,
            CrossoverFinder,
            SensitivityParameter,
            PhaseDiagram2D,
        )
        from src.visualization import plot_1d_sensitivity, plot_sensitivity_dashboard

        param_map = {
            "deg": SensitivityParameter.TYRE_DEG_MULTIPLIER,
            "pit-loss": SensitivityParameter.PIT_LOSS,
            "soft-delta": SensitivityParameter.COMPOUND_DELTA_SOFT,
            "track-evo": SensitivityParameter.TRACK_EVOLUTION,
            "fuel-penalty": SensitivityParameter.FUEL_PENALTY,
        }
        chosen_param = param_map[args.param]

        # Determine sweep range
        if args.range:
            if len(args.range) == 2:
                r_min, r_max, n_steps = args.range[0], args.range[1], 25
            elif len(args.range) >= 3:
                r_min, r_max, n_steps = args.range[0], args.range[1], int(args.range[2])
            else:
                r_min, r_max, n_steps = 0.6, 2.2, 25
        else:
            default_ranges = {
                SensitivityParameter.TYRE_DEG_MULTIPLIER: (0.6, 2.2, 25),
                SensitivityParameter.PIT_LOSS: (16.0, 32.0, 25),
                SensitivityParameter.COMPOUND_DELTA_SOFT: (-1.2, 0.0, 25),
                SensitivityParameter.TRACK_EVOLUTION: (0.0, 1.0, 21),
                SensitivityParameter.FUEL_PENALTY: (0.020, 0.045, 21),
            }
            r_min, r_max, n_steps = default_ranges[chosen_param]

        if args.crossover:
            print(f"\nComputing exact critical tipping point for {circuit_name} on {chosen_param.value}...")
            cf = CrossoverFinder(model, compounds)
            c_pt = cf.find_1_vs_2_stop_crossover(chosen_param, (r_min, r_max))
            print_crossover_table(c_pt, circuit_name)

        if args.sensitivity:
            print(f"\nRunning 1D Sensitivity Sweep on {chosen_param.value} [{r_min:.2f} to {r_max:.2f}] for {circuit_name}...")
            values = np.linspace(r_min, r_max, n_steps)
            sweeper = ParameterSweep1D(model, compounds)

            if args.strategy:
                # Custom strategies provided via CLI
                strategies = [parse_strategy_string(s, compounds) for s in args.strategy]
                sweep_res = sweeper.sweep_strategies(strategies, chosen_param, values)
            else:
                # Re-optimization sweep comparing optimal 1-stop vs optimal 2-stop
                sweep_res = sweeper.sweep_reoptimization(chosen_param, values, max_stops=args.max_stops)

            print_sensitivity_sweep_table(sweep_res, circuit_name)

            if args.plot:
                if not args.strategy and chosen_param == SensitivityParameter.TYRE_DEG_MULTIPLIER:
                    # Generate full 4-panel dashboard!
                    pit_vals = np.linspace(16.0, 32.0, 25)
                    sweep_pit = sweeper.sweep_reoptimization(SensitivityParameter.PIT_LOSS, pit_vals, max_stops=args.max_stops)
                    p2d = PhaseDiagram2D(model, compounds)
                    phase_res = p2d.compute_phase_map(resolution=(20, 20))
                    plot_sensitivity_dashboard(sweep_res, sweep_pit, phase_res, save_path="sensitivity_dashboard.png")
                else:
                    plot_1d_sensitivity(sweep_res, save_path="sensitivity_curve.png")
        return
    # Check for strategy optimization request
    if args.optimize:

        from src.optimization import StrategyOptimizer, OptimizationObjective

        obj_map = {
            "time": OptimizationObjective.DETERMINISTIC_TIME,
            "expected": OptimizationObjective.EXPECTED_TIME,
            "risk": OptimizationObjective.MIN_RISK_P95,
        }
        optimizer = StrategyOptimizer(model, compounds)
        mc_iters = args.mc if args.mc else 500

        print(f"\nRunning Strategy Optimization Engine for {circuit_name} (Era: {args.era}, Max Stops: {args.max_stops})...")
        opt_res = optimizer.optimize(
            max_stops=args.max_stops,
            objective=obj_map[args.objective],
            mc_iterations=mc_iters,
        )
        print_optimization_table(opt_res, show_pit_windows=args.pit_windows)


        if args.laps:
            winning_result = simulate_race(opt_res.optimal_strategy, model)
            print_lap_telemetry(winning_result)
        return

    # Determine strategies to run

    if args.strategy:
        strategies = []
        for s_str in args.strategy:
            try:
                strat = parse_strategy_string(s_str, compounds)
                is_valid, msg = strat.validate(total_laps)
                if not is_valid:
                    print(f"Error in strategy '{s_str}': {msg}", file=sys.stderr)
                    sys.exit(1)
                strategies.append(strat)
            except Exception as e:
                print(f"Failed to parse strategy '{s_str}': {e}", file=sys.stderr)
                sys.exit(1)
    else:
        strategies = get_default_strategies(args.circuit, compounds)

    if args.mc:
        from src.stochastic import StochasticParameters
        from src.montecarlo import simulate_monte_carlo
        
        params = StochasticParameters()
        print(f"\nRunning {args.mc} Monte Carlo simulations per strategy (Era: {args.era})...")
        
        results = [simulate_monte_carlo(strat, model, params, n_iterations=args.mc) for strat in strategies]
        
        print_comparison_table_mc(results, circuit_name, total_laps)
        
        if args.plot:
            from src.visualization import (
                plot_monte_carlo_dashboard,
                plot_strategy_boxplots,
                plot_strategy_cdf,
                plot_strategy_distributions,
                plot_strategy_histograms,
                plot_win_probability_matrix,
            )
            plot_strategy_distributions(results, save_path="mc_distributions.png")
            plot_strategy_histograms(results, save_path="mc_histograms.png")
            plot_strategy_cdf(results, save_path="mc_cdf.png")
            plot_strategy_boxplots(results, save_path="mc_boxplots.png")
            plot_win_probability_matrix(results, save_path="mc_win_matrix.png")
            plot_monte_carlo_dashboard(results, save_path="mc_dashboard.png")
    else:
        # Run deterministic simulations
        results = [
            simulate_race(strat, model, safety_car_laps=safety_car_laps or None)
            for strat in strategies
        ]
        era_title = f"{circuit_name} [{args.era} Regs]" if args.era != "2024" else circuit_name
        print_comparison_table(results, era_title, total_laps)

        if args.laps_all:
            for res in results:
                print_lap_telemetry(res)
        elif args.laps:
            winning_result = min(results, key=lambda r: r.total_time)
            print_lap_telemetry(winning_result)


if __name__ == "__main__":
    main()
