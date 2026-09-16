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
        action="store_true",
        help="Calculate and display allowable tactical pit windows for the winning strategy.",
    )


    args = parser.parse_args()

    from src.config import get_circuit_compounds
    compounds = get_circuit_compounds(args.circuit)
    model = create_race_model(args.circuit, era=args.era)
    circuit_name = model.circuit.name
    total_laps = model.circuit.total_laps

    # Build safety car lap map if provided
    safety_car_laps = {}
    if args.sc_lap:
        for lap in args.sc_lap:
            safety_car_laps[lap] = "safety_car"
    if args.vsc_lap:
        for lap in args.vsc_lap:
            safety_car_laps[lap] = "vsc"

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
        print_optimization_table(opt_res, show_pit_windows=args.pit_windows or True)

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
