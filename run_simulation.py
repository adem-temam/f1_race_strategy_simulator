#!/usr/bin/env python3
"""Interactive CLI runner for F1 Race Strategy Simulator."""

import argparse
import sys
from typing import Optional

from src.config import CIRCUIT_PRESETS, create_race_model, get_default_compounds
from src.simulation import simulate_race
from src.strategies import Stint, Strategy
from src.tyres import TyreCompound


def parse_strategy_string(strategy_str: str, compounds: dict[str, TyreCompound]) -> Strategy:
    """Parse a shorthand string like 'S15-M21-S21' or 'M26-H31' into a Strategy."""
    tokens = strategy_str.strip().split("-")
    stints = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        compound_char = token[0].upper()
        lap_str = token[1:]
        if not lap_str.isdigit():
            raise ValueError(f"Invalid stint token '{token}'. Expected format like 'S15' or 'M26'.")
        
        laps = int(lap_str)
        char_map = {"S": "Soft", "M": "Medium", "H": "Hard"}
        if compound_char not in char_map:
            raise ValueError(f"Unknown compound code '{compound_char}'. Use S (Soft), M (Medium), or H (Hard).")
        
        comp_name = char_map[compound_char]
        stints.append(Stint(compound=compounds[comp_name], laps=laps))
    
    return Strategy(stints=stints, name=strategy_str)


def get_default_strategies(circuit_key: str, compounds: dict[str, TyreCompound]) -> list[Strategy]:
    """Return realistic competing strategies for a given circuit."""
    s = compounds["Soft"]
    m = compounds["Medium"]
    h = compounds["Hard"]

    if circuit_key == "bahrain":
        # 57 laps
        return [
            Strategy([Stint(m, 26), Stint(h, 31)], name="1-Stop (M-H)"),
            Strategy([Stint(s, 15), Stint(m, 21), Stint(s, 21)], name="2-Stop (S-M-S)"),
            Strategy([Stint(s, 14), Stint(h, 22), Stint(m, 21)], name="2-Stop (S-H-M)"),
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
        "--laps",
        action="store_true",
        help="Print detailed lap-by-lap breakdown for the winning strategy.",
    )
    parser.add_argument(
        "--laps-all",
        action="store_true",
        help="Print detailed lap-by-lap breakdown for all simulated strategies.",
    )

    args = parser.parse_args()

    compounds = get_default_compounds()
    model = create_race_model(args.circuit)
    circuit_name = model.circuit.name
    total_laps = model.circuit.total_laps

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

    # Run simulations
    results = [simulate_race(strat, model) for strat in strategies]

    # Print summary table
    print_comparison_table(results, circuit_name, total_laps)

    # Optional lap telemetry
    if args.laps_all:
        for res in results:
            print_lap_telemetry(res)
    elif args.laps:
        winning_result = min(results, key=lambda r: r.total_time)
        print_lap_telemetry(winning_result)


if __name__ == "__main__":
    main()
