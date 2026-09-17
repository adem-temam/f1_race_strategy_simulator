"""Interactive CLI tool for Phase 5: Historical Data Validation & Empirical Parameter Estimation."""

import argparse
from pathlib import Path
import sys

from src.data_pipeline import load_historical_data
from src.estimation import estimate_empirical_parameters
from src.validation import DiscrepancyAnalyzer, RaceBacktester
from src.visualization import (
    plot_empirical_degradation_fit,
    plot_lap_residuals,
    plot_strategy_backtest_gantt,
)

DEFAULT_WINNERS = {
    "bahrain": (1, "Max Verstappen"),
    "barcelona": (1, "Max Verstappen"),
    "monza": (16, "Charles Leclerc"),
}


def print_empirical_params_table(params, circuit_name: str) -> None:
    """Print ASCII telemetry calibration scorecard."""
    print("\n" + "=" * 95)
    print(f"  EMPIRICAL TELEMETRY PARAMETER FIT: {circuit_name.upper()} ({params.year})")
    print("=" * 95)
    print(f"  Circuit Baseline Pace:   {params.base_lap_time:.2f} s")
    print(f"  Pit Lane Loss Duration:  {params.pit_loss_mean:.2f} s (std: {params.pit_loss_std:.2f} s)")
    print(f"  Lap Noise Dispersion:    {params.lap_noise_std:.3f} s")
    print(f"  Fuel Mass Sensitivity:   {params.fuel_penalty_per_kg:.3f} s/kg")
    print("-" * 95)
    print(
        f"  {'Compound':<10} {'Alpha (s/lap)':<16} {'Beta (s/lap^2)':<16} "
        f"{'Pace Delta':<14} {'Laps (N)':<10} {'Fit R^2':<10} {'RMSE':<8}"
    )
    print("  " + "-" * 91)
    for c_name, cp in params.compounds.items():
        delta_str = f"{cp.base_delta:+.2f}s" if cp.base_delta != 0 else "0.00s (Ref)"
        print(
            f"  {c_name:<10} {cp.alpha:<16.4f} {cp.beta:<16.6f} "
            f"{delta_str:<14} {cp.sample_count:<10} {cp.r_squared:<10.2f} {cp.rmse:<8.3f}"
        )
    print("=" * 95 + "\n")


def print_backtest_table(m) -> None:
    """Print backtest comparison against historical driver outcome."""
    print("\n" + "=" * 95)
    print(f"  HISTORICAL STRATEGY BACKTEST: {m.circuit_name.upper()} — {m.driver_name}")
    print("=" * 95)
    print(f"  Actual Strategy:         {m.actual_strategy_desc} ({m.actual_stops} stops)")
    print(f"  Model-Optimal Strategy:  {m.optimal_strategy_desc} ({m.optimal_stops} stops)")
    print(f"  Stop Count Agreement:    {'MATCH [PASS]' if m.stop_count_matches else 'DISCREPANCY'}")
    print(f"  Stint Length MAE:        {m.stint_length_mae:.2f} laps")
    print(f"  Clean Lap Pace RMSE:     {m.clean_lap_rmse:.3f} s")
    print(f"  Pit Window Hit Rate:     {m.pit_windows_covered}/{m.total_pit_windows} ({m.pit_window_hit_rate:.0%})")
    print("-" * 95)
    print(f"  [Physics Model Fidelity — Simulating Actual Strategy]")
    print(f"    Sim Actual Race Time:  {m.formatted_sim_actual_time} ({m.sim_actual_strategy_time:.2f}s)")
    print(f"    Recorded Clock Time:   {m.formatted_actual_time} ({m.actual_time:.2f}s)")
    print(f"    Physics Delta Error:   {m.sim_actual_delta:+.2f}s ({m.sim_actual_error_pct:.2f}%)")
    print(f"  [Strategy Optimization Evaluation]")
    print(f"    Optimal Strategy Time: {m.formatted_predicted_time} ({m.predicted_time:.2f}s)")
    print(f"    Strategic Gain Delta:  {m.strategic_gain_delta:+.2f}s ({'Faster' if m.strategic_gain_delta > 0 else 'Slower'} than actual strategy)")
    print("=" * 95 + "\n")


def print_discrepancy_diagnostic(diag) -> None:
    """Print discrepancy case study."""
    print("\n" + "#" * 95)
    print(f"  DISCREPANCY CASE STUDY: {diag.case_title}")
    print("#" * 95)
    print(f"\n* Observed Outcome:\n  {diag.observed_phenomenon}")
    print(f"\n* Pure Theoretical Prediction:\n  {diag.theoretical_prediction}")
    print("\n* Root Cause Physical & Regulatory Factors:")
    for idx, f in enumerate(diag.root_cause_factors, 1):
        print(f"  {idx}. {f}")
    print(f"\n* Engineering Solution / Model Enhancement:\n  {diag.engineering_solution}")
    print("#" * 95 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F1 Race Strategy Simulator — Phase 5 Validation & Telemetry Calibration."
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="bahrain",
        choices=["bahrain", "barcelona", "monza", "all"],
        help="Target Grand Prix circuit (default: bahrain).",
    )
    parser.add_argument(
        "--driver",
        type=int,
        default=None,
        help="Driver car number to backtest (e.g. 1 for Verstappen, 16 for Leclerc).",
    )
    parser.add_argument(
        "--fit-params",
        action="store_true",
        help="Estimate empirical tyre degradation, pit loss, and noise parameters from telemetry.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Execute strategy backtest against historical race outcome.",
    )
    parser.add_argument(
        "--discrepancy",
        action="store_true",
        help="Run causal discrepancy diagnostic explaining theoretical vs reality gaps.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate empirical wear curves, lap residual charts, and Gantt timelines.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save generated visual plots (default: current directory).",
    )

    args = parser.parse_args()

    # If no specific mode chosen, default to all three analytical views
    if not (args.fit_params or args.backtest or args.discrepancy or args.plot):
        args.fit_params = True
        args.backtest = True
        args.discrepancy = True

    circuits = ["bahrain", "barcelona", "monza"] if args.circuit == "all" else [args.circuit]
    all_metrics = []

    for c_key in circuits:
        data = load_historical_data(c_key)
        params = estimate_empirical_parameters(data)

        if args.fit_params:
            print_empirical_params_table(params, data.circuit_name)

        if args.backtest or args.plot:
            tester = RaceBacktester(data, empirical_params=params)
            default_d_num, default_d_name = DEFAULT_WINNERS[c_key]
            d_num = args.driver if args.driver else default_d_num
            d_name = data.drivers.get(d_num, default_d_name)

            metrics = tester.backtest_driver(d_num, driver_name=d_name)
            all_metrics.append(metrics)

            if args.backtest:
                print_backtest_table(metrics)

        if args.discrepancy:
            diag = DiscrepancyAnalyzer.analyze_circuit(c_key)
            print_discrepancy_diagnostic(diag)

        if args.plot:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1. Empirical wear curve
            wear_plot_file = out_dir / f"empirical_wear_{c_key}.png"
            clean_laps = [l for l in data.laps if l.is_clean]
            plot_empirical_degradation_fit(params, clean_laps, save_path=wear_plot_file)

            # 2. Residual error trace
            d_clean_laps = data.get_driver_laps(d_num, clean_only=True)
            if d_clean_laps:
                from src.simulation import simulate_race
                from src.strategies import Stint, Strategy

                actual_stints = data.get_driver_stints(d_num)
                sim_stints = [
                    Stint(
                        compound=params.to_tyre_compounds().get(s.compound.capitalize(), list(params.to_tyre_compounds().values())[0]),
                        laps=s.stint_length,
                    )
                    for s in actual_stints
                ]
                sim_res = simulate_race(Strategy(sim_stints), params.to_race_model(data.total_laps))
                sim_map = {rec.lap_number: rec.lap_time for rec in sim_res.lap_records}

                lap_nums = [l.lap_number for l in d_clean_laps if l.lap_number in sim_map]
                actual_times = [l.lap_duration for l in d_clean_laps if l.lap_number in sim_map]
                sim_times = [sim_map[n] for n in lap_nums]

                res_plot_file = out_dir / f"lap_residuals_{c_key}_{d_name.replace(' ', '_').lower()}.png"
                plot_lap_residuals(
                    lap_numbers=lap_nums,
                    sim_laps=sim_times,
                    actual_laps=actual_times,
                    circuit_name=data.circuit_name,
                    driver_name=d_name,
                    save_path=res_plot_file,
                )

    if args.plot and all_metrics:
        gantt_file = Path(args.output_dir) / "strategy_backtest_gantt.png"
        plot_strategy_backtest_gantt(all_metrics, save_path=gantt_file)


if __name__ == "__main__":
    main()
