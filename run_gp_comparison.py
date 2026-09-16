"""Script to execute 5 stochastic simulation trials for all strategies across Bahrain, Barcelona, and Monza.
Compares simulated outputs with real-world Formula 1 races (2024-2026).
"""

import numpy as np
from src.config import create_race_model, get_circuit_compounds
from run_simulation import get_default_strategies
from src.montecarlo import simulate_monte_carlo
from src.simulation import simulate_race
from src.stochastic import StochasticParameters
from src.strategies import Stint, Strategy


def format_seconds(seconds: float) -> str:
    hours = int(seconds // 3600)
    rem = seconds % 3600
    minutes = int(rem // 60)
    secs = rem % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:06.3f}s"
    return f"{minutes:02d}m {secs:06.3f}s"


def get_all_test_strategies(circuit_key: str, compounds: dict) -> list[Strategy]:
    """Combine preset strategies with actual real-world race strategies from 2024."""
    presets = get_default_strategies(circuit_key, compounds)
    s = compounds["Soft"]
    m = compounds["Medium"]
    h = compounds["Hard"]

    if circuit_key == "bahrain":
        # Total laps: 57
        real_strats = [
            Strategy([Stint(s, 17), Stint(h, 20), Stint(s, 20)], name="2024 Winner (Verstappen S-H-S)"),
            Strategy([Stint(s, 14), Stint(h, 21), Stint(h, 22)], name="2024 Ferrari (Sainz P3 S-H-H)"),
            Strategy([Stint(s, 12), Stint(h, 24), Stint(s, 21)], name="2024 Perez P2 (S-H-S)"),
        ]
        return presets + real_strats

    elif circuit_key == "barcelona":
        # Total laps: 66
        real_strats = [
            Strategy([Stint(s, 17), Stint(m, 27), Stint(s, 22)], name="2024 Winner (Verstappen S-M-S)"),
            Strategy([Stint(s, 23), Stint(m, 24), Stint(s, 19)], name="2024 Norris P2 (S-M-S)"),
            Strategy([Stint(s, 15), Stint(m, 21), Stint(h, 30)], name="2024 Russell P4 (S-M-H)"),
        ]
        return presets + real_strats

    elif circuit_key == "monza":
        # Total laps: 53
        real_strats = [
            Strategy([Stint(m, 15), Stint(h, 38)], name="2024 Winner (Leclerc 1-Stop M-H)"),
            Strategy([Stint(m, 16), Stint(h, 22), Stint(h, 15)], name="2024 McLaren (Piastri P2 2-Stop M-H-H)"),
            Strategy([Stint(m, 19), Stint(h, 34)], name="2024 Ferrari (Sainz P4 1-Stop M-H)"),
        ]
        return presets + real_strats

    return presets


def run_comparative_benchmark(era: str = "2024"):
    circuits = ["bahrain", "barcelona", "monza"]
    stochastic_params = StochasticParameters()

    all_data = {}

    for c_key in circuits:
        model = create_race_model(c_key, era=era)
        compounds = get_circuit_compounds(c_key)
        strategies = get_all_test_strategies(c_key, compounds)
        c_name = model.circuit.name
        total_laps = model.circuit.total_laps

        circuit_results = []

        for strat in strategies:
            # 1. Deterministic baseline
            det_res = simulate_race(strat, model)

            # 2. Run 5 distinct stochastic race trials (seeds 1 to 5)
            trial_times = []
            for seed in [101, 202, 303, 404, 505]:
                mc_res = simulate_monte_carlo(strat, model, stochastic_params, n_iterations=1, seed=seed)
                trial_times.append(float(mc_res.race_times[0]))

            # 3. Also run 5,000 iterations for distribution statistics
            full_mc = simulate_monte_carlo(strat, model, stochastic_params, n_iterations=5000, seed=42)

            strat_summary = {
                "name": strat.name,
                "description": strat.description,
                "stops": strat.num_stops,
                "deterministic_time": det_res.total_time,
                "formatted_deterministic": det_res.formatted_total_time,
                "trial_times": trial_times,
                "formatted_trials": [format_seconds(t) for t in trial_times],
                "trial_mean": float(np.mean(trial_times)),
                "trial_std": float(np.std(trial_times)),
                "trial_min": float(np.min(trial_times)),
                "trial_max": float(np.max(trial_times)),
                "mc_mean": full_mc.mean_time,
                "mc_std": full_mc.std_dev,
                "mc_p05": full_mc.p05_time,
                "mc_p95": full_mc.p95_time,
                "pure_racing_time": det_res.pure_racing_time,
                "pit_loss": det_res.total_pit_loss,
                "avg_lap": det_res.average_lap_time,
            }
            circuit_results.append(strat_summary)

        # Sort by deterministic time
        circuit_results.sort(key=lambda x: x["deterministic_time"])
        all_data[c_key] = {
            "name": c_name,
            "total_laps": total_laps,
            "results": circuit_results,
        }

    return all_data


if __name__ == "__main__":
    for era in ["2024", "2026"]:
        print("\n" + "#" * 110)
        print(f"       BENCHMARK EVALUATION: {era} FORMULA 1 TECHNICAL REGULATIONS")
        print("#" * 110)
        benchmark_data = run_comparative_benchmark(era=era)

        for c_key, c_info in benchmark_data.items():
            print("\n" + "=" * 110)
            print(f"CIRCUIT: {c_info['name']} ({c_info['total_laps']} Laps) [{era} Regs]")
            print("=" * 110)
            print(
                f"{'Pos':<4} {'Strategy Name':<34} {'Stops':<6} {'Deterministic':<16} "
                f"{'5-Run Mean':<16} {'Std Dev':<10} {'Min (Best)':<16} {'Max (Worst)':<16}"
            )
            print("-" * 110)
            leader_time = c_info["results"][0]["deterministic_time"]
            for pos, r in enumerate(c_info["results"], 1):
                delta = "LEADER" if pos == 1 else f"+{r['deterministic_time'] - leader_time:.2f}s"
                print(
                    f"{pos:<4} {r['name']:<34} {r['stops']:<6} {format_seconds(r['deterministic_time']):<16} "
                    f"{format_seconds(r['trial_mean']):<16} {r['trial_std']:<10.2f} "
                    f"{format_seconds(r['trial_min']):<16} {format_seconds(r['trial_max']):<16}"
                )

            print(f"\nIndividual 5 Stochastic Runs per strategy ({era} Regs):")
            for r in c_info["results"]:
                print(f"  * {r['name']}:")
                for idx, t_str in enumerate(r["formatted_trials"], 1):
                    raw = r["trial_times"][idx - 1]
                    print(f"      Run {idx}: {t_str} ({raw:.2f}s)")

