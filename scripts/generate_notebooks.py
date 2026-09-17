"""Script to generate all demonstration Jupyter notebooks using nbformat."""

import sys
from pathlib import Path
import nbformat as nbf


def make_notebook(title: str, cells_data: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.14"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "title": title,
    }
    for cell_type, content in cells_data:
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(content.strip()))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(content.strip()))
    return nb


def generate_all_notebooks(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Notebook 01: Physics & Synthetic Model Exploration
    # ---------------------------------------------------------
    nb1_cells = [
        ("markdown", """# Formula 1 Race Strategy Simulator
## Notebook 01: Physics & Synthetic Model Exploration

This notebook explores the analytical physics engine developed in **Phase 1**:
1. **Lap-Time Decomposition**: Breaking down a lap into base track pace, fuel burn, tyre degradation, traffic, and pit loss.
2. **Non-Linear Tyre Degradation**: Modeling quadratic wear $\\alpha a + \\beta a^2$ and exponential thermal cliff $\\gamma e^{\\kappa (a - L_{cliff})}$.
3. **Fuel Burn Dynamics**: Mass depletion $m_{fuel}(l) = m_0 - \\beta_{burn} \\cdot l$ and acceleration penalty.
4. **Deterministic Race Simulation**: Simulating and comparing 1-stop vs 2-stop strategies at the Bahrain Grand Prix.
"""),
        ("code", """import sys
from pathlib import Path

# Add repository root to Python path
repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import CircuitConfig, TyreCompound, FuelModel, PitStopModel, RaceModel
from src.strategies import Strategy, Stint
from src.simulation import simulate_race
from src.config import BAHRAIN_CONFIG, DEFAULT_COMPOUNDS

sns.set_theme(style="darkgrid")
print("Environment and modules successfully initialized.")
"""),
        ("markdown", """### 1. Non-Linear Tyre Degradation Mechanics

Tyre degradation is modeled with two distinct regimes:
1. **Progressive Structural & Thermal Wear**:
   $$\\Delta T_{wear}(a) = \\alpha a + \\beta a^2$$
2. **Thermal Degradation Cliff**:
   $$\\Delta T_{cliff}(a) = \\gamma \\exp\\big(\\kappa (a - L_{cliff})\\big) \\quad \\text{for } a > L_{cliff}$$

Let us plot the degradation curves across Soft, Medium, and Hard compounds over 40 laps.
"""),
        ("code", """laps = np.arange(1, 41)
fig, ax = plt.subplots(figsize=(10, 5))

for compound_name, compound in DEFAULT_COMPOUNDS.items():
    wear = [compound.degradation_at_lap(a) for a in laps]
    ax.plot(laps, wear, label=f"{compound.name} (base {compound.base_time_delta:+.2f}s)", linewidth=2.5)

ax.set_title("Non-Linear Tyre Compound Degradation vs Tyre Age", fontsize=14, fontweight="bold")
ax.set_xlabel("Tyre Age (Laps)", fontsize=12)
ax.set_ylabel("Degradation Lap Time Penalty (s)", fontsize=12)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
"""),
        ("markdown", """### 2. Fuel Mass Depletion & Weight Advantage

As fuel burns at $\\sim 1.84\\text{ kg/lap}$, the vehicle sheds mass:
$$m_{fuel}(l) = m_0 - \\beta_{burn} \\cdot (l - 1), \\quad \\Delta T_{fuel}(l) = \\lambda_{fuel} \\cdot m_{fuel}(l)$$

This weight reduction provides a significant lap-time gain (over $3.0\\text{s}$ across a Grand Prix).
"""),
        ("code", """model = RaceModel(BAHRAIN_CONFIG)
total_laps = model.circuit.total_laps
lap_indices = np.arange(1, total_laps + 1)

fuel_masses = [model.fuel_model.fuel_mass_at_lap(l) for l in lap_indices]
fuel_penalties = [model.fuel_model.lap_time_penalty(m) for m in fuel_masses]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

ax1.plot(lap_indices, fuel_masses, color="crimson", linewidth=2)
ax1.set_title("Fuel Mass Depletion Over Race Distance", fontweight="bold")
ax1.set_xlabel("Lap Number")
ax1.set_ylabel("Remaining Fuel (kg)")

ax2.plot(lap_indices, fuel_penalties, color="navy", linewidth=2)
ax2.set_title("Fuel Lap-Time Penalty Evolution", fontweight="bold")
ax2.set_xlabel("Lap Number")
ax2.set_ylabel("Time Penalty (s)")

plt.tight_layout()
plt.show()
"""),
        ("markdown", """### 3. Deterministic Race Simulation: 1-Stop vs 2-Stop

We compare two standard tactical approaches for the 57-lap Bahrain Grand Prix:
* **Strategy A (1-Stop)**: Soft (Laps 1-18) $\\to$ Hard (Laps 19-57).
* **Strategy B (2-Stop)**: Soft (Laps 1-15) $\\to$ Medium (Laps 16-36) $\\to$ Medium (Laps 37-57).
"""),
        ("code", """strat_1stop = Strategy(
    stints=[
        Stint(compound=DEFAULT_COMPOUNDS["Soft"], laps=18),
        Stint(compound=DEFAULT_COMPOUNDS["Hard"], laps=39),
    ],
    name="1-Stop (S -> H)"
)

strat_2stop = Strategy(
    stints=[
        Stint(compound=DEFAULT_COMPOUNDS["Soft"], laps=15),
        Stint(compound=DEFAULT_COMPOUNDS["Medium"], laps=21),
        Stint(compound=DEFAULT_COMPOUNDS["Medium"], laps=21),
    ],
    name="2-Stop (S -> M -> M)"
)

res_1stop = simulate_race(strat_1stop, model)
res_2stop = simulate_race(strat_2stop, model)

print(f"Strategy 1-Stop Duration: {res_1stop.total_time:.3f} s  ({res_1stop.summary()['formatted_time']})")
print(f"Strategy 2-Stop Duration: {res_2stop.total_time:.3f} s  ({res_2stop.summary()['formatted_time']})")
delta = res_1stop.total_time - res_2stop.total_time
winner = "2-Stop" if delta > 0 else "1-Stop"
print(f"Delta: {abs(delta):.3f} s ({winner} is faster)")
"""),
        ("markdown", """### 4. Lap-by-Lap Telemetry Profiling

Let us visualize the effective lap times over the 57 laps, highlighting the pit stop transit losses and compound degradation differences.
"""),
        ("code", """laps_1 = [lr.lap_number for lr in res_1stop.lap_records]
times_1 = [lr.effective_lap_time for lr in res_1stop.lap_records]

laps_2 = [lr.lap_number for lr in res_2stop.lap_records]
times_2 = [lr.effective_lap_time for lr in res_2stop.lap_records]

plt.figure(figsize=(12, 5))
plt.plot(laps_1, times_1, label="1-Stop (S -> H)", color="tab:blue", linewidth=2)
plt.plot(laps_2, times_2, label="2-Stop (S -> M -> M)", color="tab:orange", linewidth=2)

plt.title("Lap-by-Lap Effective Pace Profile: Bahrain Grand Prix", fontsize=14, fontweight="bold")
plt.xlabel("Lap Number", fontsize=12)
plt.ylabel("Effective Lap Time (s)", fontsize=12)
plt.ylim(90, 130)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
"""),
    ]

    # ---------------------------------------------------------
    # Notebook 02: Monte Carlo & Optimization
    # ---------------------------------------------------------
    nb2_cells = [
        ("markdown", """# Formula 1 Race Strategy Simulator
## Notebook 02: Monte Carlo Uncertainty & Dynamic Programming Optimization

This notebook demonstrates **Phase 2** and **Phase 3** mathematical algorithms:
1. **Vectorized Monte Carlo Uncertainty**: Simulating lap pace stochasticity $\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)$, pit stop transit variations, and random incidents.
2. **Statistical Risk Metrics**: Expectation, Standard Deviation, Value-at-Risk ($P_{95}$ VaR), and Expected Shortfall ($CVaR_{95}$).
3. **Win Probability Matrix**: Quantifying pairwise head-to-head dominance.
4. **Dynamic Programming Optimization**: Bellman backward induction across the Directed Acyclic Graph (DAG) of valid race stints.
5. **Pit Window Tolerance**: Computing the flexible pit window where strategic loss $\\le \\tau = 2.0\\text{s}$.
"""),
        ("code", """import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import RaceModel
from src.config import BAHRAIN_CONFIG, DEFAULT_COMPOUNDS
from src.strategies import Strategy, Stint
from src.stochastic import StochasticConfig
from src.montecarlo import MonteCarloEngine, compute_win_probability_matrix
from src.optimization import StrategyOptimizer

sns.set_theme(style="darkgrid")
print("Optimization & Monte Carlo engine initialized.")
"""),
        ("markdown", """### 1. Vectorized Monte Carlo Simulation

Under race conditions, lap times vary due to tyre graining, traffic, and driver precision. Pit stops also feature execution variance:
$$T_{pit} \\sim \\mathcal{N}(\\mu_{pit}, \\sigma_{pit}^2)$$

We simulate $N = 1,000$ stochastic iterations for three candidate strategies:
1. **1-Stop (S-H)**: Soft (18) $\\to$ Hard (39)
2. **2-Stop (S-M-M)**: Soft (15) $\\to$ Medium (21) $\\to$ Medium (21)
3. **2-Stop (S-H-S)**: Soft (14) $\\to$ Hard (28) $\\to$ Soft (15)
"""),
        ("code", """model = RaceModel(BAHRAIN_CONFIG)

strategies = [
    Strategy([Stint(DEFAULT_COMPOUNDS["Soft"], 18), Stint(DEFAULT_COMPOUNDS["Hard"], 39)], name="1-Stop (S-H)"),
    Strategy([Stint(DEFAULT_COMPOUNDS["Soft"], 15), Stint(DEFAULT_COMPOUNDS["Medium"], 21), Stint(DEFAULT_COMPOUNDS["Medium"], 21)], name="2-Stop (S-M-M)"),
    Strategy([Stint(DEFAULT_COMPOUNDS["Soft"], 14), Stint(DEFAULT_COMPOUNDS["Hard"], 28), Stint(DEFAULT_COMPOUNDS["Soft"], 15)], name="2-Stop (S-H-S)"),
]

stoch_config = StochasticConfig(lap_noise_std=0.25, pit_noise_std=0.6, num_simulations=1000)
mc_engine = MonteCarloEngine(model, stoch_config, seed=42)
mc_results = mc_engine.run_simulation(strategies)

print(f"Executed {len(strategies)} strategies across {stoch_config.num_simulations} stochastic iterations.")
"""),
        ("markdown", """### 2. Risk Metrics & Distribution Analysis

Let us inspect the distribution moments and tail risks:
* **Mean & Std**: Expected total duration and volatility.
* **VaR 95%**: 95th percentile worst-case race time.
* **CVaR 95%**: Conditional expected time given that the outcome falls in the worst 5% tail.
"""),
        ("code", """import pandas as pd

table = []
for res in mc_results:
    table.append({
        "Strategy": res.strategy_name,
        "Mean (s)": round(res.mean_time, 2),
        "Std (s)": round(res.std_dev, 2),
        "Median (s)": round(res.median_time, 2),
        "P95 VaR (s)": round(res.p95_time, 2),
        "CVaR 95 (s)": round(res.cvar_95, 2),
    })

df_metrics = pd.DataFrame(table)
print(df_metrics.to_string(index=False))
"""),
        ("markdown", """### 3. Visualizing Uncertainty: Empirical Distributions & Boxplots"""),
        ("code", """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# KDE Plot
for res in mc_results:
    sns.kdeplot(res.raw_times, ax=ax1, label=res.strategy_name, fill=True, alpha=0.3, linewidth=2)

ax1.set_title("Kernel Density Estimation of Race Duration", fontweight="bold")
ax1.set_xlabel("Race Time (s)")
ax1.set_ylabel("Probability Density")
ax1.legend()

# Boxplot
data = [res.raw_times for res in mc_results]
labels = [res.strategy_name for res in mc_results]
ax2.boxplot(data, tick_labels=labels, showmeans=True)
ax2.set_title("Strategy Race Duration Boxplots & Dispersion", fontweight="bold")
ax2.set_ylabel("Total Race Time (s)")
ax2.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.show()
"""),
        ("markdown", """### 4. Pairwise Win Probability Matrix

In head-to-head racing, the probability that strategy $A$ defeats strategy $B$ is given by:
$$P(T_A < T_B) = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{I}\\big(T_A^{(i)} < T_B^{(i)}\\big)$$
"""),
        ("code", """win_matrix, strat_names = compute_win_probability_matrix(mc_results)

plt.figure(figsize=(7, 5))
sns.heatmap(win_matrix, annot=True, fmt=".1%", cmap="coolwarm", cbar=False,
            xticklabels=strat_names, yticklabels=strat_names)
plt.title("Head-to-Head Win Probability Matrix", fontsize=13, fontweight="bold")
plt.xlabel("Opponent Strategy")
plt.ylabel("Candidate Strategy")
plt.tight_layout()
plt.show()
"""),
        ("markdown", """### 5. Dynamic Programming (Bellman Backward Induction)

The strategy optimization problem is formulated as a shortest-path problem on a Directed Acyclic Graph (DAG), where each node is a lap index $l \\in \\{0, 1, \\dots, N\\}$ and edges represent stints of compound $c$:
$$V(l) = \\min_{c \\in \\mathcal{C}, l' \\in (l, N]} \\Big[ \\text{LapPace}(l \\to l', c) + T_{pit}(l') + V(l') \\Big]$$
"""),
        ("code", """optimizer = StrategyOptimizer(model)

opt_result = optimizer.optimize_dp(max_stops=2)

print(f"Optimal Strategy Found: {opt_result.optimal_strategy.name}")
print(f"Optimal Race Time: {opt_result.optimal_strategy.total_time:.3f} s")
print(f"Graph Nodes Evaluated: {opt_result.dp_nodes_evaluated}")
print(f"Execution Time: {opt_result.computation_time_ms:.2f} ms")

print("\nStint Breakdown:")
for i, stint in enumerate(opt_result.optimal_strategy.stints, 1):
    print(f"  Stint {i}: {stint.compound.name} for {stint.laps} laps")
"""),
        ("markdown", """### 6. Pit Stop Window Sensitivity

In real racing, pit stops cannot always happen on the exact theoretical lap due to yellow flags, overtaking opportunities, or traffic.
We calculate the **pit window tolerance** $[\\text{lap}_{open}, \\text{lap}_{close}]$ within which stopping incurs $\\le \\tau = 2.0\\text{s}$ penalty.
"""),
        ("code", """windows = optimizer.compute_pit_windows(opt_result.optimal_strategy, tolerance_seconds=2.0)

for w in windows:
    print(f"Pit Stop #{w.pit_index + 1}:")
    print(f"  Optimal Lap: Lap {w.optimal_lap}")
    print(f"  Window Range: Lap {w.window_open_lap} to Lap {w.window_close_lap} (Size: {w.window_size} laps)")
"""),
    ]

    # ---------------------------------------------------------
    # Notebook 03: Sensitivity & Decision Phase Maps
    # ---------------------------------------------------------
    nb3_cells = [
        ("markdown", """# Formula 1 Race Strategy Simulator
## Notebook 03: Sensitivity Sweeps & Decision Phase Boundaries

This notebook explores **Phase 4**:
1. **1D Sensitivity Sweeps**: Continuous sweeps of pit lane transit loss $t_{loss} \\in [15.0\\text{s}, 35.0\\text{s}]$ and tyre degradation multiplier $\\mu_{deg} \\in [0.5, 2.5]$.
2. **Crossover Tipping Point Root-Finding**: Exact determination of threshold parameters via Brent's method.
3. **2D Decision Phase Boundaries**: Isoline contour mapping of $\\Delta T(t_{pit}, \\mu_{deg}) = 0$ separating 1-stop and 2-stop dominance regimes.
4. **Circuit Strategy Profiles**: Comparing high-degradation Bahrain with low-degradation Monza.
"""),
        ("code", """import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import RaceModel
from src.config import BAHRAIN_CONFIG, MONZA_CONFIG, DEFAULT_COMPOUNDS
from src.strategies import Strategy, Stint
from src.sensitivity import SensitivityAnalyzer

sns.set_theme(style="darkgrid")
print("Sensitivity analyzer ready.")
"""),
        ("markdown", """### 1. 1D Sensitivity: Pit Lane Time Loss Sweep

How sensitive is optimal strategy selection to the pit lane time loss?
We evaluate the delta between 1-Stop and 2-Stop:
$$\\Delta T(t_{pit}) = T(S_{\\text{1-stop}}; t_{pit}) - T(S_{\\text{2-stop}}; t_{pit})$$
* If $\\Delta T > 0$: 2-Stop is faster.
* If $\\Delta T < 0$: 1-Stop is faster.
* Root $\\Delta T(t_{pit}^*) = 0$ identifies the exact tipping point.
"""),
        ("code", """model = RaceModel(BAHRAIN_CONFIG)
analyzer = SensitivityAnalyzer(model)

strat_1stop = Strategy([Stint(DEFAULT_COMPOUNDS["Soft"], 18), Stint(DEFAULT_COMPOUNDS["Hard"], 39)], name="1-Stop (S-H)")
strat_2stop = Strategy([Stint(DEFAULT_COMPOUNDS["Soft"], 15), Stint(DEFAULT_COMPOUNDS["Medium"], 21), Stint(DEFAULT_COMPOUNDS["Medium"], 21)], name="2-Stop (S-M-M)")

pit_losses = np.linspace(15.0, 32.0, 35)
curve = analyzer.sweep_pit_loss(strat_1stop, strat_2stop, pit_losses)

# Find exact crossover tipping point using Brent's method
crossover = analyzer.find_crossover(strat_1stop, strat_2stop, "pit_loss", bounds=(15.0, 32.0))
if crossover.crossover_value is not None:
    print(f"Crossover Tipping Point: t_pit = {crossover.crossover_value:.2f} s")
else:
    print("No crossover observed in given bounds.")
"""),
        ("markdown", """### 2. Plotting 1D Crossover Curve"""),
        ("code", """plt.figure(figsize=(10, 5))
plt.plot(pit_losses, curve.delta_times, color="purple", linewidth=2.5, label=r"$\\Delta T = T_{1stop} - T_{2stop}$")
plt.axhline(0, color="gray", linestyle="--", alpha=0.7)

if crossover.crossover_value is not None:
    plt.axvline(crossover.crossover_value, color="red", linestyle=":", label=f"Tipping Point: {crossover.crossover_value:.2f}s")
    plt.scatter([crossover.crossover_value], [0], color="red", s=80, zorder=5)

plt.fill_between(pit_losses, curve.delta_times, 0, where=(curve.delta_times > 0), color="orange", alpha=0.2, label="2-Stop Dominates")
plt.fill_between(pit_losses, curve.delta_times, 0, where=(curve.delta_times < 0), color="blue", alpha=0.2, label="1-Stop Dominates")

plt.title("1D Pit Loss Sensitivity & Strategy Crossover: Bahrain GP", fontsize=13, fontweight="bold")
plt.xlabel("Pit Loss (seconds)", fontsize=11)
plt.ylabel("Time Delta: 1-Stop - 2-Stop (s)", fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
"""),
        ("markdown", """### 3. 2D Decision Phase Boundaries

In competitive racing, pit loss and track degradation fluctuate simultaneously (e.g. wet weather transition, green track evolution, or hot track temps).
The 2D decision boundary is defined by the level set:
$$\\mathcal{B} = \\big\\{ (t_{pit}, \\mu_{deg}) \\in \\mathbb{R}^2 \\mid T(S_1; t_{pit}, \\mu_{deg}) - T(S_2; t_{pit}, \\mu_{deg}) = 0 \\big\\}$$
"""),
        ("code", """phase_res = analyzer.compute_phase_diagram_2d(
    strat_1stop,
    strat_2stop,
    pit_loss_range=(16.0, 30.0),
    deg_range=(0.7, 1.8),
    grid_res=30
)

plt.figure(figsize=(10, 6))
cp = plt.contourf(phase_res.deg_grid, phase_res.pit_grid, phase_res.delta_grid, levels=25, cmap="RdYlBu_r")
cbar = plt.colorbar(cp)
cbar.set_label("Time Advantage: 2-Stop minus 1-Stop (s)")

# Decision boundary contour line
cs = plt.contour(phase_res.deg_grid, phase_res.pit_grid, phase_res.delta_grid, levels=[0.0], colors="black", linewidths=2.5, linestyles="--")
plt.clabel(cs, inline=True, fmt="Decision Boundary (0s)", fontsize=11)

nom_deg, nom_pit = phase_res.nominal_point
plt.scatter([nom_deg], [nom_pit], color="lime", edgecolor="black", s=150, zorder=10, label=f"Nominal Bahrain Point ({nom_deg}x, {nom_pit}s)")

plt.title("2D Strategic Phase Boundary: Pit Loss vs Tyre Degradation", fontsize=14, fontweight="bold")
plt.xlabel("Tyre Degradation Multiplier", fontsize=12)
plt.ylabel("Pit Loss (s)", fontsize=12)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
"""),
        ("markdown", """### 4. Cross-Circuit Comparison: Monza vs Bahrain

Why does Monza favor a 1-stop strategy while Bahrain strongly favors a 2-stop?
Let us run the DP optimizer on both circuits.
"""),
        ("code", """from src.optimization import StrategyOptimizer

monza_model = RaceModel(MONZA_CONFIG)
optimizer_bahrain = StrategyOptimizer(model)
optimizer_monza = StrategyOptimizer(monza_model)

opt_bah = optimizer_bahrain.optimize_dp(max_stops=2)
opt_mon = optimizer_monza.optimize_dp(max_stops=2)

print("=== CIRCUIT STRATEGY COMPARISON ===")
print(f"Bahrain: {model.circuit.name} ({model.circuit.total_laps} laps, pit loss: {model.pitstop_model.pit_loss}s)")
print(f"  Optimal: {opt_bah.optimal_strategy.name}")
print(f"\nMonza: {monza_model.circuit.name} ({monza_model.circuit.total_laps} laps, pit loss: {monza_model.pitstop_model.pit_loss}s)")
print(f"  Optimal: {opt_mon.optimal_strategy.name}")
"""),
    ]

    # ---------------------------------------------------------
    # Notebook 04: Historical Data Validation
    # ---------------------------------------------------------
    nb4_cells = [
        ("markdown", """# Formula 1 Race Strategy Simulator
## Notebook 04: Real-World Telemetry Ingestion & Model Validation

This notebook covers **Phase 5**:
1. **Historical Telemetry Ingestion**: Loading OpenF1 2024 race weekend timing data.
2. **Clean Lap Filtering**: Eliminating pit in/out laps, standing starts, safety cars, and unrepresentative pace outliers ($107\\%$ median filter).
3. **Fixed-Effects OLS Regression**: Correcting for fuel mass depletion and fitting compound degradation parameters $\\alpha_c, \\beta_c, \\delta_c$.
4. **Driver Backtesting**: Quantitative validation against 2024 Grand Prix winners (Max Verstappen, Charles Leclerc).
5. **Discrepancy Diagnostics**: Engineering root-cause analysis of real-world tactical deviations.
"""),
        ("code", """import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_pipeline import load_historical_data
from src.estimation import estimate_empirical_parameters
from src.validation import RaceBacktester, DiscrepancyAnalyzer
from run_validation import DEFAULT_WINNERS

sns.set_theme(style="darkgrid")
print("Validation & telemetry pipeline initialized.")
"""),
        ("markdown", """### 1. Ingesting Real-World Telemetry

We load the historical 2024 Bahrain Grand Prix dataset, which contains complete telemetry records:
* Total recorded laps across all drivers
* Stint compound history
* Real pit stop transit and stationary times
"""),
        ("code", """data = load_historical_data("bahrain")
print(f"Circuit: {data.circuit_name} ({data.year})")
print(f"Total Laps Recorded: {len(data.laps)}")
print(f"Total Pit Stops Recorded: {len(data.pit_stops)}")
print(f"Participating Drivers: {len(data.stints.driver_number.unique())}")
"""),
        ("markdown", """### 2. Empirical Parameter Estimation via OLS Regression

Raw lap times $T(n)$ mask degradation because fuel burn continuously accelerates the car:
$$T_{\\text{corr}}(a) = T_{\\text{raw}}(n) + \\gamma_{\\text{fuel}} \\cdot (m_0 - m(n))$$

We fit a quadratic wear model for each tyre compound:
$$T_{\\text{corr}}(a) = T_{base} + \\delta_c + \\alpha_c a + \\beta_c a^2 + \\epsilon$$
"""),
        ("code", """emp_params = estimate_empirical_parameters(data)

print(f"Base Lap Time: {emp_params.base_lap_time:.3f} s")
print(f"Mean Pit Loss: {emp_params.pit_loss_mean:.2f} s (std: {emp_params.pit_loss_std:.2f} s)")

fitted_rows = []
for c_name, cp in emp_params.compounds.items():
    fitted_rows.append({
        "Compound": c_name,
        "Linear Alpha (s/lap)": f"{cp.alpha:.5f}",
        "Quadratic Beta (s/lap^2)": f"{cp.beta:.6f}",
        "Base Delta (s)": f"{cp.base_delta:+.3f}",
        "R-squared": f"{cp.r_squared:.3f}",
        "Clean Samples": cp.sample_count,
    })

print(pd.DataFrame(fitted_rows).to_string(index=False))
"""),
        ("markdown", """### 3. Backtesting Against 2024 Bahrain Winner: Max Verstappen

We test whether the calibrated simulator accurately reproduces the race-winning strategy of Max Verstappen (#1).
"""),
        ("code", """backtester = RaceBacktester(data, empirical_params=emp_params)
driver_num, driver_name = DEFAULT_WINNERS["bahrain"]
metrics = backtester.backtest_driver(driver_num, driver_name=driver_name)

print("=== VERSTAPPEN BACKTEST METRICS ===")
print(f"Actual Strategy:              {metrics.actual_strategy_desc} ({metrics.actual_stops} stops)")
print(f"Simulated Optimal Strategy:   {metrics.optimal_strategy_desc} ({metrics.optimal_stops} stops)")
print(f"Stop Count Matched:           {metrics.stop_count_matches}")
print(f"Stint Length MAE:             {metrics.stint_length_mae:.2f} laps")
print(f"Clean Lap Pace RMSE:          {metrics.clean_lap_rmse:.3f} s")
print(f"Actual Race Duration:         {metrics.actual_time:.2f} s")
print(f"Simulated Actual Duration:    {metrics.sim_actual_strategy_time:.2f} s")
print(f"Physics Duration Error:       {metrics.sim_actual_error_pct:.3f}%")
print(f"Strategic Gain Delta:         {metrics.strategic_gain_delta:+.2f} s")
"""),
        ("markdown", """### 4. Residual Diagnostic Analysis

Let us inspect the distribution of lap-by-lap residuals $T_{\\text{actual}} - T_{\\text{simulated}}$.
"""),
        ("code", """res_df = backtester.get_lap_residuals(driver_num)

plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.scatter(res_df.lap_number, res_df.residual, color="royalblue", alpha=0.7, edgecolor="k")
plt.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
plt.title("Lap Pace Residuals vs Race Lap", fontweight="bold")
plt.xlabel("Lap Number")
plt.ylabel("Residual: Actual - Simulated (s)")

plt.subplot(1, 2, 2)
sns.histplot(res_df.residual, kde=True, color="navy")
plt.axvline(0, color="crimson", linestyle="--")
plt.title("Residual Error Distribution", fontweight="bold")
plt.xlabel("Residual (s)")

plt.tight_layout()
plt.show()
"""),
        ("markdown", """### 5. Tactical Discrepancy Diagnostics

Why do real Formula 1 pit stop laps sometimes differ from analytical predictions?
The `DiscrepancyAnalyzer` categorizes tactical mechanisms:
* **Thermal tyre management**: Drivers delta-managing pace in early stint laps.
* **Dirty air mitigation**: Pitting into clean track windows rather than optimal wear points.
* **Undercut/Overcut tactics**: Tactical responses to rival pit window positioning.
"""),
        ("code", """diag = DiscrepancyAnalyzer.analyze_circuit("bahrain")

print(f"Diagnostic Case: {diag.case_title}")
print(f"\nObserved Phenomenon:\n  {diag.observed_phenomenon}")
print(f"\nTheoretical Model Prediction:\n  {diag.theoretical_prediction}")
print("\nRoot Cause Factors:")
for f in diag.root_cause_factors:
    print(f"  * {f}")
print(f"\nEngineering Solution:\n  {diag.engineering_solution}")
"""),
    ]

    # ---------------------------------------------------------
    # Notebook 05: What-If Scenario Analysis
    # ---------------------------------------------------------
    nb5_cells = [
        ("markdown", """# Formula 1 Race Strategy Simulator
## Notebook 05: What-If Counterfactual Scenario Analysis

This notebook demonstrates **Phase 6: Scenario Analysis & Counterfactual What-If Engine**:
1. **Perturbation Manifolds**: Formalizing counterfactual changes to circuit, pit crew, and tyre parameters $\\theta' = \\mathcal{P}(\\theta_0, \\mathbf{p})$.
2. **Strategic Regret Theory**: Measuring the time cost of running the nominal plan under perturbed reality:
   $$\\mathcal{R}(S; \\theta') = T(S; \\theta') - \\min_{S'} T(S'; \\theta')$$
3. **Core Research Scenarios from project.md**:
   * Pit loss variations ($\\pm 3.0\\text{s}$, botched stop $+10\\text{s}$)
   * Extreme thermal degradation spikes ($+50\\%$)
   * Soft tyre pace differential shifts
   * Sprint vs Full Grand Prix distance
   * Safety Car neutralization at Lap 18
   * 2026 Technical Regulation shifts
4. **Winning Strategy Pivots**: Identifying when tactical pivots become mandatory.
"""),
        ("code", """import sys
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import RaceModel
from src.config import BAHRAIN_CONFIG
from src.scenarios import ScenarioEngine, Scenario, PRESET_SCENARIOS

sns.set_theme(style="darkgrid")
print(f"Loaded {len(PRESET_SCENARIOS)} preset scenarios.")
"""),
        ("markdown", """### 1. The Scenario Analysis Engine

The `ScenarioEngine` perturbs physical model parameters while holding strategy candidate sets constant, allowing us to compute:
* Nominal optimal strategy time under counterfactual conditions
* Adapted optimal strategy time
* **Strategic Regret** $\\mathcal{R}$: Time forfeited by not adapting the strategy
* **Strategy Pivot**: Whether the optimal tactic switches (e.g. 1-Stop $\\to$ 2-Stop)
"""),
        ("code", """model = RaceModel(BAHRAIN_CONFIG)
engine = ScenarioEngine(model)

print(f"Base Circuit: {model.circuit.name} ({model.circuit.total_laps} laps)")
print("Available Scenarios:")
for s_id, s_obj in PRESET_SCENARIOS.items():
    print(f"  * {s_id:20s}: {s_obj.name}")
"""),
        ("markdown", """### 2. Evaluating the Full What-If Scenario Matrix

Let us run all 11 pre-configured scenarios and compile a strategic impact table.
"""),
        ("code", """results = engine.run_all_scenarios()

summary_rows = []
for res in results:
    summary_rows.append({
        "Scenario": res.scenario_name,
        "Nominal Winner": res.nominal_best_strategy,
        "Adapted Winner": res.adapted_best_strategy,
        "Strategy Pivot?": "YES" if res.strategy_pivoted else "No",
        "Regret (s)": round(res.strategic_regret, 2),
        "Time Delta (s)": round(res.delta_to_nominal_seconds, 2),
    })

df_scenarios = pd.DataFrame(summary_rows)
print(df_scenarios.to_string(index=False))
"""),
        ("markdown", """### 3. Visualizing Strategic Regret Across Scenarios

Strategic Regret $\\mathcal{R}$ quantifies the competitive disaster of remaining on the baseline plan when track conditions mutate.
"""),
        ("code", """fig, ax = plt.subplots(figsize=(12, 5))

names = [r.scenario_name for r in results]
regrets = [r.strategic_regret for r in results]
pivots = [r.strategy_pivoted for r in results]
colors = ["crimson" if p else "steelblue" for p in pivots]

bars = ax.barh(names, regrets, color=colors, edgecolor="black", height=0.6)
ax.set_title("Strategic Regret of Nominal Plan Under Counterfactual Scenarios", fontsize=13, fontweight="bold")
ax.set_xlabel("Strategic Regret (seconds forfeited)", fontsize=11)

# Annotate pivot flag
for bar, p in zip(bars, pivots):
    if p:
        ax.text(bar.get_width() + 0.3, bar.get_y() + 0.15, "PIVOT", color="crimson", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.show()
"""),
        ("markdown", """### 4. Deep-Dive: Scenario 1 - Severe Tyre Degradation Spike (+50%)

When track temperature surges and degradation rises by $+50\\%$, how does the race time and optimal strategy shift?
"""),
        ("code", """res_deg = engine.evaluate_scenario("high_deg")

print(f"=== SCENARIO: {res_deg.scenario_name} ===")
print(f"Insight: {res_deg.summary_insight}")
print(f"Nominal Winner: {res_deg.nominal_best_strategy}")
print(f"Adapted Winner: {res_deg.adapted_best_strategy}")
print(f"Strategic Regret: {res_deg.strategic_regret:.2f} s")

print("\nStrategy Outcomes under High Degradation:")
for outcome in res_deg.strategy_outcomes:
    print(f"  * {outcome.strategy_name:25s} Duration: {outcome.total_time:.2f}s  Delta to P1: {outcome.delta_to_p1:+.2f}s")
"""),
        ("markdown", """### 5. Deep-Dive: Scenario 2 - Safety Car Neutralization at Lap 18

Under a Safety Car, pit loss drops by $\\approx 35\\%$ because field speed is capped under the delta:
$$T_{pit}^{SC} = T_{transit} \\cdot \\frac{v_{racing}}{v_{delta}} + T_{stationary}$$

Does this trigger an immediate opportunistic pit stop?
"""),
        ("code", """res_sc = engine.evaluate_scenario("safety_car_lap18")

print(f"=== SCENARIO: {res_sc.scenario_name} ===")
print(f"Insight: {res_sc.summary_insight}")
print(f"Strategy Pivot Occurred: {res_sc.strategy_pivoted}")
print(f"Strategic Regret: {res_sc.strategic_regret:.2f} s")
"""),
        ("markdown", """### 6. Deep-Dive: Scenario 3 - 2026 Technical Regulations Shift

The 2026 regulations introduce:
* Reduced fuel capacity ($70\\text{ kg}$ vs $105\\text{ kg}$)
* Active aerodynamics (reduced drag on straights)
* Altered power unit deployment ($50\\%$ ICE / $50\\%$ Electrical)
"""),
        ("code", """res_2026 = engine.evaluate_scenario("regulations_2026")

print(f"=== SCENARIO: {res_2026.scenario_name} ===")
print(f"Insight: {res_2026.summary_insight}")
print(f"Nominal 2024 Winner: {res_2026.nominal_best_strategy}")
print(f"Adapted 2026 Winner: {res_2026.adapted_best_strategy}")
print(f"Time Delta to 2024 Baseline: {res_2026.delta_to_nominal_seconds:+.2f} s")
"""),
    ]

    notebooks_to_build = [
        ("01_synthetic_model_exploration.ipynb", "01: Physics & Synthetic Model Exploration", nb1_cells),
        ("02_monte_carlo_and_optimization.ipynb", "02: Monte Carlo Uncertainty & Strategy Optimization", nb2_cells),
        ("03_sensitivity_and_decision_maps.ipynb", "03: Sensitivity Sweeps & Decision Phase Boundaries", nb3_cells),
        ("04_historical_data_validation.ipynb", "04: Real-World Telemetry Ingestion & Model Validation", nb4_cells),
        ("05_scenario_analysis_and_what_if.ipynb", "05: What-If Counterfactual Scenario Analysis", nb5_cells),
    ]

    for filename, title, cells in notebooks_to_build:
        nb_obj = make_notebook(title, cells)
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            nbf.write(nb_obj, f)
        print(f"Successfully generated {out_path} ({len(nb_obj.cells)} cells)")


if __name__ == "__main__":
    notebooks_dir = Path(__file__).resolve().parent.parent / "notebooks"
    generate_all_notebooks(notebooks_dir)
