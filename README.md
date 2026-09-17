# F1 Race Strategy Simulator & Analysis 🏎️⏱️

A data-driven simulation and optimization engine for Formula 1 race strategy. The platform evaluates competing tyre and pit-stop strategies under uncertainty, computes stint pace profiles, models non-linear tyre degradation and fuel burn dynamics, discovers mathematically optimal strategies, and calculates tactical pit windows.

---

## 📌 Features

- **Lap-Time Decomposition Engine**: Synthesizes circuit baseline pace, compound pace offsets, dynamic fuel burn load, non-linear tyre degradation, track rubbering evolution, and traffic penalties into individual lap times.
- **Dynamic Programming Optimizer**: Exact global strategy solver utilizing Bellman backward induction on a Directed Acyclic Graph (DAG) in `< 15ms`.
- **Combinatorial Grid Search**: High-throughput strategy evaluation capable of screening 19,000+ candidate strategies in `~0.5 seconds`.
- **Tactical Pit Window Discovery**: Calculates allowable undercut and overcut pit windows around optimal stop laps within an acceptable tactical delta ($\delta_{\text{tol}}$).
- **Sensitivity Analysis & Tipping Point Engine**: Calculates exact critical crossover thresholds (e.g., degradation rate where 2-stop beats 1-stop) using Brent's method root-finding down to $10^{-4}$ precision.
- **2D Decision Phase Maps**: Maps the entire parameter space (Pit Loss vs Tyre Degradation), generating decision boundary contours, iso-delta surfaces, and tactical safety margins.
- **Dimensionless Elasticity & Minimax Regret**: Measures strategy vulnerability ($\% \Delta T_{\text{race}} / \% \Delta \theta$) and minimax regret under environmental misestimation.
- **Multi-Objective & Pareto Frontier Analysis**: Discovers non-dominated strategies balancing raw pace ($\mathbb{E}[T_{\text{race}}]$) against downside risk ($P_{95}$ Value at Risk).
- **Monte Carlo Engine**: Runs vectorized probability simulations modeling lap pace noise ($\mathcal{N}$), pit stop delay distributions (heavy-tailed LogNormal), and tyre wear multipliers.
- **Statistical Analytics & Distribution Visualizer**: Computes Expected Time, Medians, standard deviations, P95 Value at Risk, and head-to-head win probability matrix, with automated plotting for KDE curves, Histograms, empirical CDFs, Boxplots, and Heatmaps.
- **Traffic & Dirty Air Dynamics**: Models Virtual Field Spread and DRS train dirty air penalties after pit stops.
- **Sporting Regulations**: Validates lap count conservation and the mandatory FIA two-compound dry race regulation.
- **Interactive CLI Runner**: Evaluate preset/custom strategies, trigger automated optimization, or conduct parameter sensitivity sweeps with rich terminal telemetry tables.
- **Automated Test Suite**: 64 comprehensive unit tests validating physical equations, conservation laws, dynamic programming optimality, Pareto frontiers, Brent crossover precision, and 2D phase maps.

---

## 📐 Mathematical Formulation

The deterministic lap time on lap $n \in \{1, \dots, N\}$ during stint $k$ on tyre compound $c$ is given by:

$$T_{\text{lap}}(n) = T_{\text{base}} + \Delta_{\text{compound}}(c) + D_c\big(a(n)\big) + F\big(m_f(n)\big) - E_{\text{track}}(n) + T_{\text{traffic}}(n)$$

### 1. Tyre Degradation Model
$$D_c(a) = \alpha_c \cdot a + \beta_c \cdot a^2 + \kappa_c \cdot \big(\max(0, a - a_{\text{cliff}, c})\big)^2$$
* $a$: Effective tyre age in laps (including dirty air degradation acceleration).
* $\alpha_c$: Linear wear coefficient ($\text{s/lap}$).
* $\beta_c$: Non-linear fatigue coefficient ($\text{s/lap}^2$).
* $a_{\text{cliff}, c}$: Stint age threshold beyond which tyre performance sharply drops.

### 2. Fuel Load & Penalty Model
$$m_f(n) = m_{f, 0} - (n - 1) \cdot \Delta m_f$$
$$F\big(m_f(n)\big) = \gamma_{\text{fuel}} \cdot m_f(n)$$
* $m_{f, 0}$: Initial fuel load ($100 - 105\text{ kg}$).
* $\Delta m_f$: Fuel burn rate ($\approx 1.7 - 1.9\text{ kg/lap}$).
* $\gamma_{\text{fuel}}$: Fuel weight sensitivity ($\approx 0.030 - 0.035\text{ s/kg}$).

### 3. Total Race Time
$$T_{\text{race}}(S) = \sum_{n=1}^N T_{\text{lap}}(n) + (K - 1) \cdot \Delta T_{\text{pit}}$$
where $K$ is the number of stints and $\Delta T_{\text{pit}}$ is the net pit lane time loss.

---

## 📁 Project Architecture

```text
race_strategy_simulator/
├── data/
│   └── raw/             # Bundled offline datasets (bahrain_2024.json, barcelona_2024.json, monza_2024.json)
├── docs/
│   ├── phase_1_mathematical_model.md
│   ├── phase_2_mathematical_model.md
│   ├── phase_3_mathematical_model.md
│   ├── phase_4_mathematical_model.md
│   ├── phase_5_mathematical_model.md
│   └── phase_6_mathematical_model.md
├── notebooks/           # Interactive demonstration Jupyter notebook curriculum (Phases 1-6)
│   ├── 01_synthetic_model_exploration.ipynb
│   ├── 02_monte_carlo_and_optimization.ipynb
│   ├── 03_sensitivity_and_decision_maps.ipynb
│   ├── 04_historical_data_validation.ipynb
│   └── 05_scenario_analysis_and_what_if.ipynb
├── scripts/
│   └── generate_notebooks.py # Programmatic notebook suite generator
├── src/
│   ├── tyres.py         # TyreCompound dataclass, wear curves & Pirelli compound specs
│   ├── fuel.py          # FuelModel consumption & weight penalty calculations
│   ├── pitstop.py       # PitStopModel transit loss and stationary stop time
│   ├── strategies.py    # Stint, Strategy classes & FIA rule validation
│   ├── model.py         # CircuitConfig, LapRecord, and RaceModel synthesis
│   ├── simulation.py    # Deterministic simulate_race() engine & RaceResult
│   ├── config.py        # Circuit presets (Bahrain, Barcelona, Monza) & compounds
│   ├── stochastic.py    # Stochastic noise parameters & probability distributions
│   ├── montecarlo.py    # Vectorized Monte Carlo engine & win probability matrix
│   ├── optimization.py  # DP Solver, Combinatorial Grid Search, Pit Windows & Pareto
│   ├── sensitivity.py   # 1D sweeps, Brent's root-finding & 2D decision phase boundaries
│   ├── data_pipeline.py # OpenF1 REST API client, local JSON caching & clean telemetry filter
│   ├── estimation.py    # Fuel mass correction & multi-driver polynomial regression for wear
│   ├── validation.py    # Historical strategy backtester, accuracy metrics & discrepancy diagnostics
│   ├── scenarios.py     # What-If scenario engine, counterfactual perturbations & regret
│   ├── dashboard/       # Interactive FastAPI server & dark-theme telemetry web UI
│   │   ├── app.py       # REST API endpoints & server setup
│   │   └── static/      # index.html, styles.css, app.js
│   └── visualization.py # Empirical wear curves, residuals, Gantt backtest & executive dashboards
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_estimation.py
│   ├── test_validation.py
│   ├── test_circuit_evaluations.py
│   ├── test_limitations_fixed.py
│   ├── test_tyres.py
│   ├── test_fuel.py
│   ├── test_pitstop.py
│   ├── test_strategies.py
│   ├── test_model.py
│   ├── test_simulation.py
│   ├── test_montecarlo.py
│   ├── test_visualization.py
│   ├── test_optimization.py
│   ├── test_sensitivity.py
│   ├── test_scenarios.py
│   ├── test_dashboard_api.py
│   └── test_notebooks.py
├── run_simulation.py    # Interactive CLI runner (simulation, optimization, sensitivity, validation, scenarios, dashboard)
├── run_validation.py    # Dedicated Phase 5 validation CLI runner
├── run_dashboard.py     # Dedicated Phase 6 interactive dashboard runner
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/adem-temam/race_strategy_simulator.git
cd race_strategy_simulator
pip install -r requirements.txt
```

### 2. Strategy Optimization (Phase 3)
Automatically discover the mathematically optimal strategy for a circuit:
```bash
# Optimize Bahrain GP (finds optimal 2-stop strategy and pit windows in ~0.5s)
python3 run_simulation.py --circuit bahrain --optimize

# Enforce maximum pit stops (e.g., find optimal 1-stop strategy at Monza)
python3 run_simulation.py --circuit monza --optimize --max-stops 1

# Optimize for lowest downside risk (P95 Value at Risk) using Monte Carlo
python3 run_simulation.py --circuit bahrain --optimize --objective risk --mc 500

# Suppress tactical pit windows table if only leaderboard is desired
python3 run_simulation.py --circuit barcelona --optimize --no-pit-windows
```

### 3. Sensitivity Analysis & Decision Phase Maps (Phase 4)
Investigate why and when strategies transition between 1-stop and 2-stop regimes:
```bash
# Compute exact critical tipping point (root finding) for tyre degradation
python3 run_simulation.py --circuit bahrain --crossover

# Compute tipping point for pit stop time loss (e.g. at Monza)
python3 run_simulation.py --circuit monza --crossover --param pit-loss

# Run 1D parameter sweep comparing optimal 1-stop vs 2-stop with elasticity
python3 run_simulation.py --circuit bahrain --sensitivity --param deg

# Generate 2D decision boundary phase diagram across Pit Loss and Degradation
python3 run_simulation.py --circuit bahrain --phase-map --plot

# Generate full 4-panel executive sensitivity dashboard
python3 run_simulation.py --circuit bahrain --sensitivity --plot
```

### 4. Historical Data Validation & Empirical Telemetry Calibration (Phase 5)
Ingest real Formula 1 race telemetry, fit empirical wear coefficients ($\alpha_c, \beta_c$), and backtest model recommendations against real 2024 Grand Prix winners:
```bash
# Run strategy backtest and causal discrepancy diagnostics on Bahrain GP
python3 run_simulation.py --circuit bahrain --validate

# Inspect empirical parameter regression table (wear alpha/beta, pit loss, lap noise)
python3 run_simulation.py --circuit barcelona --fit-params

# Run full dedicated validation suite with visual plots (wear curves, residuals, Gantt charts)
python3 run_validation.py --circuit monza --plot

# Execute backtest across all 3 benchmark circuits simultaneously
python3 run_validation.py --circuit all
```

### 5. Interactive Strategy Dashboard (Phase 6)
Launch the dark-theme web dashboard powered by FastAPI and Chart.js:
```bash
# Launch dashboard on http://127.0.0.1:8000 with automatic browser opening
python3 run_dashboard.py

# Alternatively, launch directly via the CLI runner
python3 run_simulation.py --dashboard --port 8000
```

### 6. What-If Counterfactual Scenario Analysis (Phase 6)
Evaluate counterfactual track interventions, regulation changes, Safety Cars, and calculate Strategic Regret:
```bash
# Run a specific preset scenario (e.g., severe thermal degradation spike)
python3 run_simulation.py --circuit bahrain --scenario high_deg

# Evaluate an unexpected Safety Car neutralization at Lap 18
python3 run_simulation.py --circuit bahrain --scenario safety_car_lap18

# Run the complete 11-scenario counterfactual matrix across all tactical events
python3 run_simulation.py --circuit bahrain --run-all-scenarios
```

### 7. Interactive Demonstration Notebooks (Phase 6)
Explore the interactive curriculum spanning Phases 1 through 6 in `notebooks/`:
```bash
jupyter notebook notebooks/
```
* `01_synthetic_model_exploration.ipynb`: Physics engine, tyre wear curves, and fuel mass burn.
* `02_monte_carlo_and_optimization.ipynb`: Stochastic risk modeling, win probability matrix, and Bellman DP.
* `03_sensitivity_and_decision_maps.ipynb`: Brent root-finding crossover tipping points and 2D phase diagrams.
* `04_historical_data_validation.ipynb`: OpenF1 telemetry ingestion, OLS regression, and Verstappen/Leclerc backtesting.
* `05_scenario_analysis_and_what_if.ipynb`: Counterfactual perturbations, strategic regret, and 2026 regulations.

### 8. Evaluate Preset Strategies via CLI
Run the default strategy comparison on Bahrain GP (57 laps):
```bash
python3 run_simulation.py --circuit bahrain
```

### 9. Test Custom Stints
Specify custom compound sequences using shorthand (`S15-M21-S21`, `M26-H31`):
```bash
python3 run_simulation.py --circuit bahrain -s "S15-M21-S21" -s "M26-H31" -s "S12-H45"
```

### 10. Inspect Lap Telemetry
View per-lap physical breakdown (degradation, remaining fuel, effective lap time):
```bash
python3 run_simulation.py --strategy "M26-H31" --laps
```

### 11. Monte Carlo Probabilistic Simulation
Run thousands of iterations introducing real-world stochastic variance:
```bash
# Run 10,000 iterations to calculate Expected Time, Risk (P95), and Head-to-Head Win Probability
python3 run_simulation.py --mc 10000

# Run iterations and generate complete Seaborn visual suite
python3 run_simulation.py --mc 5000 --plot
```

---

## 🐍 Python API Example

### Strategy Optimization (Phase 3)
```python
from src.config import create_race_model, get_circuit_compounds
from src.optimization import StrategyOptimizer, OptimizationObjective

# 1. Initialize circuit and compounds
model = create_race_model("bahrain")
compounds = get_circuit_compounds("bahrain")

# 2. Initialize optimizer
optimizer = StrategyOptimizer(model, compounds)

# 3. Discover optimal strategy
result = optimizer.optimize(max_stops=2, objective=OptimizationObjective.DETERMINISTIC_TIME)
print(f"Optimal Strategy: {result.optimal_strategy.description}")
print(f"Total Race Time: {result.formatted_optimal_time}")

# 4. Inspect tactical pit windows
for window in result.pit_windows:
    print(f"Stop {window.pit_index}: Window Laps {window.window_open_lap}–{window.window_close_lap} (Size: {window.window_size}L)")
```

### What-If Scenario Analysis (Phase 6)
```python
from src.model import RaceModel
from src.config import BAHRAIN_CONFIG
from src.scenarios import ScenarioEngine, PRESET_SCENARIOS

model = RaceModel(BAHRAIN_CONFIG)
engine = ScenarioEngine(model)

# Evaluate high thermal degradation spike scenario (+50% tyre wear)
res = engine.evaluate_scenario("high_deg")
print(f"Scenario: {res.scenario_name}")
print(f"Strategic Regret: {res.strategic_regret:.2f}s")
print(f"Adapted Optimal Strategy: {res.adapted_best_strategy}")
print(f"Strategy Pivot Triggered: {res.strategy_pivoted}")
```

---

## 🧪 Testing

Run the automated test suite with `pytest`:
```bash
pytest -v
```

All 100 unit tests verify physical model mechanics, fuel mass conservation, pit stop transit accounting, FIA rule compliance, Bellman DAG optimality, Pareto risk frontiers, Brent crossover root-finding, 2D phase boundaries, OpenF1 offline cache ingestion, empirical polynomial wear regression, strategy backtests, causal discrepancy diagnostics, counterfactual scenario permutations, FastAPI REST endpoints, and Jupyter notebook structures.

---

## 🗺️ Roadmap

- [x] **Phase 1: Mathematical Modeling Core & Deterministic Simulation**
- [x] **Phase 2: Monte Carlo Simulation & Stochastic Uncertainty** (lap variance, pit stop delays, wear deviations, distribution visualizer)
- [x] **Phase 3: Strategy Optimization Engine** (Dynamic Programming DAG, Combinatorial Search, tactical pit windows, Pareto risk modeling)
- [x] **Phase 4: Sensitivity Analysis & Decision Phase Maps** (crossover tipping points, Brent's method root-finding, 2D decision phase boundaries, elasticity, and minimax regret)
- [x] **Phase 5: Real-World Historical Data Integration & Validation** (OpenF1 ingestion, local offline JSON cache, fuel correction, empirical polynomial regression for tyre wear, strategy backtesting against 2024 winners, and discrepancy diagnostics)
- [x] **Phase 6: Interactive Dashboard, Scenario Analysis & Demonstration Notebooks** (FastAPI REST server, dark-theme telemetry web UI, What-If counterfactual scenario engine, strategic regret evaluation, and 5 interactive Jupyter curriculum notebooks)

