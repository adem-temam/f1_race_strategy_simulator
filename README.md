# F1 Race Strategy Simulator & Analysis 🏎️⏱️

A data-driven simulation and optimization engine for Formula 1 race strategy. The platform evaluates competing tyre and pit-stop strategies under uncertainty, computes stint pace profiles, models non-linear tyre degradation and fuel burn dynamics, discovers mathematically optimal strategies, and calculates tactical pit windows.

---

## 📌 Features

- **Lap-Time Decomposition Engine**: Synthesizes circuit baseline pace, compound pace offsets, dynamic fuel burn load, non-linear tyre degradation, track rubbering evolution, and traffic penalties into individual lap times.
- **Dynamic Programming Optimizer**: Exact global strategy solver utilizing Bellman backward induction on a Directed Acyclic Graph (DAG) in `< 15ms`.
- **Combinatorial Grid Search**: High-throughput strategy evaluation capable of screening 19,000+ candidate strategies in `~0.5 seconds`.
- **Tactical Pit Window Discovery**: Calculates allowable undercut and overcut pit windows around optimal stop laps within an acceptable tactical delta ($\delta_{\text{tol}}$).
- **Multi-Objective & Pareto Frontier Analysis**: Discovers non-dominated strategies balancing raw pace ($\mathbb{E}[T_{\text{race}}]$) against downside risk ($P_{95}$ Value at Risk).
- **Monte Carlo Engine**: Runs vectorized probability simulations modeling lap pace noise ($\mathcal{N}$), pit stop delay distributions (heavy-tailed LogNormal), and tyre wear multipliers.
- **Statistical Analytics & Distribution Visualizer**: Computes Expected Time, Medians, standard deviations, P95 Value at Risk, and head-to-head win probability matrix, with automated plotting for KDE curves, Histograms, empirical CDFs, Boxplots, and Heatmaps.
- **Traffic & Dirty Air Dynamics**: Models Virtual Field Spread and DRS train dirty air penalties after pit stops.
- **Sporting Regulations**: Validates lap count conservation and the mandatory FIA two-compound dry race regulation.
- **Interactive CLI Runner**: Evaluate preset/custom strategies or trigger automated optimization with rich terminal telemetry tables.
- **Automated Test Suite**: 51 comprehensive unit tests validating physical equations, conservation laws, dynamic programming optimality, and Pareto frontiers.

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
│   └── visualization.py # KDE, Histograms, CDFs, Boxplots, Heatmaps & Dashboard
├── tests/
│   ├── test_tyres.py
│   ├── test_fuel.py
│   ├── test_pitstop.py
│   ├── test_strategies.py
│   ├── test_model.py
│   ├── test_simulation.py
│   ├── test_montecarlo.py
│   ├── test_visualization.py
│   └── test_optimization.py
├── run_simulation.py    # Interactive CLI runner with simulation & optimization modes
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

### 3. Evaluate Preset Strategies via CLI
Run the default strategy comparison on Bahrain GP (57 laps):
```bash
python3 run_simulation.py --circuit bahrain
```

### 4. Test Custom Stints
Specify custom compound sequences using shorthand (`S15-M21-S21`, `M26-H31`):
```bash
python3 run_simulation.py --circuit bahrain -s "S15-M21-S21" -s "M26-H31" -s "S12-H45"
```

### 5. Inspect Lap Telemetry
View per-lap physical breakdown (degradation, remaining fuel, effective lap time):
```bash
python3 run_simulation.py --strategy "M26-H31" --laps
```

### 6. Monte Carlo Probabilistic Simulation
Run thousands of iterations introducing real-world stochastic variance:
```bash
# Run 10,000 iterations to calculate Expected Time, Risk (P95), and Head-to-Head Win Probability
python3 run_simulation.py --mc 10000

# Run iterations and generate complete Seaborn visual suite (KDE, Histograms, CDF, Boxplots, Win Heatmap, Dashboard)
python3 run_simulation.py --mc 5000 --plot
```

---

## 🐍 Python API Example

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

---

## 🧪 Testing

Run the automated test suite with `pytest`:
```bash
pytest -v
```

All 51 unit tests verify model mechanics, fuel consumption conservation, pit stop accounting, FIA rule compliance, Bellman DAG optimality, and Pareto efficiency.

---

## 🗺️ Roadmap

- [x] **Phase 1: Mathematical Modeling Core & Deterministic Simulation**
- [x] **Phase 2: Monte Carlo Simulation & Stochastic Uncertainty** (lap variance, pit stop delays, wear deviations, distribution visualizer)
- [x] **Phase 3: Strategy Optimization Engine** (Dynamic Programming DAG, Combinatorial Search, tactical pit windows, Pareto risk modeling)
- [ ] **Phase 4: Sensitivity Analysis & Decision Phase Maps** (crossover points and 2D parameter heatmaps)
- [ ] **Phase 5: Real-World Historical Data Integration** (FastF1 / Ergast parameter fitting and race validation)
- [ ] **Phase 6: Interactive Dashboard & Scientific Visualization**

