# F1 Race Strategy Simulator & Analysis 🏎️⏱️

A data-driven simulation and optimization engine for Formula 1 race strategy. The platform evaluates competing tyre and pit-stop strategies under uncertainty, computes stint pace profiles, models non-linear tyre degradation and fuel burn dynamics, discovers mathematically optimal strategies, and calculates tactical pit windows.

---

## 📌 Features

- **Lap-Time Decomposition Engine**: Synthesizes circuit baseline pace, compound pace offsets, dynamic fuel burn load, non-linear tyre degradation, track rubbering evolution, and traffic penalties into individual lap times.
- **Dynamic Programming Optimizer**: Exact global strategy solver utilizing Bellman backward induction on a Directed Acyclic Graph (DAG) in `< 15ms`.
- **Combinatorial Grid Search**: High-throughput strategy evaluation capable of screening 19,000+ candidate strategies in `~0.5 seconds`.
- **Tactical Pit Window Discovery**: Calculates allowable undercut and overcut pit windows around optimal stop laps within an acceptable tactical delta ($\delta_{\te- **Sensitivity Analysis & Tipping Point Engine**: Calculates exact critical crossover thresholds (e.g., degradation rate where 2-stop beats 1-stop) using Brent's method root-finding down to $10^{-4}$ precision.
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
│   ├── sensitivity.py   # Parameter sweeps, Brent's root-finding, 2D phase maps & regret
│   └── visualization.py # KDE, Histograms, CDFs, Boxplots, 1D/2D Phase Maps & Dashboard
├── tests/
│   ├── test_tyres.py
│   ├── test_fuel.py
│   ├── test_pitstop.py
│   ├── test_strategies.py
│   ├── test_model.py
│   ├── test_simulation.py
│   ├── test_montecarlo.py
│   ├── test_visualization.py
│   ├── test_optimization.py
│   └── test_sensitivity.py
├── run_simulation.py    # Interactive CLI runner with simulation, optimization & sensitivity modes
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

### 4. Evaluate Preset Strategies via CLI
Run the default strategy comparison on Bahrain GP (57 laps):
```bash
python3 run_simulation.py --circuit bahrain
```

### 5. Test Custom Stints
Specify custom compound sequences using shorthand (`S15-M21-S21`, `M26-H31`):
```bash
python3 run_simulation.py --circuit bahrain -s "S15-M21-S21" -s "M26-H31" -s "S12-H45"
```

### 6. Inspect Lap Telemetry
View per-lap physical breakdown (degradation, remaining fuel, effective lap time):
```bash
python3 run_simulation.py --strategy "M26-H31" --laps
```

### 7. Monte Carlo Probabilistic Simulation
Run thousands of iterations introducing real-world stochastic variance:
```bash
# Run 10,000 iterations to calculate Expected Time, Risk (P95), and Head-to-Head Win Probability
python3 run_simulation.py --mc 10000

# Run iterations and generate complete Seaborn visual suite
python3 run_simulation.py --mc 5000 --plot
```

---

## 🐍 Python API Example

```python
from src.config import create_race_model, get_circuit_compounds
from src.sensitivity import CrossoverFinder, PhaseDiagram2D, SensitivityParameter

# 1. Initialize circuit and compounds
model = create_race_model("bahrain")
compounds = get_circuit_compounds("bahrain")

# 2. Find exact critical tipping point where 2-stop beats 1-stop
finder = CrossoverFinder(model, compounds)
c_pt = finder.find_1_vs_2_stop_crossover(
    param=SensitivityParameter.TYRE_DEG_MULTIPLIER,
    search_range=(0.8, 1.6),
)
print(f"Crossover Degradation Multiplier: {c_pt.crossover_value:.4f}")
print(f"Circuit Nominal Baseline: {c_pt.nominal_value:.4f}")
print(f"Distance to Boundary: {c_pt.distance_from_nominal:+.4f}")

# 3. Compute 2D Decision Phase Diagram
phase_mapper = PhaseDiagram2D(model, compounds)
phase_result = phase_mapper.compute_phase_map(
    param_x=SensitivityParameter.PIT_LOSS,
    x_range=(16.0, 32.0),
    param_y=SensitivityParameter.TYRE_DEG_MULTIPLIER,
    y_range=(0.6, 2.2),
)
print(f"Nominal Operating Regime: {phase_result.nominal_stops}-Stop Strategy")
```

---

## 🧪 Testing

Run the automated test suite with `pytest`:
```bash
pytest -v
```

All 64 unit tests verify model mechanics, fuel consumption conservation, pit stop accounting, FIA rule compliance, Bellman DAG optimality, Pareto frontiers, Brent crossover precision, and 2D phase maps.

---

## 🗺️ Roadmap

- [x] **Phase 1: Mathematical Modeling Core & Deterministic Simulation**
- [x] **Phase 2: Monte Carlo Simulation & Stochastic Uncertainty** (lap variance, pit stop delays, wear deviations, distribution visualizer)
- [x] **Phase 3: Strategy Optimization Engine** (Dynamic Programming DAG, Combinatorial Search, tactical pit windows, Pareto risk modeling)
- [x] **Phase 4: Sensitivity Analysis & Decision Phase Maps** (crossover tipping points, Brent's method root-finding, 2D decision phase boundaries, elasticity, and minimax regret)
- [ ] **Phase 5: Real-World Historical Data Integration** (FastF1 / Ergast parameter fitting and race validation)
- [ ] **Phase 6: Interactive Dashboard & Scientific Visualization**

