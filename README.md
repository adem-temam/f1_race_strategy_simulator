# F1 Race Strategy Simulator & Analysis 🏎️⏱️

A data-driven simulation and optimization engine for Formula 1 race strategy. The platform evaluates competing tyre and pit-stop strategies under uncertainty, computes stint pace profiles, models non-linear tyre degradation and fuel burn dynamics, and determines optimal pit windows.

---

## 📌 Features

- **Lap-Time Decomposition Engine**: Synthesizes circuit baseline pace, compound pace offsets, dynamic fuel load, tyre degradation, and track evolution into individual lap times.
- **Realistic Tyre Degradation**: Models linear wear ($\alpha$), quadratic thermal fatigue ($\beta$), and tyre performance cliffs ($\kappa$).
- **Dynamic Fuel Mass Model**: Simulates linear fuel burn per lap with weight sensitivity penalties ($\gamma_{\text{fuel}}$).
- **Pit Loss Modeling**: Accounts for pit lane transit loss and stationary stop duration.
- **Sporting Regulations**: Validates lap count conservation and the mandatory FIA two-compound dry race regulation.
- **Interactive CLI Runner**: Evaluate preset or custom multi-stop strategies across circuits (Bahrain, Barcelona, Monza) with instant telemetry tables.
- **Automated Test Suite**: 18 unit tests validating physical equations, conservation laws, and edge cases.

---

## 📐 Mathematical Formulation

The deterministic lap time on lap $n \in \{1, \dots, N\}$ during stint $k$ on tyre compound $c$ is given by:

$$T_{\text{lap}}(n) = T_{\text{base}} + \Delta_{\text{compound}}(c) + D_c\big(a(n)\big) + F\big(m_f(n)\big) - E_{\text{track}}(n)$$

### 1. Tyre Degradation Model
$$D_c(a) = \alpha_c \cdot a + \beta_c \cdot a^2 + \kappa_c \cdot \big(\max(0, a - a_{\text{cliff}, c})\big)^2$$
* $a$: Tyre age in laps on the current set.
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
│   ├── tyres.py         # TyreCompound dataclass, wear curves & cliff penalties
│   ├── fuel.py          # FuelModel consumption & weight penalty calculations
│   ├── pitstop.py       # PitStopModel transit loss and stationary stop time
│   ├── strategies.py    # Stint, Strategy classes & FIA rule validation
│   ├── model.py         # CircuitConfig, LapRecord, and RaceModel synthesis
│   ├── simulation.py    # Deterministic simulate_race() engine & RaceResult
│   └── config.py        # Circuit presets (Bahrain, Barcelona, Monza) & compounds
├── tests/
│   ├── test_tyres.py
│   ├── test_fuel.py
│   ├── test_pitstop.py
│   ├── test_strategies.py
│   ├── test_model.py
│   └── test_simulation.py
├── run_simulation.py    # Interactive CLI runner
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

### 2. Compare Strategies via CLI
Run the default comparison on Bahrain GP (57 laps):
```bash
python3 run_simulation.py
```

Output:
```text
============================================================================================
  F1 RACE STRATEGY SIMULATION RESULTS: Sakhir (Bahrain GP) (57 Laps)
============================================================================================
Pos  Strategy Name      Stints                         Stops  Total Time      Delta     
--------------------------------------------------------------------------------------------
1    2-Stop (S-M-S)     Soft (15L) -> Medium (21L) -> Soft (21L) 2      1h 30m 20.333s  LEADER    
2    2-Stop (M-H-S)     Medium (20L) -> Hard (20L) -> Soft (17L) 2      1h 30m 26.548s  +6.214s   
3    2-Stop (S-H-M)     Soft (14L) -> Hard (22L) -> Medium (21L) 2      1h 30m 26.963s  +6.630s   
4    1-Stop (M-H)       Medium (26L) -> Hard (31L)     1      1h 30m 27.394s  +7.060s   
============================================================================================
```

### 3. Switch Circuits
Evaluate low-degradation or high-speed tracks:
```bash
python3 run_simulation.py --circuit monza
python3 run_simulation.py --circuit barcelona
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

---

## 🐍 Python API Example

```python
from src.config import create_race_model, get_default_compounds
from src.strategies import Stint, Strategy
from src.simulation import simulate_race

# 1. Initialize track and tyre compounds
model = create_race_model("bahrain")
compounds = get_default_compounds()

# 2. Define competing strategies
strat = Strategy([
    Stint(compound=compounds["Soft"], laps=15),
    Stint(compound=compounds["Medium"], laps=21),
    Stint(compound=compounds["Soft"], laps=21),
], name="2-Stop (S-M-S)")

# 3. Simulate and inspect
result = simulate_race(strat, model)
print(f"Total Race Time: {result.formatted_total_time}")

# 4. Export full telemetry as DataFrame
df = result.to_dataframe()
print(df[["lap", "compound", "tyre_age", "fuel_kg", "effective_time"]].head())
```

---

## 🧪 Testing

Run the automated test suite with `pytest`:
```bash
pytest -v
```

All 18 tests verify model mechanics, fuel consumption conservation, pit stop accounting, and FIA rule compliance.

---

## 🗺️ Roadmap

- [x] **Phase 1: Mathematical Modeling Core & Deterministic Simulation**
- [ ] **Phase 2: Monte Carlo Simulation & Stochastic Uncertainty** (lap variance, pit stop delays, wear deviations)
- [ ] **Phase 3: Strategy Optimization Engine** (automated pit window discovery via continuous & discrete search)
- [ ] **Phase 4: Sensitivity Analysis & Decision Phase Maps** (crossover points and 2D parameter heatmaps)
- [ ] **Phase 5: Real-World Historical Data Integration** (FastF1 / Ergast parameter fitting and race validation)
- [ ] **Phase 6: Interactive Dashboard & Scientific Visualization**
