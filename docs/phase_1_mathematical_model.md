# Mathematical Formulation & Architecture: Phase 1

This document formalizes the mathematical model and software design for **Phase 1: Mathematical Modeling Core & Deterministic Simulation** of the Formula 1 Race Strategy Simulator.

---

## 1. Overview & System Objectives

The core question of Phase 1 is:
> **Given deterministic car, circuit, fuel, and tyre characteristics, what is the exact lap-by-lap pace profile and total race time for any candidate tyre pit-stop strategy?**

Phase 1 provides the mathematical foundation upon which subsequent phases (Monte Carlo uncertainty, optimization algorithms, and historical parameter fitting) are constructed.

---

## 2. Mathematical Modeling

### 2.1 Race Strategy Representation
Let $N \in \mathbb{N}$ denote the total number of race laps.

A strategy $S$ is defined as an ordered sequence of $K$ stints:
$$S = \big[ (c_1, \ell_1), (c_2, \ell_2), \dots, (c_K, \ell_K) \big]$$

where:
* $K \ge 1$ is the total number of stints.
* $c_k \in \mathcal{C}$ is the tyre compound used in stint $k$ ($\mathcal{C} = \{\text{Soft}, \text{Medium}, \text{Hard}\}$).
* $\ell_k \in \mathbb{N}_{\ge 1}$ is the length of stint $k$ in completed laps.

#### Constraints & FIA Regulations
1. **Lap Conservation:** The sum of stint lengths must equal total race distance:
   $$\sum_{k=1}^K \ell_k = N$$

2. **Pit Stop Scheduling:** The strategy implies $K - 1$ pit stops. Pit stop $k$ occurs at the end of lap $p_k$:
   $$p_k = \sum_{j=1}^k \ell_j \quad \text{for } k \in \{1, \dots, K - 1\}$$

3. **Mandatory Compound Diversity (FIA Sporting Regulations):** In a dry race, a driver must use at least two distinct dry-weather tyre specifications:
   $$\big| \{c_1, c_2, \dots, c_K\} \big| \ge 2$$

---

### 2.2 Lap-Time Decomposition
For any lap $n \in \{1, 2, \dots, N\}$, let:
* $k(n) \in \{1, \dots, K\}$ be the active stint index.
* $c = c_{k(n)}$ be the active tyre compound.
* $a(n) \in \{1, 2, \dots, \ell_{k(n)}\}$ be the tyre age (number of laps completed on the current set of tyres).
* $m_f(n)$ be the remaining fuel mass (kg) at the start of lap $n$.

The deterministic lap time $T_{\text{lap}}(n)$ is decomposed into physical components:
$$T_{\text{lap}}(n) = T_{\text{base}} + \Delta_{\text{compound}}(c) + D_c\big(a(n)\big) + F\big(m_f(n)\big) - E_{\text{track}}(n)$$

#### Parameter Definitions
| Symbol | Component | Description | Units |
| :--- | :--- | :--- | :--- |
| $T_{\text{base}}$ | Baseline Circuit Pace | Lap time of reference car at zero fuel with baseline reference tyre | seconds (s) |
| $\Delta_{\text{compound}}(c)$ | Compound Inherent Delta | Baseline pace offset of compound $c$ relative to Medium ($\Delta_{\text{Med}} = 0$) | seconds (s) |
| $D_c(a)$ | Tyre Degradation | Time lost due to wear, thermal degradation, and surface abrasion at tyre age $a$ | seconds (s) |
| $F(m_f)$ | Fuel Load Penalty | Time penalty caused by carrying remaining fuel mass $m_f$ | seconds (s) |
| $E_{\text{track}}(n)$ | Track Evolution | Pace improvement as rubber is deposited on track over distance | seconds (s) |

---

### 2.3 Tyre Degradation Model $D_c(a)$
Tyre degradation represents the progressive loss of grip as the tyre undergoes thermal cycles, graining, and tread rubber depletion.

For compound $c$ at tyre age $a$:
$$D_c(a) = \alpha_c \cdot a + \beta_c \cdot a^2 + \text{Cliff}_c(a)$$

where:
* $\alpha_c \ge 0$: Linear degradation coefficient ($\text{s} \cdot \text{lap}^{-1}$).
* $\beta_c \ge 0$: Quadratic degradation coefficient ($\text{s} \cdot \text{lap}^{-2}$), modeling non-linear wear acceleration.
* $\text{Cliff}_c(a)$: High degradation penalty once tyre surpasses critical age $a_{\text{cliff}, c}$:
  $$\text{Cliff}_c(a) = \kappa_c \cdot \left(\max\big(0, \, a - a_{\text{cliff}, c}\big)\right)^2$$
  where $\kappa_c$ governs the severity of the cliff drop-off.

#### Compound Trade-off Properties
$$\Delta_{\text{Soft}} < \Delta_{\text{Medium}} < \Delta_{\text{Hard}}$$
$$\alpha_{\text{Soft}} > \alpha_{\text{Medium}} > \alpha_{\text{Hard}}$$
* **Soft tyre**: Highest peak grip (lowest initial lap time), but steep degradation slope.
* **Hard tyre**: Lower peak grip (higher initial lap time), but gentle degradation slope suitable for long stints.

---

### 2.4 Fuel Consumption & Penalty Model $F(m_f)$
Formula 1 cars start with a maximum permissible race fuel load and burn fuel almost linearly across laps.

> [!NOTE]
> **Regulatory Evolution Note**: Initial iterations of the mathematical model were configured using outdated FIA regulations ($100 - 110\text{ kg}$ maximum fuel capacity, $m_{f, 0} = 105.0\text{ kg}$, $798\text{ kg}$ vehicle minimum mass, and $\gamma_{\text{fuel}} \approx 0.033\text{ s/kg}$). Under modernized FIA regulations incorporating 100% sustainable fuels and upgraded hybrid electrical delivery (50% ICE / 50% MGU-K), fuel capacity is reduced to $70 - 75\text{ kg}$ (standard benchmark $m_{f, 0} = 75.0\text{ kg}$), vehicle minimum weight drops to $768\text{ kg}$, and fuel sensitivity decreases to $\gamma_{\text{fuel}} \approx 0.025 - 0.028\text{ s/kg}$.

Let:
* $m_{f, 0}$: Initial fuel mass at race start ($n = 1$).
* $m_{f, \text{res}}$: Minimum reserve fuel mass remaining at race end (for FIA scrutineering sample).
* $\Delta m_f$: Fuel burn rate per lap:
  $$\Delta m_f = \frac{m_{f, 0} - m_{f, \text{res}}}{N}$$

The remaining fuel mass on lap $n$ is:
$$m_f(n) = m_{f, 0} - (n - 1) \cdot \Delta m_f$$

The lap-time penalty induced by fuel mass is directly proportional to weight:
$$F\big(m_f(n)\big) = \gamma_{\text{fuel}} \cdot m_f(n)$$

where:
* $\gamma_{\text{fuel}}$ is the fuel sensitivity coefficient ($\approx 0.025 - 0.028\text{ s} \cdot \text{kg}^{-1}$ under modernized regulations, meaning $10\text{ kg}$ of fuel costs $\approx 0.28\text{ s}$ per lap; note that the previous formula was using the outdated FIA regulation value of $\approx 0.033\text{ s} \cdot \text{kg}^{-1}$).

---

### 2.5 Pit Stop Time Loss Model
In Formula 1, entering the pit lane incurs a net time loss known as the **pit loss delta** ($\Delta T_{\text{pit}}$):
$$\Delta T_{\text{pit}} = T_{\text{transit}} + T_{\text{stationary}} - T_{\text{track, equiv}}$$

where:
* $T_{\text{transit}}$ is the transit time through the pit lane at pit speed limit (e.g. $80\text{ km/h}$).
* $T_{\text{stationary}}$ is the time the car is stationary for the 4-wheel change (typically $\approx 2.2 - 2.8\text{ s}$).
* $T_{\text{track, equiv}}$ is the time a car on the racing circuit would have taken to travel the equivalent distance between pit entry and pit exit at racing speeds.

In modern F1 circuits, net pit loss $\Delta T_{\text{pit}}$ typically ranges from $19.0\text{ s}$ to $25.0\text{ s}$.

---

### 2.6 Total Race Time Formulation
The total race time for strategy $S$ is the sum of all individual lap times plus the pit loss incurred at each scheduled pit stop:
$$T_{\text{race}}(S) = \sum_{n=1}^N T_{\text{lap}}(n) + \sum_{k=1}^{K-1} \Delta T_{\text{pit}}$$

Alternatively, pit loss can be recorded directly onto the pit-in lap $p_k$:
$$T_{\text{lap, eff}}(n) = T_{\text{lap}}(n) + \mathbb{I}_{\{n \in \{p_1, \dots, p_{K-1}\}\}} \cdot \Delta T_{\text{pit}}$$
such that:
$$T_{\text{race}}(S) = \sum_{n=1}^N T_{\text{lap, eff}}(n)$$

---

## 3. Reference Circuit Benchmark: Sakhir (Bahrain GP)

To validate the model with realistic numbers:
* **Circuit Laps:** $N = 57$ laps
* **Baseline Pace:** $T_{\text{base}} = 92.000\text{ s}$ ($1\text{m } 32.000\text{s}$)
* **Pit Loss Delta:** $\Delta T_{\text{pit}} = 22.500\text{ s}$
* **Fuel Parameters:**
  - $m_{f, 0} = 75.0\text{ kg}$ (modernized standard regulation; note the original formula was using the outdated FIA regulation of $105.0\text{ kg}$), $m_{f, \text{res}} = 2.0\text{ kg}$
  - Burn rate $\Delta m_f \approx \frac{75.0 - 2.0}{57} \approx 1.281\text{ kg/lap}$ ($\approx 1.807\text{ kg/lap}$ under outdated $105\text{ kg}$ rules)
  - Sensitivity $\gamma_{\text{fuel}} = 0.028\text{ s/kg}$ (adjusted from outdated $0.033\text{ s/kg}$)
* **Compounds:**
  - **Soft (C3):** $\Delta_{\text{Soft}} = -0.650\text{ s}$, $\alpha = 0.110\text{ s/lap}$, $\beta = 0.0015\text{ s/lap}^2$, cliff at lap 18 ($\kappa = 0.05$).
  - **Medium (C2):** $\Delta_{\text{Med}} = 0.000\text{ s}$, $\alpha = 0.065\text{ s/lap}$, $\beta = 0.0008\text{ s/lap}^2$, cliff at lap 28 ($\kappa = 0.04$).
  - **Hard (C1):** $\Delta_{\text{Hard}} = +0.700\text{ s}$, $\alpha = 0.035\text{ s/lap}$, $\beta = 0.0003\text{ s/lap}^2$, cliff at lap 40 ($\kappa = 0.03$).

---

## 4. Software Architecture for Phase 1

```text
src/
├── __init__.py
├── tyres.py         # TyreCompound dataclass, degradation & cliff computation
├── fuel.py          # FuelModel dataclass, consumption tracking & penalty
├── pitstop.py       # PitStopModel dataclass, transit loss and delta
├── strategies.py    # Stint, Strategy dataclasses & FIA rule validation
├── model.py         # CircuitConfig, LapRecord, and RaceModel synthesis
├── simulation.py    # RaceResult, deterministic simulate_race() engine
└── config.py        # Realistic circuit & compound presets (Bahrain, Barcelona, etc.)
```
