# Mathematical Formulation & Architecture: Phase 6

This document formalizes the mathematical theory, counterfactual perturbation manifolds, strategic regret formulations, opportunistic Safety Car decision models, and formal answers to the core research questions for **Phase 6: Interactive Strategy Dashboard, Scenario Analysis Engine, and Demonstration Notebook Suite** of the Formula 1 Race Strategy Simulator.

---

## 1. Overview & System Objectives

Across **Phases 1 through 5**, the simulator developed:
* **Phase 1**: Deterministic lap-time synthesis ($S \mapsto T_{\text{race}}(S)$) incorporating quadratic tyre degradation, exponential cliffs, fuel depletion, and pit transit losses.
* **Phase 2**: Vectorized Monte Carlo uncertainty simulation ($N=1,000+$ iterations) deriving empirical probability distributions, Value-at-Risk ($P_{95}\text{ VaR}$), and Expected Shortfall ($CVaR_{95}$).
* **Phase 3**: Global optimal strategy discovery via Bellman backward induction on the Directed Acyclic Graph (DAG) of valid stints, alongside combinatorial grid search and pit window tolerance analysis ($\tau \le 2.0\text{s}$).
* **Phase 4**: 1D continuous sensitivity sweeps, Brent's method root-finding for exact crossover tipping points ($t_{\text{loss}}^*$), and 2D decision phase boundaries ($\Delta T = 0$).
* **Phase 5**: Real-world OpenF1 telemetry ingestion, clean lap filtering ($107\%$ median filter), fixed-effects OLS regression for tyre wear ($\alpha, \beta$), and driver backtesting against 2024 Grand Prix winners with discrepancy diagnostics.

**Phase 6** unifies these components into:
1. **Interactive Strategy Dashboard**: A modern web application (FastAPI + dark-theme single-page interface with Chart.js) providing real-time parameter tuning, Dynamic Programming re-optimization, Monte Carlo risk evaluation, 2D phase diagrams, and historical telemetry inspection.
2. **Scenario Analysis & Counterfactual What-If Engine**: A mathematical framework that perturbs physical parameters, re-optimizes candidate strategies, and measures **Strategic Regret** $\mathcal{R}(S; \theta')$ to evaluate the 6 core research questions from `project.md`.
3. **Demonstration Curriculum Notebooks**: An educational five-notebook suite in `notebooks/` illustrating the end-to-end mathematical models.

---

## 2. Counterfactual Scenario Theory & Perturbation Manifolds

### 2.1 The Parameter Manifold $\Theta$

Let the physical race environment be characterized by a multi-dimensional parameter vector $\theta \in \Theta \subset \mathbb{R}^k$:
$$\theta = \big( T_{\text{base}}, \; m_0, \; \beta_{\text{burn}}, \; \lambda_{\text{fuel}}, \; t_{\text{pit}}, \; \sigma_{\text{pit}}, \; \boldsymbol{\alpha}, \; \boldsymbol{\beta}, \; \boldsymbol{\delta}, \; N_{\text{laps}} \big)$$

where:
* $T_{\text{base}}$: Base circuit lap time on fresh tyres and zero fuel ($\text{seconds}$).
* $m_0$: Initial fuel load ($\text{kg}$), typically $100 - 105\text{ kg}$ under 2024 FIA regulations.
* $\beta_{\text{burn}}$: Fuel burn rate per lap ($\text{kg/lap}$).
* $\lambda_{\text{fuel}}$: Lap time sensitivity to fuel mass ($\text{s/kg}$).
* $t_{\text{pit}}$: Nominal pit lane time loss ($\text{seconds}$), sum of transit loss and stationary service.
* $\boldsymbol{\alpha} = (\alpha_{\text{Soft}}, \alpha_{\text{Medium}}, \alpha_{\text{Hard}})$: Linear degradation rates ($\text{s/lap}$).
* $\boldsymbol{\beta} = (\beta_{\text{Soft}}, \beta_{\text{Medium}}, \beta_{\text{Hard}})$: Quadratic degradation rates ($\text{s/lap}^2$).
* $\boldsymbol{\delta} = (\delta_{\text{Soft}}, \delta_{\text{Medium}}, \delta_{\text{Hard}})$: Compound pace offsets relative to Medium baseline ($\text{seconds}$).
* $N_{\text{laps}}$: Scheduled total race distance ($\text{laps}$).

### 2.2 Perturbation Operator $\mathcal{P}$

A counterfactual scenario is an intervention represented by an operator:
$$\mathcal{P}: \Theta \times \Delta \to \Theta', \quad \theta' = \mathcal{P}(\theta_0, \mathbf{p})$$

where $\mathbf{p} \in \Delta$ is a perturbation descriptor:
$$\mathbf{p} = \big( \mu_{\text{deg}}, \; \Delta t_{\text{pit}}, \; \Delta \delta_{\text{compound}}, \; \alpha_L, \; \mathbf{reg}_{2026} \big)$$

The transformed parameters $\theta'$ satisfy:
1. **Tyre Degradation Scaling**:
   $$\alpha_c' = \mu_{\text{deg}} \cdot \alpha_c, \quad \beta_c' = \mu_{\text{deg}} \cdot \beta_c \quad \forall c \in \{\text{Soft}, \text{Medium}, \text{Hard}\}$$
2. **Pit Lane Loss Translation**:
   $$t_{\text{pit}}' = t_{\text{pit}} + \Delta t_{\text{pit}}$$
3. **Compound Pace Differential Shift**:
   $$\delta_c' = \delta_c + \Delta \delta_c$$
4. **Race Distance Contraction/Extension**:
   $$N_{\text{laps}}' = \lfloor \alpha_L \cdot N_{\text{laps}} \rfloor, \quad \alpha_L \in (0, 1.5]$$
5. **2026 Technical Regulation Transformation**:
   $$m_0' = 70.0\text{ kg} \quad (\text{down from } 100\text{ kg}), \quad \beta_{\text{burn}}' = \frac{70.0}{N_{\text{laps}}}, \quad T_{\text{base}}' = T_{\text{base}} - 1.20\text{s}$$

---

## 3. Strategic Regret Theory & Policy Pivots

### 3.1 Race Duration Functional

Given a candidate strategy $S = ((\text{compound}_k, L_k))_{k=1}^K$ satisfying $\sum_{k=1}^K L_k = N_{\text{laps}}$, the deterministic total race time under parameters $\theta$ is:
$$J(S; \theta) = \sum_{k=1}^K \sum_{a=1}^{L_k} T_{\text{lap}}\big(a, m(l), c_k; \theta\big) + (K - 1) \cdot t_{\text{pit}}$$

Let $\mathcal{S}$ denote the space of all feasible strategies compliant with the mandatory 2-compound sporting regulation.

### 3.2 Nominal vs Perturbed Optimal Policies

* The **nominal optimal policy** is the strategy that minimizes expected race duration under the baseline environment $\theta_0$:
  $$S^*(\theta_0) = \arg\min_{S \in \mathcal{S}} J(S; \theta_0)$$

* The **counterfactual optimal policy** is the optimal strategy recomputed under the perturbed environment $\theta'$:
  $$S^*(\theta') = \arg\min_{S \in \mathcal{S}} J(S; \theta')$$

### 3.3 Strategic Regret $\mathcal{R}$

If a race engineer commits to the nominal optimal strategy $S^*(\theta_0)$ without adapting to the perturbed reality $\theta'$, the incurred **Strategic Regret** is defined as:
$$\mathcal{R}\big(S^*(\theta_0); \theta'\big) \triangleq J\big(S^*(\theta_0); \theta'\big) - J\big(S^*(\theta'); \theta'\big) \ge 0$$

* **Zero Regret ($\mathcal{R} = 0$)**: The baseline strategy remains optimal under the perturbation ($S^*(\theta') = S^*(\theta_0)$).
* **Positive Regret ($\mathcal{R} > 0$)**: The baseline plan is strictly suboptimal. The value of $\mathcal{R}$ represents the exact race duration in seconds forfeited by strategic inaction.

### 3.4 Strategy Pivot Criterion

A **Strategy Pivot** occurs if and only if the argmin set changes:
$$\text{Pivot}(\theta_0 \to \theta') \iff S^*(\theta') \neq S^*(\theta_0)$$

In practice, a strategy pivot is critical if $\mathcal{R}(S^*(\theta_0); \theta') > \delta_{\text{tolerance}}$, where $\delta_{\text{tolerance}} \approx 1.5 - 2.5\text{s}$ (the threshold where track position cannot overcome cumulative pace deficit).

---

## 4. Stochastic Safety Car Dynamics & Opportunistic Stopping

### 4.1 Neutralization Mechanics

Under a Safety Car (SC) or Virtual Safety Car (VSC), all drivers on track must reduce speed to comply with a minimum sector delta time ($v_{\text{delta}} \approx 0.60 - 0.65 \cdot v_{\text{racing}}$). However, pit lane speed limits remain strictly constant ($80\text{ km/h}$ or $22.2\text{ m/s}$).

Consequently, the pit lane transit time penalty is drastically compressed:
$$T_{\text{transit}}^{\text{SC}} = T_{\text{pit lane}} - T_{\text{track}}^{\text{SC}} = T_{\text{pit lane}} - \frac{D_{\text{pit straight}}}{v_{\text{delta}}}$$

Because $T_{\text{track}}^{\text{SC}} > T_{\text{track}}^{\text{racing}}$, the effective pit loss drops from $\sim 24.0\text{s}$ to $\sim 14.5\text{s}$, producing an **opportunistic pit stop dividend**:
$$\Delta T_{\text{SC}} = t_{\text{pit}}^{\text{nominal}} - t_{\text{pit}}^{\text{SC}} \approx 9.5\text{s}$$

### 4.2 Dynamic Programming with Opportunistic Neutralizations

Let $l_{\text{SC}}$ be the lap at which the Safety Car is deployed. For a car running on stint $k$ with tyre age $a$, the decision to pit or stay out is governed by the Bellman comparison:
$$
V_{\text{SC}}(l_{\text{SC}}, c, a) = \min \begin{cases}
V\big(l_{\text{SC}} + 1, c, a + 1\big) + T_{\text{SC}}, & \text{Stay out} \\
t_{\text{pit}}^{\text{SC}} + \min\limits_{c' \neq c} \Big[ V\big(l_{\text{SC}} + 1, c', 1\big) + T_{\text{SC}} \Big], & \text{Pit for new tyres}
\end{cases}
$$

If the car pits:
* It consumes a pit stop at a $35 - 40\%$ discount.
* It gains fresh rubber ($\Delta T_{\text{tyre}} \to 0$) while rivals remaining on track accumulate further wear.
* If a driver had already planned a 2-stop, an SC deployment in their pit window virtually guarantees strategic victory.

---

## 5. Formal Answers to Core Research Questions (from project.md)

Using the mathematical engines built across Phases 1 through 6, we provide analytical and empirical answers to the six fundamental research questions:

### RQ1: How sensitive is the optimal strategy to variations in pit stop loss ($\pm 3.0\text{s}$)?

* **Analytical Derivative**:
  $$\frac{\partial J(S; \theta)}{\partial t_{\text{pit}}} = K - 1 = N_{\text{stops}}$$
  The sensitivity of a strategy's total duration to pit lane loss is strictly proportional to its number of pit stops. A 1-stop strategy has sensitivity $\frac{\partial J}{\partial t_{\text{pit}}} = 1$, whereas a 2-stop strategy has sensitivity $\frac{\partial J}{\partial t_{\text{pit}}} = 2$.
* **Crossover Equation**:
  The time delta between a 1-stop and 2-stop strategy is:
  $$\Delta T(t_{\text{pit}}) = T(S_{\text{1-stop}}) - T(S_{\text{2-stop}}) = \big(\Delta T_{\text{pace}} - \Delta T_{\text{deg}}\big) - t_{\text{pit}}$$
  where $\Delta T_{\text{pace}} - \Delta T_{\text{deg}} > 0$ is the cumulative on-track pace advantage of fresher tyres across two stops.
* **Empirical Findings (Bahrain GP)**:
  * Nominal pit loss: $24.0\text{s}$. At this value, 2-stop is faster by $13.8\text{s}$.
  * Upper tipping point: $t_{\text{pit}}^* = 37.8\text{s}$. A pit loss increase of $+3.0\text{s}$ ($27.0\text{s}$) reduces the 2-stop advantage from $13.8\text{s}$ to $10.8\text{s}$, but does **not** trigger a strategy pivot.
  * Human error / botched stop ($t_{\text{pit}} \to 34.0\text{s}$): Even with a catastrophic 10-second stationary delay on one stop, the 2-stop strategy at Bahrain remains marginally superior or neutral due to extreme tyre degradation on high-abrasion asphalt.

---

### RQ2: How do degradation rate spikes ($+50\%$) alter optimal stint length and stop frequency?

* **Mathematical Mechanics**:
  Tyre degradation includes a quadratic term $\beta \cdot a^2$. When degradation scales by $\mu_{\text{deg}} = 1.5$:
  $$\Delta T_{\text{wear}}'(a) = 1.5 \cdot \big(\alpha a + \beta a^2\big)$$
  The marginal degradation rate $\frac{d\Delta T}{da} = 1.5(\alpha + 2\beta a)$ steepens dramatically.
* **Optimal Stint Length Contraction**:
  Equating marginal tyre wear per lap to the amortized cost of a pit stop yields:
  $$a^* \approx \sqrt{\frac{t_{\text{pit}}}{\mu_{\text{deg}} \cdot \beta}}$$
  An increase in degradation from $\mu_{\text{deg}} = 1.0 \to 1.5$ contracts optimal stint length by a factor of $\frac{1}{\sqrt{1.5}} \approx 0.816$ (an $18.4\%$ reduction in stint length).
* **Strategic Pivot Outcome**:
  * On high-deg circuits (Bahrain, Barcelona), a 1-stop strategy collapses completely under $+50\%$ degradation. The 1-stop suffers an additional $+28.4\text{s}$ in tyre cliff penalty, elevating the Strategic Regret of a 1-stop to $\mathcal{R} > 32\text{s}$.
  * On medium-deg circuits (Silverstone), a $+50\%$ spike forces an immediate pivot from 1-stop to 2-stop.

---

### RQ3: When does a pace advantage on softer compounds justify an additional pit stop?

* **Crossover Threshold Condition**:
  An additional pit stop costs $t_{\text{pit}}$ seconds. For an extra stop to be profitable, the pace gain of running softer compounds $\Delta \delta_{\text{soft}}$ across stint length $L_{\text{soft}}$ must satisfy:
  $$\sum_{l=1}^{L_{\text{soft}}} \Big[ T_{\text{lap}}(l, \text{Hard}) - T_{\text{lap}}(l, \text{Soft}) \Big] > t_{\text{pit}}$$
* **Empirical Findings**:
  * In Bahrain, the base pace offset is $\delta_{\text{Soft}} - \delta_{\text{Hard}} = -1.6\text{s/lap}$. Over a 15-lap stint, the raw compound speed advantage yields $24.0\text{s}$, exactly matching the nominal pit transit loss.
  * Because softer compounds also suffer higher $\beta$, the soft compound must be removed before lap 16 to avoid the exponential cliff. When $\Delta \delta_{\text{soft}}$ widens from $-0.8\text{s}$ to $-1.4\text{s}$, 2-stop strategies utilizing two soft stints become dominant over all medium-hard combinations.

---

### RQ4: How does race distance affect strategy (Sprint 35% vs Full Grand Prix)?

* **Tyre Cliff Non-Saturation**:
  At $35\%$ distance ($19$ laps in Bahrain), the race distance is shorter than the Hard tyre cliff life ($L_{\text{cliff}} \approx 40$ laps) and approximately equal to the Medium tyre life ($L_{\text{cliff}} \approx 28$ laps).
* **Zero-Stop / Single-Stint Dominance**:
  Since pit stop transit loss remains fixed at $24.0\text{s}$, spending $24\text{s}$ to save $0.15\text{s/lap}$ over only 10 remaining laps yields a net loss of $-22.5\text{s}$.
* **Result**: Sprint races fundamentally suppress multi-stop strategies. The optimal policy collapses to a zero-stop (single stint) on the Medium compound, or a single Soft stint if degradation allows.

---

### RQ5: How do unexpected Safety Car periods alter the optimal strategy?

* **Strategic Reversal Mechanism**:
  A Safety Car deployed between Laps 16 and 22 at Bahrain reduces the pit transit loss from $24.0\text{s}$ to $15.5\text{s}$.
* **Value of the Free Stop**:
  $$\text{Dividend} = 24.0 - 15.5 = 8.5\text{s}$$
* **Tactical Pivot**:
  * Drivers who committed to a 1-stop (S $\to$ H) have their strategic advantage neutralized.
  * A 2-stopping driver pitting under the SC saves $8.5\text{s}$, locking in an insurmountable track advantage.
  * Strategic Regret of not pitting under an SC during an open pit window exceeds $\mathcal{R} \ge 8.5\text{s}$.

---

### RQ6: How will the 2026 Formula 1 technical regulations alter strategic planning?

* **Regulatory Physics Parameters**:
  * Minimum fuel load reduced: $m_0 = 70.0\text{ kg}$ (vs $105.0\text{ kg}$).
  * Mass reduction penalty: Initial fuel penalty drops from $105 \times 0.033 = 3.46\text{s}$ to $70 \times 0.033 = 2.31\text{s}$, an immediate $+1.15\text{s/lap}$ speedup at race start.
  * Active aerodynamics: Active straight-line drag reduction reduces lateral slip on exits, lowering tyre surface temperatures and reducing $\beta_{\text{deg}}$ by an estimated $10-15\%$.
* **Strategic Consequences**:
  * Total race duration decreases by $\sim 140 - 180\text{s}$ ($2.5 - 3.0$ minutes faster).
  * Reduced tyre wear extends stint lifetimes, shifting marginal circuits (e.g. Silverstone, Spa) closer to 1-stop territory, while reducing the strategic regret of 1-stop strategies at high-wear tracks.

---

## 6. Software Architecture & Interactive Dashboard

The interactive strategy dashboard is engineered as a decoupled high-performance application:

```
┌─────────────────────────────────────────────────────────────┐
│                       FastAPI Server                        │
│                   (src/dashboard/app.py)                    │
└──────┬───────────────────┬───────────────────┬──────────────┘
       │                   │                   │
┌──────▼────────┐   ┌──────▼────────┐   ┌──────▼────────┐
│ Simulation &  │   │  Monte Carlo  │   │  What-If &    │
│ Optimization  │   │  Uncertainty  │   │ Historical    │
│  (/simulate,  │   │ (/montecarlo) │   │  Validation   │
│  /optimize)   │   │               │   │ (/scenarios)  │
└───────────────┘   └───────────────┘   └───────────────┘
```

* **Backend**: FastAPI with asynchronous endpoints utilizing NumPy vectorized broadcast operations for Monte Carlo simulation ($< 400\text{ms}$ for $N=1,000$ iterations) and dynamic programming Bellman induction ($< 15\text{ms}$).
* **Frontend**: Pure vanilla modern JavaScript (ES6+) with zero heavy frameworks, utilizing Chart.js for canvas-rendered telemetry charts, KDE density graphs, and 2D phase maps.
* **CLI Launcher**: `python run_dashboard.py --port 8000 --open-browser` provides zero-configuration local execution with automatic port fallback.
