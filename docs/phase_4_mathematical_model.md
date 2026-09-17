# Mathematical Formulation & Architecture: Phase 4

This document formalizes the mathematical theory, analytical derivations, numerical methods, and software architecture for **Phase 4: Sensitivity Analysis & Decision Phase Maps** of the Formula 1 Race Strategy Simulator.

---

## 1. Overview & System Objectives

In **Phase 1**, we established the deterministic lap-time evaluation engine:
$$S \mapsto T_{\text{race}}(S)$$

In **Phase 2**, we introduced stochastic Monte Carlo distributions:
$$S \mapsto \mathbb{E}\big[T_{\text{race}}(S)\big], \quad \text{Var}\big(T_{\text{race}}(S)\big), \quad P_{95}\big(T_{\text{race}}(S)\big)$$

In **Phase 3**, we implemented global combinatorial and dynamic programming strategy discovery:
$$S^*(\mathbf{\theta}) = \arg\min_{S \in \mathcal{S}_{\text{valid}}} \mathcal{J}(S; \mathbf{\theta})$$

In **Phase 4**, the system transitions from finding the optimal strategy for a single static parameter set to **investigating the structure of the decision space across parameter variations**:
> **Given a race environment parameterized by $\mathbf{\theta} = (\mu_{\text{deg}}, t_{\text{pit}}, \mathbf{\Delta}_{\text{compound}}, E_{\text{track}}, \gamma_{\text{fuel}})$, why does a strategy become optimal, when does the optimal strategy transition (tipping points), where are the decision boundaries in multi-dimensional parameter space, and how robust is a chosen strategy to environmental misestimation?**

---

## 2. Analytical Theory: The Strategy Trade-off Mechanics

### 2.1 The Fundamental Trade-off Equation
Consider an $N$-lap Grand Prix. Let $S_1$ denote an optimal 1-stop strategy (2 stints) and $S_2$ denote an optimal 2-stop strategy (3 stints).

The total race time difference is:
$$\Delta T(\mathbf{\theta}) = T_{\text{race}}(S_2; \mathbf{\theta}) - T_{\text{race}}(S_1; \mathbf{\theta}) = \Big[ T_{\text{track}}(S_2; \mathbf{\theta}) - T_{\text{track}}(S_1; \mathbf{\theta}) \Big] + \Delta T_{\text{pit}}$$

where:
* $\Delta T_{\text{pit}} = t_{\text{pit}}$ is the net pit loss penalty incurred by taking an additional stop.
* $T_{\text{track}}(S)$ is the net elapsed time on track (excluding stationary and transit pit loss).

The 2-stop strategy is strictly faster if and only if:
$$\Delta T_{\text{pace}} = T_{\text{track}}(S_1; \mathbf{\theta}) - T_{\text{track}}(S_2; \mathbf{\theta}) > t_{\text{pit}}$$

The strategy problem fundamentally reduces to whether running shorter stints on fresher rubber and softer compounds recovers more on-track time than the pit lane transit loss $t_{\text{pit}}$.

---

### 2.2 Scaling Laws of Tyre Degradation
Let compound $c$ have linear wear coefficient $\alpha_c$ and quadratic wear coefficient $\beta_c$, modulated by circuit degradation multiplier $\mu_{\text{deg}}$. The degradation at tyre age $a$ is:
$$D_c(a) = \mu_{\text{deg}} \cdot \big( \alpha_c a + \beta_c a^2 \big)$$

The cumulative degradation time lost over a stint of length $\ell$ is given by the integral (or discrete summation):
$$\int_0^\ell D_c(a) \, da = \mu_{\text{deg}} \left[ \frac{\alpha_c \ell^2}{2} + \frac{\beta_c \ell^3}{3} \right]$$

For a strategy dividing the race into $K$ equal stints of length $\ell = N / K$:
$$\text{Total Degradation Time}(K) = K \cdot \mu_{\text{deg}} \left[ \frac{\alpha (N/K)^2}{2} + \frac{\beta (N/K)^3}{3} \right] = \mu_{\text{deg}} \left[ \frac{\alpha N^2}{2K} + \frac{\beta N^3}{3K^2} \right]$$

#### Comparison between 1-Stop ($K=2$) and 2-Stop ($K=3$):
$$\text{Degradation}(K=2) = \mu_{\text{deg}} \left[ \frac{\alpha N^2}{4} + \frac{\beta N^3}{12} \right]$$
$$\text{Degradation}(K=3) = \mu_{\text{deg}} \left[ \frac{\alpha N^2}{6} + \frac{\beta N^3}{27} \right]$$

The tyre degradation savings gained by adding a second stop is:
$$\Delta \text{Deg}(K=2 \to K=3) = \mu_{\text{deg}} \left[ \frac{\alpha N^2}{12} + \frac{5 \beta N^3}{108} \right]$$

#### Mathematical Insights:
1. **Linearity in $\mu_{\text{deg}}$**: The on-track degradation savings scale directly with $\mu_{\text{deg}}$.
2. **Super-Linearity in Race Distance $N$**: The savings scale with $N^2$ and $N^3$. Consequently, longer races (e.g. Barcelona 66 laps vs Monza 53 laps) heavily penalize 1-stop strategies.
3. **Compound Cliff Trigger**: If any stint length $\ell > a_{\text{cliff}}$, an additional penalty $\kappa_c (a - a_{\text{cliff}})^2$ triggers, creating an exponential divergence where 1-stop times rapidly explode.

---

## 3. Mathematical Formulations of Sensitivity Analysis

### 3.1 1D Parameter Response Curves & Lower Envelope
Let $\theta \in [\theta_{\min}, \theta_{\max}]$ denote a single scalar parameter (e.g. $\mu_{\text{deg}}$ or $t_{\text{pit}}$), holding all other parameters $\mathbf{\theta}_{-\theta}$ fixed.

For a finite set of competing candidate strategies $\mathcal{S} = \{S_1, S_2, \dots, S_M\}$:
$$T_m(\theta) = T_{\text{race}}(S_m; \theta)$$

The **Lower Envelope (Global Optimal Frontier)** is:
$$\mathcal{L}(\theta) = \min_{S \in \mathcal{S}} T_{\text{race}}(S; \theta)$$

The parameter space $[\theta_{\min}, \theta_{\max}]$ is partitioned into contiguous optimality intervals:
$$\mathcal{I}_k = \big\{ \theta \in [\theta_{\min}, \theta_{\max}] \;\big|\; S_k = \arg\min_{S \in \mathcal{S}} T_{\text{race}}(S; \theta) \big\}$$

The boundaries between these intervals are the **critical tipping points** (crossover values).

---

### 3.2 Marginal Sensitivity & Dimensionless Elasticity
To quantify how sensitive a strategy $S$ is to environmental fluctuations around nominal baseline $\theta_0$:

1. **Marginal Sensitivity (Gradient)**:
   $$\mathcal{S}_\theta(S) = \left. \frac{\partial T_{\text{race}}(S; \theta)}{\partial \theta} \right|_{\theta_0} \approx \frac{T_{\text{race}}(S; \theta_0 + \epsilon) - T_{\text{race}}(S; \theta_0 - \epsilon)}{2\epsilon}$$

2. **Dimensionless Elasticity**:
   $$E_\theta(S) = \left. \frac{\partial \ln T_{\text{race}}}{\partial \ln \theta} \right|_{\theta_0} = \frac{\theta_0}{T_{\text{race}}(S; \theta_0)} \cdot \mathcal{S}_\theta(S)$$
   $E_\theta(S)$ represents the percentage change in total race time resulting from a $1\%$ change in parameter $\theta$.

#### Physical Interpretation:
- If $E_{\mu_{\text{deg}}}(S_{\text{1-stop}}) \gg E_{\mu_{\text{deg}}}(S_{\text{2-stop}})$, the 1-stop strategy is highly vulnerable to track temperature increases.
- If $E_{T_{\text{pit}}}(S_{\text{2-stop}}) = 2 \cdot E_{T_{\text{pit}}}(S_{\text{1-stop}})$, the 2-stop strategy suffers double the penalty from pit-lane delays.

---

### 3.3 Exact Crossover Root Finding (Brent's Method)
A critical tipping point $\theta^*$ between two strategy classes (e.g. best 1-stop vs best 2-stop) satisfies:
$$f(\theta) = \min_{S \in \mathcal{S}_{\text{1-stop}}} T_{\text{race}}(S; \theta) - \min_{S \in \mathcal{S}_{\text{2-stop}}} T_{\text{race}}(S; \theta) = 0$$

Because $f(\theta)$ is continuous, strictly monotonic over physical ranges, and cheap to evaluate, we compute the root $\theta^*$ using **Brent's method** (`scipy.optimize.brentq`), which combines bisection, secant method, and inverse quadratic interpolation:
$$\theta^{(k+1)} = \text{BrentStep}\big(f, \theta^{(k)}, \theta^{(k-1)}\big)$$
guaranteeing superlinear convergence with tolerance $|f(\theta^*)| < 10^{-4}\text{ seconds}$ in $\le 8$ iterations.

At the crossover point, the **transition sharpness** is given by the derivative:
$$\left. \frac{d f}{d \theta} \right|_{\theta^*}$$
A steep derivative implies a brittle boundary where slight parameter misestimation decisively flips the winning strategy.

---

### 3.4 2D Decision Phase Maps
Let $\boldsymbol{\Theta} = [\theta_{1, \min}, \theta_{1, \max}] \times [\theta_{2, \min}, \theta_{2, \max}] \subset \mathbb{R}^2$ be a two-dimensional parameter slice (e.g., Pit Loss $t_{\text{pit}}$ on the X-axis and Degradation Multiplier $\mu_{\text{deg}}$ on the Y-axis).

We partition $\boldsymbol{\Theta}$ into discrete **strategy decision regimes**:
$$\mathcal{R}_k = \big\{ (\theta_1, \theta_2) \in \boldsymbol{\Theta} \;\big|\; \text{OptimalStops}(\theta_1, \theta_2) = k \big\} \quad \text{for } k \in \{1, 2, 3\}$$

The **Decision Boundary (Crossover Manifold)** separating the 1-stop regime $\mathcal{R}_1$ and 2-stop regime $\mathcal{R}_2$ is the implicit curve:
$$\mathcal{B}_{1,2} = \big\{ (\theta_1, \theta_2) \in \boldsymbol{\Theta} \;\big|\; \Delta T(\theta_1, \theta_2) = 0 \big\}$$
where $\Delta T(\theta_1, \theta_2) = T_{\text{1-stop}}^*(\theta_1, \theta_2) - T_{\text{2-stop}}^*(\theta_1, \theta_2)$.

#### Analytical Crossover Curve Approximation:
Equating tyre degradation savings to pit loss:
$$\mu_{\text{deg}}^*(t_{\text{pit}}) \approx \frac{t_{\text{pit}}}{\frac{\alpha N^2}{12} + \frac{5\beta N^3}{108} + N_{\text{soft}} \cdot \Delta_{\text{soft}}}$$
This reveals that the decision boundary $\mathcal{B}_{1,2}$ in the $(t_{\text{pit}}, \mu_{\text{deg}})$ plane is approximately an affine ray through the origin.

#### Distance to Boundary (Tactical Safety Margin):
For a circuit with nominal parameters $\mathbf{\theta}_0 = (T_{\text{pit}, 0}, \mu_{\text{deg}, 0})$:
$$d_{\text{margin}} = \min_{(\theta_1, \theta_2) \in \mathcal{B}_{1,2}} \left\| \begin{bmatrix} \frac{\theta_1 - T_{\text{pit}, 0}}{\sigma_{T_{\text{pit}}}} \\ \frac{\theta_2 - \mu_{\text{deg}, 0}}{\sigma_{\mu_{\text{deg}}}} \end{bmatrix} \right\|_2$$
A large safety margin indicates that race engineers can commit to the strategy without fear of weather or track evolution flipping the optimum.

---

### 3.5 Strategy Robustness & Minimax Regret
If a team commits to strategy $S \in \mathcal{S}$ based on prior parameter estimate $\hat{\theta}$, but nature realizes true condition $\theta \sim p(\theta)$, the **Regret** incurred is:
$$\text{Regret}(S; \theta) = T_{\text{race}}(S; \theta) - \min_{S' \in \mathcal{S}} T_{\text{race}}(S'; \theta) \ge 0$$

We evaluate two robust decision criteria:
1. **Minimax Regret**:
   $$S^*_{\text{minimax}} = \arg\min_{S \in \mathcal{S}} \max_{\theta \in \Theta} \text{Regret}(S; \theta)$$
2. **Expected Regret**:
   $$\mathbb{E}[\text{Regret}(S)] = \int_{\Theta} \text{Regret}(S; \theta) p(\theta) \, d\theta$$

A strategy that is slightly sub-optimal at nominal conditions but avoids catastrophic tyre cliff drop-offs under higher degradation has significantly lower maximum regret than an aggressive 1-stop on the brink of failure.

---

## 4. Software Architecture & Implementation Plan

```text
src/
├── sensitivity.py            # Core Phase 4 engine
│   ├── SensitivityParameter  # Enum of swappable parameters
│   ├── CrossoverPoint        # Tipping point data structure
│   ├── SweepResult1D         # 1D sweep results & elasticities
│   ├── PhaseDiagramResult2D  # 2D phase map & boundary data
│   ├── ParameterSweep1D      # 1D sweep executor
│   ├── CrossoverFinder       # Brent's method root finder
│   ├── PhaseDiagram2D        # Vectorized 2D grid evaluator
│   └── RobustnessAnalyzer    # Minimax & expected regret analyzer
├── visualization.py          # Seaborn/Matplotlib plotting extensions
│   ├── plot_1d_sensitivity
│   ├── plot_2d_phase_diagram
│   └── plot_sensitivity_dashboard
└── optimization.py          # Existing Phase 3 solvers (DP & fast race simulation)
```

---

## 5. Verification & Validation Metrics

Phase 4 will be verified through unit tests in `tests/test_sensitivity.py`:
1. **Monotonicity Tests**: Assert that race time strictly increases monotonically with pit loss and tyre degradation.
2. **Root Finding Precision**: Assert that at any computed crossover point $\theta^*$, $|\Delta T(\theta^*)| < 0.01\text{s}$.
3. **Circuit Operating Point Validation**:
   - Assert that Bahrain nominal point $(22.5\text{s}, 1.30)$ lies strictly inside $\mathcal{R}_2$ (2-stop regime).
   - Assert that Monza nominal point $(24.0\text{s}, 0.80)$ lies strictly inside $\mathcal{R}_1$ (1-stop regime).
4. **Grid Consistency**: Assert that 2D phase grid shapes strictly match coordinate matrices and decision boundaries are continuous.
5. **CLI Integration**: Verify that `--sensitivity`, `--param`, `--crossover`, and `--phase-map` flags execute without exceptions.
