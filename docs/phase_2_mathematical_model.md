# Mathematical Formulation & Architecture: Phase 2

This document formalizes the mathematical models of uncertainty and the software architecture for **Phase 2: Monte Carlo Simulation & Uncertainty Analysis** of the Formula 1 Race Strategy Simulator.

---

## 1. Overview & System Objectives

In Phase 1, we calculated the exact deterministic lap-time profile $T_{\text{lap}}(n)$. However, real Formula 1 races are highly stochastic. 

The core objective of Phase 2 is:
> **To evaluate the statistical distribution of race outcomes for a given strategy by introducing controlled randomness into lap times, pit stops, and tyre degradation, allowing us to compute confidence intervals and win probabilities.**

We transition the objective function from minimizing deterministic time:
$$S^* = \arg\min_S T_{\text{race}}(S)$$
to minimizing the expected time, or minimizing risk (variance):
$$S^* = \arg\min_S \mathbb{E}[T_{\text{race}}(S)]$$

---

## 2. Mathematical Models of Uncertainty

### 2.1 Driver Lap-to-Lap Pace Variance ($\epsilon_{\text{lap}}$)
Even in clean air, a driver cannot produce identical lap times. We model natural pace variation and minor traffic encounters as independent, identically distributed (i.i.d.) Gaussian noise:
$$\epsilon_{\text{lap}}(n) \sim \mathcal{N}(0, \sigma_{\text{lap}}^2)$$

where:
* $\sigma_{\text{lap}}$ is the standard deviation of lap times (typically $\approx 0.3 - 0.7$ seconds for top drivers).
* Over an $N$-lap race, the total accumulated variance simply adds up: $\text{Var}(T_{\text{race}}) \propto N \cdot \sigma_{\text{lap}}^2$.

### 2.2 Pit Stop Execution Variance ($\epsilon_{\text{pit}}$)
A pit stop can never be faster than the physical limit of the crew, but it can be significantly slower (e.g., a sticky wheel nut). This creates a heavily right-skewed distribution. We model the delay using a **Log-Normal distribution** (or Gamma distribution) added to the theoretical perfect pit loss:

$$T_{\text{pit}} = T_{\text{pit, ideal}} + \epsilon_{\text{pit}}$$
$$\epsilon_{\text{pit}} \sim \text{LogNormal}(\mu_{\text{pit}}, \sigma_{\text{pit}}^2)$$

where:
* $\epsilon_{\text{pit}} > 0$ strictly represents time *lost* above the ideal baseline.
* The parameters $\mu_{\text{pit}}, \sigma_{\text{pit}}$ are calibrated such that a normal stop adds $\approx 0.2\text{s}$, but there is a $5\%$ tail probability of a delay $>2.0\text{s}$.

### 2.3 Tyre Degradation Uncertainty ($\delta_{\text{wear}}$)
Tyre wear rates ($\alpha, \beta$) vary based on unpredictable track temperatures, setup nuances, and driving style. We apply a stochastic multiplier to the entire degradation curve for a given simulation run:

$$D_c(a, m) = (1 + \delta_{\text{wear}}^{(m)}) \cdot D_{c, \text{ideal}}(a)$$
$$\delta_{\text{wear}}^{(m)} \sim \mathcal{N}(0, \sigma_{\text{wear}}^2)$$

where:
* $m \in \{1, \dots, M\}$ is the simulation iteration index.
* $\sigma_{\text{wear}}$ dictates the uncertainty in tyre life (e.g., $\sigma_{\text{wear}} = 0.10$ implies a $10\%$ standard deviation in degradation severity).

---

## 3. Monte Carlo Aggregation Metrics

For $M$ iterations (e.g., $M = 10,000$), we sample from the above distributions to generate $M$ total race times: $\mathbf{T} = [T^{(1)}, T^{(2)}, \dots, T^{(M)}]$.

We compute the following decision metrics:
1. **Expected Race Time:** $\mathbb{E}[\mathbf{T}] = \frac{1}{M} \sum_{m=1}^M T^{(m)}$
2. **Median Race Time:** $P_{50}(\mathbf{T})$ (robust to extreme pit stop outliers).
3. **Pace Variance:** $\text{Var}(\mathbf{T})$.
4. **Value at Risk (P95):** The 95th percentile race time, answering *"What is the worst-case scenario if things go wrong?"*
5. **Head-to-Head Win Probability:**
   When comparing Strategy A to Strategy B under identical random seeds (to correlate track evolution/safety cars):
   $$P(S_A \text{ beats } S_B) = \frac{1}{M} \sum_{m=1}^M \mathbb{I}\Big(T^{(m)}(S_A) < T^{(m)}(S_B)\Big)$$

---

## 4. Software Architecture & Vectorization

Performing $10,000$ iterations using traditional `for` loops in Python is slow. Instead, Phase 2 will utilize **NumPy Broadcasting** to execute all simulations simultaneously in milliseconds.

```text
src/
├── stochastic.py        # Distributions and StochasticParameters dataclasses
├── montecarlo.py        # Vectorized MonteCarloEngine and MonteCarloResult
└── visualization.py     # KDE, histograms, and cumulative distribution plots
```

### Vectorized Execution Pipeline
1. **Base Array:** Extract the deterministic $1 \times N$ lap times and $1 \times N$ degradation terms from Phase 1.
2. **Degradation Scaling:** Sample an $M \times 1$ array of $\delta_{\text{wear}}$. Multiply to get an $M \times N$ degradation matrix.
3. **Pace Noise:** Sample an $M \times N$ matrix from $\mathcal{N}(0, \sigma_{\text{lap}}^2)$.
4. **Pit Stop Noise:** Sample an $M \times (K-1)$ matrix from the Log-Normal distribution.
5. **Summation:** Add the matrices and sum along the $N$ axis (axis=1) to yield an $M \times 1$ vector of total race times instantly.
