"""Visualization tools for Monte Carlo strategy distributions."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.montecarlo import MonteCarloResult, compute_win_probability_matrix


def plot_strategy_distributions(
    results: list[MonteCarloResult],
    title: str = "Race Strategy Monte Carlo Distributions",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot Kernel Density Estimate (KDE) distributions of simulated race times.

    Args:
        results: List of MonteCarloResult objects to compare.
        title: Plot title.
        save_path: Optional file path to save the generated plot.
    """
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    for res in results:
        strategy_name = res.deterministic_result.strategy.name or "Unknown Strategy"
        # Convert seconds to minutes for easier reading on the x-axis
        times_minutes = res.race_times / 60.0

        sns.kdeplot(
            x=times_minutes,
            fill=True,
            linewidth=2,
            label=f"{strategy_name} (Mean: {res.mean_time / 60.0:.2f}m)",
        )

        # Plot the mean as a vertical dashed line
        plt.axvline(
            x=res.mean_time / 60.0,
            linestyle="--",
            alpha=0.7,
        )

    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Total Race Time (Minutes)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(title="Strategies", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_strategy_histograms(
    results: list[MonteCarloResult],
    bins: int = 30,
    title: str = "Race Strategy Race Time Histograms",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot binned frequency histograms of simulated race times.

    Args:
        results: List of MonteCarloResult objects to compare.
        bins: Number of histogram bins.
        title: Plot title.
        save_path: Optional file path to save the generated plot.
    """
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    for res in results:
        strategy_name = res.deterministic_result.strategy.name or "Unknown Strategy"
        times_minutes = res.race_times / 60.0
        sns.histplot(
            x=times_minutes,
            bins=bins,
            stat="density",
            element="step",
            alpha=0.3,
            label=f"{strategy_name} (Mean: {res.mean_time / 60.0:.2f}m)",
        )
        plt.axvline(
            x=res.mean_time / 60.0,
            linestyle="--",
            alpha=0.7,
        )

    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Total Race Time (Minutes)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title="Strategies", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Histograms saved to {save_path}")
    else:
        plt.show()


def plot_strategy_cdf(
    results: list[MonteCarloResult],
    title: str = "Race Strategy Cumulative Distribution Functions (CDF)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot empirical Cumulative Distribution Functions (CDF) of simulated race times.

    Illustrates P(T_race <= t), answering "What is the probability a strategy completes
    the race within time t?"

    Args:
        results: List of MonteCarloResult objects to compare.
        title: Plot title.
        save_path: Optional file path to save the generated plot.
    """
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    for res in results:
        strategy_name = res.deterministic_result.strategy.name or "Unknown Strategy"
        times_minutes = res.race_times / 60.0

        sorted_times = np.sort(times_minutes)
        p_cumulative = np.linspace(0.0, 1.0, len(sorted_times))

        plt.plot(
            sorted_times,
            p_cumulative,
            linewidth=2.5,
            label=f"{strategy_name} (Median: {res.median_time / 60.0:.2f}m, P95: {res.p95_time / 60.0:.2f}m)",
        )

    # Reference lines for key percentiles: 50% (Median) and 95% (VaR)
    plt.axhline(0.50, color="gray", linestyle=":", alpha=0.6, label="50th Percentile (Median)")
    plt.axhline(0.95, color="red", linestyle=":", alpha=0.6, label="95th Percentile (VaR)")

    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Total Race Time (Minutes)", fontsize=12)
    plt.ylabel("Cumulative Probability P(Time ≤ t)", fontsize=12)
    plt.ylim(-0.02, 1.05)
    plt.legend(title="Strategies & Key Percentiles", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"CDF plot saved to {save_path}")
    else:
        plt.show()


def plot_strategy_boxplots(
    results: list[MonteCarloResult],
    title: str = "Race Strategy Distribution Comparison (Boxplots)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot comparative boxplots showing median, IQR, whiskers, and outliers.

    Args:
        results: List of MonteCarloResult objects to compare.
        title: Plot title.
        save_path: Optional file path to save the generated plot.
    """
    names: list[str] = []
    times: list[np.ndarray] = []
    for res in results:
        strat_name = res.deterministic_result.strategy.name or "Unknown Strategy"
        names.extend([strat_name] * len(res.race_times))
        times.append(res.race_times / 60.0)

    df = pd.DataFrame({
        "Strategy": names,
        "Race Time (Minutes)": np.concatenate(times),
    })

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    sns.boxplot(
        data=df,
        x="Strategy",
        y="Race Time (Minutes)",
        hue="Strategy",
        legend=False,
        palette="Set2",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 6},
    )

    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Strategy", fontsize=12)
    plt.ylabel("Total Race Time (Minutes)", fontsize=12)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Boxplots saved to {save_path}")
    else:
        plt.show()


def plot_monte_carlo_dashboard(
    results: list[MonteCarloResult],
    title: str = "Monte Carlo Race Strategy Uncertainty Dashboard",
    save_path: Optional[str | Path] = None,
) -> None:
    """Generate a comprehensive 4-panel dashboard containing KDE, Histograms, CDFs, and Boxplots.

    Args:
        results: List of MonteCarloResult objects to compare.
        title: Dashboard super-title.
        save_path: Optional file path to save the generated plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sns.set_theme(style="whitegrid")

    # 1. Top-Left: KDE
    ax_kde = axes[0, 0]
    for res in results:
        strategy_name = res.deterministic_result.strategy.name or "Unknown Strategy"
        times_minutes = res.race_times / 60.0
        sns.kdeplot(
            x=times_minutes,
            fill=True,
            linewidth=2,
            label=f"{strategy_name} (Mean: {res.mean_time / 60.0:.2f}m)",
            ax=ax_kde,
        )
        ax_kde.axvline(x=res.mean_time / 60.0, linestyle="--", alpha=0.7)
    ax_kde.set_title("Kernel Density Estimation (KDE)", fontsize=13, pad=10)
    ax_kde.set_xlabel("Race Time (Minutes)", fontsize=11)
    ax_kde.set_ylabel("Density", fontsize=11)
    ax_kde.legend(fontsize=9)

    # 2. Top-Right: Histograms
    ax_hist = axes[0, 1]
    for res in results:
        strategy_name = res.deterministic_result.strategy.name or "Unknown Strategy"
        times_minutes = res.race_times / 60.0
        sns.histplot(
            x=times_minutes,
            bins=30,
            stat="density",
            element="step",
            alpha=0.3,
            label=strategy_name,
            ax=ax_hist,
        )
        ax_hist.axvline(x=res.mean_time / 60.0, linestyle="--", alpha=0.7)
    ax_hist.set_title("Binned Frequency Histograms", fontsize=13, pad=10)
    ax_hist.set_xlabel("Race Time (Minutes)", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.legend(fontsize=9)

    # 3. Bottom-Left: Empirical CDF
    ax_cdf = axes[1, 0]
    for res in results:
        strategy_name = res.deterministic_result.strategy.name or "Unknown Strategy"
        times_minutes = res.race_times / 60.0
        sorted_times = np.sort(times_minutes)
        p_cumulative = np.linspace(0.0, 1.0, len(sorted_times))
        ax_cdf.plot(
            sorted_times,
            p_cumulative,
            linewidth=2,
            label=f"{strategy_name} (P95: {res.p95_time / 60.0:.2f}m)",
        )
    ax_cdf.axhline(0.50, color="gray", linestyle=":", alpha=0.6, label="50% Median")
    ax_cdf.axhline(0.95, color="red", linestyle=":", alpha=0.6, label="95% VaR")
    ax_cdf.set_title("Cumulative Distribution Functions P(T ≤ t)", fontsize=13, pad=10)
    ax_cdf.set_xlabel("Race Time (Minutes)", fontsize=11)
    ax_cdf.set_ylabel("Cumulative Probability", fontsize=11)
    ax_cdf.set_ylim(-0.02, 1.05)
    ax_cdf.legend(fontsize=9)

    # 4. Bottom-Right: Boxplots
    ax_box = axes[1, 1]
    names: list[str] = []
    times: list[np.ndarray] = []
    for res in results:
        strat_name = res.deterministic_result.strategy.name or "Unknown Strategy"
        names.extend([strat_name] * len(res.race_times))
        times.append(res.race_times / 60.0)

    df = pd.DataFrame({
        "Strategy": names,
        "Race Time (Minutes)": np.concatenate(times),
    })
    sns.boxplot(
        data=df,
        x="Strategy",
        y="Race Time (Minutes)",
        hue="Strategy",
        legend=False,
        palette="Set2",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 6},
        ax=ax_box,
    )

    ax_box.set_title("Boxplot Comparison (Median, IQR, Outliers)", fontsize=13, pad=10)
    ax_box.set_xlabel("Strategy", fontsize=11)
    ax_box.set_ylabel("Race Time (Minutes)", fontsize=11)
    ax_box.tick_params(axis="x", rotation=15)

    plt.suptitle(title, fontsize=16, y=0.99)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Monte Carlo dashboard saved to {save_path}")
    else:
        plt.show()


def plot_win_probability_matrix(
    results: list[MonteCarloResult],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a heatmap showing the probability of row strategy beating column strategy."""
    prob_matrix, names = compute_win_probability_matrix(results)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        prob_matrix,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        center=0.5,
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "Win Probability (Row beats Column)"},
    )

    plt.title("Head-to-Head Strategy Win Probabilities", fontsize=14, pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Win probability matrix saved to {save_path}")
    else:
        plt.show()


def plot_1d_sensitivity(
    sweep_res: "SweepResult1D",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot 1D parameter sensitivity curves with crossover tipping points.

    Args:
        sweep_res: SweepResult1D object containing sweep data and crossovers.
        title: Optional custom plot title.
        save_path: Optional path to save the generated image.
    """
    from src.sensitivity import SweepResult1D

    plt.figure(figsize=(11, 6))
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("tab10", len(sweep_res.strategy_times))
    x_vals = sweep_res.parameter_values

    # Plot each strategy curve
    for idx, (name, times) in enumerate(sweep_res.strategy_times.items()):
        times_min = times / 60.0
        plt.plot(
            x_vals,
            times_min,
            label=name,
            color=palette[idx],
            linewidth=2.2,
            alpha=0.85,
        )

    # Plot lower envelope
    plt.plot(
        x_vals,
        sweep_res.optimal_times / 60.0,
        label="Optimal Strategy Envelope",
        color="black",
        linewidth=1.2,
        linestyle=":",
        alpha=0.6,
    )

    # Mark nominal baseline value
    nom_val = sweep_res.nominal_value
    if np.min(x_vals) <= nom_val <= np.max(x_vals):
        plt.axvline(
            x=nom_val,
            color="grey",
            linestyle="-.",
            linewidth=1.5,
            label=f"Nominal Baseline ({nom_val:.2f})",
            alpha=0.7,
        )

    # Annotate crossover tipping points
    for c in sweep_res.crossovers:
        if np.min(x_vals) <= c.crossover_value <= np.max(x_vals):
            plt.axvline(
                x=c.crossover_value,
                color="crimson",
                linestyle="--",
                linewidth=1.8,
                alpha=0.85,
            )
            y_pos = (c.time_at_crossover / 60.0) if c.time_at_crossover > 0 else np.mean(sweep_res.optimal_times / 60.0)
            plt.scatter([c.crossover_value], [y_pos], color="crimson", s=70, zorder=5)
            plt.annotate(
                f"Tipping Point: {c.crossover_value:.3f}\n({c.strategy_a_name} ↔ {c.strategy_b_name})",
                xy=(c.crossover_value, y_pos),
                xytext=(10, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6, ec="orange"),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="crimson"),
                fontsize=9,
                fontweight="bold",
            )

    plot_title = title or f"Strategy Sensitivity: {sweep_res.parameter_display_name}"
    plt.title(plot_title, fontsize=15, pad=12)
    plt.xlabel(sweep_res.parameter_display_name, fontsize=12)
    plt.ylabel("Total Race Time (Minutes)", fontsize=12)
    plt.legend(title="Strategies", fontsize=10, loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Sensitivity plot saved to {save_path}")
    else:
        plt.show()


def plot_2d_phase_diagram(
    phase_res: "PhaseDiagramResult2D",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a 2D decision boundary phase diagram showing 1-stop vs 2-stop regimes.

    Args:
        phase_res: PhaseDiagramResult2D object with grids, contours, and nominal point.
        title: Optional plot title.
        save_path: Optional path to save image.
    """
    from src.sensitivity import PhaseDiagramResult2D

    plt.figure(figsize=(11, 8))
    sns.set_theme(style="white")

    # Diverging contour surface of Delta T = T(1-stop) - T(2-stop)
    # Positive (green/warm) => 2-stop wins; Negative (blue/cool) => 1-stop wins
    max_abs_delta = max(10.0, float(np.percentile(np.abs(phase_res.delta_grid), 95)))
    levels = np.linspace(-max_abs_delta, max_abs_delta, 31)

    cf = plt.contourf(
        phase_res.x_grid,
        phase_res.y_grid,
        phase_res.delta_grid,
        levels=levels,
        cmap="coolwarm",
        alpha=0.85,
        extend="both",
    )
    cbar = plt.colorbar(cf)
    cbar.set_label("Time Advantage: 2-Stop over 1-Stop (s)\n[Positive = 2-Stop Wins | Negative = 1-Stop Wins]", fontsize=11)

    # Iso-delta contours
    iso_levels = [-15.0, -10.0, -5.0, 5.0, 10.0, 15.0]
    valid_iso = [l for l in iso_levels if np.min(phase_res.delta_grid) < l < np.max(phase_res.delta_grid)]
    if valid_iso:
        cs = plt.contour(
            phase_res.x_grid,
            phase_res.y_grid,
            phase_res.delta_grid,
            levels=valid_iso,
            colors="grey",
            linewidths=0.8,
            linestyles="--",
            alpha=0.7,
        )
        plt.clabel(cs, inline=True, fontsize=8, fmt="%+1.0fs")

    # Critical Decision Boundary (Delta T = 0)
    has_boundary = np.min(phase_res.delta_grid) < 0 < np.max(phase_res.delta_grid)
    if has_boundary:
        cs_zero = plt.contour(
            phase_res.x_grid,
            phase_res.y_grid,
            phase_res.delta_grid,
            levels=[0.0],
            colors="black",
            linewidths=3.0,
            linestyles="-",
        )
        plt.clabel(cs_zero, inline=True, fontsize=11, fmt="Decision Boundary (Delta = 0s)")

    # Mark nominal circuit point
    nom_x, nom_y = phase_res.nominal_point
    regime_str = f"{phase_res.nominal_stops}-Stop Regime"
    plt.scatter(
        [nom_x],
        [nom_y],
        color="gold",
        edgecolors="black",
        s=200,
        marker="*",
        zorder=6,
        label=f"{phase_res.circuit_name} Nominal ({nom_x:.1f}s, {nom_y:.2f}) -> {regime_str}",
    )
    plt.annotate(
        f"Nominal: {regime_str}",
        xy=(nom_x, nom_y),
        xytext=(15, -15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
        fontweight="bold",
        fontsize=10,
    )

    plot_title = title or f"Strategy Decision Phase Map: {phase_res.circuit_name}"
    plt.title(plot_title, fontsize=15, pad=12)
    plt.xlabel(phase_res.param_x_label, fontsize=12)
    plt.ylabel(phase_res.param_y_label, fontsize=12)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"2D Phase Diagram saved to {save_path}")
    else:
        plt.show()


def plot_sensitivity_dashboard(
    sweep_deg: "SweepResult1D",
    sweep_pit: "SweepResult1D",
    phase_res: "PhaseDiagramResult2D",
    title: str = "F1 Strategy Sensitivity & Decision Phase Dashboard",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot an executive 4-panel dashboard of strategy sensitivity and decision boundaries."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    sns.set_theme(style="whitegrid")

    # Panel 1: Degradation Multiplier 1D Sweep
    ax1 = axes[0, 0]
    for name, times in sweep_deg.strategy_times.items():
        ax1.plot(sweep_deg.parameter_values, times / 60.0, label=name, linewidth=2)
    for c in sweep_deg.crossovers:
        ax1.axvline(c.crossover_value, color="crimson", linestyle="--", alpha=0.8)
        ax1.annotate(
            f"Tipping Point: {c.crossover_value:.3f}",
            xy=(c.crossover_value, np.mean(sweep_deg.optimal_times / 60.0)),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6),
        )
    ax1.axvline(sweep_deg.nominal_value, color="grey", linestyle="-.", alpha=0.6, label="Nominal")
    ax1.set_title("Tyre Degradation Sensitivity (1-Stop vs 2-Stop)", fontsize=13)
    ax1.set_xlabel("Tyre Degradation Multiplier", fontsize=11)
    ax1.set_ylabel("Total Race Time (Minutes)", fontsize=11)
    ax1.legend(fontsize=9)

    # Panel 2: Pit Loss Time 1D Sweep
    ax2 = axes[0, 1]
    for name, times in sweep_pit.strategy_times.items():
        ax2.plot(sweep_pit.parameter_values, times / 60.0, label=name, linewidth=2)
    for c in sweep_pit.crossovers:
        ax2.axvline(c.crossover_value, color="crimson", linestyle="--", alpha=0.8)
        ax2.annotate(
            f"Tipping Point: {c.crossover_value:.2f}s",
            xy=(c.crossover_value, np.mean(sweep_pit.optimal_times / 60.0)),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6),
        )
    ax2.axvline(sweep_pit.nominal_value, color="grey", linestyle="-.", alpha=0.6, label="Nominal")
    ax2.set_title("Pit Loss Sensitivity & Safety Car Window", fontsize=13)
    ax2.set_xlabel("Pit Stop Time Loss (Seconds)", fontsize=11)
    ax2.set_ylabel("Total Race Time (Minutes)", fontsize=11)
    ax2.legend(fontsize=9)

    # Panel 3: 2D Phase Diagram
    ax3 = axes[1, 0]
    max_abs = max(10.0, float(np.percentile(np.abs(phase_res.delta_grid), 95)))
    cf = ax3.contourf(
        phase_res.x_grid,
        phase_res.y_grid,
        phase_res.delta_grid,
        levels=np.linspace(-max_abs, max_abs, 25),
        cmap="coolwarm",
        alpha=0.85,
    )
    fig.colorbar(cf, ax=ax3, label="Time Delta: 2-Stop vs 1-Stop (s)")
    if np.min(phase_res.delta_grid) < 0 < np.max(phase_res.delta_grid):
        cs_zero = ax3.contour(
            phase_res.x_grid, phase_res.y_grid, phase_res.delta_grid, levels=[0.0], colors="black", linewidths=2.5
        )
        ax3.clabel(cs_zero, inline=True, fontsize=10, fmt="Boundary (Delta=0s)")
    nom_x, nom_y = phase_res.nominal_point
    ax3.scatter([nom_x], [nom_y], color="gold", edgecolors="black", s=180, marker="*", zorder=5, label=f"Nominal ({phase_res.nominal_stops}-Stop)")
    ax3.set_title("2D Decision Phase Map (Pit Loss vs Degradation)", fontsize=13)
    ax3.set_xlabel(phase_res.param_x_label, fontsize=11)
    ax3.set_ylabel(phase_res.param_y_label, fontsize=11)
    ax3.legend(fontsize=9, loc="upper right")

    # Panel 4: Dimensionless Elasticity Comparison Bar Chart
    ax4 = axes[1, 1]
    strats = list(sweep_deg.strategy_times.keys())
    # Compute elasticity for deg and pit loss
    nom_deg = sweep_deg.nominal_value
    nom_pit = sweep_pit.nominal_value
    deg_elast = [
        (nom_deg / sweep_deg.strategy_times[s][len(sweep_deg.parameter_values)//2])
        * ((sweep_deg.strategy_times[s][-1] - sweep_deg.strategy_times[s][0]) / (sweep_deg.parameter_values[-1] - sweep_deg.parameter_values[0]))
        for s in strats
    ]
    pit_elast = [
        (nom_pit / sweep_pit.strategy_times[s][len(sweep_pit.parameter_values)//2])
        * ((sweep_pit.strategy_times[s][-1] - sweep_pit.strategy_times[s][0]) / (sweep_pit.parameter_values[-1] - sweep_pit.parameter_values[0]))
        for s in strats
    ]

    y_pos = np.arange(len(strats))
    width = 0.35
    ax4.barh(y_pos - width/2, deg_elast, width, label="Tyre Deg Elasticity (%T / %Deg)", color="indianred")
    ax4.barh(y_pos + width/2, pit_elast, width, label="Pit Loss Elasticity (%T / %PitLoss)", color="cornflowerblue")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(strats, fontsize=10)
    ax4.set_xlabel("Dimensionless Elasticity E_theta (% Race Time Change per 1% Parameter Shift)", fontsize=11)
    ax4.set_title("Strategy Elasticity & Environmental Vulnerability", fontsize=13)
    ax4.legend(fontsize=9)

    plt.suptitle(title, fontsize=16, y=0.99)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Sensitivity dashboard saved to {save_path}")
    else:
        plt.show()


def plot_empirical_degradation_fit(
    params: "EmpiricalCircuitParameters",
    clean_laps: list["HistoricalLap"],
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot scatter of empirical tyre degradation with fitted quadratic curves."""
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    palette = {
        "Soft": "#E10600",
        "Medium": "#FFD700",
        "Hard": "#A0A0A0",
        "C1": "#A0A0A0",
        "C2": "#FFD700",
        "C3": "#E10600",
        "C4": "#9932CC",
        "C5": "#FF69B4",
    }

    # Group clean laps by compound and compute empirical degradation
    compounds_present = [c for c in params.compounds.keys() if params.compounds[c].sample_count > 0]
    stints_dict: dict[tuple[int, int], list] = {}
    for l in clean_laps:
        if l.is_clean and l.compound and l.stint_number is not None and l.tyre_age and l.tyre_age >= 1:
            stints_dict.setdefault((l.driver_number, l.stint_number), []).append(l)

    # Plot scatter points per compound
    for comp_name in compounds_present:
        color = palette.get(comp_name, "#333333")
        cp = params.compounds[comp_name]

        # Gather points
        ages = []
        deltas = []
        for (d_num, s_num), laps in stints_dict.items():
            if not laps or laps[0].compound.upper() != comp_name.upper():
                continue
            early = [l.lap_duration for l in laps if l.tyre_age and 2 <= l.tyre_age <= 5]
            base_t = float(np.median(early)) if early else laps[0].lap_duration
            for l in laps:
                ages.append(l.tyre_age)
                deltas.append(l.lap_duration - base_t)

        if ages:
            ax.scatter(
                ages,
                deltas,
                alpha=0.25,
                s=20,
                color=color,
                label=f"{comp_name} Telemetry (N={len(ages)})",
            )

        # Plot fitted curve: D(a) = alpha * a + beta * a^2
        max_age = max(ages) if ages else 35
        curve_ages = np.linspace(1, max(30, max_age), 100)
        curve_deg = cp.alpha * curve_ages + cp.beta * (curve_ages**2)
        ax.plot(
            curve_ages,
            curve_deg,
            color=color,
            linewidth=3,
            label=f"{comp_name} Fit: alpha={cp.alpha:.4f}, beta={cp.beta:.6f} (R2={cp.r_squared:.2f})",
        )

    ax.set_title(title or f"Empirical Tyre Degradation Curves — {params.circuit_name} ({params.year})", fontsize=14)
    ax.set_xlabel("Tyre Age (Laps Completed in Stint)", fontsize=12)
    ax.set_ylabel("Pace Degradation Delta (Seconds)", fontsize=12)
    ax.set_ylim(-1.0, 5.5)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Empirical wear plot saved to {save_path}")
    else:
        plt.show()


def plot_lap_residuals(
    lap_numbers: list[int],
    sim_laps: list[float],
    actual_laps: list[float],
    circuit_name: str = "Circuit",
    driver_name: str = "Driver",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot lap-by-lap comparison and residual error (Actual - Simulated)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    sns.set_theme(style="whitegrid")

    # Panel 1: Lap Times
    ax1.plot(lap_numbers, actual_laps, label=f"Actual Telemetry ({driver_name})", color="black", linewidth=1.8, marker="o", markersize=3)
    ax1.plot(lap_numbers, sim_laps, label="Calibrated Simulation", color="crimson", linewidth=2.0, linestyle="--")
    ax1.set_title(f"Lap Pace Fidelity — {circuit_name}: {driver_name}", fontsize=14)
    ax1.set_ylabel("Lap Time (Seconds)", fontsize=12)
    ax1.legend(fontsize=11)

    # Panel 2: Residuals
    residuals = np.array(actual_laps) - np.array(sim_laps)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    ax2.axhline(0, color="grey", linestyle="-", alpha=0.8)
    ax2.bar(lap_numbers, residuals, color=np.where(residuals > 0, "salmon", "skyblue"), width=0.8, alpha=0.8)
    ax2.axhline(mae, color="orange", linestyle=":", label=f"+MAE ({mae:.2f}s)")
    ax2.axhline(-mae, color="orange", linestyle=":", label=f"-MAE (-{mae:.2f}s)")
    ax2.set_title(f"Residual Error (Actual - Simulated) | RMSE = {rmse:.3f}s, MAE = {mae:.3f}s", fontsize=12)
    ax2.set_xlabel("Lap Number", fontsize=12)
    ax2.set_ylabel("Residual Delta (s)", fontsize=11)
    ax2.set_ylim(-3.0, 3.0)
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Lap residuals plot saved to {save_path}")
    else:
        plt.show()


def plot_strategy_backtest_gantt(
    metrics_list: list["ValidationMetrics"],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot side-by-side Gantt stint bars comparing Actual Strategies vs Optimizer."""
    fig, ax = plt.subplots(figsize=(14, max(4, len(metrics_list) * 2.2)))
    sns.set_theme(style="whitegrid")

    palette = {
        "Soft": "#E10600",
        "Medium": "#FFD700",
        "Hard": "#D3D3D3",
        "C1": "#D3D3D3",
        "C2": "#FFD700",
        "C3": "#E10600",
        "C4": "#9932CC",
        "C5": "#FF69B4",
    }

    y_labels = []
    y_positions = []
    y_idx = 0

    for m in metrics_list:
        # 1. Actual
        y_positions.append(y_idx)
        y_labels.append(f"{m.circuit_name}\nActual: {m.driver_name}")

        start_lap = 1
        for part in m.actual_strategy_desc.split(" -> "):
            comp, l_str = part.split(" (")
            stint_len = int(l_str.replace("L)", ""))
            color = palette.get(comp.capitalize(), "#666666")
            ax.barh(y_idx, stint_len, left=start_lap - 1, color=color, edgecolor="black", height=0.5, alpha=0.85)
            ax.text(start_lap - 1 + stint_len / 2, y_idx, f"{comp} ({stint_len}L)", ha="center", va="center", fontsize=9, fontweight="bold")
            start_lap += stint_len

        # 2. Predicted Optimal
        y_idx += 1
        y_positions.append(y_idx)
        y_labels.append(f"{m.circuit_name}\nModel Optimal")

        start_lap = 1
        for part in m.optimal_strategy_desc.split(" -> "):
            comp, l_str = part.split(" (")
            stint_len = int(l_str.replace("L)", ""))
            color = palette.get(comp.capitalize(), "#666666")
            ax.barh(y_idx, stint_len, left=start_lap - 1, color=color, edgecolor="black", height=0.5, alpha=0.6, hatch="//")
            ax.text(start_lap - 1 + stint_len / 2, y_idx, f"{comp} ({stint_len}L)", ha="center", va="center", fontsize=9, fontweight="bold")
            start_lap += stint_len

        y_idx += 1.5

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Race Lap Number", fontsize=12)
    ax.set_title("Historical Strategy Backtest Comparison (Actual vs Model Optimal)", fontsize=14)
    ax.set_xlim(0, max(m.actual_stops * 30 for m in metrics_list) + 70)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Backtest Gantt plot saved to {save_path}")
    else:
        plt.show()


