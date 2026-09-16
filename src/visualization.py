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

