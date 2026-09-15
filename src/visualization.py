"""Visualization tools for Monte Carlo strategy distributions."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.montecarlo import MonteCarloResult


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
            label=f"{strategy_name} (Mean: {res.mean_time/60.0:.2f}m)",
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
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_win_probability_matrix(
    results: list[MonteCarloResult],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a heatmap showing the probability of row strategy beating column strategy."""
    n = len(results)
    names = [r.deterministic_result.strategy.name for r in results]
    
    # Calculate probability matrix
    prob_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                prob_matrix[i, j] = 0.5
            else:
                wins = np.sum(results[i].race_times < results[j].race_times)
                prob_matrix[i, j] = wins / results[i].n_iterations

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        prob_matrix,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        center=0.5,
        xticklabels=names,
        yticklabels=names,
        cbar_kws={'label': 'Win Probability (Row beats Column)'}
    )
    
    plt.title("Head-to-Head Strategy Win Probabilities", fontsize=14, pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Win probability matrix saved to {save_path}")
    else:
        plt.show()
