"""Exploratory data analysis: summary stats and distribution plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import ensure_dir

PathLike = Union[str, Path]

DEFAULT_DIST_COLS: tuple[str, ...] = (
    "SOC",
    "SOH",
    "terminal_voltage",
    "battery_current",
    "battery_temp",
    "ambient_temp",
)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column descriptive statistics (numeric columns only).

    Adds skew, kurtosis, and NaN counts to the standard `describe()`.
    """
    num = df.select_dtypes(include="number")
    desc = num.describe().T
    desc["skew"] = num.skew()
    desc["kurtosis"] = num.kurtosis()
    desc["n_missing"] = num.isna().sum()
    return desc


def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    out_dir: PathLike = "outputs",
    filename: str = "distributions.png",
) -> Path:
    """Plot histograms + KDE for each numeric column of interest.

    Returns the saved PNG path.
    """
    cols = list(columns) if columns else [c for c in DEFAULT_DIST_COLS if c in df.columns]
    if not cols:
        raise ValueError("No plottable columns found.")

    out_dir = ensure_dir(out_dir)
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#3b7dd8")
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Variable distributions", y=1.02)
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_correlation(
    df: pd.DataFrame,
    out_dir: PathLike = "outputs",
    filename: str = "correlation.png",
) -> Path:
    """Plot a Pearson correlation heatmap over numeric columns."""
    out_dir = ensure_dir(out_dir)
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson correlation")
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
