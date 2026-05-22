"""Visualization helpers for charging curves, grid load, solar, and battery temp.

Each function accepts a DataFrame and returns the saved file path (PNG for
matplotlib, HTML for plotly). Designed to be importable as a library and
also driven by `src.__main__`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import detect_charge_segments, ensure_dir, ev_load_from_battery

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# 1. Charging curves
# ---------------------------------------------------------------------------
def plot_charging_curves(
    df: pd.DataFrame,
    out_dir: PathLike = "outputs",
    filename: str = "charging_curves.png",
    *,
    show: bool = False,
) -> Path:
    """Overlay SOC and terminal voltage vs time on a dual y-axis chart.

    Charge-segment start/stop markers are annotated using
    `utils.detect_charge_segments`.

    Parameters
    ----------
    df : DataFrame
        Must contain `datetime`, `SOC`, and `terminal_voltage`.
    out_dir, filename : path bits
        Where to save the PNG.
    show : bool
        If True, also call `plt.show()`.

    Returns
    -------
    Path to saved PNG.
    """
    required = {"datetime", "SOC", "terminal_voltage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"plot_charging_curves missing columns: {missing}")

    out_dir = ensure_dir(out_dir)
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df["datetime"], df["SOC"], color="#1f77b4", label="SOC")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("SOC (fraction)", color="#1f77b4")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(df["datetime"], df["terminal_voltage"], color="#d62728", alpha=0.7, label="V_term")
    ax2.set_ylabel("Terminal voltage (V)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # Annotate charge segments
    for start, end in detect_charge_segments(df):
        ax1.axvline(df["datetime"].iloc[start], color="green", ls="--", alpha=0.4)
        ax1.axvline(df["datetime"].iloc[end], color="red", ls="--", alpha=0.4)
    ax1.text(
        0.01, 0.97,
        "green dash = charge start  |  red dash = charge stop",
        transform=ax1.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7, ec="none"),
    )

    fig.suptitle("Charging curves: SOC and terminal voltage")
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2. Grid load (stacked area, base + EV, optional solar subtraction)
# ---------------------------------------------------------------------------
def plot_grid_load(
    df: pd.DataFrame,
    solar_df: Optional[pd.DataFrame] = None,
    battery_df: Optional[pd.DataFrame] = None,
    *,
    base_load_col: str = "base_load_kw",
    ev_load_col: str = "ev_load_kw",
    solar_col: str = "solar_kw",
    out_dir: PathLike = "outputs",
    filename: str = "grid_load.html",
) -> Path:
    """Stacked-area plot of grid load with optional solar subtraction.

    Behavior
    --------
    - If `df` already has `base_load_kw`, use it. Otherwise raise.
    - EV load: from `df[ev_load_col]` if present, else from
      `battery_df` (converted via `utils.ev_load_from_battery`), else zero.
    - Solar: from `solar_df[solar_col]` if given.

    Saves an interactive Plotly HTML and returns its path.
    """
    if base_load_col not in df.columns:
        raise ValueError(
            f"`{base_load_col}` not in df. Add a base load column or "
            "use utils.synthesize_grid_load to generate one."
        )

    out_dir = ensure_dir(out_dir)
    t = df["datetime"]
    base = df[base_load_col]

    if ev_load_col in df.columns:
        ev = df[ev_load_col]
    elif battery_df is not None:
        ev = ev_load_from_battery(battery_df).reindex(df.index).fillna(0)
    else:
        ev = pd.Series(0.0, index=df.index)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=t, y=base, name="Base load (kW)",
        stackgroup="load", line=dict(color="#a6cee3"),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=ev, name="EV load (kW)",
        stackgroup="load", line=dict(color="#1f78b4"),
    ))

    if solar_df is not None and solar_col in solar_df.columns:
        solar = solar_df[solar_col].reindex(df.index).fillna(0)
        net = base + ev - solar
        fig.add_trace(go.Scatter(
            x=t, y=-solar, name="Solar offset (kW)",
            line=dict(color="#ff7f00", dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=t, y=net, name="Net grid draw (kW)",
            line=dict(color="black", width=2),
        ))

    fig.update_layout(
        title="Grid load: base + EV (with optional solar offset)",
        xaxis_title="Time", yaxis_title="Power (kW)",
        hovermode="x unified", template="plotly_white",
    )
    out_path = out_dir / filename
    fig.write_html(out_path)
    return out_path


# ---------------------------------------------------------------------------
# 3. Solar output
# ---------------------------------------------------------------------------
def plot_solar_output(
    df: pd.DataFrame,
    *,
    solar_col: str = "solar_kw",
    out_dir: PathLike = "outputs",
    filename: str = "solar_output.png",
) -> Path:
    """Plot solar generation over time, shading daylight portions."""
    if solar_col not in df.columns:
        raise ValueError(f"`{solar_col}` not in df.")
    out_dir = ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(df["datetime"], 0, df[solar_col], color="#f6c244", alpha=0.85)
    ax.plot(df["datetime"], df[solar_col], color="#c98a00", lw=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Solar output (kW)")
    ax.set_title("PV generation profile")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 4. Battery temperature
# ---------------------------------------------------------------------------
def plot_battery_temp(
    df: pd.DataFrame,
    *,
    out_dir: PathLike = "outputs",
    filename: str = "battery_temp.png",
    temp_warn_c: float = 45.0,
    temp_crit_c: float = 60.0,
) -> Path:
    """Plot battery temperature vs ambient with warning/critical bands.

    Note: synthetic Kaggle data can contain unrealistic temperature spikes
    (>1000°C). We clip the y-axis to the 1st/99th percentiles for legibility
    but still draw all points so outliers are visible at the edges.
    """
    if "battery_temp" not in df.columns:
        raise ValueError("`battery_temp` not in df.")

    out_dir = ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["datetime"], df["battery_temp"], color="#d62728", lw=1.2, label="Battery °C")
    if "ambient_temp" in df.columns:
        ax.plot(df["datetime"], df["ambient_temp"], color="#7f7f7f",
                lw=1, alpha=0.7, label="Ambient °C")

    # Threshold lines
    ax.axhline(temp_warn_c, color="orange", ls="--", lw=1, alpha=0.7, label=f"Warn {temp_warn_c}°C")
    ax.axhline(temp_crit_c, color="red", ls="--", lw=1, alpha=0.7, label=f"Crit {temp_crit_c}°C")

    # Clip y-axis to robust quantiles so plot stays readable with outliers
    lo, hi = df["battery_temp"].quantile([0.01, 0.99])
    pad = (hi - lo) * 0.1
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Battery vs ambient temperature")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
