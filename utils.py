"""Small helpers: paths, synthetic-signal generation, charge-segment detection."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create directory (and parents) if missing; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def detect_charge_segments(
    df: pd.DataFrame,
    *,
    current_col: str = "battery_current",
    soc_col: str = "SOC",
    min_run: int = 5,
) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx)] segments where the battery is charging.

    Charging = positive current AND non-decreasing SOC for at least
    `min_run` consecutive samples. Indices are inclusive.
    """
    if current_col not in df.columns:
        return []
    charging = df[current_col] > 0
    if soc_col in df.columns:
        charging &= df[soc_col].diff().fillna(0) >= 0

    segments: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, flag in enumerate(charging.values):
        if flag and not in_seg:
            start = i
            in_seg = True
        elif not flag and in_seg:
            if i - start >= min_run:
                segments.append((start, i - 1))
            in_seg = False
    if in_seg and len(charging) - start >= min_run:
        segments.append((start, len(charging) - 1))
    return segments


def synthesize_grid_load(
    df: pd.DataFrame,
    *,
    base_kw: float = 50.0,
    diurnal_amp_kw: float = 20.0,
    seed: int = 42,
) -> pd.Series:
    """Generate a plausible baseline neighborhood grid load (kW) aligned to df.

    Not from the CSV — this is a synthetic stand-in so the grid-load
    visualization has something to plot. Replace with real telemetry
    when available.
    """
    rng = np.random.default_rng(seed)
    hours = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
    diurnal = diurnal_amp_kw * np.sin((hours - 6) * np.pi / 12.0)
    noise = rng.normal(0, 2.0, size=len(df))
    return pd.Series(base_kw + diurnal + noise, index=df.index, name="base_load_kw")


def synthesize_solar_output(
    df: pd.DataFrame,
    *,
    peak_kw: float = 30.0,
    seed: int = 7,
) -> pd.Series:
    """Generate a synthetic PV output profile (kW) aligned to df.

    Zero at night; bell curve peaking near solar noon with small noise.
    """
    rng = np.random.default_rng(seed)
    hours = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
    bell = np.clip(np.sin((hours - 6) * np.pi / 12.0), 0, None) ** 2
    noise = rng.normal(0, 0.5, size=len(df))
    return pd.Series(np.clip(peak_kw * bell + noise, 0, None), index=df.index, name="solar_kw")


def ev_load_from_battery(
    df: pd.DataFrame,
    *,
    nominal_voltage: float = 400.0,
    current_col: str = "battery_current",
) -> pd.Series:
    """Convert per-cell battery current into a rough pack-level EV load (kW).

    Assumes `nominal_voltage` volts at the pack and that the per-row current
    in the CSV is the charging current to the pack. This is a heuristic,
    not a calibrated measurement.
    """
    if current_col not in df.columns:
        return pd.Series(0.0, index=df.index, name="ev_load_kw")
    kw = (df[current_col].clip(lower=0) * nominal_voltage) / 1000.0
    return kw.rename("ev_load_kw")
