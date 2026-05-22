"""Load and clean the EV battery charging CSV.

The reference dataset (Kaggle NEV battery charging) has columns:
    timestamp, SOC, SOH, terminal_voltage, battery_current, battery_temp,
    ambient_temp, internal_resistance, action_current, action_voltage,
    dT_dt, dV_dt, soc_delta, thermal_stress_index, aging_indicator,
    charging_efficiency, charging_time, cycle_degradation,
    over_temp_flag, over_voltage_flag, balancing_time

`timestamp` is an integer tick (0, 1, 2, ...). We treat each tick as one
second elapsed from a configurable epoch and produce a `datetime` column
so downstream plots have a real time axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# Columns we expect in a well-formed file. Missing ones trigger a warning,
# not a hard failure, so partial/derived datasets still load.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "SOC",
    "SOH",
    "terminal_voltage",
    "battery_current",
    "battery_temp",
    "ambient_temp",
    "internal_resistance",
)


def _normalize_soc(soc: pd.Series) -> pd.Series:
    """Coerce SOC to the [0, 1] convention.

    The source CSV mixes fractions (0.06) and percentages (96.75). We detect
    the scale per-row: anything > 1.5 is divided by 100. Vectorized.
    """
    soc = pd.to_numeric(soc, errors="coerce")
    return np.where(soc > 1.5, soc / 100.0, soc)


def load_csv(
    path: PathLike,
    *,
    epoch: str = "2024-01-01",
    tick_seconds: int = 1,
    required_columns: Optional[Iterable[str]] = None,
    dropna_subset: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load the EV battery charging CSV into a tidy DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    epoch : str, default "2024-01-01"
        Anchor date. The integer `timestamp` column is interpreted as
        `epoch + timestamp * tick_seconds` seconds.
    tick_seconds : int, default 1
        Seconds per tick. Set to 60 if your ticks are minutes, etc.
    required_columns : iterable of str, optional
        If given, raise `ValueError` when any are missing.
    dropna_subset : iterable of str, optional
        Drop rows where any of these columns is NaN. Defaults to the
        intersection of EXPECTED_COLUMNS and what is present.

    Returns
    -------
    pd.DataFrame
        Cleaned frame with an added `datetime` column (pd.Timestamp,
        timezone-naive) and SOC normalized to [0, 1].

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If `required_columns` is set and a column is missing, or if the
        `timestamp` column is absent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)

    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column.")

    # Required-column check
    if required_columns is not None:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # Parse timestamp -> datetime
    df["datetime"] = pd.to_datetime(epoch) + pd.to_timedelta(
        df["timestamp"].astype("int64") * tick_seconds, unit="s"
    )

    # Sort + dedupe by time (defensive)
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    # Normalize SOC if present
    if "SOC" in df.columns:
        df["SOC"] = _normalize_soc(df["SOC"])

    # Drop rows missing critical fields
    if dropna_subset is None:
        dropna_subset = [c for c in EXPECTED_COLUMNS if c in df.columns]
    df = df.dropna(subset=list(dropna_subset)).reset_index(drop=True)

    return df
