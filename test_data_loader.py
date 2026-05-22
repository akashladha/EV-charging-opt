"""Unit tests for src.data_loader."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import EXPECTED_COLUMNS, load_csv


def _write_minimal_csv(tmp_path: Path) -> Path:
    """Write a tiny well-formed CSV that mirrors the Kaggle schema."""
    rows = [
        "timestamp,SOC,SOH,terminal_voltage,battery_current,battery_temp,ambient_temp,internal_resistance",
        "0,0.10,0.99,3.7,5.0,25.0,22.0,0.012",
        "1,0.25,0.99,3.8,7.0,28.0,22.1,0.012",
        "2,55.0,0.98,3.9,9.0,30.0,22.3,0.013",   # SOC as percentage; loader should normalize
        "3,0.80,0.98,4.0,8.0,32.0,22.4,0.013",
    ]
    p = tmp_path / "mini.csv"
    p.write_text("\n".join(rows))
    return p


def test_load_csv_parses_and_normalizes(tmp_path: Path) -> None:
    p = _write_minimal_csv(tmp_path)
    df = load_csv(p)

    # All expected columns present
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"missing column {col}"

    # datetime column added and monotonic
    assert "datetime" in df.columns
    assert df["datetime"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])

    # Row count preserved
    assert len(df) == 4

    # SOC normalized to [0, 1]
    assert df["SOC"].between(0.0, 1.0).all(), f"SOC out of range: {df['SOC'].tolist()}"
    # The percentage row (55.0) should have become 0.55
    assert np.isclose(df.loc[df["timestamp"] == 2, "SOC"].iloc[0], 0.55)


def test_load_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "does_not_exist.csv")


def test_load_csv_required_columns_enforced(tmp_path: Path) -> None:
    p = tmp_path / "bare.csv"
    p.write_text("timestamp,SOC\n0,0.1\n1,0.2\n")
    # SOC is present; terminal_voltage is not — should raise
    with pytest.raises(ValueError, match="Missing required columns"):
        load_csv(p, required_columns=["timestamp", "SOC", "terminal_voltage"])


def test_load_csv_rejects_csv_without_timestamp(tmp_path: Path) -> None:
    p = tmp_path / "no_ts.csv"
    p.write_text("SOC,terminal_voltage\n0.1,3.7\n")
    with pytest.raises(ValueError, match="timestamp"):
        load_csv(p)


def test_load_csv_tick_seconds(tmp_path: Path) -> None:
    p = _write_minimal_csv(tmp_path)
    df = load_csv(p, tick_seconds=60)
    # 4 rows, 60s apart -> last row should be 3 minutes after first
    delta = df["datetime"].iloc[-1] - df["datetime"].iloc[0]
    assert delta == pd.Timedelta(minutes=3)
