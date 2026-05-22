"""Runnable demo:
    python -m src --csv data/nev_battery_charging.csv

Loads the CSV, prints summary stats, saves all PNG/HTML plots, and runs
a tiny optimization demo so the wiring is end-to-end verified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_csv
from .eda import plot_correlation, plot_distributions, summary_stats
from .optimization import ChargingProblem, naive_baseline, solve_peak_minimization
from .utils import (
    ensure_dir,
    ev_load_from_battery,
    synthesize_grid_load,
    synthesize_solar_output,
)
from .viz import (
    plot_battery_temp,
    plot_charging_curves,
    plot_grid_load,
    plot_solar_output,
)


def _build_demo_problem(df: pd.DataFrame) -> ChargingProblem:
    """Construct a 6-EV, 96-step toy problem driven by the dataset's hourly profile."""
    # Resample base load to 15-min steps over 24h
    T = 96
    hours = np.linspace(0, 24, T, endpoint=False)
    base = 40 + 20 * np.sin((hours - 6) * np.pi / 12.0)
    solar = np.clip(np.sin((hours - 6) * np.pi / 12.0), 0, None) ** 2 * 30
    N = 6
    rng = np.random.default_rng(0)
    return ChargingProblem(
        base_load_kw=base,
        p_max_kw=np.full(N, 11.0),
        energy_required_kwh=rng.uniform(20, 40, size=N),
        dt_hours=0.25,
        solar_kw=solar,
        arrival_idx=rng.integers(0, 20, size=N),    # arrive evening-ish
        departure_idx=rng.integers(70, 96, size=N),  # leave next morning
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EV charging analysis & optimization demo")
    parser.add_argument(
        "--csv", default="data/nev_battery_charging.csv",
        help="Path to the EV battery CSV",
    )
    parser.add_argument(
        "--out", default="outputs", help="Directory for plots & artifacts",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    print(f"[1/5] Loading {args.csv}")
    df = load_csv(args.csv)
    print(f"      Loaded {len(df):,} rows, columns: {list(df.columns)[:8]}...")

    print("[2/5] Summary stats")
    stats = summary_stats(df)
    stats.to_csv(out_dir / "summary_stats.csv")
    plot_distributions(df, out_dir=out_dir)
    plot_correlation(df, out_dir=out_dir)

    print("[3/5] Charging-curve and battery-temp plots")
    plot_charging_curves(df, out_dir=out_dir)
    plot_battery_temp(df, out_dir=out_dir)

    print("[4/5] Synthesizing grid + solar (CSV has no grid columns) and plotting")
    df = df.copy()
    df["base_load_kw"] = synthesize_grid_load(df)
    df["solar_kw"] = synthesize_solar_output(df)
    df["ev_load_kw"] = ev_load_from_battery(df)
    plot_solar_output(df, out_dir=out_dir)
    plot_grid_load(df, solar_df=df, out_dir=out_dir)

    print("[5/5] Running optimization demo (6 EVs, 96 timesteps)")
    problem = _build_demo_problem(df)
    naive = naive_baseline(problem)
    try:
        opt = solve_peak_minimization(problem)
        print(f"      naive peak: {naive.peak_kw:.2f} kW")
        print(f"      optimized peak ({opt.status}): {opt.peak_kw:.2f} kW")
        print(f"      reduction: {(1 - opt.peak_kw / naive.peak_kw) * 100:.1f}%")
    except Exception as e:  # pragma: no cover - solver-environment dependent
        print(f"      solver failed: {e}\n      naive peak: {naive.peak_kw:.2f} kW")

    print(f"\nDone. Artifacts in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
