"""CVXPy stub: schedule N EVs over T timesteps to minimize peak grid load.

Formulation
-----------
Decision variables
    p[n, t]  : power drawn by EV n at timestep t      (kW, ≥ 0)
    peak     : scalar peak grid draw across the horizon (kW)

Parameters
    base[t]            : non-EV (baseline) grid load at time t (kW)
    solar[t]           : PV output at time t (kW, ≥ 0)  → subtracted from net
    p_max[n]           : max charging power per EV (kW)
    E_req[n]           : energy each EV needs to deliver to its battery (kWh)
    dt                 : timestep length in hours
    arrive[n], depart[n] : earliest/latest indices EV n can charge

Objective
    minimize  peak  +  lambda_soft * sum(p)        # soft term keeps it tame

Constraints
    p[n, t] ≥ 0
    p[n, t] ≤ p_max[n]
    p[n, t] = 0  if  t ∉ [arrive[n], depart[n]]
    sum_t  p[n, t] * dt = E_req[n]                  # energy delivered
    base[t] + sum_n p[n, t] - solar[t] ≤ peak       # peak definition

This is a stub: drop in real arrivals / OCPP telemetry to make it useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp
import numpy as np


@dataclass
class ChargingProblem:
    """Inputs to the smart-charging optimizer."""

    base_load_kw: np.ndarray              # shape (T,)
    p_max_kw: np.ndarray                  # shape (N,)
    energy_required_kwh: np.ndarray       # shape (N,)
    dt_hours: float = 0.25                # 15-minute steps by default
    solar_kw: Optional[np.ndarray] = None # shape (T,)
    arrival_idx: Optional[np.ndarray] = None  # shape (N,)
    departure_idx: Optional[np.ndarray] = None  # shape (N,)
    lambda_soft: float = 1e-3

    def __post_init__(self) -> None:
        self.base_load_kw = np.asarray(self.base_load_kw, dtype=float)
        self.p_max_kw = np.asarray(self.p_max_kw, dtype=float)
        self.energy_required_kwh = np.asarray(self.energy_required_kwh, dtype=float)
        if self.solar_kw is None:
            self.solar_kw = np.zeros_like(self.base_load_kw)
        else:
            self.solar_kw = np.asarray(self.solar_kw, dtype=float)
        T, N = self.T, self.N
        if self.arrival_idx is None:
            self.arrival_idx = np.zeros(N, dtype=int)
        if self.departure_idx is None:
            self.departure_idx = np.full(N, T - 1, dtype=int)

    @property
    def T(self) -> int:
        return int(self.base_load_kw.shape[0])

    @property
    def N(self) -> int:
        return int(self.p_max_kw.shape[0])


@dataclass
class ChargingSolution:
    """Outputs of the optimizer."""
    schedule_kw: np.ndarray  # shape (N, T)
    peak_kw: float
    status: str
    objective: float
    extra: dict = field(default_factory=dict)


def solve_peak_minimization(problem: ChargingProblem, solver: Optional[str] = None) -> ChargingSolution:
    """Solve the peak-grid-load minimization LP.

    Parameters
    ----------
    problem : ChargingProblem
    solver : str, optional
        Pass a CVXPy solver name (e.g. "ECOS", "SCS", "CLARABEL"). Defaults
        to CVXPy's auto-pick.

    Returns
    -------
    ChargingSolution
    """
    T, N = problem.T, problem.N
    p = cp.Variable((N, T), nonneg=True)
    peak = cp.Variable(nonneg=True)

    constraints = []

    # Per-EV power cap + availability windows
    for n in range(N):
        constraints.append(p[n, :] <= problem.p_max_kw[n])
        a, d = int(problem.arrival_idx[n]), int(problem.departure_idx[n])
        if a > 0:
            constraints.append(p[n, :a] == 0)
        if d < T - 1:
            constraints.append(p[n, d + 1:] == 0)
        # Energy delivered
        constraints.append(cp.sum(p[n, :]) * problem.dt_hours == problem.energy_required_kwh[n])

    # Peak definition (net of solar, never negative)
    total_ev = cp.sum(p, axis=0)
    net = problem.base_load_kw + total_ev - problem.solar_kw
    constraints.append(net <= peak)

    objective = cp.Minimize(peak + problem.lambda_soft * cp.sum(p))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    sched = np.zeros((N, T)) if p.value is None else np.asarray(p.value)
    return ChargingSolution(
        schedule_kw=sched,
        peak_kw=float(peak.value) if peak.value is not None else float("nan"),
        status=str(prob.status),
        objective=float(prob.value) if prob.value is not None else float("nan"),
        extra={"net_load_kw": (np.asarray(problem.base_load_kw) + sched.sum(0)
                               - np.asarray(problem.solar_kw))},
    )


def naive_baseline(problem: ChargingProblem) -> ChargingSolution:
    """Charge every EV at full p_max from arrival until its energy is met.

    A reference policy for comparing the optimized schedule against.
    """
    T, N = problem.T, problem.N
    sched = np.zeros((N, T))
    for n in range(N):
        a, d = int(problem.arrival_idx[n]), int(problem.departure_idx[n])
        e_left = float(problem.energy_required_kwh[n])
        for t in range(a, d + 1):
            if e_left <= 0:
                break
            kw = min(problem.p_max_kw[n], e_left / problem.dt_hours)
            sched[n, t] = kw
            e_left -= kw * problem.dt_hours
    net = problem.base_load_kw + sched.sum(0) - problem.solar_kw
    return ChargingSolution(
        schedule_kw=sched,
        peak_kw=float(net.max()),
        status="naive",
        objective=float(net.max()),
        extra={"net_load_kw": net},
    )
