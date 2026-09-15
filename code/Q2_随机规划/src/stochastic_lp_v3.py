from __future__ import annotations
import numpy as np
from scipy.optimize import linprog


def _terminal_value_coeff(price, eta_c=0.9):
    """
    Approximate tomorrow-value of 1 kWh stored energy.
    Use the lower quartile purchase price / charging efficiency:
    conservative enough to prevent draining, but not high enough to make
    the optimizer buy expensive energy just to fill the battery.
    """
    return float(np.quantile(np.asarray(price, float), 0.25) / eta_c)


def solve_two_stage_continuous(
    net_scenarios_kw,
    price,
    *,
    soc0,
    terminal_ref_kwh=6000.0,
    enforce_terminal=False,
    dt=1/6,
    soc_min=1200.0,
    soc_max=10800.0,
    pmax_kw=5000.0,
    eta_c=0.9,
    eta_d=0.9,
    emergency_multiplier=5.0,
    terminal_penalty_yuan_per_kwh=None,
):
    """
    Two-stage stochastic LP for a single day with cross-day SOC continuity.

    First-stage:
      grid[t] = day-ahead purchase plan, common to all scenarios.

    Second-stage by scenario:
      emergency, charge, discharge, spill, SOC.

    Unlike V1/V2, SOC at 24:00 is NOT forced to 6000 every day.
    Instead, a one-sided terminal shortfall penalty represents tomorrow's value:
      z_s >= terminal_ref - SOC_end_s
      terminal_penalty * z_s

    On the final date, enforce_terminal=True can force SOC_end = terminal_ref.
    """
    scenarios = np.asarray(net_scenarios_kw, float)
    price = np.asarray(price, float)
    S, T = scenarios.shape
    if T != len(price):
        raise ValueError("scenario horizon and price length mismatch")

    net = scenarios * dt
    pmax = pmax_kw * dt
    terminal_penalty = (
        _terminal_value_coeff(price, eta_c)
        if terminal_penalty_yuan_per_kwh is None
        else float(terminal_penalty_yuan_per_kwh)
    )

    # x = grid[T] + each scenario [emergency, charge, discharge, spill, soc, terminal_shortfall]
    block = 5 * T + 1
    n = T + S * block
    c = np.zeros(n)
    c[:T] = price
    prob = 1.0 / S

    A_eq, b_eq = [], []
    A_ub, b_ub = [], []

    for s in range(S):
        b = T + s * block
        ie = b
        ic = b + T
        idis = b + 2*T
        iw = b + 3*T
        isoc = b + 4*T
        iz = b + 5*T

        c[ie:ie+T] = prob * emergency_multiplier * price
        # Tiny throughput/spill penalty only breaks degenerate cycles.
        c[ic:iw+T] += 1e-7
        if not enforce_terminal:
            c[iz] = prob * terminal_penalty

        # Energy balance
        for t in range(T):
            row = np.zeros(n)
            row[t] = 1.0
            row[ie+t] = 1.0
            row[idis+t] = 1.0
            row[ic+t] = -1.0
            row[iw+t] = -1.0
            A_eq.append(row)
            b_eq.append(net[s, t])

        # SOC transition
        for t in range(T):
            row = np.zeros(n)
            row[isoc+t] = 1.0
            row[ic+t] = -eta_c
            row[idis+t] = 1.0 / eta_d
            if t == 0:
                rhs = float(soc0)
            else:
                row[isoc+t-1] = -1.0
                rhs = 0.0
            A_eq.append(row)
            b_eq.append(rhs)

        if enforce_terminal:
            row = np.zeros(n)
            row[isoc+T-1] = 1.0
            A_eq.append(row)
            b_eq.append(float(terminal_ref_kwh))
        else:
            # terminal_ref - SOC_end <= z  ->  -SOC_end - z <= -terminal_ref
            row = np.zeros(n)
            row[isoc+T-1] = -1.0
            row[iz] = -1.0
            A_ub.append(row)
            b_ub.append(-float(terminal_ref_kwh))

    bounds = [(0, None)] * T
    for s in range(S):
        bounds += [(0, None)] * T
        bounds += [(0, pmax)] * T
        bounds += [(0, pmax)] * T
        bounds += [(0, None)] * T
        bounds += [(soc_min, soc_max)] * T
        bounds += [(0, None)]  # terminal shortfall

    res = linprog(
        c,
        A_ub=np.asarray(A_ub) if A_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        A_eq=np.asarray(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"two-stage LP failed: {res.message}")

    return res.x[:T], {
        "objective_with_terminal_value": float(res.fun),
        "terminal_penalty_yuan_per_kwh": terminal_penalty,
    }


def solve_recourse_continuous(
    actual_net_kw,
    grid_kwh,
    price,
    *,
    soc0,
    terminal_ref_kwh=6000.0,
    enforce_terminal=False,
    dt=1/6,
    soc_min=1200.0,
    soc_max=10800.0,
    pmax_kw=5000.0,
    eta_c=0.9,
    eta_d=0.9,
    emergency_multiplier=5.0,
    terminal_penalty_yuan_per_kwh=None,
):
    """
    Execute a fixed day-ahead purchase plan on actual realized net load.

    The reported 'total_cost_yuan' contains ONLY real electricity costs:
      plan purchase + emergency purchase.
    The terminal penalty is a decision regularizer / future-value proxy and is
    reported separately, not counted as actual electricity expenditure.
    """
    actual_net_kw = np.asarray(actual_net_kw, float)
    grid_kwh = np.asarray(grid_kwh, float)
    price = np.asarray(price, float)
    T = len(actual_net_kw)
    net = actual_net_kw * dt
    pmax = pmax_kw * dt
    terminal_penalty = (
        _terminal_value_coeff(price, eta_c)
        if terminal_penalty_yuan_per_kwh is None
        else float(terminal_penalty_yuan_per_kwh)
    )

    # x = emergency[T], charge[T], discharge[T], spill[T], soc[T], z
    n = 5*T + 1
    ie, ic, idis, iw, isoc, iz = 0, T, 2*T, 3*T, 4*T, 5*T

    c = np.zeros(n)
    c[ie:ie+T] = emergency_multiplier * price
    c[ic:iw+T] += 1e-7
    if not enforce_terminal:
        c[iz] = terminal_penalty

    A_eq, b_eq = [], []
    A_ub, b_ub = [], []

    for t in range(T):
        row = np.zeros(n)
        row[ie+t] = 1.0
        row[idis+t] = 1.0
        row[ic+t] = -1.0
        row[iw+t] = -1.0
        A_eq.append(row)
        b_eq.append(net[t] - grid_kwh[t])

    for t in range(T):
        row = np.zeros(n)
        row[isoc+t] = 1.0
        row[ic+t] = -eta_c
        row[idis+t] = 1.0 / eta_d
        if t == 0:
            rhs = float(soc0)
        else:
            row[isoc+t-1] = -1.0
            rhs = 0.0
        A_eq.append(row)
        b_eq.append(rhs)

    if enforce_terminal:
        row = np.zeros(n)
        row[isoc+T-1] = 1.0
        A_eq.append(row)
        b_eq.append(float(terminal_ref_kwh))
    else:
        row = np.zeros(n)
        row[isoc+T-1] = -1.0
        row[iz] = -1.0
        A_ub.append(row)
        b_ub.append(-float(terminal_ref_kwh))

    bounds = []
    bounds += [(0, None)] * T
    bounds += [(0, pmax)] * T
    bounds += [(0, pmax)] * T
    bounds += [(0, None)] * T
    bounds += [(soc_min, soc_max)] * T
    bounds += [(0, None)]

    res = linprog(
        c,
        A_ub=np.asarray(A_ub) if A_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        A_eq=np.asarray(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"recourse LP failed: {res.message}")

    x = res.x
    emergency = x[ie:ie+T]
    charge = x[ic:ic+T]
    discharge = x[idis:idis+T]
    spill = x[iw:iw+T]
    soc = x[isoc:isoc+T]
    shortfall = float(x[iz]) if not enforce_terminal else max(0.0, terminal_ref_kwh - soc[-1])

    plan_cost = float(np.dot(price, grid_kwh))
    emergency_cost = float(np.dot(emergency_multiplier * price, emergency))
    actual_cost = plan_cost + emergency_cost

    balance = grid_kwh + emergency + discharge - charge - spill - net

    return {
        "emergency_kwh": emergency,
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "spill_kwh": spill,
        "soc_kwh": soc,
        "soc_start_kwh": float(soc0),
        "soc_end_kwh": float(soc[-1]),
        "terminal_shortfall_kwh": shortfall,
        "terminal_value_penalty_yuan": float(shortfall * terminal_penalty),
        "terminal_penalty_yuan_per_kwh": terminal_penalty,
        "plan_cost_yuan": plan_cost,
        "emergency_cost_yuan": emergency_cost,
        "total_cost_yuan": actual_cost,
        "total_emergency_kwh": float(emergency.sum()),
        "total_grid_kwh": float(grid_kwh.sum()),
        "max_abs_balance_residual_kwh": float(np.max(np.abs(balance))),
        "simultaneous_charge_discharge_periods": int(
            np.sum((charge > 1e-7) & (discharge > 1e-7))
        ),
    }


def solve_deterministic_plan_continuous(pred_net_kw, price, **kwargs):
    grid, meta = solve_two_stage_continuous(
        np.asarray(pred_net_kw, float)[None, :], price, **kwargs
    )
    return grid, meta
