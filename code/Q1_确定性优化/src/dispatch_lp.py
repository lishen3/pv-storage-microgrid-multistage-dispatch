from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass
class DispatchParams:
    dt_hours: float = 1 / 6
    soc_min_kwh: float = 1200.0
    soc_max_kwh: float = 10800.0
    soc_initial_kwh: float = 6000.0
    soc_final_kwh: float = 6000.0
    charge_power_max_kw: float = 5000.0
    discharge_power_max_kw: float = 5000.0
    eta_charge: float = 0.90
    eta_discharge: float = 0.90


def solve_q1_lp(df: pd.DataFrame, params: DispatchParams):
    """
    Q1：确定性线性规划。

    变量（每个时段 t）：
      grid[t]      普通购电量(kWh)
      charge[t]    充电量(kWh，交流侧)
      discharge[t] 放电量(kWh，向负荷侧)
      spill[t]     弃光量(kWh)
    状态：
      soc[t]       时段结束储能量(kWh)，共 T 个

    能量平衡：
      grid + pv + discharge = load + charge + spill

    SOC:
      soc_t = soc_{t-1} + eta_c*charge - discharge/eta_d
    """
    T = len(df)
    dt = params.dt_hours

    load = df["load_kw"].to_numpy(float) * dt
    pv = df["pv_kw"].to_numpy(float) * dt
    price = df["price"].to_numpy(float)

    # Vector layout:
    # [grid T | charge T | discharge T | spill T | soc T]
    n = 5 * T
    i_grid = slice(0, T)
    i_charge = slice(T, 2*T)
    i_dis = slice(2*T, 3*T)
    i_spill = slice(3*T, 4*T)
    i_soc = slice(4*T, 5*T)

    c = np.zeros(n)
    c[i_grid] = price  # yuan/kWh * kWh

    A_eq = []
    b_eq = []

    # 1) energy balance
    for t in range(T):
        row = np.zeros(n)
        row[t] = 1.0                    # grid
        row[2*T + t] = 1.0              # discharge
        row[T + t] = -1.0               # charge
        row[3*T + t] = -1.0             # spill
        A_eq.append(row)
        b_eq.append(load[t] - pv[t])

    # 2) SOC transition
    for t in range(T):
        row = np.zeros(n)
        row[4*T + t] = 1.0
        row[T + t] = -params.eta_charge
        row[2*T + t] = 1.0 / params.eta_discharge
        if t == 0:
            rhs = params.soc_initial_kwh
        else:
            row[4*T + t - 1] = -1.0
            rhs = 0.0
        A_eq.append(row)
        b_eq.append(rhs)

    # 3) terminal SOC
    row = np.zeros(n)
    row[5*T - 1] = 1.0
    A_eq.append(row)
    b_eq.append(params.soc_final_kwh)

    charge_max = params.charge_power_max_kw * dt
    discharge_max = params.discharge_power_max_kw * dt

    bounds = []
    bounds += [(0, None)] * T                                    # grid
    bounds += [(0, charge_max)] * T                              # charge
    bounds += [(0, discharge_max)] * T                           # discharge
    bounds += [(0, None)] * T                                    # spill
    bounds += [(params.soc_min_kwh, params.soc_max_kwh)] * T     # soc

    result = linprog(
        c,
        A_eq=np.asarray(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"LP求解失败: {result.message}")

    x = result.x

    out = df.copy()
    out["load_kwh"] = load
    out["pv_kwh"] = pv
    out["grid_kwh"] = x[i_grid]
    out["charge_kwh"] = x[i_charge]
    out["discharge_kwh"] = x[i_dis]
    out["spill_kwh"] = x[i_spill]
    out["soc_kwh"] = x[i_soc]
    out["cost_yuan"] = out["grid_kwh"] * out["price"]

    # independent checks
    balance_residual = (
        out["grid_kwh"] + out["pv_kwh"] + out["discharge_kwh"]
        - out["load_kwh"] - out["charge_kwh"] - out["spill_kwh"]
    )

    summary = {
        "solver": "scipy.optimize.linprog / HiGHS",
        "status": result.message,
        "total_grid_kwh": float(out["grid_kwh"].sum()),
        "total_cost_yuan": float(out["cost_yuan"].sum()),
        "total_charge_kwh": float(out["charge_kwh"].sum()),
        "total_discharge_kwh": float(out["discharge_kwh"].sum()),
        "total_spill_kwh": float(out["spill_kwh"].sum()),
        "soc_start_kwh": float(params.soc_initial_kwh),
        "soc_end_kwh": float(out["soc_kwh"].iloc[-1]),
        "soc_min_observed_kwh": float(out["soc_kwh"].min()),
        "soc_max_observed_kwh": float(out["soc_kwh"].max()),
        "max_abs_balance_residual_kwh": float(np.abs(balance_residual).max()),
        "simultaneous_charge_discharge_periods": int(
            ((out["charge_kwh"] > 1e-7) & (out["discharge_kwh"] > 1e-7)).sum()
        ),
    }

    return out, summary
