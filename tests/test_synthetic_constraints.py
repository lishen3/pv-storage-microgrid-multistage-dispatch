"""Small, data-free checks for the physical constraints in all five stages."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-6


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SyntheticConstraintTests(unittest.TestCase):
    def test_q1_energy_balance_soc_and_terminal_state(self):
        module = load_module("q1_dispatch", "code/Q1_确定性优化/src/dispatch_lp.py")
        frame = pd.DataFrame({
            "load_kw": [6.0, 5.0, 4.0, 6.0],
            "pv_kw": [0.0, 2.0, 7.0, 0.0],
            "price": [0.4, 0.5, 0.9, 1.0],
        })
        params = module.DispatchParams(
            dt_hours=1.0, soc_min_kwh=0.0, soc_max_kwh=10.0,
            soc_initial_kwh=5.0, soc_final_kwh=5.0,
            charge_power_max_kw=5.0, discharge_power_max_kw=5.0,
        )
        result, summary = module.solve_q1_lp(frame, params)
        self.assertLessEqual(summary["max_abs_balance_residual_kwh"], TOL)
        self.assertAlmostEqual(summary["soc_end_kwh"], 5.0, places=6)
        self.assertTrue(result["soc_kwh"].between(0.0, 10.0).all())

    def test_q2_recourse_balance_soc_and_no_simultaneous_flow(self):
        module = load_module("q2_lp", "code/Q2_随机规划/src/stochastic_lp_v3.py")
        net = np.array([4.0, 3.0, -2.0, 5.0])
        price = np.array([0.4, 0.5, 0.8, 1.0])
        plan, _ = module.solve_deterministic_plan_continuous(
            net, price, soc0=5.0, terminal_ref_kwh=5.0, enforce_terminal=True,
            dt=1.0, soc_min=0.0, soc_max=10.0, pmax_kw=5.0,
        )
        result = module.solve_recourse_continuous(
            net, plan, price, soc0=5.0, terminal_ref_kwh=5.0,
            enforce_terminal=True, dt=1.0, soc_min=0.0,
            soc_max=10.0, pmax_kw=5.0,
        )
        self.assertLessEqual(result["max_abs_balance_residual_kwh"], TOL)
        self.assertAlmostEqual(result["soc_end_kwh"], 5.0, places=6)
        self.assertEqual(result["simultaneous_charge_discharge_periods"], 0)

    def test_q3_rolling_execution_obeys_physical_constraints(self):
        module = load_module("q3_opt", "code/Q3_滚动优化/src/q3_opt_final2.py")
        load = np.array([5.0, 4.0, 3.0, 6.0])
        pv = np.array([0.0, 2.0, 5.0, 0.0])
        purchase = np.array([4.0, 3.0, 0.0, 5.0])
        result = module.execute_segment_milp(
            load, pv, purchase, np.ones(4), soc0=5.0, dt=1.0,
            soc_min=0.0, soc_max=10.0, pmax_kw=5.0,
            terminal_ref=5.0, force_terminal=True,
        )
        self.assertLessEqual(float(np.max(np.abs(result["balance_residual"]))), TOL)
        self.assertAlmostEqual(result["soc_end"], 5.0, places=6)
        self.assertEqual(result["simultaneous_count"], 0)
        self.assertTrue(np.all(result["pv_spill"] <= pv + TOL))

    def _check_q4_module(self, name: str, path: str):
        module = load_module(name, path)
        net = np.array([[4.0, 3.0, -1.0, 5.0], [5.0, 2.0, 0.0, 4.0]])
        prices = np.array([[0.4, 0.5, 0.9, 1.0], [0.45, 0.55, 0.8, 0.95]])
        plan, _ = module.solve_two_stage_joint(
            net, prices, soc0=5.0, terminal_ref_kwh=5.0,
            enforce_terminal=True, dt=1.0, soc_min=0.0,
            soc_max=10.0, pmax_kw=5.0,
        )
        result = module.execute_actual(
            net[0], plan, prices[0], soc0=5.0, terminal_ref_kwh=5.0,
            enforce_terminal=True, dt=1.0, soc_min=0.0,
            soc_max=10.0, pmax_kw=5.0,
        )
        self.assertLessEqual(result["max_abs_balance_residual_kwh"], TOL)
        self.assertAlmostEqual(result["soc_end_kwh"], 5.0, places=6)
        self.assertEqual(result["simultaneous_charge_discharge_periods"], 0)

    def test_q4_2_joint_scenarios_and_execution(self):
        self._check_q4_module("q4_2_lp", "code/Q4_2动态电价/src/q4_stochastic_lp.py")

    def test_q4_3_joint_scenarios_and_execution(self):
        self._check_q4_module("q4_3_lp", "code/Q4_3动态滚动优化/src/q4_stochastic_lp.py")


if __name__ == "__main__":
    unittest.main()
