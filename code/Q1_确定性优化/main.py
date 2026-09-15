import json
from pathlib import Path

import pandas as pd

from src.config import (
    ATTACHMENT1, OUTPUT_DIR, DT_HOURS,
    SOC_MIN_KWH, SOC_MAX_KWH, SOC_INITIAL_KWH, SOC_FINAL_KWH,
    CHARGE_POWER_MAX_KW, DISCHARGE_POWER_MAX_KW,
    ETA_CHARGE, ETA_DISCHARGE,
)
from src.data_loader import load_attachment1
from src.dispatch_lp import DispatchParams, solve_q1_lp


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 读取数据: {ATTACHMENT1}")
    df = load_attachment1(ATTACHMENT1)

    params = DispatchParams(
        dt_hours=DT_HOURS,
        soc_min_kwh=SOC_MIN_KWH,
        soc_max_kwh=SOC_MAX_KWH,
        soc_initial_kwh=SOC_INITIAL_KWH,
        soc_final_kwh=SOC_FINAL_KWH,
        charge_power_max_kw=CHARGE_POWER_MAX_KW,
        discharge_power_max_kw=DISCHARGE_POWER_MAX_KW,
        eta_charge=ETA_CHARGE,
        
        eta_discharge=ETA_DISCHARGE,
    )

    print("[2/4] 求解 Q1 确定性 LP...")
    result_df, summary = solve_q1_lp(df, params)

    print("[3/4] 保存结果...")
    result_path = OUTPUT_DIR / "q1_dispatch_result.xlsx"
    summary_path = OUTPUT_DIR / "q1_summary.json"

    result_df.to_excel(result_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[4/4] 完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果表: {result_path}")
    print(f"汇总:   {summary_path}")


if __name__ == "__main__":
    main()
