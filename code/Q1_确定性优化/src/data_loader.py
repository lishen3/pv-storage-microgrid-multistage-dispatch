from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["时间", "电价", "小区负载", "光伏发电预测功率"]


def load_attachment1(path) -> pd.DataFrame:
    """读取附件1，并统一为 Q1 优化所需的 144 个 10 分钟时段。"""
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"附件1缺少字段: {missing}; 当前字段={list(df.columns)}")

    df = df[REQUIRED_COLUMNS].copy()
    df.columns = ["time", "price", "load_kw", "pv_kw"]

    for c in ["price", "load_kw", "pv_kw"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    if len(df) != 144:
        raise ValueError(f"Q1预期144个10分钟时段，实际读取到 {len(df)} 行。")

    if (df[["price", "load_kw", "pv_kw"]] < 0).any().any():
        raise ValueError("检测到负电价/负负荷/负光伏，请先核对原始数据。")

    return df.reset_index(drop=True)
