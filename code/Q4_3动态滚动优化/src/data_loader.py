from __future__ import annotations
import numpy as np
import pandas as pd


def load_price(path):
    df = pd.read_excel(path, sheet_name=0)
    if len(df) != 144:
        raise ValueError(f'附件1应有144个10分钟时段，实际{len(df)}')
    return pd.to_numeric(df['电价'], errors='raise').to_numpy(float)


def load_actual_matrix(path):
    load_df = pd.read_excel(path, sheet_name='小区负载')
    pv_df = pd.read_excel(path, sheet_name='光伏发电实际功率')
    dates = pd.to_datetime(load_df.iloc[:,0])
    if not dates.equals(pd.to_datetime(pv_df.iloc[:,0])):
        raise ValueError('负荷与光伏日期不一致')
    load = load_df.iloc[:,1:].apply(pd.to_numeric, errors='raise').to_numpy(float)
    pv = pv_df.iloc[:,1:].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if load.shape != (365,144) or pv.shape != (365,144):
        raise ValueError(f'附件2预期365x144，实际负荷{load.shape}，光伏{pv.shape}')
    if np.any(load < 0) or np.any(pv < 0):
        raise ValueError('检测到负负荷或负光伏')
    net = load - pv
    return dates, load, pv, net
