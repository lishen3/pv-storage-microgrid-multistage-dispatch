from __future__ import annotations
import numpy as np
import pandas as pd

ISSUE_HOURS = (0, 6, 12, 18)
# Attachment2 / result3 slots are aligned to timestamps:
# 0:10, 0:20, ..., 23:50, 0:00+1 (24:00).
SLOT_HOURS = np.arange(1, 145, dtype=float) / 6.0


def load_pv_forecasts(path):
    df = pd.read_excel(path)
    df['日期'] = df['日期'].ffill()
    df['日期'] = pd.to_datetime(df['日期']).dt.normalize()
    cols = [f'预报{i}小时' for i in range(1,25)]
    out = {}
    for _, r in df.iterrows():
        issue = int(str(r['预报时刻']).split(':')[0])
        out[(r['日期'], issue)] = pd.to_numeric(r[cols], errors='raise').to_numpy(float)
    return out


def issue_start_slot(issue_hour: int) -> int:
    """First not-yet-executed 10-min slot that can be changed at issue time.

    Attachment2/result templates are timestamped by interval END:
    6:00 is the already-finished 5:50-6:00 interval, so a forecast released
    at 6:00 may first affect the 6:00-6:10 interval whose timestamp is 6:10.
    """
    if issue_hour == 0:
        return 0
    return issue_hour * 6


def _known_anchor_pv(actual_pv_kw, day_idx: int, issue_hour: int) -> float:
    """PV known exactly at the release instant; never reads a future slot."""
    if issue_hour == 0:
        # 0:00 is previous day's 0:00+1 value, already observed.
        return float(actual_pv_kw[day_idx-1, -1]) if day_idx > 0 else 0.0
    return float(actual_pv_kw[day_idx, issue_hour*6 - 1])


def forecast_for_remaining_day(forecast_dict, dates, actual_pv_kw, day_idx, issue_hour):
    """
    Linear interpolation, exactly aligned to attachment2/result3 timestamps.

    Attachment3 '预报1小时' ... '预报24小时' are treated as forecasts at
    issue+1h ... issue+24h. The value at the release instant is already known
    from measurements and is used only as the interpolation anchor.
    """
    date = pd.Timestamp(dates[day_idx]).normalize()
    hourly = np.asarray(forecast_dict[(date, issue_hour)], float)
    anchor = _known_anchor_pv(actual_pv_kw, day_idx, issue_hour)
    xp = np.arange(0, 25, dtype=float)
    yp = np.r_[anchor, hourly]

    start = issue_start_slot(issue_hour)
    target_abs = SLOT_HOURS[start:]
    rel = target_abs - issue_hour
    pred = np.interp(rel, xp, yp)
    return np.maximum(pred, 0.0)


def historical_pv_error_profiles(forecast_dict, dates, actual_pv_kw, day_idx,
                                  issue_hour, window_days=28):
    start = issue_start_slot(issue_hour)
    profiles=[]
    for d in range(max(1, day_idx-window_days), day_idx):
        key=(pd.Timestamp(dates[d]).normalize(), issue_hour)
        if key not in forecast_dict:
            continue
        pred=forecast_for_remaining_day(forecast_dict, dates, actual_pv_kw, d, issue_hour)
        actual=actual_pv_kw[d, start:]
        if len(pred)==len(actual):
            profiles.append(actual-pred)
    if not profiles:
        return np.zeros((1, 144-start))
    return np.asarray(profiles,float)
