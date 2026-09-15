from __future__ import annotations
import numpy as np
from src.forecast import _row_features


def build_recent_calibration(net_kw, dates, day_idx, med, window_days=21):
    """
    Build a strictly causal calibration set:
    only days before target day are used.
    """
    start = max(14, day_idx - window_days)
    X, y, days, slots = [], [], [], []
    for d in range(start, day_idx):
        for t in range(144):
            X.append(_row_features(net_kw, dates, d, t))
            y.append(net_kw[d, t])
            days.append(d)
            slots.append(t)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    days = np.asarray(days)
    slots = np.asarray(slots)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X, y, days, slots


def _higher_quantile(values, q):
    values = np.asarray(values, float)
    if len(values) == 0:
        return 0.0
    q = min(1.0, max(0.0, q))
    return float(np.quantile(values, q, method="higher"))


def adaptive_block_qhat(
    models, Xcal, ycal, cal_slots,
    target_coverage=0.82, n_blocks=6
):
    """
    Time-local adaptive conformal expansion.
    Separate qhat is estimated for each intraday block using only recent,
    already observed errors.

    This reacts to seasonal/distribution shifts without using target-day truth.
    """
    raw_lo = models[0.1].predict(Xcal)
    raw_hi = models[0.9].predict(Xcal)
    lo = np.minimum(raw_lo, raw_hi)
    hi = np.maximum(raw_lo, raw_hi)

    scores = np.maximum.reduce([lo - ycal, ycal - hi, np.zeros_like(ycal)])

    block_width = 144 // n_blocks
    qhat_blocks = np.zeros(n_blocks)

    for b in range(n_blocks):
        s0 = b * block_width
        s1 = 144 if b == n_blocks - 1 else (b + 1) * block_width
        mask = (cal_slots >= s0) & (cal_slots < s1)
        sb = scores[mask]
        n = len(sb)
        if n == 0:
            qhat_blocks[b] = _higher_quantile(scores, target_coverage)
        else:
            # finite-sample split-conformal style quantile
            level = min(1.0, np.ceil((n + 1) * target_coverage) / n)
            qhat_blocks[b] = _higher_quantile(sb, level)

    qhat_by_slot = np.zeros(144)
    for b in range(n_blocks):
        s0 = b * block_width
        s1 = 144 if b == n_blocks - 1 else (b + 1) * block_width
        qhat_by_slot[s0:s1] = qhat_blocks[b]

    return qhat_by_slot, qhat_blocks


def calibration_coverage(models, Xcal, ycal, cal_slots, qhat_by_slot):
    r10 = models[0.1].predict(Xcal)
    r50 = models[0.5].predict(Xcal)
    r90 = models[0.9].predict(Xcal)

    qh = qhat_by_slot[cal_slots]
    lo = np.minimum(r10, r90) - qh
    hi = np.maximum(r10, r90) + qh

    return (
        float(np.mean((ycal >= lo) & (ycal <= hi))),
        float(np.mean(np.abs(ycal - r50))),
    )


def make_v4_scenarios(
    models, Xcal, ycal, cal_days, cal_slots,
    Xtarget, qhat_by_slot, n_scenarios=7, seed=0
):
    """
    Intervals and scenarios are deliberately separated:
    - adaptive conformal controls interval coverage;
    - optimization scenarios are bootstrapped COMPLETE recent residual days,
      preserving temporal correlation.

    Extreme residual profiles are real historical days, not artificial Q90 paths.
    """
    raw10 = models[0.1].predict(Xtarget)
    q50 = models[0.5].predict(Xtarget)
    raw90 = models[0.9].predict(Xtarget)

    q10 = np.minimum(raw10, raw90) - qhat_by_slot
    q90 = np.maximum(raw10, raw90) + qhat_by_slot

    pred_cal = models[0.5].predict(Xcal)
    resid = ycal - pred_cal

    profiles = []
    for d in np.unique(cal_days):
        m = cal_days == d
        if m.sum() == 144:
            order = np.argsort(cal_slots[m])
            profiles.append(resid[m][order])
    R = np.asarray(profiles, float)
    if len(R) < 3:
        raise RuntimeError("近期完整残差日不足")

    # Recent residuals get slightly more sampling weight.
    ages = np.arange(len(R))[::-1]
    w = np.exp(-np.log(2) * ages / 7.0)
    w = w / w.sum()

    rng = np.random.default_rng(seed)
    n_random = max(1, n_scenarios - 2)
    picks = rng.choice(len(R), size=n_random, replace=True, p=w)

    # Include actual historical low/high daily residual-energy profiles.
    energy_err = R.sum(axis=1)
    low = R[int(np.argmin(energy_err))]
    high = R[int(np.argmax(energy_err))]

    scenarios = q50[None, :] + np.vstack([R[picks], low[None, :], high[None, :]])

    lo_phys = float(np.percentile(ycal, 0.1) - 1200)
    hi_phys = float(np.percentile(ycal, 99.9) + 1200)
    scenarios = np.clip(scenarios, lo_phys, hi_phys)

    return q10, q50, q90, scenarios
