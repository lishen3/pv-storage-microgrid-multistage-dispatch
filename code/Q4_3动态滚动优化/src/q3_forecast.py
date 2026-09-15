from __future__ import annotations
import numpy as np

from src.forecast import (
    build_rolling_sets, target_features, fit_quantile_models
)
from src.forecast_v4 import (
    build_recent_calibration, adaptive_block_qhat
)


def build_load_model(load_kw, dates, month_start, month,
                     train_window_days=120, calibration_days=21,
                     half_life_days=45, seed=20260911):
    """
    Forecast LOAD only. PV comes from official attachment3 forecasts.
    """
    fit_pack, _ = build_rolling_sets(
        load_kw, dates, month_start, train_window_days, calibration_days
    )
    Xf, yf, fd, fs, med = fit_pack
    models = fit_quantile_models(
        Xf, yf, fd, month_start, half_life_days, seed + month
    )
    return models, fit_pack


def load_forecast_at_00(load_kw, dates, day_idx, pack,
                        calibration_days=21, target_coverage=0.82, n_blocks=6):
    models, fit_pack = pack
    Xf, yf, fd, fs, med = fit_pack

    Xc, yc, cd, cs = build_recent_calibration(
        load_kw, dates, day_idx, med, calibration_days
    )
    qhat_slot, _ = adaptive_block_qhat(
        models, Xc, yc, cs,
        target_coverage=target_coverage, n_blocks=n_blocks
    )
    Xt = target_features(load_kw, dates, day_idx, med)
    q50 = models[0.5].predict(Xt)

    # historical complete-day load residual trajectories
    pred_cal = models[0.5].predict(Xc)
    resid = yc - pred_cal
    profiles = []
    for d in np.unique(cd):
        m = cd == d
        if m.sum() == 144:
            order = np.argsort(cs[m])
            profiles.append(resid[m][order])
    if not profiles:
        profiles = [np.zeros(144)]
    return q50, np.asarray(profiles, float), qhat_slot
