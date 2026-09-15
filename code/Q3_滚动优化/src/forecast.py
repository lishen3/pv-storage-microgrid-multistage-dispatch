from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def _row_features(net_kw: np.ndarray, dates, day_idx: int, slot: int):
    def lag(k):
        return net_kw[day_idx-k, slot] if day_idx >= k else np.nan
    def rmean(k):
        a = max(0, day_idx-k)
        return float(np.mean(net_kw[a:day_idx, slot])) if day_idx > a else np.nan
    d = dates[day_idx]
    return np.array([
        slot,
        np.sin(2*np.pi*slot/144), np.cos(2*np.pi*slot/144),
        d.dayofweek,
        np.sin(2*np.pi*d.dayofyear/365.25), np.cos(2*np.pi*d.dayofyear/365.25),
        lag(1), lag(2), lag(7), lag(14),
        rmean(3), rmean(7), rmean(14), rmean(28),
    ], float)


def _build_rows(net_kw, dates, day_start, day_end, min_day_idx=14):
    X, y, day_ids, slots = [], [], [], []
    for d in range(max(min_day_idx, day_start), day_end):
        for t in range(144):
            X.append(_row_features(net_kw, dates, d, t))
            y.append(net_kw[d,t])
            day_ids.append(d); slots.append(t)
    X = np.asarray(X); y = np.asarray(y)
    if len(X) == 0:
        raise RuntimeError('可用历史样本不足')
    med = np.nanmedian(X, axis=0)
    inds=np.where(np.isnan(X)); X[inds]=np.take(med,inds[1])
    return X,y,np.asarray(day_ids),np.asarray(slots),med


def build_rolling_sets(net_kw, dates, history_end_day_idx,
                       train_window_days=120, calibration_days=14):
    history_start=max(0, history_end_day_idx-train_window_days)
    cal_start=max(history_start+15, history_end_day_idx-calibration_days)
    # Ensure enough fit data for early February.
    if cal_start-history_start < 10:
        cal_start=max(history_start+10, history_end_day_idx-7)
    Xfit,yfit,fit_days,fit_slots,med=_build_rows(net_kw,dates,history_start,cal_start)
    Xcal,ycal,cal_days,cal_slots,_=_build_rows(net_kw,dates,cal_start,history_end_day_idx)
    # Re-impute calibration with fit medians for strict train/cal separation.
    inds=np.where(np.isnan(Xcal)); Xcal[inds]=np.take(med,inds[1])
    return (Xfit,yfit,fit_days,fit_slots,med),(Xcal,ycal,cal_days,cal_slots)


def target_features(net_kw, dates, day_idx, med):
    X=np.vstack([_row_features(net_kw,dates,day_idx,t) for t in range(144)])
    inds=np.where(np.isnan(X)); X[inds]=np.take(med,inds[1])
    return X


def fit_quantile_models(X,y,day_ids,history_end_day_idx,half_life_days=45.0,seed=0):
    age=np.maximum(0, history_end_day_idx-1-day_ids)
    w=np.exp(-np.log(2)*age/half_life_days)
    models={}
    for q in (0.1,0.5,0.9):
        m=HistGradientBoostingRegressor(
            loss='quantile',quantile=q,max_iter=160,learning_rate=0.05,
            max_leaf_nodes=25,min_samples_leaf=30,l2_regularization=1.5,
            random_state=seed,
        )
        m.fit(X,y,sample_weight=w)
        models[q]=m
    return models


def conformal_qhat(models,Xcal,ycal,target_coverage=0.80):
    lo=models[0.1].predict(Xcal); hi=models[0.9].predict(Xcal)
    lo,hi=np.minimum(lo,hi),np.maximum(lo,hi)
    scores=np.maximum.reduce([lo-ycal,ycal-hi,np.zeros_like(ycal)])
    n=len(scores)
    # Finite-sample split-conformal quantile.
    q_level=min(1.0, np.ceil((n+1)*target_coverage)/n)
    return float(np.quantile(scores,q_level,method='higher'))


def calibration_diagnostics(models,Xcal,ycal,qhat):
    q10=models[0.1].predict(Xcal)-qhat
    q50=models[0.5].predict(Xcal)
    q90=models[0.9].predict(Xcal)+qhat
    coverage=float(np.mean((ycal>=q10)&(ycal<=q90)))
    mae=float(np.mean(np.abs(ycal-q50)))
    return coverage,mae


def make_scenarios(models,Xcal,ycal,cal_days,cal_slots,Xtarget,qhat,
                   n_scenarios=9,seed=0):
    raw10=models[0.1].predict(Xtarget); q50=models[0.5].predict(Xtarget); raw90=models[0.9].predict(Xtarget)
    q10=np.minimum(raw10,raw90)-qhat
    q90=np.maximum(raw10,raw90)+qhat

    pred_cal=models[0.5].predict(Xcal)
    resid=ycal-pred_cal
    profiles=[]
    for d in np.unique(cal_days):
        m=cal_days==d
        if m.sum()==144:
            order=np.argsort(cal_slots[m]); profiles.append(resid[m][order])
    R=np.asarray(profiles)
    if len(R)<3:
        raise RuntimeError('校准期完整残差轨迹太少')

    hist_scale=np.std(R,axis=0,ddof=1)
    hist_scale=np.where(hist_scale<50,50.0,hist_scale)
    Z=R/hist_scale
    sigma=(q90-q10)/(2*1.28155)
    sigma=np.maximum(sigma,0.45*hist_scale)

    rng=np.random.default_rng(seed)
    n_boot=max(1,n_scenarios-3)
    picks=rng.choice(len(Z),size=n_boot,replace=len(Z)<n_boot)
    boot=q50[None,:]+Z[picks,:]*sigma[None,:]
    # Always include lower, median, upper calibrated paths so tail risk cannot be missed by bootstrap luck.
    scenarios=np.vstack([boot,q10[None,:],q50[None,:],q90[None,:]])
    return q10,q50,q90,scenarios
