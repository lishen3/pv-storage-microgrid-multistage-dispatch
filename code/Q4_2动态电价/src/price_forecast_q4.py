from __future__ import annotations
import numpy as np
import pandas as pd

def load_dynamic_price(path):
    df=pd.read_excel(path, sheet_name=0)
    dates=pd.DatetimeIndex(pd.to_datetime(df.iloc[:,0]))
    P=df.iloc[:,1:].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if P.shape!=(365,144):
        raise ValueError(f'附件4预期365x144，实际{P.shape}')
    if np.any(P<=0):
        raise ValueError('动态电价存在非正数')
    return dates,P

def point_price_forecast(price, dates, day_idx, history_days=28):
    """Strictly causal next-day price forecast."""
    start=max(0,day_idx-history_days)
    hist=np.arange(start,day_idx)
    if len(hist)<7:
        raise RuntimeError('历史价格不足7天')
    target_dow=dates[day_idx].dayofweek
    same=np.array([i for i in hist if dates[i].dayofweek==target_dow],dtype=int)

    # Exponentially weighted same-slot recent mean
    ages=day_idx-hist
    w=np.exp(-np.log(2)*ages/10.0)
    w=w/w.sum()
    recent=(price[hist]*w[:,None]).sum(axis=0)

    if len(same)>=2:
        ages2=day_idx-same
        w2=np.exp(-np.log(2)*ages2/21.0); w2=w2/w2.sum()
        weekday=(price[same]*w2[:,None]).sum(axis=0)
        pred=0.65*weekday+0.35*recent
    else:
        pred=recent
    return np.clip(pred,0.001,None)

def historical_price_residual_profiles(price, dates, day_idx, window_days=21):
    """Residual profiles computed causally for prior days."""
    start=max(7, day_idx-window_days)
    prof=[]
    for d in range(start,day_idx):
        try:
            pred=point_price_forecast(price,dates,d)
        except RuntimeError:
            continue
        prof.append(price[d]-pred)
    if len(prof)<3:
        raise RuntimeError('近期价格残差轨迹不足')
    return np.asarray(prof,float)

def price_scenarios(price, dates, day_idx, n_scenarios=5, seed=0):
    point=point_price_forecast(price,dates,day_idx)
    R=historical_price_residual_profiles(price,dates,day_idx,21)
    rng=np.random.default_rng(seed)
    if n_scenarios<=2:
        picks=rng.choice(len(R),n_scenarios,replace=True)
        S=point[None,:]+R[picks]
    else:
        nrand=n_scenarios-2
        picks=rng.choice(len(R),nrand,replace=True)
        meanerr=R.mean(axis=1)
        low=R[int(np.argmin(meanerr))]
        high=R[int(np.argmax(meanerr))]
        S=point[None,:]+np.vstack([R[picks],low[None,:],high[None,:]])
    return point,np.clip(S,0.001,None)

def metrics(point,actual):
    e=np.asarray(point)-np.asarray(actual)
    return {
        'mae':float(np.mean(np.abs(e))),
        'rmse':float(np.sqrt(np.mean(e*e))),
        'mape':float(np.mean(np.abs(e)/np.maximum(actual,0.05))),
    }
