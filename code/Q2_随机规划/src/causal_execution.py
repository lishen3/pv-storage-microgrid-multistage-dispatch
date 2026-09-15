from __future__ import annotations
import numpy as np


def causal_execute_plan(actual_load_kw, actual_pv_kw, purchase_kwh, price, *, soc0,
                        dt=1/6, soc_min=1200.0, soc_max=10800.0,
                        pmax_kw=5000.0, eta_c=0.9, eta_d=0.9,
                        emergency_multiplier=5.0):
    """Strictly causal 10-minute real-time execution of a contracted grid schedule.

    At slot t the controller sees only current actual load/PV and current SOC.
    It never optimizes against future actual values.

    Rule:
      1) contracted grid + current PV first serves current load;
      2) surplus charges storage as much as current constraints allow;
      3) remaining surplus first curtails PV, then is counted as unused contracted grid;
      4) deficit discharges storage as much as current constraints allow;
      5) remaining deficit is emergency purchase at 5x current price.

    This policy is intentionally simple, transparent and identical for Q2 and Q3,
    so the economic value of Q3 comes only from new forecasts / plan adjustments.
    """
    load=np.asarray(actual_load_kw,float)*dt
    pv=np.asarray(actual_pv_kw,float)*dt
    grid=np.asarray(purchase_kwh,float)
    p=np.asarray(price,float)
    T=len(grid)
    if not (len(load)==len(pv)==len(p)==T):
        raise ValueError('causal execution length mismatch')
    pmax=pmax_kw*dt

    em=np.zeros(T); ch=np.zeros(T); dis=np.zeros(T)
    spill=np.zeros(T); unused=np.zeros(T); socs=np.zeros(T); residual=np.zeros(T)
    soc=float(soc0)
    for t in range(T):
        # surplus before storage: grid + PV - load
        surplus=grid[t]+pv[t]-load[t]
        if surplus >= 0:
            # charge only from currently available surplus
            max_by_room=max(0.0,(soc_max-soc)/eta_c)
            c=min(surplus,pmax,max_by_room)
            ch[t]=c
            soc += eta_c*c
            rem=surplus-c
            # If excess remains, curtail free PV first; paid grid is unused only after PV is exhausted.
            spill[t]=min(rem,pv[t])
            unused[t]=max(0.0,rem-spill[t])
        else:
            deficit=-surplus
            max_by_soc=max(0.0,(soc-soc_min)*eta_d)
            d=min(deficit,pmax,max_by_soc)
            dis[t]=d
            soc -= d/eta_d
            em[t]=deficit-d
        # numerical clipping only
        if abs(soc-soc_min)<1e-9: soc=soc_min
        if abs(soc-soc_max)<1e-9: soc=soc_max
        socs[t]=soc
        residual[t]=grid[t]+em[t]+dis[t]+pv[t]-(load[t]+ch[t]+spill[t]+unused[t])

    plan_cost=float(np.dot(p,grid))
    em_cost=float(np.dot(emergency_multiplier*p,em))
    return {
        'emergency_kwh':em,'charge_kwh':ch,'discharge_kwh':dis,
        'pv_spill_kwh':spill,'unused_contract_kwh':unused,'soc_kwh':socs,
        'soc_start_kwh':float(soc0),'soc_end_kwh':float(socs[-1]),
        'plan_cost_yuan':plan_cost,'emergency_cost_yuan':em_cost,
        'total_cost_yuan':plan_cost+em_cost,
        'total_emergency_kwh':float(em.sum()),'total_grid_kwh':float(grid.sum()),
        'total_external_purchase_kwh':float(grid.sum()+em.sum()),
        'total_pv_spill_kwh':float(spill.sum()),
        'total_unused_contract_kwh':float(unused.sum()),
        'max_abs_balance_residual_kwh':float(np.max(np.abs(residual))),
        'simultaneous_charge_discharge_periods':int(np.sum((ch>1e-7)&(dis>1e-7))),
        'balance_residual_kwh':residual,
    }


def causal_execute_segment(actual_load_kw, actual_pv_kw, purchase_kwh, price, *, soc0, **kwargs):
    return causal_execute_plan(actual_load_kw,actual_pv_kw,purchase_kwh,price,soc0=soc0,**kwargs)
