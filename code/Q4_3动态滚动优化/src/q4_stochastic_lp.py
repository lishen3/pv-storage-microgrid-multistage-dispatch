from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

def _terminal_value_coeff(price,eta_c=0.9):
    return float(np.quantile(np.asarray(price,float),0.25)/eta_c)

def solve_two_stage_joint(net_scenarios_kw, price_scenarios, *,
                          soc0, terminal_ref_kwh=6000.0,
                          enforce_terminal=False, dt=1/6,
                          soc_min=1200.0,soc_max=10800.0,pmax_kw=5000.0,
                          eta_c=0.9,eta_d=0.9,emergency_multiplier=5.0):
    netS=np.asarray(net_scenarios_kw,float)
    prS=np.asarray(price_scenarios,float)
    if netS.shape!=prS.shape:
        raise ValueError(f'源荷/价格情景维度不一致 {netS.shape} vs {prS.shape}')
    S,T=netS.shape
    net=netS*dt; pmax=pmax_kw*dt
    pbar=prS.mean(axis=0)
    term=_terminal_value_coeff(pbar,eta_c)

    # grid + each scenario [emergency, charge, discharge, spill, soc, z]
    block=5*T+1
    n=T+S*block
    c=np.zeros(n); c[:T]=pbar
    Aeq=[]; beq=[]; Aub=[]; bub=[]
    prob=1.0/S

    for s in range(S):
        b=T+s*block
        ie=b; ic=b+T; idis=b+2*T; iw=b+3*T; isoc=b+4*T; iz=b+5*T
        c[ie:ie+T]=prob*emergency_multiplier*prS[s]
        c[ic:iw+T]+=1e-7
        if not enforce_terminal:
            c[iz]=prob*term
        for t in range(T):
            row=np.zeros(n)
            row[t]=1; row[ie+t]=1; row[idis+t]=1
            row[ic+t]=-1; row[iw+t]=-1
            Aeq.append(row); beq.append(net[s,t])
        for t in range(T):
            row=np.zeros(n)
            row[isoc+t]=1; row[ic+t]=-eta_c; row[idis+t]=1/eta_d
            if t==0: rhs=float(soc0)
            else:
                row[isoc+t-1]=-1; rhs=0.0
            Aeq.append(row); beq.append(rhs)
        if enforce_terminal:
            row=np.zeros(n); row[isoc+T-1]=1
            Aeq.append(row); beq.append(float(terminal_ref_kwh))
        else:
            row=np.zeros(n); row[isoc+T-1]=-1; row[iz]=-1
            Aub.append(row); bub.append(-float(terminal_ref_kwh))

    bounds=[(0,None)]*T
    for _ in range(S):
        bounds += [(0,None)]*T + [(0,pmax)]*T + [(0,pmax)]*T
        bounds += [(0,None)]*T + [(soc_min,soc_max)]*T + [(0,None)]
    res=linprog(c,A_ub=np.asarray(Aub) if Aub else None,b_ub=np.asarray(bub) if bub else None,
                A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),bounds=bounds,method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[:T], {'objective':float(res.fun),'forecast_price_mean':pbar}

def solve_point_plan(pred_net_kw,pred_price,**kwargs):
    return solve_two_stage_joint(np.asarray(pred_net_kw)[None,:],
                                 np.asarray(pred_price)[None,:],**kwargs)

def execute_actual(actual_net_kw,grid_kwh,actual_price,*,soc0,
                   terminal_ref_kwh=6000.0,enforce_terminal=False,dt=1/6,
                   soc_min=1200.0,soc_max=10800.0,pmax_kw=5000.0,
                   eta_c=0.9,eta_d=0.9,emergency_multiplier=5.0):
    net=np.asarray(actual_net_kw,float)*dt
    g=np.asarray(grid_kwh,float); p=np.asarray(actual_price,float)
    T=len(net); pmax=pmax_kw*dt
    term=_terminal_value_coeff(p,eta_c)

    # emergency, charge, discharge, spill, soc, z
    n=5*T+1; ie=0; ic=T; idis=2*T; iw=3*T; isoc=4*T; iz=5*T
    c=np.zeros(n); c[ie:ie+T]=emergency_multiplier*p
    c[ic:iw+T]+=1e-7
    if not enforce_terminal: c[iz]=term
    Aeq=[];beq=[];Aub=[];bub=[]
    for t in range(T):
        row=np.zeros(n); row[ie+t]=1;row[idis+t]=1;row[ic+t]=-1;row[iw+t]=-1
        Aeq.append(row);beq.append(net[t]-g[t])
    for t in range(T):
        row=np.zeros(n);row[isoc+t]=1;row[ic+t]=-eta_c;row[idis+t]=1/eta_d
        if t==0: rhs=float(soc0)
        else: row[isoc+t-1]=-1;rhs=0
        Aeq.append(row);beq.append(rhs)
    if enforce_terminal:
        row=np.zeros(n);row[isoc+T-1]=1;Aeq.append(row);beq.append(terminal_ref_kwh)
    else:
        row=np.zeros(n);row[isoc+T-1]=-1;row[iz]=-1;Aub.append(row);bub.append(-terminal_ref_kwh)

    bounds=[(0,None)]*T+[(0,pmax)]*T+[(0,pmax)]*T+[(0,None)]*T+[(soc_min,soc_max)]*T+[(0,None)]
    res=linprog(c,A_ub=np.asarray(Aub) if Aub else None,b_ub=np.asarray(bub) if bub else None,
                A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),bounds=bounds,method='highs')
    if not res.success: raise RuntimeError(res.message)
    x=res.x
    em=x[ie:ie+T];ch=x[ic:ic+T];dis=x[idis:idis+T];spill=x[iw:iw+T];soc=x[isoc:isoc+T]
    bal=g+em+dis-ch-spill-net
    return {
        'emergency_kwh':em,'charge_kwh':ch,'discharge_kwh':dis,'spill_kwh':spill,
        'soc_kwh':soc,'soc_start_kwh':float(soc0),'soc_end_kwh':float(soc[-1]),
        'plan_cost_yuan':float(np.dot(p,g)),
        'emergency_cost_yuan':float(np.dot(5*p,em)),
        'total_cost_yuan':float(np.dot(p,g)+np.dot(5*p,em)),
        'total_emergency_kwh':float(em.sum()),
        'total_grid_kwh':float(g.sum()),
        'max_abs_balance_residual_kwh':float(np.max(np.abs(bal))),
        'simultaneous_charge_discharge_periods':int(np.sum((ch>1e-6)&(dis>1e-6)))
    }
