from __future__ import annotations
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def _terminal_value(price, eta_c=0.9):
    # Keep Q2 V4's conservative continuation-value convention.
    return float(np.quantile(np.asarray(price, float), 0.25) / eta_c)


def solve_initial_plan(net_scenarios_kw, price, soc0, *, dt=1/6,
                       soc_min=1200, soc_max=10800, pmax_kw=5000,
                       eta_c=0.9, eta_d=0.9, emergency_multiplier=5.0,
                       terminal_ref=6000.0, force_terminal=False):
    """
    0:00 stochastic plan, inherited from the accepted Q2/Q3-final structure.
    This stage remains scenario based; the FINAL-2 correction is targeted at
    the 6/12/18 intraday adjustment stage.
    """
    scenarios=np.asarray(net_scenarios_kw,float)
    price=np.asarray(price,float)
    S,T=scenarios.shape
    net=scenarios*dt
    pmax=pmax_kw*dt
    terminal_pen=_terminal_value(price,eta_c)

    block=4*T+1
    n=T+S*block
    c=np.zeros(n); c[:T]=price
    Aeq=[]; beq=[]; Aub=[]; bub=[]
    prob=1.0/S

    for s in range(S):
        b=T+s*block
        ie=b; ic=b+T; idis=b+2*T; isoc=b+3*T; iz=b+4*T
        c[ie:ie+T]=prob*emergency_multiplier*price
        c[ic:ic+T]+=1e-8
        c[idis:idis+T]+=1e-8
        if not force_terminal:
            c[iz]=prob*terminal_pen

        # Q2-style adequacy at 0:00.
        for t in range(T):
            row=np.zeros(n)
            row[ic+t]=1; row[t]=-1; row[ie+t]=-1; row[idis+t]=-1
            Aub.append(row); bub.append(-net[s,t])

        for t in range(T):
            row=np.zeros(n)
            row[isoc+t]=1; row[ic+t]=-eta_c; row[idis+t]=1/eta_d
            if t==0:
                rhs=float(soc0)
            else:
                row[isoc+t-1]=-1; rhs=0.0
            Aeq.append(row); beq.append(rhs)

        if force_terminal:
            row=np.zeros(n); row[isoc+T-1]=1
            Aeq.append(row); beq.append(terminal_ref)
        else:
            row=np.zeros(n); row[isoc+T-1]=-1; row[iz]=-1
            Aub.append(row); bub.append(-terminal_ref)

    bounds=[(0,None)]*T
    for _ in range(S):
        bounds += [(0,None)]*T + [(0,pmax)]*T + [(0,pmax)]*T
        bounds += [(soc_min,soc_max)]*T + [(0,None)]

    res=linprog(c,A_ub=np.asarray(Aub),b_ub=np.asarray(bub),
                A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),
                bounds=bounds,method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[:T], {'objective':float(res.fun)}


def solve_deterministic_adjustment(original_plan, load_forecast_kw, pv_forecast_kw,
                                   price, soc0, *, dt=1/6,
                                   soc_min=1200, soc_max=10800, pmax_kw=5000,
                                   eta_c=0.9, eta_d=0.9,
                                   terminal_ref=6000.0, force_terminal=False):
    """
    FINAL-2 intraday adjustment.

    Candidate plans MUST have a physical destination for every purchased kWh:
        A + D + (PV - W) = L + C

    W is PV curtailment only and satisfies 0 <= W <= PV.
    No 'grid spill' / free supply slack exists in the candidate optimization.
    Charge/discharge mutual exclusion is enforced with a binary variable.
    """
    P=np.asarray(original_plan,float)
    load=np.asarray(load_forecast_kw,float)*dt
    pv=np.asarray(pv_forecast_kw,float)*dt
    price=np.asarray(price,float)
    T=len(P); pmax=pmax_kw*dt
    term=_terminal_value(price,eta_c)

    # x = A, down, up, charge, discharge, pv_spill, soc, binary, terminal_shortfall
    n=8*T+1
    iA=0; idn=T; iup=2*T; ic=3*T; idis=4*T
    iw=5*T; isoc=6*T; ib=7*T; iz=8*T

    c=np.zeros(n)
    c[iA:iA+T]=price
    c[idn:idn+T]=0.5*price
    c[iup:iup+T]=0.5*price
    c[ic:ic+T]+=1e-8
    c[idis:idis+T]+=1e-8
    c[iw:iw+T]+=1e-10
    if not force_terminal:
        c[iz]=term

    rows=[]; lb=[]; ub=[]

    # A = P - down + up
    for t in range(T):
        rows.append({iA+t:1, idn+t:1, iup+t:-1})
        lb.append(P[t]); ub.append(P[t])

    # Exact physical energy balance:
    # A + D + PV - W = L + C
    # => A + D - W - C = L - PV
    for t in range(T):
        rows.append({iA+t:1, idis+t:1, iw+t:-1, ic+t:-1})
        rhs=load[t]-pv[t]
        lb.append(rhs); ub.append(rhs)

    # SOC transition
    for t in range(T):
        r={isoc+t:1, ic+t:-eta_c, idis+t:1/eta_d}
        rhs=float(soc0) if t==0 else 0.0
        if t>0:
            r[isoc+t-1]=-1
        rows.append(r); lb.append(rhs); ub.append(rhs)

    # strict mutual exclusion
    for t in range(T):
        rows.append({ic+t:1, ib+t:-pmax})
        lb.append(-np.inf); ub.append(0.0)
        rows.append({idis+t:1, ib+t:pmax})
        lb.append(-np.inf); ub.append(pmax)

    if force_terminal:
        rows.append({isoc+T-1:1})
        lb.append(terminal_ref); ub.append(terminal_ref)
    else:
        # z >= terminal_ref - S_end
        rows.append({isoc+T-1:-1, iz:-1})
        lb.append(-np.inf); ub.append(-terminal_ref)

    M=lil_matrix((len(rows),n),dtype=float)
    for i,r in enumerate(rows):
        for j,v in r.items():
            M[i,j]=v

    lower=np.zeros(n)
    upper=np.full(n,np.inf)
    upper[ic:ic+T]=pmax
    upper[idis:idis+T]=pmax
    upper[iw:iw+T]=pv               # only PV may be curtailed
    lower[isoc:isoc+T]=soc_min
    upper[isoc:isoc+T]=soc_max
    upper[ib:ib+T]=1.0
    integrality=np.zeros(n,dtype=int)
    integrality[ib:ib+T]=1

    res=milp(c,integrality=integrality,
             bounds=Bounds(lower,upper),
             constraints=LinearConstraint(M.tocsr(),
                                          np.asarray(lb,float),
                                          np.asarray(ub,float)),
             options={'time_limit':30,'disp':False})
    if not res.success:
        raise RuntimeError(f'Adjustment MILP failed: {res.message}')
    x=res.x
    return x[iA:iA+T], {
        'objective':float(res.fun),
        'down_kwh':x[idn:idn+T],
        'up_kwh':x[iup:iup+T],
        'pv_spill_kwh':x[iw:iw+T],
        'simultaneous_count':int(np.sum((x[ic:ic+T]>1e-7)&(x[idis:idis+T]>1e-7)))
    }


def deterministic_keep_cost(current_plan, original_plan,
                            load_forecast_kw, pv_forecast_kw,
                            price, soc0, *,
                            dt=1/6, soc_min=1200, soc_max=10800,
                            pmax_kw=5000, eta_c=0.9, eta_d=0.9,
                            emergency_multiplier=5.0,
                            terminal_ref=6000.0):
    """
    Cost of keeping the already contracted schedule.

    A fixed old plan can be larger than the newest forecast. To keep this
    comparison always feasible, 'unused' represents already-paid contracted
    energy that cannot be consumed after the forecast update. It is NOT
    available in solve_deterministic_adjustment, so the optimizer cannot
    deliberately buy electricity and discard it.
    """
    A=np.asarray(current_plan,float)
    P=np.asarray(original_plan,float)
    load=np.asarray(load_forecast_kw,float)*dt
    pv=np.asarray(pv_forecast_kw,float)*dt
    price=np.asarray(price,float)
    T=len(A); pmax=pmax_kw*dt
    term=_terminal_value(price,eta_c)

    fixed=float(np.dot(price,A)+np.dot(0.5*price,np.abs(A-P)))

    # emergency, charge, discharge, pv_spill, unused_contract, soc, z
    n=6*T+1
    ie=0; ic=T; idis=2*T; iw=3*T; iu=4*T; isoc=5*T; iz=6*T
    c=np.zeros(n)
    c[ie:ie+T]=emergency_multiplier*price
    c[ic:ic+T]+=1e-8
    c[idis:idis+T]+=1e-8
    c[iw:iw+T]+=1e-10
    c[iu:iu+T]+=1e-10
    c[iz]=term

    Aeq=[]; beq=[]; Aub=[]; bub=[]

    # A + E + D + PV - W - U = L + C
    for t in range(T):
        row=np.zeros(n)
        row[ie+t]=1
        row[idis+t]=1
        row[iw+t]=-1
        row[iu+t]=-1
        row[ic+t]=-1
        Aeq.append(row)
        beq.append(load[t]-pv[t]-A[t])

    for t in range(T):
        row=np.zeros(n)
        row[isoc+t]=1; row[ic+t]=-eta_c; row[idis+t]=1/eta_d
        if t==0:
            rhs=float(soc0)
        else:
            row[isoc+t-1]=-1; rhs=0.0
        Aeq.append(row); beq.append(rhs)

    row=np.zeros(n); row[isoc+T-1]=-1; row[iz]=-1
    Aub.append(row); bub.append(-terminal_ref)

    bounds=[]
    bounds += [(0,None)]*T
    bounds += [(0,pmax)]*T
    bounds += [(0,pmax)]*T
    bounds += [(0,float(v)) for v in pv]
    bounds += [(0,None)]*T
    bounds += [(soc_min,soc_max)]*T
    bounds += [(0,None)]

    res=linprog(c,A_ub=np.asarray(Aub),b_ub=np.asarray(bub),
                A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),
                bounds=bounds,method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    return fixed+float(res.fun)


def execute_segment_milp(actual_load_kw, actual_pv_kw, purchase_kwh, price, soc0, *,
                         dt=1/6, soc_min=1200, soc_max=10800, pmax_kw=5000,
                         eta_c=0.9, eta_d=0.9, emergency_multiplier=5.0,
                         terminal_ref=6000.0, terminal_value=0.0,
                         force_terminal=False):
    """
    Realized operation.

    Exact accounting:
       planned grid + emergency + discharge + actual PV
       = load + charge + PV curtailment + unused contracted grid.

    'unused contracted grid' is a diagnostic recourse caused by forecast error.
    It is never a decision variable in the candidate adjustment optimizer.
    """
    load=np.asarray(actual_load_kw,float)*dt
    pv=np.asarray(actual_pv_kw,float)*dt
    A=np.asarray(purchase_kwh,float)
    price=np.asarray(price,float)
    T=len(A); pmax=pmax_kw*dt

    # emergency, charge, discharge, pv_spill, unused, soc, binary, z
    n=7*T+1
    ie=0; ic=T; idis=2*T; iw=3*T; iu=4*T; isoc=5*T; ib=6*T; iz=7*T

    c=np.zeros(n)
    c[ie:ie+T]=emergency_multiplier*price
    c[ic:ic+T]+=1e-10
    c[idis:idis+T]+=1e-10
    # Lexicographic tie-break only (not part of settlement):
    # use storage first, then curtail PV, and leave paid grid unused only last.
    c[iw:iw+T]+=1e-7
    c[iu:iu+T]+=1e-5
    if not force_terminal:
        c[iz]=float(terminal_value)

    rows=[]; lb=[]; ub=[]

    # A + E + D + PV - W - U = L + C
    for t in range(T):
        rhs=load[t]-pv[t]-A[t]
        rows.append({ie+t:1, idis+t:1, iw+t:-1, iu+t:-1, ic+t:-1})
        lb.append(rhs); ub.append(rhs)

    for t in range(T):
        r={isoc+t:1, ic+t:-eta_c, idis+t:1/eta_d}
        rhs=float(soc0) if t==0 else 0.0
        if t>0:
            r[isoc+t-1]=-1
        rows.append(r); lb.append(rhs); ub.append(rhs)

    for t in range(T):
        rows.append({ic+t:1, ib+t:-pmax})
        lb.append(-np.inf); ub.append(0.0)
        rows.append({idis+t:1, ib+t:pmax})
        lb.append(-np.inf); ub.append(pmax)

    if force_terminal:
        rows.append({isoc+T-1:1})
        lb.append(terminal_ref); ub.append(terminal_ref)
    elif terminal_value>0:
        rows.append({isoc+T-1:-1, iz:-1})
        lb.append(-np.inf); ub.append(-terminal_ref)

    M=lil_matrix((len(rows),n),dtype=float)
    for i,r in enumerate(rows):
        for j,v in r.items():
            M[i,j]=v

    lower=np.zeros(n)
    upper=np.full(n,np.inf)
    upper[ic:ic+T]=pmax
    upper[idis:idis+T]=pmax
    upper[iw:iw+T]=pv
    lower[isoc:isoc+T]=soc_min
    upper[isoc:isoc+T]=soc_max
    upper[ib:ib+T]=1.0
    integrality=np.zeros(n,dtype=int)
    integrality[ib:ib+T]=1

    res=milp(c,integrality=integrality,
             bounds=Bounds(lower,upper),
             constraints=LinearConstraint(M.tocsr(),
                                          np.asarray(lb,float),
                                          np.asarray(ub,float)),
             options={'time_limit':30,'disp':False})
    if not res.success:
        raise RuntimeError(f'Execution MILP failed: {res.message}')

    x=res.x
    em=x[ie:ie+T]; ch=x[ic:ic+T]; dis=x[idis:idis+T]
    spill=x[iw:iw+T]; unused=x[iu:iu+T]; soc=x[isoc:isoc+T]

    residual=A+em+dis+pv-(load+ch+spill+unused)
    return {
        'emergency':em,
        'charge':ch,
        'discharge':dis,
        'pv_spill':spill,
        'unused_contract':unused,
        'soc':soc,
        'soc_end':float(soc[-1]),
        'balance_residual':residual,
        'simultaneous_count':int(np.sum((ch>1e-7)&(dis>1e-7)))
    }


def settlement_cost(original_plan, final_plan, price, emergency):
    P=np.asarray(original_plan,float)
    A=np.asarray(final_plan,float)
    p=np.asarray(price,float)
    E=np.asarray(emergency,float)
    down=np.maximum(P-A,0)
    up=np.maximum(A-P,0)
    normal=float(np.dot(p,A))
    adjust=float(np.dot(0.5*p,down+up))
    emerg=float(np.dot(5*p,E))
    literal=float(np.dot(p,P)+np.dot(0.5*p,down)+np.dot(1.5*p,up)+emerg)
    return {
        'adjusted_purchase_cost':normal,
        'adjustment_fee':adjust,
        'emergency_cost':emerg,
        'total_cost':normal+adjust+emerg,
        'literal_total_cost':literal,
        'up_kwh':float(up.sum()),
        'down_kwh':float(down.sum())
    }
