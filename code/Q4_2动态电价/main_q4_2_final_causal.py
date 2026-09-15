from __future__ import annotations
import argparse,json,time
import numpy as np
import pandas as pd
from src.q4_config import *
from src.data_loader import load_actual_matrix
from src.forecast import build_rolling_sets,target_features,fit_quantile_models
from src.forecast_v4 import build_recent_calibration,adaptive_block_qhat,calibration_coverage,make_v4_scenarios
from src.price_forecast_q4 import load_dynamic_price,price_scenarios,metrics
from src.q4_stochastic_lp import solve_two_stage_joint,solve_point_plan
from src.causal_execution import causal_execute_plan
from src.terminal_normalization import terminal_soc_equivalent_cost
from src.q4_export import export_result42,validate_result42

def build_model_pack(net,dates,month_start,month):
    fit_pack,_=build_rolling_sets(net,dates,month_start,TRAIN_WINDOW_DAYS,CALIBRATION_DAYS)
    Xf,yf,fd,fs,med=fit_pack
    models=fit_quantile_models(Xf,yf,fd,month_start,RECENCY_HALF_LIFE_DAYS,RANDOM_SEED+month)
    return models,fit_pack

def forecast_net(day_idx,dates,net,pack,seed):
    models,fit_pack=pack
    Xf,yf,fd,fs,med=fit_pack
    Xc,yc,cd,cs=build_recent_calibration(net,dates,day_idx,med,ADAPTIVE_CALIBRATION_DAYS)
    qhat_slot,qhat_blocks=adaptive_block_qhat(models,Xc,yc,cs,target_coverage=TARGET_COVERAGE,n_blocks=ADAPTIVE_TIME_BLOCKS)
    cal_cov,cal_mae=calibration_coverage(models,Xc,yc,cs,qhat_slot)
    Xt=target_features(net,dates,day_idx,med)
    q10,q50,q90,scenarios=make_v4_scenarios(models,Xc,yc,cd,cs,Xt,qhat_slot,n_scenarios=N_SCENARIOS,seed=seed)
    actual=net[day_idx]
    return {'q10':q10,'q50':q50,'q90':q90,'scenarios':scenarios,'actual':actual,
            'coverage':float(np.mean((actual>=q10)&(actual<=q90))),
            'mae':float(np.mean(np.abs(q50-actual))), 'cal_cov':cal_cov}

def run_one(di,dates,load,pv,net,price,pack,soc0,kind,is_last=False):
    fc=forecast_net(di,dates,net,pack,RANDOM_SEED+di)
    pp,ps=price_scenarios(price,dates,di,N_SCENARIOS,RANDOM_SEED+10000+di)
    if kind=='stochastic':
        grid,_=solve_two_stage_joint(fc['scenarios'],ps,soc0=soc0,enforce_terminal=False)
    elif kind=='point':
        grid,_=solve_point_plan(fc['q50'],pp,soc0=soc0,enforce_terminal=False)
    elif kind=='perfect':
        grid,_=solve_point_plan(fc['actual'],price[di],soc0=soc0,enforce_terminal=False)
    else: raise ValueError(kind)
    ex=causal_execute_plan(
        load[di],pv[di],grid,price[di],soc0=soc0,
        dt=DT_HOURS,soc_min=SOC_MIN_KWH,soc_max=SOC_MAX_KWH,
        pmax_kw=P_CHARGE_MAX_KW,eta_c=ETA_C,eta_d=ETA_D,
        emergency_multiplier=EMERGENCY_MULTIPLIER
    )
    return fc,pp,grid,ex

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--full',action='store_true')
    args=ap.parse_args()
    out=ROOT/'outputs_q4_2_final_causal';out.mkdir(exist_ok=True)

    dates,load,pv,net=load_actual_matrix(ATTACHMENT2);dates=pd.DatetimeIndex(dates)
    pdates,price=load_dynamic_price(ATTACHMENT4)
    if not dates.equals(pdates): raise ValueError('附件2与附件4日期不一致')

    start=int(np.where(dates==pd.Timestamp('2025-02-01'))[0][0])
    if args.full:
        targets=list(range(start,len(dates)))
    else:
        targets=[int(np.where(dates==pd.Timestamp(d))[0][0]) for d in REPRESENTATIVE_DATES]

    state={k:SOC_INITIAL_KWH for k in ('stochastic','point','perfect')}
    cache={};rows=[];details=[];chosen=[];t0=time.time()

    for j,di in enumerate(targets,1):
        d=dates[di];key=(d.year,d.month)
        if key not in cache:
            mi=int(np.where(dates==pd.Timestamp(d.year,d.month,1))[0][0])
            cache[key]=build_model_pack(net,dates,mi,d.month)
            print(f'[源荷模型] {d.year}-{d.month:02d}，历史截止 {dates[mi-1].date()}')

        day={}
        for kind in ('stochastic','point','perfect'):
            socin=state[kind] if args.full else SOC_INITIAL_KWH
            fc,pp,g,ex=run_one(di,dates,load,pv,net,price,cache[key],socin,kind,args.full and di==targets[-1])
            day[kind]=(fc,pp,g,ex)
            if args.full: state[kind]=ex['soc_end_kwh']

        fc,pp,g,ex=day['stochastic']
        _,_,gp,ep=day['point'];_,_,gpi,epi=day['perfect']
        pm=metrics(pp,price[di])

        rows.append({
            '日期':str(d.date()),
            '源荷Q10-Q90覆盖率':fc['coverage'],
            '源荷预测MAE_kW':fc['mae'],
            '电价预测MAE_元每kWh':pm['mae'],
            '电价预测RMSE_元每kWh':pm['rmse'],
            '随机_SOC起点_kWh':ex['soc_start_kwh'],
            '随机_SOC终点_kWh':ex['soc_end_kwh'],
            '随机_计划购电量_kWh':ex['total_grid_kwh'],
            '随机_计划购电费_元':ex['plan_cost_yuan'],
            '随机_应急购电量_kWh':ex['total_emergency_kwh'],
            '随机_应急购电费_元':ex['emergency_cost_yuan'],
            '随机_总成本_元':ex['total_cost_yuan'],
            '点预测_总成本_元':ep['total_cost_yuan'],
            '完全信息_总成本_元':epi['total_cost_yuan'],
            '相对点预测节省_元':ep['total_cost_yuan']-ex['total_cost_yuan'],
            '距完全信息下界_元':ex['total_cost_yuan']-epi['total_cost_yuan'],
            '最大供需残差_kWh':ex['max_abs_balance_residual_kwh'],
            '同时充放电时段数':ex['simultaneous_charge_discharge_periods']
        })
        for t in range(144):
            details.append({
                '日期':str(d.date()),'时段序号':t+1,
                '实际净负荷_kW':fc['actual'][t],'Q50预测净负荷_kW':fc['q50'][t],
                '预测电价_元每kWh':pp[t],'实际电价_元每kWh':price[di,t],
                '计划购电_kWh':g[t],'紧急购电_kWh':ex['emergency_kwh'][t],
                '充电_kWh':ex['charge_kwh'][t],'放电_kWh':ex['discharge_kwh'][t],
                'SOC_kWh':ex['soc_kwh'][t]
            })
        chosen.append({'date':str(d.date()),'day_idx':di,'grid':g,'emergency':ex['emergency_kwh'],
                       'charge':ex['charge_kwh'],'discharge':ex['discharge_kwh'],
                       'soc_start':ex['soc_start_kwh'],'soc_end':ex['soc_end_kwh']})
        print(f"[{j}/{len(targets)}] {d.date()} 成本={ex['total_cost_yuan']:.2f} "
              f"购电={ex['total_grid_kwh']:.1f} 应急={ex['total_emergency_kwh']:.1f} "
              f"价格MAE={pm['mae']:.4f} SOC={ex['soc_end_kwh']:.1f}")

    s=pd.DataFrame(rows);d=pd.DataFrame(details);mode='full' if args.full else 'quick4'
    with pd.ExcelWriter(out/f'q4_2_final_causal_results_{mode}.xlsx',engine='openpyxl') as w:
        s.to_excel(w,index=False,sheet_name='汇总');d.to_excel(w,index=False,sheet_name='时段明细')

    raw_total=float(s['随机_总成本_元'].sum())
    point_total=float(s['点预测_总成本_元'].sum())
    final_soc=float(s['随机_SOC终点_kWh'].iloc[-1])
    # Fair finite-horizon comparison only: value end SOC at the next-slot causal price reference.
    terminal_price=float(price[-1,-1])
    terminal_adj=terminal_soc_equivalent_cost(
        final_soc,reference_soc_kwh=SOC_INITIAL_KWH,
        settlement_price_yuan_per_kwh=terminal_price,eta_c=ETA_C,eta_d=ETA_D
    )
    stats={
        'mode':mode,'days':len(s),
        'mean_cost_yuan':float(s['随机_总成本_元'].mean()),
        'total_cost_yuan':raw_total,
        'terminal_soc_equivalent_adjustment_yuan':float(terminal_adj),
        'terminal_normalized_cost_yuan':float(raw_total+terminal_adj),
        'terminal_normalization_reference_soc_kwh':float(SOC_INITIAL_KWH),
        'terminal_normalization_price_yuan_per_kwh':terminal_price,
        'point_total_cost_yuan':point_total,
        'total_saving_vs_point_yuan':float(point_total-raw_total),
        'total_plan_purchase_kwh':float(s['随机_计划购电量_kWh'].sum()),
        'total_emergency_kwh':float(s['随机_应急购电量_kWh'].sum()),
        'total_emergency_cost_yuan':float(s['随机_应急购电费_元'].sum()),
        'total_external_purchase_kwh':float(s['随机_计划购电量_kWh'].sum()+s['随机_应急购电量_kWh'].sum()),
        'mean_price_mae_yuan_per_kwh':float(s['电价预测MAE_元每kWh'].mean()),
        'mean_netload_mae_kw':float(s['源荷预测MAE_kW'].mean()),
        'mean_saving_vs_point_yuan':float(s['相对点预测节省_元'].mean()),
        'final_soc_kwh':final_soc,
        'max_balance_residual_kwh':float(s['最大供需残差_kWh'].max()),
        'simultaneous_total':int(s['同时充放电时段数'].sum()),
        'runtime_seconds':time.time()-t0
    }
    (out/f'q4_2_final_causal_summary_{mode}.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')

    if args.full:
        export_result42(RESULT42_TEMPLATE,out/'result4-2.xlsx',chosen,price)
        validate_result42(out/'result4-2.xlsx',chosen,price)
        print('[表格终检] result4-2.xlsx通过。')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
