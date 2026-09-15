from __future__ import annotations
import argparse,json,time
import numpy as np
import pandas as pd
from src.config import *
from src.data_loader import load_price,load_actual_matrix
from src.forecast import build_rolling_sets,target_features,fit_quantile_models
from src.forecast_v4 import build_recent_calibration,adaptive_block_qhat,calibration_coverage,make_v4_scenarios
from src.stochastic_lp_v3 import solve_two_stage_continuous,solve_deterministic_plan_continuous
from src.causal_execution import causal_execute_plan
from src.terminal_normalization import terminal_soc_equivalent_cost

OUT=ROOT/'outputs_q2_final_causal'
REP=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']

def build_model_pack(net,dates,month_start,month):
    fit_pack,_=build_rolling_sets(net,dates,month_start,TRAIN_WINDOW_DAYS,CALIBRATION_DAYS)
    Xf,yf,fd,fs,med=fit_pack
    models=fit_quantile_models(Xf,yf,fd,month_start,RECENCY_HALF_LIFE_DAYS,RANDOM_SEED+month)
    return models,fit_pack

def forecast_day(day_idx,dates,net,pack,seed):
    models,fit_pack=pack
    Xf,yf,fd,fs,med=fit_pack
    Xc,yc,cd,cs=build_recent_calibration(net,dates,day_idx,med,ADAPTIVE_CALIBRATION_DAYS)
    qhat,_=adaptive_block_qhat(models,Xc,yc,cs,target_coverage=TARGET_COVERAGE,n_blocks=ADAPTIVE_TIME_BLOCKS)
    cal_cov,_=calibration_coverage(models,Xc,yc,cs,qhat)
    Xt=target_features(net,dates,day_idx,med)
    q10,q50,q90,scenarios=make_v4_scenarios(models,Xc,yc,cd,cs,Xt,qhat,n_scenarios=N_SCENARIOS,seed=seed)
    actual=net[day_idx]
    return {'q10':q10,'q50':q50,'q90':q90,'scenarios':scenarios,'actual':actual,
            'coverage':float(np.mean((actual>=q10)&(actual<=q90))),
            'mae':float(np.mean(np.abs(q50-actual))),'cal_cov':cal_cov}

def make_plan(kind,fc,price,soc0):
    common=dict(soc0=soc0,terminal_ref_kwh=SOC_INITIAL_KWH,enforce_terminal=False)
    if kind=='stochastic': return solve_two_stage_continuous(fc['scenarios'],price,**common)[0]
    if kind=='point': return solve_deterministic_plan_continuous(fc['q50'],price,**common)[0]
    raise ValueError(kind)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    price=load_price(ATTACHMENT1)
    dates,load,pv,net=load_actual_matrix(ATTACHMENT2);dates=pd.DatetimeIndex(dates)
    start=int(np.where(dates==pd.Timestamp('2025-02-01'))[0][0])
    targets=list(range(start,len(dates))) if args.full else [int(np.where(dates==pd.Timestamp(d))[0][0]) for d in REP]
    states={'stochastic':SOC_INITIAL_KWH,'point':SOC_INITIAL_KWH}
    cache={};rows=[];details=[];t0=time.time()
    for j,di in enumerate(targets,1):
        d=dates[di];key=(d.year,d.month)
        if key not in cache:
            ms=int(np.where(dates==pd.Timestamp(d.year,d.month,1))[0][0])
            cache[key]=build_model_pack(net,dates,ms,d.month)
            print(f'[模型更新] {d.year}-{d.month:02d}')
        fc=forecast_day(di,dates,net,cache[key],RANDOM_SEED+di)
        day={}
        for kind in ('stochastic','point'):
            soc0=states[kind] if args.full else SOC_INITIAL_KWH
            grid=make_plan(kind,fc,price,soc0)
            ex=causal_execute_plan(load[di],pv[di],grid,price,soc0=soc0)
            day[kind]=(grid,ex)
            if args.full: states[kind]=ex['soc_end_kwh']
        gs,es=day['stochastic'];gp,ep=day['point']
        rows.append({
            '日期':str(d.date()),'覆盖率':fc['coverage'],'预测MAE_kW':fc['mae'],
            '随机_SOC起点_kWh':es['soc_start_kwh'],'随机_SOC终点_kWh':es['soc_end_kwh'],
            '随机_计划购电量_kWh':es['total_grid_kwh'],'随机_计划购电费_元':es['plan_cost_yuan'],
            '随机_紧急购电量_kWh':es['total_emergency_kwh'],'随机_紧急购电费_元':es['emergency_cost_yuan'],
            '随机_总成本_元':es['total_cost_yuan'],'随机_外部购电总量_kWh':es['total_external_purchase_kwh'],
            '随机_弃光_kWh':es['total_pv_spill_kwh'],'随机_未利用计划购电_kWh':es['total_unused_contract_kwh'],
            '点预测_总成本_元':ep['total_cost_yuan'],'相对点预测节省_元':ep['total_cost_yuan']-es['total_cost_yuan'],
            '最大平衡残差_kWh':es['max_abs_balance_residual_kwh'],'同时充放电':es['simultaneous_charge_discharge_periods']})
        for t in range(144):
            details.append({'日期':str(d.date()),'时段序号':t+1,'实际负荷_kW':load[di,t],'实际光伏_kW':pv[di,t],
                            '计划购电_kWh':gs[t],'紧急购电_kWh':es['emergency_kwh'][t],
                            '充电_kWh':es['charge_kwh'][t],'放电_kWh':es['discharge_kwh'][t],
                            'SOC_kWh':es['soc_kwh'][t],'未利用计划购电_kWh':es['unused_contract_kwh'][t]})
        print(f"[{j}/{len(targets)}] {d.date()} 随机={es['total_cost_yuan']:.2f} 点预测={ep['total_cost_yuan']:.2f} 应急={es['total_emergency_kwh']:.1f} SOC={es['soc_end_kwh']:.1f}")
    s=pd.DataFrame(rows); det=pd.DataFrame(details); mode='full' if args.full else 'quick4'
    with pd.ExcelWriter(OUT/f'q2_final_causal_results_{mode}.xlsx',engine='openpyxl') as w:
        s.to_excel(w,index=False,sheet_name='汇总');det.to_excel(w,index=False,sheet_name='时段明细')
    raw_total=float(s['随机_总成本_元'].sum())
    raw_point=float(s['点预测_总成本_元'].sum())
    final_soc=float(s['随机_SOC终点_kWh'].iloc[-1])
    terminal_price=float(price[0])
    terminal_adj=terminal_soc_equivalent_cost(final_soc,reference_soc_kwh=SOC_INITIAL_KWH,settlement_price_yuan_per_kwh=terminal_price,eta_c=ETA_C,eta_d=ETA_D)
    stats={'mode':mode,'days':len(s),'total_cost_yuan':raw_total,
           'terminal_soc_equivalent_adjustment_yuan':float(terminal_adj),
           'terminal_normalized_cost_yuan':float(raw_total+terminal_adj),
           'terminal_normalization_reference_soc_kwh':float(SOC_INITIAL_KWH),
           'terminal_normalization_price_yuan_per_kwh':terminal_price,
           'point_total_cost_yuan':raw_point,
           'total_saving_vs_point_yuan':float(s['相对点预测节省_元'].sum()),
           'total_external_purchase_kwh':float(s['随机_外部购电总量_kWh'].sum()),
           'total_emergency_kwh':float(s['随机_紧急购电量_kWh'].sum()),
           'total_emergency_cost_yuan':float(s['随机_紧急购电费_元'].sum()),
           'total_unused_contract_kwh':float(s['随机_未利用计划购电_kWh'].sum()),
           'mean_mae_kw':float(s['预测MAE_kW'].mean()),'final_soc_kwh':final_soc,
           'max_balance_residual_kwh':float(s['最大平衡残差_kWh'].max()),'simultaneous_total':int(s['同时充放电'].sum()),
           'runtime_seconds':time.time()-t0}
    (OUT/f'q2_final_causal_summary_{mode}.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
