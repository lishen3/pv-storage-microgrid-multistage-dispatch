from __future__ import annotations
import argparse,json,time
import numpy as np
import pandas as pd
from src.config import *
from src.data_loader import load_price,load_actual_matrix
from src.forecast import build_rolling_sets,target_features,fit_quantile_models
from src.forecast_v4 import build_recent_calibration,adaptive_block_qhat,make_v4_scenarios
from src.stochastic_lp_v3 import solve_two_stage_continuous
from src.q3_data_fixed import load_pv_forecasts,forecast_for_remaining_day
from src.q3_forecast import build_load_model,load_forecast_at_00
from src.q3_opt_final2 import solve_deterministic_adjustment,deterministic_keep_cost,settlement_cost
from src.causal_execution import causal_execute_segment
from src.terminal_normalization import terminal_soc_equivalent_cost

ATTACHMENT3=DATA/'attachment3.xlsx'
OUT=ROOT/'outputs_q3_final_causal'
REP=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
SEGMENTS=[(0,36,0),(36,72,6),(72,108,12),(108,144,18)]
THRESHOLD=1.0

def build_q2_pack(net,dates,month_start,month):
    fit_pack,_=build_rolling_sets(net,dates,month_start,TRAIN_WINDOW_DAYS,CALIBRATION_DAYS)
    Xf,yf,fd,fs,med=fit_pack
    models=fit_quantile_models(Xf,yf,fd,month_start,RECENCY_HALF_LIFE_DAYS,RANDOM_SEED+month)
    return models,fit_pack

def q2_scenarios(net,dates,di,pack):
    models,fit_pack=pack; Xf,yf,fd,fs,med=fit_pack
    Xc,yc,cd,cs=build_recent_calibration(net,dates,di,med,ADAPTIVE_CALIBRATION_DAYS)
    qhat,_=adaptive_block_qhat(models,Xc,yc,cs,target_coverage=TARGET_COVERAGE,n_blocks=ADAPTIVE_TIME_BLOCKS)
    Xt=target_features(net,dates,di,med)
    q10,q50,q90,sc=make_v4_scenarios(models,Xc,yc,cd,cs,Xt,qhat,n_scenarios=N_SCENARIOS,seed=RANDOM_SEED+di)
    return q50,sc

def q2_plan(net,dates,di,pack,price,soc0):
    q50,sc=q2_scenarios(net,dates,di,pack)
    grid,_=solve_two_stage_continuous(sc,price,soc0=soc0,terminal_ref_kwh=SOC_INITIAL_KWH,enforce_terminal=False)
    return grid,q50

def run_day(di,dates,load,pv,net,price,pv_fc,q2pack,loadpack,soc_start,strategy):
    original,_=q2_plan(net,dates,di,q2pack,price,soc_start)
    load_q50,_,_=load_forecast_at_00(load,dates,di,loadpack)
    current=original.copy();final=original.copy();soc=float(soc_start)
    em=np.zeros(144);ch=np.zeros(144);dis=np.zeros(144);spill=np.zeros(144);unused=np.zeros(144);socs=np.zeros(144);resid=np.zeros(144)
    updates=[]
    for a,b,issue in SEGMENTS:
        if issue>0:
            # Latest PV forecast affects only not-yet-executed interval [a,144).
            pv_pt=forecast_for_remaining_day(pv_fc,dates,pv,di,issue)
            load_pt=load_q50[a:]
            P0=original[a:]; keep=current[a:]
            keep_obj=deterministic_keep_cost(keep,P0,load_pt,pv_pt,price[a:],soc)
            cand,meta=solve_deterministic_adjustment(P0,load_pt,pv_pt,price[a:],soc,force_terminal=False)
            gain=keep_obj-meta['objective']
            do=(strategy=='always') or (strategy=='threshold' and gain>THRESHOLD)
            if strategy=='never':do=False
            if do: current[a:]=cand;final[a:]=cand
            updates.append({'时刻':f'{issue:02d}:00','预计净收益':float(gain),'是否调整':bool(do),
                            '调整增加量_kWh':float(np.maximum(cand-P0,0).sum()),'调整减少量_kWh':float(np.maximum(P0-cand,0).sum())})
        ex=causal_execute_segment(load[di,a:b],pv[di,a:b],current[a:b],price[a:b],soc0=soc)
        em[a:b]=ex['emergency_kwh'];ch[a:b]=ex['charge_kwh'];dis[a:b]=ex['discharge_kwh'];spill[a:b]=ex['pv_spill_kwh'];unused[a:b]=ex['unused_contract_kwh'];socs[a:b]=ex['soc_kwh'];resid[a:b]=ex['balance_residual_kwh'];soc=ex['soc_end_kwh']
    cost=settlement_cost(original,final,price,em)
    return {'date':str(dates[di].date()),'soc_start':float(soc_start),'soc_end':float(soc),'original_plan':original,'final_plan':final,
            'emergency':em,'charge':ch,'discharge':dis,'spill':spill,'unused':unused,'soc':socs,'resid':resid,'updates':updates,'cost':cost}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    price=load_price(ATTACHMENT1)
    dates,load,pv,net=load_actual_matrix(ATTACHMENT2);dates=pd.DatetimeIndex(dates)
    pv_fc=load_pv_forecasts(ATTACHMENT3)
    start=int(np.where(dates==pd.Timestamp('2025-02-01'))[0][0])
    targets=list(range(start,len(dates))) if args.full else [int(np.where(dates==pd.Timestamp(d))[0][0]) for d in REP]
    states={s:SOC_INITIAL_KWH for s in ('never','always','threshold')};q2cache={};loadcache={};rows=[];ups=[];detail=[];t0=time.time()
    for j,di in enumerate(targets,1):
        d=dates[di];key=(d.year,d.month)
        if key not in q2cache:
            ms=int(np.where(dates==pd.Timestamp(d.year,d.month,1))[0][0])
            q2cache[key]=build_q2_pack(net,dates,ms,d.month)
            loadcache[key]=build_load_model(load,dates,ms,d.month,TRAIN_WINDOW_DAYS,CALIBRATION_DAYS,RECENCY_HALF_LIFE_DAYS,RANDOM_SEED)
            print(f'[模型更新] {d.year}-{d.month:02d}')
        rr={}
        for strategy in ('never','always','threshold'):
            soc0=states[strategy] if args.full else SOC_INITIAL_KWH
            r=run_day(di,dates,load,pv,net,price,pv_fc,q2cache[key],loadcache[key],soc0,strategy)
            rr[strategy]=r
            if args.full:states[strategy]=r['soc_end']
        r=rr['threshold'];never=rr['never'];always=rr['always']
        # critical audit: never uses exactly the same original plan + causal executor family as Q2
        rows.append({'日期':r['date'],'阈值策略总成本_元':r['cost']['total_cost'],'全调整总成本_元':always['cost']['total_cost'],'不调整总成本_元':never['cost']['total_cost'],
                     '相对不调整节省_元':never['cost']['total_cost']-r['cost']['total_cost'],'正常调整后购电费_元':r['cost']['adjusted_purchase_cost'],
                     '调整费用_元':r['cost']['adjustment_fee'],'紧急购电费_元':r['cost']['emergency_cost'],'0时计划购电量_kWh':float(r['original_plan'].sum()),
                     '最终正常购电量_kWh':float(r['final_plan'].sum()),'紧急购电量_kWh':float(r['emergency'].sum()),'外部购电总量_kWh':float(r['final_plan'].sum()+r['emergency'].sum()),
                     '弃光_kWh':float(r['spill'].sum()),'未利用计划购电_kWh':float(r['unused'].sum()),'SOC起点_kWh':r['soc_start'],'SOC终点_kWh':r['soc_end'],
                     '触发调整次数':sum(u['是否调整'] for u in r['updates']),'最大平衡残差_kWh':float(np.max(np.abs(r['resid']))),
                     '同时充放电':int(np.sum((r['charge']>1e-7)&(r['discharge']>1e-7)))})
        for u in r['updates']:ups.append({'日期':r['date'],**u})
        for t in range(144):detail.append({'日期':r['date'],'时段序号':t+1,'0时计划购电_kWh':r['original_plan'][t],'最终调整购电_kWh':r['final_plan'][t],
                                            '紧急购电_kWh':r['emergency'][t],'充电_kWh':r['charge'][t],'放电_kWh':r['discharge'][t],'SOC_kWh':r['soc'][t],'未利用计划购电_kWh':r['unused'][t]})
        print(f"[{j}/{len(targets)}] {r['date']} 阈值={r['cost']['total_cost']:.2f} 不调整={never['cost']['total_cost']:.2f} 节省={never['cost']['total_cost']-r['cost']['total_cost']:.2f} 应急={r['emergency'].sum():.1f}")
    s=pd.DataFrame(rows);u=pd.DataFrame(ups);de=pd.DataFrame(detail);mode='full' if args.full else 'quick4'
    with pd.ExcelWriter(OUT/f'q3_final_causal_results_{mode}.xlsx',engine='openpyxl') as w:
        s.to_excel(w,index=False,sheet_name='策略汇总');u.to_excel(w,index=False,sheet_name='更新价值');de.to_excel(w,index=False,sheet_name='时段明细')
    raw_total=float(s['阈值策略总成本_元'].sum()); raw_never=float(s['不调整总成本_元'].sum())
    final_soc=float(s['SOC终点_kWh'].iloc[-1]); never_final_soc=float(states['never']) if args.full else float(rr['never']['soc_end'])
    terminal_price=float(price[0])
    threshold_terminal_adj=terminal_soc_equivalent_cost(final_soc,reference_soc_kwh=SOC_INITIAL_KWH,settlement_price_yuan_per_kwh=terminal_price,eta_c=ETA_C,eta_d=ETA_D)
    never_terminal_adj=terminal_soc_equivalent_cost(never_final_soc,reference_soc_kwh=SOC_INITIAL_KWH,settlement_price_yuan_per_kwh=terminal_price,eta_c=ETA_C,eta_d=ETA_D)
    normalized_total=raw_total+threshold_terminal_adj; normalized_never=raw_never+never_terminal_adj
    stats={'mode':mode,'days':len(s),'total_cost_yuan':raw_total,'never_total_cost_yuan':raw_never,
           'total_saving_vs_never_yuan':float(raw_never-raw_total),
           'terminal_soc_equivalent_adjustment_yuan':float(threshold_terminal_adj),
           'never_terminal_soc_equivalent_adjustment_yuan':float(never_terminal_adj),
           'terminal_normalized_cost_yuan':float(normalized_total),
           'never_terminal_normalized_cost_yuan':float(normalized_never),
           'terminal_normalized_saving_vs_never_yuan':float(normalized_never-normalized_total),
           'terminal_normalization_reference_soc_kwh':float(SOC_INITIAL_KWH),
           'terminal_normalization_price_yuan_per_kwh':terminal_price,
           'total_external_purchase_kwh':float(s['外部购电总量_kWh'].sum()),
           'total_emergency_kwh':float(s['紧急购电量_kWh'].sum()),'total_emergency_cost_yuan':float(s['紧急购电费_元'].sum()),
           'total_adjustment_triggers':int(s['触发调整次数'].sum()),'total_unused_contract_kwh':float(s['未利用计划购电_kWh'].sum()),
           'final_soc_kwh':final_soc,'never_final_soc_kwh':never_final_soc,
           'max_balance_residual_kwh':float(s['最大平衡残差_kWh'].max()),'simultaneous_total':int(s['同时充放电'].sum()),
           'runtime_seconds':time.time()-t0}
    (OUT/f'q3_final_causal_summary_{mode}.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
