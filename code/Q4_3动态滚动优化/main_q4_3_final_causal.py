from __future__ import annotations
import argparse,json,time
import numpy as np
import pandas as pd

from src.q4_config import *
from src.data_loader import load_actual_matrix
from src.forecast import build_rolling_sets,target_features,fit_quantile_models
from src.forecast_v4 import build_recent_calibration,adaptive_block_qhat,calibration_coverage,make_v4_scenarios
from src.q3_data_fixed import load_pv_forecasts,forecast_for_remaining_day
from src.q3_forecast import build_load_model,load_forecast_at_00
from src.price_forecast_q4 import load_dynamic_price,price_scenarios,point_price_forecast,metrics
from src.q4_stochastic_lp import solve_two_stage_joint
from src.q3_opt_final2 import (
    solve_deterministic_adjustment,deterministic_keep_cost,
    settlement_cost
)
from src.causal_execution import causal_execute_segment
from src.terminal_normalization import terminal_soc_equivalent_cost
from src.q4_3_export import export_result43,validate_result43_template,validate_result43_output

ATTACHMENT3=DATA/'attachment3.xlsx'
ATTACHMENT4=DATA/'attachment4.xlsx'
RESULT43_TEMPLATE=DATA/'result4-3_template.xlsx'
OUT=ROOT/'outputs_q4_3_final_causal'
REP_DATES=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']
SEGMENTS=[(0,36,0),(36,72,6),(72,108,12),(108,144,18)]
THRESHOLD=1.0

def build_q42_pack(net,dates,month_start,month):
    fit_pack,_=build_rolling_sets(net,dates,month_start,TRAIN_WINDOW_DAYS,CALIBRATION_DAYS)
    Xf,yf,fd,fs,med=fit_pack
    models=fit_quantile_models(Xf,yf,fd,month_start,RECENCY_HALF_LIFE_DAYS,RANDOM_SEED+month)
    return models,fit_pack

def q42_forecast(day_idx,dates,net,pack,seed):
    models,fit_pack=pack
    Xf,yf,fd,fs,med=fit_pack
    Xc,yc,cd,cs=build_recent_calibration(net,dates,day_idx,med,ADAPTIVE_CALIBRATION_DAYS)
    qhat,_=adaptive_block_qhat(
        models,Xc,yc,cs,target_coverage=TARGET_COVERAGE,n_blocks=ADAPTIVE_TIME_BLOCKS
    )
    Xt=target_features(net,dates,day_idx,med)
    q10,q50,q90,scenarios=make_v4_scenarios(
        models,Xc,yc,cd,cs,Xt,qhat,n_scenarios=N_SCENARIOS,seed=seed
    )
    return q50,scenarios

def q42_original_plan(di,dates,net,price,pack,soc0,final_day):
    q50,netS=q42_forecast(di,dates,net,pack,RANDOM_SEED+di)
    p0,pS=price_scenarios(price,dates,di,N_SCENARIOS,RANDOM_SEED+10000+di)
    grid,_=solve_two_stage_joint(
        netS,pS,soc0=soc0,enforce_terminal=False
    )
    return grid,p0

def updated_source_point(load_q50,pv_fc,dates,pv,di,issue):
    a=issue*6
    return load_q50[a:],forecast_for_remaining_day(pv_fc,dates,pv,di,issue)

def updated_price_point(price,dates,di,issue):
    base=point_price_forecast(price,dates,di)
    if issue==0: return base
    a=issue*6
    # At issue time, slots ending at or before issue are already observed.
    err=price[di,max(0,a-18):a]-base[max(0,a-18):a]
    if len(err):
        w=np.exp(np.linspace(-2.0,0.0,len(err))); w=w/w.sum()
        bias=float(np.dot(w,err))
    else: bias=0.0
    out=base.copy(); out[a:]=np.clip(out[a:]+bias,0.001,None)
    return out

def run_day(di,dates,load,pv,net,price,pv_fc,
            q42_pack,load_pack,soc_start,strategy='threshold',final_day=False):
    # 0:00 plan = Q4-2 V1 formal joint source-load/price stochastic plan.
    original,p0=q42_original_plan(di,dates,net,price,q42_pack,soc_start,final_day)
    load_q50,_,_=load_forecast_at_00(load,dates,di,load_pack)

    current=original.copy(); final=original.copy(); soc=float(soc_start)
    emergency=np.zeros(144); charge=np.zeros(144); discharge=np.zeros(144)
    pv_spill=np.zeros(144); unused=np.zeros(144)
    soc_curve=np.full(144,np.nan); residual=np.full(144,np.nan)
    simultaneous=0; updates=[]

    for a,b,issue in SEGMENTS:
        pforecast=updated_price_point(price,dates,di,issue)
        if issue>0:
            load_pt,pv_pt=updated_source_point(load_q50,pv_fc,dates,pv,di,issue)
            P0=original[a:]; keep=current[a:]; pf=pforecast[a:]
            keep_obj=deterministic_keep_cost(keep,P0,load_pt,pv_pt,pf,soc)
            cand,meta=solve_deterministic_adjustment(
                P0,load_pt,pv_pt,pf,soc,force_terminal=False
            )
            gain=keep_obj-meta['objective']
            do=(strategy=='always') or (strategy=='threshold' and gain>THRESHOLD)
            if strategy=='never': do=False
            if do:
                current[a:]=cand; final[a:]=cand
            updates.append({
                '时刻':f'{issue:02d}:00','调整起始时段':f'{issue:02d}:00-{issue:02d}:10',
                '预计保持成本':keep_obj,'预计调整成本':meta['objective'],
                '预计净收益':gain,'是否调整':bool(do),
                '价格预测MAE_剩余时域':float(np.mean(np.abs(pf-price[di,a:]))),
                '调整增加量_kWh':float(np.maximum(cand-P0,0).sum()),
                '调整减少量_kWh':float(np.maximum(P0-cand,0).sum())
            })

        # Strict causal realization inside the segment. Although we process a block
        # for convenience, causal_execute_segment loops slot-by-slot and never optimizes
        # against future actual load/PV/price inside that block.
        ex=causal_execute_segment(
            load[di,a:b],pv[di,a:b],current[a:b],price[di,a:b],soc0=soc,
            dt=DT_HOURS,soc_min=SOC_MIN_KWH,soc_max=SOC_MAX_KWH,
            pmax_kw=P_CHARGE_MAX_KW,eta_c=ETA_C,eta_d=ETA_D,
            emergency_multiplier=EMERGENCY_MULTIPLIER
        )
        emergency[a:b]=ex['emergency_kwh']; charge[a:b]=ex['charge_kwh']
        discharge[a:b]=ex['discharge_kwh']; pv_spill[a:b]=ex['pv_spill_kwh']
        unused[a:b]=ex['unused_contract_kwh']; soc_curve[a:b]=ex['soc_kwh']
        residual[a:b]=ex['balance_residual_kwh']; soc=ex['soc_end_kwh']
        simultaneous+=ex['simultaneous_charge_discharge_periods']

    cost=settlement_cost(original,final,price[di],emergency)
    return {
        'date':str(pd.Timestamp(dates[di]).date()),'day_idx':di,
        'soc_start':float(soc_start),'soc_end':float(soc),
        'original_plan':original,'final_plan':final,'emergency':emergency,
        'charge':charge,'discharge':discharge,'pv_spill':pv_spill,
        'unused_contract':unused,'soc_curve':soc_curve,'balance_residual':residual,
        'updates':updates,'cost':cost,'simultaneous_count':simultaneous,
        'price_mae_00':metrics(p0,price[di])['mae']
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--full',action='store_true'); args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    dates,load,pv,net=load_actual_matrix(ATTACHMENT2); dates=pd.DatetimeIndex(dates)
    pdates,price=load_dynamic_price(ATTACHMENT4)
    if not dates.equals(pdates): raise ValueError('附件2和附件4日期不一致')
    pv_fc=load_pv_forecasts(ATTACHMENT3)
    validate_result43_template(RESULT43_TEMPLATE)
    print('[表格预检] result4-3模板结构正确。')

    if args.full:
        start=int(np.where(dates==pd.Timestamp('2025-02-01'))[0][0]); target=list(range(start,len(dates)))
    else:
        target=[int(np.where(dates==pd.Timestamp(d))[0][0]) for d in REP_DATES]

    states={s:SOC_INITIAL_KWH for s in ('never','always','threshold')}
    q42_cache={}; load_cache={}; summary=[]; updates=[]; detail=[]; chosen=[]; t0=time.time()

    for j,di in enumerate(target,1):
        d=dates[di]; key=(d.year,d.month)
        if key not in q42_cache:
            ms=int(np.where(dates==pd.Timestamp(d.year,d.month,1))[0][0])
            q42_cache[key]=build_q42_pack(net,dates,ms,d.month)
            load_cache[key]=build_load_model(
                load,dates,ms,d.month,TRAIN_WINDOW_DAYS,CALIBRATION_DAYS,
                RECENCY_HALF_LIFE_DAYS,RANDOM_SEED
            )
            print(f'[模型更新] {d.year}-{d.month:02d} | 0时计划=Q4-2 V1')

        res={}
        for strategy in ('never','always','threshold'):
            socin=states[strategy] if args.full else SOC_INITIAL_KWH
            r=run_day(
                di,dates,load,pv,net,price,pv_fc,q42_cache[key],load_cache[key],
                socin,strategy,bool(args.full and di==target[-1])
            )
            res[strategy]=r
            if args.full: states[strategy]=r['soc_end']

        r=res['threshold']; chosen.append(r)
        summary.append({
            '日期':r['date'],'阈值策略总成本_元':r['cost']['total_cost'],
            '全调整总成本_元':res['always']['cost']['total_cost'],
            '不调整总成本_元':res['never']['cost']['total_cost'],
            '相对不调整节省_元':res['never']['cost']['total_cost']-r['cost']['total_cost'],
            '0时电价预测MAE_元每kWh':r['price_mae_00'],
            '正常调整后购电费_元':r['cost']['adjusted_purchase_cost'],
            '调整费用_元':r['cost']['adjustment_fee'],
            '紧急购电费_元':r['cost']['emergency_cost'],
            '0时计划购电量_kWh':float(r['original_plan'].sum()),
            '最终正常购电量_kWh':float(r['final_plan'].sum()),
            '紧急购电量_kWh':float(r['emergency'].sum()),
            '外部购电总量_kWh':float(r['final_plan'].sum()+r['emergency'].sum()),
            '实际弃光量_kWh':float(r['pv_spill'].sum()),
            '未利用计划购电量_kWh':float(r['unused_contract'].sum()),
            'SOC起点_kWh':r['soc_start'],'SOC终点_kWh':r['soc_end'],
            '触发调整次数':sum(u['是否调整'] for u in r['updates']),
            '同时充放电次数':r['simultaneous_count'],
            '最大能量平衡残差_kWh':float(np.max(np.abs(r['balance_residual'])))
        })
        for u in r['updates']: updates.append({'日期':r['date'],**u})
        for t in range(144):
            detail.append({
                '日期':r['date'],'时段序号':t+1,'实际电价_元每kWh':price[di,t],
                '0时计划购电_kWh':r['original_plan'][t],
                '最终调整购电_kWh':r['final_plan'][t],
                '紧急购电_kWh':r['emergency'][t],'充电_kWh':r['charge'][t],
                '放电_kWh':r['discharge'][t],'弃光_kWh':r['pv_spill'][t],
                '未利用计划购电_kWh':r['unused_contract'][t],'SOC_kWh':r['soc_curve'][t]
            })
        print(f"[{j}/{len(target)}] {r['date']} 成本={r['cost']['total_cost']:.2f} "
              f"不调整={res['never']['cost']['total_cost']:.2f} "
              f"应急={r['emergency'].sum():.1f} 价格MAE={r['price_mae_00']:.4f} "
              f"SOC={r['soc_end']:.1f} 触发={sum(u['是否调整'] for u in r['updates'])}")

    sdf=pd.DataFrame(summary); udf=pd.DataFrame(updates); ddf=pd.DataFrame(detail)
    mode='full' if args.full else 'quick4'
    with pd.ExcelWriter(OUT/f'q4_3_final_causal_results_{mode}.xlsx',engine='openpyxl') as w:
        sdf.to_excel(w,index=False,sheet_name='策略汇总')
        udf.to_excel(w,index=False,sheet_name='更新价值')
        ddf.to_excel(w,index=False,sheet_name='时段明细')

    raw_total=float(sdf['阈值策略总成本_元'].sum())
    raw_never=float(sdf['不调整总成本_元'].sum())
    final_soc=float(sdf['SOC终点_kWh'].iloc[-1])
    never_final_soc=float(states['never']) if args.full else float(res['never']['soc_end'])
    terminal_price=float(price[-1,-1])
    adj_terminal=terminal_soc_equivalent_cost(
        final_soc,reference_soc_kwh=SOC_INITIAL_KWH,
        settlement_price_yuan_per_kwh=terminal_price,eta_c=ETA_C,eta_d=ETA_D)
    never_terminal=terminal_soc_equivalent_cost(
        never_final_soc,reference_soc_kwh=SOC_INITIAL_KWH,
        settlement_price_yuan_per_kwh=terminal_price,eta_c=ETA_C,eta_d=ETA_D)
    stats={
        'mode':mode,'days':len(sdf),'mean_cost_yuan':float(sdf['阈值策略总成本_元'].mean()),
        'total_cost_yuan':raw_total,
        'never_total_cost_yuan':raw_never,
        'total_saving_vs_never_yuan':float(raw_never-raw_total),
        'terminal_soc_equivalent_adjustment_yuan':float(adj_terminal),
        'never_terminal_soc_equivalent_adjustment_yuan':float(never_terminal),
        'terminal_normalized_cost_yuan':float(raw_total+adj_terminal),
        'never_terminal_normalized_cost_yuan':float(raw_never+never_terminal),
        'terminal_normalized_saving_vs_never_yuan':float((raw_never+never_terminal)-(raw_total+adj_terminal)),
        'terminal_normalization_reference_soc_kwh':float(SOC_INITIAL_KWH),
        'terminal_normalization_price_yuan_per_kwh':terminal_price,
        'total_external_purchase_kwh':float(sdf['外部购电总量_kWh'].sum()),
        'total_emergency_kwh':float(sdf['紧急购电量_kWh'].sum()),
        'total_emergency_cost_yuan':float(sdf['紧急购电费_元'].sum()),
        'total_adjustment_triggers':int(sdf['触发调整次数'].sum()),
        'mean_price_mae_00':float(sdf['0时电价预测MAE_元每kWh'].mean()),
        'total_unused_contract_kwh':float(sdf['未利用计划购电量_kWh'].sum()),
        'total_pv_spill_kwh':float(sdf['实际弃光量_kWh'].sum()),
        'simultaneous_total':int(sdf['同时充放电次数'].sum()),
        'max_balance_residual_kwh':float(sdf['最大能量平衡残差_kWh'].max()),
        'final_soc_kwh':final_soc,'never_final_soc_kwh':never_final_soc,
        'runtime_seconds':time.time()-t0
    }
    (OUT/f'q4_3_final_causal_summary_{mode}.json').write_text(
        json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8'
    )
    if args.full:
        export_result43(RESULT43_TEMPLATE,OUT/'result4-3.xlsx',chosen,price)
        validate_result43_output(OUT/'result4-3.xlsx',chosen,price)
        print('[表格终检] result4-3.xlsx通过。')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
