"""Construct an explicitly modeled, continuous three-line Bideal reference.

Uses the reviewed current-pair baseline thresholds, never edits raw observations.
Default evaluation stops at 20%; continuation to zero requires explicit opt-in.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import sqlite3
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'analysis_results/bideal_20260908'
OUT=ROOT/'analysis_results/bideal_three_stage_v1_20260908'
STAGES=[('high','high95_75',100.,75.),('medium','medium75_40',75.,40.),('low','low40_20',40.,0.)]


def build_model():
    rates=pd.read_csv(PREVIOUS/'coefficients.csv')
    assert len(rates)==15 and rates.battery_id.nunique()==5
    assert set(rates[rates.battery_id.eq('B12')].drone)=={'drone_5'}
    diagnostics=[]
    # Validate all 15 scalar rates independently against original threshold times.
    for _,r in rates.iterrows():
        path=ROOT/r.source
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r.sha256
        raw=pd.read_csv(path)
        d=raw[raw.phase.eq('hover_to_10_percent')].copy()
        a=d[d.battery.le(r.upper)].iloc[0]
        b=d[(d.elapsed_time>a.elapsed_time)&d.battery.le(r.lower)].iloc[0]
        rate=60*(r.upper-r.lower)/(b.elapsed_time-a.elapsed_time)
        assert math.isclose(rate,r.rate_pp_min,rel_tol=1e-10)
        seg=d[d.elapsed_time.between(a.elapsed_time,b.elapsed_time)]
        pred=r.upper-rate*(seg.elapsed_time-a.elapsed_time)/60
        diagnostics.append(dict(battery_id=r.battery_id,stage=r.band,run_id=r.run_id,
            rmse_pp=float(np.sqrt(np.mean((seg.battery-pred)**2))),
            max_abs_error_pp=float(np.max(abs(seg.battery-pred))),
            n_samples=len(seg),source=r.source))
    time=0.;segments=[]
    for name,band,upper,lower in STAGES:
        g=rates[rates.band.eq(band)]
        assert len(g)==5 and g.battery_id.nunique()==5
        rate=float(g.rate_pp_min.mean())
        duration=60*(upper-lower)/rate
        segments.append(dict(stage=name,calibration_band=band,upper_soc=upper,lower_soc=lower,
            rate_pp_min=rate,rate_pp_s=rate/60,slope_pp_s=-rate/60,
            start_s=time,end_s=time+duration,duration_s=duration,
            intercept_pp=upper+rate*time/60,n_batteries=5,
            observed_upper_soc=float(g.upper.iloc[0]),observed_lower_soc=float(g.lower.iloc[0])))
        time+=duration
    model=dict(version='bideal_three_stage_v1_20260908',kind='modeled_reference_not_measured_battery',
        method='Endpoint-calibrated battery rates; equal-battery arithmetic mean in each stage; continuity by cumulative segment duration.',
        source_date='2026-09-06',mathematical_domain_soc=[0,100],default_domain_soc=[20,100],
        directly_calibrated_soc=[20,95],extrapolated_soc_intervals=[[95,100],[0,20]],
        boundaries_soc=[100,75,40,0],segments=segments,
        high_extension_note='100→95 uses the slope measured over 95→75; not a measured 100→75 average.',
        low_extension_note='20→0 is an unvalidated extension of the 40→20 slope; it is not a safe flight target.',
        unit='reported SOC percentage points per minute',
        excluded_run='B12/drone_2/20260906_165812',
        normalization='For a drop, split its original SOC range at 75 and 40; multiply each part by stage Bideal_rate / own_baseline_rate. Not an absolute SOC multiplier.',
        limitations=['One contemporary complete run per battery-airframe pair; coefficients remain provisional.',
                     'Piecewise linearity is a deliberate approximation, not a test result proving true linear discharge.',
                     'Wind/load transfer and pure battery/airframe separation are not validated.'])
    return model,rates,pd.DataFrame(diagnostics)


def soc_after(model,start_soc,duration_s,allow_low_extrapolation=False):
    floor=0. if allow_low_extrapolation else 20.
    if not math.isfinite(start_soc) or not floor<=start_soc<=100:
        raise ValueError('Initial SOC outside allowed model range')
    if not math.isfinite(duration_s) or duration_s<0:
        raise ValueError('Duration must be finite and nonnegative')
    soc=float(start_soc);remaining=float(duration_s)
    for s in model['segments']:
        bottom=max(floor,s['lower_soc'])
        if not bottom<soc<=s['upper_soc']:continue
        until_boundary=(soc-bottom)/s['rate_pp_s']
        if remaining<=until_boundary+1e-10:
            return max(bottom,soc-remaining*s['rate_pp_s'])
        remaining-=until_boundary;soc=bottom
    if remaining>1e-9:
        raise ValueError('Requested duration crosses the supported SOC floor')
    return soc


def normalized_drop(model,rates,battery_id,start_soc,end_soc):
    if not 20<=end_soc<=start_soc<=100:
        raise ValueError('Drop normalization is restricted to 100→20')
    own=rates[rates.battery_id.eq(battery_id)].set_index('band')
    if len(own)!=3:raise ValueError('Unknown or incomplete battery calibration')
    total=0.
    for s in model['segments']:
        overlap=max(0.,min(start_soc,s['upper_soc'])-max(end_soc,s['lower_soc'],20.))
        total+=overlap*s['rate_pp_min']/float(own.loc[s['calibration_band'],'rate_pp_min'])
    return total


def validate(model,rates):
    s=model['segments'];assert len(s)==3
    for a,b in zip(s,s[1:]):
        assert a['end_s']==b['start_s'] and a['lower_soc']==b['upper_soc']
        assert math.isclose(a['intercept_pp']+a['slope_pp_s']*a['end_s'],b['upper_soc'],abs_tol=1e-9)
    assert all(r['slope_pp_s']<0 for r in s)
    for r in s:
        assert math.isclose(soc_after(model,100,r['end_s'],True),r['lower_soc'],abs_tol=1e-8)
    grid=np.linspace(0,s[-1]['end_s'],501)
    values=np.array([soc_after(model,100,t,True) for t in grid])
    assert np.all(np.diff(values)<0)
    try:soc_after(model,100,s[-1]['end_s'])
    except ValueError:pass
    else:raise AssertionError('Default evaluator must reject below20')
    for b in rates.battery_id.unique():
        assert math.isclose(normalized_drop(model,rates,b,80,35),
            normalized_drop(model,rates,b,80,75)+normalized_drop(model,rates,b,75,40)+normalized_drop(model,rates,b,40,35))
    return dict(status='passed',raw_threshold_checks=15,continuous_boundaries=2,
        monotonicity_grid_points=501,below20_guard=True,normalization_additivity=True)


def build():
    model,rates,diagnostics=build_model();qa=validate(model,rates)
    OUT.mkdir(parents=True,exist_ok=True)
    s=pd.DataFrame(model['segments']);s.to_csv(OUT/'segments.csv',index=False)
    (OUT/'model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2))
    (OUT/'validation.json').write_text(json.dumps(qa,indent=2))
    diagnostics.to_csv(OUT/'approximation_diagnostics.csv',index=False)
    rates[['battery_id','drone','band','rate_pp_min','physical_to_bideal_scale','source','sha256']].to_csv(OUT/'battery_scales.csv',index=False)
    # This SQL materially generates the model curve; rows are model evaluations, not telemetry.
    times=sorted(set([float(t) for t in np.linspace(0,s.end_s.max(),109)]+list(s.start_s)+list(s.end_s)+[
        5/model['segments'][0]['rate_pp_s'],model['segments'][2]['start_s']+20/model['segments'][2]['rate_pp_s']]))
    sql='''SELECT g.time_s,s.stage,s.upper_soc,s.lower_soc,s.rate_pp_min,s.start_s,s.end_s,
           s.upper_soc-s.rate_pp_s*(g.time_s-s.start_s) AS soc_pct,
           'model_not_observation' AS data_kind
           FROM model_grid g JOIN model_segments s ON g.time_s>=s.start_s
           AND (g.time_s<s.end_s OR (s.stage='low' AND g.time_s<=s.end_s))
           ORDER BY g.time_s'''
    with sqlite3.connect(':memory:') as conn:
        s.to_sql('model_segments',conn,index=False)
        pd.DataFrame({'time_s':times}).to_sql('model_grid',conn,index=False)
        curve=pd.read_sql_query(sql,conn)
    curve['basis']=np.where(curve.soc_pct>95,'upper_extension',np.where(curve.soc_pct<20,'unvalidated_low_extension','calibrated_interval_model'))
    assert len(curve)==len(times) and np.allclose(curve.soc_pct,[soc_after(model,100,t,True) for t in times])
    curve.to_csv(OUT/'model_curve.csv',index=False)
    # Extend the existing complete report; unaffected sections/data stay intact.
    artifact=json.loads((PREVIOUS/'artifact.json').read_text())
    source=dict(id='three-stage',label='三段Bideal参考模型（非实测曲线）',path='analysis_results/bideal_three_stage_v1_20260908/model.json',
        query=dict(sql=sql,engine='sqlite',tables_used=['model_grid','model_segments'],
        description='model_segments由五个已采用Hover baseline的分区率等权计算；model_grid仅用于求值绘图，不是新增实验样本。',
        codePath='output_py/build_bideal_three_stage.py',filters=['B12 current calibration uses D5 only','95→100 and below20 are model extensions']))
    m=artifact['manifest'];m['sources'].append(source)
    stage_text='''## 已生成三条连续直线组成的Bideal
High：100→75%，Medium：75→40%，Low：40→0%；分别使用固定的下降速度。曲线在75%和40%处相接，没有跳变。
这是一条人为定义的参考电池模型，不是新增实测数据。High斜率用95→75%的数据计算并延至100%；Low斜率用40→20%的数据计算，20→0仅为未验证延长线。默认计算拒绝进入20%以下。
'''
    table_text='|阶段|下降速度（百分点/分钟）|模型累计时间（秒）|\n|---|---:|---:|\n'
    for r in model['segments']:
        table_text+=f"|{r['stage']}|{r['rate_pp_min']:.4f}|{r['start_s']:.2f}–{r['end_s']:.2f}|\n"
    method_text='''## 三段直线的构造与使用
每块电池每阶段的平均率由真实SOC阈值之间的下降量除以实际时间计算；五块电池等权平均得到Bideal斜率。以100%为起点，阶段耗时=该阶段SOC差÷斜率，再累加阶段耗时，保证两处连接连续。
这是端点平均率线性化，不是无约束最小二乘拟合；不声称它是误差最小的曲线。每块电池自身直线近似相对原始样本的RMSE和最大误差已另存，计算时没有把Bideal参考曲线误当作各电池的共同观测真值。
在原始SOC所在区间，转换下降量=实际下降量×Bideal率÷该电池baseline率；跨区间拆段求和。参考模型的绝对SOC不通过直接乘系数获得。
模型可直接用于下一步数据处理，但本次不自动替换旧冻结耗电率表或训练输入。先固定这版定义，再统一重算；不能将此模型等同于实飞安全依据或已验证的全负载电池模型。
'''
    formulas='## 三阶段时间函数\n\nt单位为秒；B(t)为模型SOC百分数。\n\n'
    for r in model['segments']:
        formulas+=f"- {r['stage']}: B(t) = {r['upper_soc']:g} − {r['rate_pp_s']:.9f} × (t − {r['start_s']:.6f})，{r['start_s']:.6f} ≤ t ≤ {r['end_s']:.6f}。\n"
    m['blocks'][1:1]=[dict(id='three-stage-summary',type='markdown',body=stage_text+'\n'+table_text),
        dict(id='three-stage-curve',type='chart',chartId='three-stage-curve'),
        dict(id='three-stage-method',type='markdown',body=method_text),dict(id='three-stage-formulas',type='markdown',body=formulas)]
    m['charts'].append(dict(id='three-stage-curve',type='line',title='Bideal三阶段SOC–时间模型',
        dataset='three-stage-curve',source=source,sourceId='three-stage',
        encodings=dict(x=dict(field='time_s',label='模型悬停时间（秒）'),y=dict(field='soc_pct',label='模型SOC（%）')),
        palette=dict(kind='sequential',colors=['#3B6FB6'])))
    artifact['snapshot']['datasets']['three-stage-curve']=json.loads(curve.to_json(orient='records'))
    (OUT/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
    (OUT/'README.md').write_text('# Bideal三阶段直线模型 v1\n\n'+stage_text+'\n'+table_text+'\n'+formulas+'\n'+method_text+
        '\n验证：15个原始阈值重算、75/40连续性、501点单调性、默认20%下限、跨段归一化加和均通过。\n')
    # Reuse the already executed companion as the full audit history, then execute new model cells.
    nb=copy.deepcopy(json.loads((PREVIOUS/'bideal_analysis.ipynb').read_text()))
    cells=[dict(cell_type='markdown',metadata={},source=['## 三段连续直线模型\n',stage_text,table_text]),
        dict(cell_type='code',metadata={},source=["from output_py.build_bideal_three_stage import build_model, validate, soc_after\n",
             "model, rates, diagnostics = build_model()\nprint(validate(model,rates))\n",
             "print(__import__('pandas').DataFrame(model['segments'])[['stage','rate_pp_min','start_s','end_s']].round(4).to_string(index=False))\n"],execution_count=None,outputs=[])]
    import contextlib,io,sys
    sys.path.insert(0,str(ROOT))
    nb['cells'].extend(cells);env={};count=0
    for c in nb['cells']:
        if c['cell_type']!='code':continue
        count+=1;buf=io.StringIO()
        with contextlib.redirect_stdout(buf):exec(compile(''.join(c['source']),'<notebook-cell>','exec'),env)
        c['execution_count']=count;c['outputs']=[dict(output_type='stream',name='stdout',text=buf.getvalue().splitlines(keepends=True))]
    (OUT/'three_stage_analysis.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))
    print(s[['stage','rate_pp_min','start_s','end_s']].round(6).to_string(index=False))
    print(json.dumps(qa));print('Notebook executed:',count,'code cells')
    print('Anchored linear approximation RMSE range:',diagnostics.rmse_pp.min(),diagnostics.rmse_pp.max())


if __name__=='__main__':build()
