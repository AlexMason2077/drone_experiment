"""Search shared SOC boundaries for endpoint-anchored three-line approximations.

All errors are evaluated on raw hover samples, with equal weight per flight.
Boundaries are SOC percentages, not arbitrary time knots. No source writes.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'analysis_results/bideal_20260908'
OUT=ROOT/'analysis_results/bideal_boundary_search_20260908'
PAIRS={'B10':'drone_2','B11':'drone_1','B12':'drone_5','B13':'drone_3','B14':'drone_4'}


def load_trace(row):
    path=ROOT/row.source
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row.sha256
    raw=pd.read_csv(path)
    d=raw[raw.phase.eq('hover_to_10_percent')].sort_values('elapsed_time').copy()
    if len(d)<10:return None
    assert not d.elapsed_time.duplicated().any()
    assert d.battery.between(0,100).all()
    trace=dict(battery_id=row.battery_id,drone=row.drone,run_id=row.run_id,source=row.source,
               sha256=row.sha256,t=d.elapsed_time.to_numpy(float),soc=d.battery.to_numpy(float))
    return trace


def prepare(trace,top=95,bottom=20):
    t,s=trace['t'],trace['soc']
    if s.max()<top or s.min()>bottom:return None
    times={}
    for soc in range(bottom,top+1):
        loc=np.flatnonzero(s<=soc)
        if not len(loc) or s[loc[0]]!=soc:return None
        times[soc]=t[loc[0]]
    keep=(t>=times[top])&(t<=times[bottom])
    t,s=t[keep],s[keep]
    if np.diff(t).max()>5 or abs(np.diff(s)).max()>2 or np.diff(s).max()>1:return None
    return dict(**{k:v for k,v in trace.items() if k not in ['t','soc']},t=t,soc=s,
                crossing=times,top=top,bottom=bottom)


def fit(trace,upper,lower,min_seconds=10):
    levels=np.array([trace['top'],upper,lower,trace['bottom']],float)
    times=np.array([trace['crossing'][int(level)] for level in levels])
    durations=np.diff(times)
    if (durations<min_seconds).any():return None
    pred=np.interp(trace['t'],times,levels)
    resid=trace['soc']-pred
    return dict(mse=float(np.mean(resid**2)),rmse=float(np.sqrt(np.mean(resid**2))),
                max_error=float(abs(resid).max()),pred=pred,anchor_times=times,
                rates=(levels[:-1]-levels[1:])*60/durations,durations=durations)


def search(traces,top=95,bottom=20,min_width=5,min_seconds=10):
    prepared=[prepare(t,top,bottom) for t in traces]
    if any(t is None for t in prepared):raise ValueError('Incomplete common range')
    rows=[]
    for upper in range(bottom+2*min_width,top-min_width+1):
        for lower in range(bottom+min_width,upper-min_width+1):
            fits=[fit(t,upper,lower,min_seconds) for t in prepared]
            if any(f is None for f in fits):continue
            r=dict(upper=upper,lower=lower,rmse=float(np.sqrt(np.mean([f['mse'] for f in fits]))),
                   worst_battery_rmse=max(f['rmse'] for f in fits),min_stage_seconds=min(min(f['durations']) for f in fits),
                   top=top,bottom=bottom,min_width=min_width,min_seconds=min_seconds)
            for t,f in zip(prepared,fits):r[t['battery_id']+'_rmse']=f['rmse']
            rows.append(r)
    return pd.DataFrame(rows).sort_values(['rmse','upper','lower']).reset_index(drop=True),prepared


def run():
    sources=pd.read_csv(PREVIOUS/'coefficients.csv').drop_duplicates(['battery_id','run_id']).sort_values('battery_id')
    assert len(sources)==5 and sources.battery_id.nunique()==5
    assert all(PAIRS[r.battery_id]==r.drone for r in sources.itertuples())
    assert not sources.run_id.eq('20260906_165812').any()
    traces=[load_trace(r) for r in sources.itertuples()]
    grid,prepared=search(traces)
    best=grid.iloc[0]
    baseline=grid[(grid.upper==75)&(grid.lower==40)].iloc[0]
    rounded=grid[(grid.upper%5==0)&(grid.lower%5==0)].iloc[0]
    choices=[('original',75,40),('best',int(best.upper),int(best.lower)),('rounded',int(rounded.upper),int(rounded.lower))]
    comparison=[];points=[];rates=[];residuals=[]
    for name,u,l in choices:
        for t in prepared:
            f=fit(t,u,l)
            comparison.append(dict(model=name,upper=u,lower=l,battery_id=t['battery_id'],run_id=t['run_id'],
                n_samples=len(t['t']),rmse_pp=f['rmse'],max_error_pp=f['max_error'],source=t['source']))
            for time,soc,pred in zip(t['t'],t['soc'],f['pred']):
                residuals.append(dict(model=name,upper=u,lower=l,battery_id=t['battery_id'],run_id=t['run_id'],
                    elapsed_time=float(time),observed_soc=float(soc),modeled_soc=float(pred),
                    residual_pp=float(soc-pred),source=t['source']))
            for k in range(3):rates.append(dict(model=name,battery_id=t['battery_id'],stage=['high','medium','low'][k],
                 upper=[95,u,l][k],lower=[u,l,20][k],rate_pp_min=float(f['rates'][k]),duration_s=float(f['durations'][k]),
                 source=t['source']))
            # Model preview against actual samples; elapsed time begins at observed95.
            for i in range(0,len(t['t']),max(1,len(t['t'])//100)):
                points.append(dict(model=name,battery_id=t['battery_id'],run_id=t['run_id'],
                    t_s=float(t['t'][i]-t['t'][0]),observed_soc=float(t['soc'][i]),model_soc=float(f['pred'][i]),
                    residual_pp=float(t['soc'][i]-f['pred'][i]),source=t['source'],upper=u,lower=l))
    # A robustness check on boundary selection, not an independent prediction test:
    # held-out battery still supplies its own endpoint calibration rates.
    folds=[]
    for held in traces:
        g,p=search([t for t in traces if t['battery_id']!=held['battery_id']])
        b=g.iloc[0];h=prepare(held);f=fit(h,int(b.upper),int(b.lower));old=fit(h,75,40)
        folds.append(dict(held_battery=held['battery_id'],upper=int(b.upper),lower=int(b.lower),
                    training_rmse=float(b.rmse),held_rmse=f['rmse'] if f else None,original_held_rmse=old['rmse']))
    sensitivity=[]
    for top,bottom,width,seconds in [(95,20,10,10),(90,20,5,10),(95,25,5,10),(95,15,5,10),(95,11,5,10),(95,20,5,15)]:
        g,_=search(traces,top,bottom,width,seconds);b=g.iloc[0]
        old=g[(g.upper==75)&(g.lower==40)]
        sensitivity.append(dict(top=top,bottom=bottom,min_width=width,min_seconds=seconds,
            upper=int(b.upper),lower=int(b.lower),rmse=float(b.rmse),
            original_rmse=float(old.iloc[0].rmse) if len(old) else None))
    # Historical same-ID pair checks do not pool dates with current calibration.
    inventory=pd.read_csv(PREVIOUS/'inventory.csv')
    history=[]
    for row in inventory.itertuples():
        if row.status!='observed_hover' or str(row.run_id).startswith('20260906') or PAIRS.get(row.battery_id)!=row.drone:continue
        raw=load_trace(row)
        if raw is None:continue
        t=prepare(raw)
        if t is None:continue
        for name,u,l in choices:
            f=fit(t,u,l)
            if f:history.append(dict(battery_id=row.battery_id,run_id=row.run_id,model=name,
                upper=u,lower=l,rmse_pp=f['rmse'],source=row.source))
    summary=dict(method='shared SOC knots, endpoint-anchored continuous lines, equal-flight pooled RMSE on real95→20 samples',
        current_runs=5,candidates=len(grid),best=best.to_dict(),original=baseline.to_dict(),rounded=rounded.to_dict(),
        in_sample_improvement_fraction=1-best.rmse/baseline.rmse,
        near_optimal_2pct=grid[grid.rmse<=best.rmse*1.02][['upper','lower','rmse']].to_dict('records'),
        validation_scope='LOBO measures shared-boundary sensitivity with held-battery endpoint recalibration; not held-out-flight prediction accuracy.',
        source_rows=sources[['battery_id','drone','run_id','source','sha256']].to_dict('records'))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,df in [('search',grid),('comparison',pd.DataFrame(comparison)),('curve_preview',pd.DataFrame(points)),('residuals',pd.DataFrame(residuals)),
                    ('stage_rates',pd.DataFrame(rates)),('leave_one_battery_out',pd.DataFrame(folds)),
                    ('sensitivity',pd.DataFrame(sensitivity)),('historical_checks',pd.DataFrame(history))]:df.to_csv(OUT/(name+'.csv'),index=False)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Candidates:',len(grid));print(grid.head(10).round(4).to_string(index=False))
    print('Original:',baseline[['upper','lower','rmse']].to_dict())
    print('Rounded:',rounded[['upper','lower','rmse']].to_dict())
    print(pd.DataFrame(comparison).pivot(index='battery_id',columns='model',values='rmse_pp').round(4).to_string())
    print('Leave one battery out:');print(pd.DataFrame(folds).round(4).to_string(index=False))
    print('Sensitivity:');print(pd.DataFrame(sensitivity).round(4).to_string(index=False))
    print('History:');print(pd.DataFrame(history).pivot(index=['battery_id','run_id'],columns='model',values='rmse_pp').round(4).to_string())
    return summary


if __name__=='__main__':run()
