"""Read-only source analysis: battery/pair baseline characterization; no flight imports.

Run from repository root. Generated evidence goes to analysis_results/bideal_20260908.
Threshold times are first observed crossings, not fabricated telemetry.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis_results' / 'bideal_20260908'
BATTERIES = ['B10', 'B11', 'B12', 'B13', 'B14']
BANDS = {'high95_75': (95, 75), 'medium75_40': (75, 40),
         'low40_20': (40, 20), 'low20_10': (20, 10)}
PAIRS = {'B10': 'drone_2', 'B11': 'drone_1', 'B12': 'drone_5',
         'B13': 'drone_3', 'B14': 'drone_4'}
EXCLUDED_BASELINE_RUNS = {
    ('B12', 'drone_2', '20260906_165812'):
        'User excluded this B12/D2 run from the Bideal analysis; raw files retained.'
}


def summarize_band(d, upper, lower):
    d = d.sort_values('elapsed_time').drop_duplicates('elapsed_time')
    if d.empty or d.battery.max() < upper or d.battery.min() > lower:
        return None
    a = d[d.battery <= upper].iloc[0]
    tail = d[d.elapsed_time > a.elapsed_time]
    tail = tail[tail.battery <= lower]
    if tail.empty:
        return None
    b = tail.iloc[0]
    seg = d[d.elapsed_time.between(a.elapsed_time, b.elapsed_time)]
    duration = float(b.elapsed_time - a.elapsed_time)
    if len(seg) < 10 or duration < 10:
        return None
    # Crossing overshoots invalidate fixed-boundary comparison.
    valid = bool(a.battery == upper and b.battery == lower
                 and seg.elapsed_time.diff().max() <= 5
                 and seg.battery.diff().abs().max() <= 2
                 and seg.battery.diff().max() <= 1)
    slope, intercept = np.polyfit(seg.elapsed_time - a.elapsed_time, seg.battery, 1)
    resid = seg.battery - (intercept + slope * (seg.elapsed_time - a.elapsed_time))
    result = dict(upper=upper, lower=lower, start_s=float(a.elapsed_time),
                  end_s=float(b.elapsed_time), duration_s=duration,
                  rate_pp_min=(upper-lower)*60/duration, ols_pp_min=float(-slope*60),
                  rmse_pp=float(np.sqrt(np.mean(resid**2))), n=len(seg), valid=valid,
                  max_gap_s=float(seg.elapsed_time.diff().max()),
                  upward_steps=int((seg.battery.diff()>0).sum()))
    for col in ['h', 'tof', 'templ', 'temph', 'pitch', 'roll']:
        if col in seg:
            s = pd.to_numeric(seg[col], errors='coerce')
            result[col+'_median'] = float(s.median()) if s.notna().any() else None
    if 'mid' in seg and 'mission_pad' in seg:
        own = pd.to_numeric(seg.mid,errors='coerce') == pd.to_numeric(seg.mission_pad,errors='coerce')
        result['own_pad_fraction'] = float(own.mean())
        if 'x' in seg and 'y' in seg:
            result['own_pad_near20_fraction'] = float((own & (seg.x.abs()<=20) & (seg.y.abs()<=20)).mean())
    return result


def baselines():
    inventory, bands, traces = [], [], []
    seen = set()
    for meta in sorted((ROOT/'database/baselines').glob('*/*hover*metadata.json')):
        m = json.loads(meta.read_text())
        battery = m.get('battery_id')
        if battery not in BATTERIES:
            continue
        stem = meta.name.removesuffix('_metadata.json')
        files = sorted(meta.parent.glob(stem+'*all_coordination.csv'))
        if not files:
            files = sorted(meta.parent.glob(stem+'*timeseries.csv'))
        info = dict(battery_id=battery, drone=m.get('drone_name'), run_id=m['run_id'],
                    date=str(m['run_id'])[-15:-7], baseline_id=stem,
                    metadata=str(meta.relative_to(ROOT)))
        if not files:
            inventory.append(dict(**info, status='no_telemetry')); continue
        p = files[0]
        d = pd.read_csv(p)
        info.update(source=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), rows=len(d))
        excluded_reason = EXCLUDED_BASELINE_RUNS.get((battery, info['drone'], m['run_id']))
        if excluded_reason:
            inventory.append(dict(**info,status='excluded_by_user',exclusion_reason=excluded_reason))
            continue
        key = (battery, m['run_id'])
        if key in seen:
            inventory.append(dict(**info,status='duplicate_run')); continue
        seen.add(key)
        for c in ['battery','elapsed_time']:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        info['duplicate_time_rows'] = int(d.duplicated('elapsed_time').sum())
        info['invalid_soc_rows'] = int((~d.battery.between(0,100)).sum())
        h = d[d.phase.eq('hover_to_10_percent') & d.battery.between(0,100)].copy()
        if h.empty:
            inventory.append(dict(**info,status='no_hover')); continue
        info.update(status='observed_hover', hover_s=float(h.elapsed_time.max()-h.elapsed_time.min()),
                    start_soc=float(h.battery.iloc[0]),end_soc=float(h.battery.iloc[-1]))
        info['full95_10'] = info['start_soc']>=95 and info['end_soc']<=10
        inventory.append(info)
        for name,(upper,lower) in BANDS.items():
            r = summarize_band(h,upper,lower)
            if r:
                bands.append(dict(**info,band=name,**r))
        # Bounded plotting preview; raw samples unchanged in source.
        preview=h.iloc[::max(1,len(h)//100)]
        for _,r in preview.iterrows():
            traces.append(dict(battery_id=battery,drone=info['drone'],run_id=info['run_id'],
                               t=float(r.elapsed_time-h.elapsed_time.iloc[0]),soc=float(r.battery)))
    return pd.DataFrame(inventory),pd.DataFrame(bands),pd.DataFrame(traces)


def wind_tunnel():
    registry={r['experiment_id']:r for r in json.loads((ROOT/'database/experiment_registry.json').read_text())['experiments']}
    rows,inventory=[],[]
    for p in sorted((ROOT/'database').glob('wind_tunnel*/*_all_coordination.csv')):
        exp=p.parent.name
        rec=registry.get(exp,{})
        info=dict(experiment_id=exp,source=str(p.relative_to(ROOT)))
        if 'merged' in p.name or 'prepare' in exp or rec.get('is_outlier'):
            inventory.append(dict(**info,status='excluded_prepare_merged_or_outlier'));continue
        wanted=['run_id','battery_id','drone_name','elapsed_time','battery','phase','formation',
                'wind_direction','wind_speed','inter_drone_distance_cm','mid','mission_pad','x','y','h','tof']
        d=pd.read_csv(p,usecols=lambda c:c in wanted)
        if d.empty:
            inventory.append(dict(**info,status='empty'));continue
        if d.phase.str.contains('merge|simulat|interpol',case=False).any():
            inventory.append(dict(**info,status='synthetic_phase'));continue
        start_times=d[d.phase.eq('wind_tunnel_hover')].groupby('drone_name').elapsed_time.min()
        if len(start_times)!=5:
            inventory.append(dict(**info,status='not_five_hovering'));continue
        start=max(10,float(start_times.max()))
        end_rows=d[d.phase.str.contains('land|fault|recoverable_error',case=False)]
        end=float(end_rows.elapsed_time.min()) if not end_rows.empty else float(d.elapsed_time.max())
        inventory.append(dict(**info,status='screened',start_s=start,end_s=end))
        active=d[d.elapsed_time.between(start,end,inclusive='left')]
        for (drone,battery),g in active.groupby(['drone_name','battery_id']):
            if battery not in BATTERIES:continue
            for band,(upper,lower) in BANDS.items():
                r=summarize_band(g,upper,lower)
                if r:
                    rows.append(dict(**info,run_id=str(g.run_id.iloc[0]),drone=drone,battery_id=battery,
                            formation=str(g.formation.iloc[0]).replace('echalon','echelon'),
                            wind_direction=g.wind_direction.iloc[0],wind_speed=g.wind_speed.iloc[0],
                            spacing=float(g.inter_drone_distance_cm.iloc[0]),band=band,**r))
    return pd.DataFrame(inventory),pd.DataFrame(rows)


def analyze():
    inventory, bands, traces = baselines()
    valid=bands[bands.valid].copy()
    recent=valid[valid.run_id.str.startswith('20260906')].copy()
    matched=recent[recent.apply(lambda r: PAIRS[r.battery_id]==r.drone,axis=1)].copy()
    # Equal battery weighting, using one contemporary run for each operational pair.
    assert not matched.duplicated(['battery_id','band']).any()
    core=matched[matched.band.ne('low20_10')]
    ideal=core.groupby('band').agg(bideal_pp_min=('rate_pp_min','mean'),battery_count=('battery_id','nunique')).reset_index()
    assert (ideal.battery_count==5).all()
    coefficients=matched.merge(ideal,on='band')
    coefficients['physical_to_bideal_scale']=coefficients.bideal_pp_min/coefficients.rate_pp_min
    coefficients['status']='PROVISIONAL_battery_airframe_pair_not_pure_battery'
    assert np.allclose(coefficients.rate_pp_min*coefficients.physical_to_bideal_scale,coefficients.bideal_pp_min)
    wi,wr=wind_tunnel()
    # Within-run, within-battery low/medium multiplier: no pooling of wind/configuration.
    piv=wr[wr.valid].pivot_table(index=['source','battery_id','drone','formation','wind_direction','wind_speed','spacing'],columns='band',values='rate_pp_min')
    paired=piv.dropna(subset=['medium75_40','low40_20']).copy()
    paired['low_medium_multiplier']=paired.low40_20/paired.medium75_40
    paired=paired.reset_index()
    result=dict(inventory=inventory,bands=bands,traces=traces,recent=recent,matched=matched,
                coefficients=coefficients,wind_inventory=wi,wind_rates=wr,wind_pairs=paired)
    return result


if __name__=='__main__':
    result=analyze()
    OUT.mkdir(parents=True,exist_ok=True)
    for name,df in result.items():
        df.to_csv(OUT/(name+'.csv'),index=False)
    print('Baseline inventory:',result['inventory'].status.value_counts().to_dict())
    print('\nLatest same-day bands:')
    print(result['recent'][['battery_id','drone','run_id','band','duration_s','rate_pp_min','ols_pp_min','h_median','tof_median']].round(3).to_string(index=False))
    print('\nProvisional coefficients:')
    print(result['coefficients'][['battery_id','band','bideal_pp_min','physical_to_bideal_scale']].round(4).to_string(index=False))
    print('\nWind tunnel paired counts and multipliers:')
    print(result['wind_pairs'].groupby('battery_id').low_medium_multiplier.agg(['count','median','min','max']).round(3).to_string())
