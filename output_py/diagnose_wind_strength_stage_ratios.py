"""Matched wind-level diagnosis; observation only, no raw-data/model mutation."""
from pathlib import Path
import hashlib
import itertools
import json
import math
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'output_py'))
from audit_wind_tunnel_stage_ratios import stage_stats, crossing, sha, FIELDS
from battery_normalization import BatteryNormalizer

PRIOR = ROOT/'analysis_results/wind_tunnel_stage_ratio_check_20260909/first_drop_recalculation'
OUT = ROOT/'analysis_results/wind_strength_stage_ratio_diagnosis_20260909'
MODEL = ROOT/'analysis_results/battery_normalization_v3_with_b15_20260909/model.json'
KEYS = ['wind_direction','formation','spacing_cm','battery_id','drone_id']
SQL = '''SELECT a.wind_direction,a.formation,a.spacing_cm,a.battery_id,a.drone_id,
 a.source AS source_lv1,b.source AS source_lv2,
 a.experiment_id AS experiment_lv1,b.experiment_id AS experiment_lv2,
 a.run_id AS run_lv1,b.run_id AS run_lv2,
 a.high_k/a.medium_k AS high_ratio_lv1,b.high_k/b.medium_k AS high_ratio_lv2,
 a.low_k/a.medium_k AS low_ratio_lv1,b.low_k/b.medium_k AS low_ratio_lv2,
 100.0*((b.high_k/b.medium_k)/(a.high_k/a.medium_k)-1.0) AS high_change_pct,
 100.0*((b.low_k/b.medium_k)/(a.low_k/a.medium_k)-1.0) AS low_change_pct,
 100.0*(b.high_raw_rate_pp_min/a.high_raw_rate_pp_min-1.0) AS high_rate_change_pct,
 100.0*(b.medium_raw_rate_pp_min/a.medium_raw_rate_pp_min-1.0) AS medium_rate_change_pct,
 100.0*(b.low_raw_rate_pp_min/a.low_raw_rate_pp_min-1.0) AS low_rate_change_pct,
 a.high_starts_after_all_ready AND b.high_starts_after_all_ready AS both_high_after_ready,
 a.low_ends_before_first_landing AND b.low_ends_before_first_landing AS both_full_low_before_landing,
 a.low_after_first_landing_fraction AS low_after_landing_fraction_lv1,
 b.low_after_first_landing_fraction AS low_after_landing_fraction_lv2
FROM full_curves a JOIN full_curves b
 ON a.wind_direction=b.wind_direction AND a.formation=b.formation
 AND a.spacing_cm=b.spacing_cm AND a.battery_id=b.battery_id AND a.drone_id=b.drone_id
WHERE a.wind_level=1 AND b.wind_level=2
ORDER BY a.wind_direction,a.formation,a.spacing_cm,a.battery_id'''


def describe(x):
    x=pd.Series(x).dropna()
    return dict(n=len(x), median=float(x.median()), median_abs=float(x.abs().median()),
                mean=float(x.mean()), mean_abs=float(x.abs().mean()), minimum=float(x.min()), maximum=float(x.max()),
                increased=int(x.gt(0).sum()), decreased=int(x.lt(0).sum()), within10=int(x.abs().le(10).sum()))


def scan_partial_pairs():
    """Allow complete stage pairs in an otherwise incomplete flight, with trace QA."""
    norm=BatteryNormalizer.load(MODEL)
    registry_path=ROOT/'database/experiment_registry.json'
    registry={r['experiment_id']:r for r in json.loads(registry_path.read_text())['experiments']}
    paths=sorted((ROOT/'database').glob('wind_tunnel*/*_all_coordination.csv'))
    outputs=[]; audit=[]; hashes={str(MODEL.relative_to(ROOT)):sha(MODEL),str(registry_path.relative_to(ROOT)):sha(registry_path)}
    seen=set()
    for p in paths:
        source=str(p.relative_to(ROOT)); exp=p.parent.name; rec=registry.get(exp,{})
        if 'prepare' in exp.lower() or 'merged' in p.name.lower() or rec.get('is_outlier'):
            audit.append(dict(source=source,status='excluded_prepare_merged_outlier'));continue
        hashes[source]=sha(p)
        d=pd.read_csv(p,usecols=lambda c:c in FIELDS,low_memory=False)
        if d.empty or not {'phase','battery','elapsed_time'}.issubset(d):
            audit.append(dict(source=source,status='empty_or_missing'));continue
        if d.phase.str.contains('merge|simulat|interpol',case=False,na=False).any():
            audit.append(dict(source=source,status='synthetic_phase'));continue
        for col in ['elapsed_time','battery','x','y','h','tof']:
            d[col]=pd.to_numeric(d[col],errors='coerce')
        ready=d[d.phase.eq('wind_tunnel_hover')].groupby('drone_name').elapsed_time.min()
        if len(ready)!=5:
            audit.append(dict(source=source,status='not_all_five_reached_hover'));continue
        all_ready=float(ready.max())
        land=d[d.phase.str.contains('land',case=False,na=False)]
        firstland=float(land.elapsed_time.min()) if len(land) else math.inf
        for (drone,battery),g in d.groupby(['drone_name','battery_id']):
            info=dict(source=source,experiment_id=exp,drone_id=drone,battery_id=battery,run_id=str(g.run_id.iloc[0]))
            if battery not in norm.curves or norm.pairs[battery]!=drone:
                audit.append(dict(**info,status='unsupported_battery_pair'));continue
            ident=(info['run_id'],drone,battery)
            if ident in seen:
                audit.append(dict(**info,status='duplicate_run'));continue
            seen.add(ident)
            g=g.sort_values('elapsed_time').copy()
            takeoff=g[g.phase.eq('wind_tunnel_takeoff')]
            if takeoff.empty:
                audit.append(dict(**info,status='no_takeoff'));continue
            g=g[g.elapsed_time.ge(takeoff.elapsed_time.iloc[0])]
            if g[['elapsed_time','battery']].isna().any().any() or not g.battery.between(0,100).all() or g.elapsed_time.duplicated().any():
                audit.append(dict(**info,status='invalid_samples'));continue
            ownland=g[g.phase.str.contains('land',case=False,na=False)]
            if len(ownland):g=g[g.elapsed_time.le(ownland.elapsed_time.iloc[0])]
            curve=norm.curve_for(battery,drone)
            upper,lower=curve.boundaries[1:3]
            tupper=crossing(g,upper); tlower=crossing(g,lower,tupper if tupper is not None else -math.inf)
            t20=crossing(g,20,tlower if tlower is not None else -math.inf)
            firstdrop=g[g.battery.diff().lt(0)]
            start=None
            if g.battery.iloc[0]==100 and len(firstdrop) and firstdrop.battery.iloc[0]==99:
                start=float(firstdrop.elapsed_time.iloc[0])
            pairs={}
            for stage,i,t0,t1,hi,lo in [('high',0,start,tupper,99,upper),
                                      ('medium',1,tupper,tlower,upper,lower),
                                      ('low',2,tlower,t20,lower,20)]:
                if t0 is None or t1 is None or t1<=t0:continue
                window=g[g.elapsed_time.between(t0,t1)]
                if window.phase.str.contains('fault|recoverable_error|uncommanded',case=False,na=False).any():continue
                st=stage_stats(g,t0,t1,hi,lo,curve.rates_pp_min[i],norm.reference.rates_pp_min[i],all_ready,firstland)
                if st is not None:pairs[stage]=st
            levelstr=str(rec.get('wind_speed',g.wind_speed.iloc[0])).lower()
            match=re.search(r'(\d+)',levelstr); level=int(match.group(1)) if match else 0
            if 'no_wind' in exp:level=0
            info.update(wind_level=level,wind_direction=str(rec.get('wind_direction',g.wind_direction.iloc[0])).lower().replace(' wind',''),
                        formation=str(rec.get('formation',g.formation.iloc[0])).lower().replace('echalon','echelon').replace('echolon','echelon'),
                        spacing_cm=float(rec.get('inter_drone_distance_cm',g.inter_drone_distance_cm.iloc[0])))
            added=0
            for stage in ['high','low']:
                if stage not in pairs or 'medium' not in pairs:continue
                s=pairs[stage];m=pairs['medium']
                outputs.append(dict(**info,stage=stage,ratio=s['k']/m['k'],stage_k=s['k'],medium_k=m['k'],
                                    starts_after_all_ready=s['start_s']>=all_ready and m['start_s']>=all_ready,
                                    both_stages_before_first_landing=s['end_s']<=firstland and m['end_s']<=firstland,
                                    medium_start_s=m['start_s'],medium_end_s=m['end_s'],
                                    stage_start_s=s['start_s'],stage_end_s=s['end_s']))
                added+=1
            audit.append(dict(**info,status='usable_stage_pair' if added else 'no_complete_stage_pair',stage_pairs=added))
    for source,expected in hashes.items():assert sha(ROOT/source)==expected
    return pd.DataFrame(outputs),pd.DataFrame(audit),hashes


def common_low_sensitivity(pairs, full):
    norm=BatteryNormalizer.load(MODEL); rows=[]
    cache={}
    for _,pair in pairs.iterrows():
        records=[];raws=[];curve=norm.curve_for(pair.battery_id,pair.drone_id)
        upper,lower=curve.boundaries[1:3]
        for level in [1,2]:
            source=pair[f'source_lv{level}']
            p=full[full.source.eq(source)&full.drone_id.eq(pair.drone_id)].iloc[0]
            if source not in cache:cache[source]=pd.read_csv(ROOT/source,low_memory=False)
            g=cache[source];g=g[g.drone_name.eq(pair.drone_id)].sort_values('elapsed_time')
            t0=crossing(g,lower)
            before=g[g.elapsed_time.between(t0,p.first_landing_s)] if t0 is not None else g.iloc[:0]
            raws.append((g,p,t0,before))
        base={k:pair[k] for k in KEYS}
        if any(len(before)==0 for g,p,t0,before in raws):
            rows.append(dict(**base,included=False,reason='No Low before first landing'));continue
        endpoint=max(20,max(float(before.battery.min()) for g,p,t0,before in raws))
        if lower-endpoint<10:
            rows.append(dict(**base,included=False,reason='Common pre-landing Low spans less than 10 percentage points'));continue
        values=[]
        for g,p,t0,before in raws:
            t1=crossing(g,endpoint,t0)
            st=stage_stats(g,t0,t1,lower,endpoint,curve.rates_pp_min[2],norm.reference.rates_pp_min[2],p.all_ready_s,p.first_landing_s)
            assert st is not None and st['end_s']<=p.first_landing_s
            values.append(st['k']/p.medium_k)
        rows.append(dict(**base,included=True,reason='Matched battery, matched SOC window; before first landing in both flights',
                         lower_start_soc=lower,end_soc=endpoint,ratio_lv1=values[0],ratio_lv2=values[1],
                         change_pct=100*(values[1]/values[0]-1)))
    return pd.DataFrame(rows)


def main():
    assert not OUT.exists(),'Use a fresh output directory'
    full=pd.read_csv(PRIOR/'paired_ratios.csv')
    prior_manifest=json.loads((PRIOR/'manifest.json').read_text())
    for p,expected in prior_manifest['input_files_sha256'].items():assert sha(ROOT/p)==expected,p
    assert not full.duplicated(KEYS+['wind_level']).any()
    with sqlite3.connect(':memory:') as conn:
        full.to_sql('full_curves',conn,index=False)
        paired=pd.read_sql_query(SQL,conn)
    assert len(paired)==7
    # Own-battery calibration cancels exactly in the between-level ratio test.
    for _,pair in paired.iterrows():
        a=full[full.source.eq(pair.source_lv1)&full.drone_id.eq(pair.drone_id)].iloc[0]
        b=full[full.source.eq(pair.source_lv2)&full.drone_id.eq(pair.drone_id)].iloc[0]
        for stage in ['high','low']:
            raw=100*((b[stage+'_raw_rate_pp_min']/b.medium_raw_rate_pp_min)/(a[stage+'_raw_rate_pp_min']/a.medium_raw_rate_pp_min)-1)
            assert np.isclose(raw,pair[stage+'_change_pct'],atol=1e-7)
    expanded,audit,hashes=scan_partial_pairs()
    repeats=expanded.groupby(KEYS+['wind_level','stage']).agg(curves=('source','nunique'),ratio_min=('ratio','min'),
                     ratio_max=('ratio','max'),ratio_median=('ratio','median')).reset_index()
    repeats=repeats[repeats.curves.gt(1)]
    repeats['range_relative_to_median_pct']=100*(repeats.ratio_max-repeats.ratio_min)/repeats.ratio_median
    expanded_paired=[]
    for key,g in expanded[expanded.wind_level.gt(0)].groupby(KEYS+['stage']):
        if set(g.wind_level)!={1,2}:continue
        a=g[g.wind_level.eq(1)];b=g[g.wind_level.eq(2)]
        expanded_paired.append(dict(**dict(zip(KEYS+['stage'],key)),n_lv1=len(a),n_lv2=len(b),
            ratio_lv1=float(a.ratio.median()),ratio_lv2=float(b.ratio.median()),change_pct=100*(b.ratio.median()/a.ratio.median()-1)))
    expanded_paired=pd.DataFrame(expanded_paired)
    common=common_low_sensitivity(paired,full)
    no_wind=[]
    for _,a in full[full.wind_level.eq(0)].iterrows():
        same=full[(full.formation.eq(a.formation)) & full.spacing_cm.eq(a.spacing_cm) & full.battery_id.eq(a.battery_id)]
        for _,b in same[same.wind_direction.eq(a.wind_direction)&same.wind_level.gt(0)].iterrows():
            no_wind.append(dict(battery_id=a.battery_id,wind_level=int(b.wind_level),source_control=a.source,source_wind=b.source,
                                high_change_pct=100*((b.high_k/b.medium_k)/(a.high_k/a.medium_k)-1),
                                low_change_pct=100*((b.low_k/b.medium_k)/(a.low_k/a.medium_k)-1)))
    clusters=paired.groupby(['wind_direction','formation','spacing_cm']).agg(high_change_pct=('high_change_pct','median'),
        low_change_pct=('low_change_pct','median'),drone_pairs=('drone_id','size')).reset_index()
    details=dict(full_curves=len(full),full_flights=int(full.source.nunique()),
                 wind_counts=full.groupby('wind_level').agg(curves=('source','size'),flights=('source','nunique')).reset_index().to_dict('records'),
                 matched_drone_pairs=len(paired),matched_conditions=len(clusters),
                 high_change=describe(paired.high_change_pct),low_change=describe(paired.low_change_pct),
                 configuration_median_high=describe(clusters.high_change_pct),configuration_median_low=describe(clusters.low_change_pct),
                 primary_repeated_same_condition_battery_cells=0,
                 extended_stage_pairs_by_stage=expanded.groupby('stage').agg(pairs=('source','size'),flights=('source','nunique')).reset_index().to_dict('records'),
                 expanded_repeated_cells=len(repeats),common_prelanding_low=describe(common.loc[common.included,'change_pct']),
                 input_hashes=hashes,source_registry_modified=False,
                 conclusion_scope='Descriptive matched comparisons; non-randomized and insufficient replication to distinguish random variation from wind-dependent effects',
                 aggregation='Primary: one curve per condition/wind-level/battery; stage ratios compared within identical conditions. Five configuration blocks, not seven independent experiments.',
                 metric='100*((r_stage/r_medium)_Lv2/(r_stage/r_medium)_Lv1 - 1); baseline cancels for the matched battery',
                 exclusions='prepare, outlier, synthetic/merged files; known battery pairs; original primary cohort unchanged; expanded study requires only the compared full stages')
    OUT.mkdir(parents=True)
    for name,df in [('matched_full_curves',paired),('condition_blocks',clusters),('expanded_stage_pairs',expanded),
                    ('expanded_source_audit',audit),('repeated_cells',repeats),('expanded_matched_pairs',expanded_paired),
                    ('common_prelanding_low',common),('no_wind_comparisons',pd.DataFrame(no_wind))]:
        df.to_csv(OUT/(name+'.csv'),index=False,float_format='%.12g')
    (OUT/'summary.json').write_text(json.dumps(details,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (OUT/'matched_comparison.sql').write_text(SQL+';\n')
    print(json.dumps({k:v for k,v in details.items() if k!='input_hashes'},ensure_ascii=False,indent=2))
    print('EXPANDED MATCHED',expanded_paired.round(3).to_string(index=False))
    print('REPEATS',repeats.round(3).to_string(index=False))
    print('COMMON LOW',common.round(3).to_string(index=False))


if __name__=='__main__':main()
