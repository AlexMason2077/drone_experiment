"""Test stage-invariant relative discharge factors in complete wind-tunnel traces.

Inputs are read-only. Own-battery SOC boundaries and baseline slopes are frozen.
Each comparison is within one source flight, drone, battery and position.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from battery_normalization import BatteryNormalizer

MODEL=ROOT/'analysis_results/battery_normalization_v3_with_b15_20260909/model.json'
OUT=ROOT/'analysis_results/wind_tunnel_stage_ratio_check_20260909'
STAGES=['high','medium','low']
FIELDS=['run_id','experiment_id','drone_name','battery_id','phase','elapsed_time','battery',
        'formation','wind_direction','wind_speed','inter_drone_distance_cm','mid','mission_pad','x','y','h','tof']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def crossing(g,level,start=-math.inf):
    after=g[(g.elapsed_time>=start)&(g.battery<=level)]
    if after.empty or float(after.battery.iloc[0])!=level:return None
    return float(after.elapsed_time.iloc[0])


def stage_stats(g,start,end,upper,lower,baseline_rate,reference_rate,all_ready,first_landing):
    s=g[g.elapsed_time.between(start,end)].copy()
    dt=end-start
    if dt<10 or len(s)<10:return None
    t=s.elapsed_time.to_numpy(float)-start;y=s.battery.to_numpy(float)
    if np.diff(t).max()>5 or abs(np.diff(y)).max()>2 or np.diff(y).max()>1:return None
    raw=(upper-lower)*60/dt
    k=raw/baseline_rate
    slope,intercept=np.polyfit(t,y,1)
    own=pd.to_numeric(s.mid,errors='coerce').eq(pd.to_numeric(s.mission_pad,errors='coerce'))
    own=own & pd.to_numeric(s.mid,errors='coerce').gt(0)
    near=own & pd.to_numeric(s.x,errors='coerce').abs().le(20)&pd.to_numeric(s.y,errors='coerce').abs().le(20)
    return dict(soc_upper=upper,soc_lower=lower,start_s=start,end_s=end,duration_s=dt,
        observed_drop_pp=upper-lower,raw_rate_pp_min=raw,baseline_rate_pp_min=baseline_rate,k=k,
        Bideal_rate_pp_min=k*reference_rate,ols_raw_rate_pp_min=float(-slope*60),
        ols_k=float(-slope*60/baseline_rate),ols_rmse_pp=float(np.sqrt(np.mean((y-(intercept+slope*t))**2))),
        n=len(s),max_gap_s=float(np.diff(t).max()),upward_steps=int((np.diff(y)>0).sum()),
        own_pad_fraction=float(own.mean()),own_pad_near20_fraction=float(near.mean()),
        h_median_cm=float(s.h.median()),tof_median_cm=float(s.tof.median()),
        starts_after_all_ready=bool(start>=all_ready),ends_before_first_landing=bool(end<=first_landing),
        after_first_landing_fraction=max(0,end-max(start,first_landing))/dt,
        before_all_ready_fraction=max(0,min(end,all_ready)-start)/dt)


def summary(group,label):
    result=dict(group=label,drone_curves=len(group),flights=int(group.source.nunique()) if len(group) else 0)
    if not len(group):return result
    for band in ['high','low']:
        x=group[f'{band}_difference_pct']
        result.update({f'{band}_median_signed_pct':float(x.median()),
            f'{band}_median_absolute_pct':float(x.abs().median()),
            f'{band}_mean_absolute_pct':float(x.abs().mean()),
            f'{band}_signed_q25_pct':float(x.quantile(.25)),f'{band}_signed_q75_pct':float(x.quantile(.75)),
            f'{band}_min_pct':float(x.min()),f'{band}_max_pct':float(x.max()),
            f'{band}_within10_count':int(x.abs().le(10).sum()),f'{band}_within20_count':int(x.abs().le(20).sum()),
            f'{band}_ols_median_signed_pct':float(group[f'{band}_ols_difference_pct'].median())})
    result['both_within10_count']=int((group.high_difference_pct.abs().le(10)&group.low_difference_pct.abs().le(10)).sum())
    result['both_within20_count']=int((group.high_difference_pct.abs().le(20)&group.low_difference_pct.abs().le(20)).sum())
    return result


def run():
    if OUT.exists():raise FileExistsError('Use a new output directory')
    norm=BatteryNormalizer.load(MODEL)
    registry_path=ROOT/'database/experiment_registry.json'
    registry={r['experiment_id']:r for r in json.loads(registry_path.read_text())['experiments']}
    paths=sorted((ROOT/'database').glob('wind_tunnel*/*_all_coordination.csv'))
    audit=[]; stages=[]; full=[]; hashes={str(MODEL.relative_to(ROOT)):sha(MODEL),str(registry_path.relative_to(ROOT)):sha(registry_path)}
    seen=set()
    for counter,p in enumerate(paths,1):
        exp=p.parent.name;rec=registry.get(exp,{})
        basic=dict(experiment_id=exp,source=str(p.relative_to(ROOT)))
        if 'prepare' in exp.lower() or 'merged' in p.name.lower() or rec.get('is_outlier'):
            audit.append(dict(**basic,status='excluded_prepare_merged_or_outlier'));continue
        hashes[basic['source']]=sha(p)
        d=pd.read_csv(p,usecols=lambda c:c in FIELDS,low_memory=False)
        if d.empty or not {'phase','battery','elapsed_time'}.issubset(d):
            audit.append(dict(**basic,status='empty_or_missing_columns'));continue
        if d.phase.str.contains('merge|simulat|interpol',case=False,na=False).any():
            audit.append(dict(**basic,status='synthetic_phase'));continue
        for c in ['elapsed_time','battery','x','y','h','tof']:
            d[c]=pd.to_numeric(d[c],errors='coerce')
        ready=d[d.phase.eq('wind_tunnel_hover')].groupby('drone_name').elapsed_time.min()
        if len(ready)!=5:
            audit.append(dict(**basic,status='not_five_hovering'));continue
        all_ready=float(ready.max())
        landing=d[d.phase.str.contains('land',case=False,na=False)]
        first_landing=float(landing.elapsed_time.min()) if len(landing) else math.inf
        for (drone,battery),g in d.groupby(['drone_name','battery_id']):
            info=dict(**basic,drone_id=drone,battery_id=battery,run_id=str(g.run_id.iloc[0]))
            if battery not in norm.curves or norm.pairs[battery]!=drone:
                audit.append(dict(**info,status='unsupported_battery_pair'));continue
            ident=(info['run_id'],drone,battery)
            if ident in seen:
                audit.append(dict(**info,status='duplicate_run_drone'));continue
            seen.add(ident)
            curve=norm.curve_for(battery,drone)
            g=g[g.elapsed_time.ge(float(ready[drone]))].sort_values('elapsed_time').copy()
            if g[['elapsed_time','battery']].isna().any().any() or not g.battery.between(0,100).all():
                audit.append(dict(**info,status='invalid_time_or_soc'));continue
            if g.duplicated('elapsed_time').any():
                audit.append(dict(**info,status='duplicate_time'));continue
            own_land=g[g.phase.str.contains('land',case=False,na=False)]
            if len(own_land):g=g[g.elapsed_time.le(float(own_land.elapsed_time.min()))]
            levels=[95,curve.boundaries[1],curve.boundaries[2],20]
            times=[]
            for lev in levels:
                times.append(crossing(g,lev,times[-1] if times and times[-1] is not None else -math.inf))
            info.update(start_hover_soc=float(g.battery.iloc[0]),last_airborne_soc=float(g.battery.iloc[-1]),
                has_all_crossings=all(t is not None for t in times))
            if not info['has_all_crossings']:
                audit.append(dict(**info,status='incomplete_95_to_20_after_hover_start'));continue
            span=g[g.elapsed_time.between(times[0],times[-1])]
            if span.phase.str.contains('fault|recoverable_error|uncommanded',case=False,na=False).any():
                audit.append(dict(**info,status='control_fault_during_full_curve'));continue
            formation=str(rec.get('formation',g.formation.iloc[0])).lower().replace('echalon','echelon').replace('echolon','echelon')
            wind=str(rec.get('wind_direction',g.wind_direction.iloc[0])).lower().replace(' wind','')
            level_str=str(rec.get('wind_speed',g.wind_speed.iloc[0])).lower()
            level_match=re.search(r'(\d+)',level_str)
            level=int(level_match.group(1)) if level_match else 0
            if 'no_wind' in exp:level=0
            info.update(formation=formation,wind_direction=wind,wind_level=level,
                spacing_cm=float(rec.get('inter_drone_distance_cm',g.inter_drone_distance_cm.iloc[0])),
                position=int(drone.split('_')[-1]),all_ready_s=all_ready,first_landing_s=first_landing)
            parts=[]
            for i,band in enumerate(STAGES):
                stat=stage_stats(g,times[i],times[i+1],levels[i],levels[i+1],curve.rates_pp_min[i],
                                 norm.reference.rates_pp_min[i],all_ready,first_landing)
                if stat is None:break
                parts.append(dict(**info,stage=band,**stat))
            if len(parts)!=3:
                audit.append(dict(**info,status='stage_telemetry_or_duration_quality_failure'));continue
            audit.append(dict(**info,status='included_full_curve'))
            stages.extend(parts)
            item=dict(**info)
            for part in parts:
                band=part['stage']
                for k in ['raw_rate_pp_min','k','Bideal_rate_pp_min','duration_s','ols_k','own_pad_fraction','own_pad_near20_fraction',
                          'h_median_cm','tof_median_cm','after_first_landing_fraction','before_all_ready_fraction']:
                    item[band+'_'+k]=part[k]
            for band in ['high','low']:
                item[band+'_difference_pct']=100*(item[band+'_k']/item['medium_k']-1)
                item[band+'_ols_difference_pct']=100*(item[band+'_ols_k']/item['medium_ols_k']-1)
            item['same_swarm_entire_curve']=times[0]>=all_ready and times[-1]<=first_landing
            item['low_ends_before_first_landing']=times[-1]<=first_landing
            item['high_starts_after_all_ready']=times[0]>=all_ready
            item['all_stages_own_pad_near20_ge80']=all(p['own_pad_near20_fraction']>=.8 for p in parts)
            # Sensitivity: drop end-of-flight landing changes by stopping Low at 30%.
            t30=crossing(g,30,times[2])
            if t30 is not None and times[2]<t30<=first_landing:
                shorter=stage_stats(g,times[2],t30,levels[2],30,curve.rates_pp_min[2],norm.reference.rates_pp_min[2],all_ready,first_landing)
                if shorter:
                    item['low_to30_k']=shorter['k']
                    item['low_to30_difference_pct']=100*(shorter['k']/item['medium_k']-1)
            full.append(item)
        if counter%25==0:print(f'Checked {counter}/{len(paths)} files',flush=True)
    full=pd.DataFrame(full);stages=pd.DataFrame(stages);audit=pd.DataFrame(audit)
    assert not full.duplicated(['source','drone_id']).any()
    assert len(stages)==3*len(full)
    groups=[summary(full,'all_complete'),summary(full[full.wind_level.gt(0)],'with_wind'),
        summary(full[full.wind_level.eq(0)],'no_wind'),
        summary(full[full.same_swarm_entire_curve],'all_five_ready_and_airborne'),
        summary(full[full.all_stages_own_pad_near20_ge80],'own_pad_near20_ge80')]
    batteries=[summary(g,b) for b,g in full[full.wind_level.gt(0)].groupby('battery_id')]
    conditions=[dict(**dict(zip(['wind_direction','wind_level','formation','spacing_cm'],key)),**summary(g,'condition'))
        for key,g in full.groupby(['wind_direction','wind_level','formation','spacing_cm'])]
    # Within-flight position shares: unnormalized / baseline-normalized share is
    # a separate question from absolute stage-to-Medium load-factor equality.
    shares=[]
    for source,g in full.groupby('source'):
        if len(g)<2:continue
        mshare=g.medium_k/g.medium_k.sum()
        for band in ['high','low']:
            stage_share=g[band+'_k']/g[band+'_k'].sum()
            for idx in g.index:
                shares.append(dict(source=source,experiment_id=g.loc[idx,'experiment_id'],battery_id=g.loc[idx,'battery_id'],
                    drone_id=g.loc[idx,'drone_id'],stage=band,compared_drones=len(g),
                    medium_share=float(mshare.loc[idx]),stage_share=float(stage_share.loc[idx]),
                    share_change_percentage_points=float((stage_share.loc[idx]-mshare.loc[idx])*100)))
    for p,expected in hashes.items():assert sha(ROOT/p)==expected,p
    OUT.mkdir(parents=True)
    for name,df in [('source_audit',audit),('stage_rates',stages),('paired_ratios',full),
                    ('summary',pd.DataFrame(groups)),('by_battery',pd.DataFrame(batteries)),
                    ('by_condition',pd.DataFrame(conditions)),('position_shares',pd.DataFrame(shares))]:
        df.to_csv(OUT/(name+'.csv'),index=False,float_format='%.12g')
    manifest=dict(model_version=norm.version,model_sha256=norm.sha256,input_files_sha256=hashes,
        file_count=len(paths),counts=audit.status.value_counts().to_dict(),
        main_formula='k_stage = observed stage rate / own-battery frozen baseline stage rate; deviation = (k_stage/k_medium - 1)*100',
        observed_stage_rate='Observed SOC drop / first-crossing elapsed seconds * 60; exact boundary hits; no interpolation',
        full_curve_range='95 to 20 after own hover start, before own first landing; own battery knots',
        Bideal_kept_fixed=True,sources_modified=False,comparison_scope='Wind tunnel only; not proof for 2.5m forward flight',
        known_limitations=['Only complete curves retained: selection bias possible','Some curves extend beyond another drone landing: sensitivity provided',
            'Wind start time/control changes may not be fully logged','95..20 validation avoids modeled 100..95 extension',
            'No stage coefficient or SOC knot was refitted on these wind-tunnel runs'],summary=groups)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('SOURCE AUDIT',manifest['counts'])
    print(pd.DataFrame(groups).round(2).to_string(index=False))
    print('BY BATTERY');print(pd.DataFrame(batteries).round(2).to_string(index=False))
    print('FULL FIVE DRONE FILES',full.groupby('experiment_id').size()[lambda x:x.eq(5)].to_dict())
    print('LOW TO 30 SENSITIVITY',full.low_to30_difference_pct.describe().round(2).to_dict())
    return OUT


if __name__=='__main__':run()
