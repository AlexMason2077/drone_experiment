"""Offline empirical wind-tunnel simulator components; never imports aircraft SDKs."""
from pathlib import Path
import hashlib
import itertools
import json
import math
import re
import sys
import argparse

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from battery_normalization import BatteryNormalizer

BASELINE_PATH=ROOT/'analysis_results/battery_normalization_v3_with_b15_20260909/model.json'
OUTPUT=ROOT/'analysis_results/wind_tunnel_simulator_20260909'
BATTERIES={1:'B11',2:'B10',3:'B13',4:'B14',5:'B12'}
STAGES=('high','medium','low')
FORMATIONS=('front','column','vee','echelon','diamond')
WINDS=('head','tail','side')
BASE_SEED=20260909
READ_COLS=['run_id','drone_name','battery_id','phase','elapsed_time','battery','formation','wind_direction','wind_speed',
 'inter_drone_distance_cm','mid','mission_pad','x','y','h','target_x','target_y','target_z','pitch','roll','yaw','templ','temph',
 'vgx','vgy','vgz','agx','agy','agz','tof','baro','motor_time','z','mission_pad_pitch','mission_pad_roll','mission_pad_yaw']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def stage_of(soc,boundaries):
    return 0 if soc>boundaries[1] else (1 if soc>boundaries[2] else 2)


def condition_id(formation,spacing,wind,level):
    return f'{formation}_{int(spacing)}_{wind}_lv{int(level)}'


def normalize_formation(x):
    return str(x).strip().lower().replace('echalon','echelon').replace('echolon','echelon')


def extract():
    """Observed one-percentage-point dwell intervals, not interpolated trajectories."""
    norm=BatteryNormalizer.load(BASELINE_PATH)
    reg_path=ROOT/'database/experiment_registry.json'
    registry={r['experiment_id']:r for r in json.loads(reg_path.read_text())['experiments']}
    hashes={str(BASELINE_PATH.relative_to(ROOT)):sha(BASELINE_PATH),str(reg_path.relative_to(ROOT)):sha(reg_path)}
    steps=[];curves=[];audit=[];inventory=[];seen=set()
    paths=sorted((ROOT/'database').glob('wind_tunnel*/*_all_coordination.csv'))
    for p in paths:
        source=str(p.relative_to(ROOT));exp=p.parent.name;rec=registry.get(exp,{})
        hashes[source]=sha(p)
        excluded=('prepare' in exp.lower() or 'practice' in exp.lower() or 'merged' in p.name.lower() or rec.get('is_outlier'))
        if excluded:
            audit.append(dict(source=source,reason='prepare_practice_merged_outlier'));continue
        d=pd.read_csv(p,usecols=lambda c:c in READ_COLS,low_memory=False)
        if d.empty or not set(['run_id','phase','elapsed_time','battery','drone_name','battery_id']).issubset(d):
            audit.append(dict(source=source,reason='empty_missing_columns'));continue
        if d.phase.str.contains('simulat|merged|interpol',case=False,na=False).any():
            audit.append(dict(source=source,reason='synthetic_phase'));continue
        for col in set(READ_COLS)-{'run_id','drone_name','battery_id','phase','formation','wind_direction','wind_speed'}:
            if col in d:d[col]=pd.to_numeric(d[col],errors='coerce')
        formation=normalize_formation(rec.get('formation',d.formation.iloc[0]))
        wind=str(rec.get('wind_direction',d.wind_direction.iloc[0])).lower().replace(' wind','')
        lv=re.search(r'(\d+)',str(rec.get('wind_speed',d.wind_speed.iloc[0])))
        level=int(lv.group(1)) if lv else 0
        if 'no_wind' in exp:level=0
        spacing=float(rec.get('inter_drone_distance_cm',d.inter_drone_distance_cm.iloc[0]))
        if formation not in FORMATIONS or wind not in WINDS or spacing not in (50,75) or level not in (0,1,2):
            audit.append(dict(source=source,reason='unsupported_condition'));continue
        cid=condition_id(formation,spacing,wind,level)
        meta=dict(source=source,experiment_id=exp,condition_id=cid,formation=formation,wind_direction=wind,
                  wind_level=level,spacing_cm=spacing)
        ready=d[d.phase.eq('wind_tunnel_hover')].groupby('drone_name').elapsed_time.min()
        inventory.append(dict(**meta,drones_entered_hover=len(ready),rows=len(d),status='observed_record'))
        # Partial failed attempts stay in inventory, but do not calibrate stable swarm simulations.
        if len(ready)!=5:
            audit.append(dict(**meta,reason='not_all_five_entered_hover'));continue
        land=d[d.phase.str.contains('land',case=False,na=False)]
        first_landing=float(land.elapsed_time.min()) if len(land) else math.inf
        landing_times=land.groupby('drone_name').elapsed_time.min().to_dict()
        for (drone,battery),g in d.groupby(['drone_name','battery_id']):
            info=dict(**meta,drone_id=drone,battery_id=battery,run_id=str(g.run_id.iloc[0]))
            if battery not in norm.curves or norm.pairs[battery]!=drone:
                audit.append(dict(**info,reason='no_matching_individual_battery_calibration'));continue
            uid=(info['run_id'],drone,battery)
            if uid in seen:
                audit.append(dict(**info,reason='duplicate_run_drone'));continue
            seen.add(uid)
            g=g.sort_values('elapsed_time').copy()
            take=g[g.phase.eq('wind_tunnel_takeoff')]
            if take.empty:continue
            take_s=float(take.elapsed_time.iloc[0]);g=g[g.elapsed_time.ge(take_s)].reset_index(drop=True)
            if g[['elapsed_time','battery']].isna().any().any() or g.elapsed_time.duplicated().any() or not g.battery.between(0,100).all():
                audit.append(dict(**info,reason='invalid_time_soc'));continue
            ownland=g[g.phase.str.contains('land',case=False,na=False)]
            if len(ownland):g=g[g.elapsed_time.le(ownland.elapsed_time.iloc[0])].reset_index(drop=True)
            drops=np.flatnonzero(g.battery.diff().to_numpy()<0)
            if not len(drops):continue
            first=g.iloc[drops[0]]
            bc=norm.curve_for(battery,drone);position=int(drone.split('_')[-1])
            initial=float(g.battery.iloc[0]);first_drop_time=float(first.elapsed_time)
            curve=dict(**info,position=position,initial_soc=initial,first_drop_soc=float(first.battery),
                first_drop_s=first_drop_time,initial_plateau_s=first_drop_time-take_s,
                all_ready_s=float(ready.max()),own_ready_s=float(ready[drone]),last_soc=float(g.battery.iloc[-1]),
                last_s=float(g.elapsed_time.iloc[-1]),target_height_cm=float(g.target_z.median()) if 'target_z' in g else np.nan)
            count=0
            for a,b in zip(drops[:-1],drops[1:]):
                s=g.iloc[a:b+1];start=s.iloc[0];end=s.iloc[-1]
                drop=float(start.battery-end.battery);dt=round(float(end.elapsed_time-start.elapsed_time),9)
                if drop!=1 or not .5<=dt<=90 or len(s)<3:continue
                if s.elapsed_time.diff().max()>5 or s.battery.diff().max()>0:continue
                if s.phase.iloc[:-1].str.contains('fault|recoverable_error|uncommanded|land',case=False,na=False).any():continue
                soc=(float(start.battery)+float(end.battery))/2
                if not 20<=soc<=100:continue
                stage=stage_of(soc,bc.boundaries)
                own=s.mid.eq(s.mission_pad)&s.mid.gt(0)
                near=own&s.x.abs().le(25)&s.y.abs().le(25)
                own_fraction=float(own.mean());near_fraction=float(near.mean())
                height=float(s.h.median())
                before_landing=float(end.elapsed_time)<=first_landing
                # Pose quality is explicit, not inferred from a nice-looking SOC curve.
                fit_ok=own_fraction>=.5 and near_fraction>=.5 and 30<=height<=140
                active_count=sum(float(landing_times.get(f'drone_{j}',math.inf))>float(start.elapsed_time) for j in range(1,6))
                row=dict(**info,position=position,stage=STAGES[stage],stage_index=stage,soc=soc,
                    soc_start=float(start.battery),soc_end=float(end.battery),duration_s=dt,
                    start_s=float(start.elapsed_time),end_s=float(end.elapsed_time),initial_soc=initial,
                    drops_since_start=initial-soc,seconds_since_first_drop=float(start.elapsed_time)-first_drop_time,
                    stage_progress=(bc.boundaries[stage]-soc)/(bc.boundaries[stage]-bc.boundaries[stage+1]),
                    baseline_dwell_s=60/bc.rates_pp_min[stage],baseline_rate_pp_min=bc.rates_pp_min[stage],
                    raw_rate_pp_min=60/dt,own_pad_fraction=own_fraction,near25_fraction=near_fraction,height_cm=height,
                    before_all_ready=float(start.elapsed_time)<float(ready.max()),before_first_landing=before_landing,
                    active_count=active_count,
                    fit_eligible=fit_ok,observed=True,n_samples=len(s))
                steps.append(row);count+=1
            curve['eligible_step_intervals']=count;curves.append(curve)
    frame=pd.DataFrame(steps)
    counts=frame[frame.fit_eligible].groupby(['source','drone_id']).size()
    frame['curve_eligible_count']=[int(counts.get((a,b),0)) for a,b in zip(frame.source,frame.drone_id)]
    frame['fit_eligible'] &= frame.curve_eligible_count.ge(5)
    assert not frame.duplicated(['source','drone_id','start_s','end_s']).any()
    for p,h in hashes.items():assert sha(ROOT/p)==h,p
    return frame,pd.DataFrame(curves),pd.DataFrame(inventory),pd.DataFrame(audit),hashes


def main_extract():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'observed_steps.csv').exists():raise FileExistsError('Do not overwrite extraction')
    steps,curves,inventory,audit,hashes=extract()
    for name,frame in [('observed_steps',steps),('observed_curves',curves),('record_inventory',inventory),('source_exclusions',audit)]:
        frame.to_csv(OUTPUT/(name+'.csv'),index=False,float_format='%.12g')
    manifest=dict(input_sha256=hashes,scope='wind_tunnel_hover_only',normalizer_version=BatteryNormalizer.load(BASELINE_PATH).version,
                  original_records_modified=False,simulation_not_experimental_replicates=True,
                  raw_step_count=len(steps),fit_step_count=int(steps.fit_eligible.sum()),
                  fit_quality='Own pad and within25cm fractions >=0.5, median height30..140cm; >=5 valid steps per curve; active-hovering count recorded as other drones begin landing',
                  field_policy='Integer measured SOC step dwells; no interpolation; initial plateau model separate; no aircraft SDK imported')
    (OUTPUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    fit=steps[steps.fit_eligible]
    print('Steps',len(steps),'fit',len(fit),'flights',fit.source.nunique(),'conditions',fit.condition_id.nunique())
    print(fit.groupby(['spacing_cm','wind_direction','wind_level']).agg(conditions=('condition_id','nunique'),flights=('source','nunique'),steps=('source','size')).to_string())


def features(d, condition=True):
    """No measured future time, arrival SOC, or measured attitude is a predictor."""
    result=[]
    for r in d.to_dict('records'):
        soc=r['soc']/100
        q={'battery':r['battery_id'],'stage':r['stage'],
           'battery_stage':r['battery_id']+'_'+r['stage'],
           'soc':soc,'soc2':soc*soc,'initial_soc':r['initial_soc']/100,
           'stage_progress':r['stage_progress'],
           'startup':math.exp(-r['drops_since_start']/5),
           'startup_long':math.exp(-r['drops_since_start']/15),
           'active_fraction':r['active_count']/5}
        for knot in (.4,.55,.7,.8,.9):q[f'hinge_{knot}']=max(0,soc-knot)
        if condition:
            q.update(formation=r['formation'],wind=r['wind_direction'],
                     level=str(int(r['wind_level'])),spacing=r['spacing_cm']/75,
                     formation_position=f"{r['formation']}_{r['position']}",
                     wind_formation=f"{r['wind_direction']}_{r['formation']}",
                     wind_position_stage=f"{r['wind_direction']}_{r['position']}_{r['stage']}",
                     wind_level=f"{r['wind_direction']}_{r['wind_level']}")
        result.append(q)
    return result


def weights(d):
    # A long trace is not hundreds of independent experimental replicates.
    count=d.groupby(['source','drone_id','stage']).duration_s.transform('size')
    w=1/count.to_numpy();return w/w.mean()


def fit_predictor(d, name):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    if name=='battery_three_stage':return {'name':name}
    cond=name!='battery_soc_only'
    vec=DictVectorizer(sparse=False)
    x=vec.fit_transform(features(d,cond))
    y=np.log(d.duration_s.to_numpy()/d.baseline_dwell_s.to_numpy())
    model=Ridge(alpha=100) if name=='condition_ridge' else HistGradientBoostingRegressor(
        max_iter=140,max_leaf_nodes=8,min_samples_leaf=45,l2_regularization=15,
        learning_rate=.06,early_stopping=False,random_state=BASE_SEED)
    model.fit(x,y,sample_weight=weights(d))
    return {'name':name,'vec':vec,'model':model,'condition':cond}


def predict_dwell(model,d):
    baseline=d.baseline_dwell_s.to_numpy()
    if model['name']=='battery_three_stage':return baseline
    pred=model['model'].predict(model['vec'].transform(features(d,model['condition'])))
    return np.clip(baseline*np.exp(pred),.5,60)


def error_metrics(d,pred):
    z=d[['source','drone_id','stage','duration_s']].copy();z['prediction_s']=pred
    # Evaluate time for the SAME eligible measured SOC steps, not nominal whole stages.
    a=z.groupby(['source','drone_id','stage'])[['duration_s','prediction_s']].sum()
    ape=(a.prediction_s/a.duration_s-1).abs()*100
    return dict(step_mae_s=float(np.average(abs(pred-d.duration_s),weights=weights(d))),
                stage_time_mape_pct=float(ape.mean()),stage_time_median_ape_pct=float(ape.median()),
                stage_time_bias_pct=float(((a.prediction_s/a.duration_s-1)*100).mean()),
                stage_segments=len(a),flights=d.source.nunique(),conditions=d.condition_id.nunique())


def train():
    import joblib
    from sklearn.model_selection import GroupKFold
    d=pd.read_csv(OUTPUT/'observed_steps.csv');d=d[d.fit_eligible].reset_index(drop=True)
    # One deterministic, untouched holdout. All five drones and repeat flights stay together.
    ids=sorted(d.condition_id.unique());rng=np.random.default_rng(BASE_SEED)
    positive=[c for c in ids if not c.endswith('lv0')]
    held=sorted(rng.choice(positive,size=6,replace=False).tolist())
    dev=d[~d.condition_id.isin(held)].copy();test=d[d.condition_id.isin(held)].copy()
    split={'seed':BASE_SEED,'holdout_conditions':held,'development_conditions':sorted(dev.condition_id.unique()),
           'group_key':'condition_id; every run and all drones stay in one partition',
           'selection_metric':'mean held-out-condition-fold stage-time MAPE; equal curve-stage weight',
           'warning':'Held-out fields active_count and initial_soc are observed covariates; this is conditional dwell-time validation, not a fully closed-loop physical-flight validation.'}
    (OUTPUT/'split_manifest.json').write_text(json.dumps(split,indent=2)+'\n')
    names=['battery_three_stage','battery_soc_only','condition_ridge','condition_tree']
    rows=[];oofs={};folds=list(GroupKFold(n_splits=5).split(dev,groups=dev.condition_id))
    for name in names:
        pred=np.zeros(len(dev));print('Model',name,flush=True)
        for k,(a,b) in enumerate(folds):
            m=fit_predictor(dev.iloc[a],name);p=predict_dwell(m,dev.iloc[b]);pred[b]=p
            rows.append(dict(model=name,fold=k,**error_metrics(dev.iloc[b],p)))
        oofs[name]=pred
    scores=pd.DataFrame(rows);scores.to_csv(OUTPUT/'development_cv.csv',index=False)
    ranking=scores.groupby('model').stage_time_mape_pct.mean().sort_values()
    selected=ranking.index[0];print('Selected',selected,ranking.to_dict(),flush=True)
    oof=dev.copy();oof['predicted_dwell_s']=oofs[selected]
    oof['log_residual']=np.log(oof.duration_s/oof.predicted_dwell_s)
    oof.to_csv(OUTPUT/'development_oof.csv',index=False)
    # Residual interval calibrated solely on development out-of-fold errors.
    lower,upper=np.quantile(oof.log_residual,[.05,.95])
    chosen=fit_predictor(dev,selected);p=predict_dwell(chosen,test)
    validation=[]
    # Baseline comparisons are specified beforehand, not used to reselect on the holdout.
    for name in names:
        m=chosen if name==selected else fit_predictor(dev,name)
        pred=predict_dwell(m,test);validation.append(dict(model=name,selected=name==selected,**error_metrics(test,pred)))
    pd.DataFrame(validation).to_csv(OUTPUT/'holdout_model_metrics.csv',index=False)
    test['predicted_dwell_s']=p
    test['interval90_low_s']=p*np.exp(lower);test['interval90_high_s']=p*np.exp(upper)
    test['interval90_contains_observed']=test.duration_s.between(test.interval90_low_s,test.interval90_high_s)
    test.to_csv(OUTPUT/'holdout_predictions.csv',index=False)
    by_stage=[]
    for stage,g in test.groupby('stage'):
        by_stage.append(dict(stage=stage,**error_metrics(g,g.predicted_dwell_s),
                             interval90_coverage=float(g.interval90_contains_observed.mean())))
    pd.DataFrame(by_stage).to_csv(OUTPUT/'holdout_stage_metrics.csv',index=False)
    joblib.dump(chosen,OUTPUT/'validation_model.joblib')
    production=fit_predictor(d,selected);joblib.dump(production,OUTPUT/'simulator_model.joblib')
    summary=dict(selected_model=selected,development_cv_mean_stage_mape_pct=float(ranking.iloc[0]),
        holdout_metrics=next(x for x in validation if x['selected']),
        holdout_interval90_coverage=float(test.interval90_contains_observed.mean()),
        interval_log_residual_quantiles=[float(lower),float(upper)],fit_steps=len(d),
        fit_flights=d.source.nunique(),fit_conditions=d.condition_id.nunique(),
        production_refit='All eligible observations AFTER locked holdout evaluation; production model itself has no independent test left.',
        simulation_semantics='Conditional empirical surrogate, not aerodynamic physics or independent real experiments.')
    (OUTPUT/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['extract','train'],nargs='?',default='extract')
    args=parser.parse_args()
    main_extract() if args.action=='extract' else train()
