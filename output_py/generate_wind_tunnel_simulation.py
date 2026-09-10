"""Generate separately labelled synthetic SOC logger files. No aircraft connections."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import sys

import joblib
import numpy as np
import pandas as pd

from wind_tunnel_simulation_model import (ROOT,OUTPUT,BASELINE_PATH,BATTERIES,STAGES,
    FORMATIONS,WINDS,BASE_SEED,BatteryNormalizer,condition_id,stage_of,predict_dwell,sha)

DEST=ROOT/'simulation_data/wind_tunnel_empirical_v1_20260909'
MODEL_VERSION='wind_tunnel_empirical_v1_20260909'
TELEMETRY_COLUMNS=['mid','x','y','z','X_global','Y_global','Z_global',
 'position_error_x','position_error_y','position_error_z','position_error_dist',
 'mean_spacing_error','max_spacing_error','yaw','pitch','roll',
 'mission_pad_pitch','mission_pad_roll','mission_pad_yaw','vgx','vgy','vgz',
 'agx','agy','agz','templ','temph','tof','h','baro','motor_time']


def coverage():
    d=pd.read_csv(OUTPUT/'observed_steps.csv');fit=d[d.fit_eligible]
    inventory=pd.read_csv(OUTPUT/'record_inventory.csv');rows=[];cells=[]
    for f,s,w,l in __import__('itertools').product(FORMATIONS,(50,75),WINDS,(1,2)):
        cid=condition_id(f,s,w,l);g=fit[fit.condition_id.eq(cid)];complete=0
        for j,battery in BATTERIES.items():
            for stage in STAGES:
                z=g[g.position.eq(j)&g.battery_id.eq(battery)&g.stage.eq(stage)]
                supported=len(z)>=5
                complete+=supported
                cells.append(dict(condition_id=cid,position=j,battery_id=battery,stage=stage,
                    observed_steps=len(z),observed_flights=z.source.nunique(),
                    local_stage_supported=supported,whole_stage_observed=False))
        status='locally_supported_all_15_cells' if complete==15 else ('partial' if len(g) else 'missing_usable_curves')
        nrec=inventory[inventory.condition_id.eq(cid)].source.nunique()
        rows.append(dict(condition_id=cid,formation=f,spacing_cm=s,wind_direction=w,wind_level=l,
            observed_record_files=nrec,usable_flights=g.source.nunique(),usable_steps=len(g),
            supported_cells=complete,total_cells=15,coverage=status,
            generate_missing_or_partial=complete<15))
    return pd.DataFrame(rows),pd.DataFrame(cells)


def missing_allowed_conditions(cov):
    """Latest user policy: any usable real curve excludes the whole condition."""
    frozen=ROOT/'frozen_data/discharge_rates/medium_bideal_v1_20260908/manifest.json'
    unsafe={condition_id(f,s,w,l) for w,l,f,s in json.loads(frozen.read_text())['unsafe_condition_cells']}
    return cov[cov.usable_flights.eq(0)&~cov.condition_id.isin(unsafe)].copy()


class NoiseBank:
    """Bootstrap OOF prediction errors, retaining flight and cross-stage structure."""
    def __init__(self):
        self.d=pd.read_csv(OUTPUT/'development_oof.csv')
        d=self.d
        # Short/one-drone fragments cannot determine an entire swarm's offset.
        size=d.groupby('source').agg(steps=('source','size'),drones=('drone_id','nunique'))
        eligible=size[(size.steps>=100)&(size.drones>=3)].index
        self.flight=d[d.source.isin(eligible)].groupby('source').log_residual.median()
        all_flight=d.groupby('source').log_residual.median()
        d['flight_component']=d.source.map(all_flight)
        cs_size=d.groupby(['source','drone_id','stage']).size()
        self.cs=d.groupby(['source','drone_id','stage']).log_residual.median()
        d['local_component']=d.log_residual-[self.cs.loc[(a,b,c)] for a,b,c in zip(d.source,d.drone_id,d.stage)]
        d['stage_sample_count']=[cs_size.loc[(a,b,c)] for a,b,c in zip(d.source,d.drone_id,d.stage)]
        self.offsets={}
        for battery in BATTERIES.values():
            q=d[d.battery_id.eq(battery)&d.stage_sample_count.ge(5)]
            piv=q.groupby(['source','stage']).log_residual.median().unstack('stage').reindex(columns=STAGES)
            self.offsets[battery]=piv.sub(all_flight.reindex(piv.index),axis=0).to_numpy()
        self.local={}
        self.bounds={stage:tuple(d[d.stage.eq(stage)].duration_s.quantile([.005,.995])) for stage in STAGES}
        for stage in STAGES:
            chunks=[]
            for _,g in d[d.stage.eq(stage)&d.stage_sample_count.ge(5)].groupby(['source','drone_id']):
                g=g.sort_values('start_s')
                # A block never bridges an excluded telemetry interval.
                breaks=g.start_s.sub(g.end_s.shift()).abs().gt(.001).cumsum()
                for _,part in g.groupby(breaks):
                    v=part.local_component.to_numpy()
                    if len(v)>=2:chunks.append(v)
            self.local[stage]=chunks
        self.curves=pd.read_csv(OUTPUT/'observed_curves.csv')
        self.curves=self.curves[self.curves.source.isin(d.source.unique())]

    def bound(self,values,stages):
        return np.array([np.clip(v,*self.bounds[stage]) for v,stage in zip(values,stages)])

    def sample(self,rng,battery,stages,scale=1):
        v=self.offsets[battery];offset=v[rng.integers(len(v))].copy()
        for k in range(3):
            if not np.isfinite(offset[k]):offset[k]=rng.choice(v[np.isfinite(v[:,k]),k])
        vals=np.zeros(len(stages))
        for k,stage in enumerate(STAGES):
            ix=np.flatnonzero(np.array(stages)==k);seq=[]
            while len(seq)<len(ix):
                block=self.local[stage][rng.integers(len(self.local[stage]))]
                start=rng.integers(len(block));seq.extend(block[start:start+5].tolist())
            vals[ix]=(offset[k]+np.array(seq[:len(ix)]))*scale
        return vals

    def plateau(self,rng,battery,initial):
        q=self.curves[self.curves.battery_id.eq(battery)]
        z=q[q.initial_soc.between(initial-3,initial+3)]
        if len(z)<3:z=q
        p=float(rng.choice(z.initial_plateau_s))
        # Small jitter avoids exactly replaying a logged initial plateau.
        return float(np.clip(p+rng.normal(0,.5),.5,60))


def scenario_features(meta,initials,norm):
    rows=[]
    for j,battery in BATTERIES.items():
        bc=norm.curve_for(battery,f'drone_{j}')
        for active in range(1,6):
            for end_soc in range(int(initials[j])-1,19,-1):
                soc=end_soc+.5;k=stage_of(soc,bc.boundaries)
                rows.append(dict(**meta,position=j,drone_id=f'drone_{j}',battery_id=battery,
                    soc=soc,soc_end=end_soc,stage_index=k,stage=STAGES[k],
                    initial_soc=initials[j],drops_since_start=initials[j]-soc,
                    stage_progress=(bc.boundaries[k]-soc)/(bc.boundaries[k]-bc.boundaries[k+1]),
                    active_count=active,baseline_dwell_s=60/bc.rates_pp_min[k]))
    return pd.DataFrame(rows)


def simulate_events(meta,seed,model,norm,bank,initials=None,uncertainty_scale=1.0):
    """Coupled step clock: a drone leaves active_count when it reaches SOC20."""
    initials=initials or {j:100 for j in BATTERIES};rng=np.random.default_rng(seed)
    feat=scenario_features(meta,initials,norm);feat['prediction']=predict_dwell(model,feat)
    lookup={(r.position,r.active_count,r.soc_end):r.prediction for r in feat.itertuples()}
    common=float(rng.choice(bank.flight.to_numpy()))*uncertainty_scale
    noise={};soc=initials.copy();events=[];last={j:0.0 for j in BATTERIES};remain={};ready={}
    for j,battery in BATTERIES.items():
        bc=norm.curve_for(battery,f'drone_{j}')
        end=list(range(initials[j]-1,19,-1));stages=[stage_of(b+.5,bc.boundaries) for b in end]
        ns=bank.sample(rng,battery,stages,uncertainty_scale)
        noise[j]={b:float(v) for b,v in zip(end,ns)}
        remain[j]=1.0
        ready[j]=bank.plateau(rng,battery,initials[j])
    now=0.;active=5
    while active:
        speed={}
        for j in BATTERIES:
            if soc[j]<=20:continue
            dwell=ready[j] if soc[j]==initials[j] else lookup[j,active,soc[j]-1]*math.exp(common+noise[j][soc[j]-1])
            if soc[j]==initials[j]:dwell=float(np.clip(dwell,.5,60))
            else:
                bc=norm.curve_for(BATTERIES[j],f'drone_{j}')
                k=stage_of(soc[j]-.5,bc.boundaries)
                dwell=float(np.clip(dwell,*bank.bounds[STAGES[k]]))
            speed[j]=1/dwell
        dt=min(remain[j]/speed[j] for j in speed);now+=dt
        for j in speed:remain[j]-=dt*speed[j]
        finished=[j for j in speed if remain[j]<1e-8]
        for j in finished:
            start_soc=soc[j];soc[j]-=1
            k=stage_of(soc[j]+.5,norm.curve_for(BATTERIES[j],f'drone_{j}').boundaries)
            events.append(dict(**meta,drone_id=f'drone_{j}',position=j,battery_id=BATTERIES[j],
                start_s=last[j],end_s=now,soc_start=start_soc,soc_end=soc[j],duration_s=now-last[j],
                stage=STAGES[k],initial_plateau=start_soc==initials[j],active_count_at_end=active,
                is_simulated=True,seed=seed,model_version=MODEL_VERSION))
            last[j]=now;remain[j]=1.0
        active=sum(s>20 for s in soc.values())
    return pd.DataFrame(events)


def logger_frame(events,experiment_id,configs):
    template=ROOT/'database/wind_tunnel_front_75_tail_lv1_001/wind_tunnel_front_75_tail_lv1_001_20260903_172757_all_coordination.csv'
    original_cols=pd.read_csv(template,nrows=0).columns.tolist()
    last=events.groupby('position').end_s.max();end=math.ceil((last.max()+3)*10)/10
    t=np.round(np.arange(0,end+.05,.1),1);frames=[]
    meta=events.iloc[0]
    for j,battery in BATTERIES.items():
        e=events[events.position.eq(j)].sort_values('end_s');c=configs[j-1]
        edges=e.end_s.to_numpy();index=np.searchsorted(edges,t,side='right')
        initial=int(e.soc_start.iloc[0]);soc=initial-index;land=float(last[j])
        phase=np.where(t>=land+3,'wind_tunnel_landed',np.where(t>=land,'wind_tunnel_landing_20_percent',
              np.where(t<3,'wind_tunnel_takeoff',np.where(t<6,'wind_tunnel_centering','wind_tunnel_hover'))))
        data={k:np.nan for k in original_cols}
        data.update(run_id=experiment_id,experiment_id=experiment_id,formation=meta.formation,
            wind_direction=meta.wind_direction+' wind',wind_speed=f'Level{int(meta.wind_level)}',
            inter_drone_distance_cm=int(meta.spacing_cm),soc_mode='all_stages_simulated',
            target_soc_percent=20,soc_tolerance_percent=0,drone_name=f'drone_{j}',battery_id=battery,
            takeoff_order=j,drone_role=f'wind_tunnel_position_{j}',mission_pad=c['mission_pad'],
            grid_column=c['grid_column'],grid_row=c['grid_row'],phase=phase,elapsed_time=t,
            hover_elapsed_time=np.maximum(0,np.minimum(t,land)-6),target_x=c['target_x'],
            target_y=c['target_y'],target_z=c['target_z'],target_pad=c['mission_pad'],
            node_forward_distance_cm=0,node_speed_cm_s=0,battery=soc,battery_hover_start=initial,
            battery_hover_end=np.where(t>=land,20,np.nan))
        frame=pd.DataFrame(data,columns=original_cols)
        frame['is_simulated']=True;frame['data_origin']='empirical_stochastic_simulation'
        frame['simulation_model']=MODEL_VERSION;frame['simulation_seed']=int(meta.seed)
        frame['simulation_time_s']=t;frame['telemetry_policy']='unmodelled_fields_blank'
        frame['phase_policy']='assumed_3s_takeoff_3s_centering_3s_landing'
        frames.append(frame)
    result=pd.concat(frames).sort_values(['elapsed_time','takeoff_order'],kind='stable').reset_index(drop=True)
    assert result[TELEMETRY_COLUMNS].isna().all().all()
    return result


def generate():
    if DEST.exists():raise FileExistsError('Refusing to overwrite simulation export')
    DEST.mkdir(parents=True);(DEST/'runs').mkdir();(DEST/'examples').mkdir()
    cov,cells=coverage();cov.to_csv(DEST/'coverage.csv',index=False);cells.to_csv(DEST/'stage_position_coverage.csv',index=False)
    norm=BatteryNormalizer.load(BASELINE_PATH);bank=NoiseBank();model=joblib.load(OUTPUT/'simulator_model.joblib')
    # AST-isolated, read-only geometry helpers; never load or execute the aircraft SDK.
    from test_wind_tunnel_layouts import load_offline,experiment
    wt=load_offline();manifest=[];rate_rows=[];event_frames=[]
    targets=missing_allowed_conditions(cov)
    for k,r in enumerate(targets.to_dict('records')):
        meta={key:r[key] for key in ('condition_id','formation','spacing_cm','wind_direction','wind_level')}
        cfg=wt.build_configs(experiment(r['formation'],r['wind_direction']+' wind',r['spacing_cm'],f"Level{r['wind_level']}"))
        # Extra uncertainty is a disclosed sensitivity assumption, not calibrated truth for missing conditions.
        scale=1.25 if r['usable_steps']==0 else (1.10 if r['supported_cells']<15 else 1.0)
        for rep in range(1,4):
            seed=BASE_SEED+k*100+rep;eid=f"SIM_{r['condition_id']}_r{rep:02d}_s{seed}"
            events=simulate_events(meta,seed,model,norm,bank,uncertainty_scale=scale)
            events['experiment_id']=eid;event_frames.append(events)
            frame=logger_frame(events,eid,cfg)
            folder='runs'
            path=DEST/folder/(eid+'_all_coordination.csv.gz')
            frame.to_csv(path,index=False,float_format='%.5f',compression={'method':'gzip','compresslevel':5,'mtime':0})
            manifest.append(dict(**meta,experiment_id=eid,seed=seed,file=str(path.relative_to(DEST)),
                rows=len(frame),sha256=sha(path),purpose='missing_usable_real_condition_only',
                source_coverage=r['coverage'],supported_cells=r['supported_cells'],uncertainty_scale=scale,
                is_simulated=True,duration_s=float(events.end_s.max())))
            for (j,stage),g in events[~events.initial_plateau].groupby(['position','stage']):
                pp=len(g);seconds=g.duration_s.sum();bc=norm.curve_for(BATTERIES[j],f'drone_{j}');st=STAGES.index(stage)
                raw=60*pp/seconds;ideal=raw/bc.rates_pp_min[st]*norm.reference.rates_pp_min[st]
                rate_rows.append(dict(**meta,experiment_id=eid,position=j,battery_id=BATTERIES[j],stage=stage,
                    soc_start=g.soc_start.max(),soc_end=g.soc_end.min(),simulated_soc_drop_pp=pp,
                    duration_s=seconds,raw_rate_pp_min=raw,bideal_rate_pp_min=ideal,is_simulated=True))
        print('Generated',r['condition_id'],flush=True)
    pd.DataFrame(manifest).to_csv(DEST/'run_manifest.csv',index=False)
    pd.concat(event_frames).to_csv(DEST/'integer_soc_events.csv.gz',index=False,compression='gzip')
    rates=pd.DataFrame(rate_rows);rates.to_csv(DEST/'simulated_stage_rates_long.csv',index=False)
    ix=['condition_id','formation','spacing_cm','wind_direction','wind_level','experiment_id','stage']
    wide=rates.pivot(index=ix,columns='position',values=['raw_rate_pp_min','bideal_rate_pp_min'])
    wide.columns=[f'{measure}_position{j}' for measure,j in wide.columns]
    wide.reset_index().assign(is_simulated=True).to_csv(DEST/'simulated_stage_rates_wide.csv',index=False)
    meta=dict(model_version=MODEL_VERSION,simulation_only=True,not_real_experimental_replicates=True,
        nominal_initial_soc=100,landing_soc=20,sampling_interval_s=.1,simulation_runs=len(manifest),
        missing_conditions=len(targets),coverage_policy='missing_usable_real_conditions_only_except_legacy_safety_exclusions',
        logger_original_columns=62,unmodelled_telemetry_fields=TELEMETRY_COLUMNS,
        unmodelled_fields_note='Preserved as empty columns. Synthetic SOC is not a full physical sensor simulation.',
        phase_times_note='Takeoff3s, centering3s, landing3s are display assumptions, not fitted flight dynamics.',
        initial_plateau_note='Empirical delay in displayed SOC. Do not interpret it as zero physical energy consumption; exclude from fitted stage-rate estimates.',
        geometry_source='Current AST-isolated build_configs; historical front layout changes are not identified by the statistical model.',
        source_hashes_manifest=str((OUTPUT/'input_manifest.json').relative_to(ROOT)),
        model_sha256=sha(OUTPUT/'simulator_model.joblib'),normalizer_sha256=sha(BASELINE_PATH),
        stochastic_model='OOF flight error bootstrap (>=100 steps and >=3 drones) + battery-specific correlated stage offsets (>=5 local steps) + blocks of5 contiguous local residual steps; fixed seeds',
        nominal_dwell_envelope_s=bank.bounds,
        envelope_note='Generated non-initial dwells bounded by development stage0.5th..99.5th percentile. Explicit nominal healthy-hover assumption; rare sensor/fault tails intentionally not recreated.',
        missing_condition_uncertainty='Heuristic log-residual scale1.25 unseen,1.10 partial,1.0 supported; not validated coverage for missing conditions',
        forbidden_interpretations=['No forced leader ranking','No causal spacing effect','No substitute for measured repetitions','No actual coordinate or attitude claims'])
    (DEST/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':generate()
