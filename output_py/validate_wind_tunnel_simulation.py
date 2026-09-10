"""Read-only source checks, held-out diagnostics and synthetic export QA."""
import json
import math
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from wind_tunnel_simulation_model import (ROOT,OUTPUT,BASELINE_PATH,BASE_SEED,BATTERIES,STAGES,
    BatteryNormalizer,fit_predictor,predict_dwell,error_metrics,sha)
from generate_wind_tunnel_simulation import DEST,NoiseBank,simulate_events,TELEMETRY_COLUMNS


def transfer_checks():
    d=pd.read_csv(OUTPUT/'observed_steps.csv');d=d[d.fit_eligible]
    selected=json.loads((OUTPUT/'training_summary.json').read_text())['selected_model']
    cases=[('75_to_50',d.spacing_cm.eq(50))]
    cases += [('unseen_formation_'+f,d.formation.eq(f)) for f in sorted(d.formation.unique())]
    rows=[]
    for name,mask in cases:
        m=fit_predictor(d[~mask],selected);p=predict_dwell(m,d[mask])
        rows.append(dict(test=name,**error_metrics(d[mask],p)))
    pd.DataFrame(rows).to_csv(OUTPUT/'transfer_stress_tests.csv',index=False)
    return rows


def stochastic_diagnostics():
    bank=NoiseBank();test=pd.read_csv(OUTPUT/'holdout_predictions.csv')
    rng=np.random.default_rng(BASE_SEED+999);samples=[]
    for rep in range(20):
        for source,g in test.groupby('source'):
            common=float(rng.choice(bank.flight.to_numpy()))
            for drone,q in g.groupby('drone_id'):
                q=q.sort_values('start_s').copy();battery=q.battery_id.iloc[0]
                ns=bank.sample(rng,battery,q.stage.map(dict(zip(STAGES,range(3)))).tolist())
                q['synthetic_dwell_s']=bank.bound(q.predicted_dwell_s*np.exp(common+ns),q.stage)
                q['replicate']=rep;samples.append(q[['source','drone_id','stage','soc','duration_s','predicted_dwell_s','synthetic_dwell_s','replicate']])
    syn=pd.concat(samples,ignore_index=True);syn.to_csv(OUTPUT/'matched_holdout_stochastic_draws.csv.gz',index=False,compression='gzip')
    rows=[]
    for stage in STAGES:
        a=test[test.stage.eq(stage)].duration_s.to_numpy()
        b=syn[syn.stage.eq(stage)].synthetic_dwell_s.to_numpy()
        rows.append(dict(stage=stage,observed_steps=len(a),simulation_draws=len(b),
            observed_median_s=float(np.median(a)),simulated_median_s=float(np.median(b)),
            observed_p10_s=float(np.quantile(a,.1)),simulated_p10_s=float(np.quantile(b,.1)),
            observed_p90_s=float(np.quantile(a,.9)),simulated_p90_s=float(np.quantile(b,.9)),
            wasserstein_s=float(wasserstein_distance(a,b)),ks_distance=float(ks_2samp(a,b).statistic)))
    pd.DataFrame(rows).to_csv(OUTPUT/'matched_distribution_metrics.csv',index=False)
    return rows


def qa_exports():
    src=json.loads((OUTPUT/'input_manifest.json').read_text())
    for p,h in src['input_sha256'].items():assert sha(ROOT/p)==h,p
    raw=pd.read_csv(OUTPUT/'observed_steps.csv');fit=raw[raw.fit_eligible]
    # Independently reconcile a deterministic sample against original SOC/timestamp endpoints.
    traced=0
    for source,g in fit.groupby('source'):
        obs=pd.read_csv(ROOT/source,usecols=['drone_name','battery_id','elapsed_time','battery'])
        sample=g.groupby(['drone_id','stage']).head(1)
        for r in sample.itertuples():
            a=obs[(obs.drone_name==r.drone_id)&np.isclose(obs.elapsed_time,r.start_s,rtol=0,atol=1e-7)]
            b=obs[(obs.drone_name==r.drone_id)&np.isclose(obs.elapsed_time,r.end_s,rtol=0,atol=1e-7)]
            assert len(a)==len(b)==1 and a.battery.iloc[0]-b.battery.iloc[0]==1
            assert np.isclose(b.elapsed_time.iloc[0]-a.elapsed_time.iloc[0],r.duration_s)
            traced+=1
    manifest=pd.read_csv(DEST/'run_manifest.csv');events=pd.read_csv(DEST/'integer_soc_events.csv.gz')
    original=pd.read_csv(ROOT/'database/wind_tunnel_front_75_tail_lv1_001/wind_tunnel_front_75_tail_lv1_001_20260903_172757_all_coordination.csv',nrows=0).columns.tolist()
    total=0
    for r in manifest.itertuples():
        path=DEST/r.file;assert sha(path)==r.sha256
        d=pd.read_csv(path,low_memory=False);assert d.columns[:len(original)].tolist()==original
        assert len(d)==r.rows and d.is_simulated.all() and d.experiment_id.str.startswith('SIM_').all()
        assert d[TELEMETRY_COLUMNS].isna().all().all()
        assert d.battery.between(20,100).all() and np.equal(d.battery,np.floor(d.battery)).all()
        assert d.drone_name.nunique()==5
        assert not d.duplicated(['drone_name','elapsed_time']).any()
        for drone,q in d.groupby('drone_name'):
            assert q.battery.iloc[0]==100 and q.battery.iloc[-1]==20
            assert q.battery.diff().dropna().isin([-1,0]).all()
            assert np.allclose(q.elapsed_time.diff().dropna(),.1)
            landing=q[q.phase.str.contains('land')]
            assert landing.battery.eq(20).all()
            assert q[q.battery.gt(20)].phase.str.contains('land').sum()==0
        e=events[events.experiment_id.eq(r.experiment_id)]
        assert len(e)==400 and e.duration_s.between(.5-1e-8,90+1e-8).all()
        assert (e.soc_start-e.soc_end).eq(1).all()
        total+=len(d)
    norm=BatteryNormalizer.load(BASELINE_PATH);model=joblib.load(OUTPUT/'simulator_model.joblib');bank=NoiseBank()
    r=manifest.iloc[0];meta={key:r[key] for key in ['condition_id','formation','spacing_cm','wind_direction','wind_level']}
    scale=r.uncertainty_scale
    e1=simulate_events(meta,int(r.seed),model,norm,bank,uncertainty_scale=scale)
    e2=simulate_events(meta,int(r.seed),model,norm,bank,uncertainty_scale=scale)
    pd.testing.assert_frame_equal(e1,e2)
    saved=events[events.experiment_id.eq(r.experiment_id)].drop(columns='experiment_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(e1,saved,check_dtype=False,rtol=1e-10,atol=1e-10)
    e3=simulate_events(meta,int(r.seed)+9999,model,norm,bank,uncertainty_scale=scale)
    assert not np.allclose(e1.sort_values(['position','soc_end']).end_s,e3.sort_values(['position','soc_end']).end_s)
    # None of the files registers itself as a measured experiment.
    registry=json.loads((ROOT/'database/experiment_registry.json').read_text())
    assert not set(manifest.experiment_id)&{r['experiment_id'] for r in registry['experiments']}
    report=dict(assessment='Share with caveats — synthetic exploratory data only',
        source_hashes_unchanged=len(src['input_sha256']),raw_endpoint_intervals_traced=traced,
        runs_verified=len(manifest),logger_rows_verified=total,integer_soc_events=len(events),
        original_logger_columns_retained=len(original),unknown_telemetry_blank=True,
        only_land_at20=True,reproducible_seed=True,different_seeds_differ=True,
        source_experiment_registry_unchanged=True,real_flight_controls_used=False,
        full_sensor_physics_validated=False,missing_conditions_experimentally_validated=False)
    (OUTPUT/'export_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    print('Transfer tests',transfer_checks(),flush=True)
    print('Distribution checks',stochastic_diagnostics(),flush=True)
    print('Export QA',qa_exports(),flush=True)
