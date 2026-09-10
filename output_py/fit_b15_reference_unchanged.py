"""Add B15 calibration from the user-selected 13 May baseline, without refitting Bideal."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from battery_normalization import BatteryNormalizer, DischargeCurve
from output_py.search_bideal_boundaries import load_trace, search, fit

BASE=ROOT/'analysis_results/battery_normalization_extended_v3_20260909/model.json'
OUT=ROOT/'analysis_results/battery_normalization_v3_with_b15_20260909'
SOURCE=ROOT/'database/baselines/drone_5_B15/drone_5_B15_hover_20260513_150444_timeseries.csv'
METADATA=SOURCE.with_name('drone_5_B15_hover_20260513_150444_metadata.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    if OUT.exists():
        raise FileExistsError('Refusing to overwrite existing B15 candidate')
    old=json.loads(BASE.read_text())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [BASE,SOURCE,METADATA]}
    meta=json.loads(METADATA.read_text())
    assert (meta['run_id'],meta['battery_id'],meta['drone_name'],meta['mode']) == ('20260513_150444','B15','drone_5','hover')
    row=SimpleNamespace(source=str(SOURCE.relative_to(ROOT)),sha256=sha(SOURCE),
        battery_id='B15',drone='drone_5',run_id=meta['run_id'])
    raw=load_trace(row)
    grid,traces=search([raw],top=95,bottom=20,min_width=10,min_seconds=20)
    best=grid.iloc[0]
    fit95=fit(traces[0],int(best.upper),int(best.lower),20)
    curve=DischargeCurve((100,float(best.upper),float(best.lower),20),tuple(fit95['rates']))
    model=copy.deepcopy(old)
    model['version']='individual_three_stage_v3_extended100_with_B15_20260909'
    model['previous_model_sha256']=sha(BASE)
    model['b15_addition']={
        'run_id':meta['run_id'],'source_selection':'User explicitly selected 2026-05-13 only',
        'excluded_other_run_ids':['20260609_180213'],
        'role':'Normalization target only; not a Bideal contributor',
        'reference_unchanged':True,
        'reference_contributors':['B10','B11','B12','B13','B14'],
        'parent_model_version':old['version'],'parent_model_sha256':sha(BASE)}
    model['fitting']['individual_by_battery']['B15']='Single-run raw-sample SOC RMSE; endpoint-anchored integer-SOC knots, 95..20'
    model['batteries']['B15']=dict(drone_id='drone_5',**curve.to_dict(),
        run_id=meta['run_id'],source=str(SOURCE.relative_to(ROOT)),sha256=sha(SOURCE),
        source_95_to_20_fit_diagnostics=dict(in_sample_rmse_pp=fit95['rmse'],
            max_error_pp=fit95['max_error'],observed_samples=len(traces[0]['t']),
            boundaries_soc=[95,float(best.upper),float(best.lower),20],
            anchor_times_s=(fit95['anchor_times']-fit95['anchor_times'][0]).tolist()),
        anchor_times_s=[curve.equivalent_seconds(100,s) for s in curve.boundaries],
        time_origin_soc=100,calibrated_soc_range=[20,95],supported_soc_range=[20,100],
        modeled_100_to_95_duration_s=curve.equivalent_seconds(100,95),
        extrapolation_method='Continue fitted first slope to 100 without refitting; no added telemetry',
        included_in_Bideal=False)
    assert model['reference']==old['reference']
    assert model['fitting']['battery_weights']==old['fitting']['battery_weights']
    assert all(model['batteries'][b]==old['batteries'][b] for b in old['batteries'])
    normalizer=BatteryNormalizer(model)
    assert np.isclose(normalizer.relative_drain('B15','drone_5',80,54,
        fit95['anchor_times'][2]-fit95['anchor_times'][1]),1)
    for p,expected in hashes.items():
        assert sha(ROOT/p)==expected
    summary=dict(battery_id='B15',drone_id='drone_5',run_id=meta['run_id'],
        boundaries_soc=list(curve.boundaries),rates_pp_min=list(curve.rates_pp_min),
        in_sample_rmse_pp=fit95['rmse'],
        Bideal_medium_scale=normalizer.reference.rates_pp_min[1]/curve.rates_pp_min[1],
        Bideal_reference_unchanged=True,other_battery_curves_unchanged=True,
        input_files_sha256=hashes,method='Same individual three-stage endpoint fitting rules as B12/B13/B14')
    OUT.mkdir(parents=True)
    (OUT/'model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n')
    (OUT/'fit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    grid.to_csv(OUT/'boundary_search.csv',index=False)
    pd.DataFrame(dict(elapsed_s_since_95=traces[0]['t']-traces[0]['t'][0],
        observed_soc=traces[0]['soc'],fitted_soc=fit95['pred'])).to_csv(OUT/'fit_evidence.csv',index=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    build()
