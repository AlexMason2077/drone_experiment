"""Recoverably remove synthetic runs for observed or safety-excluded conditions.

Reads and verifies exact manifests before moving any file. Never modifies
database/ or a frozen calibration. Filtered aggregate replacements are staged
before their old versions move to Trash.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from wind_tunnel_simulation_model import ROOT, OUTPUT, sha
from generate_wind_tunnel_simulation import coverage, missing_allowed_conditions
from complete_wind_tunnel_simulation_coverage import unsafe_condition_ids

EXPORTS = (
    ROOT/'simulation_data/wind_tunnel_empirical_v1_20260909',
    ROOT/'simulation_data/wind_tunnel_all_allowed_v2_20260909',
    OUTPUT/'prototype_unbounded',
)


def prepare():
    cov, _ = coverage()
    observed = set(cov.loc[cov.usable_flights.gt(0), 'condition_id'])
    unsafe = unsafe_condition_ids()
    allowed = set(missing_allowed_conditions(cov).condition_id)
    assert len(observed)==24 and len(unsafe)==4 and len(allowed)==32
    assert not observed & unsafe
    manifest = json.loads((OUTPUT/'input_manifest.json').read_text())
    hashes = manifest['input_sha256']
    for path, expected in hashes.items():
        assert sha(ROOT/path)==expected, path
    plans=[]
    for root in EXPORTS:
        assert root.is_dir() and not root.is_symlink()
        runs=pd.read_csv(root/'run_manifest.csv')
        remove=runs[runs.condition_id.isin(observed|unsafe)].copy()
        keep=runs[~runs.condition_id.isin(observed|unsafe)].copy()
        assert set(keep.condition_id)==allowed
        assert keep.groupby('condition_id').size().eq(3).all()
        assert set(runs.experiment_id)==set(keep.experiment_id)|set(remove.experiment_id)
        resolved=[]
        for r in runs.itertuples():
            p=root/r.file
            assert p.resolve().is_relative_to(root.resolve()) and not p.is_symlink()
            assert p.name.startswith('SIM_') and p.name.endswith('_all_coordination.csv.gz')
            assert p.is_file() and sha(p)==r.sha256, p
            resolved.append(p.resolve())
        actual={p.resolve() for folder in ('runs','examples') for p in (root/folder).glob('*_all_coordination.csv.gz')}
        assert actual==set(resolved), root
        plans.append((root,runs,remove,keep))
    return cov, observed, unsafe, allowed, hashes, plans


def prune(apply=False):
    cov, observed, unsafe, allowed, hashes, plans=prepare()
    summary=[dict(export=str(root.relative_to(ROOT)),remove_run_files=len(remove),keep_runs=len(keep),
                  observed_run_files=int(remove.condition_id.isin(observed).sum()),
                  unsafe_run_files=int(remove.condition_id.isin(unsafe).sum()))
             for root,runs,remove,keep in plans]
    print(json.dumps(summary,indent=2),flush=True)
    if not apply:return
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    trash=Path('/Users/alexmason/.Trash')/('drone_simulation_cleanup_'+stamp)
    trash.mkdir(parents=False,exist_ok=False)
    audit=dict(policy='remove_when_any_usable_real_curve_exists; also exclude four legacy safety cells',
               observed_conditions=sorted(observed),safety_exclusions=sorted(unsafe),
               remaining_conditions=sorted(allowed),remaining_runs_per_export=96,
               trash_path=str(trash),exports=summary,moved_files=[])
    staging=Path(tempfile.mkdtemp(prefix='drone_simulation_prune_'))
    operations=[]
    for root,runs,remove,keep in plans:
        stage=staging/root.name;stage.mkdir()
        ids=set(keep.experiment_id)
        keep=keep.copy();keep['purpose']='missing_usable_real_condition_only'
        keep.to_csv(stage/'run_manifest.csv',index=False)
        # Materialize the reduced aggregate files before moving originals.
        for name in ('integer_soc_events.csv.gz','simulated_stage_rates_long.csv','simulated_stage_rates_wide.csv'):
            d=pd.read_csv(root/name)
            assert set(d.experiment_id)==set(runs.experiment_id), (root,name)
            filtered=d[d.experiment_id.isin(ids)].copy()
            assert set(filtered.experiment_id)==ids
            filtered.to_csv(stage/name,index=False,compression='gzip' if name.endswith('.gz') else None)
        c=pd.read_csv(root/'coverage.csv')
        c['has_usable_real_data']=c.condition_id.isin(observed)
        c['safety_excluded']=c.condition_id.isin(unsafe)
        c['simulation_replicates']=c.condition_id.map(keep.groupby('condition_id').size()).fillna(0).astype(int)
        c['requested_simulation_replicates']=np.where(c.condition_id.isin(allowed),3,0)
        c['export_status']=np.where(c.safety_excluded,'excluded_existing_safety_rule',
                                  np.where(c.has_usable_real_data,'real_data_exists_no_simulation','included_missing_condition'))
        c.to_csv(stage/'coverage.csv',index=False)
        cells=pd.read_csv(root/'stage_position_coverage.csv')
        cells['has_usable_real_data']=cells.condition_id.isin(observed)
        cells['safety_excluded']=cells.condition_id.isin(unsafe)
        cells['simulation_included']=cells.condition_id.isin(allowed)
        cells.to_csv(stage/'stage_position_coverage.csv',index=False)
        meta=json.loads((root/'manifest.json').read_text())
        for key in ('missing_partial_conditions','covered_conditions','reused_unchanged_runs','newly_generated_runs',
                    'excluded_legacy_runs','realizations_per_condition'):
            meta.pop(key,None)
        meta.update(coverage_policy='missing_usable_real_conditions_only_except_legacy_safety_exclusions',
            simulation_runs=len(keep),covered_conditions=len(allowed),realizations_per_condition=3,
            removed_observed_condition_runs=int(remove.condition_id.isin(observed).sum()),
            removed_safety_excluded_runs=int(remove.condition_id.isin(unsafe).sum()),
            observed_conditions_without_simulations=sorted(observed),excluded_conditions=sorted(unsafe),
            cleanup_stamp=stamp,removed_files_recoverable_at=str(trash),
            validation_status='filtered_export_hashes_and_membership_verified',
            legacy_status='not_for_use' if root.name=='prototype_unbounded' else 'missing_conditions_only')
        (stage/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        removed=remove.copy()
        removed['removal_reason']=np.where(removed.condition_id.isin(unsafe),'existing_safety_exclusion','usable_real_data_exists')
        removed.to_csv(stage/'removed_simulation_runs.csv',index=False)
        operations.append((root,stage,remove,keep))
    # Exact validated targets only. Retained run files are never moved or overwritten.
    for root,stage,remove,keep in operations:
        backup=trash/root.name;backup.mkdir()
        for r in remove.itertuples():
            original=root/r.file;destination=backup/r.file
            destination.parent.mkdir(parents=True,exist_ok=True)
            original.rename(destination)
            audit['moved_files'].append(dict(original=str(original),recoverable=str(destination),sha256=sha(destination)))
        for new in sorted(stage.iterdir()):
            original=root/new.name
            if original.exists():
                destination=backup/new.name
                original.rename(destination)
                audit['moved_files'].append(dict(original=str(original),recoverable=str(destination),sha256=sha(destination)))
            new.rename(original)
        current=pd.read_csv(root/'run_manifest.csv')
        assert len(current)==96 and set(current.condition_id)==allowed
        assert current.groupby('condition_id').size().eq(3).all()
        for r in current.itertuples():assert sha(root/r.file)==r.sha256
        assert not any((root/r.file).exists() for r in remove.itertuples())
        for name in ('integer_soc_events.csv.gz','simulated_stage_rates_long.csv','simulated_stage_rates_wide.csv'):
            d=pd.read_csv(root/name)
            assert set(d.experiment_id)==set(current.experiment_id)
            assert not set(d.condition_id)&(observed|unsafe)
        # Actual directory scan must agree with the manifest, not merely its row count.
        on_disk={p.name for sub in ('runs','examples') for p in (root/sub).glob('*_all_coordination.csv.gz')}
        assert on_disk=={Path(x).name for x in current.file}
    for path,expected in hashes.items():assert sha(ROOT/path)==expected,path
    audit.update(source_hashes_unchanged=len(hashes),verification_passed=True,
                 total_removed_run_file_copies=sum(x['remove_run_files'] for x in summary))
    (trash/'RESTORE_MANIFEST.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (OUTPUT/'simulation_cleanup_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ('trash_path','verification_passed','source_hashes_unchanged',
                      'remaining_runs_per_export','total_removed_run_file_copies')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    prune(parser.parse_args().apply)
