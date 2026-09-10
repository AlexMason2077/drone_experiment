"""Coverage-only revision: all requested combinations except the frozen safety mask.

Keep the fitted model, noise logic and original data unchanged. Reuse existing
allowed synthetic runs byte-for-byte; only generate missing allowed conditions.
"""
import ast
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wind_tunnel_simulation_model import (
    ROOT, OUTPUT, BASELINE_PATH, BATTERIES, STAGES, BatteryNormalizer, condition_id, sha,
)
from generate_wind_tunnel_simulation import (
    DEST as PREVIOUS, coverage, NoiseBank, simulate_events, logger_frame, MODEL_VERSION,
)

DEST = ROOT / 'simulation_data/wind_tunnel_all_allowed_v2_20260909'
SAFETY_MANIFEST = ROOT / 'frozen_data/discharge_rates/medium_bideal_v1_20260908/manifest.json'
ORACLE_SOURCE = ROOT / 'ml_policy/oracle_optimizer.py'
EXPORT_VERSION = 'wind_tunnel_all_allowed_v2_20260909'


def unsafe_condition_ids():
    frozen = json.loads(SAFETY_MANIFEST.read_text())['unsafe_condition_cells']
    expected = {condition_id(f, s, w, l) for w, l, f, s in frozen}
    tree = ast.parse(ORACLE_SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name) and n.target.id == 'UNSAFE_STRUCTURES_BY_CONDITION')
    # Evaluate this one literal mapping only, without importing application code.
    current = eval(compile(ast.Expression(node.value), '<safety mapping>', 'eval'),
                   {'__builtins__': {}, 'frozenset': frozenset})
    actual = {condition_id(f.replace('echalon', 'echelon'), s, w, l)
              for (w, l), cells in current.items() for f, s in cells}
    if actual != expected:
        raise ValueError('Frozen and current safety masks disagree; do not guess the exclusions.')
    return expected


def planned_coverage():
    cov, cells = coverage()
    unsafe = unsafe_condition_ids()
    cov = cov.rename(columns={'generate_missing_or_partial': 'historically_missing_or_partial'})
    cov['safety_excluded'] = cov.condition_id.isin(unsafe)
    cov['requested_simulation_replicates'] = np.where(cov.safety_excluded, 0, 3)
    cov['export_status'] = np.where(cov.safety_excluded, 'excluded_existing_safety_rule', 'included_all_allowed')
    cells['safety_excluded'] = cells.condition_id.isin(unsafe)
    return cov, cells


def stage_rows(events, meta, eid, norm):
    rows = []
    for (j, stage), g in events[~events.initial_plateau].groupby(['position', 'stage']):
        pp, seconds = len(g), float(g.duration_s.sum())
        bc = norm.curve_for(BATTERIES[j], f'drone_{j}')
        k = STAGES.index(stage)
        raw = 60 * pp / seconds
        rows.append(dict(**meta, experiment_id=eid, position=j, battery_id=BATTERIES[j], stage=stage,
                         soc_start=g.soc_start.max(), soc_end=g.soc_end.min(), simulated_soc_drop_pp=pp,
                         duration_s=seconds, raw_rate_pp_min=raw,
                         bideal_rate_pp_min=raw / bc.rates_pp_min[k] * norm.reference.rates_pp_min[k],
                         is_simulated=True))
    return rows


def build():
    if DEST.exists():
        raise FileExistsError(f'Refusing to overwrite existing export: {DEST}')
    cov, cells = planned_coverage()
    unsafe = unsafe_condition_ids()
    old_manifest = pd.read_csv(PREVIOUS / 'run_manifest.csv')
    old_events = pd.read_csv(PREVIOUS / 'integer_soc_events.csv.gz')
    old_rates = pd.read_csv(PREVIOUS / 'simulated_stage_rates_long.csv')
    previous_hashes = {str(p.relative_to(PREVIOUS)): sha(p)
                       for p in PREVIOUS.rglob('*') if p.is_file()}
    before = json.loads((OUTPUT / 'input_manifest.json').read_text())['input_sha256']
    for path, expected in before.items():
        assert sha(ROOT / path) == expected, path
    norm = BatteryNormalizer.load(BASELINE_PATH)
    model = joblib.load(OUTPUT / 'simulator_model.joblib')
    bank = NoiseBank()
    from test_wind_tunnel_layouts import load_offline, experiment
    wt = load_offline()
    DEST.mkdir(parents=True)
    (DEST / 'runs').mkdir()
    manifest, event_frames, rate_frames = [], [], []
    allowed = cov[~cov.safety_excluded].sort_values('condition_id')
    for condition_index, r in enumerate(allowed.to_dict('records')):
        cid = r['condition_id']
        old = old_manifest[old_manifest.condition_id.eq(cid)].sort_values('seed')
        if len(old) not in (0, 3):
            raise ValueError(f'Unexpected partial old simulation group: {cid}, {len(old)}')
        if len(old) == 3:
            ids = set(old.experiment_id)
            for row in old.to_dict('records'):
                source = PREVIOUS / row['file']
                assert sha(source) == row['sha256']
                target = DEST / 'runs' / source.name
                shutil.copy2(source, target)
                row.update(file=str(target.relative_to(DEST)), purpose='all_allowed_condition',
                           previous_file=str(source.relative_to(ROOT)), reused_unchanged=True,
                           export_version=EXPORT_VERSION)
                manifest.append(row)
            event_frames.append(old_events[old_events.experiment_id.isin(ids)].copy())
            rate_frames.append(old_rates[old_rates.experiment_id.isin(ids)].copy())
            continue
        meta = {key: r[key] for key in ('condition_id', 'formation', 'spacing_cm', 'wind_direction', 'wind_level')}
        configs = wt.build_configs(experiment(r['formation'], r['wind_direction'] + ' wind',
                                             r['spacing_cm'], f"Level{r['wind_level']}"))
        scale = 1.25 if r['usable_steps'] == 0 else (1.10 if r['supported_cells'] < 15 else 1.0)
        for rep in range(1, 4):
            # Separate deterministic namespace: no collision with existing v1 seeds.
            seed = 20270909 + condition_index * 100 + rep
            eid = f'SIM_{cid}_r{rep:02d}_s{seed}'
            events = simulate_events(meta, seed, model, norm, bank, uncertainty_scale=scale)
            events['experiment_id'] = eid
            frame = logger_frame(events, eid, configs)
            target = DEST / 'runs' / (eid + '_all_coordination.csv.gz')
            frame.to_csv(target, index=False, float_format='%.5f',
                         compression={'method': 'gzip', 'compresslevel': 5, 'mtime': 0})
            manifest.append(dict(**meta, experiment_id=eid, seed=seed,
                file=str(target.relative_to(DEST)), rows=len(frame), sha256=sha(target),
                purpose='all_allowed_condition', source_coverage=r['coverage'],
                supported_cells=r['supported_cells'], uncertainty_scale=scale,
                is_simulated=True, duration_s=float(events.end_s.max()),
                previous_file='', reused_unchanged=False, export_version=EXPORT_VERSION))
            event_frames.append(events)
            rate_frames.append(pd.DataFrame(stage_rows(events, meta, eid, norm)))
        print('Added 3 simulations:', cid, flush=True)
    m = pd.DataFrame(manifest).sort_values(['condition_id', 'seed']).reset_index(drop=True)
    counts = m.groupby('condition_id').size()
    assert set(counts.index) == set(allowed.condition_id)
    assert counts.eq(3).all() and not set(counts.index) & unsafe
    assert not m.experiment_id.duplicated().any() and not m.seed.duplicated().any()
    cov['simulation_replicates'] = cov.condition_id.map(counts).fillna(0).astype(int)
    assert cov.simulation_replicates.equals(cov.requested_simulation_replicates)
    cov.to_csv(DEST / 'coverage.csv', index=False)
    cells.to_csv(DEST / 'stage_position_coverage.csv', index=False)
    cov[cov.safety_excluded].to_csv(DEST / 'safety_exclusions.csv', index=False)
    m.to_csv(DEST / 'run_manifest.csv', index=False)
    events = pd.concat(event_frames, ignore_index=True)
    events.to_csv(DEST / 'integer_soc_events.csv.gz', index=False, compression='gzip')
    rates = pd.concat(rate_frames, ignore_index=True)
    rates.to_csv(DEST / 'simulated_stage_rates_long.csv', index=False)
    keys = ['condition_id', 'formation', 'spacing_cm', 'wind_direction', 'wind_level', 'experiment_id', 'stage']
    wide = rates.pivot(index=keys, columns='position', values=['raw_rate_pp_min', 'bideal_rate_pp_min'])
    wide.columns = [f'{measure}_position{j}' for measure, j in wide.columns]
    wide.reset_index().assign(is_simulated=True).to_csv(DEST / 'simulated_stage_rates_wide.csv', index=False)
    inherited = json.loads((PREVIOUS / 'manifest.json').read_text())
    inherited.pop('missing_partial_conditions', None)
    inherited.update(export_version=EXPORT_VERSION, coverage_policy='all_60_combinations_except_existing_four_safety_cells',
        theoretical_conditions=len(cov), excluded_conditions=sorted(unsafe), covered_conditions=len(counts),
        simulation_runs=len(m), realizations_per_condition=3,
        reused_unchanged_runs=int(m.reused_unchanged.sum()), newly_generated_runs=int((~m.reused_unchanged).sum()),
        excluded_legacy_runs=int(old_manifest.condition_id.isin(unsafe).sum()),
        previous_export_preserved=str(PREVIOUS.relative_to(ROOT)),
        safety_source=str(SAFETY_MANIFEST.relative_to(ROOT)), safety_source_sha256=sha(SAFETY_MANIFEST),
        safety_code_sha256=sha(ORACLE_SOURCE),
        scope_note='User explicitly applies the old Medium safety exclusions to this wind-tunnel simulation export. This is not a new physical safety certification.',
        method_changed=False, model_retrained=False, previous_exports_deleted=False)
    for path, expected in before.items():
        assert sha(ROOT / path) == expected, path
    for path, expected in previous_hashes.items():
        assert sha(PREVIOUS / path) == expected, path
    inherited['validation_status'] = 'coverage_checked_full_file_QA_pending'
    (DEST / 'manifest.json').write_text(json.dumps(inherited, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: inherited[key] for key in ('covered_conditions', 'simulation_runs',
          'reused_unchanged_runs', 'newly_generated_runs', 'excluded_legacy_runs')}, indent=2), flush=True)


if __name__ == '__main__':
    raise SystemExit('Superseded by user: do not simulate conditions with usable real data. Use the missing-only generator.')
