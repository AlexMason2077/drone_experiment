"""Recalculate the same audited curves, removing only the initial SOC plateau."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'output_py'))
from audit_wind_tunnel_stage_ratios import stage_stats, summary, sha

BASE = ROOT / 'analysis_results/wind_tunnel_stage_ratio_check_20260909'
OUT = BASE / 'first_drop_recalculation'


def main():
    pairs = pd.read_csv(BASE / 'paired_ratios.csv')
    stages = pd.read_csv(BASE / 'stage_rates.csv')
    updated, audit, stage_rows = [], [], []
    for source, group in pairs.groupby('source'):
        raw = pd.read_csv(ROOT / source, low_memory=False)
        for _, p in group.iterrows():
            g = raw[raw.drone_name.eq(p.drone_id)].sort_values('elapsed_time').copy()
            takeoff = g[g.phase.eq('wind_tunnel_takeoff')]
            if takeoff.empty:
                raise ValueError(f'No takeoff phase: {source} {p.drone_id}')
            takeoff_s = float(takeoff.elapsed_time.iloc[0])
            g = g[g.elapsed_time.ge(takeoff_s)]
            initial_soc = float(g.battery.iloc[0])
            drop = g[g.battery.diff().lt(0)]
            if drop.empty:
                raise ValueError('No actual SOC decrease')
            first = drop.iloc[0]
            start, soc = float(first.elapsed_time), float(first.battery)
            old = stages[(stages.source.eq(source)) & stages.drone_id.eq(p.drone_id)].set_index('stage')
            high = old.loc['high']
            span = g[g.elapsed_time.between(start, high.end_s)]
            if span.phase.str.contains('fault|recoverable_error|uncommanded|land', case=False, na=False).any():
                raise ValueError(f'New High window has a fault/landing: {source} {p.drone_id}')
            assert soc > high.soc_lower
            assert soc <= initial_soc
            stat = stage_stats(g, start, high.end_s, soc, high.soc_lower,
                               high.baseline_rate_pp_min, high.Bideal_rate_pp_min / high.k,
                               p.all_ready_s, p.first_landing_s)
            assert stat is not None
            item = p.to_dict()
            item.update(initial_soc=initial_soc, first_drop_soc=soc, first_drop_s=start,
                        initial_plateau_excluded_s=start-takeoff_s, first_drop_phase=str(first.phase),
                        previous_95_high_difference_pct=p.high_difference_pct)
            for key in ['raw_rate_pp_min','k','Bideal_rate_pp_min','duration_s','ols_k','own_pad_fraction',
                        'own_pad_near20_fraction','h_median_cm','tof_median_cm','after_first_landing_fraction',
                        'before_all_ready_fraction']:
                item['high_' + key] = stat[key]
            item['high_difference_pct'] = 100 * (stat['k'] / p.medium_k - 1)
            item['high_ols_difference_pct'] = 100 * (stat['ols_k'] / p.medium_ols_k - 1)
            item['high_starts_after_all_ready'] = start >= p.all_ready_s
            item['same_swarm_entire_curve'] = item['high_starts_after_all_ready'] and p.low_ends_before_first_landing
            item['all_stages_own_pad_near20_ge80'] = all([stat['own_pad_near20_fraction'] >= .8,
                                                        p.medium_own_pad_near20_fraction >= .8,
                                                        p.low_own_pad_near20_fraction >= .8])
            # Control: old 95% start already excluded the initial plateau.
            item['high_difference_change_percentage_points'] = item['high_difference_pct'] - p.high_difference_pct
            updated.append(item)
            audit.append({k:item[k] for k in ['source','experiment_id','drone_id','battery_id','wind_level',
                          'initial_soc','first_drop_soc','first_drop_s','initial_plateau_excluded_s',
                          'first_drop_phase','high_starts_after_all_ready','previous_95_high_difference_pct',
                          'high_difference_pct','high_difference_change_percentage_points']})
            for stage in ['high','medium','low']:
                row = old.loc[stage].to_dict()
                row['stage'] = stage
                if stage == 'high':
                    row.update(stat)
                stage_rows.append(row)
    p = pd.DataFrame(updated)
    w = p[p.wind_level.gt(0)]
    summaries = [summary(p,'all_complete'),summary(w,'with_wind'),summary(p[p.wind_level.eq(0)],'no_wind'),
                 summary(w[w.high_starts_after_all_ready],'with_wind_first_drop_after_all_ready')]
    manifest = json.loads((BASE / 'manifest.json').read_text())
    for source, expected in manifest['input_files_sha256'].items():
        assert sha(ROOT / source) == expected
    manifest.update(full_curve_range='First observed post-takeoff SOC decrease to 20%; High SOC begins at the first decreased sample, not 100',
                    update_reason='User requested exclusion of initial no-decrease period',
                    compared_cohort='Same 47 audited complete curves (43 with wind); no cohort expansion',
                    initial_plateau_policy='Remove only initial waiting time; retain all later integer-SOC step plateaus; do not count the removed first percentage point',
                    summary=summaries,counts={},
                    known_limitations=manifest['known_limitations'] + ['First drop can occur before all drones finish centering; separate sensitivity',
                       'The old 95%-start comparison already excluded initial plateau; updated High adds observed samples above 95%',
                       'Observed High above 95% is compared with the frozen line extended to 100%; no synthetic observations used'])
    OUT.mkdir(exist_ok=False)
    for name, df in [('paired_ratios',p),('stage_rates',pd.DataFrame(stage_rows)),('initial_plateau_audit',pd.DataFrame(audit)),
                     ('summary',pd.DataFrame(summaries)),
                     ('by_battery',pd.DataFrame([summary(g,b) for b,g in w.groupby('battery_id')]))]:
        df.to_csv(OUT/(name+'.csv'),index=False,float_format='%.12g')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(summaries).round(2).to_string(index=False))
    print('First drop/plateau details',w[['initial_soc','first_drop_soc','initial_plateau_excluded_s']].describe().round(2).to_string())
    print('First drop phases',w.first_drop_phase.value_counts().to_dict())
    print('Example',w[w.experiment_id.eq('wind_tunnel_front_75_tail_lv1_001')][
        ['drone_id','battery_id','initial_soc','first_drop_soc','initial_plateau_excluded_s','high_k','medium_k','low_k','high_difference_pct','low_difference_pct']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
