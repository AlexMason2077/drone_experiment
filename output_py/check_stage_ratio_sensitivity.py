"""Read-only robustness check: keep Low only while all five drones are airborne."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'output_py'))
from audit_wind_tunnel_stage_ratios import crossing, stage_stats, sha

OUT = ROOT / 'analysis_results/wind_tunnel_stage_ratio_check_20260909'


def calculate():
    pairs = pd.read_csv(OUT / 'paired_ratios.csv')
    stages = pd.read_csv(OUT / 'stage_rates.csv')
    rows = []
    for source, group in pairs.groupby('source'):
        d = pd.read_csv(ROOT / source, low_memory=False)
        for _, p in group.iterrows():
            g = d[d.drone_name.eq(p.drone_id)].sort_values('elapsed_time')
            s = stages[(stages.source.eq(source)) & stages.drone_id.eq(p.drone_id)].set_index('stage')
            low, medium = s.loc['low'], s.loc['medium']
            before = g[g.elapsed_time.between(low.start_s, min(low.end_s, p.first_landing_s))]
            base = dict(source=source, experiment_id=p.experiment_id, battery_id=p.battery_id,
                        drone_id=p.drone_id, wind_level=int(p.wind_level), first_landing_s=p.first_landing_s)
            if medium.end_s > p.first_landing_s or before.empty:
                rows.append(dict(**base, included=False, reason='Medium ends after first landing, or no pre-landing Low'))
                continue
            end_soc = float(before.battery.min())
            end = crossing(before, end_soc, low.start_s)
            if low.soc_upper - end_soc < 10:
                rows.append(dict(**base, included=False, reason='Less than 10 percentage points of pre-landing Low'))
                continue
            stat = stage_stats(g, low.start_s, end, low.soc_upper, end_soc,
                               low.baseline_rate_pp_min, low.Bideal_rate_pp_min / low.k,
                               p.all_ready_s, p.first_landing_s)
            if stat is None:
                rows.append(dict(**base, included=False, reason='Telemetry/duration quality failure'))
                continue
            rows.append(dict(**base, included=True, reason='Included', **stat,
                             medium_k=medium.k,
                             difference_pct=100 * (stat['k'] / medium.k - 1)))
    result = pd.DataFrame(rows)
    return pairs, stages, result


def stats(values):
    x = values.dropna()
    return dict(n=len(x), median_signed_pct=float(x.median()), median_absolute_pct=float(x.abs().median()),
                mean_absolute_pct=float(x.abs().mean()), min_pct=float(x.min()), max_pct=float(x.max()),
                within10=int(x.abs().le(10).sum()), within20=int(x.abs().le(20).sum()))


def main():
    pairs, stages, result = calculate()
    wind = pairs[pairs.wind_level.gt(0)]
    preland = result[result.included & result.wind_level.gt(0)]
    summary = {'pre_landing_low': stats(preland.difference_pct),
               'pre_landing_low_flights': int(preland.source.nunique()),
               'pre_landing_low_batteries': preland.battery_id.value_counts().to_dict(),
               'to30_low_withwind': stats(wind.low_to30_difference_pct)}
    for stage in ['high', 'low']:
        summary[stage + '_OLS_withwind'] = stats(wind[stage + '_ols_difference_pct'])
        summary[stage + '_pose_quality_withwind'] = stats(wind.loc[wind.all_stages_own_pad_near20_ge80, stage + '_difference_pct'])
    # Independent arithmetic checks from saved endpoints and source records.
    for _, s in stages.iterrows():
        expected = (s.soc_upper - s.soc_lower) * 60 / (s.end_s - s.start_s)
        assert np.isclose(expected, s.raw_rate_pp_min, rtol=1e-8)
        assert np.isclose(s.k, expected / s.baseline_rate_pp_min, rtol=1e-8)
    for stage in ['high', 'low']:
        assert np.allclose(wind[stage + '_difference_pct'], 100 * (wind[stage + '_k'] / wind.medium_k - 1))
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for source, expected in manifest['input_files_sha256'].items():
        assert sha(ROOT / source) == expected, source
    summary['validation'] = 'All 141 stage formulas, paired ratios, and audited source SHA256 hashes verified'
    result.to_csv(OUT / 'pre_landing_low_sensitivity.csv', index=False, float_format='%.12g')
    (OUT / 'sensitivity_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(preland[['experiment_id', 'battery_id', 'soc_upper', 'soc_lower', 'difference_pct']].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
