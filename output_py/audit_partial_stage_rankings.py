"""Read-only raw SOC rankings from deterministic within-stage partial windows."""
from pathlib import Path
import hashlib
import json
import math
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battery_normalization import BatteryNormalizer

MODEL = ROOT / 'analysis_results/battery_normalization_v3_with_b15_20260909/model.json'
OUT = ROOT / 'analysis_results/partial_stage_raw_rankings_20260909'
FIELDS = ['run_id', 'drone_name', 'battery_id', 'phase', 'elapsed_time', 'battery',
          'wind_speed', 'wind_direction', 'formation', 'inter_drone_distance_cm',
          'mid', 'mission_pad', 'x', 'y', 'h']
POLICIES = [(3, 'first'), (5, 'first'), (8, 'first'), (5, 'after_all_ready'), (5, 'second')]
SQL = '''WITH rates AS (
 SELECT *, 'endpoint' AS method, 60.0*(soc_start-soc_end)/ROUND(end_s-start_s,9) AS rate FROM windows
 UNION ALL
 SELECT *, 'ols' AS method, ols_rate AS rate FROM windows
), pairs AS (
 SELECT a.*, b.rate AS medium_rate, b.start_s AS medium_start_s, b.end_s AS medium_end_s
 FROM rates a JOIN rates b
 ON a.source=b.source AND a.drone_id=b.drone_id AND a.battery_id=b.battery_id
 AND a.threshold_pp=b.threshold_pp AND a.policy=b.policy AND a.method=b.method
 WHERE a.stage IN ('high','low') AND b.stage='medium' AND a.wind_level>0
), complete AS (
 SELECT *, COUNT(*) OVER (PARTITION BY source,stage,threshold_pp,policy,method) AS fleet_count
 FROM pairs
)
SELECT *, RANK() OVER (PARTITION BY source,stage,threshold_pp,policy,method ORDER BY rate DESC) AS stage_rank,
 RANK() OVER (PARTITION BY source,stage,threshold_pp,policy,method ORDER BY medium_rate DESC) AS medium_rank
FROM complete WHERE fleet_count=5
ORDER BY threshold_pp,policy,method,stage,source,drone_id'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_window(g, upper, lower, threshold, policy, ready):
    """Earliest drop-to-drop window; never cross bad telemetry or stage bounds."""
    valid = g.battery.between(lower, upper) & ~g.phase.str.contains('fault|recoverable_error|uncommanded', case=False, na=False)
    if policy == 'after_all_ready':
        valid &= g.elapsed_time.ge(ready)
    dt = g.elapsed_time.diff()
    dy = g.battery.diff()
    # Breaks split records, not time-compress them. Later integer flat steps remain.
    breaks = (~valid | ~valid.shift(fill_value=False) | dt.gt(5) | dt.le(0) | dy.abs().gt(2) | dy.gt(0))
    chunks = breaks.cumsum()
    selected = []
    for _, seg in g[valid].groupby(chunks[valid]):
        drops = seg.loc[seg.is_drop].index.to_numpy()
        for start in drops:
            if selected and g.at[start, 'elapsed_time'] < selected[0].elapsed_time.iloc[-1]:
                continue
            t0 = g.at[start, 'elapsed_time']; y0 = g.at[start, 'battery']
            ends = drops[(g.loc[drops, 'elapsed_time'].to_numpy() - t0 >= 10)
                         & (y0 - g.loc[drops, 'battery'].to_numpy() >= threshold)]
            if not len(ends):
                continue
            end = ends[0]
            window = seg.loc[start:end]
            if len(window) < 10:
                continue
            selected.append(window)
            if policy != 'second' or len(selected) == 2:
                return window.copy()
    return None


def fit_window(w):
    t = w.elapsed_time.to_numpy(float); y = w.battery.to_numpy(float)
    x = t-t[0]
    slope, intercept = np.polyfit(x, y, 1)
    residual = y-(intercept+slope*x)
    own = w.mid.eq(w.mission_pad) & w.mid.gt(0)
    duration = round(float(t[-1]-t[0]), 9)  # Remove subtraction noise, not telemetry precision.
    return dict(start_s=float(t[0]), end_s=float(t[-1]), duration_s=duration,
                soc_start=float(y[0]), soc_end=float(y[-1]), drop_pp=float(y[0]-y[-1]),
                raw_rate=60*float(y[0]-y[-1])/duration, ols_rate=float(-60*slope),
                ols_intercept=float(intercept), rmse_pp=float(np.sqrt(np.mean(residual**2))),
                n=len(w), max_gap_s=float(np.diff(t).max()), own_pad_fraction=float(own.mean()),
                near20_fraction=float((own & w.x.abs().le(20) & w.y.abs().le(20)).mean()),
                height_median_cm=float(w.h.median()))


def scan():
    norm = BatteryNormalizer.load(MODEL)
    registry_path = ROOT/'database/experiment_registry.json'
    registry = {r['experiment_id']: r for r in json.loads(registry_path.read_text())['experiments']}
    paths = sorted((ROOT/'database').glob('wind_tunnel*/*_all_coordination.csv'))
    hashes = {str(MODEL.relative_to(ROOT)): digest(MODEL), str(registry_path.relative_to(ROOT)): digest(registry_path)}
    windows = []; audit = []; seen = set()
    for p in paths:
        exp = p.parent.name; rec = registry.get(exp, {}); source = str(p.relative_to(ROOT))
        hashes[source] = digest(p)
        if 'prepare' in exp.lower() or 'merged' in p.name.lower() or rec.get('is_outlier'):
            audit.append(dict(source=source, status='excluded_prepare_merged_outlier')); continue
        data = pd.read_csv(p, usecols=lambda c: c in FIELDS, low_memory=False)
        if data.empty or not set(FIELDS).issubset(data.columns):
            audit.append(dict(source=source, status='empty_missing_columns')); continue
        if data.phase.str.contains('merge|simulat|interpol', case=False, na=False).any():
            audit.append(dict(source=source, status='synthetic_phase')); continue
        for col in ['elapsed_time', 'battery', 'mid', 'mission_pad', 'x', 'y', 'h']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        ready = data[data.phase.eq('wind_tunnel_hover')].groupby('drone_name').elapsed_time.min()
        if len(ready) != 5:
            audit.append(dict(source=source, status='not_all_five_reached_hover')); continue
        all_ready = float(ready.max())
        lands = data[data.phase.str.contains('land', case=False, na=False)]
        first_land = float(lands.elapsed_time.min()) if len(lands) else math.inf
        for (drone, battery), g in data.groupby(['drone_name', 'battery_id']):
            info = dict(source=source, experiment_id=exp, drone_id=drone, battery_id=battery, run_id=str(g.run_id.iloc[0]))
            if battery not in norm.curves or norm.pairs[battery] != drone:
                audit.append(dict(**info, status='unsupported_battery_pair')); continue
            ident = (info['run_id'], drone, battery)
            if ident in seen:
                audit.append(dict(**info, status='duplicate_run')); continue
            seen.add(ident)
            g = g.sort_values('elapsed_time').copy()
            takeoff = g[g.phase.eq('wind_tunnel_takeoff')]
            if takeoff.empty:
                audit.append(dict(**info, status='no_takeoff')); continue
            g = g[g.elapsed_time.ge(takeoff.elapsed_time.iloc[0])].copy()
            if g[['elapsed_time','battery']].isna().any().any() or not g.battery.between(0,100).all() or g.elapsed_time.duplicated().any():
                audit.append(dict(**info, status='invalid_samples')); continue
            ownland = g[g.phase.str.contains('land', case=False, na=False)]
            if len(ownland):
                g = g[g.elapsed_time.le(ownland.elapsed_time.iloc[0])].copy()
            g = g.reset_index(drop=True)
            g['is_drop'] = g.battery.diff().lt(0)
            level = re.search(r'(\d+)', str(rec.get('wind_speed', g.wind_speed.iloc[0])))
            level = int(level.group(1)) if level else 0
            if 'no_wind' in exp: level = 0
            info.update(wind_level=level, wind_direction=str(rec.get('wind_direction', g.wind_direction.iloc[0])).lower().replace(' wind',''),
                        formation=str(rec.get('formation',g.formation.iloc[0])).lower().replace('echalon','echelon').replace('echolon','echelon'),
                        spacing_cm=float(rec.get('inter_drone_distance_cm',g.inter_drone_distance_cm.iloc[0])),
                        initial_soc=float(g.battery.iloc[0]), all_ready_s=all_ready)
            boundaries = norm.curve_for(battery,drone).boundaries
            for i, stage in enumerate(['high','medium','low']):
                for threshold, policy in POLICIES:
                    w = select_window(g, boundaries[i], boundaries[i+1], threshold, policy, all_ready)
                    tag = dict(**info, stage=stage, threshold_pp=threshold, policy=policy,
                               stage_upper=boundaries[i], stage_lower=boundaries[i+1])
                    if w is None:
                        audit.append(dict(**tag,status='no_eligible_partial_window')); continue
                    stats = fit_window(w)
                    assert stats['drop_pp'] >= threshold and stats['duration_s'] >= 10
                    assert boundaries[i] >= stats['soc_start'] > stats['soc_end'] >= boundaries[i+1]
                    assert w.is_drop.iloc[0] and w.is_drop.iloc[-1]
                    assert np.diff(w.battery).max() <= 0 and stats['max_gap_s'] <= 5
                    windows.append(dict(**tag,**stats, starts_after_all_ready=stats['start_s']>=all_ready,
                                        ends_before_first_landing=stats['end_s']<=first_land))
                    audit.append(dict(**tag,status='included'))
    for src,h in hashes.items(): assert digest(ROOT/src)==h,src
    return pd.DataFrame(windows),pd.DataFrame(audit),hashes,len(paths)


def rank_windows(windows):
    assert not windows.duplicated(['source','drone_id','stage','threshold_pp','policy']).any()
    with sqlite3.connect(':memory:') as conn:
        windows.to_sql('windows', conn, index=False)
        ranked = pd.read_sql_query(SQL, conn)
    comparisons = []
    keys = ['threshold_pp','policy','method','stage','source']
    for key,g in ranked.groupby(keys):
        assert len(g)==5 and g.drone_id.nunique()==5
        assert np.allclose(g.rate.rank(ascending=False,method='min'),g.stage_rank)
        assert np.allclose(g.medium_rate.rank(ascending=False,method='min'),g.medium_rank)
        m = g.sort_values(['medium_rank','drone_id']); s = g.sort_values(['stage_rank','drone_id'])
        m_leaders = set(g[g.medium_rank.eq(1)].drone_id)
        s_leaders = set(g[g.stage_rank.eq(1)].drone_id)
        strict = len(m_leaders)==len(s_leaders)==1
        leader = m.iloc[0].drone_id
        current = g[g.drone_id.eq(leader)].iloc[0]
        def order_text(frame, rank_column):
            return ' > '.join(' = '.join(part.drone_id.str.replace('drone_','D')) for _,part in frame.groupby(rank_column,sort=True))
        comparisons.append(dict(zip(keys,key),experiment_id=g.experiment_id.iloc[0],
            medium_leader=' = '.join(sorted(m_leaders)),stage_leader=' = '.join(sorted(s_leaders)),unique_leaders=strict,
            same_fastest=bool(m_leaders & s_leaders),
            medium_leader_count=len(m_leaders),stage_leader_count=len(s_leaders),
            medium_order=order_text(m,'medium_rank'),
            stage_order=order_text(s,'stage_rank'),
            leader_new_rank=int(current.stage_rank),gap_pct=100*(s.rate.iloc[0]/g[g.drone_id.isin(m_leaders)].rate.max()-1),
            top_gap_pct=100*(s.rate.iloc[0]/s.rate.iloc[1]-1),
            min_stage_duration_s=float(g.duration_s.min()),min_stage_drop_pp=float(g.drop_pp.min()),
            all_stage_windows_after_ready=bool(g.starts_after_all_ready.all()),
            all_stage_windows_before_landing=bool(g.ends_before_first_landing.all())))
    comparisons = pd.DataFrame(comparisons)
    summary = comparisons.groupby(['threshold_pp','policy','method','stage']).agg(
        flights=('source','size'), unique_leader_flights=('unique_leaders','sum'), same_fastest=('same_fastest','sum'),
        within5=('gap_pct',lambda s: int(s.le(5).sum())),within10=('gap_pct',lambda s:int(s.le(10).sum())),
        near_ties_2pct=('top_gap_pct',lambda s:int(s.lt(2).sum()))).reset_index()
    return ranked, comparisons, summary


def main():
    assert not OUT.exists(), 'Use a new output directory, do not overwrite evidence.'
    windows,audit,hashes,count = scan()
    ranked,comparisons,summary = rank_windows(windows)
    OUT.mkdir(parents=True)
    for name,df in [('windows',windows),('source_audit',audit),('ranked_values',ranked),('comparisons',comparisons),('summary',summary)]:
        df.to_csv(OUT/(name+'.csv'),index=False,float_format='%.12g')
    # One flight/stage row, five drone rate columns; detail remains in windows.csv.
    main_windows = windows[(windows.threshold_pp==5)&(windows.policy=='first')&(windows.wind_level>0)]
    main_windows.pivot(index=['experiment_id','source','stage'],columns='drone_id',values='raw_rate').reset_index().to_csv(
        OUT/'raw_rates_wide.csv',index=False,float_format='%.12g')
    (OUT/'ranking.sql').write_text(SQL+';\n')
    manifest = dict(as_of='2026-09-09',raw_sources_modified=False,Bideal_modified=False,normalization_applied=False,
        input_hashes=hashes,files_scanned=count,
        metric='60 * observed SOC percentage-point drop / elapsed seconds, within own-battery stage boundaries',
        selection='Earliest contiguous drop-to-drop window >= threshold pp and >=10 seconds; >=10 samples. Initial plateau excluded; normal later flats retained.',
        primary='5 pp first window; 3 pp, 8 pp, after-all-ready and second non-overlapping window sensitivity',
        exclusions='prepare/merged/outlier/synthetic; unsupported calibrated battery/drone; no five-drone hover; invalid samples; windows split at faults, gaps>5s, upward SOC or jumps>2pp',
        caveats=['Local-window rate is not automatically the whole-stage average',
                 'Different batteries may be compared at different SOC subranges and elapsed times',
                 'Primary allows initial centering; after-all-ready sensitivity isolates that selection',
                 'Low can occur after another drone lands; flags retained',
                 'One source flight counts once per stage comparison, not once per time window',
                 '3/5/8 pp thresholds are analysis conventions, not statistical confidence thresholds'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(summary.to_string(index=False))
    print('Primary individual windows:',main_windows.groupby('stage').size().to_dict())


if __name__=='__main__': main()
