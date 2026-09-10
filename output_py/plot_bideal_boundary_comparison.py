"""Prepare five auditable native line charts; no raw/model/flight-code writes.

Chart contract: one SOC-time overlay per battery, actual samples plus the
four endpoint anchors of each continuous three-line approximation. These
five repeated line views intentionally answer the same shape-comparison
question. Actual samples use solid lines, 75/40 dashed, 80/56 dotted. Native
widget categorical colors are renderer-owned; line style and labels carry
the distinction without relying on color. Preserve every change of SOC and
its preceding sample; removing only collinear plateau interiors leaves the
observed polyline unchanged. No smoothing, invented samples or extrapolation.
"""
from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd

from search_bideal_boundaries import ROOT, PREVIOUS, OUT, load_trace, prepare, fit, PAIRS

DEST = OUT / 'curve_comparison'
QUERY = """SELECT elapsed_time, battery AS observed_soc
FROM raw_baseline_csv
WHERE phase = 'hover_to_10_percent'
ORDER BY elapsed_time"""


def main():
    sources = pd.read_csv(PREVIOUS / 'coefficients.csv').drop_duplicates(
        ['battery_id', 'run_id']).sort_values('battery_id')
    previous = pd.read_csv(OUT / 'comparison.csv')
    assert len(sources) == 5
    assert not sources.run_id.astype(str).eq('20260906_165812').any()
    DEST.mkdir(parents=True, exist_ok=True)
    all_anchors, all_rows, checks = [], [], []
    columns = [
        {'key': 't_s', 'label': '从95%起的时间（秒）', 'type': 'number'},
        {'key': 'soc', 'label': 'SOC（%）', 'type': 'number'},
        {'key': 'series', 'label': '曲线', 'type': 'text'},
        {'key': 'line_style', 'label': '线型', 'type': 'text'},
        {'key': 'point_label', 'label': '点说明', 'type': 'text'},
        {'key': 'battery_id', 'label': '电池', 'type': 'text'},
        {'key': 'drone', 'label': '无人机', 'type': 'text'},
        {'key': 'run_id', 'label': '实验', 'type': 'text'},
        {'key': 'elapsed_time', 'label': '原始elapsed_time（秒）', 'type': 'number'},
        {'key': 'observed_soc', 'label': '真实SOC（%）', 'type': 'number'},
        {'key': 'fit_75_40', 'label': '75/40在此刻的SOC（%）', 'type': 'number'},
        {'key': 'fit_80_56', 'label': '80/56在此刻的SOC（%）', 'type': 'number'},
        {'key': 'kind', 'label': '原始观测或模型锚点', 'type': 'text'},
    ]
    for src in sources.itertuples():
        assert PAIRS[src.battery_id] == src.drone
        trace = load_trace(src)  # verifies original SHA256
        with sqlite3.connect(':memory:') as db:
            pd.read_csv(ROOT / src.source).to_sql('raw_baseline_csv', db, index=False)
            queried = pd.read_sql_query(QUERY, db)
        assert np.array_equal(queried.elapsed_time.to_numpy(), trace['t'])
        assert np.array_equal(queried.observed_soc.to_numpy(), trace['soc'])
        p = prepare(trace)
        fits = [('75% / 40%', 'dashed', fit(p, 75, 40), [95, 75, 40, 20], 'original'),
                ('80% / 56%', 'dotted', fit(p, 80, 56), [95, 80, 56, 20], 'best')]
        t, s, origin = p['t'], p['soc'], p['t'][0]
        transitions = np.flatnonzero(np.diff(s) != 0) + 1
        keep = set([0, len(t) - 1, *transitions.tolist(), *(transitions - 1).tolist()])
        for _, _, f, _, _ in fits:
            keep.update(np.flatnonzero(np.isin(t, f['anchor_times'])).tolist())
        keep = np.array(sorted(keep))
        # Exact polyline preservation, not approximate downsampling.
        np.testing.assert_allclose(np.interp(t, t[keep], s[keep]), s, atol=1e-10)

        def base(i):
            return dict(battery_id=src.battery_id, drone=src.drone, run_id=src.run_id,
                        t_s=float(t[i] - origin), elapsed_time=float(t[i]),
                        observed_soc=float(s[i]), fit_75_40=float(fits[0][2]['pred'][i]),
                        fit_80_56=float(fits[1][2]['pred'][i]))

        rows = [dict(**base(i), soc=float(s[i]), series='真实记录', line_style='solid',
                     kind='observed', point_label=f'真实记录 {s[i]:.0f}%') for i in keep]
        for label, style, f, levels, model in fits:
            expected = previous[(previous.battery_id == src.battery_id) & (previous.model == model)].iloc[0]
            assert abs(f['rmse'] - expected.rmse_pp) < 1e-10
            for j, (time, level) in enumerate(zip(f['anchor_times'], levels)):
                i = int(np.flatnonzero(t == time)[0])
                assert s[i] == level
                description = ('共同起点' if j == 0 else '共同终点' if j == 3 else '分界点')
                rows.append(dict(**base(i), soc=float(level), series=label, line_style=style,
                                 kind='model_anchor', point_label=f'{label} {description} {level}%'))
                all_anchors.append(dict(battery_id=src.battery_id, drone=src.drone,
                    run_id=src.run_id, model=label, anchor=description, soc=level,
                    t_s=float(time - origin), elapsed_time=float(time), source=src.source))
        assert len(rows) < 2000
        old, new = fits[0][2]['rmse'], fits[1][2]['rmse']
        payload = dict(
            title=f'{src.battery_id} / {src.drone.replace("drone_", "D")}：真实SOC与两套三段直线',
            subtitle=f'95%→20%；75/40误差 {old:.3f}，80/56误差 {new:.3f} 个百分点。两套模型各4个点，连接后即为三段直线。',
            source=dict(id='baseline_' + src.battery_id, label=f'{src.battery_id}无风单机Hover，2026-09-06',
                path=str(ROOT / src.source), query=dict(sql=QUERY, engine='SQLite (raw CSV loaded unchanged)',
                    language='sql', tables_used=['raw_baseline_csv'],
                    description='真实悬停样本：按首次95%至首次20%裁剪；每套模型以实际SOC首次经过四个端点的时刻连成三段直线。可复现变换：output_py/plot_bideal_boundary_comparison.py。',
                    filters=['phase=hover_to_10_percent', '2026-09-06 current battery/airframe pair',
                             'B12/D2 run 20260906_165812 excluded', '95% to 20%; no extrapolation'],
                    metric_definitions=['SOC is reported percentage, not an energy measurement.',
                        'Time zero = first observed 95%. No rescaling or time-gap compression.',
                        'Only collinear plateau interiors omitted; displayed observed polyline exactly preserves raw samples.',
                        'RMSE computed over all original samples, not the reduced chart vertices.',
                        'Model anchors are observed threshold crossings, not synthetic measurements.'])),
            table=dict(columns=columns, rows=rows, row_count=len(rows), truncated=False,
                       original_sample_count=len(t)),
            chart=dict(type='line', fields=dict(
                x=dict(field='t_s', type='quantitative', label='从95%起的时间（秒）'),
                y=dict(field='soc', type='quantitative', aggregate='none', label='SOC（%）'),
                color=dict(field='series', type='nominal', label='曲线'),
                lineStyle=dict(field='line_style', type='nominal'),
                label=dict(field='point_label', type='nominal')),
                options=dict(points='always')),
            display=dict(unit='%', baseline=20, controls=True,
                         x_axis_title='从首次95%开始的悬停时间（秒）', y_axis_title='SOC（%）'))
        (DEST / f'{src.battery_id}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        all_rows.extend(rows)
        checks.append(dict(battery_id=src.battery_id, raw_samples=len(t), plotted_rows=len(rows),
                           real_polyline_preserved=True, anchors_checked=8, old_rmse=old, new_rmse=new))
    pd.DataFrame(all_anchors).to_csv(DEST / 'anchor_points.csv', index=False)
    pd.DataFrame(all_rows).to_csv(DEST / 'plotted_values.csv', index=False)
    (DEST / 'validation.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    (DEST / 'README.md').write_text('''# 五块电池的真实SOC与两套三段直线

数据：2026-09-06无风、相同设定高度的单机Hover。B10/D2、B11/D1、B12/D5、B13/D3、B14/D4；B12/D2的20260906_165812不采用。

每个原生交互图叠加三条曲线：真实记录（实线）、75%/40%（虚线）、80%/56%（点线）。模型各有4个标记点：共同95%起点、各自两个分界点、共同20%终点。95%与20%的模型标记完全重叠是正常的。鼠标移到点上可查看SOC和时刻。

横轴从该电池首次记录95%开始；没有平移SOC、拉伸时间、删除中断或补造数据。为降低标记密度，只去掉水平平台内部的共线点，逐点验证折线形状与全部原始数据一致。误差仍用所有原始样本计算。

这里比较的是每块电池自身的两种三段近似，并非把同一条平均Bideal硬套给五块电池。两套方案均使用端点连接法，不是另外换成最小二乘回归。

`anchor_points.csv`给出全部40个模型锚点；`plotted_values.csv`保留绘图值和对应原始时刻；原始CSV与既有模型保持不变。
''')
    print(pd.DataFrame(checks).to_string(index=False))
    print(pd.DataFrame(all_anchors).to_string(index=False))


if __name__ == '__main__':
    main()
