"""Package reviewed results, an executed audit notebook, and a canonical report."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import io
import json
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'analysis_results/wind_tunnel_stage_ratio_check_20260909'
OUT = BASE / 'first_drop_recalculation'


def records(df):
    return json.loads(df.to_json(orient='records', force_ascii=False, double_precision=10))


def main():
    p = pd.read_csv(OUT / 'paired_ratios.csv')
    # Material recomputation in SQLite supplies the report reader's SQL-native
    # provenance contract. This is executed, not an invented display query.
    sql = '''WITH rates AS (
      SELECT source, drone_id, stage,
        (soc_upper - soc_lower) * 60.0 / (end_s - start_s) / baseline_rate_pp_min AS k
      FROM wind_tunnel_stage_rates
    )
    SELECT h.source, h.drone_id, h.k AS high_k, m.k AS medium_k, l.k AS low_k,
      100.0 * (h.k / m.k - 1.0) AS high_difference_pct,
      100.0 * (l.k / m.k - 1.0) AS low_difference_pct
    FROM rates h
    JOIN rates m ON h.source=m.source AND h.drone_id=m.drone_id AND m.stage='medium'
    JOIN rates l ON h.source=l.source AND h.drone_id=l.drone_id AND l.stage='low'
    WHERE h.stage='high'
    ORDER BY h.source, h.drone_id'''
    with sqlite3.connect(':memory:') as conn:
        pd.read_csv(OUT/'stage_rates.csv').to_sql('wind_tunnel_stage_rates',conn,index=False)
        checked = pd.read_sql_query(sql,conn)
    keys=['source','drone_id']
    computed=[c for c in checked if c not in keys]
    validation=p.merge(checked,on=keys,suffixes=('_saved','_sql'),validate='one_to_one')
    for col in computed:
        assert ((validation[col+'_saved']-validation[col+'_sql']).abs()<1e-7).all()
    p=p.drop(columns=computed).merge(checked,on=keys,validate='one_to_one')
    w = p[p.wind_level.gt(0)].copy()
    from audit_wind_tunnel_stage_ratios import summary as make_summary
    b = pd.DataFrame([make_summary(g,battery) for battery,g in w.groupby('battery_id')])
    sens = json.loads((BASE / 'sensitivity_summary.json').read_text())
    sample = w[w.experiment_id.eq('wind_tunnel_front_75_tail_lv1_001')].copy()
    sample['drone_battery'] = sample.drone_id.str.replace('drone_', 'D') + ' / ' + sample.battery_id
    sample['high_ratio'] = sample.high_k / sample.medium_k
    sample['low_ratio'] = sample.low_k / sample.medium_k
    summary = []
    chart = []
    for band, label in [('high', 'High'), ('low', 'Low')]:
        x = w[band + '_difference_pct']
        summary.append(dict(stage=label, curves=len(w), flights=w.source.nunique(),
                            median_absolute_pct=x.abs().median(), mean_absolute_pct=x.abs().mean(),
                            median_signed_pct=x.median(), min_pct=x.min(), max_pct=x.max(),
                            within10=f'{x.abs().le(10).sum()}/{len(x)}'))
        for _, row in b.iterrows():
            chart.append(dict(battery=row['group'], stage=label, curves=int(row.drone_curves),
                              flights=int(row.flights), median_absolute_pct=row[band + '_median_absolute_pct'],
                              mean_absolute_pct=row[band + '_mean_absolute_pct'],
                              median_signed_pct=row[band + '_median_signed_pct'],
                              within10_count=int(row[band + '_within10_count']),
                              within20_count=int(row[band + '_within20_count'])))
    selected = ['experiment_id', 'run_id', 'drone_id', 'battery_id', 'wind_direction', 'wind_level', 'formation',
                'spacing_cm', 'high_k', 'medium_k', 'low_k', 'high_difference_pct', 'low_difference_pct',
                'high_duration_s', 'medium_duration_s', 'low_duration_s', 'low_after_first_landing_fraction']
    snapshots = {'summary': records(pd.DataFrame(summary)), 'battery_comparison': chart,
                 'example': records(sample[['drone_battery', 'high_ratio', 'low_ratio', 'high_difference_pct',
                                            'low_difference_pct', 'high_raw_rate_pp_min', 'medium_raw_rate_pp_min',
                                            'low_raw_rate_pp_min']]),
                 'reviewed_pairs': records(w[selected]),
                 'preland': records(pd.read_csv(BASE / 'pre_landing_low_sensitivity.csv').query('included and wind_level > 0')),
                 'plateaus': records(pd.read_csv(OUT / 'initial_plateau_audit.csv'))}
    source = dict(id='stage_audit', label='Wind-tunnel stage-ratio audit · 2026-09-09',
                  path='analysis_results/wind_tunnel_stage_ratio_check_20260909/first_drop_recalculation/manifest.json',
                  query=dict(engine='SQLite (in-memory); Python source extraction and summary statistics', language='sql',
                             description='Executed SQL recomputes stage rates, baseline-normalized k and paired relative deviations from the audited endpoints. wind_tunnel_stage_rates is loaded from first_drop_recalculation/stage_rates.csv. Python then computes medians/counts for the report. Source inventory and additional pre-landing sensitivity are preserved in the audit manifest and notebook.',
                             sql=sql,
                             tables_used=['wind_tunnel_stage_rates','database/experiment_registry.json',
                                          'analysis_results/battery_normalization_v3_with_b15_20260909/model.json',
                                          'analysis_results/wind_tunnel_stage_ratio_check_20260909/first_drop_recalculation/stage_rates.csv',
                                          'analysis_results/wind_tunnel_stage_ratio_check_20260909/first_drop_recalculation/paired_ratios.csv',
                                          'analysis_results/wind_tunnel_stage_ratio_check_20260909/pre_landing_low_sensitivity.csv'],
                             filters=['No prepare, merged/synthetic, or registry-outlier flights',
                                      'Known battery/drone calibration pair; complete own-hover 95–20% trace',
                                      'Main results: wind level > 0; no-wind flight reported separately'],
                             metric_definitions={'rate': 'SOC drop in percentage points / first-crossing duration in seconds * 60',
                                                 'k': 'observed stage rate / frozen own-battery baseline rate for that stage',
                                                 'difference_pct': '100 * (k_stage / k_medium - 1)',
                                                 'typical_deviation': 'Median of absolute relative deviations across eligible drone-flight curves'},
                             executed_at=datetime.now(timezone.utc).isoformat()))
    blocks = []
    def md(id, body, backed=True):
        block = dict(id=id, type='markdown', body=body)
        if backed:
            block['sourceId'] = 'stage_audit'
        blocks.append(block)
    title = 'Wind-tunnel ratios after first SOC decrease'
    md('title', '# ' + title, False)
    md('summary_text', '## 结论：不能把三段耗电比例视为完全相同\n\n'
       '按要求去掉起飞后首次显示掉电之前的初始平台。在 **16 次有风飞行的 43 条完整单机曲线**中，按 Medium 和各电池自身的基线比例推算：High 的典型绝对偏差为 **23.4%**，Low 为 **7.5%**。'
       'Low 全段结果更接近比例相同，但只保留五架都未降落的 Low 部分时，典型偏差为 **12.2%**（37 条曲线、12 次飞行）。'
       '因此，High 的统一比例缺乏支持；Low 可以作为近似假设继续检验，不能当作已经证明的恒定关系。')
    md('definitions', '## 比较的“比例”是什么\n\n'
       '本次检验的不是五架无人机原始电量百分比下降速度是否相等，而是：**同一次飞行、同一位置的耗电负载倍率，在 High、Medium、Low 是否保持不变**。\n\n'
       '`k阶段 = 实测阶段耗电率 ÷ 该电池自身基线阶段耗电率`\n\n'
       '`相对偏差(%) = (k阶段 ÷ kMedium − 1) × 100`\n\n'
       '0% 表示与 Medium 按基线比例推算完全一致；−20% 表示实测值比该推算低 20%。这里的“%”是耗电率的相对偏差，不是电池多掉了几个百分点。'
       '等价地，若三个 k 相等，统一到当前 Bideal 后，High/Medium 应为 2.213，Low/Medium 应为 1.605。')
    md('scope', '## 样本与区间\n\n'
       '检查 163 个 wind-tunnel 协调记录文件，筛出 17 次飞行的 47 条完整单机曲线；其中 16 次有风飞行、43 条曲线为主分析，另 1 次无风飞行、4 条曲线单列检查。'
       '有风样本的运行日期范围为 2026-08-29 至 2026-09-08。43 条曲线不是 43 次独立实验，同一次飞行中的无人机结果有关联。\n\n'
       '使用每块电池自身的分界点。High 从起飞后电量第一次实际下降的样本算到各自上分界点；Medium 为两个分界点之间；Low 为下分界点到 20%。'
       '43 条有风曲线最初都显示 100%，首次下降都显示 99%，所以 High 实际从 99% 算起，不能把被删掉的第一个百分点计入分子。'
       '只排除起始平台，保留之后所有正常的整数阶梯平台；使用真实的 99%–95% 记录，不制造补点。冻结基线在 95% 以上仍是原先延伸的直线，本次是用真实记录检查它。'
       'B10/B11/B12/B13/B14 的基线保持冻结；B06 缺少匹配校准，不纳入。')
    md('plateau_policy', '## 初始平台已逐机排除，不删中途正常阶梯\n\n'
       '起飞阶段开始到首次电量下降的等待时间为 **9.2–53.0 秒**，中位数 **19.6 秒**。首次下降的样本时刻作为新的计时起点。'
       '例如 Front / 75 cm / Tail / Lv1 / 001：D1/B11 排除 27.414 秒，D2/B10 排除 18.925 秒。'
       '此前从 95% 开始的比较本来也没有包含初始满电平台；本次变化主要是加入真实的 99%–95% 段，而不是再次压缩 Medium 或 Low 的时间。'
       '这是显示 SOC 的耗电率计算口径，不表示平台期间真实物理耗电为零。')
    md('findings', '## High 的比例偏差明显大于 Low\n\n'
       '以下典型偏差使用绝对值中位数，避免正负偏差相互抵消。High 仅 10/43 条在 ±10% 内，Low 为 35/43 条。'
       'High 的有符号偏差范围为 −71.4% 至 +72.0%；Low 为 −16.9% 至 +9.4%。这些范围是样本范围，不是置信区间。')
    blocks.append(dict(id='overall_table', type='table', tableId='overall'))
    blocks.append(dict(id='battery_chart', type='chart', chartId='battery_deviation'))
    md('example_note', '## 一次完整五机实验的例子\n\n'
       '`wind_tunnel_front_75_tail_lv1_001`（20260903_172757）。下面把各机 Medium 的 k 设为 1，列出 High、Low 相对它的倍率。'
       '如 D1/B11：High 只有应有比例的约 0.66 倍（−33.9%），Low 为约 0.96 倍（−3.7%）。同一架飞机的 High 与 Low 并不一致。'
       '该实验只有 D5 在自己结束 Low 之前始终保持五架都未降落，其他四架的完整 Low 包含了队友先降落后的时间。')
    blocks.append(dict(id='example_table', type='table', tableId='example'))
    md('robustness', '## 低电量结果受比较窗口影响\n\n'
       '43 条有风曲线中，38 条 Low 包含其他无人机先降落后的时间。为检查这一影响，再截取第一次队友降落之前的 Low，要求至少有 10 个百分点真实下降量，且 Medium 也在首次降落前结束。'
       '保留 37 条曲线、12 次飞行：绝对偏差中位数 **12.2%**，平均绝对偏差 **12.1%**；16/37 在 ±10% 内，31/37 在 ±20% 内。'
       '这一检查同时缩短了 Low 区间，不能把结果变化全部归因于队友降落。\n\n'
       '使用阶段内所有样本的直线拟合斜率复算，有风样本 High 的绝对偏差中位数为 19.7%，Low 为 6.7%；限制各阶段至少 80% 样本在自身 Pad 的 x/y ±20 cm 内后，High 为 23.9%，Low 为 7.6%。'
       'High 不稳定的结论没有因为这两种检查而消失。\n\n'
       '首次掉电时有 12 条曲线尚未完成五机初始回正。仅保留首次掉电已在五机首次 hover 之后的 31 条曲线（14 次飞行），High 绝对偏差中位数仍为 **19.7%**。')
    md('methods', '## 计算与复核方法\n\n'
       '使用边界 SOC 的首次实际到达时间，按阶段电量下降量除以时长计算平均耗电率，无插值、无补点。'
       '每段至少 10 秒和 10 条样本，最大采样间隔不超过 5 秒；排除重复时间、缺失电量、明显电量跳变以及完整比较期间的控制故障标记。'
       '按用户要求，新的 High 起点是首次实际掉电，可能包含尚未完成的起飞回正；该影响用独立敏感性分析检查。排除该机自身降落后的记录。\n\n'
       '共复核 141 个阶段的计算式、所有阶段倍率公式及读取来源的 SHA256；原始实验文件、Bideal 和控制代码均未修改。'
       '结果可由同目录中的 notebook 及逐曲线 CSV 复核。下面展示按实验及无人机排序的前 10 条主分析数据，完整 43 条保存于结果文件。')
    snapshots['preview'] = snapshots['reviewed_pairs'][:10]
    blocks.append(dict(id='preview_table', type='table', tableId='preview'))
    md('limitations', '## 当前证据能说明什么，不能说明什么\n\n'
       '- 可以说明：当前冻结的电池基线在这些 wind-tunnel 记录中，不能用一套完全不变的倍率准确解释三个 SOC 阶段。\n'
       '- 不能仅凭本次结果确定原因：基线跨日期差异、High 曲线形状、风扇实际开启时点、纠偏动作和 SOC 显示特性都可能有关。无风对照仅 1 次，无法分辨各因素。\n'
       '- 完整曲线筛选会遗漏高段在回正前已结束或中途失败的飞行，尤其 B13 只有 3 条有风完整曲线，不能把它当成所有实验的代表。\n'
       '- 本次不覆盖首次 100%→99% 下降的真实发生区间，也不覆盖 20% 以下；wind tunnel 的结论不能直接视为 2.5 m 前进实验的证明。')
    md('next_steps', '## 对耗电率表的影响\n\n'
       '现阶段不宜把仅由 Medium 乘固定比例得到的 High/Low 耗电率当作已经由真实实验验证的数值。'
       '优先使用同条件的实际分段结果，并在整个群体仍悬停的共同窗口内复核 Low；缺少覆盖时保留缺失或明确列为模型推算。'
       '这次只做核查，不改冻结表，也不重拟合 Bideal。')
    md('questions', '## 下一步需要确认的问题\n\n'
       '是否在 High 开始之前已经开启风扇并稳定？是否有可核对的风扇开启时刻？同一条件是否有多次完整曲线可估计重复性？'
       '若要专门检验五个位置之间的耗电“份额”是否一致，需要另设同一时窗、五架同时在空中的比较；不能与本次的阶段基线倍率检验混为一谈。', False)
    def table(id, dataset, title, columns, sort):
        return dict(id=id, title=title, dataset=dataset, sourceId='stage_audit', defaultSort=dict(field=sort, direction='asc'),
                    columns=[dict(field=f, label=l, **({'type':'text'} if t else {'format':'number'})) for f,l,t in columns])
    tables = [table('overall', 'summary', '有风完整曲线的阶段比例偏差',
                    [('stage','阶段',True),('curves','曲线数',False),('median_absolute_pct','典型绝对偏差 (%)',False),
                     ('mean_absolute_pct','平均绝对偏差 (%)',False),('within10','±10% 内',True)], 'stage'),
              table('example','example','Front / 75 cm / Tail / Lv1 / 001',
                    [('drone_battery','无人机 / 电池',True),('high_ratio','High 相对倍率',False),('low_ratio','Low 相对倍率',False),
                     ('high_difference_pct','High 偏差 (%)',False),('low_difference_pct','Low 偏差 (%)',False)],'drone_battery'),
              table('preview','preview','逐曲线复核预览（前 10 条）',
                    [('experiment_id','实验',True),('drone_id','无人机',True),('battery_id','电池',True),
                     ('high_k','High k',False),('medium_k','Medium k',False),('low_k','Low k',False)],'experiment_id')]
    chart_spec = dict(id='battery_deviation', title='各电池 High、Low 相对 Medium 的比例偏差',
                      subtitle='有风完整曲线；绝对偏差中位数 (%)；B10 n=10、B11 n=15、B12 n=8、B13 n=3、B14 n=7',
                      type='bar', dataset='battery_comparison', sourceId='stage_audit', valueFormat='number', unit='%',
                      intent='comparison', layout='full',
                      encodings=dict(x=dict(field='battery',type='nominal',label='电池'),
                                     y=dict(field='median_absolute_pct',type='quantitative',label='绝对相对偏差中位数 (%)'),
                                     color=dict(field='stage',type='nominal',label='阶段'),
                                     tooltip=[dict(field='curves',type='quantitative',label='完整单机曲线数'),
                                              dict(field='mean_absolute_pct',type='quantitative',label='平均绝对偏差 (%)')]),
                      settings=dict(groupMode='grouped',orientation='vertical',sort='none',showValues=True),
                      palette=dict(kind='categorical'),legend=dict(position='bottom',sort='spec'),labels=dict(values='all'))
    generated=datetime.now(timezone.utc).isoformat()
    payload=dict(surface='report', manifest=dict(version=1,surface='report',title=title,generatedAt=generated,
                                               blocks=blocks,charts=[chart_spec],tables=tables,sources=[source]),
                 snapshot=dict(version=1,status='ready',generatedAt=generated,datasets=snapshots),sources=[source])
    (OUT/'artifact.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    # Notebook cells are executed sequentially here, not via a Jupyter kernel.
    cells=[]
    cells.append(dict(cell_type='markdown',id='purpose',metadata={},source=[
        '# Wind-tunnel stage-ratio audit\n',
        'Read-only reproduction and formula checks. Cells were executed in order with Python; a Jupyter kernel was not available.\n',
        'All source values remain observed. High starts at the first post-takeoff decrease (99% in the with-wind cohort); later integer-step plateaus remain included. Frozen baseline slopes are not refitted.\n']))
    code_cells=[
        "from pathlib import Path\nimport sys, json\nimport pandas as pd\nimport numpy as np\n"
        "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'battery_normalization.py').exists())\n"
        "BASE = ROOT/'analysis_results/wind_tunnel_stage_ratio_check_20260909'\n"
        "OUT = BASE/'first_drop_recalculation'\n"
        "sys.path.insert(0,str(ROOT/'output_py'))\nfrom check_stage_ratio_sensitivity import calculate, stats\n"
        "from audit_wind_tunnel_stage_ratios import sha\n"
        "_, _, preland = calculate()\npairs=pd.read_csv(OUT/'paired_ratios.csv')\nstages=pd.read_csv(OUT/'stage_rates.csv')\nwind = pairs[pairs.wind_level.gt(0)]\n"
        "print('Wind flights:', wind.source.nunique(), 'complete drone curves:', len(wind))\n",
        "for source, rows in stages.groupby('source'):\n"
        "    raw=pd.read_csv(ROOT/source,usecols=['drone_name','elapsed_time','battery'])\n"
        "    for _,s in rows.iterrows():\n"
        "        g=raw[raw.drone_name.eq(s.drone_id)]\n"
        "        for t,soc in [(s.start_s,s.soc_upper),(s.end_s,s.soc_lower)]:\n"
        "            endpoint=g[np.isclose(g.elapsed_time,t,rtol=0,atol=1e-6)]\n"
        "            assert len(endpoint)==1 and endpoint.battery.iloc[0]==soc\n"
        "        expected=(s.soc_upper-s.soc_lower)*60/(s.end_s-s.start_s)\n"
        "        assert np.isclose(expected,s.raw_rate_pp_min,rtol=1e-8)\n"
        "        assert np.isclose(expected/s.baseline_rate_pp_min,s.k,rtol=1e-8)\n"
        "manifest=json.loads((OUT/'manifest.json').read_text())\n"
        "for source, expected in manifest['input_files_sha256'].items():\n"
        "    assert sha(ROOT/source)==expected\n"
        "print('All 282 raw SOC/time endpoints, 141 rates and source hashes passed.')\n",
        "for stage in ['high','low']:\n"
        "    recomputed=100*(wind[stage+'_k']/wind.medium_k-1)\n"
        "    assert np.allclose(recomputed,wind[stage+'_difference_pct'])\n"
        "    print(stage,stats(recomputed))\n"
        "print('Low before first swarm landing:',stats(preland.loc[preland.included & preland.wind_level.gt(0),'difference_pct']))\n"
        "print(wind[['experiment_id','drone_id','battery_id','high_difference_pct','low_difference_pct']].head(10).to_string(index=False))\n"
    ]
    ns={}
    for count,code in enumerate(code_cells,1):
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(code,'stage_ratio_audit.ipynb','exec'),ns)
        cells.append(dict(cell_type='code',id=f'check-{count}',metadata={},source=code.splitlines(keepends=True),
                          execution_count=count,outputs=[dict(output_type='stream',name='stdout',text=capture.getvalue().splitlines(keepends=True))]))
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':sys.version.split()[0]},
        'execution_note':'Executed sequentially in Python; not validated in a Jupyter kernel.'})
    (OUT/'stage_ratio_audit.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')
    print('Created canonical artifact and executed notebook. Raw endpoint checks passed.')


if __name__ == '__main__':
    import sys
    main()
