"""Package the matched wind-level analysis without modifying the prior report."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import io
import json
import sqlite3
import sys

import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'output_py'))
from diagnose_wind_strength_stage_ratios import OUT, PRIOR, SQL, describe


def records(df):return json.loads(df.to_json(orient='records',force_ascii=False,double_precision=10))


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    full=pd.read_csv(PRIOR/'paired_ratios.csv')
    expanded=pd.read_csv(OUT/'expanded_stage_pairs.csv')
    matched=pd.read_csv(OUT/'matched_full_curves.csv')
    extended_sql='''SELECT a.wind_direction,a.formation,a.spacing_cm,a.battery_id,a.drone_id,a.stage,
      a.source AS source_lv1,b.source AS source_lv2,a.ratio AS ratio_lv1,b.ratio AS ratio_lv2,
      100.0*(b.ratio/a.ratio-1.0) AS change_pct
      FROM expanded_stage_pairs a JOIN expanded_stage_pairs b
      ON a.wind_direction=b.wind_direction AND a.formation=b.formation AND a.spacing_cm=b.spacing_cm
      AND a.battery_id=b.battery_id AND a.drone_id=b.drone_id AND a.stage=b.stage
      WHERE a.wind_level=1 AND b.wind_level=2
      ORDER BY a.stage,a.wind_direction,a.formation,a.battery_id'''
    with sqlite3.connect(':memory:') as conn:
        full.to_sql('full_curves',conn,index=False)
        expanded.to_sql('expanded_stage_pairs',conn,index=False)
        checked=pd.read_sql_query(SQL,conn)
        extended=pd.read_sql_query(extended_sql,conn)
    for col in ['high_change_pct','low_change_pct']:
        assert np.allclose(checked[col],matched[col])
    assert len(extended)==36
    labels={'head':'Head','side':'Side','tail':'Tail'}
    matched['comparison']=matched.apply(lambda r:f'{labels[r.wind_direction]} · {r.formation.title()} · {r.battery_id}',axis=1)
    chart=[]
    for _,r in matched.iterrows():
        for stage in ['high','low']:
            chart.append(dict(comparison=r.comparison,stage='High/Medium' if stage=='high' else 'Low/Medium',
                              change_pct=r[stage+'_change_pct'],ratio_lv1=r[stage+'_ratio_lv1'],ratio_lv2=r[stage+'_ratio_lv2'],
                              battery=r.battery_id,wind_direction=r.wind_direction,formation=r.formation,spacing_cm=r.spacing_cm,
                              experiment_lv1=r.experiment_lv1,experiment_lv2=r.experiment_lv2,
                              both_high_after_ready=bool(r.both_high_after_ready)))
    condition=[]
    for (stage,wind,formation,spacing),g in extended.groupby(['stage','wind_direction','formation','spacing_cm']):
        condition.append(dict(stage=stage,condition=f'{formation.title()} / {labels[wind]} / {int(spacing)} cm',
                              pairs=len(g),median_change_pct=g.change_pct.median(),min_change_pct=g.change_pct.min(),
                              max_change_pct=g.change_pct.max(),positive=int(g.change_pct.gt(0).sum()),negative=int(g.change_pct.lt(0).sum())))
    condition=pd.DataFrame(condition)
    source_root='analysis_results/wind_strength_stage_ratio_diagnosis_20260909'
    generated=datetime.now(timezone.utc).isoformat()
    sources=[dict(id='primary',label='Matched complete wind-tunnel curves',path=source_root+'/matched_comparison.sql',
                  query=dict(engine='SQLite (in-memory)',language='sql',sql=SQL,executed_at=generated,
                             description='full_curves is loaded from the frozen first-drop recalculation paired_ratios.csv. Matches the same wind direction, formation, spacing, drone and battery across Lv1 and Lv2. SOC-stage ratios and between-level changes are computed by this executed query.',
                             tables_used=['full_curves','analysis_results/wind_tunnel_stage_ratio_check_20260909/first_drop_recalculation/paired_ratios.csv'],
                             filters=['Complete 99%–20% observed curves from the prior audited cohort','Only identical configuration and physical battery pairs','Lv1 versus Lv2; no synthetic samples'],
                             metric_definitions={'ratio_change':'100 * ((rate_stage / rate_medium)_Lv2 / (rate_stage / rate_medium)_Lv1 - 1)',
                                                 'positive':'The stage/Medium ratio is larger at Lv2; not necessarily a larger raw discharge rate'})),
             dict(id='extended',label='Complete stage-pair sensitivity audit',path=source_root+'/summary.json',
                  query=dict(engine='SQLite (in-memory); Python telemetry extraction and descriptive summaries',language='sql',sql=extended_sql,executed_at=generated,
                             description='expanded_stage_pairs is loaded from expanded_stage_pairs.csv. The source-flight scan accepts complete High+Medium or Medium+Low even if the third stage is absent. Matching and ratio changes are independently recomputed here; summary.json preserves exclusions and source hashes.',
                             tables_used=['expanded_stage_pairs',source_root+'/expanded_stage_pairs.csv',source_root+'/expanded_source_audit.csv',
                                          source_root+'/common_prelanding_low.csv',source_root+'/no_wind_comparisons.csv','database/experiment_registry.json'],
                             filters=['Five drones reached hover; known battery/drone baseline pairs','No prepare/merged/outlier files','Complete pair of stages; telemetry and within-window fault QA'],
                             metric_definitions={'change_pct':'100 * (stage_to_medium_ratio_Lv2 / stage_to_medium_ratio_Lv1 - 1)',
                                                 'replicate_cell':'Same wind direction, level, formation, spacing, physical drone, battery and stage-pair, in more than one source flight'}))]
    blocks=[]
    def md(id,body,source=None):
        block=dict(id=id,type='markdown',body=body)
        if source:block['sourceId']=source
        blocks.append(block)
    title='Wind strength and discharge-stage ratios'
    md('title','# '+title)
    md('answer','## 结论：有差异，但目前不能判定是风力效应还是实验波动\n\n'
       '**没有观察到“风越大，三段比例就统一变大或变小”的规律。** 控制风向、队形、间距、电池和无人机之后，High/Medium 的变化仍然有正有负，而且可达到几十个百分点的相对变化。'
       'Low/Medium 通常变化较小，但个别配置也有明显差异。\n\n'
       '**不能据此称为纯随机，也不能证明与风力无关。** 符合完整阶段口径的同条件重复记录不足，尚不能把风力变化与当天飞行状态、位置保持、风扇启动时点等因素分开。')
    md('definition','## 本次比较的是“比例随风力怎么变”\n\n'
       '沿用最新口径：起飞后第一次显示掉电才开始计时，去掉前面的初始平台；之后正常整数阶梯平台全部保留。High、Medium、Low 按每块电池自己的分界点计算。\n\n'
       '`风力变化效应(%) = [(阶段耗电率 / Medium耗电率)Lv2 ÷ (阶段耗电率 / Medium耗电率)Lv1 − 1] × 100`\n\n'
       '+20% 的含义是：该阶段相对 Medium 的耗电率比例在 Lv2 比 Lv1 高 20%，不是电量多消耗了 20 个百分点，也不是该阶段绝对耗电率一定高了 20%。'
       '同一块电池跨风力配对时，冻结的基线比例在这个比值中会抵消。因此，这个比较不依赖 Bideal 的绝对斜率是否准确。','primary')
    md('scope','## 先严格配对，避免配置混杂\n\n'
       '完整曲线样本中，Lv1 有 30 条单机曲线、11 次飞行，Lv2 有 13 条单机曲线、5 次飞行；不能直接把这两组的总体平均值当成风力效应。'
       '严格匹配后只有 **7 组单机配对、5 组配置条件**，涉及 10 次飞行，全部为 75 cm 间距。它们不能代表所有电池、队形或 50 cm 实验。'
       '这些记录覆盖 2026 年 8 月 29 日至 9 月 8 日；风力按实验记录的 Lv1/Lv2 标签区分，文件没有逐机实测风速字段。','primary')
    md('matched_finding','## High 的差异较大，方向却不统一\n\n'
       '7 组严格配对中，High/Medium 在 Lv2 有 3 组上升、4 组下降，典型变化幅度为 **27.2%**（绝对变化中位数），范围 **−37.9% 至 +38.8%**。'
       'Low/Medium 的典型变化幅度为 **3.7%**，范围 **−4.7% 至 +6.0%**。\n\n'
       '下图的零线表示 Lv1、Lv2 的比例相同，正负方向表示比例增加或降低。两种颜色是 High/Medium 与 Low/Medium；每行比较同一块电池和同一位置。'
       '这里是 5 组配置下的观察结果，不是 7 个独立随机实验。','primary')
    blocks.append(dict(id='paired_chart',type='chart',chartId='paired_changes'))
    md('example','## 同一块 B11，也没有统一的风力方向\n\n'
       '例如 Front + Tail 下，High/Medium 从 Lv1 到 Lv2 增加 **38.8%**；Diamond + Tail 下却减少 **37.9%**。'
       'Diamond + Side 下同一块 B11 减少 **27.2%**。这说明不能仅用一个统一的“风力倍率”解释当前结果。'
       '这几组 High 都在五机首次进入 hover 后才开始掉电，因此初始回正不是这些相反方向结果的充分解释；但仍可能存在其他飞行状态差异。','primary')
    blocks.append(dict(id='matched_table',type='table',tableId='pairs'))
    md('extended_scope','## 增加完整阶段对后，Low 也存在局部较大变化\n\n'
       '再检查全部 163 个协调记录文件，不要求一条记录同时覆盖三个阶段：只要完整覆盖 High+Medium，或 Medium+Low，就可分别比较。'
       '有风记录得到 46 条 High/Medium 阶段对（16 次飞行）和 82 条 Low/Medium 阶段对（19 次飞行）。匹配 Lv1/Lv2 后分别为 **10 组 High 配对、26 组 Low 配对**。\n\n'
       'High 的典型变化幅度 **28.4%**，范围 −37.9% 至 +97.6%；Low 的典型变化幅度 **4.4%**，18/26 在 ±10% 内，但范围扩大到 **−22.7% 至 +25.5%**。'
       '所以，前面严格完整曲线中 Low 的 ±6% 范围不能推广到所有可用阶段记录。','extended')
    md('condition_effect','## 有按配置分组的同向变化，但尚未经过重复验证\n\n'
       '扩展的 Low 对比中，Diamond + Side 的 4 架可比无人机全部增加，Diamond + Tail 的 3 架也全部增加；Front + Tail 的 4 架则全部降低。'
       '这提示值得检查“队形与风力共同作用”，但每个条件、每块电池目前仍只有一次合格阶段记录。'
       '同一次飞行的几架无人机不是独立重复，这种分组也可能来自那次飞行共同的环境或操作变化。下表是每种配置内的描述性中位数，不是已验证的因果效应。','extended')
    blocks.append(dict(id='low_conditions',type='table',tableId='low_condition_table'))
    md('robustness','## 排除一部分干扰后，仍不能得出因果结论\n\n'
       '**起飞回正：**严格 High 配对中，只有 4 组两次飞行的首次掉电均晚于五机首次 hover。保留它们后仍同时存在 −37.9% 和 +38.8% 的变化，方向并未统一。\n\n'
       '**低电量先后降落：**严格配对中没有一组能在两次飞行里都保持五架飞到该机 Low 结束。改为同电池、相同 SOC 区间，且在首次队友降落前结束，只有 3 组至少保留 10 个百分点的下降量；变化为 +0.6%、−6.5%、−6.9%。'
       '这个更可比的检查样本很少，只说明这些窗口差异不大，不能外推到整个 Low。\n\n'
       '**无风对照：**只有 1 次合格无风完整飞行，且与有风实验日期不同，不足以判断从无风到有风的稳定效应。','extended')
    md('randomness','## 为什么现在不能判断“纯随机”\n\n'
       '估计同条件波动，需要在风力、风向、队形、间距、电池、无人机和 SOC 比较窗口都相同的条件下，有多次独立飞行。'
       '本次完整曲线以及扩展阶段对筛选后，**这样的重复单元均为 0**。这不表示原始文件中没有多次起飞，而是它们没有重复覆盖此次要求的完整阶段。\n\n'
       '因此当前无法估计“同样条件重做一次，比例通常波动多少”，就无法判断 Lv1/Lv2 的差异是否明显超过该波动。'
       '本次只作描述性比较，没有用缺乏独立重复的数据报告显著性，也没有把“未发现统一趋势”当成“证明无关”。','extended')
    md('recommendation','## 对当前研究的建议\n\n'
       '- 暂时保留 wind level，不要因缺少统一趋势就删掉这个条件。\n'
       '- 不使用一个随风力统一放大或缩小的三段比例；已有差异应保留为配置相关的实测结果和不确定性。\n'
       '- 若时间有限，优先补 Front 与 Diamond 的 Tail、75 cm 对比：这两种配置的 High 变化方向相反，最有助于检验是否为稳定的配置差异。每个风力尽量至少 3 次独立有效飞行，并交替安排 Lv1/Lv2。\n'
       '- 记录风扇何时开启并稳定、换电池及飞行顺序；比较时尽量采用五机仍在空中的共同 SOC 窗口。三次只是最低限度的重复检查，不保证能检出较小效应。')
    md('questions','## 需要进一步确认的实验条件\n\n'
       '风扇是在起飞前已开启，还是起飞后才开启？Lv1、Lv2 的风扇位置和距离是否完全相同？这些因素若不同，就不能把两条曲线的差别单独归因于风力等级。')
    def table(id,dataset,title,columns,sort,source):
        return dict(id=id,title=title,dataset=dataset,sourceId=source,defaultSort=dict(field=sort,direction='asc'),
                    columns=[dict(field=f,label=l,**({'type':'text'} if text else {'format':'number'})) for f,l,text in columns])
    tables=[table('pairs','pairs','Lv2 相对 Lv1 的阶段比例变化',
                  [('comparison','配置 / 电池（均为75 cm）',True),('high_change_pct','High/Medium 变化 (%)',False),
                   ('low_change_pct','Low/Medium 变化 (%)',False)],'comparison','primary'),
            table('low_condition_table','low_conditions','Low/Medium 按配置的匹配结果',
                  [('condition','条件',True),('pairs','单机配对数',False),('median_change_pct','变化中位数 (%)',False),
                   ('min_change_pct','最小变化 (%)',False),('max_change_pct','最大变化 (%)',False)],'condition','extended')]
    chart_spec=dict(id='paired_changes',title='Lv2 相对 Lv1 的阶段比例变化',
                    subtitle='同风向、队形、间距、电池与无人机；7 组单机配对，5 组配置；单位为相对变化 (%)',showDescription=True,
                    type='horizontalBar',intent='comparison',dataset='paired_chart',sourceId='primary',layout='full',
                    valueFormat='number',unit='%',settings=dict(groupMode='grouped',orientation='horizontal',sort='none',showValues=True),
                    encodings=dict(x=dict(field='comparison',type='nominal',label='配置与电池'),
                                   y=dict(field='change_pct',type='quantitative',label='比例相对变化 (%)'),
                                   color=dict(field='stage',type='nominal',label='阶段比例'),
                                   tooltip=[dict(field='ratio_lv1',type='quantitative',label='Lv1 基线归一化比例'),
                                            dict(field='ratio_lv2',type='quantitative',label='Lv2 基线归一化比例')]),
                    referenceLines=[dict(axis='y',value=0,color='neutral',label='无变化',lineStyle='solid')],
                    palette=dict(kind='categorical'),legend=dict(position='bottom',sort='spec'),labels=dict(values='all'))
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,generatedAt=generated,blocks=blocks,
                    charts=[chart_spec],tables=tables,sources=sources),
                  snapshot=dict(version=1,status='ready',generatedAt=generated,datasets={
                      'paired_chart':chart,'pairs':records(matched),'low_conditions':records(condition[condition.stage.eq('low')]),
                      'extended_pairs':records(extended),'expanded_conditions':records(condition),
                      'common_low':records(pd.read_csv(OUT/'common_prelanding_low.csv'))}),sources=sources)
    (OUT/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (OUT/'extended_comparison.sql').write_text(extended_sql+';\n')
    # Build a bounded, inspectable companion notebook. Execute all cells in-order
    # with Python and disclose that no Jupyter kernel is installed in this runtime.
    cells=[]
    def markdown(id,text):cells.append(dict(cell_type='markdown',id=id,metadata={},source=text.splitlines(keepends=True)))
    markdown('summary','## tl;dr\n\nMatched results do not show a uniform monotonic wind-strength effect. High/Medium changes are large and mixed in sign; Low/Medium is usually closer but has larger local exceptions. No eligible within-condition replicate cells remain, so random variation and wind-related effects cannot be separated.\n')
    markdown('methods','## Context & Methods\n\nSame wind direction, formation, spacing, drone, and battery; compare Lv1 versus Lv2. Exclude the initial SOC plateau only.\n\n### Key Assumptions\n\nRecorded levels are treatment labels, not measured local m/s. Within-flight drone observations are correlated. The primary cohort requires complete curves; the sensitivity cohort needs only complete pairs of stages.\n')
    ns={}; count=0
    def code(id,source):
        nonlocal count
        count+=1; capture=io.StringIO()
        with contextlib.redirect_stdout(capture):exec(compile(source,'wind_strength_audit.ipynb','exec'),ns)
        cells.append(dict(cell_type='code',id=id,metadata={},source=source.splitlines(keepends=True),execution_count=count,
                          outputs=[dict(output_type='stream',name='stdout',text=capture.getvalue().splitlines(keepends=True))]))
    markdown('data','## Data\n\nData and code are local to the drone_experiment project. Sources are read-only; no model or flight-control code is changed.\n')
    code('load',"from pathlib import Path\nimport sys,json,sqlite3\nimport pandas as pd\nimport numpy as np\n"
         "ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'battery_normalization.py').exists())\n"
         "sys.path.insert(0,str(ROOT/'output_py'))\nfrom diagnose_wind_strength_stage_ratios import OUT,PRIOR,SQL,describe\n"
         "from audit_wind_tunnel_stage_ratios import sha\nfull=pd.read_csv(PRIOR/'paired_ratios.csv')\n"
         "with sqlite3.connect(':memory:') as conn:\n    full.to_sql('full_curves',conn,index=False)\n    matched=pd.read_sql_query(SQL,conn)\n"
         "print(matched[['wind_direction','formation','battery_id','high_change_pct','low_change_pct']].round(3).to_string(index=False))\n")
    markdown('validate','### Check provenance and pairing\n')
    code('checks',"audit=json.loads((OUT/'summary.json').read_text())\n"
         "for source,expected in audit['input_hashes'].items():\n    assert sha(ROOT/source)==expected\n"
         "saved=pd.read_csv(OUT/'matched_full_curves.csv')\n"
         "for stage in ['high','low']:\n    assert np.allclose(matched[stage+'_change_pct'],saved[stage+'_change_pct'])\n"
         "assert not full.duplicated(['wind_direction','formation','spacing_cm','battery_id','drone_id','wind_level']).any()\n"
         "print('Source hashes, independent SQL matches and primary one-to-one joins: passed')\n")
    markdown('results','## Results\n')
    code('stats',"for stage in ['high','low']:\n    print('Primary',stage,describe(matched[stage+'_change_pct']))\n"
         "extended=pd.read_csv(OUT/'expanded_matched_pairs.csv')\n"
         "for stage,g in extended.groupby('stage'):\n    print('Extended',stage,describe(g.change_pct))\n"
         "repeats=pd.read_csv(OUT/'repeated_cells.csv')\nprint('Eligible replicate cells:',len(repeats))\n"
         "common=pd.read_csv(OUT/'common_prelanding_low.csv')\nprint(common[common.included][['wind_direction','formation','battery_id','end_soc','change_pct']].to_string(index=False))\n")
    markdown('takeaways','## Takeaways\n\nNo universal wind-level multiplier is supported. Configuration-specific differences are descriptive hypotheses, not demonstrated causation. The missing within-condition repeats prevent an empirical estimate of repeatability noise. Expanded Low pairs include landing-related environment changes; the matched pre-landing check is limited to three pairs.\n\nExecution: all code cells above ran sequentially in Python. Jupyter-kernel execution was not available (nbformat/nbclient/ipykernel not installed). To validate in a Jupyter environment, run `python -m jupyter nbconvert --execute --to notebook --inplace wind_strength_audit.ipynb` from this directory.\n')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                   'language_info':{'name':'python','version':sys.version.split()[0]},'execution_note':'Sequential Python execution; not run through a Jupyter kernel'},cells=cells)
    assert all(c['cell_type']!='code' or c['execution_count'] for c in cells)
    (OUT/'wind_strength_audit.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')
    print('Canonical report and executed companion notebook created; source/hash/SQL checks passed.')


if __name__=='__main__':main()
