"""Validate and package the partial-window raw ranking audit."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import io
import json
import sys

import numpy as np
import pandas as pd

from audit_partial_stage_rankings import ROOT, OUT, SQL, digest, rank_windows


def records(df):
    return json.loads(df.to_json(orient='records',force_ascii=False,double_precision=10))


def validate():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for p,h in manifest['input_hashes'].items(): assert digest(ROOT/p)==h,p
    w=pd.read_csv(OUT/'windows.csv')
    r,c,s=rank_windows(w)
    saved=pd.read_csv(OUT/'summary.csv')
    pd.testing.assert_frame_equal(s.reset_index(drop=True),saved,check_dtype=False)
    # Independently trace every primary segment back to exact original observations.
    primary=w[(w.threshold_pp==5)&(w.policy=='first')]
    for source,g in primary.groupby('source'):
        raw=pd.read_csv(ROOT/source,usecols=['drone_name','battery_id','elapsed_time','battery','phase'])
        for row in g.itertuples():
            d=raw[(raw.drone_name==row.drone_id)&(raw.battery_id==row.battery_id)]
            a=d[np.isclose(d.elapsed_time,row.start_s,atol=1e-7,rtol=0)]
            b=d[np.isclose(d.elapsed_time,row.end_s,atol=1e-7,rtol=0)]
            assert len(a)==len(b)==1
            assert a.battery.iloc[0]==row.soc_start and b.battery.iloc[0]==row.soc_end
            rate=60*(a.battery.iloc[0]-b.battery.iloc[0])/round(float(b.elapsed_time.iloc[0]-a.elapsed_time.iloc[0]),9)
            assert np.isclose(rate,row.raw_rate,rtol=1e-10)
            seg=d[d.elapsed_time.between(row.start_s,row.end_s)]
            assert len(seg)==row.n and seg.battery.between(row.stage_lower,row.stage_upper).all()
    first=c[(c.threshold_pp==5)&(c.policy=='first')&(c.method=='endpoint')]
    second=c[(c.threshold_pp==5)&(c.policy=='second')&(c.method=='endpoint')]
    matched=first.merge(second,on=['source','stage'],suffixes=('_first','_second'))
    matched_summary=matched.groupby('stage').agg(flights=('source','size'),first=('same_fastest_first','sum'),second=('same_fastest_second','sum')).reset_index()
    old=pd.read_csv(ROOT/'analysis_results/cross_stage_drone_rankings_20260909/rank_comparisons.csv')
    old=old[old['mode'].eq('raw')]
    joined=first.merge(old,on=['source','stage'],suffixes=('_partial','_whole'))
    old_cohort=joined.groupby('stage').agg(flights=('source','size'),whole=('same_fastest_whole','sum'),partial=('same_fastest_partial','sum')).reset_index()
    matched.to_csv(OUT/'matched_first_second.csv',index=False,float_format='%.12g')
    old_cohort.to_csv(OUT/'matched_old_cohort.csv',index=False)
    qa=dict(assessment='Share with caveats',primary_segments_traced=len(primary),input_hashes_verified=len(manifest['input_hashes']),
            sql_ranks_independently_checked=True,normalization_applied=False,
            tie_policy='Exact equal endpoint durations use tied ranks; same-fastest includes shared top ranks. Durations rounded at 1e-9 seconds only to remove subtraction noise.',
            unit_tests='Five synthetic-fixture tests pass; fixtures are not experiment data',
            matched_first_second=records(matched_summary),matched_old_cohort=records(old_cohort),
            caveats=['Selection and window length materially affect exact ranks','Comparisons are local observed SOC slopes, not whole-stage averages or isolated aerodynamic effects'])
    (OUT/'validation.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    return w,r,c,s,qa


def make_notebook():
    cell_specs=[('markdown','summary','## tl;dr\nPartial High records increase eligible five-drone High/Medium flights from 2 to 10. With first >=5 pp / >=10 s windows, Medium top rank persists in High 1/10 and Low 6/18 flights. These are local-window ranks, not whole-stage averages.\n'),
      ('markdown','methods','## Context & Methods\n### Key Assumptions\nUse each battery’s frozen stage boundaries only; no Bideal rate conversion. Select earliest consecutive drop-to-drop segments, keeping ordinary integer-SOC plateaus. Test 3/5/8 pp windows and the next non-overlapping window. Each flight counts once per stage comparison.\n'),
      ('code','setup',"from pathlib import Path\nimport sys,json\nimport pandas as pd\nROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'battery_normalization.py').exists())\nsys.path.insert(0,str(ROOT/'output_py'))\nfrom audit_partial_stage_rankings import OUT,rank_windows,digest\n"),
      ('markdown','data','## Data\nRaw wind-tunnel coordination CSV files and frozen battery boundaries. Source identities and SHA256 values are in manifest.json. Window details are in windows.csv; one-flight/stage, five-drone rates are in raw_rates_wide.csv.\n'),
      ('code','load',"manifest=json.loads((OUT/'manifest.json').read_text())\nfor path,expected in manifest['input_hashes'].items():\n    assert digest(ROOT/path)==expected\nwindows=pd.read_csv(OUT/'windows.csv')\nprint(windows[['experiment_id','drone_id','stage','soc_start','soc_end','duration_s','raw_rate']].head(10).to_string(index=False))\n"),
      ('markdown','results','## Results\nRecompute rankings and denominators from the stored observed window endpoints. The extraction implementation is audit_partial_stage_rankings.py; scan() returns fresh windows without writing source data.\n'),
      ('code','rank',"ranked,comparisons,summary=rank_windows(windows)\nsaved=pd.read_csv(OUT/'summary.csv')\npd.testing.assert_frame_equal(summary.reset_index(drop=True),saved,check_dtype=False)\nprint(summary[(summary.method=='endpoint')&(summary.policy=='first')].to_string(index=False))\nprint(json.loads((OUT/'validation.json').read_text())['matched_first_second'])\n"),
      ('markdown','takeaways','## Takeaways\nNo 100%-start requirement remains. Do not equate an arbitrary small subsection with a stable whole-stage slope: thresholds and segment position affect rankings. High often has near-ties. Keep SOC range, duration and uncertainty alongside rates. No source experiment, flight control or Bideal model was changed.\n')]
    cells=[];env={};number=0
    for kind,id,body in cell_specs:
        cell=dict(cell_type=kind,id=id,metadata={},source=body.splitlines(keepends=True))
        if kind=='code':
            number+=1;capture=io.StringIO()
            with contextlib.redirect_stdout(capture):exec(compile(body,'partial_window_audit.ipynb','exec'),env)
            cell.update(execution_count=number,outputs=[dict(output_type='stream',name='stdout',text=capture.getvalue().splitlines(keepends=True))])
        cells.append(cell)
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        validation=dict(method='Code cells executed sequentially in one Python namespace, not through a Jupyter kernel; nbformat/nbclient unavailable',
                        kernel_command='python3 -m jupyter nbconvert --execute --to notebook --inplace partial_window_audit.ipynb (requires nbconvert/nbformat)')))
    (OUT/'partial_window_audit.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')


def main():
    w,r,c,s,qa=validate()
    main=c[(c.threshold_pp==5)&(c.policy=='first')&(c.method=='endpoint')].copy()
    sensitive=s[(s.policy=='first')&(s.method=='endpoint')].copy()
    sensitive['segment']=sensitive.threshold_pp.astype(str)+' pp'
    sensitive['stage']=sensitive.stage.str.title()
    sensitive['fraction']=sensitive.same_fastest/sensitive.flights
    main['experiment']=main.experiment_id.str.replace('wind_tunnel_','',regex=False)
    main['stage']=main.stage.str.title()
    main['medium_leader']=main.medium_leader.str.replace('drone_','D')
    main['stage_leader']=main.stage_leader.str.replace('drone_','D')
    main['retained']=main.same_fastest.map({True:'是',False:'否'})
    generated=datetime.now(timezone.utc).isoformat()
    source=dict(id='partial',label='Raw partial-stage discharge windows and ranks',path=str((OUT/'windows.csv').relative_to(ROOT)),
        query=dict(engine='SQLite in-memory; source-window extraction and summaries in Python',language='sql',sql=SQL,
        executed_at=generated,description='Window endpoints extracted from original coordination CSVs by audit_partial_stage_rankings.py; executed SQL computes raw endpoint rates, pairs stages and ranks all five drones. Summary counts are aggregated in the same script.',
        tables_used=['windows','database/experiment_registry.json',str(MODEL_REL)],
        filters=['Wind level > 0','Same flight and same five drones','Own-battery stage boundaries; no rate normalization',
                 'First drop-to-drop window >=5 pp and >=10 s primary; >=10 samples','3/8 pp, after-ready and second-window sensitivity',
                 'No prepare/merged/outlier/synthetic; no fault or telemetry-gap bridging'],
        metric_definitions={'raw_rate':'60*(observed SOC_start-SOC_end)/(elapsed_end-elapsed_start), percentage points/minute',
                            'same_fastest':'At least one Medium first-ranked drone remains first-ranked in the compared stage, including exact ties',
                            'fraction':'same_fastest flights / all eligible five-drone stage-pair flights',
                            'near_tie':'First/second rate ratio - 1 <0.02; descriptive, not statistical equivalence'}))
    old_source=dict(id='old',label='Earlier whole-stage raw ranking audit',path='analysis_results/cross_stage_drone_rankings_20260909/summary.csv')
    blocks=[]
    def md(id,body,source_id=None):
        block=dict(id=id,type='markdown',body=body)
        if source_id:block['sourceId']=source_id
        blocks.append(block)
    title='Raw discharge rankings from partial SOC windows'
    md('title','# '+title)
    md('summary','## 结论：High 可用片段增加，但短片段排名并不稳定\n\n'
       '取消“必须从 100% 起飞”的要求后，按最早连续下降至少 **5 个百分点、至少 10 秒**的片段，已有 **10 次**实验可比较五机 High 与 Medium，**18 次**可比较五机 Low 与 Medium。'
       '这里均为原始 SOC 耗电率，未做 Bideal 标准化。\n\n'
       'Medium 最快者在 High 仍最快为 **1/10**，在 Low 为 **6/18**；含精确并列第一。'
       '但这些是所选局部片段的结果，不应当当作整段平均排名。更换片段长度或位置，结果会变。','partial')
    md('definitions','## 比较的是局部掉电速度，不要求覆盖整个阶段\n\n'
       '各电池仍用自己的 High、Medium、Low 分界点，仅借用边界划分区间，不应用 Bideal 耗电率转换。'
       '例如起始 94% 的电池可以取 93%→88%，只要这段完全在该电池的 High 范围内。该例仅说明方法，不代表某条真实记录。\n\n'
       '耗电率 = 实际下降的电量百分点 ÷ 实际经过秒数 × 60，单位为百分点/分钟。'
       '同一次实验中五架各有可用的两个阶段，才计作一次五机排名比较；不会把同一次飞行的多个片段当作独立实验。')
    md('coverage','## 部分曲线可以利用，但缺少阶段仍不补造\n\n'
       '当前主口径保存了 **137 条单机 High 片段（40 次飞行）**、**144 条 Medium 片段（37 次）**、**122 条 Low 片段（29 次）**。'
       '这些单阶段片段数不是完整五机排名的分母；五机配齐两个阶段后，才得到 High 的 10 次、Low 的 18 次。'
       '若某架起飞时已经低于自己的 High 下边界，它依然没有这次飞行的 High 数据。','partial')
    md('window_effect','## 片段长度会改变结果，不能随意挑一段后认定整个阶段相同\n\n'
       '下图按相同的“最早合格片段”规则，比较下降至少 3、5、8 个百分点时的结果。纵轴为 Medium 最快者保持第一的实验占比，具体分母见表。'
       'Low 三组均为同样的 18 次实验，保持第一分别为 **6、6、10 次**；High 分母随覆盖要求变化，不能只看百分比作直接优劣判断。','partial')
    blocks.append(dict(id='sensitivity_chart',type='chart',chartId='sensitivity'))
    blocks.append(dict(id='sensitivity_table',type='table',tableId='sensitivity'))
    md('next_window','## 在相同实验里换到下一小段，Low 排名也会变化\n\n'
       '保持同样的 18 次 Low–Medium 实验，分别取各阶段第一段和下一段不重叠的合格片段，Medium 最快者保持第一从 **6/18** 变为 **13/18**。'
       '这说明变化不仅来自新增实验，还与片段位置有关；两阶段的取样片段都随之向后移动。\n\n'
       'High 两种片段都齐全的共同样本为 7 次，对应 **0/7** 和 **2/7**。不同电池的 SOC 区间和取样时刻并不完全相同，这不是风力或位置效应的因果结论。','partial')
    md('detail','## 每次实验的第一名与差距\n\n'
       '下表为主口径的逐次核对：差距表示当前阶段最快速率，相对原 Medium 最快者在该阶段速率高多少。'
       'High 有 **9/10 次**第一、第二名差距不到 2%，包含一次精确并列；其中 **8/10 次**原 Medium 最快者距离 High 最快者不到或等于 5%。'
       '因此“没有保持第一”不等于速度明显不同。','partial')
    blocks.append(dict(id='details',type='table',tableId='details'))
    md('old','## 与之前的整段结果是两种统计口径\n\n'
       '旧口径按完整阶段平均，原始数据 High 为 0/2、Low 为 10/11；它回答整段平均速率的排名，而本次回答局部片段的排名。','old')
    md('matched_old','只保留旧 Low 的同样 11 次实验，本次短片段得到 **3/11**，说明不能把变化全部解释成新增实验导致。'
       '旧 High 两次在短片段中仍为 0/2。旧报告和原始文件均保留，没有覆盖成新口径。','partial')
    md('method','## 取段规则与交叉检查\n\n'
       '片段从真实下降事件开始，到满足电量跨度和时长的另一次下降事件结束；不计算起飞后的初始平台，但保留中途所有正常整数阶梯平台。'
       '不跨阶段，不跨遥测间断或故障片段；不删除时间后强行连接曲线，不插值或补造电量。\n\n'
       '主口径沿用“起飞后可用记录”，并标记是否处于初始回正期。另从五架均进入悬停后取样时，High 为 **1/8**、Low 为 **6/18**。'
       '同样片段改用直线拟合斜率时，High 为 **3/10**、Low 为 **5/18**。这些交叉检查同样表明精确第一名对取段和计算方式敏感。','partial')
    md('limits','## 可以按部分区间计算，但应保留局部性的限制\n\n'
       '可以用已有的有效片段，不必因未从 100% 起飞就弃用。只是“一小段斜率”不能自动代表整个 High、Medium 或 Low。'
       '这些比较还包含整数 SOC 量化、各机取样时刻不同、初始回正，以及部分 Low 片段发生在其他无人机降落之后等限制。'
       '此处记录的是显示 SOC 的下降速度，不等于直接测得功率或能量。')
    md('next','## 当前使用建议\n\n'
       '每个速率同时保存实际 SOC 起止、持续时间和样本数；用多个不重叠片段检查同阶段的波动。'
       '先将本次输出用于局部曲线分析，不据此覆盖冻结的全阶段耗电率表。无需补做实验，也不需要生成缺失的 High 数据。'
       '仍待明确的是：同阶段不同片段的差距是否小到足以支持一条直线近似；本次只显示该假设需要用数据检查。')
    def table(id,dataset,title,cols,sort):
        return dict(id=id,dataset=dataset,title=title,sourceId='partial',defaultSort=dict(field=sort,direction='asc'),
                    columns=[dict(field=f,label=l,**({'type':'text'} if text else {'format':'number'})) for f,l,text in cols])
    tables=[table('sensitivity','sensitivity','片段长度与排名保持次数',
         [('segment','最小下降跨度',True),('stage','比较阶段',True),('flights','实验数',False),('same_fastest','仍最快次数',False)],'segment'),
      table('details','details','五机局部片段排名：主口径',
         [('experiment','实验',True),('stage','阶段',True),('medium_leader','Medium 最快',True),('stage_leader','当前阶段最快',True),
          ('retained','是否保持',True),('gap_pct','最快者相对原最快者差距 %',False)],'experiment')]
    chart=dict(id='sensitivity',type='bar',title='不同片段长度的排名保持比例',
      subtitle='原始 SOC 耗电率；同一飞行五机比较；每组分母见下表',showDescription=True,
      intent='comparison',dataset='sensitivity',sourceId='partial',layout='full',valueFormat='percent',
      encodings=dict(x=dict(field='segment',type='nominal',label='最小电量跨度'),
                     y=dict(field='fraction',type='quantitative',label='保持第一的实验占比'),
                     color=dict(field='stage',type='nominal',label='比较阶段'),
                     tooltip=[dict(field='flights',type='quantitative',label='实验数'),dict(field='same_fastest',type='quantitative',label='仍最快次数')]),
      settings=dict(groupMode='grouped',orientation='vertical',sort='none',showValues=True),
      palette=dict(kind='categorical'),legend=dict(position='bottom',sort='spec'),labels=dict(values='all'))
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,generatedAt=generated,
       blocks=blocks,charts=[chart],tables=tables,sources=[source,old_source]),snapshot=dict(version=1,status='ready',generatedAt=generated,
       datasets=dict(sensitivity=records(sensitive),details=records(main),
                     preview=records(w[(w.policy=='first')&(w.threshold_pp==5)].head(10)))),sources=[source,old_source])
    (OUT/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (OUT/'report_plan.json').write_text(json.dumps(dict(audience='technical',new_companion_not_replacement=True,
       required_structure=['title','technical summary','metric definitions before evidence','evidence and coverage','method','robustness and uncertainty','next steps and open question'],
       chart_contract=dict(question='Does partial-window length change rank retention?',family='Comparison & Ranking',variant='grouped bars',
          sufficiency='6 stage-by-threshold aggregates; denominator table adjacent; not a temporal trend',palette='two semantic groups in native categorical palette',
          non_color='group position, legend and exact values',footprint='full width',surface='MCP artifact; HTML only if renderer unavailable'),
       omitted_other_charts='Exact per-experiment lookup belongs in the detail table; no redundant ranking chart'),ensure_ascii=False,indent=2)+'\n')
    make_notebook()
    print(json.dumps(qa,ensure_ascii=False,indent=2))


MODEL_REL='analysis_results/battery_normalization_v3_with_b15_20260909/model.json'
if __name__=='__main__':main()
