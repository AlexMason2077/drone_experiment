"""Within-flight five-drone rank consistency, raw and frozen-Bideal normalized."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import hashlib
import io
import json
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from battery_normalization import BatteryNormalizer

INPUT=ROOT/'analysis_results/wind_strength_stage_ratio_diagnosis_20260909/expanded_stage_pairs.csv'
MODEL=ROOT/'analysis_results/battery_normalization_v3_with_b15_20260909/model.json'
OUT=ROOT/'analysis_results/cross_stage_drone_rankings_20260909'
SQL='''WITH values_by_mode AS (
 SELECT source,experiment_id,drone_id,battery_id,compare_stage,'raw' AS mode,
        raw_medium AS medium_value,raw_stage AS stage_value FROM five_drone_pairs
 UNION ALL
 SELECT source,experiment_id,drone_id,battery_id,compare_stage,'Bideal' AS mode,
        ideal_medium AS medium_value,ideal_stage AS stage_value FROM five_drone_pairs
)
SELECT *,
 RANK() OVER (PARTITION BY source,compare_stage,mode ORDER BY medium_value DESC) AS medium_rank,
 RANK() OVER (PARTITION BY source,compare_stage,mode ORDER BY stage_value DESC) AS stage_rank
FROM values_by_mode ORDER BY compare_stage,source,mode,drone_id'''


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def calculate():
    norm=BatteryNormalizer.load(MODEL)
    data=pd.read_csv(INPUT)
    assert not data.duplicated(['source','drone_id','stage']).any()
    data=data[data.wind_level.gt(0)].copy()
    kept=data.groupby(['source','stage']).filter(lambda g:len(g)==5 and g.drone_id.nunique()==5).copy()
    assert set(kept.drone_id)=={f'drone_{i}' for i in range(1,6)}
    kept['compare_stage']=kept.stage
    kept['raw_stage']=[r.stage_k*norm.curve_for(r.battery_id,r.drone_id).rates_pp_min[0 if r.stage=='high' else 2] for _,r in kept.iterrows()]
    kept['raw_medium']=[r.medium_k*norm.curve_for(r.battery_id,r.drone_id).rates_pp_min[1] for _,r in kept.iterrows()]
    kept['ideal_stage']=[r.stage_k*norm.reference.rates_pp_min[0 if r.stage=='high' else 2] for _,r in kept.iterrows()]
    kept['ideal_medium']=kept.medium_k*norm.reference.rates_pp_min[1]
    with sqlite3.connect(':memory:') as conn:
        kept.to_sql('five_drone_pairs',conn,index=False)
        ranks=pd.read_sql_query(SQL,conn)
    results=[]
    for (stage,source,mode),g in ranks.groupby(['compare_stage','source','mode']):
        assert len(g)==5
        assert np.allclose(g.medium_rank,g.medium_value.rank(ascending=False))
        assert np.allclose(g.stage_rank,g.stage_value.rank(ascending=False))
        medium=g.sort_values('medium_rank');current=g.sort_values('stage_rank')
        leader=medium.drone_id.iloc[0]
        leader_now=current[current.drone_id.eq(leader)].iloc[0]
        results.append(dict(experiment_id=g.experiment_id.iloc[0],source=source,stage=stage,mode=mode,
                            medium_leader=leader,stage_leader=current.drone_id.iloc[0],same_fastest=leader==current.drone_id.iloc[0],
                            medium_leader_new_rank=int(leader_now.stage_rank),
                            medium_order=' > '.join(medium.drone_id.str.replace('drone_','D')),
                            stage_order=' > '.join(current.drone_id.str.replace('drone_','D')),
                            same_complete_order=medium.drone_id.tolist()==current.drone_id.tolist(),
                            rank_correlation=float(g.medium_rank.corr(g.stage_rank)),
                            fastest_over_medium_leader_pct=100*(current.stage_value.iloc[0]/leader_now.stage_value-1),
                            fastest_over_second_pct=100*(current.stage_value.iloc[0]/current.stage_value.iloc[1]-1)))
    results=pd.DataFrame(results)
    summaries=[]
    for (stage,mode),g in results.groupby(['stage','mode']):
        summaries.append(dict(stage=stage,mode=mode,flights=len(g),same_fastest=int(g.same_fastest.sum()),
                              same_order=int(g.same_complete_order.sum()),
                              leader_still_top2=int(g.medium_leader_new_rank.le(2).sum()),
                              leader_within5pct_of_fastest=int(g.fastest_over_medium_leader_pct.le(5).sum()),
                              leader_within10pct_of_fastest=int(g.fastest_over_medium_leader_pct.le(10).sum()),
                              top_two_within2pct=int(g.fastest_over_second_pct.lt(2).sum()),
                              median_rank_correlation=float(g.rank_correlation.median()),
                              median_fastest_over_medium_leader_pct=float(g.fastest_over_medium_leader_pct.median())))
    return kept,ranks,results,pd.DataFrame(summaries)


def records(df):return json.loads(df.to_json(orient='records',force_ascii=False,double_precision=10))


def main():
    assert not OUT.exists()
    previous=json.loads((INPUT.parent/'summary.json').read_text())
    for source,expected in previous['input_hashes'].items():assert digest(ROOT/source)==expected
    kept,ranks,results,summaries=calculate()
    OUT.mkdir(parents=True)
    for name,df in [('rates',kept),('ranked_values',ranks),('rank_comparisons',results),('summary',summaries)]:
        df.to_csv(OUT/(name+'.csv'),index=False,float_format='%.12g')
    (OUT/'ranking.sql').write_text(SQL+';\n')
    manifest=dict(source_sha256={str(INPUT.relative_to(ROOT)):digest(INPUT),str(MODEL.relative_to(ROOT)):digest(MODEL)},
                  definition='Within the same flight, rank all five observed drone rates descending; no incomplete fleet rankings.',
                  stage_policy='High begins at first post-takeoff SOC decrease; preserve later integer-step plateaus. Own-battery stage knots.',
                  normalization='own observed rate / own frozen baseline rate * Bideal reference stage rate',
                  user_confirmed='Fans on before takeoff; same position and distance for Lv1/Lv2.',
                  limitations=['Only 2 full five-drone High/Medium flights and 11 Medium/Low flights',
                               'Some High samples before full initial hover; Low can extend after other drones land',
                               'Ranks computed without rounding; small gaps are not demonstrated statistically distinct',
                               'Baseline normalization is a model correction, not a guarantee of isolated aerodynamic position effects'],
                  summary=records(summaries),raw_sources_unchanged=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    # Complete three-stage example, keep exact values and ranks together.
    exp='wind_tunnel_diamond_75_tail_lv2_001'
    example=[]; plot=[]
    for drone,g in ranks[ranks.experiment_id.eq(exp)&ranks['mode'].eq('Bideal')].groupby('drone_id'):
        h=g[g.compare_stage.eq('high')].iloc[0];l=g[g.compare_stage.eq('low')].iloc[0]
        assert np.isclose(h.medium_value,l.medium_value)
        example.append(dict(drone=drone.replace('drone_','D'),battery=h.battery_id,
                            high_rate=h.stage_value,medium_rate=h.medium_value,low_rate=l.stage_value,
                            high_rank=int(h.stage_rank),medium_rank=int(h.medium_rank),low_rank=int(l.stage_rank)))
        for stage,rate,rank in [('High',h.stage_value,h.stage_rank),('Medium',h.medium_value,h.medium_rank),('Low',l.stage_value,l.stage_rank)]:
            plot.append(dict(drone=drone.replace('drone_','D'),battery=h.battery_id,stage=stage,rate=rate,rank=int(rank),experiment_id=exp))
    generated=datetime.now(timezone.utc).isoformat()
    source=dict(id='rank_audit',label='Same-flight five-drone ranking audit',path='analysis_results/cross_stage_drone_rankings_20260909/ranking.sql',
                query=dict(engine='SQLite (in-memory); Python baseline mapping and descriptive summaries',language='sql',sql=SQL,
                           executed_at=generated,description='five_drone_pairs contains complete five-drone stage-pair observations from expanded_stage_pairs.csv. Own battery baselines map raw rates to Bideal. This executed SQL ranks within flight, comparison stage and measurement mode.',
                           tables_used=['five_drone_pairs',str(INPUT.relative_to(ROOT)),str(MODEL.relative_to(ROOT))],
                           filters=['Wind level > 0','All five matched calibrated drones present with both stages','No prepare/merged/outlier or faulty comparison windows'],
                           metric_definitions={'rank':'1 is fastest SOC percentage decrease per minute; descending rank, no rounding before ranking',
                                               'same_fastest':'Medium top-ranked physical drone is also top-ranked in the compared stage',
                                               'near_tie_gap':'100*(fastest rate / second-fastest rate - 1)'}))
    blocks=[]
    def md(id,body,backed=True):
        b=dict(id=id,type='markdown',body=body)
        if backed:b['sourceId']='rank_audit'
        blocks.append(b)
    title='Drone discharge rankings across SOC stages'
    md('title','# '+title,False)
    md('answer','## 结论：Medium 最快，不保证 High 或 Low 也最快\n\n'
       '按当前 **Bideal 标准化后的阶段平均耗电率**，Medium 第一名在 High 仍为第一的是 **1/2 次**，在 Low 仍为第一的是 **2/11 次**。'
       '因此不能直接把 Medium 的耗电快慢顺序复制到其他阶段。\n\n'
       '但 Low 排名经常很接近：11 次里有 9 次，第一与第二名的相对差距不到 2%。排名改变不一定意味着明显的性能反转，也不能当作统计显著差异。')
    md('question','## 这次回答的是排名，而不是倍率\n\n'
       '比较同一次实验的同一组五架无人机：先找 Medium 阶段谁耗电最快，再看它在 High 或 Low 是否仍最快。'
       '只有五架都具备完整可比阶段时，才计算五机排名；不把三机或四机排名称为五机排名。High/Medium 有 2 次合格飞行；Medium/Low 有 11 次。'
       '两者分母不同，不能拿 1/2 和 2/11 直接比较哪一阶段更稳定。\n\n'
       'High 从首次实际掉电开始，初始平台不计，之后正常阶梯平台保留。各电池使用自身三段分界点。所有排序都在未四舍五入的数值上完成。')
    md('modes','## 必须区分原始掉电速度与 Bideal 标准化结果\n\n'
       '原始耗电率表示电池显示百分比每分钟下降多少，仍包含电池本身差异；Bideal 标准化使用当前冻结的单电池基线进行校正。'
       '对你要减小电池差异、比较位置负载的目的，后者更相关，但仍取决于基线校正的有效性。\n\n'
       '原始数据中，Medium 最快者在 Low 仍最快为 10/11 次，这些 Medium 第一名都是 D5/B12；在 High 则为 0/2 次。'
       '标准化后，Low 保持第一名降到 2/11 次，说明“原始百分比掉得快”与“校正后负载大”不能混为一谈。')
    blocks.append(dict(id='summary_table',type='table',tableId='summary'))
    md('example','## 具体例子：Diamond / 75 cm / Tail / Lv2 / 001\n\n'
       '标准化后，这次实验 Medium 最快的是 **D2**，但 High 最快变成 **D5**。D5 在 Medium 是最慢的，在 High 却成为最快的。'
       'Low 的第一名仍是 D2，但与 D3 的差距只有约 0.022%，应视为非常接近。下图比较的是各阶段标准化平均耗电率，不是原始剩余电量。')
    blocks.append(dict(id='example_chart',type='chart',chartId='example_rates'))
    blocks.append(dict(id='example_table',type='table',tableId='example'))
    md('closeness','## Low 往往是名次改变，差距未必很大\n\n'
       '虽然只有 2/11 次保住第一，但 Medium 第一名在 Low 距离实际最快者不超过 5% 的有 6/11 次，不超过 10% 的有 9/11 次。'
       '这里按“最快耗电率 ÷ 原 Medium 第一名耗电率 − 1”计算差距，5% 和 10% 只是描述性阈值，不是统计检验。'
       '11 次 Low 比较没有一次五机完整顺序完全相同，因此即使保住第一，也不代表全部名次保持不变。')
    md('limits','## 结论的使用边界\n\n'
       '这些是已有记录的阶段平均排名，不是已证明的固定位置规律。不同电池进入同一 SOC 阶段的时间并不一致；部分 High 包含初始回正，Low 还可能包含其他无人机已经降落后的时间。'
       '小的名次差别也可能受整数 SOC 读数和基线拟合误差影响。不能把这些观察排名直接解释为位置气动效应。'
       '这次只分析现有数据，不修改原始数据、Bideal、飞行控制或冻结耗电率表，也不要求复测。')
    md('use','## 对你当前问题的直接答案\n\n'
       '**不能假定“Medium 耗电最快的无人机，在 High 和 Low 一定仍然最快”。**'
       '当前可用的是分阶段的实际耗电率和排序；Medium 可以提供参考，但不应被当作另外两个阶段的确定排序。'
       '如果后续决策关心的是节省多少时间，应使用实际耗电率差距，而不只看谁排第一。',False)
    def table(id,dataset,title,columns,sort):
        return dict(id=id,dataset=dataset,title=title,sourceId='rank_audit',defaultSort=dict(field=sort,direction='asc'),
                    columns=[dict(field=f,label=l,**({'type':'text'} if t else {'format':'number'})) for f,l,t in columns])
    display=summaries[['stage','mode','flights','same_fastest','top_two_within2pct']].copy()
    tables=[table('summary','summary','Medium 第一名能否保持第一',
                  [('stage','比较阶段',True),('mode','耗电率口径',True),('flights','五机实验数',False),('same_fastest','仍为第一的次数',False)],'mode'),
            table('example','example','Bideal 耗电率与名次：名次1代表最快；单位为百分点/分钟',
                  [('drone','无人机',True),('battery','电池',True),('high_rate','High 耗电率',False),('high_rank','High 名次',False),
                   ('medium_rate','Medium 耗电率',False),('medium_rank','Medium 名次',False),
                   ('low_rate','Low 耗电率',False),('low_rank','Low 名次',False)],'drone')]
    chart=dict(id='example_rates',title='五架无人机在三个阶段的 Bideal 耗电率',
               subtitle='Diamond / 75 cm / Tail / Lv2 / 001；单位为 Bideal 电量百分点/分钟；越高表示下降越快',showDescription=True,
               type='bar',intent='comparison',dataset='example_rates',sourceId='rank_audit',layout='full',valueFormat='number',
               encodings=dict(x=dict(field='drone',type='nominal',label='无人机'),
                              y=dict(field='rate',type='quantitative',label='Bideal 百分点/分钟'),
                              color=dict(field='stage',type='nominal',label='SOC阶段'),
                              tooltip=[dict(field='rank',type='quantitative',label='本阶段名次'),dict(field='battery',type='nominal',label='电池')]),
               settings=dict(groupMode='grouped',orientation='vertical',sort='none',showValues=True),
               palette=dict(kind='categorical'),legend=dict(position='bottom',sort='spec'),labels=dict(values='all'))
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,generatedAt=generated,
                    blocks=blocks,charts=[chart],tables=tables,sources=[source]),
                  snapshot=dict(version=1,status='ready',generatedAt=generated,datasets={'summary':records(display),
                    'example':example,'example_rates':plot,'all_comparisons':records(results)}),sources=[source])
    (OUT/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    cells=[dict(cell_type='markdown',id='context',metadata={},source=[
        '## tl;dr\nMedium fastest does not guarantee fastest in other SOC stages.\n',
        '## Context & Methods\nFive-drone, same-flight comparisons; first-drop SOC start; raw and frozen-Bideal-normalized rankings.\n',
        '### Key Assumptions\nRanks are observational; small differences need not be statistically distinct.\n',
        '## Data\nSource files, model identity and hashes are in manifest.json.\n'])]
    code="from pathlib import Path\nimport sys,json\nimport pandas as pd\nimport numpy as np\nROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'battery_normalization.py').exists())\nsys.path.insert(0,str(ROOT/'output_py'))\nfrom audit_cross_stage_drone_rankings import calculate,OUT,digest\nkept,ranks,comparisons,summary=calculate()\nfor source,expected in json.loads((OUT/'manifest.json').read_text())['source_sha256'].items():\n    assert digest(ROOT/source)==expected\nsaved=pd.read_csv(OUT/'summary.csv')\nassert np.allclose(summary.same_fastest,saved.same_fastest)\nprint(summary.to_string(index=False))\nprint(comparisons[comparisons['mode'].eq('Bideal')][['experiment_id','stage','medium_order','stage_order']].to_string(index=False))\n"
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):exec(compile(code,'ranking_audit.ipynb','exec'),{})
    cells.extend([dict(cell_type='markdown',id='results',metadata={},source=['## Results\n']),
                  dict(cell_type='code',id='recompute',metadata={},source=code.splitlines(keepends=True),execution_count=1,
                       outputs=[dict(output_type='stream',name='stdout',text=captured.getvalue().splitlines(keepends=True))]),
                  dict(cell_type='markdown',id='takeaways',metadata={},source=[
                      '## Takeaways\nUse actual stage rates rather than copying Medium ranks. Low top ranks are often near ties. No new experiments are requested.\n',
                      'Execution: cells ran sequentially in Python, not in a Jupyter kernel (nbformat/nbclient/ipykernel unavailable). To validate with Jupyter, run `python -m jupyter nbconvert --execute --to notebook --inplace ranking_audit.ipynb`.\n'])])
    (OUT/'ranking_audit.ipynb').write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,
        metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}),ensure_ascii=False,indent=2)+'\n')
    print(summaries.to_string(index=False));print(pd.DataFrame(example).round(4).to_string(index=False))


if __name__=='__main__':main()
