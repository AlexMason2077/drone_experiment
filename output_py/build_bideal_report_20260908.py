"""Build reproducible notebook and native report payload from reviewed calculations."""
from pathlib import Path
import contextlib
import io
import json
import sys
import sqlite3
import pandas as pd
from analyze_bideal_20260908 import ROOT, OUT

TITLE='B10–B14耗电规律与Bideal校准草案'
METHODS='''## Context & Methods
### Key Assumptions
用户确认2026-09-06单独Hover baseline均无风且设置相同高度；lv1是默认标签。
以最新同日、当前常用电池–机体配对为主要校准队列：B11/D1、B10/D2、B13/D3、B14/D4、B12/D5。
按用户要求排除20260906_165812（B12/D2），不用于本分析的统计、曲线或对照；原始文件保留，排除规则固定在分析脚本中。
高区间先用95%→75%以避开满电平台，并保证跨机可比较；不能声称覆盖100%→95%。Medium为75%→40%，Low为40%→20%。
只使用真实hover_to_10_percent记录，起降不计入。平均率=SOC下降百分点×60/首次观察到两个边界的时间差。
不插值、不拼接、不补数据。边界跳过、时间断点>5秒、单次SOC跳变>2个百分点等排除。
单位是设备报告的SOC百分点/分钟，不是Wh/min；每条独立飞行是一个样本，不把每秒记录当独立重复。
OLS斜率作为敏感性检查。分区均率不意味着区间内部严格线性。
历史记录分开列出；Wind Tunnel只作辅助检查，不用于分离电池与位置/机体效应。
'''
TLDR='''## tl;dr
当前配对下，B13呈现高区间快、medium慢、low再次变快的特征。五个电池low40→20均比medium75→40掉电更快，但加速比例不同。
B12在9月6日仅采用D5数据，D2那次已按用户要求排除。当前仍是电池–机体配对的SOC分段Bideal草案，跨机体通用性尚未验证。
'''
TAKEAWAYS='''## Takeaways
第一版只发布可追溯的草案，不覆盖既有冻结表和飞行控制/训练代码。以同日五个常用配对分区率的等权均值定义Bideal参考率，a_b,k=r_ideal,k/r_b,k。
同一SOC区间内归一化下降量：DeltaSOC_ideal=a_b,k*DeltaSOC_reported。跨区间需拆段求和；这不是直接把绝对SOC乘系数。
迁移到不同机体、风力和负载是待验证假设。高区间历史变化大；每个当前配对同日仅一次完整样本，暂不作为最终通用系数。
'''


def md(text):
    return dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True))


def code(text):
    return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True))


def build():
    cells=[md('# '+TITLE+'\n\n'+TLDR),md(METHODS),md('## Data\n只读取原始数据库与注册信息，输出另存，不修改原始实验。'),
      code("import sys\nfrom pathlib import Path\nroot=Path('/Users/alexmason/Downloads/drone_experiment')\nsys.path.insert(0,str(root/'output_py'))\nfrom analyze_bideal_20260908 import analyze\nresult=analyze()\nprint(result['inventory'].status.value_counts().to_string())\nprint('原始Wind Tunnel文件数:',len(result['wind_inventory']))\n"),
      md('## Results\n### 最新同日Hover：按当前配对选取，每配对一个样本'),
      code("rates=result['coefficients'].pivot(index='battery_id',columns='band',values='rate_pp_min')\nprint(rates.round(3).to_string())\nprint('low/medium倍率:')\nprint((rates.low40_20/rates.medium75_40).round(3).to_string())\n"),
      md('### B12采用D5数据及其余历史记录（已排除9月6日D2那次）'),
      code("r=result['recent']\nprint(r[r.battery_id.eq('B12')][['drone','band','rate_pp_min','tof_median']].round(3).to_string(index=False))\nb=result['bands']\nprint(b[b.valid & b.band.ne('low20_10')].pivot_table(index=['battery_id','drone','run_id'],columns='band',values='rate_pp_min').round(3).to_string())\n"),
      md('### 可追溯草案与独立计算复核'),
      code("import numpy as np\nc=result['coefficients']\nassert np.allclose(c.rate_pp_min*c.physical_to_bideal_scale,c.bideal_pp_min)\nfor _,row in c.iterrows():\n    raw=pd.read_csv(root/row.source) if 'pd' in globals() else __import__('pandas').read_csv(root/row.source)\n    raw=raw[raw.phase.eq('hover_to_10_percent')]\n    start=raw.loc[raw.battery.le(row.upper),'elapsed_time'].iloc[0]\n    end=raw.loc[(raw.elapsed_time>start)&raw.battery.le(row.lower),'elapsed_time'].iloc[0]\n    assert abs((row.upper-row.lower)*60/(end-start)-row.rate_pp_min)<1e-10\nprint('15个原始区间阈值复核通过；归一化恒等式通过。')\nprint(c[['battery_id','band','bideal_pp_min','physical_to_bideal_scale']].round(4).to_string(index=False))\nprint('Wind Tunnel同次飞行low/medium辅助对照（有删失，不作为电池校准）:')\nprint(result['wind_pairs'].groupby('battery_id').low_medium_multiplier.agg(['count','median']).round(3).to_string())\n"),md(TAKEAWAYS)]
    # Execute all code cells in one clean CPython namespace; no kernel packages required.
    # All computations are plain Python; outputs are captured in nbformat 4 structure.
    env={}
    count=0
    for cell in cells:
        if cell['cell_type']!='code':continue
        count+=1
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(''.join(cell['source']),'<notebook-cell>','exec'),env)
        cell['execution_count']=count
        cell['outputs']=[dict(output_type='stream',name='stdout',text=buf.getvalue().splitlines(keepends=True))]
    notebook=dict(nbformat=4,nbformat_minor=4,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                 execution_note='Executed sequentially in a fresh CPython namespace; no Jupyter kernel dependency.'),cells=cells)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'bideal_analysis.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2))
    result=env['result']
    for name,df in result.items():df.to_csv(OUT/(name+'.csv'),index=False)
    rates=result['coefficients'].copy()
    names={'high95_75':'High 95→75%','medium75_40':'Medium 75→40%','low40_20':'Low 40→20%'}
    rates['soc_band']=rates.band.map(names)
    rates['rate_pp_min']=rates.rate_pp_min.round(3)
    rates['scale']=rates.physical_to_bideal_scale.round(4)
    rows=json.loads(rates.to_json(orient='records'))
    for name in ['bands','traces','recent','matched','coefficients']:
        df=result[name]
        assert not ((df.battery_id=='B12') & (df.drone=='drone_2') & (df.run_id=='20260906_165812')).any()
    sql='''SELECT battery_id, drone, run_id, band, source, upper, lower,
           start_s, end_s, (end_s-start_s) AS duration_s,
           60.0*(upper-lower)/(end_s-start_s) AS rate_pp_min
           FROM baseline_threshold_crossings
           WHERE run_id LIKE '20260906%' AND valid=1 AND band <> 'low20_10'
           ORDER BY battery_id, drone, band'''
    with sqlite3.connect(':memory:') as conn:
        result['bands'].to_sql('baseline_threshold_crossings',conn,index=False)
        checked=pd.read_sql_query(sql,conn)
    original=result['recent'].query("band!='low20_10'")
    joined=checked.merge(original,on=['battery_id','drone','run_id','band'],suffixes=('_sql','_python'))
    assert len(joined)==15 and ((joined.rate_pp_min_sql-joined.rate_pp_min_python).abs()<1e-10).all()
    source=dict(id='baseline',label='真实单独Hover baseline；用户确认9月6日无风、相同设定高度',
                path='database/baselines/',query=dict(description='按metadata去重，优先原始all_coordination，其次timeseries。每次飞行按首次SOC阈值时刻求分区率。',
                sql=sql,engine='sqlite',tables=['baseline_threshold_crossings'],
                codePath='output_py/analyze_bideal_20260908.py',
                staging='baseline_threshold_crossings由本分析真实边界提取结果bands.csv加载；不是远程数据库。'))
    summary='''## 结论：分区特征明确，但不能使用全SOC固定系数
已检查B10–B14的32个Hover元数据记录：排除用户指定的B12/D2一次后，21个有实际悬停记录、10个未进入悬停；17次至少覆盖一个可比较区间，15次完整覆盖本分析三个区间。
最新9月6日采用5次完整实验，B12只采用D5；每个当前常用配对一次，未将重复采样点当成独立实验。B12/D2的20260906_165812不再用于统计或对照，原始文件保留。
B13在95→75%平均21.32 pp/min、75→40%为7.36、40→20%为14.53。B10/B11的medium几乎一致；五个配对的low相对medium增速分别为30.7%、44.5%、48.2%、97.5%、62.5%（B10至B14）。
'''
    defs='''## 定义与方法
图表均为设备SOC百分点/分钟（pp/min），不是电能。高区间仅95→75%，避免100%平台；不能把这个率宣称为整个100→75%的精确平均率。
Medium为75→40%、Low为40→20%。只使用真实悬停样本，去掉起降；区间率=下降百分点÷阈值间实际分钟数。
不修改时间、不补数据，原始文件路径及SHA256保存在inventory.csv。每区间另有OLS斜率与残差用于检查曲率。
'''
    draft='''## Bideal草案：先规范当前电池–机体配对
以五个常用配对同日区间率的等权平均作为参考，得到95→75%：10.9915，75→40%：9.0742，40→20%：14.0300 pp/min。
定义 a_b,k = r_ideal,k / r_b,k；同区间 DeltaSOC_ideal = a_b,k × DeltaSOC_reported。跨区间拆段计算，不把绝对SOC直接乘系数。
这是人为定义的参考尺度，不是额外测得的理想电池；均值选择与现有旧版（含B15、不同SOC范围）不同。草案另存，不覆盖冻结表、训练数据或飞行程序。
'''
    limits='''## 证据边界与后续验证
已移除基于9月6日B12/D2那次实验的机体对照图及差异结论。当前校准仍限于选定电池–机体配对；排除这条数据不等于证明机体之间没有差异。
历史高区间不稳定：B11从5月21.72变为9月7.20 pp/min，B14从22.54变为8.78；B13 medium在已有三次为7.36–7.89。不能把跨月记录混成一个精确常数，也不能仅据此断言老化或故障。
Wind Tunnel辅助扫描163个文件，排除prepare、合成拼接和标记异常，并截到首次落地/明确故障。严格位置质量筛选后仅15组同次飞行low/medium对照（B12 13、B13 1、B11 1），其倍率均大于1；B10和B14受首机降落截断，没有完整low40→20配对。不能以此证明五个电池在所有队形上的通用规律。
主校准每个配对只有一次最新完整样本；三段均率不代表各段严格线性。100→95%和低于20%的映射未纳入本版系数；更不能外推至0%。
下一步优先重复当前五个配对的无风baseline验证系数稳定性。若需把电池参数迁移到其他机体，需要可辨识的交叉对照；Wind Tunnel不同位置的率不能直接充当纯电池校准。
'''
    manifest=dict(version=1,surface='report',title=TITLE,sources=[source],
      blocks=[dict(id='title',type='markdown',body='# '+TITLE),dict(id='summary',type='markdown',body=summary),
              dict(id='methods',type='markdown',body=defs),dict(id='rates-chart',type='chart',chartId='rates'),
              dict(id='draft',type='markdown',body=draft),
              dict(id='coeff-table',type='table',tableId='coefficients'),dict(id='limits',type='markdown',body=limits)],
      charts=[dict(id='rates',type='bar',title='当前五个电池–机体配对的分区掉电率',dataset='rates',sourceId='baseline',source=source,
                   encodings=dict(x=dict(field='battery_id',label='电池'),y=dict(field='rate_pp_min',label='SOC百分点/分钟'),color=dict(field='soc_band',label='SOC区间')),
                   options=dict(grouping='grouped'),palette=dict(kind='categorical',colors=['#3B6FB6','#D8A739','#D66B32']))],
      tables=[dict(id='coefficients',title='分区归一化系数（草案）',dataset='rates',sourceId='baseline',source=source,
                    defaultSort=dict(field='battery_id',direction='asc'),columns=[dict(field=f,label=l) for f,l in [
                    ('battery_id','电池'),('drone','机体'),('soc_band','SOC区间'),('rate_pp_min','原始掉电率'),('scale','转换系数'),('run_id','来源Run')]])])
    artifact=dict(surface='report',manifest=manifest,snapshot=dict(version=1,status='ready',datasets=dict(rates=rows)))
    (OUT/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
    (OUT/'README.md').write_text('# '+TITLE+'\n\n'+summary+'\n'+defs+'\n'+draft+'\n'+limits+'\n\n重现：python3 output_py/build_bideal_report_20260908.py。代码与Notebook逐条核对15个主要区间的阈值计算；所有原始实验均未修改。\n')
    print('Notebook executed: 4 code cells; 15 raw threshold checks passed.')
    print(OUT/'artifact.json')


if __name__=='__main__':build()
