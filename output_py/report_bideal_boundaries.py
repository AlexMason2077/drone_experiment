"""Reproducible boundary-search report; does not change the active v1 model."""
import contextlib
import io
import json
import math
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from search_bideal_boundaries import ROOT,OUT,run,fit,prepare,load_trace


def build():
    summary=run()
    comp=pd.read_csv(OUT/'comparison.csv')
    hist=pd.read_csv(OUT/'historical_checks.csv')
    sensitivity=pd.read_csv(OUT/'sensitivity.csv')
    folds=pd.read_csv(OUT/'leave_one_battery_out.csv')
    residuals=pd.read_csv(OUT/'residuals.csv')
    sql='''SELECT model,battery_id,run_id,upper,lower,source,COUNT(*) AS n_samples,
           sqrt(AVG(residual_pp*residual_pp)) AS rmse_pp,MAX(ABS(residual_pp)) AS max_error_pp
           FROM raw_hover_fit_residuals GROUP BY model,battery_id,run_id,upper,lower,source
           ORDER BY battery_id,model'''
    with sqlite3.connect(':memory:') as conn:
        conn.create_function('sqrt',1,math.sqrt)
        residuals.to_sql('raw_hover_fit_residuals',conn,index=False)
        checked=pd.read_sql_query(sql,conn)
    joined=comp.merge(checked,on=['model','battery_id','run_id'],suffixes=('_py','_sql'))
    assert len(joined)==15 and np.allclose(joined.rmse_pp_py,joined.rmse_pp_sql,rtol=1e-12)
    assert checked.run_id.nunique()==5 and not checked.run_id.eq('20260906_165812').any()
    # Independent algebraic re-evaluation avoids relying only on interpolation code.
    for key,g in residuals.groupby(['model','battery_id']):
        source=g.source.iloc[0];raw=pd.read_csv(ROOT/source)
        raw=raw[raw.phase.eq('hover_to_10_percent')]
        levels=[95,int(g.upper.iloc[0]),int(g.lower.iloc[0]),20]
        times=[float(raw.loc[raw.battery.le(s),'elapsed_time'].iloc[0]) for s in levels]
        pred=np.empty(len(g))
        t=g.elapsed_time.to_numpy()
        for i in range(3):
            m=(t>=times[i])&(t<=times[i+1])
            pred[m]=levels[i]+(levels[i+1]-levels[i])*(t[m]-times[i])/(times[i+1]-times[i])
        assert np.allclose(pred,g.modeled_soc,atol=1e-9)
    overall=checked.groupby('model').rmse_pp.apply(lambda s:np.sqrt(np.mean(s*s)))
    old=overall['original']
    models={'original':'75% / 40%','best':'80% / 56%','rounded':'80% / 55%'}
    checked['方案']=checked.model.map(models)
    hist['方案']=hist.model.map(models)
    overview=[]
    for name in ['original','best','rounded']:
        hx=hist[hist.model.eq(name)].rmse_pp
        overview.append(dict(model=name,label=models[name],rmse_pp=float(overall[name]),
            improvement_pct=float((1-overall[name]/old)*100),n_current=5,
            historical_rmse_pp=float(np.sqrt(np.mean(hx*hx))),n_historical=len(hx)))
    title='Bideal三段分界线：75/40、80/56与80/55的比较'
    headline='''## 当前数据更支持约80%与55%–56%，而不是75%与40%
采用9月6日五个当前配对的无风Hover实验，B12只用D5，D2那次始终排除。在95%→20%共同覆盖范围内，以1个百分点为步长搜索，共有1,886组满足约束的共同分界线。
在本次搜索和端点线性化方法下，80%／56%的等权总体RMSE最低：1.667个百分点，相比75%／40%的2.504下降33.4%。取整到80%／55%为1.728，下降31.0%。
建议把80%／56%作为数值候选，80%／55%作为易于报告和执行的近似选项；56%并不是已证明的电池物理转折点。本次只比较并提出候选，不覆盖当前75%／40%的v1模型。
'''
    methods='''## 比较口径：每块电池分别拟合，再等权比较
固定三段连续直线。对每个候选边界，取真实SOC首次经过95%、上边界、下边界、20%的四个时刻，并连接这四个端点；每块电池都有自己的斜率，但五块电池共用两条SOC边界。
RMSE由整个95→20%区间的真实悬停SOC与该电池自身三段直线的差计算，单位为SOC百分点；总体值为五次飞行MSE的等权平均后开根号。不是将同一条Bideal曲线硬套到五块电池上，也不是把每秒样本当成独立重复实验。
各段至少跨5个百分点、持续10秒；还检查了更宽区间和更长持续时间。边界比较使用相同数据范围和相同计算方法，未改原始SOC、时间或补数据。
'''
    current='''## 改善主要来自B13和B14，并非每块电池改善相同
80%／56%下，B10到B14的RMSE分别为1.268、1.469、1.456、1.659、2.292；原方案分别为1.516、1.514、1.600、3.775、3.155。
图中柱越低表示该电池的三段近似越贴近原始曲线。80%／56%在五块电池上都低于原方案，但B11改善很小。取整的80%／55%会使B11略变差（1.514→1.573），因此不能声称取整方案对每块电池都更好。
'''
    robustness='''## 五次较早实验和范围敏感性检查也支持调整边界
五次较早、相同电池–机体编号配对的可比较Hover记录中，80%／56%逐次均比75%／40%误差小，等权总体RMSE从3.135降到1.933；80%／55%为1.999。旧实验与新实验的日期、状态可能不同，未合并计算当前斜率。
每次去掉一块当前电池、用另外四块选边界，得到80/56（三次）、85/56、79/51。改变共同分析范围后：90→20得到79/56；95→25和95→15仍为80/56；95→11为80/58。将最小阶段宽度增至10个百分点、或最小时长增至15秒，仍为80/56。
因此更可靠的解释是：上边界约80%，下边界位于50%多的区域，比40%更合适；不应把整数最优56%说成精确不变的物理常数。
'''
    limitations='''## 局限与下一步
上述误差是曲线近似误差，不是决策模型准确率，也不是未知飞行的SOC预测误差。旧实验检查和去掉一块电池的检查都重新使用各自真实端点来校准斜率，验证的是分界线的适用性，不是固定耗电率跨飞行迁移的准确性。
当前每个配对只采用一次最新完整实验；搜索最优存在样本内选择偏差，重复实测仍是更强的验证。取整是一种可解释性选择，不是误差最小化本身。
主搜索依据95→20%的实际悬停数据，不据此验证100→95%或20%以下；95→11仅作已有观测范围的敏感性检查，仍不能外推为0%的实测依据。
若采用新边界，需要按新边界重算三段Bideal率和每块电池的转换系数，而不是只改high、medium、low的名称。当前v1及冻结耗电率表保持不变。
'''
    source=dict(id='boundary-search',label='五次真实Hover实验的分界线搜索与原始残差复核',path='analysis_results/bideal_boundary_search_20260908/summary.json',
        query=dict(sql=sql,engine='sqlite',tables_used=['raw_hover_fit_residuals'],
        codePath='output_py/search_bideal_boundaries.py',
        description='原始Hover数据→首次SOC阈值→连续端点直线→全部样本残差→SQLite复算每次实验RMSE。',
        metric_definitions={'rmse_pp':'sqrt(mean((observed_SOC-modeled_SOC)^2)); pooled=sqrt(mean(per_run_MSE))'},
        filters=['five current battery-airframe pairs from2026-09-06','exclude B12/D2/20260906_165812','observed95→20','stage width>=5pp and duration>=10s']))
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,sources=[source],
        blocks=[dict(id='title',type='markdown',body='# '+title),dict(id='summary',type='markdown',body=headline),
            dict(id='methods',type='markdown',body=methods),dict(id='current',type='markdown',body=current),
            dict(id='current-chart',type='chart',chartId='current'),dict(id='overview-table',type='table',tableId='overview'),
            dict(id='robustness',type='markdown',body=robustness),dict(id='history-table',type='table',tableId='history'),
            dict(id='limits',type='markdown',body=limitations)],
        charts=[dict(id='current',type='bar',title='五块电池在三种分界线下的拟合RMSE',dataset='current',source=source,
            encodings=dict(x=dict(field='battery_id',label='电池'),y=dict(field='rmse_pp',label='RMSE（SOC百分点）'),color=dict(field='方案',label='分界线')),
            options=dict(grouping='grouped'),palette=dict(kind='categorical',colors=['#3B6FB6','#D8A739','#D66B32']))],
        tables=[dict(id='overview',title='当前五次实验的总体误差',dataset='overview',source=source,
            defaultSort=dict(field='rmse_pp',direction='asc'),columns=[dict(field=f,label=l) for f,l in [
                ('label','分界线'),('rmse_pp','总体RMSE（百分点）'),('improvement_pct','误差降低（%）'),('n_current','实验次数')]]),
            dict(id='history',title='五次较早同配对实验的误差复核',dataset='history',source=source,
                defaultSort=dict(field='battery_id',direction='asc'),columns=[dict(field=f,label=l) for f,l in [
                    ('battery_id','电池'),('run_id','历史Run'),('方案','分界线'),('rmse_pp','RMSE（百分点）')]])]),
        snapshot=dict(version=1,status='ready',datasets=dict(current=json.loads(checked.to_json(orient='records')),
            overview=overview,history=json.loads(hist.to_json(orient='records')))))
    (OUT/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
    (OUT/'README.md').write_text('# '+title+'\n\n'+headline+'\n'+methods+'\n'+current+'\n'+robustness+'\n'+limitations)
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',independent_residual_checks=15,source_run_count=5,
        excluded_run_absent=True,active_model_changed=False),indent=2))
    # Executable companion; all code cells are rerun sequentially in a clean namespace.
    def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True))
    def code(s):return dict(cell_type='code',metadata={},source=s.splitlines(keepends=True),execution_count=None,outputs=[])
    cells=[md('# '+title+'\n\n## tl;dr\n80/56在当前搜索中最低误差；80/55为接近的取整选项。'),
        md('## Context & Methods\n### Key Assumptions\n'+methods.replace('## 比较口径：每块电池分别拟合，再等权比较\n','')),
        md('## Data\n五个同日当前配对；B12只用D5。详细原始来源和哈希见summary.json。'),
        code("import sys,json\nfrom pathlib import Path\nroot=Path('/Users/alexmason/Downloads/drone_experiment')\nsys.path.insert(0,str(root/'output_py'))\nfrom search_bideal_boundaries import run,OUT\ns=run()\n"),
        md('## Results\n误差按独立飞行等权，不按采样点数给电池加权。'),
        code("import pandas as pd, numpy as np\ng=pd.read_csv(OUT/'search.csv')\nprint(g[['upper','lower','rmse']].head(6).to_string(index=False))\nprint(pd.read_csv(OUT/'sensitivity.csv').to_string(index=False))\nassert int(g.iloc[0].upper)==80 and int(g.iloc[0].lower)==56\n"),
        md('## Takeaways\n'+limitations.replace('## 局限与下一步\n',''))]
    env={};count=0
    for cell in cells:
        if cell['cell_type']!='code':continue
        count+=1;buf=io.StringIO()
        with contextlib.redirect_stdout(buf):exec(compile(''.join(cell['source']),'<notebook-cell>','exec'),env)
        cell['execution_count']=count;cell['outputs']=[dict(output_type='stream',name='stdout',text=buf.getvalue().splitlines(keepends=True))]
    nb=dict(nbformat=4,nbformat_minor=4,cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        execution_note='Executed sequentially in fresh CPython namespace; no Jupyter kernel packages required.'))
    (OUT/'boundary_search.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2))
    print('Independent residual QA passed; notebook executed; v1 unchanged.')


if __name__=='__main__':build()
