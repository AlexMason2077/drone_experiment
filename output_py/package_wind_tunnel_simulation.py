"""Reproducible report datasets, notebook companion, and native report artifact."""
from pathlib import Path
from datetime import datetime,timezone
import base64
import contextlib
import io
import json
import sqlite3
import sys

import numpy as np
import pandas as pd

from wind_tunnel_simulation_model import ROOT,OUTPUT,BASE_SEED,BATTERIES,STAGES,sha
from generate_wind_tunnel_simulation import DEST

SQL='''
WITH per_curve_stage AS (
 SELECT source, drone_id, stage, COUNT(*) AS observed_soc_steps,
        SUM(duration_s) AS actual_time_s,
        SUM(predicted_dwell_s) AS predicted_time_s
 FROM holdout_predictions GROUP BY source, drone_id, stage
)
SELECT source,drone_id,stage,observed_soc_steps,actual_time_s,predicted_time_s,
       100.0*ABS(predicted_time_s-actual_time_s)/actual_time_s AS absolute_error_pct,
       60.0*observed_soc_steps/actual_time_s AS actual_rate_pp_min,
       60.0*observed_soc_steps/predicted_time_s AS predicted_rate_pp_min
FROM per_curve_stage
ORDER BY absolute_error_pct DESC,source,drone_id,stage;
'''


def records(d):return json.loads(d.to_json(orient='records',force_ascii=False,double_precision=8))


def curve_data():
    ref='front_75_tail_lv1'
    p=ROOT/'database/wind_tunnel_front_75_tail_lv1_001/wind_tunnel_front_75_tail_lv1_001_20260903_172757_all_coordination.csv'
    raw=pd.read_csv(p,usecols=['drone_name','elapsed_time','battery','phase'])
    points=[]
    for drone,g in raw.groupby('drone_name'):
        g=g.sort_values('elapsed_time');t0=g[g.phase.eq('wind_tunnel_takeoff')].elapsed_time.iloc[0]
        g=g[g.elapsed_time.ge(t0)];land=g[g.phase.str.contains('land')].elapsed_time.min()
        g=g[g.elapsed_time.le(land)]
        changed=g.battery.ne(g.battery.shift());indices=np.flatnonzero(changed)
        for idx in sorted(set(indices.tolist()+[max(0,i-1) for i in indices]+[len(g)-1])):
            r=g.iloc[idx];points.append(dict(panel='Observed reference',series=drone.replace('drone_','D'),
                time_s=float(r.elapsed_time-t0),soc=int(r.battery),condition_id=ref,data_origin='observed',source=str(p.relative_to(ROOT))))
    manifest=pd.read_csv(DEST/'run_manifest.csv');events=pd.read_csv(DEST/'integer_soc_events.csv.gz')
    examples=manifest[manifest.condition_id.eq(ref)].sort_values('seed').head(2)
    for n,r in enumerate(examples.itertuples(),1):
        e=events[events.experiment_id.eq(r.experiment_id)]
        for drone,g in e.groupby('drone_id'):
            for z in g.sort_values('end_s').itertuples():
                for t,b in [(z.start_s,z.soc_start),(z.end_s-.00001,z.soc_start),(z.end_s,z.soc_end)]:
                    points.append(dict(panel=f'Simulation seed {r.seed}',series=drone.replace('drone_','D'),time_s=max(0,t),
                        soc=int(b),condition_id=ref,data_origin='simulated',source=r.experiment_id))
    return pd.DataFrame(points)


def plot_notebook_figures(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#3B6FB6','#C28824','#D5753D','#7D873D','#C46796']
    panels=curves.panel.unique();fig,axs=plt.subplots(1,len(panels),figsize=(16,5.5),sharex=True,sharey=True)
    xmax=np.ceil(curves.time_s.max()/100)*100
    for ax,panel in zip(axs,panels):
        q=curves[curves.panel.eq(panel)]
        for j,(series,g) in enumerate(q.groupby('series')):
            ax.plot(g.time_s,g.soc,label=series,color=colors[j],lw=1.6)
        ax.set_title(panel,fontsize=12);ax.grid(color='#e8e8e8',lw=.6)
        ax.set(xlabel='Elapsed time since takeoff (s)',xlim=(0,xmax),ylim=(18,102))
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Reported battery SOC (%)');axs[-1].legend(ncol=5,loc='upper right',fontsize=8)
    fig.suptitle('Front / 75 cm / tail wind / Level 1: observed and synthetic SOC curves',fontsize=14)
    fig.text(.02,.01,'Observed reference is in the production fit, not a holdout. Synthetic curves start at 100%; integer SOC, independent seeds, no fabricated sensor telemetry.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(OUTPUT/'notebook_curve_comparison.png',dpi=150);plt.close(fig)
    v=pd.read_csv(OUTPUT/'matched_distribution_metrics.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,r in zip(axs,v.itertuples()):
        for x,prefix,color in [(0,'observed','#3B6FB6'),(1,'simulated','#C28824')]:
            lo=getattr(r,prefix+'_p10_s');mid=getattr(r,prefix+'_median_s');hi=getattr(r,prefix+'_p90_s')
            ax.vlines(x,lo,hi,color=color,lw=3);ax.plot(x,mid,'o',color=color)
        ax.set_xticks([0,1],['Observed','Simulated']);ax.set_title(r.stage.title())
        ax.set_xlim(-.5,1.5);ax.set_ylim(0,max(v.observed_p90_s.max(),v.simulated_p90_s.max())*1.2)
        ax.grid(axis='y',color='#e8e8e8');ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Seconds per 1 percentage-point drop')
    fig.suptitle('Matched holdout dwell distributions: median and 10th–90th percentiles')
    fig.text(.02,.01,'Same held-out covariates; 20 stochastic draws. Descriptive distribution check, not proof that missing conditions are correct.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(OUTPUT/'notebook_dwell_comparison.png',dpi=150);plt.close(fig)


def make_notebook(summary):
    specs=[('markdown','summary','## tl;dr\nAn explicitly synthetic empirical wind-tunnel simulator is available. It preserves integer SOC and stochastic step timing. It is not a validated replacement for missing experiments.\n'),
    ('markdown','methods','## Context & Methods\n### Key Assumptions\nPer-battery baseline rates are offsets. Conditional log-dwell prediction uses SOC, own stage, startup progress and condition. Holdout groups keep conditions, flights and drones together. Randomness uses development out-of-fold residual blocks. Unmodelled coordinates, attitude and temperature are blank. The generator was revised after internal synthetic QA; held-out stochastic distribution checks are descriptive.\n'),
    ('code','setup',"from pathlib import Path\nimport sys,json\nimport pandas as pd\nROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'battery_normalization.py').exists())\nsys.path.insert(0,str(ROOT/'output_py'))\nfrom wind_tunnel_simulation_model import OUTPUT,sha\nfrom generate_wind_tunnel_simulation import DEST\nfrom package_wind_tunnel_simulation import curve_data,plot_notebook_figures,SQL\n"),
    ('markdown','data','## Data\nObserved source files and frozen baseline hashes are in input_manifest.json; generated files are listed in run_manifest.csv. The simulator never imports an aircraft SDK.\n'),
    ('code','verify',"m=json.loads((OUTPUT/'input_manifest.json').read_text())\nfor path,expected in m['input_sha256'].items():\n    assert sha(ROOT/path)==expected\nprint('Source hashes unchanged:',len(m['input_sha256']))\nprint(pd.read_csv(DEST/'coverage.csv').groupby('coverage').size().to_string())\n"),
    ('markdown','results','## Results\nDeterministic prediction is evaluated on untouched held-out conditions. Metrics sum predicted and observed duration for the SAME eligible SOC steps, not nominal whole stages.\n'),
    ('code','metrics',"import sqlite3\nd=pd.read_csv(OUTPUT/'holdout_predictions.csv')\ncon=sqlite3.connect(':memory:')\nd.to_sql('holdout_predictions',con,index=False)\nr=pd.read_sql_query(SQL,con)\nprint('Curve-stage mean absolute percentage time error:',r.absolute_error_pct.mean())\nprint(pd.read_csv(OUTPUT/'matched_distribution_metrics.csv').to_string(index=False))\n"),
    ('code','plots',"plot_notebook_figures(curve_data())\nprint('Generated notebook_curve_comparison.png and notebook_dwell_comparison.png')\n"),
    ('markdown','takeaways','## Takeaways\nUse these simulations for algorithm exploration and sensitivity analysis, not as measured replications or independent experimental validation. High/Medium transfer errors remain material; battery, drone and position are confounded. No causal leader or spacing rule was hardcoded. No 2.5 m forward-flight rates were transplanted to hover.\n')]
    cells=[];env={};count=0
    for kind,id,body in specs:
        cell=dict(cell_type=kind,id=id,metadata={},source=body.splitlines(keepends=True))
        if kind=='code':
            count+=1;stream=io.StringIO()
            with contextlib.redirect_stdout(stream):exec(compile(body,'simulation_audit.ipynb','exec'),env)
            outputs=[dict(output_type='stream',name='stdout',text=stream.getvalue().splitlines(keepends=True))]
            if id=='plots':
                for name in ['notebook_curve_comparison.png','notebook_dwell_comparison.png']:
                    outputs.append(dict(output_type='display_data',metadata={},data={'image/png':base64.b64encode((OUTPUT/name).read_bytes()).decode()}))
            cell.update(execution_count=count,outputs=outputs)
        cells.append(cell)
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        execution_note='Executed code cells sequentially in a shared Python namespace. nbformat/nbclient unavailable: no actual Jupyter-kernel execution.',
        rerun='python3 -m jupyter nbconvert --execute --to notebook --inplace simulation_audit.ipynb'))
    (OUTPUT/'simulation_audit.ipynb').write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n')


def package():
    summary=json.loads((OUTPUT/'training_summary.json').read_text());export=json.loads((OUTPUT/'export_validation.json').read_text())
    manifest=json.loads((DEST/'manifest.json').read_text());cov=pd.read_csv(DEST/'coverage.csv')
    h=pd.read_csv(OUTPUT/'holdout_predictions.csv');con=sqlite3.connect(':memory:');h.to_sql('holdout_predictions',con,index=False)
    detail=pd.read_sql_query(SQL,con);detail.to_csv(OUTPUT/'holdout_stage_time_details.csv',index=False)
    (OUTPUT/'report_metrics.sql').write_text(SQL)
    assert np.isclose(detail.absolute_error_pct.mean(),summary['holdout_metrics']['stage_time_mape_pct'])
    distribution=pd.read_csv(OUTPUT/'matched_distribution_metrics.csv');curves=curve_data();curves.to_csv(OUTPUT/'report_curve_points.csv',index=False)
    metrics=pd.read_csv(OUTPUT/'holdout_model_metrics.csv');stress=pd.read_csv(OUTPUT/'transfer_stress_tests.csv')
    generated=datetime.now(timezone.utc).isoformat();title='Wind-tunnel discharge simulation'
    src=dict(id='model',label='Observed SOC steps, grouped holdout predictions and simulator exports',path=str(OUTPUT.relative_to(ROOT)),
        query=dict(engine='SQLite',language='sql',sql=SQL,executed_at=generated,
        description='Executed over holdout_predictions.csv. Upstream extraction and model fitting: output_py/wind_tunnel_simulation_model.py; generation: output_py/generate_wind_tunnel_simulation.py; QA: output_py/validate_wind_tunnel_simulation.py.',
        tables_used=['holdout_predictions','observed_steps.csv','run_manifest.csv'],
        filters=['Original non-prepare, non-outlier, non-merged wind-tunnel records','All five reached hover; individual calibrated battery-drone pair',
                 'Valid one-percentage-point steps, own-pad and near-pad fractions >=0.5','Entire conditions held out together'],
        metric_definitions={'absolute_error_pct':'100*abs(sum(predicted step duration)-sum(observed step duration))/sum(observed step duration), per flight/drone/stage',
            'supported_cells':'5 eligible observed 1pp steps for a battery-position-stage; local support only, not necessarily full stage or >=3 repeats',
            'simulation_runs':'Separate seeded synthetic realizations, never measured experimental replicates'}))
    blocks=[]
    def md(id,body):blocks.append(dict(id=id,type='markdown',body=body,sourceId='model'))
    md('title','# '+title)
    md('summary',f'## 已生成仿真，但不能把缺失实验说成已经验证\n\n共生成 **{manifest["simulation_runs"]} 次明确标注的仿真**：53 个缺失或部分覆盖条件各 3 次，另有同一已测条件的 3 次示例。每次五架机从 100% 到各自 20% 降落，电量为整数，下降间隔具有随机性。真实实验、冻结 Bideal 和飞行代码没有改动。\n\n模型留出 6 个完整条件、10 次真实飞行后，同一批有效 SOC 片段的耗时误差平均为 **21.1%**，中位数为 **9.5%**。它适合探索和敏感性分析，尚不能称为高保真替代实验。')
    md('scope','## 相同字段不等于虚构全部传感器\n\n保留原记录的 62 列，另加仿真来源、模型版本和随机种子。已生成电量、仿真时间、实验条件、目标 Pad/目标坐标及示意阶段。**实测坐标、姿态、温度、速度、加速度等无法可靠建模的字段保留为空**；真实日期、真实设备 IP 也为空。这不是完整飞行物理仿真。起飞 3 秒、回正 3 秒、降落 3 秒是显示阶段假设，不是从运动数据拟合出的结果。')
    md('coverage','## 53 个待补条件并不表示你有 53 个实验完全没做\n\n60 个 formation × spacing × wind × level 组合中：36 个没有通过当前筛选的可用曲线，17 个只有部分电池/阶段，7 个的 15 个“位置×阶段”单元都有局部支持。这里“支持”只要求各单元至少 5 个有效 1% 下降片段，不等于完整曲线或三次重复。已做过但阶段缺失的条件也会得到独立仿真，原记录不被替换。')
    blocks.append(dict(id='coverage_table',type='table',tableId='coverage'))
    md('shape','## 随机性来自真实误差结构，不是给直线随便加噪声\n\n模型以每块电池的基准三段曲线为参照，学习每下降 1% 所需时间随 SOC、起始 SOC、起飞后的掉电进程、条件和仍悬停的机数如何变化。再叠加整次飞行的共同波动、各电池跨阶段相关偏差及连续片段的局部波动。不同随机种子产生不同曲线。以下真实参考参与了生产模型拟合，因此只用于形态示例，不是独立验证。')
    for k,panel in enumerate(curves.panel.unique()):blocks.append(dict(id=f'curve_block{k}',type='chart',chartId=f'curve{k}'))
    md('error','## 条件信息有一些帮助，但改善很有限\n\n开发集分组交叉验证中，条件模型平均耗时误差 19.9%，仅电池/SOC 模型为 20.2%；独立留出条件上反而分别为 21.1% 和 20.0%。因此不能声称已经可靠学到了风力、队形、间距的全部影响。固定电池三段基准在留出集为 29.1%。所有这些误差都按同一组已观测掉电片段计算，不是整段 100%→20% 飞行时长准确率。')
    blocks.append(dict(id='model_chart',type='chart',chartId='model_error'))
    md('stages','## High 和 Medium 比 Low 更难预测\n\n留出集 High、Medium、Low 的片段总时间平均绝对百分比误差分别约为 **26.2%、28.0%、8.3%**。单次 1% 下降时间的开发集误差区间，在留出点上的覆盖约 90.0%；这只是点级误差区间，不能解读为缺失条件整条曲线有 90% 概率正确。')
    md('distribution','## 生成后按相同条件再次核对了掉电间隔\n\n对留出记录的相同 SOC、条件等输入产生 20 组随机结果，比较每下降 1% 的中位时间和 10%–90% 范围。下面同时列出真实值与仿真值，保留差异，不把曲线调成完全一样。随机生成器在内部自检后修正过，因此本项是描述性复核，不是未经查看的最终检验。')
    blocks.append(dict(id='distribution_table',type='table',tableId='distribution'))
    md('transfer','## 缺少 50 cm 数据时，不能把 75 cm 当成完全等价\n\n仅用 75 cm 拟合，再预测已有的 50 cm front 记录，片段耗时平均误差约 **27.1%**。这项检查只覆盖 4 个 front 条件，不能替未测的其他 50 cm 队形背书。未见条件的随机偏差放大 1.25 倍、部分条件放大 1.10 倍，仅为显式敏感性假设，不是有实测保证的置信区间。')
    blocks.append(dict(id='stress_table',type='table',tableId='stress'))
    md('bounds','## 内部自检排除了把短片段波动放大为整程异常的做法\n\n最初原型出现异常快、慢的完整曲线，已保留在分析目录但不作为交付数据。最终整程公共偏差只从至少 100 个掉电片段、至少 3 架机的飞行中抽样；阶段偏差至少需要 5 个局部片段。正常掉电间隔限制在开发集各阶段 0.5%–99.5% 分位范围，目的是模拟常规悬停，而非故障或传感器异常尾部。原始记录没有裁剪或修改。初始平台单独从记录抽样，不解释为没有物理耗电，也不进入阶段耗电率计算。')
    md('limits','## 不强行写入未经验证的空气动力结论\n\n没有强制“Vee leader 必须最耗电”或“越近每架必然越耗电”。电池、无人机和位置基本固定，三者难以独立辨识；历史 front 的布局变更也未在统计模型中分离。B06 无匹配标定，不冒充 B12。新导出的目标几何读取当前布局函数，不表示生成了真实位置轨迹。100%→95% 的电池基准还包含既有模型的线性延伸。前进 2.5 m 的 Medium 耗电表属于另一种飞行工况，本次没有直接移植成悬停耗电率。')
    md('qa',f'## 文件与来源检查已完成\n\n核对了 {export["source_hashes_unchanged"]} 个输入哈希、追溯 {export["raw_endpoint_intervals_traced"]} 个原始掉电区间，检查 {export["runs_verified"]} 个导出文件的 {export["logger_rows_verified"]:,} 行。整数电量、逐机 20% 降落、单调时钟、相同种子可重现、不同种子有差异均通过。没有向无人机发送任何指令。')
    md('next','## 接下来可用于算法探索，不能用于证明实验结论\n\n使用 simulated_stage_rates_wide.csv 可查看每个条件、每次仿真、每个阶段的五位置原始/Bideal 耗电率；runs/ 是压缩的完整格式记录。请保持仿真与真实数据分开。可以用多种随机种子检验候选配置算法是否稳定，但不能用生成它们的同一模型当作独立真值验证算法。仍待真实数据回答的问题包括：未测条件的风场效应、50 cm 非 front 队形的差异、以及完整传感器联合行为。')
    md('references','## 方法依据\n\n[Tello SDK 2.0 官方说明](https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf)定义电量状态字段；[scikit-learn GroupKFold 文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)说明分组不重叠验证。数据来源、筛选与每个文件哈希见 input_manifest.json。')
    charts=[];datasets={'coverage':records(cov),'model_error':records(metrics),'distribution':records(distribution),'stress':records(stress),'holdout_details':records(detail)}
    def enc(field,kind,label):return dict(field=field,type=kind,label=label)
    for k,panel in enumerate(curves.panel.unique()):
        did=f'curve{k}';datasets[did]=records(curves[curves.panel.eq(panel)].sort_values(['series','time_s']))
        charts.append(dict(id=did,type='line',title=panel,subtitle='Front · 75 cm · tail wind · Level1；参考与仿真分开显示，单位为整数 SOC%',
            dataset=did,sourceId='model',layout='full',intent='trend',
            encodings=dict(x=enc('time_s','quantitative','Time (s)'),y=enc('soc','quantitative','SOC (%)'),color=enc('series','nominal','Drone'))))
    charts.append(dict(id='model_error',type='bar',title='Held-out condition prediction errors',subtitle='相同有效 SOC 片段的总时间 MAPE；6 个条件，10 次真实飞行，84 个单机阶段片段',
        dataset='model_error',sourceId='model',layout='full',intent='comparison',
        encodings=dict(x=enc('model','nominal','Model'),y=enc('stage_time_mape_pct','quantitative','MAPE (%)'))))
    tables=[]
    for id,label,cols,sort in [
        ('coverage','条件覆盖与待补范围',['condition_id','observed_record_files','usable_flights','supported_cells','coverage'],'condition_id'),
        ('distribution','相同留出输入下的掉电间隔（秒）',['stage','observed_steps','observed_median_s','simulated_median_s','observed_p10_s','simulated_p10_s','observed_p90_s','simulated_p90_s'],'stage'),
        ('stress','迁移压力检查',['test','conditions','flights','stage_time_mape_pct','stage_time_median_ape_pct'],'test')]:
        tables.append(dict(id=id,title=label,dataset=id,sourceId='model',columns=[dict(field=c,label=c) for c in cols],defaultSort=dict(field=sort,direction='asc')))
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,generatedAt=generated,blocks=blocks,charts=charts,tables=tables,sources=[src]),
        snapshot=dict(version=1,status='ready',datasets=datasets),sources=[src])
    (OUTPUT/'artifact.json').write_text(json.dumps(artifact,indent=2,ensure_ascii=False)+'\n')
    (OUTPUT/'report.md').write_text('\n\n'.join(b['body'] for b in blocks if b['type']=='markdown')+'\n')
    make_notebook(summary)
    plan=dict(audience='technical research; Chinese narrative',surface='mcp-artifact',companion='executed scientific notebook',
        chart_contracts=[dict(question='Do independent simulations retain non-uniform integer SOC timing?',family='trend',variants='three separate line panels; observed/in-sample reference and two seeded simulations',
            palette='relaxed five-category approved roots; series labels',grain='SOC event endpoints',qa='Notebook PNG inspection; native report validation'),
            dict(question='How does the selected predictor compare with baselines?',family='comparison',variant='bar',palette='single root',grain='four prespecified models, same held-out conditions')],
        notebook_execution='Code cells executed sequentially, not through Jupyter kernel (missing nbformat/nbclient).',
        assumptions='Independent empirical stochastic surrogate. No fabricated real telemetry; all uncertainty and exclusions recorded.')
    (OUTPUT/'report_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('Report and notebook ready',OUTPUT,flush=True)


if __name__=='__main__':package()
