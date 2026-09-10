"""Compare and save a repeated-hover mean B10/B11 calibration revision.

Primary artifact: a new candidate model, not a report or active flight setting.
User confirmed old and recent flights were no-wind with the same set height.
Average each battery's two same-pair runs equally at common SOC crossing points.
Preserve the fixed-80 recent-only model as an explicitly constrained alternative.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from battery_normalization import BatteryNormalizer, DischargeCurve, normalize_csv
from output_py.search_bideal_boundaries import load_trace, prepare, search, fit
from output_py.build_battery_normalization_candidate import mean_rate_curve, fit_reference

PREVIOUS=ROOT/'analysis_results/battery_normalization_candidate_20260908'
DEFAULT_OUT=ROOT/'analysis_results/battery_normalization_pooled_v2_20260909'
TARGETS={'B10':'drone_2','B11':'drone_1'}
UPPER_CONSTRAINT=80
MIN_WIDTH,MIN_SECONDS=10,20


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_sources():
    inventory=pd.read_csv(ROOT/'analysis_results/bideal_20260908/inventory.csv',dtype={'run_id':str})
    target=inventory[inventory.battery_id.isin(TARGETS)].copy()
    # Fresh catalog check: do not silently miss a new hover baseline after the last inventory.
    for path in (ROOT/'database/baselines').glob('drone_*_B1[01]/*hover*metadata.json'):
        metadata=json.loads(path.read_text())
        if metadata.get('run_id') not in set(target.run_id):
            raise ValueError(f'New baseline needs inventory review: {path}')
    traces,profiles,individual=[],[],[]
    for row in target.itertuples():
        record=dict(battery_id=row.battery_id,drone_id=row.drone,run_id=row.run_id,
            source=row.source,sha256=row.sha256,status=row.status,
            current_pair=row.drone==TARGETS[row.battery_id])
        if sha(ROOT/row.source)!=row.sha256:raise ValueError('Changed source: '+row.source)
        if row.status!='observed_hover':
            record['use']='excluded_no_hover';profiles.append(record);continue
        raw=load_trace(row);prepared=prepare(raw)
        samples=pd.read_csv(ROOT/row.source)
        samples=samples[samples.phase.eq('hover_to_10_percent')]
        record.update(hover_samples=len(samples),soc_min=float(samples.battery.min()),soc_max=float(samples.battery.max()),
            duplicate_times=int(samples.elapsed_time.duplicated().sum()),null_soc=int(samples.battery.isna().sum()),
            median_h_cm=float(samples.h.median()) if 'h' in samples else None,
            median_tof_cm=float(samples.tof.median()) if 'tof' in samples else None,
            max_gap_s=float(samples.elapsed_time.sort_values().diff().max()),
            conditions='user_confirmed_no_wind_same_set_height' if row.run_id.startswith('20260906') or row.run_id in ['20260513_180556','20260513_143201'] else 'not_confirmed_for_different_pair')
        if prepared is None:
            record['use']='excluded_incomplete_95_20';profiles.append(record);continue
        record['use']='primary_pooled_calibration' if record['current_pair'] else 'different_drone_sensitivity_only'
        grid,_=search([raw],min_width=MIN_WIDTH,min_seconds=MIN_SECONDS)
        for method,subset in [('free',grid),('upper_75_to_85',grid[grid.upper.between(75,85)]),
                              ('upper_fixed_80',grid[grid.upper.eq(UPPER_CONSTRAINT)])]:
            best=subset.iloc[0]
            individual.append(dict(battery_id=row.battery_id,drone_id=row.drone,run_id=row.run_id,
                method=method,upper=int(best.upper),lower=int(best.lower),rmse_pp=float(best.rmse),
                source=row.source,scope='raw_observed_95_20_samples'))
        prepared['grid']=grid
        traces.append(prepared);profiles.append(record)
    return traces,profiles,pd.DataFrame(individual)


def reference_update(model):
    curves=[DischargeCurve(tuple(m['boundaries_soc']),tuple(m['rates_pp_min'])) for m in model['batteries'].values()]
    edges,rates,times=mean_rate_curve(curves)
    curve,anchors,grid=fit_reference(edges,times)
    best=grid.iloc[0]
    model['reference']=dict(**curve.to_dict(),anchor_times_s=anchors.tolist(),
        approximation_rmse_pp=float(best.rmse_pp),approximation_max_error_pp=float(best.max_error_pp),
        mean_rate_knot_soc=edges.tolist(),mean_rate_pp_min=rates.tolist(),mean_rate_knot_times_s=times.tolist())
    return grid


def pooled_curve(traces):
    """Mean elapsed time at each SOC crossing, equal run weights, aligned at 95%."""
    levels=np.arange(95,19,-1,dtype=float)
    per_run=np.array([[t['crossing'][int(s)]-t['crossing'][95] for s in levels] for t in traces])
    times=per_run.mean(axis=0)
    curve,anchors,grid=fit_reference(levels,times)
    return curve,anchors,grid,levels,times


def model_error_on_trace(curve,trace):
    # Never flatten a below-floor prediction at 20% and include it as a fitted observation.
    elapsed=trace['t']-trace['t'][0]
    duration=curve.equivalent_seconds(95,20)
    mask=elapsed<=duration+1e-8
    predicted=np.array([curve.advance(95,t) for t in elapsed[mask]])
    crossings=np.array([trace['crossing'][s]-trace['crossing'][95] for s in range(95,19,-1)])
    predicted_crossings=np.array([curve.equivalent_seconds(95,s) for s in range(95,19,-1)])
    return dict(in_domain_soc_rmse_pp=float(np.sqrt(np.mean((trace['soc'][mask]-predicted)**2))),
        observed_samples=len(elapsed),evaluated_samples=int(mask.sum()),coverage_fraction=float(mask.mean()),
        crossing_time_rmse_s=float(np.sqrt(np.mean((crossings-predicted_crossings)**2))))


def plots(out,old,new,pooled,traces):
    # Chart contract: 2x2 static comparison; same axes per row, >400 raw points per
    # run; 2026-05-13 vs 2026-09-06. Blue/orange + neutral, line-style redundancy.
    # Left: original/current constrained fit. Right: actual runs/constructed mean.
    # No time rescaling. Inspect the exported PNG in its final standalone surface.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(15,10),sharex=True,sharey=True)
    for row,battery in enumerate(TARGETS):
        current=next(t for t in traces if t['battery_id']==battery and t['run_id']==old['batteries'][battery]['run_id'])
        ax=axes[row,0]
        ax.step(current['t']-current['t'][0],current['soc'],where='post',color='#9ca3af',lw=1.8,label='Observed: 6 Sep')
        for model,color,style,name in [(old,'#2563a6','--','Previous free fit'),(new,'#c15b20','-','Revised: first knot fixed at 80%')]:
            m=model['batteries'][battery];c=DischargeCurve(tuple(m['boundaries_soc']),tuple(m['rates_pp_min']))
            times=[c.equivalent_seconds(95,s) for s in c.boundaries]
            ax.plot(times,c.boundaries,style,color=color,lw=2.2,marker='o',label=name)
        ax.set_title(f'{battery}: recent-only fits (alternative, not selected)')
        before=old['batteries'][battery]['in_sample_rmse_pp'];after=new['batteries'][battery]['in_sample_rmse_pp']
        ax.text(.04,.06,f'Raw-sample RMSE: {before:.3f} -> {after:.3f} SOC pp',transform=ax.transAxes)
        ax=axes[row,1]
        relevant=sorted([t for t in traces if t['battery_id']==battery and t['drone']==TARGETS[battery]],key=lambda t:t['run_id'])
        for t,color,style in zip(relevant,['#2563a6','#808080'],['--','-']):
            ax.step(t['t']-t['t'][0],t['soc'],where='post',color=color,linestyle=style,lw=1.6,label='Observed '+t['run_id'][:8])
        c,anchor,_,levels,times=pooled_curve(relevant)
        ax.plot(times,levels,color='#2e2e2e',linestyle=':',lw=2,label='Mean crossing-time curve (derived)')
        ax.plot(anchor,c.boundaries,'-o',color='#c15b20',lw=2.2,label=f'Mean-curve fit: {c.boundaries[1]:g}% / {c.boundaries[2]:g}%')
        ax.set_title(f'{battery}: old/new mean calibration (selected candidate)')
    for ax in axes.flat:
        ax.set(xlim=(0,510),ylim=(18,98),xlabel='Elapsed time since first 95% SOC (s)',ylabel='SOC (%)')
        ax.tick_params(labelbottom=True);ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=9)
    fig.suptitle('B10 and B11: recent-only and repeated-hover mean fits',fontsize=19,y=.98)
    fig.text(.02,.925,'Recent fits: same raw 95%-20% observations. Right panels: 1 old + 1 recent run per battery, equal weights.',fontsize=12)
    fig.text(.02,.016,'User confirmed no wind / same set height. B11 logged median h: 80 cm (May) vs 70 cm (September); run differences remain visible.',fontsize=11)
    fig.tight_layout(rect=(0,.05,1,.91));fig.savefig(out/'B10_B11_comparison.png',dpi=170);fig.savefig(out/'B10_B11_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14.5,6.4),sharex=True,sharey=True)
    for ax,(battery,drone) in zip(axes,TARGETS.items()):
        selected=sorted([t for t in traces if t['battery_id']==battery and t['drone']==drone],key=lambda t:t['run_id'])
        for t,color,style in zip(selected,['#2563a6','#999999'],['--','-']):
            ax.step(t['t']-t['t'][0],t['soc'],where='post',color=color,linestyle=style,lw=1.5,label='Observed '+t['run_id'][:8])
        c,anchor,_,levels,times=pooled_curve(selected)
        ax.plot(times,levels,':',color='#333333',lw=1.8,label='Equal-run mean (derived)')
        ax.plot(anchor,c.boundaries,'-o',color='#c15b20',lw=2.5,label='Three-segment mean fit')
        for t,s in zip(anchor[1:-1],c.boundaries[1:-1]):
            ax.annotate(f'{s:g}%',(t,s),xytext=(7,10),textcoords='offset points',fontsize=12,color='#923e12')
        ax.set(title=f'{battery}: {c.boundaries[1]:g}% / {c.boundaries[2]:g}%',xlabel='Time since first 95% SOC (s)',ylabel='SOC (%)',
            xlim=(0,510),ylim=(18,98))
        ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=9)
    fig.suptitle('B10 / B11 repeated-hover mean calibration',fontsize=19,y=.99)
    fig.text(.03,.92,'Two no-wind runs per battery, same set height, May 13 and Sep 6, 2026. Each run has 50% weight.',fontsize=11)
    fig.text(.03,.02,'Orange is a constructed mean-fit, not a replacement for observed data. Run-to-run differences remain, especially for B11.',fontsize=11)
    fig.tight_layout(rect=(0,.05,1,.9));fig.savefig(out/'B10_B11_pooled.png',dpi=180);fig.savefig(out/'B10_B11_pooled.svg');plt.close(fig)
    ref=pooled['reference'];previous=old['reference']
    fig,ax=plt.subplots(figsize=(10.5,6.2))
    ax.plot(previous['anchor_times_s'],previous['boundaries_soc'],'--o',color='#2563a6',lw=2.2,label='Previous Bideal: 83% / 51%')
    ax.plot(ref['mean_rate_knot_times_s'],ref['mean_rate_knot_soc'],':',color='#777777',lw=2,label='Revised same-SOC mean-rate reference')
    ax.plot(ref['anchor_times_s'],ref['boundaries_soc'],'-o',color='#c15b20',lw=2.2,label=f"Revised Bideal: {ref['boundaries_soc'][1]:g}% / {ref['boundaries_soc'][2]:g}%")
    ax.set(title='Common Bideal before and after the B10/B11 revision',xlabel='Equivalent baseline-hover time since 95% (s)',ylabel='SOC (%)',ylim=(18,98))
    ax.grid(alpha=.2);ax.legend(fontsize=11)
    fig.text(.02,.02,'Only B10/B11 calibration coefficients changed. Both curves are constructed references, not measured energy.',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'Bideal_comparison.png',dpi=180);plt.close(fig)


def write_companion(out):
    # nbformat/nbclient unavailable in the task Python. Use a strictly JSON notebook
    # and run its ordinary Python cells top-to-bottom, capturing all outputs.
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None))
    md('## tl;dr\n用户确认旧/新记录均无风、相同设定高度。B10/B11各用两次同配对Hover等权平均得到自己的三段曲线；上分界没有固定80%。固定80%的近期单次方案另存为对照。')
    md('## Context & Methods\n### Key Assumptions\n95%→20%；每段至少10个百分点/20秒。新旧记录从95%对齐，在每个相同SOC取首次到达耗时的算术平均，再拟合三段。构造均值不是额外真实实验。\n这个核查本是模型的 companion。所有普通 Python 单元逐格运行；未验证 Jupyter 内核执行。')
    code(f"from pathlib import Path\nimport json, hashlib, sys\nimport pandas as pd\nroot=Path({str(ROOT)!r})\nsys.path.insert(0,str(root))\nfrom output_py.revise_b10_b11_calibration import review_sources, pooled_curve\nfrom output_py.search_bideal_boundaries import fit\nfolder=Path({str(out)!r})\nmodel=json.loads((folder/'model.json').read_text())\nalternative=json.loads((folder/'fixed_80_sensitivity_model.json').read_text())\ntraces, profiles, individual=review_sources()\nprint('Reviewed hover records:',len(profiles))")
    md('## Data\n实时重读原始记录及 SHA256；只把当前同一电池/无人机配对用于新旧平均对照。')
    code("print(pd.DataFrame(profiles)[['battery_id','drone_id','run_id','use']].to_string(index=False))")
    md('## Results\n重新计算对照与主候选，核对保存的两个电池系数；不覆盖模型。')
    code("for battery in ['B10','B11']:\n    item=alternative['batteries'][battery]\n    trace=next(t for t in traces if t['battery_id']==battery and t['run_id']==item['run_id'])\n    result=fit(trace,80,int(item['boundaries_soc'][2]),20)\n    assert abs(result['rmse']-item['in_sample_rmse_pp'])<1e-10\n    print('Alternative',battery,item['boundaries_soc'],'RMSE',result['rmse'])")
    code("for battery,drone in [('B10','drone_2'),('B11','drone_1')]:\n    selected=[t for t in traces if t['battery_id']==battery and t['drone']==drone]\n    curve, _, grid, _, _=pooled_curve(selected)\n    assert list(curve.boundaries)==model['batteries'][battery]['boundaries_soc']\n    assert all(abs(a-b)<1e-10 for a,b in zip(curve.rates_pp_min,model['batteries'][battery]['rates_pp_min']))\n    print('Primary old/new mean:',battery,curve.boundaries,'approximation RMSE',float(grid.iloc[0].rmse_pp))")
    md('## Takeaways\n主候选用重复实验的平均特性，不是只追求最新一次最低误差。B10为85/56、B11为82/51；B11新旧差异仍较大，不能宣称校准可靠性已获独立验证。原始数据、上一版及飞行逻辑未改。')
    namespace={};counter=0
    for cell in cells:
        if cell['cell_type']!='code':continue
        counter+=1;capture=io.StringIO()
        with contextlib.redirect_stdout(capture):exec(compile(''.join(cell['source']),'audit.ipynb','exec'),namespace)
        cell['execution_count']=counter
        cell['outputs']=[dict(output_type='stream',name='stdout',text=capture.getvalue().splitlines(True))]
    notebook=dict(nbformat=4,nbformat_minor=4,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        execution_validation='ordinary Python cells executed sequentially; Jupyter kernel not checked'),cells=cells)
    path=out/'audit.ipynb';path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')
    parsed=json.loads(path.read_text());assert parsed['nbformat']==4 and len(parsed['cells'])==len(cells)
    return counter


def build(out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Use a new directory; no prior model overwrite')
    old=json.loads((PREVIOUS/'model.json').read_text());old_hash=sha(PREVIOUS/'model.json')
    traces,profiles,individual=review_sources()
    new=copy.deepcopy(old);new['version']='recent_fixed80_sensitivity_20260909_not_selected'
    new['previous_model_sha256']=old_hash
    new['revision']=dict(primary='Latest raw data, first boundary explicitly fixed at 80 for B10/B11 only',
        reason='User requested about-80% stage comparability; this is a model constraint, not the unconstrained optimum',
        historical_role='not used in this recent-only alternative; user confirmed old runs no-wind and same set height',
        untouched_batteries=['B12','B13','B14'])
    new['fitting']['upper_soc_constraints']={'B10':80,'B11':80}
    comparison=[];current_traces={}
    for battery in TARGETS:
        trace=next(t for t in traces if t['battery_id']==battery and t['run_id']==old['batteries'][battery]['run_id'])
        current_traces[battery]=trace
        best=trace['grid'][trace['grid'].upper.eq(UPPER_CONSTRAINT)].iloc[0]
        result=fit(trace,UPPER_CONSTRAINT,int(best.lower),MIN_SECONDS)
        c=DischargeCurve((95,UPPER_CONSTRAINT,int(best.lower),20),tuple(result['rates']))
        new['batteries'][battery].update(c.to_dict(),in_sample_rmse_pp=result['rmse'],
            max_in_sample_error_pp=result['max_error'],anchor_elapsed_times_s=result['anchor_times'].tolist(),
            first_boundary_source='explicit_80_percent_model_constraint')
        before=old['batteries'][battery]['in_sample_rmse_pp']
        comparison.append(dict(battery_id=battery,old_upper=old['batteries'][battery]['boundaries_soc'][1],
            old_lower=old['batteries'][battery]['boundaries_soc'][2],new_upper=UPPER_CONSTRAINT,new_lower=int(best.lower),
            old_raw_rmse_pp=before,new_raw_rmse_pp=result['rmse'],rmse_increase_pp=result['rmse']-before,
            rmse_increase_fraction=result['rmse']/before-1))
    ref_grid=reference_update(new)
    pooled=copy.deepcopy(old);pooled['version']='individual_three_stage_pooled_candidate_v2_20260909'
    pooled['previous_model_sha256']=old_hash
    pooled['revision']=dict(primary=True,method='Equal-run mean first-crossing elapsed time at each SOC, aligned at 95; then unconstrained-integer-knot three-line approximation',
        historical_conditions='user confirmed 2026-09-09: no wind, same set hover height',
        logged_height_caveat='B11 median h was 80 cm vs 70 cm; does not override user confirmation about settings',
        not_held_out_validation=True,upper_80_constraint_applied=False,untouched_batteries=['B12','B13','B14'])
    pooled['fitting']['individual_by_battery']={'B10':'equal-run mean crossing-time curve; exact time-integral approximation RMSE',
        'B11':'equal-run mean crossing-time curve; exact time-integral approximation RMSE',
        'B12':'unchanged v1 raw-sample RMSE','B13':'unchanged v1 raw-sample RMSE','B14':'unchanged v1 raw-sample RMSE'}
    pooled['fitting']['individual']='B10/B11: equal-run mean-curve approximation; B12/B13/B14: unchanged raw-sample endpoint fits; see individual_by_battery'
    pooled_rows=[];pooled_curves=[]
    for battery,drone in TARGETS.items():
        selected=[t for t in traces if t['battery_id']==battery and t['drone']==drone]
        c,anchors,grid,levels,times=pooled_curve(selected)
        pooled['batteries'][battery]=dict(drone_id=drone,**c.to_dict(),
            sources=[{k:t[k] for k in ['run_id','source','sha256']} for t in selected],run_weights=[1/len(selected)]*len(selected),
            constructed_mean_curve_rmse_pp=float(grid.iloc[0].rmse_pp),anchor_times_s=anchors.tolist())
        for s,t in zip(levels,times):pooled_curves.append(dict(battery_id=battery,soc=s,mean_elapsed_s=t,kind='derived_mean_not_observed',run_count=len(selected)))
        for trace in selected:
            pooled_rows.append(dict(battery_id=battery,run_id=trace['run_id'],method='pooled_mean_primary',**model_error_on_trace(c,trace)))
    reference_update(pooled)
    for battery in ['B12','B13','B14']:
        assert pooled['batteries'][battery]==old['batteries'][battery]
    out.mkdir(parents=True)
    for name,model in [('model',pooled),('fixed_80_sensitivity_model',new)]:
        (out/(name+'.json')).write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n')
        BatteryNormalizer.load(out/(name+'.json'))
    pd.DataFrame(profiles).to_csv(out/'source_audit.csv',index=False)
    individual.to_csv(out/'individual_run_comparison.csv',index=False)
    pd.DataFrame(comparison).to_csv(out/'revision_comparison.csv',index=False)
    pd.DataFrame(pooled_rows).to_csv(out/'pooled_vs_individual_observations.csv',index=False)
    pd.DataFrame(pooled_curves).to_csv(out/'pooled_crossing_times_derived.csv',index=False)
    reference_update(pooled).to_csv(out/'Bideal_boundary_search.csv',index=False)
    segment_rows=[]
    for name,m in list(pooled['batteries'].items())+[('Bideal',pooled['reference'])]:
        for k,(hi,lo,r) in enumerate(zip(m['boundaries_soc'],m['boundaries_soc'][1:],m['rates_pp_min']),1):
            segment_rows.append(dict(battery_id=name,segment=k,upper_soc=hi,lower_soc=lo,rate_pp_min=r))
    pd.DataFrame(segment_rows).to_csv(out/'segments.csv',index=False)
    normalize_csv(out/'model.json',PREVIOUS/'normalization_example_input.csv',out/'normalization_example_output.csv',70)
    plots(out,old,new,pooled,traces)
    cells=write_companion(out)
    for record in profiles:
        assert sha(ROOT/record['source'])==record['sha256']
    assert sha(PREVIOUS/'model.json')==old_hash
    checks=dict(old_model_unchanged=True,source_files_hash_checked=len(profiles),B12_B13_B14_unchanged=True,
        normalization_csv_executed=True,audit_python_cells_executed=cells,Jupyter_kernel_execution_verified=False,
        all_outputs_candidate_only=True,model_sha256=sha(out/'model.json'),historical_records_used_in_primary_fit=True,
        historical_conditions_user_confirmed=True,B10_B11_independent_holdout_runs=0)
    (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    lines=['# B10 / B11 分界调整（候选 v2）','',
        '用户于2026-09-09确认5月13日与9月6日的B10/D2、B11/D1 Hover均无风、相同设定高度。主候选将每块电池的两次同配对记录等权平均，再拟合三段；没有强制第一处分界为80%。',
        'B12/B13/B14 的曲线逐字段保持原样；Bideal 按相同 SOC 五电池等权平均率重新生成。未改原始记录、旧模型、正式耗电表或飞行代码。','',
        '| 电池 | 旧分界（单次） | 新分界（两次平均） | 对均值曲线的近似 RMSE（SOC 百分点） |',
        '|---|---|---|---|']
    for r in comparison:
        m=pooled['batteries'][r['battery_id']]
        lines.append(f"| {r['battery_id']} | {r['old_upper']:g}/{r['old_lower']:g} | {m['boundaries_soc'][1]:g}/{m['boundaries_soc'][2]:g} | {m['constructed_mean_curve_rmse_pp']:.3f} |")
    ref=pooled['reference']
    lines += ['',f"新 Bideal：{' → '.join(f'{v:g}%' for v in ref['boundaries_soc'])}；耗电率 {' / '.join(f'{v:.4f}' for v in ref['rates_pp_min'])} SOC 百分点/分钟。",'',
        '## 历史数据提供的依据与限制','',
        '- 5月13日同一编号配对的单次自由拟合：B10/D2 为85/54，B11/D1 为80/51。因此约80%附近的上分界有历史记录依据，但不证明跨日期不变。',
        '- B10 从95%到80%：5月约34.17秒，9月约127.59秒；B11：5月约30.12秒，9月约120.54秒。高电量段的批次差异明显。',
        '- 用户确认设定高度相同。B11 的历史 h 中位数80cm、9月70cm；日志读数有差异，但不能据此否认相同设定，也不能仅据时期不同的IP判定机体更换。',
        '- 对每块电池各一次旧记录和新记录，在相同SOC求首次经过时间的等权平均、从95%对齐，再自由拟合三段。结果：B10为85/56，B11为82/51。',
        '- 上述均值的拟合误差仅代表对构造均值曲线的近似，不能和单次真实记录的RMSE混为一谈。pooled_vs_individual_observations.csv 显示它对各次真实记录的偏差；超出支持时间范围的样本不做平底夹断，并报告覆盖率。',
        '- 新旧记录均已参与主候选校准，不能再称为独立验证。B10对旧/新实测SOC的域内RMSE约2.275/3.493，B11约5.443/6.713；覆盖率分别100%/98.6%、100%/92.7%。新旧波动较大，平均曲线并不会更贴合每一次实验。',
        '- 另保存仅用最新数据、第一处分界预设80%的对照方案：B10/B11均为80/56；对应最新实测RMSE分别1.268与1.469。该对照没有选为主候选。',
        '- B10在D3上的5月6日数据另外检查，其自由拟合85/58，但不混入当前B10/D2配对。8月D1仅14秒、不覆盖95→20，排除。','',
        '## 使用与验证','',
        '- 主参数（两次平均）：model.json；近期单次固定80%的对照：fixed_80_sensitivity_model.json。两者均兼容已有 battery_normalization.py 接口。',
        '- 当前范围仍为95%–20%；没有把阶段速度强行设成其他电池的速度，也没有改写SOC实测值。',
        '- 约25秒的相同真实样本已用新模型换算，新增列单独保存；这些校准样本不是独立验证。',
        '- source_audit.csv保留全部11条B10/B11候选记录的使用/排除原因、配对、覆盖、日志高度与源哈希。主候选总共使用4条B10/B11记录；每块电池2次、各50%。',
        '- 原始源哈希、旧模型哈希、B12–B14不变性已检查。',
        '- audit.ipynb 的普通Python单元已从头逐格运行并记录输出；本机缺nbformat/nbclient，未验证Jupyter内核执行。可在含这些组件的环境运行：`python3 -m jupyter nbconvert --execute --to notebook --inplace audit.ipynb`。','',
        '## 复现','',
        '`python3 output_py/revise_b10_b11_calibration.py --output 一个不存在的新目录`','',
        '模型约束：整数SOC边界、每段至少10个百分点和20秒；原始曲线按实际采样点计算对照误差，端点锚定。主候选对均值曲线的拟合使用精确时间积分误差。B10上分界85%恰好在最小10个百分点段宽的搜索上边界，不能将85%宣称为已经验证的物理转折点。','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(output=str(out),fixed_80_comparison=comparison,Bideal=pooled['reference'],pooled_observation_checks=pooled_rows,validation=checks),ensure_ascii=False,indent=2))
    return pooled


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    build(parser.parse_args().output)
