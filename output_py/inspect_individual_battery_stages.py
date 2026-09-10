"""Inspect individual knots; no active calibration or source-data changes.

Chart contract: native grouped bars of sample-in RMSE, five batteries x
three explicitly labelled calibration methods, blue/gold/orange palette.
Exact knot/duration lookup is a separate table. Methodology-first report:
summary, definitions, evidence, correction specification, uncertainty,
next steps/open questions. Reuses the original endpoint-fit support module.
"""
import json
import sqlite3
import numpy as np
import pandas as pd
from search_bideal_boundaries import ROOT, PREVIOUS, OUT, PAIRS, load_trace, search, fit

DEST = OUT / 'individual_stages'
RESIDUAL_SQL = '''SELECT battery, method, upper, lower, run_id, source,
COUNT(*) AS samples,
sqrt(AVG((observed_soc-modeled_soc)*(observed_soc-modeled_soc))) AS rmse_pp
FROM individual_fit_residuals
GROUP BY battery, method, upper, lower, run_id, source
ORDER BY battery, method'''


def equivalent_hover_minutes(start, end, levels, rates):
    """Baseline-equivalent SOC loss, not Wh and not an absolute-SOC mapping."""
    assert levels[0] >= start >= end >= levels[-1]
    return sum(max(0., min(start, a) - max(end, b)) / r
               for a, b, r in zip(levels[:-1], levels[1:], rates))


def main():
    sources = pd.read_csv(PREVIOUS/'coefficients.csv').drop_duplicates(['battery_id','run_id']).sort_values('battery_id')
    assert len(sources) == 5 and not sources.run_id.astype(str).eq('20260906_165812').any()
    summaries, bars, sensitive, source_records, residuals = [], [], [], [], []
    for row in sources.itertuples():
        assert PAIRS[row.battery_id] == row.drone
        raw = load_trace(row)
        grid, prepared = search([raw])
        p = prepared[0]; best = grid.iloc[0]
        u, l = int(best.upper), int(best.lower)
        f = fit(p, u, l)
        levels = [95, u, l, 20]
        # Independent segment formula verifies the scored predictions.
        pred = np.zeros_like(p['t'])
        for k in range(3):
            a, b = f['anchor_times'][k:k+2]
            mask = (p['t'] >= a) & (p['t'] <= b)
            pred[mask] = levels[k] - (p['t'][mask]-a)/(b-a)*(levels[k]-levels[k+1])
        np.testing.assert_allclose(pred, f['pred'], atol=1e-10)
        minutes = equivalent_hover_minutes(95,20,levels,f['rates'])
        assert abs(minutes-(p['t'][-1]-p['t'][0])/60) < 1e-10
        assert abs(minutes-equivalent_hover_minutes(95,70,levels,f['rates'])-
                   equivalent_hover_minutes(70,20,levels,f['rates'])) < 1e-10
        near = grid[grid.rmse <= best.rmse*1.02]
        info = dict(battery=row.battery_id, drone=row.drone, run_id=row.run_id,
            upper=u, lower=l, own_rmse=f['rmse'], samples=len(p['t']),
            high_seconds=float(f['durations'][0]), middle_seconds=float(f['durations'][1]),
            low_seconds=float(f['durations'][2]), rates_pp_min=f['rates'].tolist(),
            near2_upper=[int(near.upper.min()),int(near.upper.max())],
            near2_lower=[int(near.lower.min()),int(near.lower.max())], source=row.source)
        summaries.append(info)
        source_records.append(dict(battery=row.battery_id,run_id=row.run_id,source=row.source,sha256=row.sha256))
        for label, a, b in [('共同75/40',75,40),('共同80/56',80,56),('各自分界',u,l)]:
            ff = fit(p,a,b)
            bars.append(dict(battery=row.battery_id, method=label, upper=a, lower=b,
                rmse_pp=ff['rmse'], samples=len(p['t']),run_id=row.run_id,source=row.source))
            residuals.extend(dict(battery=row.battery_id,method=label,upper=a,lower=b,
                run_id=row.run_id,source=row.source,elapsed_time=float(t),
                observed_soc=float(s),modeled_soc=float(predicted))
                for t,s,predicted in zip(p['t'],p['soc'],ff['pred']))
        for width, seconds in [(5,20),(10,10),(10,20)]:
            g,_ = search([raw],min_width=width,min_seconds=seconds); bb=g.iloc[0]
            sensitive.append(dict(battery=row.battery_id,min_width=width,min_seconds=seconds,
                upper=int(bb.upper),lower=int(bb.lower),rmse_pp=float(bb.rmse)))

    DEST.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(':memory:') as db:
        db.create_function('sqrt',1,np.sqrt)
        pd.DataFrame(residuals).to_sql('individual_fit_residuals',db,index=False)
        sql_bars=pd.read_sql_query(RESIDUAL_SQL,db)
    python_bars=pd.DataFrame(bars).sort_values(['battery','method'])
    np.testing.assert_allclose(sql_bars.rmse_pp,python_bars.rmse_pp,atol=1e-10)
    bars=sql_bars.to_dict('records')
    pd.DataFrame(residuals).to_csv(DEST/'fit_residuals.csv',index=False)
    pd.DataFrame(summaries).to_csv(DEST/'individual_candidates.csv',index=False)
    pd.DataFrame(bars).to_csv(DEST/'rmse_comparison.csv',index=False)
    pd.DataFrame(sensitive).to_csv(DEST/'sensitivity.csv',index=False)
    (DEST/'sources.json').write_text(json.dumps(source_records,ensure_ascii=False,indent=2))
    technical = [
        ('summary','每块电池可以有自己的分界线，标准化不要求边界相同',
         '共同80%/56%是共用边界的折中，不是每块电池自身的最佳边界。可以为每块电池拟合独立三段基准函数，再将实验耗电与该电池在相同SOC下的单机Hover基准相比，最后统一换算到指定的Bideal参考状态。本文提出候选和方法，不改变当前Bideal v1或真实实验数据。'),
        ('scope','本次只比较同一批五次无风Hover的95%→20%区间',
         '采用2026年9月6日无风、相同设定高度的B10/D2、B11/D1、B12/D5、B13/D3、B14/D4各一次实验；排除B12/D2的20260906_165812。分界线以1个SOC百分点为步长搜索，各段至少覆盖5个百分点且持续10秒。用真实SOC首次经过95%、两个候选边界、20%的时刻连接连续三段直线，计算全部真实样本的RMSE（SOC百分点）。这些是工程近似区间，不是已经识别的电化学阶段。'),
        ('finding','个体最优分界不同，低误差不等于边界已经稳定',
         '当前候选：B10为63/43，B11为67/47，B12为90/51，B13为81/51，B14为88/56。图中比较共同75/40、共同80/56和各自分界的样本内RMSE；柱越低表示本次曲线近似更贴近实测。各自寻优更灵活，因此不能凭样本内误差下降证明跨实验标准化更可靠。B12和B14首段很短，需要特别谨慎。'),
        ('method','先查各自基准，再计算相对Hover耗电倍数',
         '定义b_i(S)为电池i在真实SOC为S时的无风Hover基准耗电率；由该电池自己的三段直线斜率得到，单位为百分点/分钟。对不跨界的实验片段，g_i=r_experiment/b_i(S)。g=1表示接近该电池自身的Hover基准，g=1.2表示该片段的SOC下降速度是基准的1.2倍，不等价于已测得功率增加20%。\n\n阶段名称无需对齐。例如SOC=70%时，按当前候选B10位于第一段而B13位于第二段；各自查询相应b_i(70%)即可。不能直接把不同SOC范围的“第一段斜率”机械求平均作为同一阶段的Bideal。'),
        ('mapping','换算到Bideal时要指定共同参考SOC，而不是把原始SOC改名',
         '用于同条件比较时，先约定共同的Bideal参考SOC q。短片段的等效耗电率为r_equivalent(q)=g_i×b_ideal(q)，其中q是统一参考状态，不必等于电池i的原始读数S。Bideal本身仍可保持三段连续参考曲线，原始电池的两个分界点不必与它相同。\n\n对跨越个体分界点的片段，先计算基准等效悬停时长H=Σ(各段SOC下降量/该段b_i)，再除以实际片段分钟数得到平均g。需要生成有限时长Bideal轨迹时，从指定参考起点沿Bideal曲线推进H分钟，跨Bideal自身分界时也分段处理。这样不会将有限下降区间错误地只乘起点处的一个系数。真实SOC仍单独保留，不能被此等效量替代。'),
        ('uncertainty','短首段与单次记录限制了边界的可信度',
         'B12的95%→90%只约11秒，B14的95%→88%约14秒。最优上边界可能受最小段宽或起始短时变化影响。增加最小段宽/时长的结果另列于敏感性表，不将其中任一设置自动激活。当前每个配对仅一次最新实验；仍需独立重复的同条件Hover检查边界和斜率能否复现，不能按秒随机拆分同一次飞行来代替独立验证。标准化还假设实验负载对基准SOC耗电率的影响近似可用倍数表达；若这个倍数随SOC显著变化，应保留SOC相关性。'),
        ('next','下一步先验证个体校准，再统一Bideal输出',
         '保留每块电池独立候选边界与斜率，同时保留当前共用模型作为对照；在独立同条件记录上比较两者校正后的偏差和跨电池离散程度。如果单机Hover可以被校正一致、风洞同条件下仍随SOC产生系统差异，就需要SOC相关的条件模型，不能把真实的负载效应全部归因于电池。'),
        ('questions','仍待确认的是可重复性与风洞迁移，而不是阶段名称',
         '个体边界能否在重复Hover中保持接近？变化主要出现在起始短时段还是整段？相对Hover耗电倍数在不同风力、相同position下是否可迁移？在默认无人机相同的前提下可将配对基准主要解释为电池校准，但这些数据并未独立识别纯电池与残余机体效应。'),
    ]
    source = dict(id='individual_analysis',label='五个无风Hover的个体分界搜索',
        path=str(DEST/'individual_candidates.csv'),
        query=dict(engine='SQLite + Python / NumPy calibration',language='sql',sql=RESIDUAL_SQL,
            description='SQL从全部真实样本与模型残差重新计算图中的RMSE；分界搜索及段长计算入口：output_py/inspect_individual_battery_stages.py，复用search_bideal_boundaries.py。原始CSV及SHA256见sources.json。',
            filters=['2026-09-06 current five battery/drone pairs','95→20 observed hover','B12 D2 excluded'],
            metric_definitions=['RMSE over all real samples; per-flight endpoint calibration, not out-of-sample validation.']))
    title='个体电池分界线与Bideal标准化'
    blocks=[dict(id='title',type='markdown',body='# '+title)]
    for key,heading,body in technical:
        blocks.append(dict(id=key,type='markdown',body='## '+heading+'\n\n'+body))
        if key=='finding':
            blocks.extend([dict(id='rmse_block',type='chart',chartId='rmse'),dict(id='knots_block',type='table',tableId='knots')])
        if key=='uncertainty':blocks.append(dict(id='sensitivity_block',type='table',tableId='sensitivity'))
    chart=dict(id='rmse',type='bar',title='五块电池在三种分界方案下的RMSE',dataset='comparison',source=source,
        encodings=dict(x=dict(field='battery',label='电池'),y=dict(field='rmse_pp',label='RMSE（SOC百分点）'),
                       color=dict(field='method',label='分界方案')),options=dict(grouping='grouped'),
        palette=dict(kind='categorical',colors=['#3B6FB6','#D8A739','#D66B32']))
    tables=[dict(id='knots',title='每块电池的个体候选分界与段长',dataset='knots',source=source,
        columns=[dict(field=k,label=v) for k,v in [('battery','电池'),('upper','上分界SOC%'),('lower','下分界SOC%'),
        ('high_seconds','首段秒数'),('middle_seconds','中段秒数'),('low_seconds','末段秒数'),('own_rmse','RMSE（百分点）')]],
        defaultSort=dict(field='battery',direction='asc')),
        dict(id='sensitivity',title='最小段宽与时长约束的敏感性',dataset='sensitivity',source=source,
        columns=[dict(field=k,label=v) for k,v in [('battery','电池'),('min_width','最小SOC跨度'),('min_seconds','最小时长秒'),('upper','上分界'),('lower','下分界'),('rmse_pp','RMSE')]],
        defaultSort=dict(field='battery',direction='asc'))]
    flat_summaries = [{k: (' / '.join(str(x) for x in v) if isinstance(v,list) else v)
                       for k,v in item.items()} for item in summaries]
    payload=dict(surface='report',manifest=dict(version=1,surface='report',title=title,blocks=blocks,
        sources=[source],charts=[chart],tables=tables),snapshot=dict(version=1,status='ready',
        datasets=dict(comparison=bars,knots=flat_summaries,sensitivity=sensitive)),sources=[source])
    (DEST/'artifact.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    md='# '+title+'\n\n'+'\n\n'.join('## '+h+'\n\n'+b for _,h,b in technical)
    (DEST/'README.md').write_text(md)
    (DEST/'validation.json').write_text(json.dumps(dict(status='passed',source_hashes=5,
        independent_prediction_checks=5,normalization_additivity_checks=5,baseline_equivalent_duration_checks=5,
        active_model_changed=False),indent=2))
    print(pd.DataFrame(summaries).drop(columns=['source','rates_pp_min']).to_string(index=False))
    print(pd.DataFrame(sensitive).to_string(index=False))


if __name__=='__main__':main()
