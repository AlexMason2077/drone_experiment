"""Build an opt-in individual-battery / common-reference calibration from raw hover.

This creates a NEW artifact directory only. Never changes raw data, old models,
the registry, flight code, or the active discharge-rate table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battery_normalization import BatteryNormalizer, DischargeCurve, normalize_csv
from output_py.search_bideal_boundaries import PAIRS, load_trace, prepare, search, fit

DEFAULT_OUT = ROOT / 'analysis_results/battery_normalization_candidate_20260908'
SOURCE_DIR = ROOT / 'analysis_results/bideal_20260908'
RUNS = {'B10': '20260906_173013', 'B11': '20260906_164725',
        'B12': '20260906_182131', 'B13': '20260906_170951', 'B14': '20260906_171938'}
EXCLUDED_RUN = '20260906_165812'  # B12 on drone_2: explicitly excluded by user.
TOP, BOTTOM, MIN_WIDTH, MIN_SECONDS = 95, 20, 10, 20
VERSION = 'individual_three_stage_candidate_v1_20260908'


def mean_rate_curve(curves):
    """Arithmetic mean at the SAME SOC; each battery has exactly 1/N weight."""
    endpoints = {(c.boundaries[0], c.boundaries[-1]) for c in curves}
    if len(endpoints) != 1:
        raise ValueError('Curves must share a calibrated domain')
    edges = np.array(sorted({s for c in curves for s in c.boundaries}, reverse=True))
    rates = np.array([np.mean([c.rate_at((a+b)/2) for c in curves])
                      for a, b in zip(edges[:-1], edges[1:])])
    times = np.r_[0., np.cumsum((edges[:-1]-edges[1:])*60/rates)]
    return edges, rates, times


def exact_line_rmse(reference_times, reference_soc, fit_times, fit_soc):
    """Exact time-integral RMSE between two continuous piecewise-linear curves."""
    if not (reference_times[0] == fit_times[0] and reference_times[-1] == fit_times[-1]):
        raise ValueError('Time domains must match')
    knots = np.unique(np.r_[reference_times, fit_times])
    error = np.interp(knots, reference_times, reference_soc) - np.interp(knots, fit_times, fit_soc)
    integral = np.sum(np.diff(knots)*(error[:-1]**2 + error[:-1]*error[1:] + error[1:]**2)/3)
    return float(np.sqrt(integral/(knots[-1]-knots[0]))), float(np.max(np.abs(error)))


def fit_reference(edges, times):
    """Endpoint-anchored three-line approximation to the mean-rate reference."""
    rows = []
    for upper in range(BOTTOM+2*MIN_WIDTH, TOP-MIN_WIDTH+1):
        for lower in range(BOTTOM+MIN_WIDTH, upper-MIN_WIDTH+1):
            levels = np.array([TOP, upper, lower, BOTTOM], dtype=float)
            anchors = np.interp(levels, edges[::-1], times[::-1])
            if np.min(np.diff(anchors)) < MIN_SECONDS:
                continue
            rmse, max_error = exact_line_rmse(times, edges, anchors, levels)
            rows.append(dict(upper=upper, lower=lower, rmse_pp=rmse,
                             max_error_pp=max_error, min_duration_s=float(np.min(np.diff(anchors)))))
    grid = pd.DataFrame(rows).sort_values(['rmse_pp', 'upper', 'lower']).reset_index(drop=True)
    best = grid.iloc[0]
    levels = np.array([TOP, best.upper, best.lower, BOTTOM], dtype=float)
    anchors = np.interp(levels, edges[::-1], times[::-1])
    curve = DischargeCurve(tuple(levels), tuple((levels[:-1]-levels[1:])*60/np.diff(anchors)))
    return curve, anchors, grid


def historical_check(normalizer, inventory, out):
    """Use frozen current coefficients. No endpoint/slope recalibration on old runs."""
    rows, audits = [], []
    for row in inventory.itertuples():
        if row.status != 'observed_hover' or str(row.run_id).startswith('20260906'):
            continue
        if PAIRS.get(row.battery_id) != row.drone or row.run_id == EXCLUDED_RUN:
            continue
        raw = load_trace(row)
        trace = prepare(raw) if raw is not None else None
        audits.append(dict(battery_id=row.battery_id, drone_id=row.drone, run_id=row.run_id,
                           source=row.source, sha256=row.sha256,
                           status='included_complete_95_20' if trace is not None else 'insufficient_contiguous_95_20'))
        if trace is None:
            continue
        for band, start, end in [('high95_75',95,75), ('medium75_40',75,40),
                                 ('low40_20',40,20), ('whole95_20',95,20)]:
            duration = trace['crossing'][end] - trace['crossing'][start]
            rows.append(dict(battery_id=row.battery_id, drone_id=row.drone, run_id=row.run_id,
                band=band, soc_start=start, soc_end=end, observed_duration_s=duration,
                observed_rate_pp_min=(start-end)*60/duration,
                frozen_model_equivalent_s=normalizer.curve_for(row.battery_id,row.drone).equivalent_seconds(start,end),
                relative_hover_factor=normalizer.relative_drain(row.battery_id,row.drone,start,end,duration),
                source=row.source, sha256=row.sha256, coefficients_refitted=False,
                historical_wind_and_height_confirmed=False))
    df = pd.DataFrame(rows)
    df.to_csv(out/'historical_validation.csv', index=False)
    pd.DataFrame(audits).to_csv(out/'historical_source_audit.csv', index=False)
    summary = []
    for band, group in df.groupby('band'):
        # Equal battery weighting even if several historical runs exist for one battery.
        by_battery = group.groupby('battery_id')[['observed_rate_pp_min','relative_hover_factor']].mean()
        raw_cv = by_battery.observed_rate_pp_min.std(ddof=0)/by_battery.observed_rate_pp_min.mean()
        norm_cv = by_battery.relative_hover_factor.std(ddof=0)/by_battery.relative_hover_factor.mean()
        summary.append(dict(band=band, battery_count=len(by_battery), run_count=len(group),
            observed_rate_cv=float(raw_cv), normalized_factor_cv=float(norm_cv),
            cv_improvement_fraction=float(1-norm_cv/raw_cv)))
    pd.DataFrame(summary).to_csv(out/'historical_summary.csv', index=False)
    return summary, audits


def make_figures(out, prepared, fitted, reference, mean_edges, mean_times, reference_times):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,3,figsize=(17,9),sharex=True,sharey=True)
    for ax, battery in zip(axes.flat, sorted(prepared)):
        trace, result = prepared[battery], fitted[battery]
        ax.step(trace['t']-trace['t'][0], trace['soc'], where='post', color='#64748b', lw=1.5, label='Observed baseline')
        ax.plot(result['anchor_times']-result['anchor_times'][0], [TOP,*result['boundaries'],BOTTOM],
                '-o', color='#2563a6', lw=2.5, markersize=6, label='Own three-segment fit')
        ax.set_title(f"{battery} / {PAIRS[battery]} | knots {result['boundaries'][0]}%, {result['boundaries'][1]}%")
        ax.text(.04,.08,f"Fit RMSE: {result['rmse']:.2f} SOC pp",transform=ax.transAxes,fontsize=11)
    ax = axes.flat[-1]
    ax.plot(mean_times,mean_edges,color='#64748b',lw=2,label='Mean-rate reference (derived)')
    ax.plot(reference_times,reference.boundaries,'-o',color='#c15b20',lw=2.5,label='Three-segment Bideal')
    ax.set_title(f'Bideal | knots {reference.boundaries[1]:g}%, {reference.boundaries[2]:g}%')
    for ax in axes.flat:
        ax.set(xlim=(0,520),ylim=(18,98),xlabel='Time since 95% SOC (s)',ylabel='SOC (%)')
        ax.grid(alpha=.2); ax.legend(fontsize=9,loc='upper right')
    fig.suptitle('Five individual baseline curves and one common Bideal',fontsize=20)
    fig.text(.02,.012,'Fit data: 6 Sep 2026, no wind. B12/drone_2 excluded. Calibrated range 95%-20% only; candidate, not active.',fontsize=12)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(out/'six_curves.png',dpi=170); fig.savefig(out/'six_curves.svg'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11,6.4))
    ax.plot(mean_times,mean_edges,color='#64748b',lw=3,label='Same-SOC mean-rate reference (derived)')
    ax.plot(reference_times,reference.boundaries,'-o',color='#c15b20',lw=2.5,markersize=8,label='Three-segment Bideal')
    for t, soc in zip(reference_times[1:-1],reference.boundaries[1:-1]):
        ax.annotate(f'{soc:g}% at {t:.1f} s',(t,soc),xytext=(15,15),textcoords='offset points',fontsize=13)
    rates = ' / '.join(f'{r:.3f}' for r in reference.rates_pp_min)
    ax.set(title=f'Bideal reference | discharge rates: {rates} SOC pp/min',xlabel='Equivalent baseline-hover time from 95% (s)',ylabel='SOC (%)')
    ax.grid(alpha=.2); ax.legend(); ax.set_ylim(18,98)
    fig.text(.02,.015,'This is a constructed reference, not an additional physical battery measurement. No extrapolation outside 95%-20%.',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(out/'Bideal.png',dpi=180); fig.savefig(out/'Bideal.svg'); plt.close(fig)


def write_readme(out, model, historical):
    ref = model['reference']
    lines = ['# 五块电池独立三段曲线 + 三段 Bideal（候选版）','',
        '状态：离线候选校准工具已建立；未替换旧版耗电率表、未接入训练或飞行控制，未改动原始数据。','',
        '## 数据与拟合','',
        '- 使用 2026-09-06 五条真实单机悬停 baseline。用户确认均无风、相同设定高度。',
        '- 配对：B10/D2、B11/D1、B12/D5、B13/D3、B14/D4。B12/D2 的 20260906_165812 不参与。',
        '- 共同校准范围为 95%→20%；当前候选不外推到 100% 或低于 20%。',
        '- 每块电池独立搜索整数 SOC 分界点。每段至少 10 个 SOC 百分点、20 秒；这是防止过短段的建模约束，不是电池物理定律。',
        '- 线段连接真实首次达到 95%、两处分界点、20% 的采样点；用范围内原始采样 SOC 的 RMSE 选择分界。不是无约束直线回归。',
        '- 三段保证连续、下降；分界点是本次数据的经验近似，不等同于已验证的电化学阶段。','',
        '| 电池 | 三段 SOC 范围 | 三段耗电率（SOC 百分点/分钟） | 拟合 RMSE（SOC 百分点） |',
        '|---|---|---|---|']
    for battery,item in model['batteries'].items():
        bounds=' → '.join(f'{v:g}%' for v in item['boundaries_soc'])
        rates=' / '.join(f'{v:.3f}' for v in item['rates_pp_min'])
        lines.append(f"| {battery} | {bounds} | {rates} | {item['in_sample_rmse_pp']:.3f} |")
    lines += ['', '## Bideal 的定义','',
        '1. 在相同 SOC 下，取五块电池各自曲线的耗电率，各占 1/5 权重：b_mean(S) = Σ b_i(S)/5。',
        '2. 对 60/b_mean(S) 按 SOC 积分，得到共同参考的 SOC–时间曲线。由于个体分界不同，这时不止三段。',
        '3. 将该参考再次近似为三条连续线段。整数 SOC 分界与最短段限制同上；按整个时间轴的精确积分平方误差选取。','',
        '这不是直接平均各电池第一段、第二段、第三段的斜率，也不是平均整段飞行时间。','',
        f"Bideal 范围：{' → '.join(f'{v:g}%' for v in ref['boundaries_soc'])}。",
        f"Bideal 耗电率：{' / '.join(f'{v:.6f}' for v in ref['rates_pp_min'])} SOC 百分点/分钟。",
        f"参考近似 RMSE：{ref['approximation_rmse_pp']:.4f} SOC 百分点，最大偏差 {ref['approximation_max_error_pp']:.4f}。",
        '该误差比较的是两条构造曲线，并不是独立实验预测误差。','',
        '## 如何换算真实实验区间','',
        '对于电池 i 在实际 Δt 秒内从 S_start 降到 S_end：','',
        '- H = ∫[S_end,S_start] 60/b_i(S) dS：对应自身 baseline 的等效悬停秒数。跨个体分界点分别积分。',
        '- g = H/Δt：相对于自身 baseline 的 SOC 耗电因子。不是测得的功率比。',
        '- 明确选择共同参考起始 SOC q（例如 70%），让 Bideal 前进 H 秒，得到 q_end。跨 Bideal 分界点分别处理。',
        '- 标准化耗电量 = q−q_end；标准化平均耗电率 = 60(q−q_end)/Δt。','',
        '实际 SOC 原值保留，不乘比例；新字段以 bn_ 开头。SOC 回升、未知电池/不同配对、超出校准区间会报错。',
        '这里假设实验相对 baseline 的变化可通过这一等效时间关系转移到参考电池，仍需实验验证，不保证消除全部电池/机体影响。','',
        '输入必需列：battery_id, drone_id, soc_start, soc_end, duration_s。额外列原样保留。',
        'normalize_interval 可用于单个区间；normalize_csv 为每行使用同一个参考起始 SOC，适合同 SOC 横向比较。',
        '不要把这些独立区间的 Bideal 降幅直接相加当作连续电池轨迹；连续轨迹必须将上一区间 q_end 作为下一区间 q_start。','',
        '```sh',
        f'python3 battery_normalization.py --model {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}/model.json \\',
        '  --input YOUR_PROCESSED_INTERVALS.csv --output NEW_NORMALIZED_INTERVALS.csv --reference-soc 70',
        '```','',
        '只写新输出文件，拒绝覆盖已有文件、原始输入，拒绝二次应用 bn_ 列。',
        '示例 input/output 各五行，来自真实校准运行中 70% 附近约 25 秒的原始采样端点；actual duration 保留真实秒数，没有伪造精确 25 秒端点。示例只演示接口，不属于独立验证。','',
        '## 验证与限制','',
        '- 源文件 SHA256 在拟合前及输出完成后核对；sources.json 提供来源。',
        '- 五次校准自身全段 H/Δt=1 是由端点拟合构造得到的，不是标准化效果验证。',
        '- 历史检查固定本次全部参数，不在历史记录上重新拟合。旧实验风/高度条件未确认，跨日期老化等因素可能影响结果；仅作探索性稳定性检查。',
        '- 目前每个当前电池/机体配对只有一次本批完整校准；在假定机体相同的前提下使用，不能单凭此拆分纯电池效应与机体效应。',
        '- 25 秒内整数 SOC 变化只有几个百分点，量化误差不可忽略；不宜仅据单个窗口评价模型。','',
        '结论：本次历史检查中，高电量和中电量区间的差异没有缩小，仅低电量区间略有改善。因此只能说工具和候选参考已建立，不能说已验证能稳定消除电池差异。','',
        '| 历史公共 SOC 区间（只用于检查） | 原始率 CV | 换算因子 CV |',
        '|---|---|---|']
    for row in historical:
        lines.append(f"| {row['band']} | {row['observed_rate_cv']:.3f} | {row['normalized_factor_cv']:.3f} |")
    lines += ['', 'CV 是五块电池等权的总体标准差/均值；若同一电池有多次历史记录先取其均值。下降为差异缩小，上升为差异变大；不以历史结果调参。','',
        '## 文件','',
        '- model.json：机器可读取的五条个体曲线与一条 Bideal，以及拟合约束和来源。',
        '- individual_segments.csv / individual_fit_quality.csv：15 段系数及原始数据拟合误差。',
        '- mean_rate_segments.csv：相同 SOC 等权平均后的多段参考；含五块电池分别贡献的速率。',
        '- reference_segments.csv / reference_search.csv：最终三段参考及完整分界搜索。',
        '- reference_curve_derived.csv：构造曲线采样，仅作绘图/检查，不是真实观测。',
        '- normalization_example_input.csv / normalization_example_output.csv：真实区间及保留原列的换算演示。',
        '- historical_validation.csv / historical_summary.csv：冻结参数的历史检查。',
        '- validation.json：数值与来源检查；six_curves.png / Bideal.png：静态图。','',
        '重新生成：运行 output_py/build_battery_normalization_candidate.py --output 一个不存在的新目录。不会改动已生成版本。',
        '构建依赖 Python、NumPy、pandas、matplotlib；换算工具仅使用 Python 标准库。',
        '离线测试：python3 -m unittest test_battery_normalization -v（不连接无人机）。','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def build(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory; candidate versions are not overwritten')
    sources = pd.read_csv(SOURCE_DIR/'coefficients.csv',dtype={'run_id':str}).drop_duplicates(['battery_id','run_id']).sort_values('battery_id')
    if len(sources)!=5 or dict(zip(sources.battery_id,sources.run_id)) != RUNS:
        raise ValueError('Unexpected calibration cohort')
    if any(PAIRS[r.battery_id]!=r.drone or r.run_id==EXCLUDED_RUN for r in sources.itertuples()):
        raise ValueError('Calibration pair/exclusion mismatch')
    prepared, fitted, models, segments, quality, grids = {}, {}, {}, [], [], {}
    for row in sources.itertuples():
        raw = load_trace(row)  # Re-read raw data; verify recorded SHA256.
        grid, traces = search([raw],min_width=MIN_WIDTH,min_seconds=MIN_SECONDS)
        best = grid.iloc[0]; trace = traces[0]
        upper, lower = int(best.upper), int(best.lower)
        result = fit(trace,upper,lower,MIN_SECONDS)
        result['boundaries'] = [upper,lower]
        curve = DischargeCurve((TOP,upper,lower,BOTTOM),tuple(result['rates']))
        models[row.battery_id] = dict(drone_id=row.drone, **curve.to_dict(), run_id=row.run_id,
            source=row.source, sha256=row.sha256, in_sample_rmse_pp=result['rmse'],
            max_in_sample_error_pp=result['max_error'], observed_sample_count=len(trace['t']),
            anchor_elapsed_times_s=result['anchor_times'].tolist())
        prepared[row.battery_id], fitted[row.battery_id], grids[row.battery_id] = trace,result,grid
        for k,(hi,lo,rate,duration) in enumerate(zip(curve.boundaries,curve.boundaries[1:],curve.rates_pp_min,result['durations']),1):
            segments.append(dict(battery_id=row.battery_id,drone_id=row.drone,segment=k,upper_soc=hi,
                lower_soc=lo,rate_pp_min=rate,duration_s=float(duration),run_id=row.run_id,source=row.source))
        quality.append(dict(battery_id=row.battery_id,rmse_pp=result['rmse'],max_error_pp=result['max_error'],
            upper=upper,lower=lower,raw_sample_count=len(trace['t']),feasible_candidates=len(grid)))
    curves = [DischargeCurve(tuple(m['boundaries_soc']),tuple(m['rates_pp_min'])) for m in models.values()]
    edges, rates, times = mean_rate_curve(curves)
    reference, ref_times, ref_grid = fit_reference(edges,times)
    best = ref_grid.iloc[0]
    model = dict(schema_version=1,kind='individual_battery_normalization_candidate',version=VERSION,
        status='candidate_not_active_not_independently_validated',units='SOC percentage points per minute',
        calibrated_soc_range=[BOTTOM,TOP],excluded_run_ids=[EXCLUDED_RUN],
        fitting=dict(individual='raw-sample RMSE of endpoint-anchored continuous lines',
            reference='time-integral RMSE of endpoint-anchored lines against same-SOC arithmetic mean-rate curve',
            min_segment_soc_width_pp=MIN_WIDTH,min_segment_duration_s=MIN_SECONDS,boundary_grid_pp=1,
            battery_weights={b:1/5 for b in models}),
        reference=dict(**reference.to_dict(),approximation_rmse_pp=float(best.rmse_pp),
            approximation_max_error_pp=float(best.max_error_pp),anchor_times_s=ref_times.tolist(),
            mean_rate_knot_soc=edges.tolist(),mean_rate_pp_min=rates.tolist(),mean_rate_knot_times_s=times.tolist()),
        batteries=models)
    out.mkdir(parents=True)
    (out/'model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(segments).to_csv(out/'individual_segments.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'individual_fit_quality.csv',index=False)
    for battery, grid in grids.items():
        grid.to_csv(out/f'{battery}_boundary_search.csv',index=False)
    mean_rows=[]
    for a,b,r,t0,t1 in zip(edges[:-1],edges[1:],rates,times[:-1],times[1:]):
        row=dict(upper_soc=a,lower_soc=b,mean_rate_pp_min=r,start_s=t0,end_s=t1,kind='model_derived')
        row.update({battery+'_rate_pp_min':curve.rate_at((a+b)/2) for battery,curve in zip(models,curves)})
        mean_rows.append(row)
    pd.DataFrame(mean_rows).to_csv(out/'mean_rate_segments.csv',index=False)
    pd.DataFrame([dict(segment=k+1,upper_soc=a,lower_soc=b,rate_pp_min=r,duration_s=dt)
        for k,(a,b,r,dt) in enumerate(zip(reference.boundaries,reference.boundaries[1:],reference.rates_pp_min,np.diff(ref_times)))]).to_csv(out/'reference_segments.csv',index=False)
    ref_grid.to_csv(out/'reference_search.csv',index=False)
    sample_times=np.unique(np.r_[np.arange(0,times[-1],.25),times,ref_times])
    pd.DataFrame(dict(equivalent_time_s=sample_times,mean_rate_reference_soc=np.interp(sample_times,times,edges),
        Bideal_soc=np.interp(sample_times,ref_times,reference.boundaries),kind='model_derived_not_observed')).to_csv(out/'reference_curve_derived.csv',index=False)

    # Real ~25s sample endpoints; no rounding duration to 25 and no invented samples.
    examples=[]
    for battery,trace in prepared.items():
        start=float(trace['crossing'][70]); i=int(np.searchsorted(trace['t'],start))
        j=int(np.argmin(abs(trace['t']-(start+25))))
        examples.append(dict(battery_id=battery,drone_id=PAIRS[battery],run_id=trace['run_id'],
            soc_start=float(trace['soc'][i]),soc_end=float(trace['soc'][j]),
            duration_s=float(trace['t'][j]-trace['t'][i]),nominal_interval_s=25,
            source_elapsed_start_s=float(trace['t'][i]),source_elapsed_end_s=float(trace['t'][j]),
            source=trace['source'],sha256=trace['sha256'],role='calibration_data_interface_demo_not_validation'))
    pd.DataFrame(examples).to_csv(out/'normalization_example_input.csv',index=False)
    normalize_csv(out/'model.json',out/'normalization_example_input.csv',out/'normalization_example_output.csv',70)
    normalizer=BatteryNormalizer.load(out/'model.json')
    inventory=pd.read_csv(SOURCE_DIR/'inventory.csv',dtype={'run_id':str})
    history,audits=historical_check(normalizer,inventory,out)

    # Independent numerical checks: dense quadrature vs exact reference error, and inverse mapping.
    dense_t=np.linspace(0,times[-1],200001)
    dense_error=np.interp(dense_t,times,edges)-np.interp(dense_t,ref_times,reference.boundaries)
    dense_rmse=float(np.sqrt(np.trapezoid(dense_error**2,dense_t)/times[-1]))
    if abs(dense_rmse-best.rmse_pp)>1e-6:
        raise AssertionError('Reference analytic/numerical RMSE mismatch')
    max_inverse_error=0.
    for curve in [*curves,reference]:
        for start in np.linspace(20,95,25):
            for end in np.linspace(20,start,20):
                h=curve.equivalent_seconds(start,end)
                max_inverse_error=max(max_inverse_error,abs(curve.advance(start,h)-end))
    if max_inverse_error>1e-9:
        raise AssertionError('SOC/time inverse check failed')
    provenance=sources[['battery_id','drone','run_id','source','sha256']].to_dict('records')
    for item in provenance+audits:
        if hashlib.sha256((ROOT/item['source']).read_bytes()).hexdigest()!=item['sha256']:
            raise AssertionError('Source changed during build')
    (out/'sources.json').write_text(json.dumps(dict(calibration=provenance,historical=audits,
        excluded_run_ids=[EXCLUDED_RUN]),indent=2)+'\n',encoding='utf-8')
    validation=dict(source_hashes_verified_before_and_after=True,calibration_runs=5,
        excluded_run_used=False,calibration_scope='95 to 20 only',
        max_inverse_error_pp=max_inverse_error,reference_exact_rmse_pp=float(best.rmse_pp),
        reference_dense_quadrature_rmse_pp=dense_rmse,historical=history,
        independent_validation_status='not established: historical wind/height not confirmed; one current run per pair',
        activation='none: old tables, raw files, flight and app code unchanged',model_sha256=normalizer.sha256)
    (out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_readme(out,model,history)
    make_figures(out,prepared,fitted,reference,edges,times,ref_times)
    print(json.dumps(dict(output=str(out),reference=model['reference'],individual_quality=quality,historical=history),ensure_ascii=False,indent=2))
    return model


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    build(parser.parse_args().output)
