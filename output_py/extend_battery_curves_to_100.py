"""Extend existing first-stage model lines to 100%; do not add observed records.

No fitting, raw-data ingestion, flight control, or active-model replacement.
All exported curve points are explicitly model-generated, never observations.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from battery_normalization import BatteryNormalizer, DischargeCurve

SOURCE=ROOT/'analysis_results/battery_normalization_pooled_v2_20260909/model.json'
DEFAULT_OUT=ROOT/'analysis_results/battery_normalization_extended_v3_20260909'


def curve(item):
    return DischargeCurve(tuple(item['boundaries_soc']),tuple(item['rates_pp_min']))


def extend_model(original,source_sha256):
    """Retain all slopes and internal knots; extend each already-fitted first line."""
    if original.get('calibrated_soc_range') != [20,95]:
        raise ValueError('This operation requires the reviewed 20..95 calibration')
    result=copy.deepcopy(original)
    result['version']='individual_three_stage_pooled_candidate_v3_extended100_20260909'
    result['status']='candidate_with_explicit_extrapolation_not_active_not_independently_validated'
    result['supported_soc_range']=[20,100]
    result['extension']=dict(method='Continue each existing first-stage straight line without refitting',
        user_requested=True,extrapolated_soc_interval=dict(lower_exclusive=95,upper_inclusive=100),
        calibrated_soc_range_unchanged=[20,95],synthetic_not_observed=True,
        source_model_version=original['version'],source_model_sha256=source_sha256,
        time_origin='Each extended curve starts at modeled 100% at t=0',
        reference_policy='Extend existing Bideal first slope independently; do not rebuild or refit Bideal',
        existing_fit_diagnostics_scope='Original 95..20 calibration only, not validation of 100..95')
    for key,item in [('Bideal',result['reference']),*result['batteries'].items()]:
        old=curve(item)
        if old.boundaries[0]!=95:raise ValueError('Already extended or unexpected source curve: '+key)
        extension_s=5*60/old.rates_pp_min[0]
        item['boundaries_soc'][0]=100.0
        new=curve(item)
        # Keep observation-specific timestamps and approximation diagnostics as
        # provenance only. The only live anchor array now starts at model 100%.
        diagnostic_keys=['anchor_times_s','anchor_elapsed_times_s','mean_rate_knot_soc',
            'mean_rate_pp_min','mean_rate_knot_times_s','in_sample_rmse_pp',
            'max_in_sample_error_pp','constructed_mean_curve_rmse_pp',
            'approximation_rmse_pp','approximation_max_error_pp']
        provenance={k:item.pop(k) for k in diagnostic_keys if k in item}
        item['source_95_to_20_fit_diagnostics']=provenance
        item['anchor_times_s']=[new.equivalent_seconds(100,s) for s in new.boundaries]
        item['time_origin_soc']=100
        item['modeled_100_to_95_duration_s']=extension_s
        item['calibrated_soc_range']=[20,95]
        item['supported_soc_range']=[20,100]
        item['extrapolation_method']='existing_first_stage_linear_continuation'
        item['calibrated_95_time_on_extended_axis_s']=extension_s
    BatteryNormalizer(result)  # Validate domain agreement without touching devices.
    return result


def write_csv(path,rows):
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def build(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Use a new output folder; old versions are preserved')
    before=source.read_bytes();source_hash=hashlib.sha256(before).hexdigest()
    original=json.loads(before)
    model=extend_model(original,source_hash)
    encoded=(json.dumps(model,ensure_ascii=False,indent=2)+'\n').encode()
    model_hash=hashlib.sha256(encoded).hexdigest()
    segment_rows=[];points=[];summary=[];max_error=0.
    for name,item in [('Bideal',model['reference']),*model['batteries'].items()]:
        fitted=curve(item)
        previous=curve(original['reference'] if name=='Bideal' else original['batteries'][name])
        offset=item['modeled_100_to_95_duration_s']
        assert fitted.rates_pp_min==previous.rates_pp_min
        assert fitted.boundaries[1:]==previous.boundaries[1:]
        assert abs(fitted.advance(100,offset)-95)<1e-9
        for soc in range(95,19,-1):
            error=abs((fitted.equivalent_seconds(100,soc)-offset)-previous.equivalent_seconds(95,soc))
            max_error=max(max_error,error)
        for k,(hi,lo,r) in enumerate(zip(fitted.boundaries,fitted.boundaries[1:],fitted.rates_pp_min),1):
            segment_rows.append(dict(battery_id=name,segment=k,upper_soc=hi,lower_soc=lo,rate_pp_min=r,
                contains_extrapolated_100_95=k==1,record_type='model_coefficient_not_observation'))
        for soc in range(100,19,-1):
            points.append(dict(battery_id=name,soc_percent=soc,time_from_modeled_100_s=fitted.equivalent_seconds(100,soc),
                time_relative_to_original_95_s=fitted.equivalent_seconds(100,soc)-offset,
                model_rate_pp_min=fitted.rate_at(soc),is_observed=False,is_extrapolated=soc>95,
                record_type='synthetic_linear_extrapolation' if soc>95 else 'fitted_curve_sample_not_observed',
                source_model_sha256=source_hash,model_sha256=model_hash))
        summary.append(dict(battery_id=name,first_rate_pp_min=fitted.rates_pp_min[0],
            upper_internal_soc=fitted.boundaries[1],lower_internal_soc=fitted.boundaries[2],
            modeled_100_to_95_s=offset,modeled_100_to_20_s=fitted.equivalent_seconds(100,20)))
    assert max_error<1e-9
    out.mkdir(parents=True)
    with (out/'model.json').open('xb') as stream:stream.write(encoded)
    write_csv(out/'segments.csv',segment_rows)
    write_csv(out/'model_curve_samples_100_to_20.csv',points)
    write_csv(out/'model_extrapolated_100_to_95.csv',[p for p in points if p['soc_percent']>=95])
    write_csv(out/'extension_summary.csv',summary)
    normalizer=BatteryNormalizer.load(out/'model.json')
    example=normalizer.normalize_interval('B10','drone_2',100,98,10,100)
    assert example['bn_uses_extrapolation'] is True
    assert source.read_bytes()==before
    validation=dict(source_model_unchanged=True,source_model_sha256=source_hash,model_sha256=model_hash,
        six_curves_extended=True,all_slopes_and_internal_knots_unchanged=True,
        max_original_domain_time_shift_error_s=max_error,calibrated_soc_range=[20,95],supported_soc_range=[20,100],
        synthetic_sample_rows=len(points),new_top_interval_rows=sum(p['soc_percent']>95 for p in points),
        actual_observation_rows_added=0,raw_data_modified=False,flight_or_active_rate_tables_modified=False,
        extrapolation_flags_verified=True,example_role='synthetic software check, not a flight or validation sample')
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    lines=['# 100%–95% 首段直线外推（v3候选）','',
        '按用户要求，延长 Bideal 和 B10–B14 已有的第一段直线至100%。所有斜率与两个内部SOC分界点不变；没有重新拟合，也没有增加第四段。','',
        '重要：100%–95% 是模型外推，不是真实观测；模型可用范围扩为20%–100%，原始校准范围仍为20%–95%。20%以下没有外推。','',
        '## 时间轴与参数','',
        '每条曲线以模型100%为新t=0；首段 SOC(t)=100−r_high×t/60（t以秒计）。到达95%的时刻为300/r_high秒。',
        '95%以下仍是原来那条曲线，只整体平移时间轴；原始实验时间戳没有改动。Bideal沿自身现有首段延长，不重新对五条延伸曲线求均值并拟合。','',
        '| 曲线 | 第一段 | 第二段 | 第三段 | 外推100%→95%耗时（秒） |','|---|---|---|---|---|']
    for row in summary:
        name=row['battery_id'];m=model['reference'] if name=='Bideal' else model['batteries'][name]
        bands=[f'{hi:g}%→{lo:g}%：{r:.3f}' for hi,lo,r in zip(m['boundaries_soc'],m['boundaries_soc'][1:],m['rates_pp_min'])]
        lines.append(f"| {name} | {' | '.join(bands)} | {row['modeled_100_to_95_s']:.3f} |")
    lines += ['', '表中速率单位为SOC百分点/分钟。新时间轴上的所有值都是模型计算值，不是实际飞行时长保证。','',
        '## 文件与使用','',
        '- model.json：兼容battery_normalization.py的新版候选。',
        '- segments.csv：三段模型系数，首段标记包含外推区间。',
        '- model_extrapolated_100_to_95.csv：6条曲线各6个整数SOC采样点，100..96标记is_extrapolated=True，95为衔接点；所有点is_observed=False。',
        '- model_curve_samples_100_to_20.csv：完整模型曲线采样，同样不属于真实实验数据；不得直接追加到真实训练集冒充实测。',
        '- extension_summary.csv：新增区间时长；validation.json：连续性、原区间保持不变和来源检查。',
        '- source_95_to_20_fit_diagnostics 保留原拟合锚点、误差等来源信息；这些误差没有验证100%–95%的外推准确性。',
        '- 换算输出的bn_battery_uses_extrapolation、bn_reference_uses_extrapolation、bn_uses_extrapolation说明该区间是否用了外推参数。',
        '- 原始数据、旧模型、正式耗电表和飞行逻辑未改。使用这版时显式传入此目录model.json，旧模型仍拒绝95%以上。','',
        '复现：`python3 output_py/extend_battery_curves_to_100.py --output 一个不存在的新目录`。','']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(output=str(out),summary=summary,validation=validation),ensure_ascii=False,indent=2))
    return model


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();build(args.source,args.output)
