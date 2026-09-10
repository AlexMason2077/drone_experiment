"""Freeze an existing rate reference without refitting or modifying live data."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

from ml_policy.build_real_medium_training_labels import FORMATIONS, WINDS, canonical_formation, sha256
from ml_policy.oracle_optimizer import UNSAFE_STRUCTURES_BY_CONDITION


def read_rows(path):
    # Accept both upstream UTF-8 files and spreadsheet-friendly BOM snapshots.
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns=None):
    if not rows:
        raise ValueError(f'No records to freeze: {path}')
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def same(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'Numerical mismatch: {a} != {b}')


def record_key(r):
    return r['experiment_directory']+'::'+r['run_id']


def verify(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, info in manifest['files'].items():
        path = directory/name
        if path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise ValueError(f'Frozen file changed: {name}')
    refs = json.loads((directory/'rate_reference.json').read_text())
    long_rows = read_rows(directory/'discharge_rates.csv')
    coverage = read_rows(directory/'coverage.csv')
    sources = read_rows(directory/'source_run_drone_rates.csv')
    selected = read_rows(directory/'source_runs.csv')
    batteries = {r['battery_id']: r for r in read_rows(directory/'battery_calibration.csv')}
    source_index = {(record_key(r), int(r['slot_id'].rsplit('_', 1)[1])): r for r in sources}
    long_index = {(r['wind_direction'], int(r['wind_level']), r['formation'], int(r['spacing_cm']), int(r['position'])): r for r in long_rows}
    if len(source_index) != len(sources) or len(long_index) != len(long_rows):
        raise ValueError('Duplicate source/position or rate key')
    if len({record_key(r) for r in selected}) != len(selected):
        raise ValueError('Duplicate selected runs')
    if len(sources) != 5*len(selected) or len(long_rows) != 5*len(refs):
        raise ValueError('Incomplete five-drone or five-position records')
    used_groups = []
    for cell in refs:
        groups = cell['source_groups']
        if cell['source_run_count'] != len(groups):
            raise ValueError('Reference source count mismatch')
        used_groups.extend(groups)
        for p in range(1, 6):
            contributing = [source_index[(g, p)] for g in groups]
            for r in contributing:
                key = (r['wind_direction'], int(float(r['wind_level'])), canonical_formation(r['formation']), int(float(r['inter_drone_spacing_cm'])))
                if key != (cell['wind'], cell['level'], cell['formation'], cell['spacing']):
                    raise ValueError('Source condition does not match frozen rate cell')
                same(r['physical_to_Bideal_scale'], batteries[r['battery_id']]['scale_physical_drop_to_Bideal'])
            values = [float(r['curve_slope_Bideal_pp_per_min']) for r in contributing]
            # Verify, do not replace the already-approved frozen number.
            same(statistics.mean(values), cell['rates'][p-1])
            row = long_index[(cell['wind'], cell['level'], cell['formation'], cell['spacing'], p)]
            same(row['discharge_rate_Bideal_pp_per_min'], cell['rates'][p-1])
            same(row['discharge_rate_Bideal_pp_per_second'], cell['rates'][p-1]/60)
            if len(values) > 1:
                same(row['between_run_stddev_pp_per_min'], statistics.stdev(values))
            elif row['between_run_stddev_pp_per_min'] != '':
                raise ValueError('Single-run standard deviation must be blank')
    if len(used_groups) != len(set(used_groups)) or set(used_groups) != {record_key(r) for r in selected}:
        raise ValueError('Reference/source population mismatch')
    unsafe = {tuple(x) for x in manifest['unsafe_condition_cells']}
    available = {(r['wind'], r['level'], r['formation'], r['spacing']) for r in refs}
    seen = set()
    for row in coverage:
        key = (row['wind_direction'], int(row['wind_level']), row['formation'], int(row['spacing_cm']))
        seen.add(key)
        expected = 'unsafe_excluded' if key in unsafe else ('available' if key in available else 'missing_data')
        if row['status'] != expected or (key in unsafe and key in available):
            raise ValueError(f'Coverage mismatch: {key}')
    expected_keys = {(w, level, f, s) for w in WINDS for level in (1, 2) for f in FORMATIONS for s in (50, 75)}
    if seen != expected_keys or len(coverage) != len(expected_keys):
        raise ValueError('Coverage inventory incomplete/duplicated')
    return {'version': manifest['version'], 'file_checksums': 'passed', 'source_mean_reconciliation': 'passed',
            'battery_scale_reconciliation': 'passed', 'coverage_check': 'passed',
            'source_runs': len(selected), 'rate_cells': len(refs), 'position_rates': len(long_rows)}


def freeze(reference_dir, source_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError(f'Frozen version already exists; use a new version: {output_dir}')
    upstream = json.loads((reference_dir/'manifest.json').read_text())
    if upstream.get('quality_policy') != 'safety_only_keep_qc_warnings':
        raise ValueError('Expected the user-approved safety-only reference')
    for path, digest in upstream['source_files_sha256'].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f'Upstream data changed since labelling: {path}')
    ref_path = reference_dir/'rate_reference.json'
    calibration_path = source_dir/'battery_ideal_normalization.csv'
    selection_path = source_dir/'selected_runs_by_database_cell.csv'
    rates_path = source_dir/'forward_discharge_rate_run_drone.csv'
    source_files = [ref_path, reference_dir/'manifest.json', calibration_path, selection_path, rates_path]
    source_hashes = {str(p.resolve()): sha256(p) for p in source_files}
    refs = json.loads(ref_path.read_text())
    all_groups = {g for c in refs for g in c['source_groups']}
    selected = [r for r in read_rows(selection_path) if record_key(r) in all_groups]
    source_rows = [r for r in read_rows(rates_path) if record_key(r) in all_groups]
    source_index = {(record_key(r), int(r['slot_id'].rsplit('_', 1)[1])): r for r in source_rows}
    long_rows = []
    for cell in refs:
        for p, rate in enumerate(cell['rates'], 1):
            sources = [source_index[(g, p)] for g in cell['source_groups']]
            rates = [float(r['curve_slope_Bideal_pp_per_min']) for r in sources]
            same(rate, statistics.mean(rates))
            long_rows.append({
                'version': output_dir.name, 'protocol': 'processed_forward_flight', 'soc_regime': 'medium',
                'model_soc_min_pct': 40, 'model_soc_max_pct': 75,
                'wind_direction': cell['wind'], 'wind_level': cell['level'],
                'formation': cell['formation'], 'spacing_cm': cell['spacing'], 'position': p,
                'discharge_rate_Bideal_pp_per_min': rate,
                'discharge_rate_Bideal_pp_per_second': rate/60,
                'source_run_count': cell['source_run_count'],
                'between_run_stddev_pp_per_min': cell['rate_stddev'][p-1] if cell['rate_stddev'][p-1] is not None else '',
                'source_rate_min_pp_per_min': min(rates), 'source_rate_max_pp_per_min': max(rates),
                'source_start_soc_min_pct': min(float(r['start_reported_soc_pct']) for r in sources),
                'source_start_soc_max_pct': max(float(r['start_reported_soc_pct']) for r in sources),
                'source_end_soc_min_pct': min(float(r['end_reported_soc_pct']) for r in sources),
                'zero_fitted_source_count_at_position': sum(r == 0 for r in rates),
                'qc_flagged_source_count_at_position': sum(bool(r['curve_qc_flags']) for r in sources),
                'cell_qc_flagged_run_count': cell['qc_flagged_source_count'],
                'cell_metadata_warning_count': cell['metadata_warning_count'],
                'source_groups': json.dumps(cell['source_groups'], separators=(',', ':')),
            })
    unsafe = sorted((w, level, canonical_formation(f), s)
                    for (w, level), cells in UNSAFE_STRUCTURES_BY_CONDITION.items() for f, s in cells)
    ref_index = {(r['wind'], r['level'], r['formation'], r['spacing']): r for r in refs}
    coverage = []
    for w, level, f, spacing in sorted((w, lv, f, s) for w in WINDS for lv in (1, 2) for f in FORMATIONS for s in (50, 75)):
        key = (w, level, f, spacing)
        status = 'unsafe_excluded' if key in unsafe else ('available' if key in ref_index else 'missing_data')
        coverage.append({'wind_direction': w, 'wind_level': level, 'formation': f, 'spacing_cm': spacing,
                         'status': status, 'source_run_count': ref_index[key]['source_run_count'] if key in ref_index else 0,
                         'reason': {'available': 'fixed_existing_reference', 'unsafe_excluded': 'existing_collision_safety_mask',
                                    'missing_data': 'no_record_in_current_processed_medium_reference'}[status]})
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(ref_path, output_dir/'rate_reference.json')
    shutil.copyfile(calibration_path, output_dir/'battery_calibration.csv')
    shutil.copyfile(reference_dir/'manifest.json', output_dir/'upstream_answer_manifest.json')
    write_csv(output_dir/'discharge_rates.csv', long_rows)
    write_csv(output_dir/'coverage.csv', coverage)
    write_csv(output_dir/'source_runs.csv', selected)
    write_csv(output_dir/'source_run_drone_rates.csv', source_rows)
    version = output_dir.name
    readme = f'''# 耗电率冻结版本：{version}

本版固定当前已确认的medium耗电率参考表，不重新拟合、不改原始实验、不重新训练模型，也不切换现有Online代码的数据路径。未来修改必须生成新版本，冻结脚本拒绝覆盖同名目录。

## 主文件

- `discharge_rates.csv`：55组条件配置 × 5个position，共275条Bideal耗电率，供查看。
- `rate_reference.json`：与当前训练参考表逐字节相同，供程序读取。
- `battery_calibration.csv`：冻结B10–B15共6个电池的原Bideal转换参数；没有B06参数，不能默认补1。
- `coverage.csv`：60组理论条件配置逐一列明available、unsafe_excluded、missing_data。
- `source_runs.csv`、`source_run_drone_rates.csv`：134次源实验、670条单机处理结果的快照，保留QC与元数据字段。
- `manifest.json`：单位、假设、源文件/冻结文件哈希及版本。

## 固定口径

来源为已经处理的medium前进实验，55个参考单元、134次五机实验。同风向/风力等级/formation/spacing/position，对源实验已拟合的Bideal耗电率按run等权平均。导出值直接取自既有参考表；重算均值只用于一致性检查。

单位`pp/min`表示每分钟下降多少电量百分点，不是相对百分比，也不是瓦特或瓦时。`pp/second`仅为除以60后的同值换算。wind_level是原有等级1/2，本版没有虚构对应m/s。

适用模型SOC区间40%–75%，采用当前确认的区间内线性假设；不自动外推到high/low。position是formation中的slot1..5，不是无人机编号或Mission Pad编号。charging pad数量不属于耗电率表的维度。

电池换算：`a = scale_physical_drop_to_Bideal`；`Bideal_drop = physical_drop × a`；某机处于位置p时，`physical_rate = frozen_Bideal_rate[p] / a_own_battery`。不同位置转用该耗电率仍是Bideal统一化后的建模假设，不代表实际完成过全部位置交换实验。

本版冻结的是放电参考及其电池转换依据，不包含充电模型或总时间标签。Bideal转换基准仍是既有75%→30%/36%的baseline拟合，未重新拟合为严格medium，范围差异明确保留。

## 安全与缺失

只排除这些明确的风况/队形/间距组合，不排除整个formation：head lv2 column50；side lv2 column50、diamond50；tail lv2 diamond50。

side lv1 column50当前没有已处理medium参考记录，保留missing_data，不当成不安全，也不填0。其他QC标记只作警告，按用户确认保留参与计算。短时间SOC拟合率为0不等于实际零耗电；有单次来源的单元，其跨run标准差为空而非0。

实验标定长度为250cm；原forward时间由逐机清理得到，本版没有将轨迹修改成共同25秒或假称实际速度严格一致。用耗电率推算后续飞行应保留这一限制，不能把wind tunnel悬停和本版前进数据混同。

## 如何保持冻结

Offline与Online后续应显式引用本版路径，不读取会自动更新的临时汇总。当前只交付快照，没有自动切换任何消费者。校验命令：

```text
python3 -m ml_policy.freeze_discharge_rates verify --directory frozen_data/discharge_rates/{version}
```

版本号不变时不应编辑这些文件；文件哈希能够检测变更，属于版本快照约定，并非操作系统级不可修改锁。若要增加数据、修改率值或模型适用范围，请生成v2。

数据验证技能用于核对所有率值与来源、Bideal系数和覆盖表；冻结表示固定可追溯的一版计算依据，不表示消除了测量不确定性或完成了独立实飞验证。
'''
    (output_dir/'README.md').write_text(readme, encoding='utf-8')
    if source_hashes != {str(p.resolve()): sha256(p) for p in source_files}:
        raise ValueError('Source files changed during freeze')
    manifest = {
        'version': version, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen_snapshot_not_empirical_validation', 'schema_version': 1,
        'rate_value_origin': 'existing_approved_rate_reference_no_refit',
        'quality_policy': upstream['quality_policy'], 'source_runs': len(selected), 'source_drone_rows': len(source_rows),
        'available_condition_cells': len(refs), 'position_rate_rows': len(long_rows),
        'source_files_sha256': source_hashes, 'freeze_script_sha256': sha256(Path(__file__)),
        'unsafe_condition_cells': unsafe,
        'missing_condition_cells': [(r['wind_direction'], r['wind_level'], r['formation'], r['spacing_cm']) for r in coverage if r['status']=='missing_data'],
        'rate_unit': 'Bideal percentage points per minute', 'model_soc_interval_pct': [40, 75],
        'battery_scale_definition': 'Bideal_drop/physical_drop; physical_rate=assigned_Bideal_rate/own_battery_scale',
        'protocol': 'processed_forward_flight', 'source_commanded_segment_cm': 250,
        'consumer_paths_changed': False, 'readonly_os_lock': False,
        'files': {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(output_dir.iterdir()) if p.is_file()},
    }
    (output_dir/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    result = verify(output_dir)
    (output_dir/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze_parser = commands.add_parser('freeze')
    freeze_parser.add_argument('--reference-dir', type=Path, required=True)
    freeze_parser.add_argument('--source-dir', type=Path, required=True)
    freeze_parser.add_argument('--output-dir', type=Path, required=True)
    verify_parser = commands.add_parser('verify')
    verify_parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    result = (freeze(args.reference_dir, args.source_dir, args.output_dir) if args.command == 'freeze' else verify(args.directory))
    print(json.dumps(result, indent=2, ensure_ascii=False))
