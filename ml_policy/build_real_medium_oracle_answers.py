"""One oracle-answer row per observed five-drone run and charging-pad scenario.

Candidate permutations are searched only in an audit table, not added as real
training observations. This is a pooled-rate model oracle, not measured truth.
No flight commands, model training, or source-data writes occur here.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from ml_policy.build_real_medium_training_labels import (
    FEATURES, FLIGHT_SECONDS, FORMATIONS, ROOT, SOURCE, TOLERANCE, WINDS,
    cost_values, make_bundle, mission_values, read_rows, sha256, truth,
)
from ml_policy.charging_model import FULLY_CHARGED_SOC, ZERO_TO_FULLY_CHARGED_MINUTES
from ml_policy.oracle_optimizer import UNSAFE_STRUCTURES_BY_CONDITION


def load_bundles(source_dir):
    selected = {}
    for r in read_rows(source_dir / 'selected_runs_by_database_cell.csv'):
        if (r['selection_status'] == 'selected'
                and truth(r['formal_clean_trajectory_candidate'])
                and truth(r['all_five_within_75_to_40_range'])):
            key = (r['experiment_directory'], r['run_id'])
            if key in selected:
                raise ValueError(f'Duplicate source selection: {key}')
            selected[key] = r
    grouped = defaultdict(list)
    for r in read_rows(source_dir / 'forward_discharge_rate_run_drone.csv'):
        key = (r['experiment_directory'], r['run_id'])
        if key in selected:
            grouped[key].append(r)
    if set(grouped) != set(selected):
        raise ValueError('Selected source runs are missing processed drone records')
    return [make_bundle(grouped[k], selected[k]) for k in sorted(grouped)]


def structure_id(formation, spacing):
    return f'{formation}_{spacing}'


def configuration_id(formation, spacing, positions):
    return structure_id(formation, spacing) + '__p' + '-'.join(str(p + 1) for p in positions)


def make_rate_bank(bundles, *, include_flagged=False):
    """Equal-weight means; optional user policy retains QC-flagged sources."""
    groups = defaultdict(list)
    for b in bundles:
        if not b['unsafe'] and (include_flagged or not b['flags']):
            groups[(b['wind'], b['level'], b['formation'], b['spacing'])].append(b)
    bank = {}
    for key, sources in sorted(groups.items()):
        rates = [statistics.mean(b['rates'][i] for b in sources) for i in range(5)]
        bank[key] = {
            'wind': key[0], 'level': key[1], 'formation': key[2], 'spacing': key[3],
            'rates': rates, 'source_run_count': len(sources),
            'rate_stddev': [statistics.stdev(b['rates'][i] for b in sources)
                            if len(sources) > 1 else None for i in range(5)],
            'source_groups': [b['source_group'] for b in sources],
            'source_qc_flags': {b['source_group']: b['flags'] for b in sources if b['flags']},
            'qc_flagged_source_count': sum(bool(b['flags']) for b in sources),
            'zero_fitted_rate_source_count': sum(any(r == 0 for r in b['rates']) for b in sources),
            'metadata_warning_count': sum(bool(b['metadata_warning']) for b in sources),
        }
    return bank


def coverage(b, bank):
    unsafe = UNSAFE_STRUCTURES_BY_CONDITION.get((b['wind'], b['level']), ())
    missing, excluded, supported = [], [], []
    for f in FORMATIONS:
        for s in (50, 75):
            name = structure_id(f, s)
            if ('echalon' if f == 'echelon' else f, s) in unsafe:
                excluded.append(name)
            elif (b['wind'], b['level'], f, s) not in bank:
                missing.append(name)
            else:
                supported.append(name)
    return supported, missing, excluded


def evaluate_run(b, bank):
    """Same observed SOC and own-battery scale for every counterfactual."""
    by_k = {k: [] for k in range(1, 6)}
    for key, cell in sorted(bank.items()):
        if key[:2] != (b['wind'], b['level']):
            continue
        for positions in itertools.permutations(range(5)):
            assigned, physical, arrival = mission_values(b['soc'], cell['rates'], b['scales'], positions)
            valid = all(40 - TOLERANCE <= s <= 75 + TOLERANCE for s in arrival)
            base = {
                'configuration_id': configuration_id(cell['formation'], cell['spacing'], positions),
                'formation': cell['formation'], 'spacing_cm': cell['spacing'],
                'positions': list(p + 1 for p in positions),
                'assigned_Bideal_rates': assigned, 'physical_rates': physical,
                'arrival_soc': arrival, 'rate_source_run_count': cell['source_run_count'],
                'rate_metadata_warning_count': cell['metadata_warning_count'],
                'valid': valid,
                'exclusion_reason': '' if valid else 'arrival_outside_medium_40_to_75',
            }
            for k in range(1, 6):
                row = dict(base, charging_pad_count=k, charging_times_min=[], charging_pads=[],
                           charging_completion_time_min='', total_time_min='')
                if valid:
                    jobs, pads, costs = cost_values(arrival, k)
                    row.update(charging_times_min=jobs, charging_pads=[p + 1 for p in pads],
                               charging_completion_time_min=costs['charging_completion_time_min'],
                               total_time_min=costs['total_required_time_min'])
                by_k[k].append(row)
    return by_k


def add_configuration_fields(row, prefix, candidate):
    if candidate is None:
        return
    row.update({
        f'{prefix}_configuration': candidate['configuration_id'],
        f'{prefix}_formation': candidate['formation'],
        f'{prefix}_spacing_cm': candidate['spacing_cm'],
        f'{prefix}_positions': candidate['positions'],
        f'{prefix}_total_time_min': candidate['total_time_min'],
        f'{prefix}_charging_completion_time_min': candidate['charging_completion_time_min'],
        f'{prefix}_rate_source_run_count': candidate['rate_source_run_count'],
        f'{prefix}_rate_metadata_warning_count': candidate['rate_metadata_warning_count'],
    })
    for i in range(5):
        for name, source in [('arrival_soc', 'arrival_soc'), ('charging_time_min', 'charging_times_min'),
                             ('charging_pad', 'charging_pads')]:
            row[f'{prefix}_{name}_d{i+1}'] = candidate[source][i] if candidate['valid'] else ''


BASE_COLUMNS = [
    'source_experiment', 'source_run_id', 'source_group_id', 'scenario_id',
    'wind_direction', 'charging_pad_count', 'remaining_distance_cm', 'flight_seconds',
    'source_qc_flags', 'source_metadata_warning', 'training_eligible', 'exclusion_reasons',
    'oracle_answer_available', 'time_gap_available', 'candidate_count_evaluated', 'candidate_count_valid',
    'candidate_count_outside_medium', 'supported_structures', 'missing_structures', 'unsafe_structures',
    'covers_all_safe_structures', 'reference_uses_all_selected_runs', 'reference_metadata_warning',
    'source_trial_time_available', 'source_trial_time_exclusion', 'source_trial_total_time_min',
    'time_gap_min', 'target_log1p_time_gap', 'worst_minus_best_min',
    'best_tie_count', 'best_tied_configurations', 'worst_tie_count', 'worst_tied_configurations',
]
for i in range(1, 6):
    BASE_COLUMNS += [f'battery_id_d{i}', f'soc_d{i}', f'source_position_d{i}', f'source_Bideal_rate_d{i}']
CONFIG_COLUMNS = []
for prefix in ('current', 'best', 'worst'):
    CONFIG_COLUMNS += [f'{prefix}_{name}' for name in (
        'configuration', 'formation', 'spacing_cm', 'positions', 'total_time_min',
        'charging_completion_time_min', 'rate_source_run_count', 'rate_metadata_warning_count')]
    for i in range(1, 6):
        CONFIG_COLUMNS += [f'{prefix}_{name}_d{i}' for name in ('arrival_soc', 'charging_time_min', 'charging_pad')]
COLUMNS = BASE_COLUMNS + FEATURES + CONFIG_COLUMNS


def answer_row(b, bank, k, candidates, *, include_flagged=False):
    import math
    row = dict.fromkeys(COLUMNS, '')
    supported, missing, unsafe = coverage(b, bank)
    row.update(source_experiment=b['exp'], source_run_id=b['run'], source_group_id=b['source_group'],
               scenario_id=f"{b['source_group']}::K{k}", wind_direction=b['wind'], wind_level=b['level'],
               charging_pad_count=k, pad_availability_ratio=k/5, remaining_distance_cm=250,
               flight_seconds=FLIGHT_SECONDS, source_qc_flags=b['flags'],
               source_metadata_warning=b['metadata_warning'], reference_uses_all_selected_runs=True,
               supported_structures=supported, missing_structures=missing, unsafe_structures=unsafe,
               covers_all_safe_structures=not missing, candidate_count_evaluated=len(candidates),
               spacing_ratio=b['spacing']/75, oracle_answer_available=False, time_gap_available=False,
               source_trial_time_available=False, best_tie_count=0, worst_tie_count=0,
               reference_metadata_warning=any(c['rate_metadata_warning_count'] for c in candidates))
    row.update({f'wind_{w}': int(w == b['wind']) for w in WINDS})
    row.update({f'formation_{f}': int(f == b['formation']) for f in FORMATIONS})
    for i in range(5):
        row.update({f'battery_id_d{i+1}': b['rows'][i]['battery_id'], f'soc_d{i+1}': b['soc'][i],
                    f'soc_d{i+1}_ratio': b['soc'][i]/100, f'battery_scale_d{i+1}': b['scales'][i],
                    f'source_position_d{i+1}': b['observed_positions'][i]+1,
                    f'source_Bideal_rate_d{i+1}': b['rates'][b['observed_positions'][i]]})
    current_id = configuration_id(b['formation'], b['spacing'], b['observed_positions'])
    row.update(current_configuration=current_id, current_formation=b['formation'],
               current_spacing_cm=b['spacing'], current_positions=[p+1 for p in b['observed_positions']])
    current = next((c for c in candidates if c['configuration_id'] == current_id), None)
    if current:
        add_configuration_fields(row, 'current', current)
        for i, value in enumerate(current['assigned_Bideal_rates'], 1):
            row[f'assigned_Bideal_rate_d{i}'] = value
    source_reasons = ([] if include_flagged else list(b['flags'])) + (['unsafe_source_structure'] if b['unsafe'] else [])
    _, _, source_arrival = mission_values(b['soc'], b['rates'], b['scales'], b['observed_positions'])
    if not all(40-TOLERANCE <= s <= 75+TOLERANCE for s in source_arrival):
        source_reasons.append('source_trial_arrival_outside_medium')
    if not source_reasons:
        row['source_trial_total_time_min'] = cost_values(source_arrival, k)[2]['total_required_time_min']
        row['source_trial_time_available'] = True
    row['source_trial_time_exclusion'] = source_reasons
    valid = sorted((c for c in candidates if c['valid']), key=lambda c: (c['total_time_min'], c['configuration_id']))
    row['candidate_count_valid'] = len(valid)
    row['candidate_count_outside_medium'] = len(candidates)-len(valid)
    reasons = ([] if include_flagged else list(b['flags'])) + (['unsafe_source_structure'] if b['unsafe'] else [])
    if not current:
        reasons.append('current_structure_has_no_rate_reference')
    elif not current['valid']:
        reasons.append('current_pooled_arrival_outside_medium')
    if valid:
        best, worst = valid[0], valid[-1]
        add_configuration_fields(row, 'best', best)
        add_configuration_fields(row, 'worst', worst)
        row['oracle_answer_available'] = True
        for prefix, candidate in [('best', best), ('worst', worst)]:
            ties = [c['configuration_id'] for c in valid if abs(c['total_time_min']-candidate['total_time_min']) <= TOLERANCE]
            row[f'{prefix}_tied_configurations'] = ties
            row[f'{prefix}_tie_count'] = len(ties)
        row['worst_minus_best_min'] = worst['total_time_min']-best['total_time_min']
        if current and current['valid']:
            gap = current['total_time_min']-best['total_time_min']
            if gap < -TOLERANCE:
                raise AssertionError('Current is below minimum of same reference candidates')
            row.update(time_gap_min=max(0., gap), target_log1p_time_gap=math.log1p(max(0., gap)), time_gap_available=True)
    else:
        reasons.append('no_valid_candidate_in_medium')
    row['exclusion_reasons'] = reasons
    row['training_eligible'] = not reasons and row['time_gap_available']
    return row


def csv_row(row):
    return {k: json.dumps(v, ensure_ascii=False, separators=(',', ':')) if isinstance(v, (list, dict)) else v
            for k, v in row.items()}


def build(source_dir, output_dir, *, include_flagged=False):
    bundles = load_bundles(source_dir)
    bank = make_rate_bank(bundles, include_flagged=include_flagged)
    if len(COLUMNS) != len(set(COLUMNS)):
        raise AssertionError('Duplicate output columns')
    source_paths = [source_dir/'selected_runs_by_database_cell.csv', source_dir/'forward_discharge_rate_run_drone.csv']
    before_hashes = {str(p.resolve()): sha256(p) for p in source_paths}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir/'rate_reference.json').write_text(json.dumps(list(bank.values()), ensure_ascii=False, indent=2)+'\n')
    counts, exclusions = Counter(), Counter()
    answers = []
    audit_columns = ['scenario_id', 'configuration_id', 'formation', 'spacing_cm', 'positions',
                     'assigned_Bideal_rates', 'physical_rates', 'arrival_soc', 'rate_source_run_count',
                     'rate_metadata_warning_count', 'valid', 'exclusion_reason', 'charging_pad_count',
                     'charging_times_min', 'charging_pads', 'charging_completion_time_min', 'total_time_min']
    with (gzip.open(output_dir/'candidate_search_audit.csv.gz', 'wt', newline='', encoding='utf-8') as audit,
          (output_dir/'all_state_answers.csv').open('w', newline='', encoding='utf-8-sig') as all_stream,
          (output_dir/'labelled_training_records.csv').open('w', newline='', encoding='utf-8-sig') as train_stream):
        aw = csv.DictWriter(audit, fieldnames=audit_columns)
        aw.writeheader()
        writers = [csv.DictWriter(stream, fieldnames=COLUMNS) for stream in (all_stream, train_stream)]
        for writer in writers:
            writer.writeheader()
        for index, b in enumerate(bundles, 1):
            by_k = evaluate_run(b, bank)
            for k, candidates in by_k.items():
                row = answer_row(b, bank, k, candidates, include_flagged=include_flagged)
                writers[0].writerow(csv_row(row))
                answers.append(row)
                counts['all_state_rows'] += 1
                for flag in ('oracle_answer_available', 'time_gap_available', 'source_trial_time_available', 'training_eligible', 'covers_all_safe_structures', 'reference_metadata_warning'):
                    counts[flag] += bool(row[flag])
                if row['training_eligible']:
                    writers[1].writerow(csv_row(row))
                else:
                    exclusions.update(row['exclusion_reasons'])
                counts['candidate_evaluations'] += len(candidates)
                counts['valid_candidate_evaluations'] += row['candidate_count_valid']
                for candidate in candidates:
                    aw.writerow(csv_row(dict(candidate, scenario_id=row['scenario_id'])))
            if index % 20 == 0:
                print(f'Completed {index}/{len(bundles)} source runs', flush=True)
    assert before_hashes == {str(p.resolve()): sha256(p) for p in source_paths}, 'Source changed during build'
    assert len({r['scenario_id'] for r in answers}) == len(bundles)*5 == len(answers)
    manifest = {
        'status': 'pooled_reference_oracle_answers_generated_no_model_trained',
        'source_files_sha256': before_hashes,
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'ml_policy/build_real_medium_training_labels.py', ROOT/'ml_policy/charging_model.py', ROOT/'ml_policy/oracle_optimizer.py']},
        'source_run_count': len(bundles), 'source_drone_rows': 5*len(bundles),
        'quality_policy': 'safety_only_keep_qc_warnings' if include_flagged else 'exclude_qc_flagged_sources',
        'rate_reference_runs': sum(c['source_run_count'] for c in bank.values()),
        'clean_rate_reference_runs': sum(c['source_run_count']-c['qc_flagged_source_count'] for c in bank.values()),
        'qc_flagged_reference_runs': sum(c['qc_flagged_source_count'] for c in bank.values()),
        'rate_reference_cells': len(bank), 'counts': dict(counts), 'exclusion_counts_nonexclusive': dict(exclusions),
        'feature_names': FEATURES, 'recommended_regression_target': 'target_log1p_time_gap',
        'target_definition': 'log1p(current_total_time_min - best_total_time_min), NOT scheduling lower-bound residual',
        'best_label_definition': 'all best_tied_configurations; one deterministic representative in best_configuration',
        'time_units': 'minutes', 'position_definition': 'array index=drone1..5, value=assigned slot1..5, NOT mission pad ID',
        'assumptions': {'flight_seconds': FLIGHT_SECONDS, 'remaining_distance_cm': 250, 'speed_cm_per_s': 10,
                        'medium_soc_interval': [40, 75], 'charging_target_soc': FULLY_CHARGED_SOC,
                        'zero_to_target_charge_minutes': ZERO_TO_FULLY_CHARGED_MINUTES,
                        'rate_estimator': 'per-slot equal-weight mean over complete selected source runs admitted by quality_policy in same wind/level/formation/spacing',
                        'battery_transfer': 'physical_rate = assigned_Bideal_rate / own_battery_scale',
                        'scheduling': 'exact non-preemptive identical-pad makespan; all pads initially available; no charge switching/setup overhead',
                        'mission': '25/60 + charging makespan; no formation-change travel overhead'},
        'validation_scope': 'Calculation labels for current pooled reference only; not held-out accuracy evidence.',
        'split_warning': 'Do not randomly split K variants or candidate audit rows. Group original trial/session. Reference currently includes all selected runs; refit reference/calibration on training-only data before independent real-data validation. Synthetic validation using this same reference only checks agreement with this oracle.',
        'limitations': [
            '134 selected processed medium forward runs are the input scope, not all raw flights or wind-tunnel/high/low experiments.',
            'Missing/unsafe structures are excluded explicitly. Best/worst are only over supported feasible configurations.',
            'QC flags are warnings only under safety_only_keep_qc_warnings; conservative mode excludes them from references and training. Numerical/domain checks remain in both modes.',
            'Retaining a flagged measurement does not establish its accuracy. Zero fitted SOC slopes are retained as estimates in safety-only mode, not proof of zero physical consumption.',
            'Current pooled time and source-trial time are different estimates; time_gap uses only pooled time to ensure fair comparisons.',
            'Metadata mismatches are retained as warnings; labels are provisional pending source reconciliation.',
            'Single-run rate cells are retained as in the approved example; uncertainty is not resolved by enumeration.',
            'Position-transfer assumptions and shared exponential charging law are model assumptions, not measured swapped flights/charging times.',
            'Existing calibration extends below medium; source forward clocks were individually cleaned, not certified common 25-second measured windows.',
        ],
    }
    (output_dir/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--include-flagged', action='store_true',
                        help='Keep all non-unsafe selected sources; QC flags are warnings, not exclusions. Medium-domain checks still apply.')
    args = parser.parse_args()
    result = build(args.source_dir, args.output_dir, include_flagged=args.include_flagged)
    print(json.dumps(result['counts'], indent=2))
