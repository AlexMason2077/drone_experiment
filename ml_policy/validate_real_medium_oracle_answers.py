"""Reconcile every answer against its candidate audit, with independent costs."""
import argparse
import csv
import gzip
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from ml_policy.build_real_medium_training_labels import FEATURES, sha256, truth
from ml_policy.oracle_optimizer import UNSAFE_STRUCTURES_BY_CONDITION


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-8):
        raise AssertionError((a, b))


def exhaustive_schedule(jobs, k):
    if k == 1:
        return sum(jobs)
    if k == 5:
        return max(jobs)
    # Independent labelled assignment enumeration, unlike the production
    # solver's unlabelled set partitions.
    best = math.inf
    for assignment in itertools.product(range(k), repeat=5):
        loads = [0.] * k
        for q, pad in zip(jobs, assignment):
            loads[pad] += q
        best = min(best, max(loads))
    return best


def validate(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    for path, expected in manifest['source_files_sha256'].items():
        assert sha256(Path(path)) == expected, f'Source changed: {path}'
    answers = rows(directory/'all_state_answers.csv')
    answer_map = {r['scenario_id']: r for r in answers}
    assert len(answer_map) == len(answers) == 5*manifest['source_run_count']
    assert set(Counter(r['source_group_id'] for r in answers).values()) == {5}
    train = rows(directory/'labelled_training_records.csv')
    assert {r['scenario_id'] for r in train} == {r['scenario_id'] for r in answers if truth(r['training_eligible'])}
    assert all(r == answer_map[r['scenario_id']] for r in train)
    bank = {(r['wind'], r['level'], r['formation'], r['spacing']): r
            for r in json.loads((directory/'rate_reference.json').read_text())}
    source_flags = {r['source_group_id']: json.loads(r['source_qc_flags']) for r in answers}
    included_sources = []
    for key, cell in bank.items():
        f = 'echalon' if key[2] == 'echelon' else key[2]
        assert (f, key[3]) not in UNSAFE_STRUCTURES_BY_CONDITION.get(key[:2], ())
        included_sources.extend(cell['source_groups'])
        assert cell['source_run_count'] == len(cell['source_groups'])
        if manifest.get('quality_policy') != 'safety_only_keep_qc_warnings':
            assert all(not source_flags[s] for s in cell['source_groups'])
    assert len(included_sources) == len(set(included_sources))
    if manifest.get('quality_policy') == 'safety_only_keep_qc_warnings':
        expected_sources = set()
        for a in answers:
            f = 'echalon' if a['current_formation'] == 'echelon' else a['current_formation']
            key = (a['wind_direction'], int(a['wind_level']))
            if (f, int(a['current_spacing_cm'])) not in UNSAFE_STRUCTURES_BY_CONDITION.get(key, ()):
                expected_sources.add(a['source_group_id'])
        assert set(included_sources) == expected_sources
    checked = 0
    valid_checked = 0
    seen = set()
    with gzip.open(directory/'candidate_search_audit.csv.gz', 'rt', newline='', encoding='utf-8') as stream:
        for scenario, group in itertools.groupby(csv.DictReader(stream), key=lambda r: r['scenario_id']):
            assert scenario not in seen
            seen.add(scenario)
            answer = answer_map[scenario]
            candidates = list(group)
            assert len(candidates) == int(answer['candidate_count_evaluated'])
            assert len({c['configuration_id'] for c in candidates}) == len(candidates)
            assert len(candidates) == len(json.loads(answer['supported_structures']))*120
            valid = []
            for c in candidates:
                checked += 1
                assert c['charging_pad_count'] == answer['charging_pad_count']
                p = json.loads(c['positions'])
                assert sorted(p) == [1, 2, 3, 4, 5]
                cell = bank[(answer['wind_direction'], int(answer['wind_level']), c['formation'], int(c['spacing_cm']))]
                assigned, physical, arrival = [json.loads(c[f]) for f in ('assigned_Bideal_rates', 'physical_rates', 'arrival_soc')]
                for i in range(5):
                    close(assigned[i], cell['rates'][p[i]-1])
                    close(physical[i], assigned[i]/float(answer[f'battery_scale_d{i+1}']))
                    close(arrival[i], float(answer[f'soc_d{i+1}'])-physical[i]*25/60)
                in_domain = all(40-1e-9 <= s <= 75+1e-9 for s in arrival)
                assert truth(c['valid']) == in_domain
                if not in_domain:
                    assert c['total_time_min'] == '' and json.loads(c['charging_times_min']) == []
                    continue
                valid_checked += 1
                valid.append(c)
                jobs, pads = json.loads(c['charging_times_min']), json.loads(c['charging_pads'])
                for q, s in zip(jobs, arrival):
                    close(q, 90/math.log(100)*math.log(100-s))
                k = int(c['charging_pad_count'])
                assert all(1 <= pad <= k for pad in pads)
                loads = [sum(q for q, pad in zip(jobs, pads) if pad == j) for j in range(1, k+1)]
                close(c['charging_completion_time_min'], max(loads))
                close(c['total_time_min'], max(loads)+25/60)
            assert len(valid) == int(answer['candidate_count_valid'])
            assert len(candidates)-len(valid) == int(answer['candidate_count_outside_medium'])
            if not valid:
                assert not truth(answer['oracle_answer_available'])
                continue
            costs = [float(c['total_time_min']) for c in valid]
            for prefix, extreme in [('best', min(costs)), ('worst', max(costs))]:
                close(answer[f'{prefix}_total_time_min'], extreme)
                selected = next(c for c in valid if c['configuration_id'] == answer[f'{prefix}_configuration'])
                close(selected['total_time_min'], extreme)
                jobs = json.loads(selected['charging_times_min'])
                close(extreme, exhaustive_schedule(jobs, int(answer['charging_pad_count']))+25/60)
                ties = {c['configuration_id'] for c in valid if abs(float(c['total_time_min'])-extreme) <= 1e-9}
                assert ties == set(json.loads(answer[f'{prefix}_tied_configurations']))
                assert len(ties) == int(answer[f'{prefix}_tie_count'])
            close(answer['worst_minus_best_min'], max(costs)-min(costs))
            if truth(answer['time_gap_available']):
                current = next(c for c in valid if c['configuration_id'] == answer['current_configuration'])
                close(answer['current_total_time_min'], current['total_time_min'])
                close(answer['time_gap_min'], float(current['total_time_min'])-min(costs))
                close(answer['time_gap_min'], math.expm1(float(answer['target_log1p_time_gap'])))
            else:
                assert answer['time_gap_min'] == answer['target_log1p_time_gap'] == ''
            if truth(answer['training_eligible']):
                assert not json.loads(answer['exclusion_reasons'])
                if manifest.get('quality_policy') != 'safety_only_keep_qc_warnings':
                    assert not json.loads(answer['source_qc_flags'])
                assert all(math.isfinite(float(answer[f])) for f in FEATURES)
                assert float(answer['time_gap_min']) >= 0
    assert seen == set(answer_map)
    assert checked == manifest['counts']['candidate_evaluations']
    assert valid_checked == manifest['counts']['valid_candidate_evaluations']
    for field in ('oracle_answer_available', 'time_gap_available', 'source_trial_time_available', 'training_eligible', 'covers_all_safe_structures', 'reference_metadata_warning'):
        assert sum(truth(r[field]) for r in answers) == manifest['counts'][field]
    sample = answer_map['front_50_tail_lv2_new_002::20260519_154736::K2']
    if manifest.get('quality_policy') != 'safety_only_keep_qc_warnings':
        close(sample['best_total_time_min'], 215.2985459150669)
        close(sample['worst_total_time_min'], 222.28437476560518)
        close(sample['current_total_time_min'], 219.8792700970747)
    close(sample['source_trial_total_time_min'], 220.58753264261506)
    report = {
        'calculation_checks': 'passed', 'source_files_unchanged': True,
        'answer_rows_checked': len(answers), 'training_rows_checked': len(train),
        'candidate_rows_checked': checked, 'valid_candidate_rows_checked': valid_checked,
        'checks': ['Unique run/K grain and exactly five scenarios per run',
                   'Reference source inclusion matches selected quality policy and excludes unsafe condition cells',
                   'Source hashes and exact training-subset equality',
                   'Every candidate: pooled slot rate, own battery factor, arrival SOC, charging law, pad loads, total time',
                   'Every scenario: audit minimum/maximum, current cost, gap, all tied extremes',
                   'Every best/worst: independent exhaustive labelled charging-pad scheduling',
                   'Original source-trial numerical example reproduced; pooled-label golden values checked only for original conservative policy'],
        'assessment': 'Share with caveats: exact within current pooled model, not independent validation or empirical optimality',
        'not_validated': ['Real-world position transfer', 'Charging model calibration',
                          'Metadata mismatch resolution', 'Held-out predictive accuracy',
                          'Missing rate cells or single-run uncertainty'],
    }
    (directory/'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), ensure_ascii=False, indent=2))
