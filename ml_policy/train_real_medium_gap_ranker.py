"""Train only on 670 observed-configuration rows; generated states are evaluation-only."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd
import tensorflow as tf

from ml_policy.build_real_medium_training_labels import FEATURES, ROOT, sha256
from ml_policy.medium_gap_policy import candidates_for_state, exact_costs, normalize, load_policy, shortlist

SEED = 20260907
TOP_NS = [1, 3, 5, 10, 20, 50, 100]


def make_model(seed):
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input((26,), name='state_and_candidate')
    x = inp
    for i, width in enumerate((256, 256, 128), 1):
        x = tf.keras.layers.Dense(width, activation=tf.nn.gelu, name=f'dense_{i}')(x)
        x = tf.keras.layers.LayerNormalization(name=f'norm_{i}')(x)
        x = tf.keras.layers.Dropout(.08, name=f'dropout_{i}')(x)
    out = tf.keras.layers.Dense(1, activation='softplus', name='predicted_log1p_gap')(x)
    model = tf.keras.Model(inp, out, name='real_medium_gap_ranker')
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=7e-4, weight_decay=1e-4),
                  loss=tf.keras.losses.Huber(delta=.10))
    return model


def generated_states(df, count_per_wind, split, seed):
    rng = np.random.default_rng(seed)
    states = []
    for wind in ('head', 'side', 'tail'):
        for level in (1, 2):
            observed = df[(df.wind_direction == wind) & (df.wind_level == level)].drop_duplicates('source_group_id')
            if observed.empty:
                continue
            for i in range(count_per_wind):
                b = observed.iloc[int(rng.integers(len(observed)))]
                # Independent generated SOC, not a new real flight. All K copies
                # of this base state stay in the same synthetic split.
                soc = rng.uniform(45., 75., size=5).tolist()
                for k in range(1, 6):
                    states.append({'id': f'{split}_{wind}_{level}_{i}_K{k}',
                                   'base_state_id': f'{split}_{wind}_{level}_{i}',
                                   'wind': wind, 'level': level, 'k': k, 'soc': soc,
                                   'scales': [float(b[f'battery_scale_d{j}']) for j in range(1, 6)],
                                   'battery_ids': [b[f'battery_id_d{j}'] for j in range(1, 6)],
                                   'battery_parameter_source_group': b['source_group_id'], 'source_type': 'rate_generated'})
    return states


def prepare_calibration(states, reference, seed):
    rng = np.random.default_rng(seed)
    xs, ys, index = [], [], []
    for n, s in enumerate(states, 1):
        candidates, x = candidates_for_state(s, reference)
        costs = exact_costs(s, candidates)
        if not len(costs):
            continue
        order = np.argsort(costs, kind='stable')
        selected = np.unique(np.r_[order[:8], rng.choice(len(costs), min(24, len(costs)), replace=False)])
        xs.append(x[selected])
        ys.append(np.log1p(np.maximum(costs[selected]-costs.min(), 0)).astype(np.float32))
        index += [{'state_id': s['id'], 'configuration': candidates[int(i)]['configuration']} for i in selected]
        if n % 100 == 0:
            print(f'Calibration oracle: {n}/{len(states)} states', flush=True)
    return np.concatenate(xs), np.concatenate(ys), index


def assess(model, scaler, states, reference):
    rows = []
    for n, s in enumerate(states, 1):
        candidates, x = candidates_for_state(s, reference)
        if not candidates:
            rows.append({'state_id': s['id'], 'status': 'no_feasible_candidate', 'k': s['k'],
                         'wind': s['wind'], 'level': s['level']})
            continue
        t = time.perf_counter()
        costs = exact_costs(s, candidates)
        oracle_seconds = time.perf_counter()-t
        t = time.perf_counter()
        predicted = model(normalize(x, scaler), training=False).numpy().ravel()
        prediction_seconds = time.perf_counter()-t
        if not np.isfinite(predicted).all():
            raise ValueError('Nonfinite predictions')
        order = np.argsort(predicted, kind='stable')
        best = float(costs.min())
        r = {'state_id': s['id'], 'status': 'ok', 'k': s['k'], 'wind': s['wind'], 'level': s['level'],
             'candidate_count': len(costs), 'oracle_best_time_min': best,
             'oracle_best_configuration': candidates[int(np.argmin(costs))]['configuration'],
             'full_oracle_seconds_excluding_features': oracle_seconds, 'nn_seconds_excluding_features': prediction_seconds,
             'top1_configuration': candidates[int(order[0])]['configuration']}
        for size in TOP_NS:
            selected = order[:size]
            regret = max(0., float(costs[selected].min())-best)
            r[f'hit_at_{size}'] = int(regret <= 1e-8)
            r[f'regret_at_{size}_min'] = regret
        target = np.log1p(np.maximum(costs-best, 0))
        r['log_gap_mae_all_candidates'] = float(np.mean(np.abs(predicted-target)))
        r['gap_mae_min_all_candidates'] = float(np.mean(np.abs(np.expm1(np.clip(predicted, 0, 30))-(costs-best))))
        rows.append(r)
        if n % 100 == 0:
            print(f'Shortlist assessment: {n}/{len(states)} states', flush=True)
    return pd.DataFrame(rows)


def summarize(frame):
    valid = frame[frame.status == 'ok']
    result = {'all_states': len(frame), 'evaluated_states': len(valid), 'no_feasible_states': len(frame)-len(valid),
              'candidate_count_mean': float(valid.candidate_count.mean())}
    result['top_n'] = {str(n): {'optimal_hit_rate': float(valid[f'hit_at_{n}'].mean()),
                              'mean_regret_min': float(valid[f'regret_at_{n}_min'].mean()),
                              'p95_regret_min': float(valid[f'regret_at_{n}_min'].quantile(.95)),
                              'max_regret_min': float(valid[f'regret_at_{n}_min'].max())} for n in TOP_NS}
    result['by_k'] = {str(k): {str(n): {'optimal_hit_rate': float(g[f'hit_at_{n}'].mean()),
                                      'mean_regret_min': float(g[f'regret_at_{n}_min'].mean())} for n in TOP_NS}
                      for k, g in valid.groupby('k')}
    result['by_wind_level'] = {f'{w}_{lv}': {str(n): float(g[f'hit_at_{n}'].mean()) for n in TOP_NS}
                              for (w, lv), g in valid.groupby(['wind', 'level'])}
    return result


class Progress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={logs['loss']:.5f}, calibration_loss={logs['val_loss']:.5f}", flush=True)


def train(source, output, epochs=80):
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    df = pd.read_csv(source/'labelled_training_records.csv')
    manifest = json.loads((source/'manifest.json').read_text())
    validation = json.loads((source/'validation_report.json').read_text())
    if validation['calculation_checks'] != 'passed' or manifest['feature_names'] != FEATURES:
        raise ValueError('Labels must be validated and match exact feature schema')
    if len(df) != 670 or not df.training_eligible.all() or df.scenario_id.duplicated().any():
        raise ValueError('Expected the approved 670 unique real-source scenarios')
    x = df[FEATURES].to_numpy(dtype=np.float32)
    y = df.target_log1p_time_gap.to_numpy(dtype=np.float32)
    if x.shape != (670, 26) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid input or label matrix')
    if not np.allclose(y, np.log1p(df.time_gap_min.to_numpy()), atol=1e-6):
        raise ValueError('Wrong target definition')
    output.mkdir(parents=True, exist_ok=False)
    df[FEATURES].to_csv(output/'X_train.csv', index=False)
    df[['target_log1p_time_gap']].to_csv(output/'y_train.csv', index=False)
    df[['scenario_id', 'source_group_id', 'current_configuration', 'current_total_time_min',
        'best_configuration', 'best_total_time_min', 'time_gap_min', 'source_qc_flags']].to_csv(output/'training_index.csv', index=False)
    mean, scale = x.mean(axis=0, dtype=np.float64), x.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1
    scaler = {'feature_names': FEATURES, 'mean': mean.tolist(), 'scale': scale.tolist(), 'fit_source': '670 training rows only'}
    (output/'scaler.json').write_text(json.dumps(scaler, indent=2)+'\n')
    reference = json.loads((source/'rate_reference.json').read_text())
    (output/'rate_reference.json').write_text(json.dumps(reference, indent=2)+'\n')
    cal = generated_states(df, 10, 'calibration', SEED+1)
    test = generated_states(df, 20, 'synthetic_test', SEED+2)
    assert {tuple(s['soc']) for s in cal}.isdisjoint({tuple(s['soc']) for s in test})
    real_soc = {tuple(r) for r in df[[f'soc_d{i}' for i in range(1, 6)]].to_numpy()}
    assert real_soc.isdisjoint({tuple(s['soc']) for s in cal+test})
    (output/'synthetic_calibration_states.json').write_text(json.dumps(cal, indent=2)+'\n')
    (output/'synthetic_test_states.json').write_text(json.dumps(test, indent=2)+'\n')
    vx, vy, vi = prepare_calibration(cal, reference, SEED+3)
    pd.DataFrame(vx, columns=FEATURES).to_csv(output/'X_calibration.csv.gz', index=False)
    pd.DataFrame({'target_log1p_time_gap': vy}).to_csv(output/'y_calibration.csv.gz', index=False)
    pd.DataFrame(vi).to_csv(output/'calibration_index.csv', index=False)
    model = make_model(SEED)
    print(f'Training {len(x)} rows x {x.shape[1]} features; {len(vx)} generated calibration candidates', flush=True)
    started = time.perf_counter()
    history = model.fit(normalize(x, scaler), y, validation_data=(normalize(vx, scaler), vy),
                        epochs=epochs, batch_size=64, verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True), Progress()])
    training_seconds = time.perf_counter()-started
    model.save(output/'medium_gap_ranker.keras')
    pd.DataFrame(history.history).rename_axis('epoch_zero_based').to_csv(output/'training_history.csv')
    # Freeze model before full calibration ranking and independent synthetic test.
    cal_result = assess(model, scaler, cal, reference)
    cal_result.to_csv(output/'calibration_state_results.csv', index=False)
    cal_metrics = summarize(cal_result)
    recommended = next((n for n in TOP_NS if cal_metrics['top_n'][str(n)]['optimal_hit_rate'] >= .90), None)
    chosen_n = recommended or 20
    test_result = assess(model, scaler, test, reference)
    test_result.to_csv(output/'synthetic_test_state_results.csv', index=False)
    train_predictions = model(normalize(x, scaler), training=False).numpy().ravel()
    pd.DataFrame({'scenario_id': df.scenario_id, 'target_log1p_gap': y, 'predicted_log1p_gap': train_predictions,
                  'predicted_gap_min': np.expm1(train_predictions)}).to_csv(output/'training_fit_predictions.csv', index=False)
    metrics = {'calibration': cal_metrics, 'synthetic_test': summarize(test_result),
               'calibration_top_n_target_hit_rate': .90, 'calibrated_top_n': recommended,
               'example_top_n': chosen_n, 'training_seconds': training_seconds,
               'epochs_completed': len(history.history['loss']),
               'best_epoch': int(np.argmin(history.history['val_loss']))+1,
               'training_gap_mae_min': float(np.mean(np.abs(np.expm1(train_predictions)-df.time_gap_min.to_numpy()))),
               'evaluation_scope': 'Unseen synthetic SOC states under shared pooled-rate/charging oracle, NOT independent real-flight validation',
               'status': 'pilot_trained_not_deployed', 'model_selection': 'Calibration only; synthetic test never used for fitting or selection'}
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    training_manifest = {'feature_names': FEATURES, 'input_width': 26, 'target': 'target_log1p_time_gap',
                         'source_training_rows': len(df), 'source_real_runs': df.source_group_id.nunique(),
                         'no_permutation_augmented_training_rows': True, 'generated_training_rows': 0,
                         'architecture': [26, 256, 256, 128, 1], 'activation': 'GELU, LayerNorm, Dropout0.08, Softplus output',
                         'optimizer': 'AdamW lr=0.0007 weight_decay=0.0001', 'loss': 'Huber delta=0.1',
                         'random_seed': SEED, 'epochs_maximum': epochs, 'calibration_base_states': len(cal)//5,
                         'test_base_states': len(test)//5, 'calibrated_top_n': recommended,
                         'sources': {str(p.resolve()): sha256(p) for p in [source/'labelled_training_records.csv', source/'rate_reference.json', source/'manifest.json']},
                         'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'ml_policy/medium_gap_policy.py']},
                         'limitations': ['Known shared reference model; synthetic test is not independent empirical evidence.',
                                         'Position permutations absent from training; candidate generalization must be judged by shortlist metrics.',
                                         'QC warnings retained per user policy; missing side-lv1-column50 remains unsupported.',
                                         'Single seed pilot, not stability study; no live drone/app integration.',
                                         'Do not add predicted gap to scheduling lower bound: this is regret-to-optimum, not the old LB residual.']}
    (output/'training_manifest.json').write_text(json.dumps(training_manifest, indent=2)+'\n')
    # Verify saved model/scaler and the inference wrapper actually round-trip.
    policy = load_policy(output)
    restored = policy['model'](normalize(x[:10], policy['scaler']), training=False).numpy().ravel()
    np.testing.assert_allclose(restored, train_predictions[:10], atol=1e-5)
    sample_row = df[df.scenario_id == 'front_50_tail_lv2_new_002::20260519_154736::K2'].iloc[0]
    state = {'wind': sample_row.wind_direction, 'level': int(sample_row.wind_level), 'k': 2,
             'soc': [float(sample_row[f'soc_d{i}']) for i in range(1, 6)],
             'scales': [float(sample_row[f'battery_scale_d{i}']) for i in range(1, 6)]}
    (output/'example_state.json').write_text(json.dumps(state, indent=2)+'\n')
    example = shortlist(policy, state, chosen_n)
    (output/'example_shortlist.json').write_text(json.dumps(example, indent=2)+'\n')
    assert example['exact_evaluations'] == min(chosen_n, example['candidate_count'])
    # End-to-end warm runtime, not just inference, on 20 pre-fixed test states.
    timings = []
    for s in test[:20]:
        t = time.perf_counter()
        cs, _ = candidates_for_state(s, reference)
        costs = exact_costs(s, cs)
        full_time = time.perf_counter()-t
        t = time.perf_counter()
        prediction = shortlist(policy, s, chosen_n)
        short_time = time.perf_counter()-t
        timings.append({'state_id': s['id'], 'full_seconds': full_time, 'shortlist_seconds': short_time,
                        'exact_candidates': prediction['exact_evaluations'], 'all_candidates': len(cs),
                        'selected_regret_min': prediction['best_in_shortlist']['exact_total_time_min']-float(costs.min())})
    pd.DataFrame(timings).to_csv(output/'runtime_comparison.csv', index=False)
    metrics['warm_runtime_20_states'] = {'full_median_seconds': float(np.median([t['full_seconds'] for t in timings])),
                                        'shortlist_median_seconds': float(np.median([t['shortlist_seconds'] for t in timings]))}
    metrics['saved_model_roundtrip_passed'] = True
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps({'output_dir': str(output.resolve()), **metrics}, indent=2), flush=True)
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=80)
    args = p.parse_args()
    train(args.source_dir, args.output_dir, args.epochs)
