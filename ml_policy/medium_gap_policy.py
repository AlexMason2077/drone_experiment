"""26-feature shortlist scorer. No drone control and no full oracle at inference."""
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from ml_policy.build_real_medium_training_labels import (
    FEATURES, FORMATIONS, WINDS, cost_values, mission_values,
)
from ml_policy.oracle_optimizer import UNSAFE_STRUCTURES_BY_CONDITION


def candidates_for_state(state, reference):
    """Only rate lookup, battery conversion, and domain filtering; no charging costs."""
    wind, level, k = state['wind'], state['level'], state['k']
    soc, scales = state['soc'], state['scales']
    if wind not in WINDS or level not in (1, 2) or k not in range(1, 6):
        raise ValueError('Unsupported wind/level/charging-pad count')
    if len(soc) != 5 or len(scales) != 5 or not all(math.isfinite(s) and 40 <= s <= 75 for s in soc):
        raise ValueError('Five SOC values within medium 40..75 are required')
    prefix = [int(wind == w) for w in WINDS] + [level, k/5] + [s/100 for s in soc]
    candidates, features = [], []
    for cell in reference:
        if (cell['wind'], cell['level']) != (wind, level):
            continue
        f, spacing = cell['formation'], cell['spacing']
        if ('echalon' if f == 'echelon' else f, spacing) in UNSAFE_STRUCTURES_BY_CONDITION.get((wind, level), ()):
            continue
        for positions in itertools.permutations(range(5)):
            assigned, physical, arrival = mission_values(soc, cell['rates'], scales, positions)
            if not all(40-1e-9 <= s <= 75+1e-9 for s in arrival):
                continue
            config = f'{f}_{spacing}__p' + '-'.join(str(p+1) for p in positions)
            features.append(prefix + [int(f == name) for name in FORMATIONS] + [spacing/75] + assigned + scales)
            candidates.append({'configuration': config, 'formation': f, 'spacing_cm': spacing,
                               'positions': [p+1 for p in positions], 'arrival_soc': arrival})
    return candidates, np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURES))


def exact_costs(state, candidates):
    return np.asarray([cost_values(c['arrival_soc'], state['k'])[2]['total_required_time_min'] for c in candidates])


def normalize(x, scaler):
    return ((x-np.asarray(scaler['mean'], dtype=np.float32))/np.asarray(scaler['scale'], dtype=np.float32)).astype(np.float32)


def load_policy(directory):
    import tensorflow as tf
    directory = Path(directory)
    manifest = json.loads((directory/'training_manifest.json').read_text())
    if manifest['feature_names'] != FEATURES or manifest['target'] != 'target_log1p_time_gap':
        raise ValueError('Incompatible feature order/target')
    return {
        'model': tf.keras.models.load_model(directory/'medium_gap_ranker.keras', compile=False),
        'scaler': json.loads((directory/'scaler.json').read_text()),
        'reference': json.loads((directory/'rate_reference.json').read_text()),
        'manifest': manifest,
    }


def shortlist(policy, state, top_n=20):
    """Score all supported candidates, then exact-evaluate ONLY the shortlist."""
    if top_n < 1:
        raise ValueError('top_n must be positive')
    started = time.perf_counter()
    candidates, x = candidates_for_state(state, policy['reference'])
    if not candidates:
        return {'status': 'no_feasible_candidate', 'candidates': [], 'candidate_count': 0}
    scores = policy['model'](normalize(x, policy['scaler']), training=False).numpy().reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite model prediction')
    chosen = np.argsort(scores, kind='stable')[:top_n]
    rows = []
    for i in chosen:
        c = candidates[int(i)]
        costs = cost_values(c['arrival_soc'], state['k'])[2]
        rows.append(dict(c, predicted_log1p_gap=float(scores[i]),
                         exact_total_time_min=costs['total_required_time_min']))
    rows.sort(key=lambda r: (r['exact_total_time_min'], r['configuration']))
    return {'status': 'ok', 'candidate_count': len(candidates), 'exact_evaluations': len(rows),
            'candidates': rows, 'best_in_shortlist': rows[0],
            'elapsed_seconds': time.perf_counter()-started,
            'warning': 'Best within shortlist only; no global-optimum guarantee or flight command.'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--state-json', type=Path, required=True)
    parser.add_argument('--top-n', type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(shortlist(load_policy(args.model_dir), json.loads(args.state_json.read_text()), args.top_n), indent=2))
