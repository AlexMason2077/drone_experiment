"""Audit and explain the frozen real-data Medium decisions without changing them.

Run: python3 -m output_py.interpret_medium_decisions
Resampling is a sensitivity analysis, NOT new flight data or an ML validation set.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_results/medium_configuration_function_v1_20260909"
RAW = ROOT / "analysis_results/medium_forward_bideal_v3_with_b15_20260909/run_drone_rates.csv"
OUT = ROOT / "analysis_results/medium_decision_interpretability_20260909"
SEED = 20260909
REPS = 2000
KEYS = ["wind_direction", "wind_level", "formation", "inter_drone_spacing_cm"]
RATE_COLS = [f"position_{i}_discharge_rate_Bideal_pp_min" for i in range(1, 6)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partitions(pads):
    """All unlabeled set partitions of five jobs onto at most pads machines."""
    result = []
    for assignment in itertools.product(range(pads), repeat=5):
        if assignment[0] != 0:
            continue
        if any(assignment[i] > 1 + max(assignment[:i]) for i in range(1, 5)):
            continue
        result.append([[int(assignment[i] == p) for i in range(5)] for p in range(pads)])
    return np.asarray(result, dtype=float)


PARTITIONS = {k: partitions(k) for k in range(1, 6)}


def jobs(rates, model="exponential", target=99):
    arrival = 75 - np.asarray(rates) * 25 / 60
    if model == "exponential":
        return 5400 / np.log(100) * np.log((100 - arrival) / (100 - target))
    if model == "constant_current":
        return (target - arrival) * 5400 / 99
    raise ValueError(model)


def scores(q, pads):
    return np.einsum("...i,api->...ap", q, PARTITIONS[pads]).max(axis=-1).min(axis=-1) + 25


def name(formation, spacing):
    return f"{'echelon' if formation == 'echalon' else formation} {int(spacing)} cm"


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "explanations.json").exists():
        raise FileExistsError("Existing completed interpretation; do not overwrite")
    frozen_hashes = {p.name: digest(p) for p in BASE.iterdir() if p.is_file()}
    frozen = json.loads((BASE / "decision_function.json").read_text())
    pre = pd.read_csv(BASE / "preprocessed_75pct_250cm.csv")
    raw = pd.read_csv(RAW)
    raw = raw[raw.status.eq("included")].copy()
    raw["trial_key"] = raw.experiment_directory + "/" + raw.run_id
    rng = np.random.default_rng(SEED)
    explanations, resampling_rows, loo_rows, charge_rows, spacing_rows = {}, [], [], [], []
    checked = 0
    for wind, level in itertools.product(("head", "side", "tail"), (1, 2)):
        subset = pre[pre.wind_direction.eq(wind) & pre.wind_level.eq(level)].sort_values(KEYS).reset_index(drop=True)
        rates = subset[RATE_COLS].to_numpy(float)
        q = jobs(rates)
        names = [name(r.formation, r.inter_drone_spacing_cm) for r in subset.itertuples()]
        cubes, trial_matrices = [], []
        for r in subset.itertuples():
            source = raw[raw.wind_direction.eq(wind) & raw.wind_level.eq(level) & raw.formation.eq(r.formation) & raw.inter_drone_spacing_cm.eq(r.inter_drone_spacing_cm)]
            mat = source.pivot(index="trial_key", columns="position", values="discharge_rate_Bideal_pp_per_min").reindex(columns=range(1, 6))
            vals = mat.to_numpy(float)
            assert np.allclose(np.nanmean(vals, axis=0), np.asarray([getattr(r, c) for c in RATE_COLS]), atol=1e-9)
            trial_matrices.append(mat)
            draws = vals[rng.integers(0, len(vals), size=(REPS, len(vals)))]
            count = np.sum(np.isfinite(draws), axis=1)
            cube = np.divide(np.nansum(draws, axis=1), count, out=np.full((REPS, 5), np.nan), where=count > 0)
            cubes.append(cube)
        bootstrap_rates = np.stack(cubes, axis=1)
        valid = np.isfinite(bootstrap_rates).all(axis=(1, 2))
        bootstrap_jobs = jobs(bootstrap_rates[valid])
        for k in range(1, 6):
            key = f"{k}|{wind}|{level}"
            d = frozen["decisions"][key]
            total = scores(q, k)
            order = np.argsort(total, kind="stable")
            best, second = map(int, order[:2])
            b, r = subset.iloc[best], subset.iloc[second]
            assert b.formation == d["configuration"]["formation"]
            assert int(b.inter_drone_spacing_cm) == d["configuration"]["inter_drone_spacing_cm"]
            assert abs(float(total[best]) - d["total_time_seconds"]) < 1e-7
            assert abs(float(total[second]-total[best])-d["gap_to_runner_up_seconds"]) < 1e-7
            ordered = np.sort(q, axis=1)
            compact = {1: ordered.sum(1), 2: ordered[:, :3].sum(1),
                       3: np.maximum(ordered[:, 0]+ordered[:, 3], ordered[:, 1]+ordered[:, 2]),
                       4: ordered[:, :2].sum(1), 5: ordered[:, 4]}[k] + 25
            assert np.allclose(compact, total, atol=1e-7)
            checked += len(total)
            boot_total = scores(bootstrap_jobs, k)
            retain = (boot_total[:, best] <= boot_total.min(1)+1e-7).mean()
            gap = boot_total[:, second]-boot_total[:, best]
            interval = np.quantile(gap, [.025, .975])
            sens = dict(key=key, bootstrap_attempts=REPS, bootstrap_valid=int(valid.sum()),
                        bootstrap_invalid=int((~valid).sum()), nominal_winner_selection_frequency=float(retain),
                        fixed_runner_up_gap_p025_s=float(interval[0]), fixed_runner_up_gap_p975_s=float(interval[1]),
                        interpretation="Conditional whole-trial bootstrap stability, not probability of true optimality; no calibration/fan/charging-model uncertainty")
            resampling_rows.append(sens)
            loo_valid, loo_retained, loo_unavailable, winner_deleted_valid, winner_deleted_retained = 0, 0, 0, 0, 0
            for ci, mat in enumerate(trial_matrices):
                vals = mat.to_numpy(float)
                for j, trial in enumerate(mat.index):
                    rest = np.delete(vals, j, axis=0)
                    okay = len(rest) > 0 and np.isfinite(rest).any(axis=0).all()
                    record = dict(key=key, omitted_trial=trial, affected_configuration=names[ci], evaluable=bool(okay))
                    if not okay:
                        loo_unavailable += 1
                        record.update(retained=None, new_configuration=None, reason="At least one position has no remaining measured rate; comparison not computed")
                    else:
                        modified = q.copy()
                        modified[ci] = jobs(np.nanmean(rest, axis=0))
                        new_scores = scores(modified, k)
                        retained = bool(new_scores[best] <= new_scores.min()+1e-7)
                        loo_valid += 1
                        loo_retained += retained
                        winner_deleted_valid += ci == best
                        winner_deleted_retained += (ci == best and retained)
                        record.update(retained=retained, new_configuration=names[int(new_scores.argmin())], reason="Whole-flight deletion within this wind condition")
                    loo_rows.append(record)
            sens.update(loo_evaluable=loo_valid, loo_retained=loo_retained, loo_not_evaluable=loo_unavailable,
                        winner_trial_deletion_evaluable=winner_deleted_valid, winner_trial_deletion_retained=winner_deleted_retained)
            alternatives = {}
            for model, target in (("constant_current", 99), ("exponential", 80), ("exponential", 90)):
                alt_scores = scores(jobs(rates, model, target), k)
                alt = int(alt_scores.argmin())
                label = f"{model}_target_{target}"
                alternatives[label] = dict(configuration=names[alt], unchanged=bool(alt == best),
                                            regret_seconds=float(alt_scores[best]-alt_scores[alt]))
                charge_rows.append(dict(key=key, alternative=label, **alternatives[label]))
            ordered_positions = list(np.argsort(q[best])+1)
            ordered_runner_positions = list(np.argsort(q[second])+1)
            loads = d["charging_pad_loads_seconds"]
            bpad = int(np.argmax(loads))
            bottleneck = [int(x.split("_")[1]) for x in d["charging_pad_groups"][bpad]]
            # Use the exact stored optimal schedule, never a guessed geometric leader.
            bottleneck_text = ", ".join(f"P{i}" for i in bottleneck)
            driver = {
                1: "the sum of all five charging times",
                2: "the combined charging time of the three shortest charging jobs",
                3: "the larger of the two optimally paired charging workloads, with the longest job charged alone",
                4: "the combined charging time of the two shortest jobs sharing a pad",
                5: "the longest individual charging time",
            }[k]
            data_en = (f"With {k} available charging pad{'s' if k != 1 else ''}, {names[best]} gives the lowest computed total time "
                       f"({total[best]/60:.2f} min), {total[second]-total[best]:.2f} s below {names[second]}. "
                       f"It minimizes {driver}; the limiting pad serves {bottleneck_text} and finishes charging after {loads[bpad]/60:.2f} min.")
            data_zh = (f"{k}个充电位时选{name(b.formation,b.inter_drone_spacing_cm)}：计算总时间{total[best]/60:.2f}分钟，"
                       f"比第二名{names[second]}少{total[second]-total[best]:.2f}秒；最后完成的充电位承担{bottleneck_text}，"
                       f"该位充电负担为{loads[bpad]/60:.2f}分钟。")
            caveats = []
            if total[second]-total[best] < 10:
                caveats.append("The numerical margin is below 10 s; this is a small-gap flag, not a significance test.")
            min_n = min(int(b[f"position_{i}_run_count"]) for i in range(1, 6))
            if min_n < 2:
                caveats.append("At least one selected position has only one usable trial; its variability is not estimable here.")
            if d["flat_trace_position_records"]:
                caveats.append(f"The selected profile includes {d['flat_trace_position_records']} flat-SOC trace(s); an integer gauge plateau does not imply zero physical energy use.")
            if d["missing_safe_configurations"]:
                caveats.append("Column 50 cm has no eligible real profile in this wind condition and is not ranked.")
            caveats.append(f"Whole-trial resampling retained this numerical winner in {retain:.1%} of {int(valid.sum())} evaluable draws; this is not a confidence probability.")
            same = subset.index[(subset.formation == b.formation) & (subset.inter_drone_spacing_cm != b.inter_drone_spacing_cm)].tolist()
            same_spacing = None
            if same:
                si = same[0]
                same_spacing = dict(configuration=names[si], gap_to_selected_seconds=float(total[si]-total[best]),
                                    rates_Bideal_pp_min=rates[si].tolist())
            explanations[key] = dict(
                input=dict(charging_pad_availability=k, wind_direction=wind, wind_strength="low" if level == 1 else "high"),
                legacy_wind_level=level, configuration=d["configuration"], paper_configuration=names[best],
                paper_wind=f"{'low' if level == 1 else 'high'} {wind}wind", data_explanation_en=data_en,
                data_explanation_zh=data_zh, total_time_minutes=float(total[best]/60),
                gap_to_runner_up_seconds=float(total[second]-total[best]), runner_up_configuration=names[second],
                decision_driver=driver, bottleneck_positions=bottleneck, charging_pad_groups=d["charging_pad_groups"],
                charging_pad_loads_seconds=loads, position_rates_Bideal_pp_min=rates[best].tolist(),
                runner_up_position_rates_Bideal_pp_min=rates[second].tolist(),
                position_charging_seconds=q[best].tolist(), runner_up_position_charging_seconds=q[second].tolist(),
                arrival_soc_Bideal=d["arrival_soc_by_position"], lowest_arrival_soc_position=int(np.argmax(q[best])+1),
                position_order_shortest_to_longest=[int(i) for i in ordered_positions],
                runner_up_position_order_shortest_to_longest=[int(i) for i in ordered_runner_positions],
                position_run_counts=[int(b[f"position_{i}_run_count"]) for i in range(1, 6)],
                candidate_count=len(subset), caveats=caveats, sensitivity=sens, charging_model_sensitivity=alternatives,
                same_formation_other_spacing=same_spacing,
                inference_boundary="Observed rate profile plus assumed charging model; aerodynamic cause not identified by this optimization")
        for formation in subset.formation.unique():
            pair = subset[subset.formation.eq(formation)].set_index("inter_drone_spacing_cm")
            if set(pair.index) != {50, 75}:
                continue
            a, b = pair.loc[50, RATE_COLS].to_numpy(float), pair.loc[75, RATE_COLS].to_numpy(float)
            spacing_rows.append(dict(wind_direction=wind, wind_strength="low" if level == 1 else "high", formation=formation,
                                     mean_rate_50=float(a.mean()), mean_rate_75=float(b.mean()),
                                     max_rate_50=float(a.max()), max_rate_75=float(b.max()),
                                     lower_mean_at_75=bool(b.mean()<a.mean()), lower_max_at_75=bool(b.max()<a.max())))
    assert checked == 275
    assert frozen_hashes == {p.name: digest(p) for p in BASE.iterdir() if p.is_file()}
    result = dict(version=1, kind="medium_decision_interpretation", source_sha256=frozen_hashes,
                  raw_rate_sha256=digest(RAW), seed=SEED, bootstrap_attempts_per_wind_condition=REPS,
                  wind_label_mapping={"1": "low", "2": "high"}, decisions=explanations)
    (OUT / "numeric_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    pd.DataFrame(resampling_rows).to_csv(OUT / "resampling_sensitivity.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(OUT / "leave_one_trial_out.csv", index=False)
    pd.DataFrame(charge_rows).to_csv(OUT / "charging_model_sensitivity.csv", index=False)
    pd.DataFrame(spacing_rows).to_csv(OUT / "spacing_comparisons.csv", index=False)
    summary = dict(decisions=len(explanations), all_candidate_schedules_checked=checked,
                   bootstrap_retention_min=min(e["sensitivity"]["nominal_winner_selection_frequency"] for e in explanations.values()),
                   bootstrap_retention_max=max(e["sensitivity"]["nominal_winner_selection_frequency"] for e in explanations.values()),
                   states_where_loo_can_change_winner=sum(e["sensitivity"]["loo_retained"]<e["sensitivity"]["loo_evaluable"] for e in explanations.values()),
                   small_gap_states=[key for key,e in explanations.items() if e["gap_to_runner_up_seconds"]<10],
                   charging_model_changed=pd.DataFrame(charge_rows).groupby("alternative").unchanged.apply(lambda s:int((~s).sum())).to_dict(),
                   source_files_unchanged=True)
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
