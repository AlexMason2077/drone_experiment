"""Build traceable candidate labels from observed medium run bundles.

No synthetic telemetry or random SOC is used. Position permutations, K and
25-second mission costs are model-derived supervision, not measured flights.
This module has no TensorFlow dependency and never issues drone commands.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from ml_policy.charging_model import (
    FULLY_CHARGED_SOC,
    ZERO_TO_FULLY_CHARGED_MINUTES,
    exponential_charging_minutes,
)
from ml_policy.oracle_optimizer import UNSAFE_STRUCTURES_BY_CONDITION


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis_outputs/forward_discharge_rate_modeling"
FORMATIONS = ("column", "diamond", "echelon", "front", "vee")
WINDS = ("head", "side", "tail")
N = 5
FLIGHT_SECONDS = 25.0
TOLERANCE = 1e-9
FEATURES = (
    [f"wind_{w}" for w in WINDS]
    + ["wind_level", "pad_availability_ratio"]
    + [f"soc_d{i}_ratio" for i in range(1, 6)]
    + [f"formation_{f}" for f in FORMATIONS]
    + ["spacing_ratio"]
    + [f"assigned_Bideal_rate_d{i}" for i in range(1, 6)]
    + [f"battery_scale_d{i}" for i in range(1, 6)]
)
HELPERS = [
    "source_experiment", "source_run_id", "source_group_id", "scenario_id",
    "configuration_id", "formation", "wind_direction", "spacing_cm",
    "charging_pad_count", "remaining_distance_cm", "flight_time_seconds",
    "source_type", "label_source", "is_observed_position_assignment",
    "source_qc_flags", "metadata_warning", "label_available",
    "training_eligible", "exclusion_reasons", "analytic_k",
]
for i in range(1, 6):
    HELPERS += [
        f"battery_id_d{i}", f"soc_d{i}", f"position_d{i}",
        f"physical_rate_d{i}", f"arrival_soc_d{i}",
        f"charging_time_d{i}_min", f"charging_pad_d{i}",
    ]
TARGETS = [
    "charging_completion_time_min", "total_required_time_min",
    "cost_lower_bound_min", "cost_residual_min", "target_log1p_residual",
    "rank_within_source_structure", "regret_within_source_structure_min",
]
COLUMNS = HELPERS + FEATURES + TARGETS


def truth(value):
    return str(value).strip().lower() == "true"


def canonical_formation(value):
    return "echelon" if value in {"echalon", "echolon"} else value


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def partitions(n, k):
    """Unlabelled non-preemptive assignments to up to k identical pads."""
    if not 1 <= k <= n:
        raise ValueError("Expected 1 <= charging pads <= number of drones")

    def visit(prefix, max_label):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for pad in range(min(max_label + 1, k - 1) + 1):
            yield from visit(prefix + [pad], max(max_label, pad))

    return tuple(visit([0], 0))


def exact_schedule(jobs, k):
    if len(jobs) != N or not all(math.isfinite(q) and q >= 0 for q in jobs):
        raise ValueError("Expected five finite, nonnegative charging jobs")
    if not 1 <= k <= N:
        raise ValueError("K must be in 1..5")
    if k == 1:
        return sum(jobs), (0,) * N
    if k == N:
        return max(jobs), tuple(range(N))
    best = (math.inf, ())
    for assignment in partitions(N, k):
        loads = [0.0] * k
        for q, pad in zip(jobs, assignment):
            loads[pad] += q
        best = min(best, (max(loads), assignment))
    return best


def mission_values(soc, rates_by_slot, scales, positions):
    if sorted(positions) != list(range(N)):
        raise ValueError("Positions must be a permutation of 0..4")
    if not all(math.isfinite(a) and a > 0 for a in scales):
        raise ValueError("Battery scales must be finite and positive")
    if not all(math.isfinite(r) and r >= 0 for r in rates_by_slot):
        raise ValueError("Rates must be finite and nonnegative")
    assigned = [rates_by_slot[p] for p in positions]
    physical = [r / a for r, a in zip(assigned, scales)]
    arrival = [s - r * FLIGHT_SECONDS / 60 for s, r in zip(soc, physical)]
    return assigned, physical, arrival


def cost_values(arrival, k):
    if not all(40.0 - TOLERANCE <= s <= 75.0 + TOLERANCE for s in arrival):
        raise ValueError("Candidate exits the medium model range")
    jobs = [exponential_charging_minutes(s) for s in arrival]
    charging, pads = exact_schedule(jobs, k)
    flight_minutes = FLIGHT_SECONDS / 60
    total = flight_minutes + charging
    lower = flight_minutes + max(max(jobs), sum(jobs) / k)
    residual = total - lower
    if residual < -TOLERANCE:
        raise AssertionError("Exact cost is below its scheduling lower bound")
    residual = max(0.0, residual)
    return jobs, pads, {
        "charging_completion_time_min": charging,
        "total_required_time_min": total,
        "cost_lower_bound_min": lower,
        "cost_residual_min": residual,
        "target_log1p_residual": math.log1p(residual),
    }


def make_bundle(rows, selection):
    rows = sorted(rows, key=lambda r: int(r["drone_name"].rsplit("_", 1)[1]))
    if [r["drone_name"] for r in rows] != [f"drone_{i}" for i in range(1, 6)]:
        raise ValueError("Expected exactly one record per drone in a source run")
    for key in ("formation", "wind_direction", "wind_level", "inter_drone_spacing_cm"):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f"Inconsistent source metadata: {key}")
    soc = [float(r["start_reported_soc_pct"]) for r in rows]
    if not all(40 <= float(r["end_reported_soc_pct"]) <= soc[i] <= 75 for i, r in enumerate(rows)):
        raise ValueError("Selected source run does not satisfy medium SOC bounds")
    observed_positions = tuple(int(r["slot_id"].rsplit("_", 1)[1]) - 1 for r in rows)
    if sorted(observed_positions) != list(range(N)):
        raise ValueError("Source run must observe each of five positions once")
    rates = [None] * N
    for r, p in zip(rows, observed_positions):
        rates[p] = float(r["curve_slope_Bideal_pp_per_min"])
    scales = [float(r["physical_to_Bideal_scale"]) for r in rows]
    # Validate even if this source is subsequently quarantined for QC.
    mission_values(soc, rates, scales, observed_positions)
    flags = sorted({flag for r in rows for flag in r["curve_qc_flags"].split(";") if flag})
    if any(rate <= 0 for rate in rates) and "flat_forward_SOC_curve" not in flags:
        flags.append("nonpositive_fitted_rate")
    formation = canonical_formation(rows[0]["formation"])
    wind = rows[0]["wind_direction"]
    spacing = int(float(rows[0]["inter_drone_spacing_cm"]))
    level = int(float(rows[0]["wind_level"]))
    if formation not in FORMATIONS or wind not in WINDS or spacing not in (50, 75) or level not in (1, 2):
        raise ValueError("Unsupported source condition")
    unsafe_form = "echalon" if formation == "echelon" else formation
    unsafe = (unsafe_form, spacing) in UNSAFE_STRUCTURES_BY_CONDITION.get((wind, level), ())
    exp, run = rows[0]["experiment_directory"], rows[0]["run_id"]
    return {
        "rows": rows, "soc": soc, "rates": rates, "scales": scales,
        "observed_positions": observed_positions, "flags": flags,
        "formation": formation, "wind": wind, "spacing": spacing,
        "level": level, "unsafe": unsafe, "exp": exp, "run": run,
        "source_group": f"{exp}::{run}",
        "metadata_warning": "" if truth(selection["registry_metadata_matches_csv"]) else "registry_csv_mismatch_use_processed_csv_pending_review",
    }


def candidate_row(bundle, positions, k):
    b = bundle
    assigned, physical, arrival = mission_values(b["soc"], b["rates"], b["scales"], positions)
    reasons = list(b["flags"])
    if b["unsafe"]:
        reasons.append("existing_unsafe_condition")
    if not all(40 - TOLERANCE <= s <= 75 + TOLERANCE for s in arrival):
        reasons.append("candidate_arrival_outside_medium")
    config = f"{b['formation']}_{b['spacing']}__p" + "-".join(str(p + 1) for p in positions)
    row = dict.fromkeys(COLUMNS, "")
    row.update({
        "source_experiment": b["exp"], "source_run_id": b["run"],
        "source_group_id": b["source_group"], "scenario_id": f"{b['source_group']}::K{k}",
        "configuration_id": config, "formation": b["formation"],
        "wind_direction": b["wind"], "wind_level": b["level"],
        "spacing_cm": b["spacing"], "charging_pad_count": k,
        "remaining_distance_cm": 250, "flight_time_seconds": FLIGHT_SECONDS,
        "source_type": "real_processed_rate_and_observed_soc",
        "label_source": "linear_25s_battery_adjusted_exponential_charge_exact_schedule",
        "is_observed_position_assignment": positions == b["observed_positions"],
        "source_qc_flags": ";".join(b["flags"]), "metadata_warning": b["metadata_warning"],
        "label_available": False, "training_eligible": False,
        "exclusion_reasons": ";".join(reasons), "analytic_k": k in (1, 5),
        "pad_availability_ratio": k / 5, "spacing_ratio": b["spacing"] / 75,
    })
    row.update({f"wind_{w}": int(w == b["wind"]) for w in WINDS})
    row.update({f"formation_{f}": int(f == b["formation"]) for f in FORMATIONS})
    for i in range(1, 6):
        j = i - 1
        row.update({
            f"soc_d{i}": b["soc"][j], f"soc_d{i}_ratio": b["soc"][j] / 100,
            f"battery_id_d{i}": b["rows"][j]["battery_id"],
            f"position_d{i}": positions[j] + 1,
            f"assigned_Bideal_rate_d{i}": assigned[j],
            f"battery_scale_d{i}": b["scales"][j],
            f"physical_rate_d{i}": physical[j], f"arrival_soc_d{i}": arrival[j],
        })
    # Do not turn flat/flagged traces or out-of-domain projections into targets.
    if reasons:
        return row
    jobs, pads, costs = cost_values(arrival, k)
    row.update(costs)
    row["label_available"] = True
    row["training_eligible"] = True
    for i, (q, pad) in enumerate(zip(jobs, pads), 1):
        row[f"charging_time_d{i}_min"] = q
        row[f"charging_pad_d{i}"] = pad + 1
    return row


def assign_ranks(rows):
    valid = sorted((r for r in rows if r["training_eligible"]), key=lambda r: r["total_required_time_min"])
    if not valid:
        return
    best = valid[0]["total_required_time_min"]
    last_cost, rank = None, 0
    for index, row in enumerate(valid, 1):
        cost = row["total_required_time_min"]
        if last_cost is None or cost - last_cost > TOLERANCE:
            rank = index
            last_cost = cost
        row["rank_within_source_structure"] = rank
        row["regret_within_source_structure_min"] = cost - best


def build(source_dir, output_dir):
    rate_path = source_dir / "forward_discharge_rate_run_drone.csv"
    selection_path = source_dir / "selected_runs_by_database_cell.csv"
    selections = {
        (r["experiment_directory"], r["run_id"]): r
        for r in read_rows(selection_path)
        if r["selection_status"] == "selected"
        and truth(r["formal_clean_trajectory_candidate"])
        and truth(r["all_five_within_75_to_40_range"])
    }
    grouped = defaultdict(list)
    for r in read_rows(rate_path):
        key = (r["experiment_directory"], r["run_id"])
        if key in selections:
            grouped[key].append(r)
    if set(grouped) != set(selections):
        raise ValueError("Selected runs and processed rate records disagree")
    bundles = [make_bundle(grouped[key], selections[key]) for key in sorted(grouped)]
    if len(FEATURES) != 26 or len(set(COLUMNS)) != len(COLUMNS):
        raise AssertionError("Invalid feature/column schema")
    output_dir.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    reason_counts = Counter()
    run_audit = []
    labelled_groups = set()
    with (
        gzip.open(output_dir / "all_candidate_labels.csv.gz", "wt", newline="", encoding="utf-8") as audit_stream,
        gzip.open(output_dir / "training_candidates.csv.gz", "wt", newline="", encoding="utf-8") as train_stream,
        (output_dir / "observed_configuration_labels.csv").open("w", newline="", encoding="utf-8") as observed_stream,
    ):
        writers = [csv.DictWriter(s, fieldnames=COLUMNS) for s in (audit_stream, train_stream, observed_stream)]
        for writer in writers:
            writer.writeheader()
        for bundle in bundles:
            run_counts = Counter()
            for k in range(1, 6):
                rows = [candidate_row(bundle, p, k) for p in itertools.permutations(range(N))]
                assign_ranks(rows)
                for row in rows:
                    writers[0].writerow(row)
                    counts["all_candidate_rows"] += 1
                    if row["training_eligible"]:
                        writers[1].writerow(row)
                        counts["labelled_candidate_rows"] += 1
                        counts[f"labelled_k{k}"] += 1
                        run_counts["labelled"] += 1
                        labelled_groups.add(bundle["source_group"])
                        if k in (2, 3, 4):
                            counts["nonanalytic_residual_training_rows"] += 1
                    else:
                        counts["quarantined_candidate_rows"] += 1
                        reason_counts.update(row["exclusion_reasons"].split(";"))
                    if row["is_observed_position_assignment"]:
                        writers[2].writerow(row)
                        counts["observed_assignment_rows"] += 1
            run_audit.append({
                "source_experiment": bundle["exp"], "source_run_id": bundle["run"],
                "source_group_id": bundle["source_group"], "source_qc_flags": ";".join(bundle["flags"]),
                "metadata_warning": bundle["metadata_warning"], "labelled_candidates": run_counts["labelled"],
                "all_candidates": 600,
                "source_forward_durations_s": [float(r["forward_movement_sec"]) for r in bundle["rows"]],
                "source_detected_forward_distances_cm": [float(r["detected_forward_distance_cm"]) for r in bundle["rows"]],
                "source_battery_ids": [r["battery_id"] for r in bundle["rows"]],
                "source_soc_start": bundle["soc"], "source_slot_rates_Bideal": bundle["rates"],
                "source_battery_scales": bundle["scales"],
            })
    manifest = {
        "status": "labels_generated_no_model_trained",
        "sources": {str(p): sha256(p) for p in (rate_path, selection_path)},
        "builder_sha256": sha256(Path(__file__)),
        "charging_model_sha256": sha256(ROOT / "ml_policy/charging_model.py"),
        "feature_names": FEATURES, "feature_count": len(FEATURES),
        "target": "target_log1p_residual",
        "total_time_column": "total_required_time_min",
        "charging_makespan_column": "charging_completion_time_min",
        "auxiliary_only_columns": TARGETS + [f"charging_time_d{i}_min" for i in range(1, 6)] + [f"arrival_soc_d{i}" for i in range(1, 6)],
        "source_run_count": len(bundles), "source_drone_rows": len(bundles) * 5,
        "source_runs_with_labels": len(labelled_groups),
        "counts": dict(counts), "exclusion_reason_counts_nonexclusive": dict(reason_counts),
        "assumptions": {
            "remaining_distance_cm": 250, "forward_speed_cm_per_s": 10,
            "flight_seconds": FLIGHT_SECONDS, "minimum_model_soc": 40, "maximum_model_soc": 75,
            "charging_target_soc": FULLY_CHARGED_SOC,
            "zero_to_target_charging_minutes": ZERO_TO_FULLY_CHARGED_MINUTES,
            "battery_rule": "physical_rate = assigned_Bideal_rate / own_battery_scale",
            "cost_rule": "total_required_time_min = 25/60 + exact_parallel_charging_makespan",
            "cost_tolerance_minutes": TOLERANCE,
            "position_transfer": "medium slot load is transferable after battery normalisation; not experimentally confirmed permutations",
        },
        "scope": "Each observed source run supports only its own formation/spacing; enumerate five K values and 120 position permutations. No random SOC or invented rate cells.",
        "training_use": "Select columns using feature_names only. For residual-learning use charging_pad_count in [2,3,4]; K=1/5 labels are analytic baselines. Fit scaler on training sources only.",
        "split_rule": "All windows, K and permutations from the same original trial/session must stay together. source_group_id is run-level; interrupted/merged sessions require additional grouping before splitting.",
        "limitations": [
            "25-second consumption and all charging labels are calculated, not observed endpoints or measured charge times.",
            "Input source uses independently cleaned forward clocks, not certified common 25-second windows; raw distances/durations are preserved in source_run_audit.json.",
            "Existing fitted rates and battery calibration are reused; calibration was fitted over 75%-30/36% and awaits a strict-medium sensitivity audit.",
            "Registry/CSV mismatches are carried as metadata warnings; not silently resolved by this builder.",
            "Any source curve QC flag quarantines its whole five-drone bundle from numeric targets in this conservative first version.",
            "Labels are model-based supervision, not independent real position-swap experiments or proof of real charging optimality.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "source_run_audit.json").write_text(json.dumps(run_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing directories are never overwritten")
    args = parser.parse_args()
    manifest = build(args.source_dir, args.output_dir)
    print(json.dumps({"output": str(args.output_dir.resolve()), "sources": manifest["source_run_count"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
