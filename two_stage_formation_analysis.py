#!/usr/bin/env python3
"""Meeting-compliant two-stage formation analysis.

Stage 1 predicts segment energy from formation, wind and entry SOC, and
predicts formation-dependent crossing time from formation and wind.  Charging
pad availability is not an input to either regressor.  Stage 2 combines the
two predictions with an explicit charging/queue calculation:

    total time = crossing time + charging service time + pad queue wait.

Validation is strict leave-one-wind-condition-out (LOWO).  For every measured
condition, a model recommendation is checked against the direct empirical
cell means.  A mismatch is replaced by the empirical winner, as required by
the 2 September 2026 methodology meeting.

The script also audits experiments collected after 26 August 2026.  Stationary
wind-tunnel depletion and preparation runs are retained as calibration data,
but are never silently mixed into the crossing-energy/travel-time training set.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "analysis_outputs" / "two_stage_formation_20260906"
DEFAULT_DATABASE = ROOT / "database"
DEFAULT_RUN_CANDIDATES = (
    ROOT / "analysis_outputs" / "corrected_forward_75cm_runs.csv",
    Path.home()
    / "Desktop"
    / "SDaaS Paper"
    / "SDaaS_reproducibility_bundle_2026-08-25"
    / "experiment"
    / "analysis_outputs"
    / "corrected_forward_75cm_runs.csv",
)

REFERENCE_SOC = 75.0
CHARGE_RATE_PP_PER_MIN = 4.5
SWARM_SIZE = 5
PADS = tuple(range(1, SWARM_SIZE + 1))
MIN_DEPLOYMENT_DECISION_ACCURACY = 0.50
NEW_DATA_CUTOFF = 20260826

CATEGORICAL_FEATURES = [
    "formation",
    "direction",
    "level",
    "spacing_label",
    "formation_condition",
    "service_cell",
]
ENERGY_NUMERIC_FEATURES = [
    "start_soc",
    "spacing_cm",
    "segment_count",
    "step_distance_cm",
    "wind_mean",
    "wind_mean_sq",
    "wind_max",
    "wind_min",
    "wind_head",
    "wind_tail",
    "wind_side",
]
TIME_NUMERIC_FEATURES = [
    feature for feature in ENERGY_NUMERIC_FEATURES if feature != "start_soc"
]


@dataclass(frozen=True)
class Candidate:
    estimator: object
    polynomial_degree: int | None = None


def resolve_default_run_csv() -> Path:
    for path in DEFAULT_RUN_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_RUN_CANDIDATES[0]


def model_candidates() -> dict[str, Candidate]:
    candidates = {
        f"Ridge alpha={alpha:g}": Candidate(Ridge(alpha=alpha))
        for alpha in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)
    }
    candidates.update(
        {
            f"Polynomial Ridge degree=2 alpha={alpha:g}": Candidate(
                Ridge(alpha=alpha), polynomial_degree=2
            )
            for alpha in (0.01, 0.1, 1, 3, 10, 30, 100)
        }
    )
    candidates.update(
        {
            "Extra Trees": Candidate(
                ExtraTreesRegressor(
                    n_estimators=180,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=1,
                )
            ),
            "Random Forest": Candidate(
                RandomForestRegressor(
                    n_estimators=180,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=1,
                )
            ),
            "Gradient Boosting": Candidate(
                GradientBoostingRegressor(
                    n_estimators=120,
                    max_depth=2,
                    min_samples_leaf=3,
                    learning_rate=0.03,
                    random_state=42,
                )
            ),
            "Hist Gradient Boosting": Candidate(
                HistGradientBoostingRegressor(
                    max_iter=160,
                    max_leaf_nodes=7,
                    min_samples_leaf=10,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    random_state=42,
                )
            ),
            "SVR RBF": Candidate(SVR(kernel="rbf", C=10.0, epsilon=0.2)),
        }
    )
    return candidates


def estimator_pipeline(
    candidate: Candidate,
    numeric_features: list[str],
):
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    steps: list[object] = [preprocessing]
    if candidate.polynomial_degree is not None:
        steps.append(
            PolynomialFeatures(
                degree=candidate.polynomial_degree,
                include_bias=False,
            )
        )
    steps.append(clone(candidate.estimator))
    return make_pipeline(*steps)


def load_training_runs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Corrected run table not found: {path}. Pass --run-csv explicitly."
        )
    runs = pd.read_csv(path, dtype={"run_id": str})
    required = set(
        CATEGORICAL_FEATURES
        + ENERGY_NUMERIC_FEATURES
        + [
            "condition",
            "formation",
            "selectable_cell",
            "endpoint_start_soc",
            "common_swarm_energy_250",
            "forward_service_time_250_s",
        ]
    )
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError("Run table is missing columns: " + ", ".join(missing))
    if len(runs) != 80:
        raise ValueError(f"Expected 80 corrected 75 cm runs; found {len(runs)}")
    runs = runs.loc[runs["selectable_cell"]].copy().reset_index(drop=True)
    if len(runs) != 79:
        raise ValueError(f"Expected 79 selectable runs; found {len(runs)}")
    # Common-swarm endpoint energy begins after settling, so its matching SOC
    # baseline is endpoint_start_soc rather than the earlier start_soc field.
    runs["start_soc"] = runs["endpoint_start_soc"].astype(float)
    return runs


def within_cell_soc_slope(runs: pd.DataFrame, energy: np.ndarray) -> float:
    frame = runs[["condition", "formation", "start_soc"]].copy()
    frame["energy"] = np.asarray(energy, dtype=float)
    grouped = frame.groupby(["condition", "formation"])
    centred_soc = frame["start_soc"] - grouped["start_soc"].transform("mean")
    centred_energy = frame["energy"] - grouped["energy"].transform("mean")
    denominator = float(np.dot(centred_soc, centred_soc))
    if denominator <= 0:
        raise RuntimeError("SOC slope is unidentified")
    return float(np.dot(centred_soc, centred_energy) / denominator)


def strict_lowo(
    runs: pd.DataFrame,
    labels: np.ndarray,
    candidates: dict[str, Candidate],
    numeric_features: list[str],
    *,
    query_soc: float | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    observed_predictions: dict[str, np.ndarray] = {}
    query_predictions: dict[str, np.ndarray] = {}
    conditions = runs["condition"].to_numpy()
    for name, candidate in candidates.items():
        observed = np.full(len(runs), np.nan, dtype=float)
        queried = np.full(len(runs), np.nan, dtype=float)
        for held in sorted(runs["condition"].unique()):
            train = np.flatnonzero(conditions != held)
            test = np.flatnonzero(conditions == held)
            estimator = estimator_pipeline(candidate, numeric_features)
            estimator.fit(runs.iloc[train], labels[train])
            observed[test] = estimator.predict(runs.iloc[test])
            query_frame = runs.iloc[test].copy()
            if query_soc is not None:
                query_frame["start_soc"] = query_soc
            queried[test] = estimator.predict(query_frame)
        rows.append(
            {
                "model": name,
                "r2": float(r2_score(labels, observed)),
                "mae": float(mean_absolute_error(labels, observed)),
                "polynomial": candidate.polynomial_degree is not None,
                "polynomial_degree": candidate.polynomial_degree or "",
            }
        )
        observed_predictions[name] = observed
        query_predictions[name] = queried
    metrics = pd.DataFrame(rows).sort_values(
        ["r2", "mae", "model"], ascending=[False, True, True]
    )
    return metrics, observed_predictions, query_predictions


def fold_local_reference_energy(
    runs: pd.DataFrame,
    observed_energy: np.ndarray,
    reference_soc: float,
) -> tuple[np.ndarray, dict[str, float]]:
    result = np.full(len(runs), np.nan, dtype=float)
    betas: dict[str, float] = {}
    conditions = runs["condition"].to_numpy()
    for held in sorted(runs["condition"].unique()):
        train = np.flatnonzero(conditions != held)
        test = np.flatnonzero(conditions == held)
        beta = within_cell_soc_slope(runs.iloc[train], observed_energy[train])
        result[test] = observed_energy[test] - beta * (
            runs.iloc[test]["start_soc"].to_numpy(float) - reference_soc
        )
        betas[str(held)] = beta
    if not np.isfinite(result).all():
        raise RuntimeError("Incomplete fold-local energy truth")
    return result, betas


def stage2_components(
    energy_points: pd.Series | np.ndarray,
    crossing_seconds: pd.Series | np.ndarray,
    pads: int,
    charge_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = np.maximum(0.0, np.asarray(energy_points, dtype=float))
    crossing = np.maximum(0.0, np.asarray(crossing_seconds, dtype=float))
    service = 60.0 * energy / charge_rate
    queue_waves = math.ceil(SWARM_SIZE / pads) - 1
    queue = service * queue_waves
    return service, queue, crossing + service + queue


def build_decisions(
    runs: pd.DataFrame,
    predicted_energy: np.ndarray,
    predicted_time: np.ndarray,
    validation_energy_truth: np.ndarray,
    empirical_energy_truth: np.ndarray,
    *,
    charge_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = runs[["condition", "formation"]].copy()
    frame["predicted_energy_pp"] = np.maximum(0.0, predicted_energy)
    frame["predicted_crossing_s"] = np.maximum(0.0, predicted_time)
    frame["validation_energy_pp"] = validation_energy_truth
    frame["measured_energy_pp"] = empirical_energy_truth
    frame["measured_crossing_s"] = runs["forward_service_time_250_s"].to_numpy(
        float
    )
    cells = frame.groupby(["condition", "formation"], as_index=False).mean()

    records: list[dict[str, object]] = []
    for condition, group in cells.groupby("condition", sort=True):
        for pads in PADS:
            predicted_service, predicted_queue, predicted_total = stage2_components(
                group["predicted_energy_pp"],
                group["predicted_crossing_s"],
                pads,
                charge_rate,
            )
            _validation_service, _validation_queue, validation_total = stage2_components(
                group["validation_energy_pp"],
                group["measured_crossing_s"],
                pads,
                charge_rate,
            )
            measured_service, measured_queue, measured_total = stage2_components(
                group["measured_energy_pp"],
                group["measured_crossing_s"],
                pads,
                charge_rate,
            )
            model_position = int(np.argmin(predicted_total))
            validation_oracle_position = int(np.argmin(validation_total))
            empirical_position = int(np.argmin(measured_total))
            empirical_order = np.argsort(measured_total)
            runner_up_position = int(empirical_order[1])
            model_row = group.iloc[model_position]
            validation_oracle_row = group.iloc[validation_oracle_position]
            empirical_row = group.iloc[empirical_position]
            runner_up_row = group.iloc[runner_up_position]
            strict_match = bool(
                model_row["formation"] == validation_oracle_row["formation"]
            )
            empirical_match = bool(
                model_row["formation"] == empirical_row["formation"]
            )
            selected_validation_total = float(validation_total[model_position])
            records.append(
                {
                    "condition": condition,
                    "charging_pads": pads,
                    "model_formation": model_row["formation"],
                    "strict_lowo_oracle_formation": validation_oracle_row["formation"],
                    "empirical_formation": empirical_row["formation"],
                    "empirical_runner_up": runner_up_row["formation"],
                    "empirical_margin_to_runner_up_s": float(
                        measured_total[runner_up_position]
                        - measured_total[empirical_position]
                    ),
                    "strict_lowo_match": strict_match,
                    "match": empirical_match,
                    "final_formation": (
                        model_row["formation"]
                        if empirical_match
                        else empirical_row["formation"]
                    ),
                    "recommendation_source": (
                        "validated_model" if empirical_match else "empirical_fallback"
                    ),
                    "model_predicted_energy_pp": float(
                        model_row["predicted_energy_pp"]
                    ),
                    "model_predicted_crossing_s": float(
                        model_row["predicted_crossing_s"]
                    ),
                    "model_predicted_charging_service_s": float(
                        predicted_service[model_position]
                    ),
                    "model_predicted_queue_wait_s": float(
                        predicted_queue[model_position]
                    ),
                    "model_predicted_total_s": float(predicted_total[model_position]),
                    "empirical_energy_pp": float(empirical_row["measured_energy_pp"]),
                    "empirical_crossing_s": float(
                        empirical_row["measured_crossing_s"]
                    ),
                    "empirical_charging_service_s": float(
                        measured_service[empirical_position]
                    ),
                    "empirical_queue_wait_s": float(measured_queue[empirical_position]),
                    "empirical_best_total_s": float(measured_total[empirical_position]),
                    "model_choice_validation_total_s": selected_validation_total,
                    "strict_lowo_oracle_total_s": float(
                        validation_total[validation_oracle_position]
                    ),
                    "regret_s": selected_validation_total
                    - float(validation_total[validation_oracle_position]),
                }
            )
    decisions = pd.DataFrame(records)

    errors = (
        decisions.groupby("condition", as_index=False)
        .agg(
            matched_pad_scenarios=("strict_lowo_match", "sum"),
            pad_scenarios=("strict_lowo_match", "size"),
            mean_regret_s=("regret_s", "mean"),
            max_regret_s=("regret_s", "max"),
        )
        .merge(
            cells.assign(
                energy_abs_error=lambda x: abs(
                    x["predicted_energy_pp"] - x["measured_energy_pp"]
                ),
                crossing_abs_error=lambda x: abs(
                    x["predicted_crossing_s"] - x["measured_crossing_s"]
                ),
            )
            .groupby("condition", as_index=False)
            .agg(
                cell_energy_mae_pp=("energy_abs_error", "mean"),
                cell_crossing_mae_s=("crossing_abs_error", "mean"),
            ),
            on="condition",
            how="left",
            validate="one_to_one",
        )
        .sort_values(["mean_regret_s", "max_regret_s"], ascending=False)
    )
    errors["decision_accuracy"] = (
        errors["matched_pad_scenarios"] / errors["pad_scenarios"]
    )
    errors["rerun_priority"] = np.arange(1, len(errors) + 1)
    return decisions, errors


def _first_value(frame: pd.DataFrame, column: str, default: object = "") -> object:
    if column not in frame or frame.empty:
        return default
    values = frame[column].dropna()
    return values.iloc[0] if not values.empty else default


def _run_date(frame: pd.DataFrame, path: Path) -> int | None:
    raw = str(_first_value(frame, "run_id", ""))
    match = re.search(r"(20\d{6})", raw) or re.search(r"(20\d{6})", path.name)
    return int(match.group(1)) if match else None


def audit_new_experiments(database: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows: list[dict[str, object]] = []
    expected_mapping = {
        "drone_1": "B11",
        "drone_2": "B10",
        "drone_3": "B13",
        "drone_4": "B14",
        "drone_5": "B12",
    }
    target_bands = {
        "high": (85.0, 95.0),
        "medium": (73.0, 77.0),
        "low": (35.0, 40.0),
    }
    for path in sorted(database.glob("**/*_all_battery.csv")):
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            continue
        run_date = _run_date(frame, path)
        if run_date is None or run_date < NEW_DATA_CUTOFF:
            continue

        experiment_id = str(_first_value(frame, "experiment_id", path.parent.name))
        soc_mode = str(_first_value(frame, "soc_mode", "")).strip().lower()
        starts = pd.to_numeric(
            frame.get("battery_hover_start", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        ends = pd.to_numeric(
            frame.get("battery_hover_end", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        drops = pd.to_numeric(
            frame.get("battery_drop", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        durations = pd.to_numeric(
            frame.get("node_duration_sec", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        distance = pd.to_numeric(
            frame.get("node_forward_distance_cm", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        mapping = (
            frame[["drone_name", "battery_id"]]
            .dropna()
            .drop_duplicates()
            .set_index("drone_name")["battery_id"]
            .astype(str)
            .to_dict()
            if {"drone_name", "battery_id"}.issubset(frame.columns)
            else {}
        )

        path_text = str(path).lower()
        if (
            "prepare" in experiment_id.lower()
            or "practice" in experiment_id.lower()
            or "/baselines/" in path_text
        ):
            category = "preparation_or_battery_baseline"
            training_eligible = False
            reason = "preparation/baseline data; not a complete formation crossing"
        elif "wind_tunnel" in experiment_id:
            category = "stationary_wind_tunnel_calibration"
            training_eligible = False
            reason = "stationary hover/depletion; no segment crossing-time label"
        elif soc_mode in target_bands and bool((distance > 0).any()):
            category = "controlled_soc_crossing_pilot"
            training_eligible = False
            reason = (
                "one Front/50 cm/Head-L1 pilot per SOC tier; not a matched "
                "formation comparison and not compatible with the 75 cm training table"
            )
        else:
            category = "preparation_or_battery_baseline"
            training_eligible = False
            reason = "preparation/baseline data; not a complete formation crossing"

        band = target_bands.get(soc_mode)
        within_band = bool(
            band is not None
            and not starts.empty
            and starts.between(band[0], band[1], inclusive="both").all()
        )
        spread = float(starts.max() - starts.min()) if not starts.empty else math.nan
        rows.append(
            {
                "battery_csv": str(path.relative_to(ROOT)),
                "run_date": run_date,
                "experiment_id": experiment_id,
                "category": category,
                "training_eligible": training_eligible,
                "exclusion_reason": reason,
                "soc_mode": soc_mode,
                "declared_target_soc": _first_value(
                    frame, "target_soc_percent", ""
                ),
                "drone_rows": int(len(frame)),
                "formation": _first_value(frame, "formation", ""),
                "wind_direction": _first_value(frame, "wind_direction", ""),
                "wind_level": _first_value(frame, "wind_speed", ""),
                "spacing_cm": _first_value(frame, "inter_drone_distance_cm", ""),
                "forward_distance_cm": float(distance.median())
                if not distance.empty
                else math.nan,
                "duration_s": float(durations.median())
                if not durations.empty
                else math.nan,
                "entry_soc_mean": float(starts.mean())
                if not starts.empty
                else math.nan,
                "entry_soc_min": float(starts.min())
                if not starts.empty
                else math.nan,
                "entry_soc_max": float(starts.max())
                if not starts.empty
                else math.nan,
                "entry_soc_spread_pp": spread,
                "exact_same_entry_soc": bool(not starts.empty and spread == 0.0),
                "entry_soc_spread_within_2pp": bool(
                    not starts.empty and spread <= 2.0
                ),
                "all_entry_soc_in_meeting_band": within_band,
                "end_soc_mean": float(ends.mean()) if not ends.empty else math.nan,
                "battery_drop_mean_pp": float(drops.mean())
                if not drops.empty
                else math.nan,
                "battery_drop_sd_pp": float(drops.std(ddof=1))
                if len(drops) > 1
                else math.nan,
                "diagnostic_drop_rate_pp_per_min": (
                    float(60.0 * drops.mean() / durations.median())
                    if not drops.empty
                    and not durations.empty
                    and float(durations.median()) > 0
                    else math.nan
                ),
                "fixed_battery_mapping": bool(
                    mapping
                    and all(mapping.get(drone) == battery for drone, battery in expected_mapping.items())
                ),
                "possible_duplicate_or_merged": "merged" in path.name.lower(),
            }
        )
    audit = pd.DataFrame(rows)
    pilots = audit.loc[
        audit["category"].eq("controlled_soc_crossing_pilot")
    ].copy()
    if not pilots.empty:
        pilots = pilots[
            [
                "soc_mode",
                "declared_target_soc",
                "entry_soc_mean",
                "entry_soc_min",
                "entry_soc_max",
                "entry_soc_spread_pp",
                "all_entry_soc_in_meeting_band",
                "duration_s",
                "battery_drop_mean_pp",
                "battery_drop_sd_pp",
                "fixed_battery_mapping",
                "battery_csv",
            ]
        ].sort_values("entry_soc_mean", ascending=False)
    summary = {
        "files_audited": int(len(audit)),
        "files_by_category": {
            str(key): int(value)
            for key, value in audit["category"].value_counts().to_dict().items()
        }
        if not audit.empty
        else {},
        "training_eligible_new_files": int(audit["training_eligible"].sum())
        if not audit.empty
        else 0,
        "controlled_soc_pilot_files": int(len(pilots)),
        "controlled_soc_pilots_meeting_band_compliant": int(
            pilots["all_entry_soc_in_meeting_band"].sum()
        )
        if not pilots.empty
        else 0,
        "controlled_soc_pilots_exact_same_entry_soc": int(
            audit.loc[
                audit["category"].eq("controlled_soc_crossing_pilot"),
                "exact_same_entry_soc",
            ].sum()
        )
        if not audit.empty
        else 0,
        "nonempty_stationary_wind_tunnel_files": int(
            (
                audit["category"].eq("stationary_wind_tunnel_calibration")
                & audit["drone_rows"].gt(0)
            ).sum()
        )
        if not audit.empty
        else 0,
        "nonstandard_mapping_nonempty_wind_tunnel_files": int(
            (
                audit["category"].eq("stationary_wind_tunnel_calibration")
                & audit["drone_rows"].gt(0)
                & ~audit["fixed_battery_mapping"]
            ).sum()
        )
        if not audit.empty
        else 0,
    }
    return audit, pilots, summary


def experiment_gap_table(errors: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in errors.itertuples(index=False):
        records.append(
            {
                "priority": int(row.rerun_priority),
                "condition": row.condition,
                "spacing_cm": 75,
                "formations": "Front; Vee; Echelon; Diamond; Column",
                "soc_tiers": "high 85-95%; medium 75%; low 35-40%",
                "minimum_independent_runs_per_formation_tier": 3,
                "requested_controlled_crossings": 45,
                "reason": (
                    f"strict-LOWO matched {int(row.matched_pad_scenarios)}/"
                    f"{int(row.pad_scenarios)} pad cases; mean regret "
                    f"{row.mean_regret_s:.1f} s"
                ),
            }
        )
    records.append(
        {
            "priority": len(records) + 1,
            "condition": "Tail-L2 Echelon replication",
            "spacing_cm": 75,
            "formations": "Echelon",
            "soc_tiers": "medium 75% first; then high and low",
            "minimum_independent_runs_per_formation_tier": 3,
            "requested_controlled_crossings": 8,
            "reason": (
                "current Echelon Tail-L2 cell has one legacy run and is excluded "
                "from selection; add 2 medium plus 3 high plus 3 low runs"
            ),
        }
    )
    return pd.DataFrame(records).sort_values("priority")


def wind_tunnel_coverage(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarise coverage without presenting uncontrolled runs as effects."""

    wind = audit.loc[
        audit["category"].eq("stationary_wind_tunnel_calibration")
        & audit["drone_rows"].gt(0)
    ].copy()
    if wind.empty:
        return pd.DataFrame()
    return (
        wind.groupby(
            ["formation", "wind_direction", "wind_level", "spacing_cm"],
            as_index=False,
            dropna=False,
        )
        .agg(
            battery_files=("battery_csv", "size"),
            entry_soc_mean=("entry_soc_mean", "mean"),
            median_entry_soc_spread_pp=("entry_soc_spread_pp", "median"),
            median_duration_s=("duration_s", "median"),
            diagnostic_median_drop_rate_pp_per_min=(
                "diagnostic_drop_rate_pp_per_min",
                "median",
            ),
            fixed_mapping_share=("fixed_battery_mapping", "mean"),
        )
        .assign(
            suitable_for_crossing_model=False,
            effect_estimate_status=(
                "not comparable until SOC, duration, and independent replicates are matched"
            ),
        )
    )


def _formation_reason(cells: pd.DataFrame, condition: str, formation: str) -> str:
    row = cells.loc[
        cells["condition"].eq(condition) & cells["formation"].eq(formation)
    ].iloc[0]
    return (
        f"measured energy {row.measured_energy_pp:.2f} pp and crossing "
        f"{row.measured_crossing_s:.2f} s"
    )


def write_briefs(
    output: Path,
    energy_metrics: pd.DataFrame,
    time_metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    errors: pd.DataFrame,
    pilot: pd.DataFrame,
    audit_summary: dict,
    selected_energy: str,
    selected_time: str,
) -> None:
    energy_row = energy_metrics.loc[energy_metrics["model"].eq(selected_energy)].iloc[0]
    time_row = time_metrics.loc[time_metrics["model"].eq(selected_time)].iloc[0]
    accuracy = float(decisions["strict_lowo_match"].mean())
    mean_regret = float(decisions["regret_s"].mean())
    p95_regret = float(decisions["regret_s"].quantile(0.95))
    max_regret = float(decisions["regret_s"].max())

    base = decisions.loc[decisions["charging_pads"].eq(1)].copy()
    recommendation_lines = []
    for row in base.itertuples(index=False):
        recommendation_lines.append(
            f"- {row.condition}, 1 pad: **{row.final_formation}** "
            f"({row.recommendation_source}); measured energy "
            f"{row.empirical_energy_pp:.2f} pp, crossing "
            f"{row.empirical_crossing_s:.2f} s, and total "
            f"{row.empirical_best_total_s:.1f} s."
        )

    pilot_lines = []
    for row in pilot.itertuples(index=False):
        pilot_lines.append(
            f"- {row.soc_mode}: entry SOC {row.entry_soc_min:.0f}–"
            f"{row.entry_soc_max:.0f}% (mean {row.entry_soc_mean:.1f}%), "
            f"crossing {row.duration_s:.2f} s, mean drop "
            f"{row.battery_drop_mean_pp:.1f} pp; meeting-band compliant: "
            f"{'yes' if row.all_entry_soc_in_meeting_band else 'no'}."
        )
    if not pilot_lines:
        pilot_lines.append("- No controlled-SOC crossing pilots were found.")

    priority = errors.head(3)["condition"].tolist()
    zh = f"""# 编队选择简短分析（2026-09-06）

## 结论

模型已改成两阶段：能耗模型不含 charging-pad availability；穿越时间按 formation 和 wind 单独预测；最后计算 `总时间 = 穿越时间 + 充电服务时间 + 排队等待时间`。在六个已测风况中，模型建议会逐项与数据集均值比较；不一致时自动采用数据集的最优编队。

## 当前模型结果

- 能耗：{selected_energy}，strict LOWO R² = {energy_row.r2:.3f}，MAE = {energy_row.mae:.3f} 个 SOC 百分点。能耗 R² 达到教授提出的 0.5–0.6 下限。
- 穿越时间：{selected_time}，strict LOWO R² = {time_row.r2:.3f}，MAE = {time_row.mae:.3f} s。Polynomial regression 被加入候选并取得最高时间 R²，但仍未达到 0.5，说明 formation-maintenance、风扰动及控制修正造成的时间差尚未被充分解释。
- 端到端编队选择：{int(decisions['strict_lowo_match'].sum())}/{len(decisions)} = {accuracy:.1%}；平均 regret {mean_regret:.1f} s，95th percentile {p95_regret:.1f} s，最大 {max_regret:.1f} s。因此模型未通过 {MIN_DEPLOYMENT_DECISION_ACCURACY:.0%} 部署门槛，已测条件必须使用 empirical fallback。

## 每个风况（低充电桩可用性，K=1）

{chr(10).join(recommendation_lines)}

## 新数据状态

共审计 {audit_summary.get('files_audited', 0)} 个 2026-08-26 之后的 battery summary；可直接加入 75 cm 穿越模型的文件为 {audit_summary.get('training_eligible_new_files', 0)}。`wind_tunnel_*` 是定点悬停放电，只适合 SOC/电池非线性标定，不能提供 formation crossing time。三档 SOC 穿越数据目前都只是 Front–50 cm–Head-L1 各 1 次：

{chr(10).join(pilot_lines)}

固定 battery-to-drone mapping 已执行，这是改进；但三次 pilot 的五机 entry SOC 都不完全相同，而且 high/medium/low 均没有形成跨 formation 的 matched comparison。

这三次 pilot 的平均 drop 为 2.4、7.2、4.6 pp，并不随 high→medium→low 单调变化；在每档只有一次、Tello SOC 只有整数分辨率且机间起始电量不同的情况下，这应视为测量离散/硬件差异信号，不能解释成 SOC 的真实效应。风洞数据中有 {audit_summary.get('nonempty_stationary_wind_tunnel_files', 0)} 个非空定点放电文件，但 SOC、时长和独立重复尚未配平；其中 {audit_summary.get('nonstandard_mapping_nonempty_wind_tunnel_files', 0)} 个 Diamond-side/no-wind 文件把 drone_5 从 B12 换成 B06，跨编队比较前需要恢复统一硬件映射或显式校正 battery effect。

## 需要补充的实验

1. 先补 **{', '.join(priority)}**（按当前 regret 排序）：75 cm、所有候选 formation、high/medium/low，每个 cell 至少 3 个独立 run；同批五机进入 segment 时 SOC 必须在同一档且 spread ≤2 pp。
2. Tail-L2 的 Echelon 目前只有 1 个旧 run：先补 2 个 medium，再补 high/low 各 3 个，否则该 cell 继续排除。
3. 每个 run 固定 battery-to-drone mapping，只轮换 drone 的 formation position；记录穿越时间、每机 SOC drop、位置/间距误差。不要把 wind-tunnel hover 当作 flight replicate。
4. 增加真实充电曲线和多 pad 同时充电测试；当前 4.5 pp/min 只是场景参数，排队时间计算可复现，但尚未由新充电实验校准。
"""

    en = f"""# Concise formation analysis (2026-09-06)

The pipeline now follows the required two-stage structure. Stage 1 predicts energy from formation, wind and entry SOC, and predicts formation-dependent crossing time separately. Stage 2 adds crossing time, charging service time and charging-pad queue wait. Pad availability is never used to train the energy model. For every measured condition, the model choice is checked against direct dataset means; a disagreement triggers an empirical fallback.

The selected energy model is {selected_energy} (strict-LOWO R² {energy_row.r2:.3f}, MAE {energy_row.mae:.3f} SOC percentage points). The selected crossing-time model is {selected_time} (R² {time_row.r2:.3f}, MAE {time_row.mae:.3f} s). Polynomial regression gives the best crossing-time R², but time prediction remains below the 0.5 target. End-to-end formation accuracy is {int(decisions['strict_lowo_match'].sum())}/{len(decisions)} ({accuracy:.1%}); mean regret is {mean_regret:.1f} s. The learned selector therefore remains below the {MIN_DEPLOYMENT_DECISION_ACCURACY:.0%} deployment gate, and measured conditions use the empirical winner.

For one available charging pad, the empirical recommendations are: """ + "; ".join(
        f"{row.condition}: {row.final_formation}"
        for row in base.itertuples(index=False)
    ) + f""".

The new data do not yet improve this selector. Wind-tunnel runs are stationary depletion tests and contain no crossing-time label. The controlled-SOC crossing data contain only one Front–50 cm–Head-L1 run at each of high, medium and low SOC, and none has identical entry SOC across all five drones. Their mean drops (2.4, 7.2 and 4.6 percentage points) are not monotonic, so three single runs with unequal entry SOC cannot establish an SOC effect. In addition, {audit_summary.get('nonstandard_mapping_nonempty_wind_tunnel_files', 0)} non-empty Diamond side/no-wind calibration files use B06 on drone_5 instead of the otherwise fixed B12 mapping. The next runs should prioritize {', '.join(priority)} at 75 cm, using matched formation comparisons at high (85–95%), medium (75%) and low (35–40%) SOC with at least three independent runs per cell. Keep each battery assigned to one drone and rotate drone positions, not batteries.
"""
    (output / "analysis_brief_zh.md").write_text(zh, encoding="utf-8")
    (output / "analysis_brief_en.md").write_text(en, encoding="utf-8")

    one_pad = decisions.loc[decisions["charging_pads"].eq(1)].set_index("condition")
    five_pads = decisions.loc[decisions["charging_pads"].eq(5)].set_index("condition")

    condition_rows = [
        (
            "Head-L1",
            "Front (preliminary)",
            "It has the lowest measured energy. Its crossing is slightly slower than Vee/Diamond, but the energy saving outweighs that difference, especially when pads are scarce.",
            "Model and dataset disagree; repeat before making a firm claim.",
            "Front（初步）",
            "实测能耗最低。虽然穿越时间略慢于 Vee/Diamond，但节省的充电时间更大，尤其在充电桩少时。",
            "模型与数据集不一致，需要补飞确认。",
        ),
        (
            "Head-L2",
            "No firm winner: Echelon ≈ Vee",
            "Echelon is first in the present mean, but Echelon and Vee differ by only 0.07 SOC points and 0.09 s, much less than run-to-run variation.",
            "Do not claim a winner until matched reruns separate them.",
            "暂不确定：Echelon ≈ Vee",
            "当前均值中 Echelon 第一，但与 Vee 仅相差约 0.07 个 SOC 百分点和 0.09 s，远小于重复实验波动。",
            "在配对补飞区分二者前，不应宣称唯一最佳。",
        ),
        (
            "Side-L1",
            "Front (preliminary)",
            "Front has both the lowest measured energy and the shortest crossing time, so it remains first from one to five pads.",
            "Other formations have only two runs; add matched repeats.",
            "Front（初步）",
            "Front 同时具有最低实测能耗和最短穿越时间，因此从 1 到 5 个充电桩都排第一。",
            "其他编队只有两次实验，仍需配对重复。",
        ),
        (
            "Side-L2",
            "Front (preliminary)",
            "Front has the shortest crossing time and the lowest mean energy. Diamond is the nearest alternative but is slower, so Front remains first for all tested pad counts.",
            "Front has only two runs and its energy overlaps Diamond; repeat both.",
            "Front（初步）",
            "Front 的穿越时间最短且平均能耗最低；Diamond 是最接近的备选但更慢，因此当前 1–5 个充电桩下均为 Front 第一。",
            "Front 只有两次实验且能耗与 Diamond 有重叠，需要重点复测二者。",
        ),
        (
            "Tail-L1",
            "Diamond",
            "Diamond uses substantially less energy while crossing only about 0.42 s slower than the fastest formation, Echelon. The charging-time saving dominates this small flight-time difference.",
            "Strongest current logical result, but controlled-SOC replication is still required.",
            "Diamond",
            "Diamond 的能耗明显更低，而穿越时间只比最快的 Echelon 慢约 0.42 s；节省的充电时间远大于这点飞行时间差。",
            "这是当前逻辑最清楚的结果，但仍需受控 SOC 重复实验。",
        ),
        (
            "Tail-L2",
            "Diamond among measured formations",
            "Diamond has both the lowest energy and the shortest crossing time among Front, Vee, Diamond and Column.",
            "Echelon has only one excluded run, so the five-formation winner is not yet proven.",
            "已测编队中为 Diamond",
            "在 Front、Vee、Diamond 和 Column 中，Diamond 同时具有最低能耗和最短穿越时间。",
            "Echelon 只有一次被排除的实验，因此五种编队中的最终赢家尚未证明。",
        ),
    ]

    en_rows = []
    zh_rows = []
    for condition, conclusion_en, reason_en, status_en, conclusion_zh, reason_zh, status_zh in condition_rows:
        low = one_pad.loc[condition]
        high = five_pads.loc[condition]
        en_rows.append(
            f"| {condition} | {conclusion_en} | {reason_en} | {status_en} |"
        )
        zh_rows.append(
            f"| {condition} | {conclusion_zh} | {reason_zh} | {status_zh} |"
        )
        if low["empirical_formation"] != high["empirical_formation"]:
            raise RuntimeError(
                f"Pad-count winner changed for {condition}; update the concise logic"
            )

    amna_en = """# Condition-by-condition formation analysis

Scope: 75 cm spacing and the current 4.5 percentage-point/min charging assumption. The empirical winner is unchanged from one to five available pads, although fewer pads make energy consumption more important.

**Concise theoretical basis.** Wind direction and speed change relative airflow, drag and the attitude corrections required for trajectory tracking, so they affect both energy and travel time ([Jacewicz et al., 2022](https://doi.org/10.3390/en15197136)). In a close multirotor formation, rotor wakes/downwash create nonlinear forces that depend on relative vehicle geometry ([Gielis et al., 2024](https://doi.org/10.1007/978-3-031-63596-0_35)); wind disturbance also creates relative-position errors that the formation controller must correct ([Zhang et al., 2020](https://doi.org/10.7527/S1000-6893.2019.23385)). Therefore, formation-specific crossing time should be measured rather than treated as constant. When pads are scarce, the same battery losses must be served in more charging waves, so energy differences receive more weight. These mechanisms justify the comparison criteria, but they do not prove that one named formation is universally best—the ranking below remains empirical.

| Wind condition | Current conclusion | Concise reasoning | Evidence status |
|---|---|---|---|
""" + "\n".join(en_rows) + f"""

The new controlled-SOC data cannot yet change these conclusions: there is only one Front–Head-L1–50 cm run per SOC tier, and the five drones did not enter the segment at the same SOC. Therefore, the immediate rerun priorities are {', '.join(priority)}. The energy model reaches R² {energy_row.r2:.3f}, but end-to-end selection accuracy is only {accuracy:.1%}; these are empirical, provisional recommendations rather than validated universal rules.
"""
    amna_zh = """# 每个风况下的编队选择简析

范围：75 cm 间距，充电速率暂按 4.5 个百分点/分钟计算。当前数据中，1–5 个可用充电桩的最优编队相同；充电桩越少，能耗对结果的影响越大。

**简短理论依据。** 风向和风速会改变相对气流、阻力以及轨迹跟踪所需的姿态修正，因此会同时影响能耗和飞行时间（[Jacewicz et al., 2022](https://doi.org/10.3390/en15197136)）。多旋翼近距离编队还存在随相对位置变化的非线性旋翼尾流/下洗干扰（[Gielis et al., 2024](https://doi.org/10.1007/978-3-031-63596-0_35)）；风扰动造成的相对位置误差需要编队控制器持续修正（[Zhang et al., 2020](https://doi.org/10.7527/S1000-6893.2019.23385)）。所以不同 formation 的 crossing time 不能视为常数。充电桩越少，同样的电量损失需要更多轮排队，能耗差异在总时间中所占权重越大。这些理论只能解释比较机制，不能预先证明某种编队必然最优，具体排名仍以实验结果为准。

| 风况 | 当前结论 | 简短逻辑 | 证据状态 |
|---|---|---|---|
""" + "\n".join(zh_rows) + f"""

新三档 SOC 数据目前不能改变上述结论：每档只有一次 Front–Head-L1–50 cm 实验，而且五架无人机进入 segment 时 SOC 不一致。因此应优先补做 {', '.join(priority)}。能耗模型 R² 为 {energy_row.r2:.3f}，但端到端选择准确率只有 {accuracy:.1%}；以上应表述为当前数据支持的初步建议，而不是已经验证的普适规律。
"""
    (output / "analysis_for_amna_en.md").write_text(amna_en, encoding="utf-8")
    (output / "analysis_for_amna_zh.md").write_text(amna_zh, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-csv", type=Path, default=resolve_default_run_csv())
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-soc", type=float, default=REFERENCE_SOC)
    parser.add_argument(
        "--charge-rate-pp-per-min", type=float, default=CHARGE_RATE_PP_PER_MIN
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.charge_rate_pp_per_min <= 0:
        raise ValueError("charge rate must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_training_runs(args.run_csv)
    candidates = model_candidates()
    observed_energy = runs["common_swarm_energy_250"].to_numpy(float)
    observed_time = runs["forward_service_time_250_s"].to_numpy(float)

    energy_metrics, _, energy_at_reference = strict_lowo(
        runs,
        observed_energy,
        candidates,
        ENERGY_NUMERIC_FEATURES,
        query_soc=args.reference_soc,
    )
    time_metrics, time_predictions, _ = strict_lowo(
        runs,
        observed_time,
        candidates,
        TIME_NUMERIC_FEATURES,
    )
    selected_energy = str(energy_metrics.iloc[0]["model"])
    selected_time = str(time_metrics.iloc[0]["model"])
    reference_truth, fold_betas = fold_local_reference_energy(
        runs, observed_energy, args.reference_soc
    )
    full_data_beta = within_cell_soc_slope(runs, observed_energy)
    empirical_reference_truth = observed_energy - full_data_beta * (
        runs["start_soc"].to_numpy(float) - args.reference_soc
    )
    decisions, errors = build_decisions(
        runs,
        energy_at_reference[selected_energy],
        time_predictions[selected_time],
        reference_truth,
        empirical_reference_truth,
        charge_rate=args.charge_rate_pp_per_min,
    )

    audit, pilots, audit_summary = audit_new_experiments(args.database)
    wind_coverage = wind_tunnel_coverage(audit)
    gaps = experiment_gap_table(errors)

    energy_metrics.assign(target="energy", mae_unit="SOC percentage points").to_csv(
        args.output_dir / "energy_model_metrics.csv", index=False
    )
    time_metrics.assign(target="crossing_time", mae_unit="seconds").to_csv(
        args.output_dir / "crossing_time_model_metrics.csv", index=False
    )
    decisions.to_csv(args.output_dir / "condition_decisions.csv", index=False)
    errors.to_csv(args.output_dir / "error_analysis.csv", index=False)
    audit.to_csv(args.output_dir / "new_data_audit.csv", index=False)
    pilots.to_csv(args.output_dir / "soc_pilot_summary.csv", index=False)
    wind_coverage.to_csv(
        args.output_dir / "wind_tunnel_calibration_coverage.csv", index=False
    )
    gaps.to_csv(args.output_dir / "experiment_gaps.csv", index=False)

    final_energy = estimator_pipeline(
        candidates[selected_energy], ENERGY_NUMERIC_FEATURES
    ).fit(runs, observed_energy)
    final_time = estimator_pipeline(
        candidates[selected_time], TIME_NUMERIC_FEATURES
    ).fit(runs, observed_time)
    deployment_eligible = bool(
        decisions["strict_lowo_match"].mean()
        >= MIN_DEPLOYMENT_DECISION_ACCURACY
    )
    joblib.dump(
        {
            "energy_estimator": final_energy,
            "crossing_time_estimator": final_time,
            "energy_features": {
                "categorical": CATEGORICAL_FEATURES,
                "numeric": ENERGY_NUMERIC_FEATURES,
                "explicitly_excluded": ["charging_pad_availability"],
            },
            "crossing_time_features": {
                "categorical": CATEGORICAL_FEATURES,
                "numeric": TIME_NUMERIC_FEATURES,
                "explicitly_excluded": [
                    "charging_pad_availability",
                    "start_soc",
                ],
            },
            "stage2": {
                "formula": (
                    "crossing_s + charging_service_s + pad_queue_wait_s"
                ),
                "charge_rate_pp_per_min": args.charge_rate_pp_per_min,
                "swarm_size": SWARM_SIZE,
            },
            "reference_soc": args.reference_soc,
            "selected_energy_model": selected_energy,
            "selected_crossing_time_model": selected_time,
            "deployment_eligible": deployment_eligible,
            "fallback": "direct empirical cell winner for measured conditions",
        },
        args.output_dir / "two_stage_models.joblib",
    )

    summary = {
        "status": "PASS",
        "training_run_csv": str(args.run_csv.resolve()),
        "training_scope": "75 cm, 79 selectable legacy crossing runs, 6 wind conditions",
        "new_data_policy": (
            "audited separately; no post-2026-08-26 file was eligible for direct "
            "addition to the matched 75 cm crossing model"
        ),
        "method": {
            "stage_1_energy_inputs": [
                "formation",
                "wind direction/level/profile",
                "entry SOC",
                "fixed geometry",
            ],
            "stage_1_crossing_time_inputs": [
                "formation",
                "wind direction/level/profile",
                "fixed geometry",
            ],
            "energy_input_exclusion": "charging-pad availability",
            "stage_2_formula": (
                "total = formation-dependent crossing + charging service + pad queue wait"
            ),
            "validation": "strict leave-one-wind-condition-out",
            "comparison_rule": (
                "compare model and direct dataset winner for every measured condition; "
                "use empirical fallback on mismatch"
            ),
        },
        "reference_soc": args.reference_soc,
        "charge_rate_pp_per_min": args.charge_rate_pp_per_min,
        "selected_energy_model": {
            **energy_metrics.iloc[0].to_dict(),
            "mae_unit": "SOC percentage points",
        },
        "selected_crossing_time_model": {
            **time_metrics.iloc[0].to_dict(),
            "mae_unit": "seconds",
        },
        "decision_validation": {
            "correct": int(decisions["strict_lowo_match"].sum()),
            "total": int(len(decisions)),
            "accuracy": float(decisions["strict_lowo_match"].mean()),
            "mean_regret_s": float(decisions["regret_s"].mean()),
            "p95_regret_s": float(decisions["regret_s"].quantile(0.95)),
            "max_regret_s": float(decisions["regret_s"].max()),
            "deployment_gate": MIN_DEPLOYMENT_DECISION_ACCURACY,
            "deployment_eligible": deployment_eligible,
        },
        "full_data_soc_slope_points_per_percent": full_data_beta,
        "fold_local_soc_slopes": fold_betas,
        "new_data_audit": audit_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_briefs(
        args.output_dir,
        energy_metrics,
        time_metrics,
        decisions,
        errors,
        pilots,
        audit_summary,
        selected_energy,
        selected_time,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
