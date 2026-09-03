from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "research_defensibility_audit"
SELECTED_RUNS = (
    ROOT
    / "analysis_outputs"
    / "forward_discharge_rate_modeling"
    / "selected_runs_by_database_cell.csv"
)
SOC_RUNS = (
    ROOT
    / "analysis_outputs"
    / "initial_soc_effect_study"
    / "run_level_soc_rate_data.csv"
)
SOC_SUMMARY = (
    ROOT
    / "analysis_outputs"
    / "initial_soc_effect_study"
    / "analysis_summary.json"
)

CELL_KEYS = [
    "formation",
    "inter_drone_spacing_cm",
    "wind_direction",
    "wind_level",
]


def _feature_matrix(soc: pd.Series | list[float], kind: str) -> np.ndarray:
    values = np.asarray(soc, dtype=float)
    centered = (values - 65.0) / 10.0
    if kind == "linear":
        return centered[:, None]
    if kind == "quadratic":
        return np.column_stack((centered, centered**2))
    if kind == "three_bands":
        # Illustrative thresholds only. High (>=68%) is the reference band.
        return np.column_stack(
            (
                (values < 58.0).astype(float),
                ((values >= 58.0) & (values < 68.0)).astype(float),
            )
        )
    raise ValueError(kind)


def _fit_with_condition_fixed_effects(
    training: pd.DataFrame,
    kind: str,
) -> np.ndarray:
    x = _feature_matrix(training["run_start_soc_mean_pct"], kind)
    y = training["run_rate_mean_pp_per_min"].to_numpy(dtype=float)
    x_within = np.empty_like(x)
    y_within = np.empty_like(y)
    for _, source_indices in training.groupby("condition").groups.items():
        positions = training.index.get_indexer(source_indices)
        x_within[positions] = x[positions] - x[positions].mean(axis=0)
        y_within[positions] = y[positions] - y[positions].mean()
    return np.linalg.lstsq(x_within, y_within, rcond=None)[0]


def build_soc_cross_validation(run_data: pd.DataFrame) -> pd.DataFrame:
    counts = run_data.groupby("condition").size()
    evaluation = run_data[
        run_data["condition"].isin(counts[counts >= 3].index)
    ].copy()
    predictions: list[dict[str, object]] = []
    for row_index, test in evaluation.iterrows():
        training = run_data.drop(index=row_index).reset_index(drop=True)
        same_cell = training[training["condition"] == test["condition"]]
        record: dict[str, object] = {
            "condition": test["condition"],
            "experiment_directory": test["experiment_directory"],
            "run_id": test["run_id"],
            "start_soc_pct": float(test["run_start_soc_mean_pct"]),
            "actual_rate_pp_per_min": float(test["run_rate_mean_pp_per_min"]),
            "static_prediction_pp_per_min": float(
                same_cell["run_rate_mean_pp_per_min"].mean()
            ),
        }
        for kind in ("three_bands", "linear", "quadratic"):
            coefficients = _fit_with_condition_fixed_effects(training, kind)
            same_x = _feature_matrix(same_cell["run_start_soc_mean_pct"], kind)
            intercept = float(
                np.mean(
                    same_cell["run_rate_mean_pp_per_min"].to_numpy(dtype=float)
                    - same_x @ coefficients
                )
            )
            test_x = _feature_matrix([float(test["run_start_soc_mean_pct"])], kind)[0]
            record[f"{kind}_prediction_pp_per_min"] = float(
                intercept + test_x @ coefficients
            )
        predictions.append(record)
    return pd.DataFrame(predictions)


def summarize_cross_validation(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in ("static", "three_bands", "linear", "quadratic"):
        error = (
            predictions[f"{method}_prediction_pp_per_min"]
            - predictions["actual_rate_pp_per_min"]
        ).to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "held_out_runs": len(predictions),
                "represented_conditions": predictions["condition"].nunique(),
                "mae_pp_per_min": float(np.mean(np.abs(error))),
                "rmse_pp_per_min": float(np.sqrt(np.mean(error**2))),
                "median_absolute_error_pp_per_min": float(np.median(np.abs(error))),
            }
        )
    return pd.DataFrame(rows)


def build_coverage(selected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    chosen = selected[selected["selection_status"] == "selected"].copy()
    cells = (
        chosen.groupby(CELL_KEYS, dropna=False)
        .agg(
            selected_run_count=("run_id", "nunique"),
            eligible_run_count=("cell_eligible_run_count", "max"),
            selected_start_soc_min_pct=("run_start_soc_mean_pct", "min"),
            selected_start_soc_max_pct=("run_start_soc_mean_pct", "max"),
        )
        .reset_index()
    )
    cells["selected_start_soc_span_pp"] = (
        cells["selected_start_soc_max_pct"] - cells["selected_start_soc_min_pct"]
    )
    cells["additional_to_3_total"] = np.maximum(0, 3 - cells["eligible_run_count"])
    cells["additional_to_4_total"] = np.maximum(0, 4 - cells["eligible_run_count"])
    cells["additional_to_5_total"] = np.maximum(0, 5 - cells["eligible_run_count"])

    formal = selected[selected["formal_clean_trajectory_candidate"] == True].copy()  # noqa: E712

    def controlled_band(row: pd.Series) -> str:
        low = float(row["run_start_soc_min_pct"])
        high = float(row["run_start_soc_max_pct"])
        if low >= 68.0:
            return "high_all"
        if low >= 58.0 and high < 68.0:
            return "middle_all"
        if high < 58.0:
            return "low_all"
        return "mixed"

    formal["illustrative_controlled_soc_band"] = formal.apply(controlled_band, axis=1)
    band_table = formal.pivot_table(
        index=CELL_KEYS,
        columns="illustrative_controlled_soc_band",
        values="run_id",
        aggfunc="nunique",
        fill_value=0,
    )
    band_table = band_table.reindex(
        columns=["low_all", "middle_all", "high_all", "mixed"], fill_value=0
    )
    band_table = band_table.reset_index()
    cells = cells.merge(band_table, on=CELL_KEYS, how="left").fillna(
        {"low_all": 0, "middle_all": 0, "high_all": 0, "mixed": 0}
    )

    summary = {
        "observed_rate_cells": int(len(cells)),
        "selected_runs": int(cells["selected_run_count"].sum()),
        "eligible_runs_across_observed_cells": int(cells["eligible_run_count"].sum()),
        "cells_with_1_selected_run": int((cells["selected_run_count"] == 1).sum()),
        "cells_with_2_selected_runs": int((cells["selected_run_count"] == 2).sum()),
        "cells_with_3_selected_runs": int((cells["selected_run_count"] == 3).sum()),
        "cells_with_at_least_5_eligible_runs": int((cells["eligible_run_count"] >= 5).sum()),
        "additional_runs_to_reach_3_total_per_observed_cell": int(
            cells["additional_to_3_total"].sum()
        ),
        "additional_runs_to_reach_4_total_per_observed_cell": int(
            cells["additional_to_4_total"].sum()
        ),
        "additional_runs_to_reach_5_total_per_observed_cell": int(
            cells["additional_to_5_total"].sum()
        ),
        "two_new_runs_per_observed_cell": int(2 * len(cells)),
        "three_new_runs_per_observed_cell": int(3 * len(cells)),
        "illustrative_soc_thresholds": {
            "low_all": "all five start below 58%",
            "middle_all": "all five start in [58%, 68%)",
            "high_all": "all five start at or above 68%",
        },
        "formal_candidate_runs_by_illustrative_band": {
            key: int(value)
            for key, value in formal["illustrative_controlled_soc_band"]
            .value_counts()
            .to_dict()
            .items()
        },
        "cells_with_all_three_controlled_bands": int(
            (
                cells[["low_all", "middle_all", "high_all"]]
                .gt(0)
                .all(axis=1)
            ).sum()
        ),
    }
    return cells, summary


def claim_matrix() -> list[dict[str, str]]:
    return [
        {
            "claim": "The pipeline estimates a relative forward-flight reported-SOC discharge proxy over the first 2.5 m.",
            "status": "defensible_with_scope",
            "evidence": "Auditable trajectory reconstruction, forward-only mask, Bideal scaling, 156 selected runs, 780 run-drone rows, and threshold sensitivity checks.",
            "risk": "DJI SOC is integer-valued and is not electrical energy in Wh.",
            "required_action": "Use pp/min and 'reported-SOC discharge proxy'; do not claim absolute power or causal aerodynamics.",
        },
        {
            "claim": "One SOC-independent rate is valid for every starting SOC within a condition and slot.",
            "status": "not_supported",
            "evidence": "Lower-SOC runs have higher five-drone mean rates in 50/53 comparable cells; the estimated common effect is +3.56 pp/min per 10 percentage-point lower SOC.",
            "risk": "Pooling low, middle, and high runs produces a rate that is wrong for the online SOC state.",
            "required_action": "Replace the static table with r(c,p,s)=alpha(c,p)+g(s) and validate g on held-out physical runs.",
        },
        {
            "claim": "Three independent low/middle/high final models are necessary.",
            "status": "not_recommended",
            "evidence": "Three-band, linear, and quadratic SOC corrections have similar leave-one-run-out errors; the continuous forms avoid threshold jumps.",
            "risk": "Separate models discard information, create discontinuities, and are awkward in a 30 s dynamic controller.",
            "required_action": "Use low/middle/high only as controlled collection anchors; fit one continuous SOC-aware model.",
        },
        {
            "claim": "The measured slot rate can be transferred to any of the 5! drone-to-slot permutations.",
            "status": "not_supported",
            "evidence": "Drone, battery, and nominal slot were not rotated; the method document itself states that slot and hardware effects cannot be separated.",
            "risk": "The current optimizer treats a confounded observed drone-slot association as a transferable geometric position effect.",
            "required_action": "Either remove free position permutation from the defensible method, or add a crossed/cyclic drone-battery-to-slot rotation experiment.",
        },
        {
            "claim": "Formation and spacing can be compared under the observed fixed hardware mapping.",
            "status": "provisional_with_uncertainty",
            "evidence": "Most observed cells have at least three eligible runs and the processing is reproducible.",
            "risk": "Replication is unbalanced, replicate slot rankings are weak, SOC and run order are confounded, and four nominal cells lack an algorithm-facing rate.",
            "required_action": "Use partial pooling, uncertainty intervals, held-out runs, and bounded claims about the indoor testbed.",
        },
        {
            "claim": "The 5,000 generated base states expand the real experimental dataset.",
            "status": "false_wording",
            "evidence": "The generator starts from a fitted empirical table and simulates reachable SOC histories and remaining distances.",
            "risk": "Synthetic rows cannot add independent physical evidence for assumptions used to generate them.",
            "required_action": "Call them synthetic decision states or model-generated training scenarios, never additional real flight data.",
        },
        {
            "claim": "The oracle labels are physically correct and globally optimal.",
            "status": "model_internal_only",
            "evidence": "Enumeration and charging scheduling are exact conditional on the supplied rate, charging, safety, distance, and zero-transition-cost model.",
            "risk": "Exact optimization of an uncertain surrogate does not establish the real-world optimum.",
            "required_action": "Use 'model-based reference optimum' and 'optimal under the stated surrogate model'.",
        },
        {
            "claim": "98% grouped cross-validation proves real-world controller correctness.",
            "status": "not_supported",
            "evidence": "Training and validation labels are generated by the same empirical rate and charging model; grouped splitting only prevents synthetic state leakage.",
            "risk": "The metric measures imitation/search fidelity to the reference solver, not physical energy or mission performance.",
            "required_action": "Report model-consistency accuracy separately and add held-out physical predictive and end-to-end validation.",
        },
        {
            "claim": "The 90-minute exponential charging model is physically calibrated for the five batteries.",
            "status": "not_yet_supported",
            "evidence": "The code anchors 0% to 99% at 90 minutes; no battery-specific charging traces are referenced in the current methodology.",
            "risk": "Charging assumptions can change the objective and the selected configuration, especially as K changes.",
            "required_action": "Measure charging curves or present charging as a scenario assumption with sensitivity analysis.",
        },
        {
            "claim": "Rates measured over 2.5 m predict flights up to 25 m without further validation.",
            "status": "not_supported",
            "evidence": "The algorithm linearly multiplies pp/min by flight time over 0.25-25 m.",
            "risk": "SOC dependence, thermal drift, wind persistence, and integer telemetry make this a long-range extrapolation.",
            "required_action": "Validate longer sequences physically or restrict the claim to short receding-horizon segments with recalibration.",
        },
        {
            "claim": "The controller is dynamic.",
            "status": "algorithmically_true_physically_unvalidated",
            "evidence": "It refreshes wind, K, remaining distance, and SOC every 30 s.",
            "risk": "Reconfiguration time, energy, and transient collision risk are set to zero, so dynamic switching performance is not physically established.",
            "required_action": "Describe it as a dynamic decision policy evaluated in simulation; validate transition costs before claiming a deployed dynamic controller.",
        },
        {
            "claim": "All four missing nominal cells are documented consistently as safety omissions.",
            "status": "internally_inconsistent",
            "evidence": "The rate table has 56 of 60 nominal cells; the exception note lists two Column-50 collisions, while the optimizer also masks Diamond-50 Side-L2 and Tail-L2 as repeated-collision cells.",
            "risk": "Unreconciled exclusion reasons look post hoc and can change the feasible set.",
            "required_action": "Reconcile the registry, exception note, rate-table exclusions, and safety mask before further model evaluation.",
        },
        {
            "claim": "The terms primary 75%-40% runs and selected algorithm-facing runs refer to the same population.",
            "status": "terminology_needs_clarification",
            "evidence": "The forward-mask validation reports 168 strict common-window runs, while the rate model selects 156 runs under battery-specific lower bounds; 134 selected runs carry the old primary label and 22 carry a sensitivity-only label.",
            "risk": "Readers can reasonably infer that sensitivity-only rows were accidentally used in the final rate table, even though the newer eligibility code accepts them under battery-specific 30%-40% lower limits.",
            "required_action": "Name and count the two analysis populations separately and explain why the newer battery-specific eligibility supersedes the older strict 75%-40% label for algorithm calibration.",
        },
    ]


def write_claim_matrix(rows: list[dict[str, str]]) -> None:
    path = OUT / "claim_evidence_limitation_matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    coverage: dict[str, object],
    soc_summary: dict[str, object],
    cv_summary: pd.DataFrame,
) -> None:
    cv = cv_summary.set_index("method")
    report = f"""# Research Defensibility Audit

## Bottom line

The existing physical dataset and cleaning pipeline can support a bounded empirical measurement contribution. The current end-to-end algorithm claim is not yet defensible as a physically validated optimizer. It can be made coherent without discarding the existing flights by separating three layers:

1. **Physical evidence:** observed, forward-only, Bideal-normalized reported-SOC discharge over a 2.5 m indoor segment.
2. **Empirical surrogate:** a continuous SOC-aware discharge model with uncertainty and physical holdout validation.
3. **Decision simulation:** synthetic states, a model-based reference optimizer, and ML fidelity to that reference model.

The thesis must not use evidence from layer 3 as if it independently validates layer 1 or layer 2.

## What the current data genuinely supports

- {coverage['selected_runs']} selected physical runs and {int(coverage['selected_runs']) * 5} selected run-drone rows contribute to the current rate analysis.
- {coverage['observed_rate_cells']} observed condition-structure cells have an algorithm-facing rate. The nominal 5 x 2 x 3 x 2 design has 60 cells, so four nominal cells are absent from that table.
- Selected replication is 47 cells with three runs, six with two, and three with one.
- Across all currently eligible flights in the observed rate cells, {coverage['eligible_runs_across_observed_cells']} runs exist; the current method down-selects some cells to three SOC-representative runs.
- The defensible response is a relative reported-SOC discharge proxy in percentage points per minute, not Wh and not an isolated aerodynamic position effect.

## Why a continuous SOC model is the best route

The current evidence contradicts a SOC-independent lookup rate: {soc_summary['lower_soc_higher_rate_conditions']}/{soc_summary['comparable_conditions']} comparable cells have a higher five-drone mean rate in the lower-start-SOC run. The within-condition estimate is +{soc_summary['run_mean_fixed_effect_model']['effect_of_10pp_lower_soc_pp_per_min']:.2f} pp/min for a 10 percentage-point lower starting SOC (condition-bootstrap 95% interval {soc_summary['run_mean_fixed_effect_model']['bootstrap_ci_low']:.2f} to {soc_summary['run_mean_fixed_effect_model']['bootstrap_ci_high']:.2f}). This is strong evidence of predictive dependence, but not clean causal evidence because SOC is confounded with trial order and within-swarm imbalance.

Leave-one-run-out prediction was evaluated for the 47 cells with at least three selected runs ({int(cv.loc['static', 'held_out_runs'])} held-out predictions):

| Model | MAE (pp/min) | RMSE (pp/min) |
|---|---:|---:|
| Static condition mean | {cv.loc['static', 'mae_pp_per_min']:.3f} | {cv.loc['static', 'rmse_pp_per_min']:.3f} |
| Three SOC bands | {cv.loc['three_bands', 'mae_pp_per_min']:.3f} | {cv.loc['three_bands', 'rmse_pp_per_min']:.3f} |
| Continuous linear SOC term | {cv.loc['linear', 'mae_pp_per_min']:.3f} | {cv.loc['linear', 'rmse_pp_per_min']:.3f} |
| Continuous quadratic SOC term | {cv.loc['quadratic', 'mae_pp_per_min']:.3f} | {cv.loc['quadratic', 'rmse_pp_per_min']:.3f} |

The small difference among the three SOC-aware forms does not justify three disconnected final models. Use low, middle, and high SOC as controlled collection anchors, then fit one continuous model such as:

`r(c, p, s) = alpha(c, p) + g(s)`

Start with a common linear `g(s)` and retain the quadratic or a low-complexity spline only if a frozen physical holdout shows a material improvement. Online SOC should then be integrated segment by segment rather than multiplied by one constant rate over 25 m.

## The highest-risk structural problem: position permutation

The data-processing chapter correctly says that drone, battery, and nominal slot were fixed and therefore not causally separable. The algorithm chapter nevertheless allows all 5! drone-to-slot permutations and assigns the measured slot rate to any physical drone. Bideal normalization reduces battery-rate differences but does not identify a transferable geometric slot effect.

There are only two defensible choices:

- **Low-experiment route:** remove arbitrary position permutations from the physical claim and optimize formation plus spacing under the observed mapping. Position optimization may remain a clearly labelled simulation-only ablation.
- **Position-thesis route:** add a prospective cyclic/crossed rotation experiment so every drone-battery pair occupies multiple slots. Fit separate drone/battery nuisance effects and slot effects, and validate unseen assignments.

Simply adding more repetitions with the same fixed mapping does not solve this identification problem.

## Replication: what can and cannot be defended

The written advice to add 2-3 flights under every condition corresponds to {coverage['two_new_runs_per_observed_cell']}-{coverage['three_new_runs_per_observed_cell']} new flights across the 56 observed cells. Existing valid flights can reduce the total only if the supervisor agrees that previously unselected eligible runs count toward the requested replication. Numerically:

- reach at least 3 eligible runs per observed cell: {coverage['additional_runs_to_reach_3_total_per_observed_cell']} additional flights;
- reach at least 4: {coverage['additional_runs_to_reach_4_total_per_observed_cell']};
- reach at least 5: {coverage['additional_runs_to_reach_5_total_per_observed_cell']}.

These totals address count only, not controlled SOC coverage. Using illustrative bands of all-five-low (<58%), all-five-middle (58-<68%), and all-five-high (>=68%), only {coverage['cells_with_all_three_controlled_bands']} cells currently contain all three controlled anchors. The precise thresholds must be agreed prospectively; they must not be invented after looking at outcomes.

If full additional replication is impossible, the thesis must shrink its claim and obtain written approval for a pre-specified validation panel. It cannot honestly claim that a targeted subset satisfies an instruction to repeat every condition unless the supervisor explicitly changes that requirement.

## Minimum coherent validation hierarchy

1. Freeze inclusion, outlier, SOC-band, safety, and preprocessing rules before new data.
2. Keep all existing eligible runs for model development; do not discard valid repetitions merely because a cell has more than three.
3. Reserve every new controlled flight as a session-grouped physical holdout until model choices are frozen.
4. Validate segment-level discharge first: error, bias, interval coverage, and failures by SOC, wind, formation, and spacing.
5. Validate formation/spacing decisions second on held-out physical conditions or sessions.
6. Validate position assignment only if crossed slot rotations were collected.
7. Call the 5,000 groups synthetic decision states. Report ML accuracy as fidelity to a model-based reference optimizer, not physical correctness.
8. Treat the charging curve, 25 m extrapolation, 30 s interval, and zero reconfiguration cost as assumptions unless separately measured; run sensitivity analysis for each.

## Wording replacements that prevent the professor's “you assume it is correct” criticism

| Avoid | Use instead |
|---|---|
| correct labels | model-generated reference labels |
| globally optimal configuration | optimal configuration under the stated surrogate model |
| independent validation set (for synthetic states) | independent synthetic test set with no seed/group overlap |
| 5,000 expanded real data | 5,000 synthetic decision-state groups generated from the empirical surrogate |
| position-dependent aerodynamic energy | slot-associated reported-SOC rate under the fixed hardware mapping |
| the controller is validated | the controller is internally validated against the reference model; physical validation remains separate |

## Immediate corrections before presenting results

1. Reconcile the four absent nominal cells. The exception note documents two Column-50 collision omissions, while the optimizer additionally marks Diamond-50 Side-L2 and Tail-L2 as repeated-collision structures. The data show different reasons for at least some of these cells.
2. Remove or qualify every occurrence of “correct labels,” “globally optimal,” and “independent validation” that lacks the phrase “under the stated surrogate model” or “synthetic.”
3. Stop pooling SOC ranges into one constant online rate.
4. Decide whether position permutation is removed from the physical claim or supported by a crossed experiment.
5. Separate algorithmic dynamicity from physical switching validation: the decision logic is dynamic, but transition time, energy, and collision risk are currently unmeasured.
6. Clarify the two run populations: 168 strict common 75%-40% runs were used in one preprocessing validation, whereas the algorithm-facing selection uses battery-specific 30%-40% lower bounds and yields 156 runs. Of those 156, 22 still carry the older `held_outside_75_to_40_for_sensitivity_only` label. This is explainable, but the current terminology makes it look accidental.
"""
    (OUT / "research_defensibility_audit.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SELECTED_RUNS)
    soc_runs = pd.read_csv(SOC_RUNS)
    soc_summary = json.loads(SOC_SUMMARY.read_text(encoding="utf-8"))

    coverage_frame, coverage_summary = build_coverage(selected)
    coverage_frame.to_csv(OUT / "condition_replication_and_soc_coverage.csv", index=False)

    predictions = build_soc_cross_validation(soc_runs)
    predictions.to_csv(OUT / "soc_model_leave_one_run_out_predictions.csv", index=False)
    cv_summary = summarize_cross_validation(predictions)
    cv_summary.to_csv(OUT / "soc_model_cross_validation_summary.csv", index=False)

    claims = claim_matrix()
    write_claim_matrix(claims)
    write_report(coverage_summary, soc_summary, cv_summary)

    audit_summary = {
        "coverage": coverage_summary,
        "soc_cross_validation": cv_summary.to_dict(orient="records"),
        "claim_status_counts": {
            str(key): int(value)
            for key, value in pd.Series(
                [row["status"] for row in claims]
            ).value_counts().items()
        },
    }
    (OUT / "audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote research defensibility audit to {OUT}")


if __name__ == "__main__":
    main()
