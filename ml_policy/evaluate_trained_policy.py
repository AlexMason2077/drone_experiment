"""Evaluate a saved policy on a separate Oracle-labelled state dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from ml_policy.train_gradient_boosted_policy import _decision_metrics
from ml_policy.oracle_optimizer import EmpiricalRateTable, OracleState, solve_oracle
from ml_policy.predict_configuration import predict_configuration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
)
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "gradient_boosted_policy.joblib"
DEFAULT_VALIDATION_CSV = DEFAULT_OUTPUT_DIR / "oracle_validation_states_1000.csv"


def evaluate_saved_policy(
    model_path: Path,
    validation_csv: Path,
    output_dir: Path,
) -> dict[str, object]:
    model = joblib.load(model_path)
    frame = pd.read_csv(validation_csv)
    metrics, predictions = _decision_metrics(model, frame)

    by_scenario = frame.set_index("scenario_id")
    reranked_times: list[float] = []
    reranked_labels: list[str] = []
    candidate_counts: list[int] = []
    for _, prediction in predictions.iterrows():
        row = by_scenario.loc[prediction["scenario_id"]]
        candidates = prediction["predicted_top2"].split("|")
        candidate_counts.append(len(candidates))
        candidate_times = {
            label: float(row[f"time__{label}"])
            for label in candidates
        }
        selected_label = min(candidate_times, key=lambda label: (candidate_times[label], label))
        reranked_labels.append(selected_label)
        reranked_times.append(candidate_times[selected_label])

    predictions["top2_reranked_structure"] = reranked_labels
    predictions["top2_reranked_total_minutes"] = reranked_times
    predictions["top2_reranked_regret_minutes"] = (
        predictions["top2_reranked_total_minutes"]
        - predictions["oracle_total_minutes"]
    )
    reranked_regret = predictions["top2_reranked_regret_minutes"].to_numpy(dtype=float)
    metrics["top2_reranked_oracle_match"] = float(
        np.isclose(reranked_regret, 0.0, atol=1e-9).mean()
    )
    metrics["top2_reranked_mean_regret_minutes"] = float(reranked_regret.mean())
    metrics["top2_reranked_maximum_regret_minutes"] = float(reranked_regret.max())
    metrics["mean_candidate_count"] = float(np.mean(candidate_counts))

    validation_manifest = json.loads(
        validation_csv.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    benchmark_rows = frame.head(min(100, len(frame)))
    benchmark_states = [
        OracleState(
            wind_direction=row["wind_direction"],
            wind_level=int(row["wind_level"]),
            charging_pad_count=int(row["charging_pad_count"]),
            current_soc=tuple(float(row[f"soc_d{index}"]) for index in range(1, 6)),
            remaining_distance_m=float(row["remaining_distance_m"]),
            forward_speed_m_per_s=float(validation_manifest["forward_speed_m_per_s"]),
            fully_charged_soc=float(validation_manifest["fully_charged_soc"]),
            zero_to_fully_charged_minutes=float(
                validation_manifest["zero_to_fully_charged_minutes"]
            ),
            minimum_arrival_soc=float(validation_manifest["minimum_arrival_soc"]),
        )
        for _, row in benchmark_rows.iterrows()
    ]
    rate_table_path = Path(validation_manifest["source_rate_table"])
    rate_table = EmpiricalRateTable.from_csv(rate_table_path)
    # Warm the model/rate-table caches before measuring steady-state online use.
    predict_configuration(
        benchmark_states[0],
        model_path=model_path,
        rate_table_path=rate_table_path,
        top_k=2,
    )
    policy_started = time.perf_counter()
    for state in benchmark_states:
        predict_configuration(
            state,
            model_path=model_path,
            rate_table_path=rate_table_path,
            top_k=2,
        )
    policy_elapsed_ms = (time.perf_counter() - policy_started) * 1000.0

    oracle_started = time.perf_counter()
    for state in benchmark_states:
        solve_oracle(state, rate_table)
    oracle_elapsed_ms = (time.perf_counter() - oracle_started) * 1000.0
    policy_mean_ms = policy_elapsed_ms / len(benchmark_states)
    oracle_mean_ms = oracle_elapsed_ms / len(benchmark_states)
    metrics["online_benchmark_rows"] = float(len(benchmark_states))
    metrics["warm_policy_top2_end_to_end_ms_per_state"] = policy_mean_ms
    metrics["fresh_oracle_end_to_end_ms_per_state"] = oracle_mean_ms
    metrics["end_to_end_online_speedup"] = oracle_mean_ms / policy_mean_ms

    model_metadata_path = model_path.with_name("gradient_boosted_policy_metrics.json")
    model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    training_csv = Path(model_metadata["training_csv"])
    training_seeds = set(pd.read_csv(training_csv, usecols=["scenario_seed"])["scenario_seed"])
    validation_seeds = set(frame["scenario_seed"])

    report: dict[str, object] = {
        "status": "pass",
        "model": str(model_path),
        "training_csv": str(training_csv),
        "validation_csv": str(validation_csv),
        "validation_rows": len(frame),
        "training_validation_seed_overlap_count": len(training_seeds & validation_seeds),
        "metrics": metrics,
    }
    if report["training_validation_seed_overlap_count"] != 0:
        report["status"] = "fail"

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "independent_validation_predictions.csv", index=False)
    (output_dir / "independent_validation_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_saved_policy(args.model, args.validation_csv, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
