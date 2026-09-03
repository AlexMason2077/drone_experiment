"""Exhaustively re-solve every Oracle-labelled training state and audit labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

from ml_policy.oracle_optimizer import EmpiricalRateTable, OracleState, SafetyTier, solve_oracle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CSV = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "ml_policy"
    / "expanded_25m_exponential_90min_interval30s"
    / "oracle_training_states_0p25_25m.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_states(
    training_csv: Path,
    output_json: Path,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    manifest_path = training_csv.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rate_table_path = Path(manifest["source_rate_table"])
    rate_table = EmpiricalRateTable.from_csv(rate_table_path)

    mismatch_examples: list[dict[str, object]] = []
    checked_rows = 0
    full_configuration_matches = 0
    structure_matches = 0
    position_matches = 0
    time_matches = 0
    charging_time_matches = 0
    safe_selected_rows = 0
    backup_selected_rows = 0
    started = time.perf_counter()

    with training_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            state = OracleState(
                wind_direction=row["wind_direction"],
                wind_level=int(row["wind_level"]),
                charging_pad_count=int(row["charging_pad_count"]),
                current_soc=tuple(float(row[f"soc_d{index}"]) for index in range(1, 6)),
                remaining_distance_m=float(row["remaining_distance_m"]),
                forward_speed_m_per_s=float(manifest["forward_speed_m_per_s"]),
                fully_charged_soc=float(manifest["fully_charged_soc"]),
                zero_to_fully_charged_minutes=float(
                    manifest["zero_to_fully_charged_minutes"]
                ),
                minimum_arrival_soc=float(manifest["minimum_arrival_soc"]),
            )
            result = solve_oracle(state, rate_table)
            recalculated_structure = result.selected.structure.label
            recalculated_position = result.selected.position_mapping(state.drone_ids)
            stored_position = json.loads(row["oracle_position_json"])
            recalculated_total = result.selected.total_completion_minutes
            recalculated_charging = result.selected.charging_schedule.makespan_minutes

            structure_ok = recalculated_structure == row["oracle_structure"]
            position_ok = recalculated_position == stored_position
            time_ok = abs(recalculated_total - float(row["oracle_total_minutes"])) <= tolerance
            charging_ok = (
                abs(recalculated_charging - float(row["oracle_charging_minutes"]))
                <= tolerance
            )
            full_ok = structure_ok and position_ok and time_ok and charging_ok

            checked_rows += 1
            structure_matches += int(structure_ok)
            position_matches += int(position_ok)
            time_matches += int(time_ok)
            charging_time_matches += int(charging_ok)
            full_configuration_matches += int(full_ok)
            safe_selected_rows += int(result.selected.safety_tier == SafetyTier.SAFE)
            backup_selected_rows += int(result.selected.safety_tier == SafetyTier.BACKUP_ONLY)

            if not full_ok and len(mismatch_examples) < 20:
                mismatch_examples.append(
                    {
                        "scenario_id": int(row["scenario_id"]),
                        "stored_structure": row["oracle_structure"],
                        "recalculated_structure": recalculated_structure,
                        "stored_position": stored_position,
                        "recalculated_position": recalculated_position,
                        "stored_total_minutes": float(row["oracle_total_minutes"]),
                        "recalculated_total_minutes": recalculated_total,
                    }
                )
            if checked_rows % 250 == 0:
                print(f"Validated {checked_rows} rows")

    elapsed_seconds = time.perf_counter() - started
    report: dict[str, object] = {
        "status": "pass" if full_configuration_matches == checked_rows else "fail",
        "definition": (
            "Every row's stored formation, spacing, drone-to-slot position, total time, "
            "and charging makespan must exactly match a fresh exhaustive Oracle solve."
        ),
        "training_csv": str(training_csv),
        "training_csv_sha256": _sha256(training_csv),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "rate_table": str(rate_table_path),
        "rate_table_sha256": _sha256(rate_table_path),
        "checked_rows": checked_rows,
        "structure_matches": structure_matches,
        "position_matches": position_matches,
        "time_matches": time_matches,
        "charging_time_matches": charging_time_matches,
        "full_configuration_matches": full_configuration_matches,
        "safe_selected_rows": safe_selected_rows,
        "backup_selected_rows": backup_selected_rows,
        "mismatch_count": checked_rows - full_configuration_matches,
        "mismatch_examples": mismatch_examples,
        "elapsed_seconds": elapsed_seconds,
        "model_scope_caveat": (
            "This proves exact optimality under the empirical-rate, constant-speed, "
            "90-minute exponential-charging, zero-switching-cost model. It does not "
            "independently prove that rates "
            "measured over 2.5 m remain physically constant over 25 m."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_json = args.output_json or args.training_csv.with_suffix(".validation.json")
    report = validate_training_states(args.training_csv, output_json)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
