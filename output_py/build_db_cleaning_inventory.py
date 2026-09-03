from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "database"
COPY = ROOT / "db_copy_for_cleaning"
ADMIN = COPY / "_cleaning_admin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(base: Path, exclude_admin: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        relative = path.relative_to(base)
        if exclude_admin and relative.parts and relative.parts[0] == "_cleaning_admin":
            continue
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ADMIN.mkdir(parents=True, exist_ok=True)
    source_rows = manifest(SOURCE)
    copy_rows = manifest(COPY, exclude_admin=True)

    fields = ["relative_path", "size_bytes", "sha256"]
    write_csv(ADMIN / "source_file_manifest_sha256.csv", source_rows, fields)
    write_csv(ADMIN / "copy_file_manifest_before_cleaning_sha256.csv", copy_rows, fields)

    source_by_path = {str(row["relative_path"]): row for row in source_rows}
    copy_by_path = {str(row["relative_path"]): row for row in copy_rows}
    reconciliation: list[dict[str, object]] = []
    for relative in sorted(set(source_by_path) | set(copy_by_path)):
        source_row = source_by_path.get(relative)
        copy_row = copy_by_path.get(relative)
        if source_row is None:
            status = "copy_only"
        elif copy_row is None:
            status = "source_only"
        elif source_row["sha256"] == copy_row["sha256"]:
            status = "identical"
        else:
            status = "content_diff"
        reconciliation.append(
            {
                "relative_path": relative,
                "status": status,
                "source_size_bytes": "" if source_row is None else source_row["size_bytes"],
                "copy_size_bytes": "" if copy_row is None else copy_row["size_bytes"],
                "source_sha256": "" if source_row is None else source_row["sha256"],
                "copy_sha256": "" if copy_row is None else copy_row["sha256"],
            }
        )
    write_csv(
        ADMIN / "source_copy_reconciliation.csv",
        reconciliation,
        [
            "relative_path",
            "status",
            "source_size_bytes",
            "copy_size_bytes",
            "source_sha256",
            "copy_sha256",
        ],
    )

    status_counts: dict[str, int] = {}
    for row in reconciliation:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    rules = {
        "recorded_on": "2026-08-12",
        "source_directory": "database",
        "cleaning_directory": "db_copy_for_cleaning",
        "source_mutation_allowed": False,
        "analysis_scope": {
            "formal_candidate_name_rule": "top-level directory contains 'new' case-insensitively and contains neither 'pre' nor 'perpare'",
            "pre_and_perpare": "exclude",
            "no_wind": "retain but do not analyse at this stage",
            "summary_directories": "retain and mark as derived_summary; do not treat as raw runs",
            "multiple_timestamps_in_one_directory": "treat each timestamp/run_id as an independent candidate run",
        },
        "valid_run_definition": {
            "required_drone_count": 5,
            "require_all_drones_complete_target_distance": True,
            "nominal_speed_cm_s": 10,
            "comparison_distance_cm": 250,
            "distance_300_handling": "take the first 250 cm from each drone's formal movement start using mission-pad trajectory",
            "include_waiting_energy": False,
        },
        "metadata_precedence": [
            "actual run-level drone/all-drone CSV fields",
            "top-level directory name",
            "experiment_registry.json",
            "needs_confirmation",
        ],
        "anomaly_policy": {
            "trajectory_jump": "retain for now and report",
            "minor_issue": "retain for now and report",
            "deletion_policy": "do not physically delete during initial cleaning; assign status and exclusion reason",
        },
    }
    (ADMIN / "cleaning_rules.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "source_file_count": len(source_rows),
        "copy_file_count_excluding_cleaning_admin": len(copy_rows),
        "reconciliation_status_counts": status_counts,
        "source_total_bytes": sum(int(row["size_bytes"]) for row in source_rows),
        "copy_total_bytes_excluding_cleaning_admin": sum(
            int(row["size_bytes"]) for row in copy_rows
        ),
    }
    (ADMIN / "inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
