from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db_copy_for_cleaning"
ADMIN = DB / "_cleaning_admin"

STANDARD_RUN_RE = re.compile(r"_new_\d+$", re.IGNORECASE)
FORMAL_FOLDER_RE = re.compile(
    r"^(front|vee|diamond|echalon|column)_(50|75)_(head|side|tail)_lv([12])_new_\d+$",
    re.IGNORECASE,
)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_directory(name: str) -> tuple[str, str]:
    lower = name.lower()
    if name == "baselines":
        return "supporting_baseline", "Battery/drone baseline data; not a swarm run."
    if lower.endswith("_summary"):
        return "derived_summary", "Derived summary retained but not treated as a raw run."
    if "pre" in lower or "perpare" in lower:
        return "excluded_preparation", "Name contains pre/prepare/perpare."
    if "no_wind" in lower and STANDARD_RUN_RE.search(name):
        return "retained_no_wind", "Retained but outside the current analysis scope."
    if STANDARD_RUN_RE.search(name):
        return "formal_candidate", "Name satisfies the confirmed formal candidate rule."
    return "excluded_legacy_or_out_of_scope", "Does not satisfy the formal candidate rule."


def first_csv_row(path: Path) -> dict[str, str] | None:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return next(csv.DictReader(handle), None)
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def all_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def unique_nonempty(rows: list[dict[str, str]], field: str) -> set[str]:
    return {str(row.get(field, "")).strip() for row in rows if str(row.get(field, "")).strip()}


def one_value(rows: list[dict[str, str]], field: str) -> str:
    values = sorted(unique_nonempty(rows, field))
    return values[0] if len(values) == 1 else "|".join(values)


def normalize_wind(value: str) -> str:
    lower = value.strip().lower().replace("_", " ")
    if lower.startswith("head"):
        return "head"
    if lower.startswith("side"):
        return "side"
    if lower.startswith("tail"):
        return "tail"
    if "no wind" in lower:
        return "no_wind"
    return lower


def normalize_level(value: str) -> str:
    match = re.search(r"([12])", value)
    return match.group(1) if match else value.strip().lower().replace(" ", "_")


def as_int_string(value: str) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def add_issue(
    issues: list[dict[str, str]],
    level: str,
    directory: str,
    run_id: str,
    severity: str,
    code: str,
    evidence: str,
    action: str,
) -> None:
    issues.append(
        {
            "record_level": level,
            "experiment_directory": directory,
            "run_id": run_id,
            "severity": severity,
            "issue_code": code,
            "evidence": evidence,
            "proposed_action": action,
        }
    )


def main() -> None:
    ADMIN.mkdir(parents=True, exist_ok=True)
    registry_path = DB / "experiment_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_entries = {
        str(item.get("experiment_id", "")): item
        for item in registry.get("experiments", [])
        if str(item.get("experiment_id", ""))
    }
    manual_exclusions_path = ADMIN / "manual_exclusions.csv"
    manual_exclusions: dict[tuple[str, str], dict[str, str]] = {}
    if manual_exclusions_path.exists():
        for row in all_csv_rows(manual_exclusions_path):
            key = (
                str(row.get("experiment_directory", "")).strip(),
                str(row.get("run_id", "")).strip(),
            )
            manual_exclusions[key] = row

    directories = sorted(
        path for path in DB.iterdir() if path.is_dir() and path.name != "_cleaning_admin"
    )
    directory_rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    run_rows: list[dict[str, Any]] = []
    candidate_directory_names: set[str] = set()
    runs_by_directory: dict[str, int] = {}

    for directory in directories:
        classification, reason = classify_directory(directory.name)
        if classification in {"formal_candidate", "retained_no_wind"}:
            candidate_directory_names.add(directory.name)
        file_count = sum(1 for path in directory.rglob("*") if path.is_file())
        subdirectory_count = sum(1 for path in directory.rglob("*") if path.is_dir())
        directory_rows.append(
            {
                "directory_name": directory.name,
                "classification": classification,
                "classification_reason": reason,
                "file_count": file_count,
                "subdirectory_count": subdirectory_count,
                "registry_present": directory.name in registry_entries,
            }
        )

    directory_index = {row["directory_name"]: row for row in directory_rows}

    battery_bindings: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    ip_bindings: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for directory_name in sorted(candidate_directory_names):
        directory = DB / directory_name
        classification = str(directory_index[directory_name]["classification"])
        battery_by_run: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        coordination_by_run: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

        for path in sorted(directory.glob("drones/*/*_battery.csv")):
            for row in all_csv_rows(path):
                run_id = str(row.get("run_id", "")).strip()
                if run_id:
                    row["_source_file"] = path.relative_to(DB).as_posix()
                    battery_by_run[run_id].append(row)

        for path in sorted(directory.glob("drones/*/*_coordination.csv")):
            row = first_csv_row(path)
            if row:
                run_id = str(row.get("run_id", "")).strip()
                if run_id:
                    row["_source_file"] = path.relative_to(DB).as_posix()
                    coordination_by_run[run_id].append(row)

        run_ids = sorted(set(battery_by_run) | set(coordination_by_run))
        runs_by_directory[directory_name] = len(run_ids)
        if not run_ids:
            directory_index[directory_name]["run_count"] = 0
            directory_index[directory_name]["directory_quality_status"] = "incomplete"
            add_issue(
                issues,
                "directory",
                directory_name,
                "",
                "high",
                "NO_NONEMPTY_RUN_ROWS",
                "No non-empty per-drone battery or coordination rows were found.",
                "Exclude from formal analysis unless a recoverable source is identified.",
            )
            continue

        if len(run_ids) > 1:
            add_issue(
                issues,
                "directory",
                directory_name,
                "",
                "low",
                "MULTIPLE_RUN_IDS_IN_DIRECTORY",
                f"Found {len(run_ids)} run IDs: {', '.join(run_ids)}.",
                "Treat each run ID as an independent candidate run.",
            )

        directory_index[directory_name]["run_count"] = len(run_ids)
        directory_index[directory_name]["directory_quality_status"] = "indexed"

        for run_id in run_ids:
            battery_rows = battery_by_run.get(run_id, [])
            coordination_rows = coordination_by_run.get(run_id, [])
            battery_drones = unique_nonempty(battery_rows, "drone_name")
            coordination_drones = unique_nonempty(coordination_rows, "drone_name")
            coordination_has_battery_fields = (
                len(coordination_drones) == 5
                and all(
                    str(row.get("battery", "")).strip()
                    and str(row.get("battery_id", "")).strip()
                    for row in coordination_rows
                )
            )
            battery_summary_recoverable = (
                len(battery_drones) == 0 and coordination_has_battery_fields
            )

            metadata_rows = battery_rows + coordination_rows
            fields_to_check = [
                "experiment_id",
                "formation",
                "wind_direction",
                "wind_speed",
                "inter_drone_distance_cm",
                "node_forward_distance_cm",
            ]
            consistency = {
                field: len(unique_nonempty(metadata_rows, field)) <= 1
                for field in fields_to_check
            }
            metadata_consistent = all(consistency.values())

            experiment_id = one_value(metadata_rows, "experiment_id")
            raw_formation = one_value(metadata_rows, "formation").lower()
            raw_wind_value = one_value(metadata_rows, "wind_direction")
            raw_wind = normalize_wind(raw_wind_value) if raw_wind_value else ""
            raw_level_value = one_value(metadata_rows, "wind_speed")
            raw_level = normalize_level(raw_level_value) if raw_level_value else ""
            raw_spacing_value = one_value(metadata_rows, "inter_drone_distance_cm")
            raw_spacing = as_int_string(raw_spacing_value) if raw_spacing_value else ""
            distance_value = one_value(metadata_rows, "node_forward_distance_cm")
            distance = as_int_string(distance_value) if distance_value else ""

            folder_match = FORMAL_FOLDER_RE.match(directory_name)
            folder_parseable = folder_match is not None
            folder_metadata_matches = True
            folder_expected = ""
            expected: dict[str, str] = {}
            if classification == "formal_candidate":
                if folder_match:
                    expected = {
                        "formation": folder_match.group(1).lower(),
                        "spacing": folder_match.group(2),
                        "wind": folder_match.group(3).lower(),
                        "level": folder_match.group(4),
                    }
                    csv_values = {
                        "formation": raw_formation,
                        "spacing": raw_spacing,
                        "wind": raw_wind,
                        "level": raw_level,
                    }
                    # A blank CSV field is filled from the next source in the
                    # confirmed precedence chain; it is not a conflict.
                    folder_metadata_matches = all(
                        not csv_values[field] or csv_values[field] == expected[field]
                        for field in expected
                    )
                    folder_expected = json.dumps(expected, sort_keys=True)
                else:
                    folder_metadata_matches = False

            registry_entry = registry_entries.get(directory_name)
            registry_present = registry_entry is not None
            manual_exclusion = manual_exclusions.get((directory_name, run_id)) or manual_exclusions.get((directory_name, ""))
            registry_outlier = bool(registry_entry and registry_entry.get("is_outlier") is True)
            marked_outlier = registry_outlier or manual_exclusion is not None
            if registry_outlier and manual_exclusion is not None:
                outlier_marker_source = "registry_is_outlier_and_user_confirmed"
            elif registry_outlier:
                outlier_marker_source = "registry_is_outlier"
            elif manual_exclusion is not None:
                outlier_marker_source = str(manual_exclusion.get("exclusion_source", "manual_exclusion"))
            else:
                outlier_marker_source = ""
            outlier_note = ""
            if registry_entry is not None:
                outlier_note = str(registry_entry.get("outlier_note", "") or "")
            if manual_exclusion is not None:
                outlier_note = str(manual_exclusion.get("exclusion_reason", "") or outlier_note)
            registry_actual = {
                "formation": "",
                "spacing": "",
                "wind": "",
                "level": "",
            }
            if registry_entry is not None:
                registry_actual = {
                    "formation": str(registry_entry.get("formation", "")).lower(),
                    "spacing": as_int_string(str(registry_entry.get("inter_drone_distance_cm", ""))),
                    "wind": normalize_wind(str(registry_entry.get("wind_direction", ""))),
                    "level": normalize_level(str(registry_entry.get("wind_speed", ""))),
                }

            general_spacing_match = re.search(r"_(50|75)_", directory_name)
            folder_spacing = general_spacing_match.group(1) if general_spacing_match else ""
            folder_formation = directory_name.split("_", 1)[0].lower()
            formation = raw_formation or expected.get("formation", "") or folder_formation or registry_actual["formation"]
            spacing = raw_spacing or expected.get("spacing", "") or folder_spacing or registry_actual["spacing"]
            wind = raw_wind or expected.get("wind", "") or registry_actual["wind"]
            level = raw_level or expected.get("level", "") or registry_actual["level"]
            metadata_sources = {
                "formation": "csv" if raw_formation else ("folder" if expected.get("formation") or folder_formation else "registry"),
                "spacing": "csv" if raw_spacing else ("folder" if expected.get("spacing") or folder_spacing else "registry"),
                "wind": "csv" if raw_wind else ("folder" if expected.get("wind") else "registry"),
                "level": "csv" if raw_level else ("folder" if expected.get("level") else "registry"),
                "distance": "csv" if distance else "missing",
            }

            registry_metadata_matches = ""
            if registry_entry is not None and metadata_rows:
                csv_actual = {
                    "formation": formation,
                    "spacing": spacing,
                    "wind": wind,
                    "level": level,
                }
                registry_metadata_matches = registry_actual == csv_actual

            all_coordination = list(directory.glob(f"*_{run_id}_all_coordination.csv"))
            all_timeseries = list(directory.glob(f"*_{run_id}_all_battery_timeseries.csv"))

            issue_codes: list[str] = []
            core_complete = True
            if len(battery_drones) != 5:
                if battery_summary_recoverable:
                    issue_codes.append("BATTERY_SUMMARY_MISSING_RECOVERABLE")
                    add_issue(
                        issues,
                        "run",
                        directory_name,
                        run_id,
                        "medium",
                        "BATTERY_SUMMARY_MISSING_RECOVERABLE",
                        "Per-drone battery summaries are empty, but all five non-empty coordination logs contain battery and battery_id fields.",
                        "Attempt run reconstruction from coordination logs, then require the same trajectory-completion checks as other runs.",
                    )
                else:
                    core_complete = False
                    issue_codes.append("BATTERY_DRONE_COUNT_NOT_5")
                    add_issue(
                        issues,
                        "run",
                        directory_name,
                        run_id,
                        "high",
                        "BATTERY_DRONE_COUNT_NOT_5",
                        f"Found {len(battery_drones)} unique drones in battery summaries.",
                        "Exclude unless all five drone records can be recovered.",
                    )
            if len(coordination_drones) != 5:
                core_complete = False
                issue_codes.append("COORDINATION_DRONE_COUNT_NOT_5")
                add_issue(
                    issues,
                    "run",
                    directory_name,
                    run_id,
                    "high",
                    "COORDINATION_DRONE_COUNT_NOT_5",
                    f"Found {len(coordination_drones)} non-empty per-drone coordination logs.",
                    "Exclude unless all five trajectory logs can be recovered.",
                )
            if battery_drones and battery_drones != coordination_drones:
                core_complete = False
                issue_codes.append("DRONE_SET_MISMATCH")
                add_issue(
                    issues,
                    "run",
                    directory_name,
                    run_id,
                    "high",
                    "DRONE_SET_MISMATCH",
                    f"Battery drones={sorted(battery_drones)}; coordination drones={sorted(coordination_drones)}.",
                    "Resolve the mismatched files before trajectory extraction.",
                )
            if not metadata_consistent:
                core_complete = False
                issue_codes.append("INCONSISTENT_DRONE_METADATA")
                bad_fields = [field for field, ok in consistency.items() if not ok]
                add_issue(
                    issues,
                    "run",
                    directory_name,
                    run_id,
                    "high",
                    "INCONSISTENT_DRONE_METADATA",
                    f"Fields disagree across drone rows: {', '.join(bad_fields)}.",
                    "Use CSV evidence to resolve each field before analysis.",
                )
            if distance not in {"250", "300"}:
                core_complete = False
                issue_codes.append("UNEXPECTED_COMMANDED_DISTANCE")
                add_issue(
                    issues,
                    "run",
                    directory_name,
                    run_id,
                    "high",
                    "UNEXPECTED_COMMANDED_DISTANCE",
                    f"node_forward_distance_cm={distance!r}.",
                    "Do not analyse until the target distance is confirmed.",
                )
            if classification == "formal_candidate" and not folder_metadata_matches:
                issue_codes.append("FOLDER_CSV_METADATA_MISMATCH")
                add_issue(
                    issues,
                    "run",
                    directory_name,
                    run_id,
                    "medium",
                    "FOLDER_CSV_METADATA_MISMATCH",
                    f"Folder parseable={folder_parseable}; expected={folder_expected or 'unparseable'}; CSV={{'formation': '{formation}', 'spacing': '{spacing}', 'wind': '{wind}', 'level': '{level}'}}.",
                    "Use CSV metadata and request confirmation before final inclusion.",
                )
            if not registry_present:
                issue_codes.append("REGISTRY_ENTRY_MISSING")
                add_issue(
                    issues,
                    "run",
                    directory_name,
                    run_id,
                    "low",
                    "REGISTRY_ENTRY_MISSING",
                    "No exact experiment_id entry exists in experiment_registry.json.",
                    "Retain because run-level CSV has higher precedence; document the mismatch.",
                )
            elif registry_metadata_matches is False:
                issue_codes.append("REGISTRY_CSV_METADATA_MISMATCH")
                add_issue(
                    issues,
                    "run",
                    directory_name,
                    run_id,
                    "low",
                    "REGISTRY_CSV_METADATA_MISMATCH",
                    "Registry condition/configuration metadata differs from run-level CSV.",
                    "Use run-level CSV metadata and retain the registry discrepancy for audit.",
                )
            if not all_coordination:
                issue_codes.append("ALL_COORDINATION_FILE_MISSING")
                add_issue(
                    issues,
                    "run",
                    directory_name,
                    run_id,
                    "low",
                    "ALL_COORDINATION_FILE_MISSING",
                    "Combined all-drone coordination file is absent.",
                    "Use the five per-drone coordination logs if complete.",
                )
            if not all_timeseries:
                issue_codes.append("ALL_BATTERY_TIMESERIES_MISSING")
                add_issue(
                    issues,
                    "run",
                    directory_name,
                    run_id,
                    "low",
                    "ALL_BATTERY_TIMESERIES_MISSING",
                    "Combined battery timeseries file is absent.",
                    "Use battery values in the per-drone coordination logs if needed.",
                )
            if marked_outlier:
                issue_codes.append("MARKED_OUTLIER")
                add_issue(
                    issues,
                    "run",
                    directory_name,
                    run_id,
                    "medium",
                    "MARKED_OUTLIER",
                    f"Outlier marker source={outlier_marker_source}; note={outlier_note or 'none'}.",
                    "Retain raw files but exclude the run from trajectory extraction and formal analysis.",
                )

            if marked_outlier:
                scope_status = (
                    "no_wind_not_analysed"
                    if classification == "retained_no_wind"
                    else "formal_analysis_candidate"
                )
                overall_status = "excluded_marked_outlier"
            elif classification == "retained_no_wind":
                scope_status = "no_wind_not_analysed"
                if not core_complete:
                    overall_status = "incomplete_no_wind"
                elif battery_summary_recoverable:
                    overall_status = "retained_no_wind_recoverable"
                else:
                    overall_status = "retained_no_wind"
            else:
                scope_status = "formal_analysis_candidate"
                if not core_complete:
                    overall_status = "incomplete"
                elif not folder_metadata_matches:
                    overall_status = "needs_confirmation"
                elif battery_summary_recoverable:
                    overall_status = "recoverable_pending_trajectory"
                else:
                    overall_status = "candidate_pending_trajectory"

            drone_battery_pairs = []
            binding_source_rows = battery_rows if battery_rows else coordination_rows
            for row in binding_source_rows:
                drone = str(row.get("drone_name", "")).strip()
                battery = str(row.get("battery_id", "")).strip()
                ip = str(row.get("drone_ip", "")).strip()
                if drone and battery:
                    battery_bindings[(classification, drone)][battery] += 1
                    drone_battery_pairs.append(f"{drone}:{battery}")
                if drone and ip:
                    ip_bindings[drone][ip] += 1

            run_rows.append(
                {
                    "experiment_directory": directory_name,
                    "run_id": run_id,
                    "scope_status": scope_status,
                    "overall_status": overall_status,
                    "formation": formation,
                    "wind_direction": wind,
                    "wind_level": level,
                    "inter_drone_spacing_cm": spacing,
                    "commanded_distance_cm": distance,
                    "battery_summary_row_count": len(battery_rows),
                    "battery_unique_drone_count": len(battery_drones),
                    "coordination_nonempty_drone_count": len(coordination_drones),
                    "all_coordination_present": bool(all_coordination),
                    "all_battery_timeseries_present": bool(all_timeseries),
                    "battery_summary_recoverable_from_coordination": battery_summary_recoverable,
                    "metadata_consistent_across_drones": metadata_consistent,
                    "folder_metadata_matches_csv": folder_metadata_matches,
                    "registry_present": registry_present,
                    "registry_metadata_matches_csv": registry_metadata_matches,
                    "marked_outlier": marked_outlier,
                    "outlier_marker_source": outlier_marker_source,
                    "outlier_note": outlier_note,
                    "metadata_sources": json.dumps(metadata_sources, sort_keys=True),
                    "drone_battery_map": ";".join(sorted(set(drone_battery_pairs))),
                    "issue_codes": ";".join(issue_codes),
                }
            )

    for row in directory_rows:
        row.setdefault("run_count", 0)
        row.setdefault("directory_quality_status", "not_applicable")

    registry_orphans = []
    actual_directory_names = {path.name for path in directories}
    for experiment_id, entry in sorted(registry_entries.items()):
        if STANDARD_RUN_RE.search(experiment_id) and experiment_id not in actual_directory_names:
            registry_orphans.append(
                {
                    "experiment_id": experiment_id,
                    "formation": entry.get("formation", ""),
                    "wind_direction": entry.get("wind_direction", ""),
                    "wind_speed": entry.get("wind_speed", ""),
                    "inter_drone_distance_cm": entry.get("inter_drone_distance_cm", ""),
                    "status": "registry_entry_without_directory",
                }
            )

    binding_rows: list[dict[str, Any]] = []
    for (classification, drone), counts in sorted(battery_bindings.items()):
        binding_rows.append(
            {
                "directory_classification": classification,
                "drone_name": drone,
                "battery_ids": ";".join(sorted(counts)),
                "battery_id_count": len(counts),
                "observations_by_battery": json.dumps(dict(sorted(counts.items())), sort_keys=True),
                "fixed_binding_consistent": len(counts) == 1,
            }
        )
        if classification == "formal_candidate" and len(counts) > 1:
            add_issue(
                issues,
                "global",
                "",
                "",
                "medium",
                "DRONE_HAS_MULTIPLE_BATTERY_IDS",
                f"{drone} has formal-run battery IDs {dict(sorted(counts.items()))}.",
                "Check whether this is a metadata error or a historical battery replacement; retain IDs for battery-specific baseline correction.",
            )

    directory_fields = [
        "directory_name",
        "classification",
        "classification_reason",
        "file_count",
        "subdirectory_count",
        "registry_present",
        "run_count",
        "directory_quality_status",
    ]
    run_fields = [
        "experiment_directory",
        "run_id",
        "scope_status",
        "overall_status",
        "formation",
        "wind_direction",
        "wind_level",
        "inter_drone_spacing_cm",
        "commanded_distance_cm",
        "battery_summary_row_count",
        "battery_unique_drone_count",
        "coordination_nonempty_drone_count",
        "all_coordination_present",
        "all_battery_timeseries_present",
        "battery_summary_recoverable_from_coordination",
        "metadata_consistent_across_drones",
        "folder_metadata_matches_csv",
        "registry_present",
        "registry_metadata_matches_csv",
        "marked_outlier",
        "outlier_marker_source",
        "outlier_note",
        "metadata_sources",
        "drone_battery_map",
        "issue_codes",
    ]
    issue_fields = [
        "record_level",
        "experiment_directory",
        "run_id",
        "severity",
        "issue_code",
        "evidence",
        "proposed_action",
    ]
    write_csv(ADMIN / "directory_inventory.csv", directory_rows, directory_fields)
    write_csv(ADMIN / "run_inventory.csv", run_rows, run_fields)
    write_csv(ADMIN / "issues_to_review.csv", issues, issue_fields)
    write_csv(
        ADMIN / "registry_entries_without_directory.csv",
        registry_orphans,
        [
            "experiment_id",
            "formation",
            "wind_direction",
            "wind_speed",
            "inter_drone_distance_cm",
            "status",
        ],
    )
    write_csv(
        ADMIN / "drone_battery_binding_audit.csv",
        binding_rows,
        [
            "directory_classification",
            "drone_name",
            "battery_ids",
            "battery_id_count",
            "observations_by_battery",
            "fixed_binding_consistent",
        ],
    )

    directory_counts = Counter(str(row["classification"]) for row in directory_rows)
    run_status_counts = Counter(str(row["overall_status"]) for row in run_rows)
    issue_counts = Counter(str(row["issue_code"]) for row in issues)
    formal_runs = [row for row in run_rows if row["scope_status"] == "formal_analysis_candidate"]
    summary = {
        "top_level_directory_count": len(directory_rows),
        "directory_classification_counts": dict(sorted(directory_counts.items())),
        "indexed_run_count": len(run_rows),
        "formal_run_count": len(formal_runs),
        "no_wind_run_count": len(run_rows) - len(formal_runs),
        "run_status_counts": dict(sorted(run_status_counts.items())),
        "issue_code_counts": dict(sorted(issue_counts.items())),
        "registry_entry_without_directory_count": len(registry_orphans),
    }
    (ADMIN / "candidate_index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    high_or_medium = [row for row in issues if row["severity"] in {"high", "medium"}]
    report_lines = [
        "# Run-level candidate inventory",
        "",
        "This step creates sidecar status records only. No experimental CSV was edited or deleted.",
        "",
        f"- Indexed timestamp-level runs: {len(run_rows)}",
        f"- Formal-condition runs: {len(formal_runs)}",
        f"- No-wind runs retained but not analysed: {len(run_rows) - len(formal_runs)}",
        "",
        "## Run status counts",
        "",
    ]
    for status, count in sorted(run_status_counts.items()):
        report_lines.append(f"- `{status}`: {count}")
    report_lines.extend(["", "## High/medium issues", ""])
    if not high_or_medium:
        report_lines.append("- None found at this stage.")
    else:
        for row in high_or_medium:
            target = row["experiment_directory"] or "global"
            if row["run_id"]:
                target += f" / {row['run_id']}"
            report_lines.append(
                f"- **{row['severity'].upper()} — {row['issue_code']}** ({target}): {row['evidence']}"
            )
    report_lines.extend(
        [
            "",
            "## Next gate",
            "",
            "`candidate_pending_trajectory` and `recoverable_pending_trajectory` runs proceed to trajectory-based completion checks. Recoverable runs must first reconstruct battery summaries from the five coordination logs. `needs_confirmation` and `incomplete` remain out of the analytical dataset until reviewed.",
            "",
        ]
    )
    (ADMIN / "candidate_inventory_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
