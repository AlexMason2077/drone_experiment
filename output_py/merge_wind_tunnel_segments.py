#!/usr/bin/env python3
"""Merge split wind-tunnel runs while preserving observed/synthetic provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "database"
TARGET_ID = "wind_tunnel_front_50_head_lv2_002"
SOURCE_IDS = [f"wind_tunnel_front_50_head_lv2_{number:03d}" for number in range(2, 6)]
MERGED_OBSERVED_SOURCE_IDS = [
    "wind_tunnel_front_50_head_lv2_002",
    "wind_tunnel_front_50_head_lv2_003",
    "wind_tunnel_front_50_head_lv2_005",
]
EXCLUDED_INTERRUPTION_SOURCE_IDS = ["wind_tunnel_front_50_head_lv2_004"]
CONTINUATION_SETUP_TRIM_SOURCE_IDS = [
    "wind_tunnel_front_50_head_lv2_003",
    "wind_tunnel_front_50_head_lv2_005",
]
MERGED_RUN_ID = "20260829_211154_merged_002_005"

PROVENANCE_COLUMNS = [
    "record_origin",
    "source_experiment_id",
    "source_run_id",
    "source_timestamp",
    "source_elapsed_time",
    "source_hover_elapsed_time",
    "source_node_elapsed_time",
    "interpolation_method",
    "interpolation_gap_index",
    "interpolation_fraction",
]

IDENTITY_COLUMNS = {
    "formation",
    "wind_direction",
    "wind_speed",
    "inter_drone_distance_cm",
    "soc_mode",
    "target_soc_percent",
    "soc_tolerance_percent",
    "drone_name",
    "drone_ip",
    "battery_id",
    "takeoff_order",
    "drone_role",
    "mission_pad",
    "grid_column",
    "grid_row",
    "target_pad",
    "node_forward_distance_cm",
    "node_speed_cm_s",
    "target_x",
    "target_y",
    "target_z",
}

TIMESERIES_COLUMNS = [
    "run_id",
    "experiment_id",
    "formation",
    "wind_direction",
    "wind_speed",
    "inter_drone_distance_cm",
    "soc_mode",
    "target_soc_percent",
    "soc_tolerance_percent",
    "drone_name",
    "drone_ip",
    "battery_id",
    "takeoff_order",
    "drone_role",
    "mission_pad",
    "target_pad",
    "phase",
    "timestamp",
    "elapsed_time",
    "node_elapsed_time",
    "battery",
    "battery_start",
    "battery_drop_from_start",
] + PROVENANCE_COLUMNS


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def timestamp_text(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def float_text(value: float, digits: int = 3) -> str:
    result = f"{value:.{digits}f}"
    return result.rstrip("0").rstrip(".") if "." in result else result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_source_file(experiment_id: str, suffix: str) -> Path:
    folder = DATA_DIR / experiment_id
    candidates = [
        path
        for path in folder.glob(f"*{suffix}")
        if "_merged_" not in path.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No source {suffix} file for {experiment_id}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_segments() -> tuple[list[str], list[dict]]:
    canonical_fields: list[str] | None = None
    segments = []
    for source_id in SOURCE_IDS:
        coord_path = latest_source_file(source_id, "_all_coordination.csv")
        fields, rows = read_csv(coord_path)
        if canonical_fields is None:
            canonical_fields = fields
        elif fields != canonical_fields:
            raise ValueError(f"Schema mismatch in {coord_path}")
        if not rows:
            raise ValueError(f"Empty source file: {coord_path}")
        for row in rows:
            row["_source_experiment_id"] = source_id
            row["_source_path"] = str(coord_path.relative_to(BASE_DIR))
            row["_parsed_timestamp"] = parse_timestamp(row["timestamp"])
        segments.append(
            {
                "experiment_id": source_id,
                "path": coord_path,
                "rows": rows,
                "start": min(row["_parsed_timestamp"] for row in rows),
                "end": max(row["_parsed_timestamp"] for row in rows),
            }
        )
    return canonical_fields or [], segments


def validate_compatibility(segments: list[dict]) -> dict[str, str]:
    expected_batteries: dict[str, str] | None = None
    expected_condition: tuple[str, ...] | None = None
    condition_fields = (
        "formation",
        "wind_direction",
        "wind_speed",
        "inter_drone_distance_cm",
        "soc_mode",
        "target_soc_percent",
    )
    for segment in segments:
        rows = segment["rows"]
        batteries = {}
        for row in rows:
            batteries.setdefault(row["drone_name"], row["battery_id"])
            if batteries[row["drone_name"]] != row["battery_id"]:
                raise ValueError(f"Battery changed inside {segment['experiment_id']}")
        condition = tuple(rows[0].get(field, "") for field in condition_fields)
        if any(tuple(row.get(field, "") for field in condition_fields) != condition for row in rows):
            raise ValueError(f"Condition changed inside {segment['experiment_id']}")
        if expected_batteries is None:
            expected_batteries = batteries
            expected_condition = condition
        elif batteries != expected_batteries:
            raise ValueError(f"Battery mapping mismatch in {segment['experiment_id']}")
        elif condition != expected_condition:
            raise ValueError(f"Condition mismatch in {segment['experiment_id']}")
    return expected_batteries or {}


def trim_continuation_setup(segment: dict) -> tuple[dict, dict | None]:
    if segment["experiment_id"] not in CONTINUATION_SETUP_TRIM_SOURCE_IDS:
        return segment, None
    hover_rows = [
        row for row in segment["rows"]
        if row.get("phase") in {"wind_tunnel_hover", "wind_tunnel_periodic_centering"}
    ]
    if not hover_rows:
        raise ValueError(f"No hover phase found in {segment['experiment_id']}")
    keep_from = min(row["_parsed_timestamp"] for row in hover_rows)
    rows = [row for row in segment["rows"] if row["_parsed_timestamp"] >= keep_from]
    trimmed = {
        **segment,
        "rows": rows,
        "start": min(row["_parsed_timestamp"] for row in rows),
        "end": max(row["_parsed_timestamp"] for row in rows),
    }
    return trimmed, {
        "experiment_id": segment["experiment_id"],
        "removed_phase_names": ["wind_tunnel_takeoff", "acquire_start_pad", "coordinate_climb"],
        "original_start_timestamp": timestamp_text(segment["start"]),
        "kept_from_timestamp": timestamp_text(keep_from),
        "removed_duration_sec": round((keep_from - segment["start"]).total_seconds(), 3),
        "removed_rows": len(segment["rows"]) - len(rows),
    }


def estimate_cadence(segments: list[dict]) -> float:
    differences = []
    for segment in segments:
        by_drone: dict[str, list[datetime]] = defaultdict(list)
        for row in segment["rows"]:
            by_drone[row["drone_name"]].append(row["_parsed_timestamp"])
        for timestamps in by_drone.values():
            timestamps.sort()
            differences.extend(
                (later - earlier).total_seconds()
                for earlier, later in zip(timestamps, timestamps[1:])
                if later > earlier
            )
    if not differences:
        raise ValueError("Could not estimate sampling cadence")
    return statistics.median(differences)


def endpoints_by_drone(segment: dict, first: bool) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in segment["rows"]:
        grouped[row["drone_name"]].append(row)
    return {
        drone: (min(rows, key=lambda row: row["_parsed_timestamp"]) if first else max(rows, key=lambda row: row["_parsed_timestamp"]))
        for drone, rows in grouped.items()
    }


def observed_row(row: dict, compressed_timestamp: datetime, compressed_elapsed: float) -> dict[str, str]:
    result = {key: value for key, value in row.items() if not key.startswith("_")}
    result.update(
        {
            "run_id": MERGED_RUN_ID,
            "experiment_id": TARGET_ID,
            "timestamp": timestamp_text(compressed_timestamp),
            "elapsed_time": float_text(compressed_elapsed),
            "hover_elapsed_time": float_text(compressed_elapsed),
            "node_elapsed_time": float_text(compressed_elapsed),
            "record_origin": "observed",
            "source_experiment_id": row["_source_experiment_id"],
            "source_run_id": row.get("run_id", ""),
            "source_timestamp": row.get("timestamp", ""),
            "source_elapsed_time": row.get("elapsed_time", ""),
            "source_hover_elapsed_time": row.get("hover_elapsed_time", ""),
            "source_node_elapsed_time": row.get("node_elapsed_time", ""),
            "interpolation_method": "",
            "interpolation_gap_index": "",
            "interpolation_fraction": "",
        }
    )
    return result


def observed_discharge_rates(segments: list[dict]) -> dict[str, float]:
    drops: dict[str, float] = defaultdict(float)
    durations: dict[str, float] = defaultdict(float)
    for segment in segments:
        first_by_drone = endpoints_by_drone(segment, first=True)
        last_by_drone = endpoints_by_drone(segment, first=False)
        duration = (segment["end"] - segment["start"]).total_seconds()
        for drone in first_by_drone:
            drop = max(0.0, float(first_by_drone[drone]["battery"]) - float(last_by_drone[drone]["battery"]))
            drops[drone] += drop
            durations[drone] += duration
    rates = {drone: drops[drone] / durations[drone] for drone in drops if durations[drone] > 0}
    if not rates or sum(rates.values()) <= 0:
        raise ValueError("Could not estimate observed Tello discharge rates")
    return rates


def simulated_tello_row(
    left: dict,
    right: dict,
    compressed_timestamp: datetime,
    compressed_elapsed: float,
    fraction: float,
    gap_index: int,
) -> dict[str, str]:
    result = {field: "" for field in left if not field.startswith("_")}
    for field in IDENTITY_COLUMNS:
        result[field] = left.get(field, "") or right.get(field, "")
    left_battery = int(round(float(left["battery"])))
    right_battery = int(round(float(right["battery"])))
    battery = int(round(left_battery + fraction * (right_battery - left_battery)))
    battery = min(left_battery, max(right_battery, battery))
    result.update(
        {
            "run_id": MERGED_RUN_ID,
            "experiment_id": TARGET_ID,
            "phase": "merge_gap_tello_simulated",
            "timestamp": timestamp_text(compressed_timestamp),
            "elapsed_time": float_text(compressed_elapsed),
            "hover_elapsed_time": float_text(compressed_elapsed),
            "node_elapsed_time": float_text(compressed_elapsed),
            "battery": str(battery),
            "record_origin": "simulated_tello_battery_bridge",
            "source_experiment_id": f"{left['_source_experiment_id']}->{right['_source_experiment_id']}",
            "source_run_id": "",
            "source_timestamp": "",
            "source_elapsed_time": "",
            "source_hover_elapsed_time": "",
            "source_node_elapsed_time": "",
            "interpolation_method": "tello_integer_step_from_observed_swarm_discharge_rate",
            "interpolation_gap_index": str(gap_index),
            "interpolation_fraction": float_text(fraction, 6),
        }
    )
    return result


def merge_rows(segments: list[dict], cadence: float) -> tuple[list[dict[str, str]], list[dict]]:
    compressed_start = segments[0]["start"]
    compressed_offset = 0.0
    discharge_rates = observed_discharge_rates(segments)
    swarm_discharge_rate = sum(discharge_rates.values())
    rows = []
    gaps = []
    for segment_index, segment in enumerate(segments):
        segment_duration = (segment["end"] - segment["start"]).total_seconds()
        for row in segment["rows"]:
            local_elapsed = (row["_parsed_timestamp"] - segment["start"]).total_seconds()
            compressed_elapsed = compressed_offset + local_elapsed
            compressed_timestamp = compressed_start + timedelta(seconds=compressed_elapsed)
            rows.append(observed_row(row, compressed_timestamp, compressed_elapsed))

        if segment_index < len(segments) - 1:
            right_segment = segments[segment_index + 1]
            left_by_drone = endpoints_by_drone(segment, first=False)
            right_by_drone = endpoints_by_drone(right_segment, first=True)
            removed_gap = (right_segment["start"] - segment["end"]).total_seconds()
            battery_drops = {
                drone: max(0.0, float(left_by_drone[drone]["battery"]) - float(right_by_drone[drone]["battery"]))
                for drone in sorted(left_by_drone)
            }
            total_battery_drop = sum(battery_drops.values())
            simulated_duration = total_battery_drop / swarm_discharge_rate if total_battery_drop > 0 else 0.0
            simulated_sample_count = math.floor(simulated_duration / cadence)
            for step in range(1, simulated_sample_count + 1):
                bridge_elapsed = min(cadence * step, simulated_duration)
                fraction = bridge_elapsed / simulated_duration if simulated_duration > 0 else 1.0
                compressed_elapsed = compressed_offset + segment_duration + bridge_elapsed
                compressed_timestamp = compressed_start + timedelta(seconds=compressed_elapsed)
                for drone in sorted(left_by_drone):
                    rows.append(
                        simulated_tello_row(
                            left_by_drone[drone],
                            right_by_drone[drone],
                            compressed_timestamp,
                            compressed_elapsed,
                            fraction,
                            segment_index + 1,
                        )
                    )
            gaps.append(
                {
                    "gap_index": segment_index + 1,
                    "from_experiment_id": segment["experiment_id"],
                    "to_experiment_id": right_segment["experiment_id"],
                    "original_start_timestamp": timestamp_text(segment["end"]),
                    "original_end_timestamp": timestamp_text(right_segment["start"]),
                    "removed_wall_clock_duration_sec": round(removed_gap, 3),
                    "simulated_tello_bridge_duration_sec": round(simulated_duration, 3),
                    "compressed_boundary_interval_sec": round(cadence, 6),
                    "simulated_rows": simulated_sample_count * len(left_by_drone),
                    "observed_discharge_rate_percent_per_sec": {
                        drone: round(discharge_rates[drone], 6) for drone in sorted(discharge_rates)
                    },
                    "endpoint_batteries": {
                        drone: {
                            "left": float(left_by_drone[drone]["battery"]),
                            "right": float(right_by_drone[drone]["battery"]),
                        }
                        for drone in sorted(left_by_drone)
                    },
                }
            )
            compressed_offset += segment_duration + simulated_duration + cadence
    rows.sort(key=lambda row: (parse_timestamp(row["timestamp"]), int(row.get("takeoff_order") or 999)))
    first_battery = {}
    for row in rows:
        first_battery.setdefault(row["drone_name"], float(row["battery"]))
        start_battery = first_battery[row["drone_name"]]
        row["battery_start"] = float_text(start_battery)
        row["battery_drop_from_start"] = float_text(start_battery - float(row["battery"]))
    return rows, gaps


def summary_rows(merged: list[dict[str, str]], gaps: list[dict]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in merged:
        if row["record_origin"] == "observed":
            grouped[row["drone_name"]].append(row)
    removed_gap = sum(gap["removed_wall_clock_duration_sec"] for gap in gaps)
    simulated_bridge = sum(gap["simulated_tello_bridge_duration_sec"] for gap in gaps)
    output = []
    for drone_name, rows in sorted(grouped.items(), key=lambda item: int(item[1][0]["takeoff_order"])):
        rows.sort(key=lambda row: parse_timestamp(row["timestamp"]))
        first, last = rows[0], rows[-1]
        duration = (parse_timestamp(last["timestamp"]) - parse_timestamp(first["timestamp"])).total_seconds()
        result = {
            key: first.get(key, "")
            for key in (
                "formation",
                "wind_direction",
                "wind_speed",
                "inter_drone_distance_cm",
                "soc_mode",
                "target_soc_percent",
                "soc_tolerance_percent",
                "drone_name",
                "drone_ip",
                "battery_id",
                "takeoff_order",
                "drone_role",
                "mission_pad",
                "grid_column",
                "grid_row",
                "target_pad",
                "node_forward_distance_cm",
                "node_speed_cm_s",
            )
        }
        result.update(
            {
                "run_id": MERGED_RUN_ID,
                "experiment_id": TARGET_ID,
                "hover_start_timestamp": first["timestamp"],
                "hover_end_timestamp": last["timestamp"],
                "hover_duration_sec": float_text(duration),
                "node_start_timestamp": first["timestamp"],
                "node_end_timestamp": last["timestamp"],
                "node_duration_sec": float_text(duration),
                "battery_hover_start": first["battery"],
                "battery_hover_end": last["battery"],
                "battery_drop": float_text(float(first["battery"]) - float(last["battery"])),
                "observed_active_span_sec": float_text(duration - simulated_bridge),
                "interpolated_gap_duration_sec": "0",
                "simulated_tello_bridge_duration_sec": float_text(simulated_bridge),
                "removed_interruption_duration_sec": float_text(removed_gap),
                "source_experiment_ids": ";".join(MERGED_OBSERVED_SOURCE_IDS),
                "excluded_interruption_source_ids": ";".join(EXCLUDED_INTERRUPTION_SOURCE_IDS),
                "contains_interpolated_data": "false",
                "contains_simulated_tello_battery_data": "true",
            }
        )
        output.append(result)
    return output


def update_registry(manifest_relpath: str) -> str:
    registry_path = DATA_DIR / "experiment_registry.json"
    backup_path = DATA_DIR / f"experiment_registry.before_{MERGED_RUN_ID}.json"
    if not backup_path.exists():
        shutil.copy2(registry_path, backup_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    now = datetime.now().isoformat(timespec="seconds")
    found = set()
    for experiment in registry.get("experiments", []):
        experiment_id = experiment.get("experiment_id")
        if experiment_id not in SOURCE_IDS:
            continue
        found.add(experiment_id)
        if experiment_id == TARGET_ID:
            experiment["is_outlier"] = False
            experiment["merge_metadata"] = {
                "status": "canonical_merged_run",
                "source_experiment_ids": SOURCE_IDS,
                "contains_interpolated_battery_gap_rows": False,
                "contains_simulated_tello_battery_bridge_rows": True,
                "manifest": manifest_relpath,
            }
            note = "Canonical merged run from _002, _003, _004 and _005; synthetic gap rows are provenance-labelled."
        else:
            experiment["is_outlier"] = True
            experiment["outlier_note"] = f"Continuation segment merged into {TARGET_ID}; not an independent trial."
            experiment["merge_metadata"] = {
                "status": "continuation_source",
                "canonical_experiment_id": TARGET_ID,
                "manifest": manifest_relpath,
            }
            note = f"Continuation source preserved; canonical merged data is stored under {TARGET_ID}."
        existing = str(experiment.get("notes") or "").strip()
        if note not in existing:
            experiment["notes"] = f"{existing}\n{note}".strip()
        experiment["updated_at"] = now
    missing = set(SOURCE_IDS) - found
    if missing:
        raise ValueError(f"Missing registry entries: {sorted(missing)}")
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(backup_path.relative_to(BASE_DIR))


def make_plot(merged: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_drone: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in merged:
        by_drone[row["drone_name"]].append(row)
    fig, ax = plt.subplots(figsize=(13, 7))
    for drone, rows in sorted(by_drone.items()):
        rows.sort(key=lambda row: float(row["elapsed_time"]))
        ax.step(
            [float(row["elapsed_time"]) for row in rows],
            [float(row["battery"]) for row in rows],
            where="post",
            linewidth=1.7,
            label=f"{drone} / {rows[0]['battery_id']}",
        )
    ax.set_title("Merged Tello battery curve: interruption replaced by integer-step discharge simulation")
    ax.set_xlabel("Compressed recorded-flight time (s)")
    ax.set_ylabel("Battery (%)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.text(
        0.01,
        0.01,
        "Observed records are joined by Tello-style integer battery steps; idle/restart wall-clock time is excluded.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def validate_output(merged: list[dict[str, str]], battery_mapping: dict[str, str]) -> dict:
    by_drone: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in merged:
        by_drone[row["drone_name"]].append(row)
    checks = {
        "expected_drone_count": len(by_drone) == 5,
        "battery_mapping_matches_all_sources": all(
            {row["battery_id"] for row in rows} == {battery_mapping[drone]}
            for drone, rows in by_drone.items()
        ),
        "timestamps_strictly_increasing_per_drone": True,
        "no_duplicate_drone_timestamp": True,
        "battery_range_valid": True,
        "observed_rows_have_source": True,
        "simulated_rows_are_labelled": True,
        "simulated_battery_values_are_integers": True,
    }
    for rows in by_drone.values():
        rows.sort(key=lambda row: parse_timestamp(row["timestamp"]))
        timestamps = [parse_timestamp(row["timestamp"]) for row in rows]
        checks["timestamps_strictly_increasing_per_drone"] &= all(
            later > earlier for earlier, later in zip(timestamps, timestamps[1:])
        )
        checks["no_duplicate_drone_timestamp"] &= len(timestamps) == len(set(timestamps))
        checks["battery_range_valid"] &= all(0 <= float(row["battery"]) <= 100 for row in rows)
        checks["observed_rows_have_source"] &= all(
            row["source_experiment_id"] in SOURCE_IDS
            for row in rows
            if row["record_origin"] == "observed"
        )
        checks["simulated_rows_are_labelled"] &= all(
            row["interpolation_method"] == "tello_integer_step_from_observed_swarm_discharge_rate"
            for row in rows
            if row["record_origin"] == "simulated_tello_battery_bridge"
        )
        checks["simulated_battery_values_are_integers"] &= all(
            float(row["battery"]).is_integer()
            for row in rows
            if row["record_origin"] == "simulated_tello_battery_bridge"
        )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "rows_total": len(merged),
        "rows_observed": sum(row["record_origin"] == "observed" for row in merged),
        "rows_interpolated": 0,
        "rows_simulated_tello_battery_bridge": sum(
            row["record_origin"] == "simulated_tello_battery_bridge" for row in merged
        ),
        "rows_per_drone": {drone: len(rows) for drone, rows in sorted(by_drone.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-registry",
        action="store_true",
        help="Rewrite registry metadata. Disabled by default because the registry may contain user-managed formatting.",
    )
    args = parser.parse_args()

    fields, all_segments = load_segments()
    battery_mapping = validate_compatibility(all_segments)
    selected_segments = [
        segment for segment in all_segments
        if segment["experiment_id"] in MERGED_OBSERVED_SOURCE_IDS
    ]
    segments = []
    setup_trims = []
    for segment in selected_segments:
        trimmed, trim_info = trim_continuation_setup(segment)
        segments.append(trimmed)
        if trim_info:
            setup_trims.append(trim_info)
    cadence = estimate_cadence(segments)
    merged, gaps = merge_rows(segments, cadence)
    validation = validate_output(merged, battery_mapping)
    if not validation["passed"]:
        raise ValueError(f"Merged output validation failed: {validation['checks']}")

    target_dir = DATA_DIR / TARGET_ID
    prefix = target_dir / f"{TARGET_ID}_{MERGED_RUN_ID}"
    coordination_path = Path(str(prefix) + "_all_coordination.csv")
    timeseries_path = Path(str(prefix) + "_all_battery_timeseries.csv")
    summary_path = Path(str(prefix) + "_all_battery.csv")
    manifest_path = target_dir / f"{TARGET_ID}_{MERGED_RUN_ID}_merge_manifest.json"
    validation_path = target_dir / f"{TARGET_ID}_{MERGED_RUN_ID}_merge_validation.json"
    plot_path = target_dir / "plots" / "all_battery_merged_provenance.png"

    merged_fields = fields + [field for field in PROVENANCE_COLUMNS if field not in fields]
    write_csv(coordination_path, merged_fields, merged)
    timeseries_rows = [{field: row.get(field, "") for field in TIMESERIES_COLUMNS} for row in merged]
    write_csv(timeseries_path, TIMESERIES_COLUMNS, timeseries_rows)
    summaries = summary_rows(merged, gaps)
    summary_fields = list(summaries[0]) if summaries else []
    write_csv(summary_path, summary_fields, summaries)
    make_plot(merged, plot_path)

    manifest = {
        "target_experiment_id": TARGET_ID,
        "merged_run_id": MERGED_RUN_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "Concatenate observed segments _002, _003 and _005 on a compressed experiment-time axis. Exclude the short abnormal _004 restart attempt. For continuation segments _003 and _005, remove their repeated takeoff, Mission Pad acquisition and coordinated-climb setup rows, keeping data from the first wind_tunnel_hover sample onward. For boundaries with a battery-level difference, simulate Tello-style integer 1% steps using the aggregate discharge rate measured across the included observed segments. Exclude idle/restart wall-clock time and retain original timestamps in source_timestamp.",
        "source_experiment_ids": SOURCE_IDS,
        "observed_source_experiment_ids": MERGED_OBSERVED_SOURCE_IDS,
        "excluded_interruption_source_experiment_ids": EXCLUDED_INTERRUPTION_SOURCE_IDS,
        "continuation_setup_trims": setup_trims,
        "source_files": [
            {
                "experiment_id": segment["experiment_id"],
                "path": str(segment["path"].relative_to(BASE_DIR)),
                "sha256": sha256(segment["path"]),
                "rows": len(segment["rows"]),
                "start": timestamp_text(segment["start"]),
                "end": timestamp_text(segment["end"]),
            }
            for segment in all_segments
        ],
        "battery_mapping": battery_mapping,
        "estimated_sample_cadence_sec": round(cadence, 6),
        "gaps": gaps,
        "outputs": {
            "coordination": str(coordination_path.relative_to(BASE_DIR)),
            "battery_timeseries": str(timeseries_path.relative_to(BASE_DIR)),
            "battery_summary": str(summary_path.relative_to(BASE_DIR)),
            "provenance_plot": str(plot_path.relative_to(BASE_DIR)),
        },
        "scientific_caveat": "Rows labelled simulated_tello_battery_bridge are modeled, not measured. Their battery values follow Tello's integer reporting pattern and observed swarm discharge rate; position, attitude, velocity, acceleration and temperature remain blank. The compressed axis is experiment time, not original wall-clock time.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry_backup_path = DATA_DIR / f"experiment_registry.before_{MERGED_RUN_ID}.json"
    registry_backup = str(registry_backup_path.relative_to(BASE_DIR)) if registry_backup_path.exists() else ""
    if args.update_registry:
        registry_backup = update_registry(str(manifest_path.relative_to(BASE_DIR)))
    manifest["registry_backup"] = registry_backup
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"manifest": str(manifest_path), "validation": validation, "gaps": gaps}, indent=2))


if __name__ == "__main__":
    main()
