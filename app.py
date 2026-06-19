import csv
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file, url_for


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))
DATA_DIR = BASE_DIR / "database"
DATA_DIR.mkdir(exist_ok=True)
BASELINE_DIR = DATA_DIR / "baselines"
REGISTRY_FILE = DATA_DIR / "experiment_registry.json"
ALLOWED_PREVIEW_EXTENSIONS = {".csv", ".png", ".jpg", ".jpeg", ".webp"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CSV_EXTENSIONS = {".csv"}
IP_PREFIX = "192.168.0."
DRONE_NUMBER_TO_IP_SUFFIX = {
    "1": "102",
    "2": "109",
    "3": "103",
    "4": "106",
    "5": "107",
}
IP_SUFFIX_TO_DRONE_NUMBER = {
    suffix: number for number, suffix in DRONE_NUMBER_TO_IP_SUFFIX.items()
}
BATTERY_OPTIONS = [f"B{i:02d}" for i in range(1, 16)]
EXPERIMENT_BATTERY_WINDOW = {"low": 40, "high": 75}
RECOMMENDED_EXPERIMENT_BATTERIES = {"B02", "B04", "B06", "B07", "B10"}
FORMATION_OPTIONS = ["front", "column", "vee", "echalon", "diamond"]
INTER_DRONE_DISTANCE_OPTIONS_CM = [50, 75]
BASELINE_MODES = [
    ("hover", "hover baseline"),
    ("head_forward_250", "head wind forward 250cm"),
    ("tail_forward_250", "tail wind forward 250cm"),
    ("side_forward_250", "side wind lateral 250cm"),
]
BASELINE_DIRECTIONS = [
    ("up", "↑"),
    ("down", "↓"),
]
BASELINE_WIND_LEVELS = [
    ("lv1", "lv1"),
    ("lv2", "lv2"),
]
MISSION_PAD_COLUMNS = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8, 1],
    [5, 6, 7, 8, 1, 2],
]
COLUMN_MISSION_PAD_COLUMNS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 3, 4],
    [],
    [],
    [],
    [],
]
COLUMN_75_MISSION_PAD_COLUMNS = [
    [1, 2, 3, 4, 5, 6, 7, 8, 1],
    [],
    [],
    [],
    [],
]
VEE_MISSION_PAD_COLUMNS = [
    [1, 2, 3, 4, 5, 6],
    [5, 6, 7, 8, 1, 2],
    [3, 4, 5, 6, 7, 8],
    [5, 6, 7, 8, 1, 2],
    [1, 2, 3, 4, 7, 8],
]
DIAMOND_MISSION_PAD_COLUMNS = [
    [],
    [1, 2, 3, 4, 5, 6, 7, 8],
    [2, 3, 4, 5, 6, 7, 8, 1],
    [3, 4, 5, 6, 7, 8, 1, 2],
    [],
]
MISSION_PAD_LAYOUTS = {
    "standard": MISSION_PAD_COLUMNS,
    "column": COLUMN_MISSION_PAD_COLUMNS,
    "column_75": COLUMN_75_MISSION_PAD_COLUMNS,
    "vee": VEE_MISSION_PAD_COLUMNS,
    "diamond": DIAMOND_MISSION_PAD_COLUMNS,
}


def mission_pad_layout_for_formation(formation, inter_drone_distance_cm=None):
    formation = str(formation or "").strip().lower()
    try:
        distance_cm = int(str(inter_drone_distance_cm or "").strip())
    except ValueError:
        distance_cm = None
    if formation == "column":
        if distance_cm == 75:
            return COLUMN_75_MISSION_PAD_COLUMNS
        return COLUMN_MISSION_PAD_COLUMNS
    if formation == "diamond":
        return DIAMOND_MISSION_PAD_COLUMNS
    if formation == "vee":
        if distance_cm == 75:
            return MISSION_PAD_COLUMNS
        return VEE_MISSION_PAD_COLUMNS
    return MISSION_PAD_COLUMNS


def python_executable():
    executable = Path(sys.executable)
    if executable.exists():
        return str(executable)
    return shutil.which("python3") or "python3"
EXPERIMENT_SCRIPTS = {
    "front": "data_collector.py",
    "column": "data_collector.py",
    "vee": "data_collector.py",
    "echalon": "data_collector.py",
    "echelon": "data_collector.py",
    "echolon": "data_collector.py",
    "diamond": "data_collector.py",
}
TAKEOFF_PROMPT = "Press Enter to take off"
DISCHARGE_PROMPT = "Press Enter to discharge high-battery drones"
WIND_DIRECTION_CODES = {
    "head wind": "head",
    "tail wind": "tail",
    "side wind": "side",
}
WIND_SPEED_CODES = {
    "no wind": "no_wind",
    "Level1": "lv1",
    "Level2": "lv2",
    "Level3": "lv3",
}
WIND_SPEED_OPTIONS = ["no wind", "Level1", "Level2", "Level3"]
WIND_LEVEL_ALIASES = {
    "lv1": "Level1",
    "lv 1": "Level1",
    "level1": "Level1",
    "level 1": "Level1",
    "l1": "Level1",
    "1": "Level1",
    "lv2": "Level2",
    "lv 2": "Level2",
    "level2": "Level2",
    "level 2": "Level2",
    "l2": "Level2",
    "2": "Level2",
    "lv3": "Level3",
    "lv 3": "Level3",
    "level3": "Level3",
    "level 3": "Level3",
    "l3": "Level3",
    "3": "Level3",
}


app = Flask(__name__)
RUN_LOCK = threading.Lock()
RUN_STATE = {
    "run_id": None,
    "experiment_id": None,
    "formation": None,
    "script": None,
    "status": "idle",
    "message": "",
    "ready_for_takeoff": False,
    "ready_for_discharge": False,
    "prompt_action": "",
    "takeoff_confirmed": False,
    "started_at": None,
    "ended_at": None,
    "returncode": None,
    "process": None,
    "output": [],
    "baseline_config": None,
}


def load_registry():
    if not REGISTRY_FILE.exists():
        return {"experiments": []}
    try:
        with REGISTRY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"experiments": []}
    if not isinstance(data, dict):
        return {"experiments": []}
    data.setdefault("experiments", [])
    data.setdefault("batteries", [{"battery_id": battery_id} for battery_id in BATTERY_OPTIONS])
    return data


def save_registry(data):
    with REGISTRY_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def remove_file_references(relpath):
    registry = load_registry()
    changed = False
    for experiment in registry.get("experiments", []):
        for key in ("coordination_files", "battery_files", "image_files"):
            values = experiment.get(key, [])
            if relpath in values:
                experiment[key] = [item for item in values if item != relpath]
                experiment["updated_at"] = datetime.now().isoformat(timespec="seconds")
                changed = True
    if changed:
        save_registry(registry)


def remove_empty_parent_dirs(path):
    current = path.parent
    while current != DATA_DIR and DATA_DIR in current.parents:
        try:
            if any(current.iterdir()):
                break
            current.rmdir()
        except OSError:
            break
        current = current.parent


def safe_relative_path(raw_path):
    path = (DATA_DIR / raw_path).resolve()
    if path == DATA_DIR or DATA_DIR not in path.parents:
        abort(404)
    if not path.exists() or not path.is_file():
        abort(404)
    return path


def human_size(num_bytes):
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def safe_slug(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def normalize_wind_level(value):
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.lower().replace("_", " ")
    if normalized in {"lv1", "lv 1", "level1", "level 1", "l1", "1"}:
        return "lv1"
    if normalized in {"lv2", "lv 2", "level2", "level 2", "l2", "2"}:
        return "lv2"
    if normalized in {"lv3", "lv 3", "level3", "level 3", "l3", "3"}:
        return "lv3"
    return text if text in {key for key, _ in BASELINE_WIND_LEVELS} else ""


def wind_level_from_notes(notes):
    text = str(notes or "").lower().replace("_", " ")
    match = re.search(r"(?:^|[^a-z0-9])(?:lv|level|l)\s*([123])\b", text)
    if match:
        return f"lv{match.group(1)}"
    return ""


def baseline_wind_level(meta=None, notes=""):
    meta = meta or {}
    for key in ("baseline_wind_level", "wind_level"):
        level = normalize_wind_level(meta.get(key))
        if level:
            return level
    return wind_level_from_notes(notes or meta.get("notes", ""))


def classify_file(path):
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix == ".csv":
        if "battery" in name or "summary" in name or "calculate_time" in name:
            return "battery"
        return "coordination"
    if suffix == ".py":
        return "script"
    return "other"


def short_experiment_id(raw_id):
    text = str(raw_id or "").strip()
    if not text:
        return datetime.now().strftime("%H%M%S")
    if text.isdigit():
        return text.zfill(3)
    return text


def short_new_experiment_id(raw_id):
    short_id = short_experiment_id(raw_id)
    if short_id.startswith("new_"):
        return short_id
    return f"new_{short_id}"


def normalize_inter_drone_distance_cm(value):
    try:
        distance = int(str(value or 50).strip())
    except ValueError:
        distance = 50
    if distance not in INTER_DRONE_DISTANCE_OPTIONS_CM:
        distance = 50
    return str(distance)


def apply_distance_to_experiment_id(experiment_id, formation, inter_drone_distance_cm):
    text = str(experiment_id or "").strip()
    if not text:
        return text
    formation_code = (formation or "custom").strip().lower().replace(" ", "_")
    parts = text.split("_")
    distance_code = normalize_inter_drone_distance_cm(inter_drone_distance_cm)
    if parts and parts[0] == formation_code:
        if len(parts) > 1 and parts[1] in {"50", "75"}:
            parts[1] = distance_code
            return "_".join(parts)
        return "_".join([parts[0], distance_code] + parts[1:])
    return text


def build_full_experiment_id(raw_id, formation, wind_direction, wind_speed, inter_drone_distance_cm=50, new_prefix=False):
    text = str(raw_id or "").strip()
    if text and not text.isdigit() and "_" in text:
        return apply_distance_to_experiment_id(text, formation, inter_drone_distance_cm)
    formation_code = (formation or "custom").strip().lower().replace(" ", "_")
    distance_code = normalize_inter_drone_distance_cm(inter_drone_distance_cm)
    direction_code = WIND_DIRECTION_CODES.get(wind_direction, (wind_direction or "wind").replace(" ", "_"))
    speed_code = WIND_SPEED_CODES.get(wind_speed, (wind_speed or "level").lower())
    short_id = short_new_experiment_id(text) if new_prefix else short_experiment_id(text)
    return f"{formation_code}_{distance_code}_{direction_code}_{speed_code}_{short_id}"


def normalize_drone_identifier(value):
    text = str(value or "").strip()
    if not text:
        return "", "", ""
    if text.startswith(IP_PREFIX):
        suffix = text.removeprefix(IP_PREFIX)
        return text, suffix, IP_SUFFIX_TO_DRONE_NUMBER.get(suffix, "")
    suffix = DRONE_NUMBER_TO_IP_SUFFIX.get(text, text)
    return f"{IP_PREFIX}{suffix}", suffix, IP_SUFFIX_TO_DRONE_NUMBER.get(suffix, text if text in DRONE_NUMBER_TO_IP_SUFFIX else "")


def display_drone_number(drone):
    number = str(drone.get("drone_number") or "").strip()
    if number:
        return number
    suffix = str(drone.get("ip_suffix") or drone.get("ip", "")).replace(IP_PREFIX, "").strip()
    return IP_SUFFIX_TO_DRONE_NUMBER.get(suffix, suffix)


def normalize_battery_id(value):
    text = str(value or "").strip().upper().replace(" ", "")
    if not text:
        return ""
    if text.isdigit():
        return f"B{int(text):02d}"
    if text.startswith("B") and text[1:].isdigit():
        return f"B{int(text[1:]):02d}"
    return text


def display_battery_id(drone):
    return normalize_battery_id(drone.get("battery_id", ""))


def read_csv_header(path):
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            return next(reader, [])
    except (OSError, UnicodeDecodeError, StopIteration):
        return []


def analyze_csv(path, max_trials=8):
    header = read_csv_header(path)
    row_count = 0
    trials = Counter()
    formations = Counter()
    drones = Counter()
    battery_columns = [col for col in header if "battery" in col.lower()]
    coordination_columns = [
        col for col in header
        if col in {
            "X_global", "Y_global", "Z_global", "x", "y", "z", "mid",
            "formation_error_x", "formation_error_y", "formation_error_dist",
            "position_error_x", "position_error_y", "position_error_z",
            "position_error_dist", "dist_12_cm", "dist_23_cm", "dist_13_cm",
            "mean_intra_dist",
        }
    ]
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                trial_id = (row.get("trial_id") or row.get("actual_trial_id") or "").strip()
                formation = (row.get("formation_type") or row.get("experiment_type") or "").strip()
                drone = (row.get("drone_name") or row.get("drone_ip") or "").strip()
                if trial_id:
                    trials[trial_id] += 1
                if formation:
                    formations[formation] += 1
                if drone:
                    drones[drone] += 1
    except (OSError, UnicodeDecodeError, csv.Error):
        pass
    return {
        "header": header,
        "row_count": row_count,
        "trial_ids": [trial for trial, _ in trials.most_common(max_trials)],
        "trial_count": len(trials),
        "formations": [formation for formation, _ in formations.most_common(5)],
        "drones": [drone for drone, _ in drones.most_common(5)],
        "battery_columns": battery_columns,
        "coordination_columns": coordination_columns,
    }


def scan_workspace():
    files = []
    categories = defaultdict(list)
    for path in sorted(DATA_DIR.rglob("*"), key=lambda item: item.relative_to(DATA_DIR).as_posix().lower()):
        if path.name.startswith(".") or path.name == "__pycache__" or path.name == REGISTRY_FILE.name:
            continue
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        category = classify_file(path)
        stat = path.stat()
        rel = path.relative_to(DATA_DIR).as_posix()
        info = {
            "name": path.name,
            "relpath": rel,
            "folder": path.parent.relative_to(DATA_DIR).as_posix() if path.parent != DATA_DIR else ".",
            "suffix": suffix,
            "category": category,
            "size": human_size(stat.st_size),
            "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "previewable": suffix in ALLOWED_PREVIEW_EXTENSIONS,
            "is_image": suffix in IMAGE_EXTENSIONS,
            "is_csv": suffix in CSV_EXTENSIONS,
        }
        if suffix == ".csv":
            info["csv"] = analyze_csv(path)
        files.append(info)
        categories[category].append(info)
    return files, categories


def csv_preview(path, limit=80):
    rows = []
    header = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            for idx, row in enumerate(reader):
                if idx >= limit:
                    break
                rows.append(row)
    except (OSError, UnicodeDecodeError, StopIteration, csv.Error):
        pass
    return header, rows


def build_experiment_from_form(form):
    drone_rows = []
    for key, raw_ip in form.items():
        if not key.startswith("pad_ip_"):
            continue
        drone_identifier = raw_ip.strip()
        if not drone_identifier:
            continue
        parts = key.removeprefix("pad_ip_").split("_")
        if len(parts) != 2:
            continue
        col, row = parts
        pad_id = form.get(f"pad_id_{col}_{row}", "").strip()
        role = form.get(f"pad_role_{col}_{row}", "").strip()
        order = form.get(f"pad_order_{col}_{row}", "").strip()
        battery_id = normalize_battery_id(form.get(f"pad_battery_{col}_{row}", ""))
        ip, suffix, drone_number = normalize_drone_identifier(drone_identifier)
        drone_rows.append({
            "ip": ip,
            "ip_suffix": suffix,
            "drone_number": drone_number,
            "battery_id": battery_id,
            "takeoff_order": order,
            "role": role,
            "mission_pad": pad_id,
            "grid_column": col,
            "grid_row": row,
        })
    drone_rows.sort(key=lambda item: int(item["takeoff_order"] or 999))
    now = datetime.now().isoformat(timespec="seconds")
    formation = form.get("formation", "").strip()
    wind_direction = form.get("wind_direction", "").strip()
    wind_speed = form.get("wind_speed", "").strip()
    inter_drone_distance_cm = form.get("inter_drone_distance_cm", "50").strip() or "50"
    experiment_id = build_full_experiment_id(
        form.get("experiment_id", "").strip(),
        formation,
        wind_direction,
        wind_speed,
        inter_drone_distance_cm,
        new_prefix=True,
    )
    return {
        "experiment_id": experiment_id,
        "short_id": short_new_experiment_id(form.get("experiment_id", "").strip()),
        "formation": formation,
        "wind_direction": wind_direction,
        "wind_speed": wind_speed,
        "inter_drone_distance_cm": inter_drone_distance_cm,
        "status": form.get("status", "planned").strip() or "planned",
        "created_at": now,
        "updated_at": now,
        "coordination_files": [],
        "battery_files": [],
        "image_files": [],
        "drones": drone_rows,
        "notes": form.get("notes", "").strip(),
    }


def build_edit_drones_from_form(form):
    drone_rows = []
    ips = form.getlist("edit_drone_number")
    orders = form.getlist("edit_takeoff_order")
    roles = form.getlist("edit_drone_role")
    pads = form.getlist("edit_mission_pad")
    columns = form.getlist("edit_grid_column")
    rows = form.getlist("edit_grid_row")
    batteries = form.getlist("edit_battery_id")
    total = max(len(ips), len(orders), len(roles), len(pads), len(columns), len(rows), len(batteries))
    for idx in range(total):
        drone_identifier = ips[idx].strip() if idx < len(ips) else ""
        order = orders[idx].strip() if idx < len(orders) else ""
        role = roles[idx].strip() if idx < len(roles) else ""
        pad = pads[idx].strip() if idx < len(pads) else ""
        column = columns[idx].strip() if idx < len(columns) else ""
        row = rows[idx].strip() if idx < len(rows) else ""
        battery_id = normalize_battery_id(batteries[idx]) if idx < len(batteries) else ""
        if not any([drone_identifier, order, role, pad, column, row, battery_id]):
            continue
        ip, suffix, drone_number = normalize_drone_identifier(drone_identifier)
        drone_rows.append({
            "ip": ip,
            "ip_suffix": suffix,
            "drone_number": drone_number,
            "battery_id": battery_id,
            "takeoff_order": order,
            "role": role,
            "mission_pad": pad,
            "grid_column": column,
            "grid_row": row,
        })
    drone_rows.sort(key=lambda item: int(item["takeoff_order"] or 999))
    return drone_rows


def validate_experiment_assignments(experiment):
    drones = experiment.get("drones", [])
    drone_numbers = [str(drone.get("drone_number") or "").strip() for drone in drones if str(drone.get("drone_number") or "").strip()]
    battery_ids = [normalize_battery_id(drone.get("battery_id")) for drone in drones if normalize_battery_id(drone.get("battery_id"))]
    if len(drone_numbers) != len(set(drone_numbers)):
        abort(400, description="Each drone can only be assigned once in an experiment.")
    if len(battery_ids) != len(set(battery_ids)):
        abort(400, description="Each battery can only be assigned once in an experiment.")
    mission_pad_columns = mission_pad_layout_for_formation(
        experiment.get("formation"),
        experiment.get("inter_drone_distance_cm"),
    )
    seen_positions = set()
    for drone in drones:
        try:
            grid_column = int(str(drone.get("grid_column")).strip())
            grid_row = int(str(drone.get("grid_row")).strip())
            mission_pad = int(str(drone.get("mission_pad")).strip())
        except ValueError:
            abort(400, description="Each active drone must have valid mission pad, grid column, and grid row values.")
        if grid_column < 0 or grid_column >= len(mission_pad_columns):
            abort(400, description=f"Grid column out of range for drone {drone.get('takeoff_order')}: {grid_column}.")
        if grid_row < 0 or grid_row >= len(mission_pad_columns[grid_column]):
            abort(400, description=f"Grid row out of range for drone {drone.get('takeoff_order')}: {grid_row}.")
        expected_pad = mission_pad_columns[grid_column][grid_row]
        if mission_pad != expected_pad:
            abort(
                400,
                description=(
                    f"Mission pad mismatch for drone {drone.get('takeoff_order')}: "
                    f"record pad={mission_pad}, layout pad={expected_pad} "
                    f"at column={grid_column}, row={grid_row}."
                ),
            )
        position = (grid_column, grid_row)
        if position in seen_positions:
            abort(400, description=f"Two drones are assigned to the same grid position: {position}.")
        seen_positions.add(position)


def auto_attach_files(experiment, files):
    exp_id = experiment["experiment_id"].lower()
    formation = experiment.get("formation", "").lower()
    for item in files:
        name = item["name"].lower()
        matched = exp_id in name or (formation and formation in name)
        if not matched:
            continue
        if item["category"] == "coordination" and item["relpath"] not in experiment["coordination_files"]:
            experiment["coordination_files"].append(item["relpath"])
        if item["category"] == "battery" and item["relpath"] not in experiment["battery_files"]:
            experiment["battery_files"].append(item["relpath"])
        if item["category"] == "image" and item["relpath"] not in experiment["image_files"]:
            experiment["image_files"].append(item["relpath"])


def experiment_id_from_relpath(relpath):
    parts = Path(relpath).parts
    if len(parts) > 1:
        return parts[0]
    name = Path(relpath).name
    if "_" in name:
        return name.split("_20", 1)[0]
    return ""


def condition_key_from_experiment_id(experiment_id):
    parts = str(experiment_id or "").rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return str(experiment_id or "")


def trial_number_from_experiment_id(experiment_id):
    parts = str(experiment_id or "").rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    return ""


def is_new_experiment(exp):
    experiment_id = str(exp.get("experiment_id") or "")
    short_id = str(exp.get("short_id") or "")
    return "_new_" in experiment_id or short_id.startswith("new_")


def filter_new_experiments(experiments):
    return [exp for exp in experiments if is_new_experiment(exp)]


def group_experiments_by_condition(experiments):
    groups = []
    grouped = defaultdict(list)
    for exp in experiments:
        grouped[condition_key_from_experiment_id(exp.get("experiment_id"))].append(exp)
    for key, trials in sorted(grouped.items()):
        trials = sorted(trials, key=lambda exp: trial_number_from_experiment_id(exp.get("experiment_id")) or exp.get("experiment_id", ""))
        first = trials[0] if trials else {}
        groups.append({
            "condition_key": key,
            "formation": first.get("formation", ""),
            "wind_direction": first.get("wind_direction", ""),
            "wind_speed": first.get("wind_speed", ""),
            "inter_drone_distance_cm": first.get("inter_drone_distance_cm", 50),
            "trial_count": len(trials),
            "included_trial_count": sum(1 for trial in trials if not trial.get("is_outlier")),
            "outlier_count": sum(1 for trial in trials if trial.get("is_outlier")),
            "trials": trials,
        })
    return groups


def summarize_experiment_archive(experiment_id):
    experiment_dir = DATA_DIR / experiment_id
    summary = {
        "exists": experiment_dir.exists(),
        "files": [],
        "plots": [],
        "drones": [],
        "all_battery": [],
    }
    if not experiment_dir.exists():
        return summary
    for path in sorted(experiment_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(DATA_DIR).as_posix()
        item = {
            "name": path.name,
            "relpath": rel,
            "size": human_size(path.stat().st_size),
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "is_image": path.suffix.lower() in IMAGE_EXTENSIONS,
            "is_csv": path.suffix.lower() == ".csv",
        }
        if "/plots/" in rel and item["is_image"] and path.name.startswith("all_"):
            summary["plots"].append(item)
        else:
            summary["files"].append(item)
    battery_path = latest_archive_file(experiment_dir, "*_all_battery.csv")
    if battery_path:
        try:
            with battery_path.open("r", newline="", encoding="utf-8-sig") as f:
                summary["all_battery"] = list(csv.DictReader(f))
        except (OSError, csv.Error):
            pass
    return summary


def summarize_condition_archive(condition_key):
    summary_dir = DATA_DIR / f"{condition_key}_summary"
    plots_dir = summary_dir / "plots"
    plots = []
    if plots_dir.exists():
        for path in sorted(plots_dir.glob("*.png")):
            plots.append({
                "name": path.name,
                "relpath": path.relative_to(DATA_DIR).as_posix(),
                "size": human_size(path.stat().st_size),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
    return {"summary_dir": summary_dir, "plots": plots}


def experiment_filter_options(experiments):
    wind_speeds = {exp.get("wind_speed", "") for exp in experiments if exp.get("wind_speed")}
    return {
        "formations": FORMATION_OPTIONS,
        "wind_directions": sorted({exp.get("wind_direction", "") for exp in experiments if exp.get("wind_direction")}),
        "wind_speeds": WIND_SPEED_OPTIONS + sorted(wind_speeds - set(WIND_SPEED_OPTIONS)),
    }


def selected_values(args, name, normalizer=None):
    values = []
    for value in args.getlist(name):
        text = str(value or "").strip()
        if not text:
            continue
        if normalizer:
            text = normalizer(text)
        if text and text not in values:
            values.append(text)
    return values


def value_matches_filter(value, selected):
    if not selected:
        return True
    return value in selected


def filter_experiments(experiments, formation=None, wind_direction=None, wind_speed=None):
    formation = formation or []
    wind_direction = wind_direction or []
    wind_speed = wind_speed or []
    filtered = []
    for exp in experiments:
        if not value_matches_filter(exp.get("formation"), formation):
            continue
        if not value_matches_filter(exp.get("wind_direction"), wind_direction):
            continue
        if not value_matches_filter(exp.get("wind_speed"), wind_speed):
            continue
        filtered.append(exp)
    return filtered


def read_first_csv_row(path):
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
            return rows[0] if rows else {}
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}


def read_json_file(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def scan_baseline_runs():
    runs = []
    seen = set()
    if not BASELINE_DIR.exists():
        return runs
    for summary_path in sorted(BASELINE_DIR.rglob("*_summary.csv"), key=lambda item: item.stat().st_mtime, reverse=True):
        row = read_first_csv_row(summary_path)
        if not row:
            continue
        baseline_id = row.get("baseline_id") or summary_path.name.removesuffix("_summary.csv")
        seen.add(baseline_id)
        folder = summary_path.parent
        timeseries = folder / f"{baseline_id}_timeseries.csv"
        metadata = folder / f"{baseline_id}_metadata.json"
        meta = read_json_file(metadata) if metadata.exists() else {}
        notes = row.get("notes", meta.get("notes", ""))
        plots_dir = folder / "plots"
        plots = []
        if plots_dir.exists():
            plots = [
                path.relative_to(DATA_DIR).as_posix()
                for path in sorted(plots_dir.glob(f"{baseline_id}_*.png"))
            ]
        runs.append({
            "baseline_id": baseline_id,
            "run_id": row.get("run_id", ""),
            "drone_name": row.get("drone_name", ""),
            "drone_number": row.get("drone_number", ""),
            "drone_ip": row.get("drone_ip", ""),
            "battery_id": normalize_battery_id(row.get("battery_id", "")),
            "mode": row.get("mode", ""),
            "wind_level": row.get("wind_level", ""),
            "direction": row.get("direction", ""),
            "wind_level": baseline_wind_level(meta, notes),
            "notes": notes,
            "baseline_path": row.get("baseline_path", ""),
            "duration_sec": row.get("duration_sec", ""),
            "battery_start": row.get("battery_start", ""),
            "battery_end": row.get("battery_end", ""),
            "battery_drop": row.get("battery_drop", ""),
            "end_reason": row.get("end_reason", ""),
            "summary_relpath": summary_path.relative_to(DATA_DIR).as_posix(),
            "timeseries_relpath": timeseries.relative_to(DATA_DIR).as_posix() if timeseries.exists() else "",
            "metadata_relpath": metadata.relative_to(DATA_DIR).as_posix() if metadata.exists() else "",
            "plots": plots,
            "mtime": datetime.fromtimestamp(summary_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    for battery_path in sorted(BASELINE_DIR.rglob("*_all_battery.csv"), key=lambda item: item.stat().st_mtime, reverse=True):
        row = read_first_csv_row(battery_path)
        if not row:
            continue
        baseline_id = row.get("experiment_id") or battery_path.name.removesuffix("_all_battery.csv")
        if baseline_id in seen:
            continue
        seen.add(baseline_id)
        folder = battery_path.parent
        metadata_candidates = sorted(folder.glob(f"{baseline_id}_metadata.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        metadata = metadata_candidates[0] if metadata_candidates else None
        meta = read_json_file(metadata) if metadata else {}
        notes = meta.get("notes", "")
        coordination = latest_archive_file(folder, f"{baseline_id}_*_all_coordination.csv")
        plots_dir = folder / "plots"
        plots = []
        if plots_dir.exists():
            plots = [
                path.relative_to(DATA_DIR).as_posix()
                for path in sorted(plots_dir.glob("*.png"))
                if baseline_id in path.name
            ]
        drone_number = str(meta.get("drone_number", "") or "")
        if not drone_number:
            match = re.search(r"drone_(\d+)", row.get("drone_name", ""))
            drone_number = match.group(1) if match else ""
        runs.append({
            "baseline_id": baseline_id,
            "run_id": row.get("run_id", meta.get("run_id", "")),
            "drone_name": row.get("drone_name", meta.get("drone_name", "")),
            "drone_number": drone_number,
            "drone_ip": row.get("drone_ip", meta.get("drone_ip", "")),
            "battery_id": normalize_battery_id(row.get("battery_id", meta.get("battery_id", ""))),
            "mode": meta.get("mode", row.get("wind_direction", "")),
            "wind_level": meta.get("wind_level", row.get("wind_level", "")),
            "direction": meta.get("direction", row.get("wind_speed", "")),
            "wind_level": baseline_wind_level(meta, notes),
            "notes": notes,
            "baseline_path": " -> ".join(str(item) for item in meta.get("baseline_path", [])),
            "duration_sec": row.get("node_duration_sec", row.get("hover_duration_sec", "")),
            "battery_start": row.get("battery_hover_start", ""),
            "battery_end": row.get("battery_hover_end", ""),
            "battery_drop": row.get("battery_drop", ""),
            "end_reason": "formal-format single baseline",
            "summary_relpath": battery_path.relative_to(DATA_DIR).as_posix(),
            "timeseries_relpath": coordination.relative_to(DATA_DIR).as_posix() if coordination else "",
            "metadata_relpath": metadata.relative_to(DATA_DIR).as_posix() if metadata else "",
            "plots": plots,
            "mtime": datetime.fromtimestamp(battery_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    for metadata_path in sorted(BASELINE_DIR.rglob("*_metadata.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        meta = read_json_file(metadata_path)
        baseline_id = meta.get("baseline_id") or metadata_path.name.removesuffix("_metadata.json")
        if not baseline_id or baseline_id in seen:
            continue
        folder = metadata_path.parent
        timeseries = folder / f"{baseline_id}_timeseries.csv"
        if not timeseries.exists():
            timeseries = latest_archive_file(folder, f"{baseline_id}_*_all_coordination.csv")
        battery_path = latest_archive_file(folder, f"{baseline_id}_*_all_battery.csv")
        battery_row = read_first_csv_row(battery_path) if battery_path else {}
        timeseries_rows = []
        if timeseries and timeseries.exists():
            try:
                with timeseries.open("r", newline="", encoding="utf-8-sig") as f:
                    timeseries_rows = list(csv.DictReader(f))
            except (OSError, UnicodeDecodeError, csv.Error):
                timeseries_rows = []
        first_ts = timeseries_rows[0] if timeseries_rows else {}
        last_ts = timeseries_rows[-1] if timeseries_rows else {}
        battery_start = battery_row.get("battery_hover_start", first_ts.get("battery_start", first_ts.get("battery", "")))
        battery_end = battery_row.get("battery_hover_end", last_ts.get("battery", ""))
        try:
            battery_drop = str(int(float(battery_start)) - int(float(battery_end)))
        except (TypeError, ValueError):
            battery_drop = battery_row.get("battery_drop", "")
        duration = battery_row.get("node_duration_sec") or battery_row.get("hover_duration_sec")
        if not duration and first_ts and last_ts:
            duration = str(baseline_time_value(last_ts) or "")
        drone_number = str(meta.get("drone_number", "") or "")
        drone_name = meta.get("drone_name", "")
        if not drone_number:
            match = re.search(r"drone_(\d+)", drone_name)
            drone_number = match.group(1) if match else ""
        plots_dir = folder / "plots"
        plots = []
        if plots_dir.exists():
            plots = [
                path.relative_to(DATA_DIR).as_posix()
                for path in sorted(plots_dir.glob(f"{baseline_id}*.png"))
            ]
        seen.add(baseline_id)
        runs.append({
            "baseline_id": baseline_id,
            "run_id": meta.get("run_id", ""),
            "drone_name": drone_name,
            "drone_number": drone_number,
            "drone_ip": meta.get("drone_ip", ""),
            "battery_id": normalize_battery_id(meta.get("battery_id", "")),
            "mode": meta.get("mode", ""),
            "wind_level": meta.get("wind_level", ""),
            "direction": meta.get("direction", ""),
            "wind_level": baseline_wind_level(meta),
            "notes": meta.get("notes", ""),
            "baseline_path": " -> ".join(str(item) for item in meta.get("baseline_path", [])),
            "duration_sec": duration or "",
            "battery_start": battery_start,
            "battery_end": battery_end,
            "battery_drop": battery_drop,
            "end_reason": "metadata baseline run",
            "summary_relpath": battery_path.relative_to(DATA_DIR).as_posix() if battery_path else "",
            "timeseries_relpath": timeseries.relative_to(DATA_DIR).as_posix() if timeseries and timeseries.exists() else "",
            "metadata_relpath": metadata_path.relative_to(DATA_DIR).as_posix(),
            "plots": plots,
            "mtime": datetime.fromtimestamp(metadata_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return runs


def baseline_filter_options(runs):
    return {
        "drones": sorted({run["drone_number"] for run in runs if run.get("drone_number")}, key=lambda value: int(value)),
        "batteries": sorted({run["battery_id"] for run in runs if run.get("battery_id")}),
        "modes": sorted({run["mode"] for run in runs if run.get("mode")}),
        "wind_levels": sorted({run["wind_level"] for run in runs if run.get("wind_level")}),
    }


def filter_baseline_runs(runs, drone_number=None, battery_id=None, mode=None, wind_level=None):
    drone_number = drone_number or []
    battery_id = [normalize_battery_id(value) for value in (battery_id or []) if normalize_battery_id(value)]
    mode = mode or []
    wind_level = [normalize_wind_level(value) for value in (wind_level or []) if normalize_wind_level(value)]
    filtered = []
    for run in runs:
        if not value_matches_filter(run.get("drone_number"), drone_number):
            continue
        if not value_matches_filter(run.get("battery_id"), battery_id):
            continue
        if not value_matches_filter(run.get("mode"), mode):
            continue
        if not value_matches_filter(run.get("wind_level"), wind_level):
            continue
        filtered.append(run)
    return filtered


def baseline_summary_key(drone_number=None, battery_id=None, mode=None, wind_level=None):
    drone_number = drone_number or []
    battery_id = [normalize_battery_id(value) for value in (battery_id or []) if normalize_battery_id(value)]
    mode = mode or []
    wind_level = [normalize_wind_level(value) for value in (wind_level or []) if normalize_wind_level(value)]
    parts = ["baseline"]
    if mode:
        parts.append("mode_" + "_".join(safe_slug(value) for value in mode))
    if wind_level:
        parts.append("wind_" + "_".join(safe_slug(value) for value in wind_level))
    if drone_number:
        parts.append("drone_" + "_".join(safe_slug(value) for value in drone_number))
    if battery_id:
        parts.append("battery_" + "_".join(safe_slug(value) for value in battery_id))
    return "_".join(parts)


def latest_archive_file(folder, pattern):
    files = sorted(folder.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def drone_archive_summary(experiment_id, drone):
    suffix = str(drone.get("ip_suffix") or drone.get("ip", "")).replace(IP_PREFIX, "")
    name = str(drone.get("role") or drone.get("takeoff_order") or "drone")
    folder_prefix = f"drone_{drone.get('takeoff_order')}_{suffix}_pad{drone.get('mission_pad')}"
    drones_dir = DATA_DIR / experiment_id / "drones"
    candidates = []
    if drones_dir.exists():
        candidates = [path for path in drones_dir.iterdir() if path.is_dir() and suffix in path.name and f"pad{drone.get('mission_pad')}" in path.name]
    drone_dir = candidates[0] if candidates else drones_dir / folder_prefix
    coord = latest_archive_file(drone_dir, "*_coordination.csv")
    battery = latest_archive_file(drone_dir, "*_battery.csv")
    plot = DATA_DIR / experiment_id / "plots" / f"drone_{drone.get('takeoff_order')}_overview.png"
    battery_row = {}
    if battery and battery.exists():
        try:
            with battery.open("r", newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
                battery_row = rows[0] if rows else {}
        except (OSError, csv.Error):
            pass
    return {
        "label": name,
        "suffix": suffix,
        "folder": drone_dir.relative_to(DATA_DIR).as_posix() if drone_dir.exists() else "",
        "coordination": coord.relative_to(DATA_DIR).as_posix() if coord else "",
        "battery": battery.relative_to(DATA_DIR).as_posix() if battery else "",
        "plot": plot.relative_to(DATA_DIR).as_posix() if plot.exists() else "",
        "battery_row": battery_row,
    }


def update_experiment_from_form(experiment, form):
    now = datetime.now().isoformat(timespec="seconds")
    formation = form.get("formation", experiment.get("formation", "")).strip()
    wind_direction = form.get("wind_direction", experiment.get("wind_direction", "")).strip()
    wind_speed = form.get("wind_speed", experiment.get("wind_speed", "")).strip()
    inter_drone_distance_cm = form.get(
        "inter_drone_distance_cm",
        experiment.get("inter_drone_distance_cm", "50"),
    ).strip() or "50"
    experiment["experiment_id"] = build_full_experiment_id(
        form.get("experiment_id", experiment.get("experiment_id", "")).strip(),
        formation,
        wind_direction,
        wind_speed,
        inter_drone_distance_cm,
    )
    experiment["short_id"] = short_experiment_id(form.get("experiment_id", experiment.get("short_id", "")).strip())
    experiment["formation"] = formation
    experiment["wind_direction"] = wind_direction
    experiment["wind_speed"] = wind_speed
    experiment["inter_drone_distance_cm"] = inter_drone_distance_cm
    experiment["status"] = form.get("status", experiment.get("status", "planned")).strip() or "planned"
    experiment["notes"] = form.get("notes", experiment.get("notes", "")).strip()
    experiment["drones"] = build_edit_drones_from_form(form)
    experiment["updated_at"] = now
    return experiment


def find_experiment_index(experiments, experiment_id):
    for idx, experiment in enumerate(experiments):
        if experiment.get("experiment_id") == experiment_id:
            return idx
    return None


def experiment_by_id(experiment_id):
    if not experiment_id:
        return None
    registry = load_registry()
    idx = find_experiment_index(registry.get("experiments", []), experiment_id)
    if idx is None:
        return None
    return registry["experiments"][idx]


def battery_level_band(value):
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if percent > EXPERIMENT_BATTERY_WINDOW["high"]:
        return "above"
    if percent < EXPERIMENT_BATTERY_WINDOW["low"]:
        return "below"
    return "window"


def battery_window_progress(value):
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return 0
    low = EXPERIMENT_BATTERY_WINDOW["low"]
    high = EXPERIMENT_BATTERY_WINDOW["high"]
    if high <= low:
        return 0
    return max(0, min(100, round((percent - low) / (high - low) * 100, 1)))


def live_batteries_from_output(output_lines):
    status = {"by_ip": {}, "by_battery": {}, "by_name": {}}
    current = None
    connect_pattern = re.compile(
        r"Connecting\s+(?P<name>\S+)\s+\((?P<ip>[^,\)]+)(?:,\s+battery\s+(?P<battery>[A-Za-z0-9]+))?"
    )
    ok_pattern = re.compile(r"OK\s+-\s+battery:\s*(?P<percent>\d+(?:\.\d+)?)%")
    window_pattern = re.compile(
        r"(?P<name>drone_\d+)\s+\((?P<ip>[^,\)]+),\s+battery\s+(?P<battery>[A-Za-z0-9]+)\):\s*(?P<percent>\d+(?:\.\d+)?)%"
    )
    live_status_pattern = re.compile(
        r"(?P<name>drone_\d+)\s+battery_id=(?P<battery>[A-Za-z0-9]+)\s+battery=(?P<percent>\d+(?:\.\d+)?)%"
    )

    def remember(item):
        ip = item.get("drone_ip")
        battery_id = normalize_battery_id(item.get("battery_id", ""))
        drone_name = item.get("drone_name")
        if battery_id:
            item["battery_id"] = battery_id
        if ip:
            status["by_ip"][ip] = item
        if battery_id:
            status["by_battery"][battery_id] = item
        if drone_name:
            status["by_name"][drone_name] = item

    for line in output_lines:
        window_match = window_pattern.search(line)
        if window_match:
            remember({
                "drone_name": window_match.group("name"),
                "drone_ip": window_match.group("ip").strip(),
                "battery_id": window_match.group("battery"),
                "battery_percent": float(window_match.group("percent")),
                "source": "battery_window_check",
            })
            continue
        live_matches = list(live_status_pattern.finditer(line))
        if live_matches:
            source = "live_output" if line.startswith("Live battery status:") else "discharge_output"
            for match in live_matches:
                remember({
                    "drone_name": match.group("name"),
                    "battery_id": match.group("battery"),
                    "battery_percent": float(match.group("percent")),
                    "phase": "battery_discharge_hover" if source == "discharge_output" else "",
                    "source": source,
                })
            continue
        connect_match = connect_pattern.search(line)
        if connect_match:
            current = {
                "drone_name": connect_match.group("name"),
                "drone_ip": connect_match.group("ip").strip(),
                "battery_id": normalize_battery_id(connect_match.group("battery") or ""),
            }
            continue
        ok_match = ok_pattern.search(line)
        if ok_match and current:
            remember({
                **current,
                "battery_percent": float(ok_match.group("percent")),
                "source": "preflight",
            })
            current = None
    return status


def latest_coordination_batteries(experiment_id):
    experiment_dir = DATA_DIR / str(experiment_id or "")
    coord_path = latest_archive_file(experiment_dir, "*_all_coordination.csv")
    if not coord_path:
        return {}
    latest = {}
    try:
        with coord_path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                ip = (row.get("drone_ip") or "").strip()
                battery = (row.get("battery") or "").strip()
                if not ip or not battery:
                    continue
                try:
                    battery_percent = float(battery)
                except ValueError:
                    continue
                latest[ip] = {
                    "drone_name": row.get("drone_name", ""),
                    "drone_ip": ip,
                    "battery_id": normalize_battery_id(row.get("battery_id", "")),
                    "battery_percent": battery_percent,
                    "phase": row.get("phase", ""),
                    "timestamp": row.get("timestamp", ""),
                    "source": "live_csv",
                }
    except (OSError, csv.Error):
        return {}
    return latest


def build_live_battery_status(experiment_id, output_lines):
    experiment = experiment_by_id(experiment_id)
    with RUN_LOCK:
        baseline_config = dict(RUN_STATE.get("baseline_config") or {})
    if not experiment:
        if baseline_config:
            stdout_status = live_batteries_from_output(output_lines)
            ip = baseline_config.get("ip", "")
            battery_id = normalize_battery_id(baseline_config.get("battery_id", ""))
            drone_name = baseline_config.get("drone_name") or f"drone_{baseline_config.get('drone_number', '')}"
            status = {}
            if ip in stdout_status["by_ip"]:
                status.update(stdout_status["by_ip"][ip])
            if drone_name in stdout_status["by_name"]:
                status.update(stdout_status["by_name"][drone_name])
            if battery_id in stdout_status["by_battery"]:
                status.update(stdout_status["by_battery"][battery_id])
            percent = status.get("battery_percent")
            return {
                "window": EXPERIMENT_BATTERY_WINDOW,
                "recommended_batteries": sorted(RECOMMENDED_EXPERIMENT_BATTERIES),
                "drones": [{
                    "takeoff_order": 1,
                    "drone_number": baseline_config.get("drone_number", ""),
                    "drone_name": status.get("drone_name") or drone_name,
                    "drone_ip": ip,
                    "battery_id": status.get("battery_id") or battery_id,
                    "battery_percent": percent,
                    "battery_percent_label": f"{percent:g}%" if percent is not None else "--",
                    "band": battery_level_band(percent),
                    "window_progress": battery_window_progress(percent),
                    "phase": status.get("phase") or baseline_config.get("mode", ""),
                    "timestamp": status.get("timestamp", ""),
                    "source": status.get("source", "waiting"),
                    "recommended": battery_id in RECOMMENDED_EXPERIMENT_BATTERIES,
                }],
            }
        return {
            "window": EXPERIMENT_BATTERY_WINDOW,
            "recommended_batteries": sorted(RECOMMENDED_EXPERIMENT_BATTERIES),
            "drones": [],
        }

    stdout_status = live_batteries_from_output(output_lines)
    csv_status = latest_coordination_batteries(experiment_id)
    drones = []
    for drone in sorted(experiment.get("drones", []), key=lambda item: int(str(item.get("takeoff_order") or 999))):
        ip = str(drone.get("ip") or "").strip()
        if not ip:
            suffix = str(drone.get("ip_suffix") or "").strip()
            ip = f"{IP_PREFIX}{suffix}" if suffix else ""
        battery_id = display_battery_id(drone)
        drone_name = drone.get("role") or f"drone_{drone.get('takeoff_order', '')}"
        status = {}
        if ip in stdout_status["by_ip"]:
            status.update(stdout_status["by_ip"][ip])
        if drone_name in stdout_status["by_name"]:
            status.update(stdout_status["by_name"][drone_name])
        if battery_id in stdout_status["by_battery"]:
            status.update(stdout_status["by_battery"][battery_id])
        if ip in csv_status:
            status.update(csv_status[ip])
        percent = status.get("battery_percent")
        drones.append({
            "takeoff_order": drone.get("takeoff_order", ""),
            "drone_number": display_drone_number(drone),
            "drone_name": status.get("drone_name") or drone_name,
            "drone_ip": ip,
            "battery_id": status.get("battery_id") or battery_id,
            "battery_percent": percent,
            "battery_percent_label": f"{percent:g}%" if percent is not None else "--",
            "band": battery_level_band(percent),
            "window_progress": battery_window_progress(percent),
            "phase": status.get("phase", ""),
            "timestamp": status.get("timestamp", ""),
            "source": status.get("source", "waiting"),
            "recommended": battery_id in RECOMMENDED_EXPERIMENT_BATTERIES,
        })
    return {
        "window": EXPERIMENT_BATTERY_WINDOW,
        "recommended_batteries": sorted(RECOMMENDED_EXPERIMENT_BATTERIES),
        "drones": drones,
    }


def public_run_state():
    with RUN_LOCK:
        state = {
            key: value
            for key, value in RUN_STATE.items()
            if key != "process"
        }
        state["output"] = RUN_STATE["output"][-240:]
    state["live_batteries"] = build_live_battery_status(state.get("experiment_id"), state.get("output", []))
    return state


def set_run_state(**changes):
    with RUN_LOCK:
        RUN_STATE.update(changes)


def append_run_output(line):
    print(line, flush=True)
    with RUN_LOCK:
        RUN_STATE["output"].append(line)
        RUN_STATE["output"] = RUN_STATE["output"][-500:]
        if TAKEOFF_PROMPT in line:
            RUN_STATE["ready_for_takeoff"] = True
            RUN_STATE["ready_for_discharge"] = False
            RUN_STATE["prompt_action"] = "takeoff"
            RUN_STATE["status"] = "ready_for_takeoff"
            RUN_STATE["message"] = "Preflight checks reached takeoff confirmation."
        elif DISCHARGE_PROMPT in line:
            RUN_STATE["ready_for_takeoff"] = True
            RUN_STATE["ready_for_discharge"] = True
            RUN_STATE["prompt_action"] = "discharge"
            RUN_STATE["status"] = "ready_for_discharge"
            RUN_STATE["message"] = "High-battery drones need discharge hover before the experiment."


def reset_run_state():
    with RUN_LOCK:
        process = RUN_STATE.get("process")
        if process and process.poll() is None:
            process.terminate()
        RUN_STATE.update({
            "run_id": None,
            "experiment_id": None,
            "formation": None,
            "script": None,
            "status": "idle",
            "message": "",
            "ready_for_takeoff": False,
            "ready_for_discharge": False,
            "prompt_action": "",
            "takeoff_confirmed": False,
            "started_at": None,
            "ended_at": None,
            "returncode": None,
            "process": None,
            "output": [],
            "baseline_config": None,
        })


def monitor_experiment_process(process):
    if process.stdout:
        for line in process.stdout:
            append_run_output(line.rstrip())
    returncode = process.wait()
    ended_at = datetime.now().isoformat(timespec="seconds")
    with RUN_LOCK:
        RUN_STATE["returncode"] = returncode
        RUN_STATE["ended_at"] = ended_at
        if RUN_STATE["status"] not in {"stopped", "error"}:
            RUN_STATE["status"] = "finished" if returncode == 0 else "error"
        if returncode == 0:
            RUN_STATE["message"] = "Experiment process finished."
        else:
            RUN_STATE["message"] = f"Experiment process exited with code {returncode}."


def experiment_ip_order(experiment):
    drones = experiment.get("drones", [])
    if not drones:
        return ""
    sorted_drones = sorted(
        drones,
        key=lambda item: int(str(item.get("takeoff_order") or 999)),
    )
    suffixes = []
    for drone in sorted_drones:
        suffix = str(drone.get("ip_suffix") or drone.get("ip", "")).replace(IP_PREFIX, "").strip()
        if suffix:
            suffixes.append(suffix)
    return ",".join(suffixes)


def start_experiment_process(experiment):
    formation = experiment.get("formation", "")
    script_name = EXPERIMENT_SCRIPTS.get(formation)
    if not script_name:
        raise ValueError(f"No runnable script is configured for formation: {formation}")

    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Experiment script not found: {script_name}")

    ip_order = experiment_ip_order(experiment)
    if not ip_order:
        raise ValueError("No drone IPs are saved in this experiment record.")

    with RUN_LOCK:
        current_process = RUN_STATE.get("process")
        if current_process and current_process.poll() is None:
            raise RuntimeError(f"Another experiment is already {RUN_STATE['status']}.")

    run_id = uuid.uuid4().hex[:12]
    command = [python_executable(), "-u", str(script_path)]
    if script_name == "data_collector.py":
        command.extend(["--experiment-id", experiment["experiment_id"]])

    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if script_name != "data_collector.py" and process.stdin:
        process.stdin.write(ip_order + "\n")
        process.stdin.flush()

    set_run_state(
        run_id=run_id,
        experiment_id=experiment["experiment_id"],
        formation=formation,
        script=script_name,
        status="preflight",
        message=f"Started {script_name}; waiting for preflight and takeoff prompt.",
        ready_for_takeoff=False,
        ready_for_discharge=False,
        prompt_action="",
        takeoff_confirmed=False,
        started_at=datetime.now().isoformat(timespec="seconds"),
        ended_at=None,
        returncode=None,
        process=process,
        output=[f"$ {' '.join(command)}", f"IP order: {ip_order}"],
    )
    thread = threading.Thread(target=monitor_experiment_process, args=(process,), daemon=True)
    thread.start()
    return public_run_state()


def start_baseline_process(form):
    drone_number = str(form.get("baseline_drone_number", "")).strip()
    battery_id = normalize_battery_id(form.get("baseline_battery_id", ""))
    mode = str(form.get("baseline_mode", "")).strip()
    wind_level = str(form.get("baseline_wind_level", "")).strip()
    start_pad = str(form.get("baseline_start_pad", "")).strip()
    start_col = str(form.get("baseline_start_col", "")).strip()
    start_row = str(form.get("baseline_start_row", "")).strip()
    direction = str(form.get("baseline_direction", "up")).strip()
    wind_level = normalize_wind_level(form.get("baseline_wind_level", ""))
    notes = str(form.get("baseline_notes", "")).strip()

    if drone_number not in DRONE_NUMBER_TO_IP_SUFFIX:
        raise ValueError("Choose a valid drone number for the baseline test.")
    if not battery_id:
        raise ValueError("Choose a battery ID for the baseline test.")
    valid_modes = {key for key, _ in BASELINE_MODES}
    if mode not in valid_modes:
        raise ValueError("Choose a valid baseline mode.")
    valid_wind_levels = {key for key, _ in BASELINE_WIND_LEVELS}
    if wind_level not in valid_wind_levels:
        raise ValueError("Choose lv1 or lv2 for the baseline wind level.")
    if direction not in {"up", "down"}:
        raise ValueError("Choose a valid baseline direction.")

    script_name = "single_drone_baseline.py"
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Baseline script not found: {script_name}")

    with RUN_LOCK:
        current_process = RUN_STATE.get("process")
        if current_process and current_process.poll() is None:
            raise RuntimeError(f"Another experiment is already {RUN_STATE['status']}.")

    suffix = DRONE_NUMBER_TO_IP_SUFFIX[drone_number]
    ip = f"{IP_PREFIX}{suffix}"
    run_id = uuid.uuid4().hex[:12]
    command = [
        python_executable(),
        "-u",
        str(script_path),
        "--drone-number",
        drone_number,
        "--battery-id",
        battery_id,
        "--mode",
        mode,
        "--wind-level",
        wind_level,
    ]
    if start_pad:
        command.extend(["--start-pad", start_pad])
    if start_col:
        command.extend(["--start-col", start_col])
    if start_row:
        command.extend(["--start-row", start_row])
    command.extend(["--direction", direction])
    if wind_level:
        command.extend(["--wind-level", wind_level])
    if notes:
        command.extend(["--notes", notes])

    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    set_run_state(
        run_id=run_id,
        experiment_id=None,
        formation="single_baseline",
        script=script_name,
        status="preflight",
        message=f"Started {script_name}; waiting for single-drone preflight and takeoff prompt.",
        ready_for_takeoff=False,
        ready_for_discharge=False,
        prompt_action="",
        takeoff_confirmed=False,
        started_at=datetime.now().isoformat(timespec="seconds"),
        ended_at=None,
        returncode=None,
        process=process,
        baseline_config={
            "drone_number": drone_number,
            "drone_name": f"drone_{drone_number}",
            "ip": ip,
            "battery_id": battery_id,
            "mode": mode,
            "direction": direction,
            "wind_level": wind_level,
        },
        output=[f"$ {' '.join(command)}", f"Baseline: drone {drone_number} ({ip}) / {battery_id} / {mode}"],
    )
    thread = threading.Thread(target=monitor_experiment_process, args=(process,), daemon=True)
    thread.start()
    return public_run_state()


@app.route("/")
def index():
    files, all_categories = scan_workspace()
    registry = load_registry()
    all_experiments = sorted(
        registry["experiments"],
        key=lambda item: item.get("updated_at", item.get("created_at", "")),
        reverse=True,
    )
    selected_filters = {
        "formation": selected_values(request.args, "formation"),
        "wind_direction": selected_values(request.args, "wind_direction"),
        "wind_speed": selected_values(request.args, "wind_speed"),
    }
    all_baselines = scan_baseline_runs()
    selected_baseline_filters = {
        "drone_number": selected_values(request.args, "baseline_drone"),
        "battery_id": selected_values(request.args, "baseline_battery", normalize_battery_id),
        "mode": selected_values(request.args, "baseline_mode"),
        "wind_level": selected_values(request.args, "baseline_wind_level", normalize_wind_level),
    }
    baseline_runs = filter_baseline_runs(all_baselines, **selected_baseline_filters)
    baseline_summary_plot = (
        BASELINE_DIR / "summary" / "plots" /
        f"{baseline_summary_key(**selected_baseline_filters)}_summary.png"
    )
    baseline_summary_extra_plots = []
    for suffix in ("consumption_vs_start_battery", "mean_drop_by_drone"):
        extra_path = baseline_summary_plot.with_name(f"{baseline_summary_plot.stem}_{suffix}.png")
        if extra_path.exists():
            baseline_summary_extra_plots.append(extra_path.relative_to(DATA_DIR).as_posix())
    data_experiments = filter_experiments(all_experiments, **selected_filters)
    condition_experiments = filter_new_experiments(data_experiments)
    filter_active = True
    matched_ids = {exp.get("experiment_id") for exp in data_experiments}
    categories = all_categories
    stats = {
        "total_files": len(files),
        "coordination": len(categories["coordination"]),
        "battery": len(categories["battery"]),
        "images": len(categories["image"]),
        "experiments": len(all_experiments),
    }
    return render_template_string(
        INDEX_TEMPLATE,
        base_dir=BASE_DIR,
        data_dir=DATA_DIR,
        mission_pad_columns=MISSION_PAD_COLUMNS,
        mission_pad_layouts=MISSION_PAD_LAYOUTS,
        max_mission_cols=max(len(columns) for columns in MISSION_PAD_LAYOUTS.values()),
        max_mission_rows=max(len(column) for columns in MISSION_PAD_LAYOUTS.values() for column in columns),
        experiment_scripts=EXPERIMENT_SCRIPTS,
        run_state=public_run_state(),
        stats=stats,
        categories=categories,
        experiments=all_experiments,
        data_experiments=data_experiments,
        condition_groups=group_experiments_by_condition(condition_experiments),
        all_experiment_count=len(all_experiments),
        selected_filters=selected_filters,
        filter_options=experiment_filter_options(all_experiments),
        baseline_runs=baseline_runs,
        all_baseline_count=len(all_baselines),
        selected_baseline_filters=selected_baseline_filters,
        baseline_filter_options=baseline_filter_options(all_baselines),
        baseline_summary_plot=baseline_summary_plot.relative_to(DATA_DIR).as_posix() if baseline_summary_plot.exists() else "",
        baseline_summary_extra_plots=baseline_summary_extra_plots,
        display_drone_number=display_drone_number,
        display_battery_id=display_battery_id,
        battery_options=BATTERY_OPTIONS,
        battery_window=EXPERIMENT_BATTERY_WINDOW,
        wind_speed_options=WIND_SPEED_OPTIONS,
        inter_drone_distance_options_cm=INTER_DRONE_DISTANCE_OPTIONS_CM,
        baseline_modes=BASELINE_MODES,
        baseline_directions=BASELINE_DIRECTIONS,
        baseline_wind_levels=BASELINE_WIND_LEVELS,
        drone_options=sorted(DRONE_NUMBER_TO_IP_SUFFIX.items(), key=lambda item: int(item[0])),
    )


@app.route("/experiments", methods=["POST"])
def create_experiment():
    files, _ = scan_workspace()
    registry = load_registry()
    experiment = build_experiment_from_form(request.form)
    if find_experiment_index(registry["experiments"], experiment["experiment_id"]) is not None:
        abort(400, description=f"Experiment ID already exists: {experiment['experiment_id']}")
    validate_experiment_assignments(experiment)
    auto_attach_files(experiment, files)
    registry["experiments"].append(experiment)
    save_registry(registry)
    return redirect(url_for("index", selected=experiment["experiment_id"]))


@app.route("/experiments/<path:experiment_id>/update", methods=["POST"])
def update_experiment(experiment_id):
    files, _ = scan_workspace()
    registry = load_registry()
    experiments = registry["experiments"]
    idx = find_experiment_index(experiments, experiment_id)
    if idx is None:
        abort(404)

    new_id = build_full_experiment_id(
        request.form.get("experiment_id", experiment_id).strip(),
        request.form.get("formation", experiments[idx].get("formation", "")).strip(),
        request.form.get("wind_direction", experiments[idx].get("wind_direction", "")).strip(),
        request.form.get("wind_speed", experiments[idx].get("wind_speed", "")).strip(),
        request.form.get("inter_drone_distance_cm", experiments[idx].get("inter_drone_distance_cm", 50)).strip(),
    )
    duplicate_idx = find_experiment_index(experiments, new_id)
    if new_id and duplicate_idx is not None and duplicate_idx != idx:
        abort(400, description=f"Experiment ID already exists: {new_id}")

    experiment = update_experiment_from_form(experiments[idx], request.form)
    validate_experiment_assignments(experiment)
    auto_attach_files(experiment, files)
    experiments[idx] = experiment
    save_registry(registry)
    return redirect(url_for("index", selected=experiment["experiment_id"]))


@app.route("/experiments/<path:experiment_id>/delete", methods=["POST"])
def delete_experiment(experiment_id):
    registry = load_registry()
    experiments = registry["experiments"]
    idx = find_experiment_index(experiments, experiment_id)
    if idx is None:
        abort(404)
    del experiments[idx]
    save_registry(registry)
    return redirect(url_for("index"))


@app.route("/experiments/<path:experiment_id>/outlier", methods=["POST"])
def toggle_experiment_outlier(experiment_id):
    registry = load_registry()
    experiments = registry["experiments"]
    idx = find_experiment_index(experiments, experiment_id)
    if idx is None:
        abort(404)
    experiment = experiments[idx]
    experiment["is_outlier"] = request.form.get("is_outlier") == "1"
    experiment["outlier_note"] = request.form.get("outlier_note", experiment.get("outlier_note", "")).strip()
    experiment["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_registry(registry)
    return redirect(request.form.get("next") or url_for("experiment_detail", experiment_id=experiment_id))


@app.route("/experiments/<path:experiment_id>")
def experiment_detail(experiment_id):
    registry = load_registry()
    idx = find_experiment_index(registry["experiments"], experiment_id)
    if idx is None:
        abort(404)
    experiment = registry["experiments"][idx]
    archive = summarize_experiment_archive(experiment_id)
    selected_suffix = request.args.get("drone", "")
    drone_cards = []
    selected_drone = None
    for drone in experiment.get("drones", []):
        card = dict(drone)
        card.update(drone_archive_summary(experiment_id, drone))
        drone_cards.append(card)
        if selected_suffix and card["suffix"] == selected_suffix:
            selected_drone = card
    if selected_drone is None and drone_cards:
        selected_drone = drone_cards[0]
    mission_pad_columns = mission_pad_layout_for_formation(
        experiment.get("formation"),
        experiment.get("inter_drone_distance_cm"),
    )
    return render_template_string(
        EXPERIMENT_TEMPLATE,
        experiment=experiment,
        archive=archive,
        drone_cards=drone_cards,
        selected_drone=selected_drone,
        mission_pad_columns=mission_pad_columns,
        mission_pad_row_count=max(len(column) for column in mission_pad_columns),
        display_battery_id=display_battery_id,
    )


@app.route("/conditions/<path:condition_key>")
def condition_detail(condition_key):
    registry = load_registry()
    all_experiments = registry["experiments"]
    trials = [
        exp for exp in all_experiments
        if condition_key_from_experiment_id(exp.get("experiment_id")) == condition_key
    ]
    if not trials:
        abort(404)
    trials = sorted(trials, key=lambda exp: trial_number_from_experiment_id(exp.get("experiment_id")) or exp.get("experiment_id", ""))
    condition = {
        "condition_key": condition_key,
        "formation": trials[0].get("formation", ""),
        "wind_direction": trials[0].get("wind_direction", ""),
        "wind_speed": trials[0].get("wind_speed", ""),
        "trial_count": len(trials),
        "included_trial_count": sum(1 for trial in trials if not trial.get("is_outlier")),
        "outlier_count": sum(1 for trial in trials if trial.get("is_outlier")),
        "trials": trials,
    }
    archive = summarize_condition_archive(condition_key)
    return render_template_string(CONDITION_TEMPLATE, condition=condition, archive=archive)


@app.route("/plots/generate/<path:relpath>", methods=["POST"])
def generate_plots(relpath):
    experiment_id = experiment_id_from_relpath(relpath)
    if not experiment_id:
        abort(400, description="Could not infer experiment ID from file path.")
    script_path = BASE_DIR / "plot_generate.py"
    result = subprocess.run(
        [python_executable(), str(script_path), "--experiment-id", experiment_id],
        cwd=str(BASE_DIR),
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        abort(500, description=result.stderr or result.stdout or "Plot generation failed.")
    return redirect(url_for("experiment_detail", experiment_id=experiment_id))


@app.route("/plots/generate-condition/<path:condition_key>", methods=["POST"])
def generate_condition_plots(condition_key):
    script_path = BASE_DIR / "plot_generate.py"
    result = subprocess.run(
        [python_executable(), str(script_path), "--condition-key", condition_key],
        cwd=str(BASE_DIR),
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        abort(500, description=result.stderr or result.stdout or "Condition plot generation failed.")
    return redirect(url_for("condition_detail", condition_key=condition_key))


@app.route("/plots/generate-filtered-conditions", methods=["POST"])
def generate_filtered_condition_plots():
    registry = load_registry()
    selected_filters = {
        "formation": selected_values(request.form, "formation"),
        "wind_direction": selected_values(request.form, "wind_direction"),
        "wind_speed": selected_values(request.form, "wind_speed"),
    }
    experiments = filter_new_experiments(filter_experiments(registry["experiments"], **selected_filters))
    groups = group_experiments_by_condition(experiments)
    if not groups:
        abort(400, description="No new condition groups match the selected filters.")

    script_path = BASE_DIR / "plot_generate.py"
    errors = []
    for group in groups:
        result = subprocess.run(
            [python_executable(), str(script_path), "--condition-key", group["condition_key"]],
            cwd=str(BASE_DIR),
            text=True,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            errors.append(f"{group['condition_key']}: {result.stderr or result.stdout or 'failed'}")
    if errors:
        abort(500, description="\n".join(errors))

    query = urlencode(selected_filters, doseq=True)
    return redirect(url_for("index") + (f"?{query}" if query else ""))


def baseline_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def baseline_time_value(row):
    for column in ("node_elapsed_time", "elapsed_time", "hover_elapsed_time"):
        value = baseline_float(row.get(column))
        if value is not None:
            return value
    return None


BASELINE_WIND_LEVEL_COLORS = {
    "lv1": "#2f80a8",
    "lv2": "#c95b4d",
    "unknown": "#8b98a5",
}


def normalize_baseline_wind_level(value):
    return normalize_wind_level(value) or str(value or "").strip()


def baseline_wind_level(run=None, notes=""):
    run = run or {}
    level = normalize_baseline_wind_level(run.get("wind_level"))
    if not level:
        level = normalize_baseline_wind_level(run.get("baseline_wind_level"))
    if not level:
        level = normalize_baseline_wind_level(run.get("wind_speed"))
    if not level:
        level = wind_level_from_notes(notes or run.get("notes", ""))
    return level or "unknown"


def baseline_wind_color(run):
    return BASELINE_WIND_LEVEL_COLORS.get(baseline_wind_level(run), BASELINE_WIND_LEVEL_COLORS["unknown"])


def baseline_run_label(run):
    label_parts = [f"D{run.get('drone_number')}", run.get("battery_id"), run.get("mode"), run.get("run_id")]
    return " ".join(str(part) for part in label_parts if part)


def read_baseline_timeseries(run):
    if not run.get("timeseries_relpath"):
        return []
    path = DATA_DIR / run["timeseries_relpath"]
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def baseline_battery_points(rows):
    points = []
    for index, row in enumerate(rows):
        battery = baseline_float(row.get("battery"))
        if battery is None:
            continue
        x_value = baseline_time_value(row)
        if x_value is None:
            x_value = float(index)
        points.append((x_value, battery))
    points.sort(key=lambda item: item[0])
    return points


def baseline_numeric_summary(run, rows):
    points = baseline_battery_points(rows)
    start = baseline_float(run.get("battery_start"))
    end = baseline_float(run.get("battery_end"))
    drop = baseline_float(run.get("battery_drop"))
    if start is None and points:
        start = points[0][1]
    if end is None and points:
        end = points[-1][1]
    if drop is None and start is not None and end is not None:
        drop = start - end
    return start, end, drop, points


def baseline_summary_records(runs):
    records = []
    for run in runs:
        rows = read_baseline_timeseries(run)
        start, end, drop, points = baseline_numeric_summary(run, rows)
        if start is None and end is None and drop is None and not points:
            continue
        records.append({
            "run": run,
            "label": baseline_run_label(run),
            "group": f"D{run.get('drone_number') or '-'} / {run.get('battery_id') or '-'}",
            "start": start,
            "end": end,
            "drop": drop,
            "points": points,
        })
    return records


def generate_baseline_consumption_trend_plot(records, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    numeric_records = [
        record for record in records
        if record["start"] is not None and record["end"] is not None and record["drop"] is not None
    ]
    if not numeric_records:
        return None

    numeric_records.sort(key=lambda record: (
        record["group"],
        record["start"] if record["start"] is not None else 999,
        record["run"].get("run_id", ""),
    ), reverse=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    ax_scatter, ax_sequence = axes

    groups = sorted({record["group"] for record in numeric_records})
    for group in groups:
        group_records = [record for record in numeric_records if record["group"] == group]
        starts = [record["start"] for record in group_records]
        drops = [record["drop"] for record in group_records]
        colors = [baseline_wind_color(record["run"]) for record in group_records]
        ax_scatter.scatter(starts, drops, s=55, color=colors, alpha=0.88)
        if len(group_records) >= 2:
            ordered = sorted(group_records, key=lambda record: record["start"], reverse=True)
            ax_scatter.plot(
                [record["start"] for record in ordered],
                [record["drop"] for record in ordered],
                color="#9fb7c9",
                alpha=0.35,
                linewidth=1.2,
            )
    ax_scatter.invert_xaxis()
    ax_scatter.set_title("Does each identical flight consume more as start battery gets lower?")
    ax_scatter.set_xlabel("Start battery before flight (%)")
    ax_scatter.set_ylabel("Battery drop for one forward flight (%)")
    ax_scatter.grid(True, alpha=0.25)
    wind_levels = sorted({baseline_wind_level(record["run"]) for record in numeric_records})
    wind_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BASELINE_WIND_LEVEL_COLORS.get(level, BASELINE_WIND_LEVEL_COLORS["unknown"]), markersize=7, label=level)
        for level in wind_levels
    ]
    ax_scatter.legend(handles=wind_handles, title="Wind level", fontsize=8, title_fontsize=8, loc="best")

    x_positions = list(range(len(numeric_records)))
    labels = [record["label"] for record in numeric_records]
    starts = [record["start"] for record in numeric_records]
    ends = [record["end"] for record in numeric_records]
    drops = [record["drop"] for record in numeric_records]
    range_colors = [baseline_wind_color(record["run"]) for record in numeric_records]
    ax_sequence.vlines(x_positions, ends, starts, color=range_colors, linewidth=5, alpha=0.65, label="start to end range")
    ax_sequence.scatter(x_positions, starts, color=range_colors, s=42, marker="^", label="start")
    ax_sequence.scatter(x_positions, ends, color=range_colors, s=42, marker="v", label="end")
    ax_sequence_twin = ax_sequence.twinx()
    ax_sequence_twin.plot(x_positions, drops, color="#216c5f", marker="o", linewidth=1.8, label="drop")
    ax_sequence.set_title("Repeated flights: start/end battery and consumed battery")
    ax_sequence.set_ylabel("Battery level (%)")
    ax_sequence_twin.set_ylabel("Battery drop (%)")
    ax_sequence.set_xticks(x_positions)
    ax_sequence.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax_sequence.grid(True, axis="y", alpha=0.25)
    lines_a, labels_a = ax_sequence.get_legend_handles_labels()
    lines_b, labels_b = ax_sequence_twin.get_legend_handles_labels()
    ax_sequence.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def generate_baseline_mean_drop_by_drone_plot(records, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for record in records:
        drop = record.get("drop")
        if drop is None:
            continue
        run = record["run"]
        drone_number = str(run.get("drone_number") or "").strip()
        if not drone_number:
            continue
        grouped[(drone_number, baseline_wind_level(run))].append(drop)
    if not grouped:
        return None

    def mean(values):
        return sum(values) / len(values)

    def sample_std(values):
        if len(values) < 2:
            return 0
        avg = mean(values)
        return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5

    drone_numbers = sorted({key[0] for key in grouped}, key=lambda value: int(value) if value.isdigit() else 999)
    wind_levels = sorted({key[1] for key in grouped})
    x_positions = list(range(len(drone_numbers)))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if len(wind_levels) <= 1:
        wind_level = wind_levels[0]
        means = [mean(grouped.get((drone, wind_level), [0])) for drone in drone_numbers]
        stds = [sample_std(grouped.get((drone, wind_level), [])) for drone in drone_numbers]
        color = BASELINE_WIND_LEVEL_COLORS.get(wind_level, BASELINE_WIND_LEVEL_COLORS["unknown"])
        ax.bar(x_positions, means, yerr=stds, capsize=4, color=color, alpha=0.9, label=wind_level)
    else:
        width = min(0.34, 0.8 / len(wind_levels))
        center_offset = (len(wind_levels) - 1) * width / 2
        for index, wind_level in enumerate(wind_levels):
            offsets = [position - center_offset + index * width for position in x_positions]
            means = []
            stds = []
            for drone in drone_numbers:
                values = grouped.get((drone, wind_level), [])
                means.append(mean(values) if values else 0)
                stds.append(sample_std(values))
            color = BASELINE_WIND_LEVEL_COLORS.get(wind_level, BASELINE_WIND_LEVEL_COLORS["unknown"])
            ax.bar(offsets, means, width=width, yerr=stds, capsize=4, color=color, alpha=0.9, label=wind_level)

    ax.set_title("Mean battery drop by drone")
    ax.set_ylabel("Mean battery drop (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"drone_{drone}" for drone in drone_numbers], rotation=25, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.28)
    if wind_levels:
        ax.legend(title="Wind level", fontsize=8, title_fontsize=8, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def generate_baseline_summary_plot(runs, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_runs = [run for run in runs if run.get("timeseries_relpath")]
    if not plot_runs:
        raise ValueError("No baseline runs with time-series data matched this filter.")
    records = baseline_summary_records(plot_runs)

    fig, axes = plt.subplots(4, 1, figsize=(11, 15), sharex=False)
    ax_battery, ax_temp, ax_drop, ax_error = axes
    drop_labels = []
    drop_values = []
    duration_values = []
    error_plotted = False
    modes = sorted({run.get("mode", "") for run in plot_runs if run.get("mode")})
    wind_levels = sorted({baseline_wind_level(run) for run in plot_runs})

    for record in records:
        run = record["run"]
        rows = read_baseline_timeseries(run)
        points = []
        temp_low = []
        temp_high = []
        errors = []
        for row in rows:
            elapsed = baseline_time_value(row)
            battery = baseline_float(row.get("battery"))
            templ = baseline_float(row.get("templ"))
            temph = baseline_float(row.get("temph"))
            position_error = baseline_float(row.get("position_error_dist"))
            if elapsed is not None and battery is not None:
                points.append((elapsed, battery))
            if elapsed is not None and templ is not None:
                temp_low.append((elapsed, templ))
            if elapsed is not None and temph is not None:
                temp_high.append((elapsed, temph))
            if elapsed is not None and position_error is not None:
                errors.append((elapsed, position_error))
        label = record["label"]
        color = baseline_wind_color(run)
        if points:
            ax_battery.plot([p[0] for p in points], [p[1] for p in points], color=color, alpha=0.78, linewidth=1.8, label=label)
        if temp_low:
            ax_temp.plot([p[0] for p in temp_low], [p[1] for p in temp_low], color=color, alpha=0.72, linewidth=1.3, label=f"{label} templ")
        if temp_high:
            ax_temp.plot([p[0] for p in temp_high], [p[1] for p in temp_high], color=color, alpha=0.72, linewidth=1.3, linestyle="--", label=f"{label} temph")
        if errors:
            ax_error.plot([p[0] for p in errors], [p[1] for p in errors], color=color, alpha=0.78, linewidth=1.5, label=label)
            error_plotted = True
        drop = baseline_float(run.get("battery_drop"))
        duration = baseline_float(run.get("duration_sec"))
        if drop is not None:
            drop_labels.append(label)
            drop_values.append(drop)
            duration_values.append(duration or 0)

    title_mode = ", ".join(modes) if modes else "matched modes"
    wind_title = ", ".join(wind_levels) if wind_levels else "matched wind levels"
    wind_handles = [
        Line2D([0], [0], color=BASELINE_WIND_LEVEL_COLORS.get(level, BASELINE_WIND_LEVEL_COLORS["unknown"]), linewidth=3, label=level)
        for level in wind_levels
    ]
    if modes == ["hover"]:
        ax_battery.axhline(10, color="#a23b3b", linestyle="--", linewidth=1, label="10% landing threshold")
    ax_battery.set_title(f"Baseline battery percentage: {title_mode} / {wind_title}")
    ax_battery.set_xlabel("Flight time (s)")
    ax_battery.set_ylabel("Battery (%)")
    ax_battery.grid(True, alpha=0.25)
    if wind_handles:
        ax_battery.legend(handles=wind_handles, title="Wind level", fontsize=7, title_fontsize=7, loc="best")

    ax_temp.set_title("Baseline temperature")
    ax_temp.set_xlabel("Flight time (s)")
    ax_temp.set_ylabel("Temperature (C)")
    ax_temp.grid(True, alpha=0.25)
    if wind_handles:
        ax_temp.legend(handles=wind_handles, title="Wind level", fontsize=7, title_fontsize=7, loc="best")

    if drop_values:
        x_positions = list(range(len(drop_values)))
        bar_colors = [baseline_wind_color(record["run"]) for record in records if baseline_float(record["run"].get("battery_drop")) is not None]
        ax_drop.bar(x_positions, drop_values, color=bar_colors, alpha=0.85, label="battery drop (%)")
        ax_drop.set_xticks(x_positions)
        ax_drop.set_xticklabels(drop_labels, rotation=35, ha="right", fontsize=8)
        ax_drop.set_ylabel("Battery drop (%)")
        ax_drop.set_title("Baseline total battery drop and duration")
        ax_duration = ax_drop.twinx()
        ax_duration.plot(x_positions, duration_values, color="#3d4b59", marker="o", linewidth=1.8, label="duration (s)")
        ax_duration.set_ylabel("Duration (s)")
        ax_drop.grid(True, axis="y", alpha=0.25)
        if wind_handles:
            ax_drop.legend(handles=wind_handles, title="Wind level", fontsize=7, title_fontsize=7, loc="best")

    ax_error.set_title("Baseline position error")
    ax_error.set_xlabel("Flight time (s)")
    ax_error.set_ylabel("Position error (cm)")
    ax_error.grid(True, alpha=0.25)
    if error_plotted:
        if wind_handles:
            ax_error.legend(handles=wind_handles, title="Wind level", fontsize=7, title_fontsize=7, loc="best")
    else:
        ax_error.text(0.5, 0.5, "No position-error series available for these runs.", ha="center", va="center", transform=ax_error.transAxes, color="#657286")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    generate_baseline_consumption_trend_plot(
        records,
        output_path.with_name(f"{output_path.stem}_consumption_vs_start_battery.png"),
    )
    generate_baseline_mean_drop_by_drone_plot(
        records,
        output_path.with_name(f"{output_path.stem}_mean_drop_by_drone.png"),
    )


@app.route("/baselines/generate-summary", methods=["POST"])
def generate_baseline_summary():
    drone_number = selected_values(request.form, "baseline_drone")
    battery_id = selected_values(request.form, "baseline_battery", normalize_battery_id)
    mode = selected_values(request.form, "baseline_mode") or ["hover"]
    wind_level = selected_values(request.form, "baseline_wind_level", normalize_wind_level)
    runs = filter_baseline_runs(
        scan_baseline_runs(),
        drone_number=drone_number,
        battery_id=battery_id,
        mode=mode,
        wind_level=wind_level,
    )
    key = baseline_summary_key(drone_number, battery_id, mode, wind_level)
    output_path = BASELINE_DIR / "summary" / "plots" / f"{key}_summary.png"
    try:
        generate_baseline_summary_plot(runs, output_path)
    except ValueError as exc:
        abort(400, description=str(exc))
    query = urlencode({
        "baseline_drone": drone_number,
        "baseline_battery": battery_id,
        "baseline_mode": mode,
        "baseline_wind_level": wind_level,
    }, doseq=True)
    return redirect(url_for("index") + (f"?{query}" if query else ""))


@app.route("/experiments/<path:experiment_id>/start", methods=["POST"])
def start_experiment(experiment_id):
    registry = load_registry()
    idx = find_experiment_index(registry["experiments"], experiment_id)
    if idx is None:
        abort(404)
    try:
        state = start_experiment_process(registry["experiments"][idx])
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        return jsonify({"ok": False, "error": str(exc), "state": public_run_state()}), 400
    return jsonify({"ok": True, "state": state})


@app.route("/baselines/start", methods=["POST"])
def start_baseline():
    try:
        state = start_baseline_process(request.form)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        return jsonify({"ok": False, "error": str(exc), "state": public_run_state()}), 400
    return jsonify({"ok": True, "state": state})


@app.route("/experiment-run/status")
def experiment_run_status():
    return jsonify({"ok": True, "state": public_run_state()})


@app.route("/experiment-run/takeoff", methods=["POST"])
def confirm_takeoff():
    with RUN_LOCK:
        process = RUN_STATE.get("process")
        ready_for_takeoff = RUN_STATE.get("ready_for_takeoff")
        status = RUN_STATE.get("status")
        prompt_action = RUN_STATE.get("prompt_action")
    if not process or process.poll() is not None:
        return jsonify({"ok": False, "error": "No active experiment process.", "state": public_run_state()}), 400
    if not ready_for_takeoff:
        return jsonify({"ok": False, "error": f"Experiment is not ready for takeoff yet. Current status: {status}", "state": public_run_state()}), 400
    try:
        if process.stdin:
            process.stdin.write("\n")
            process.stdin.flush()
    except BrokenPipeError:
        return jsonify({"ok": False, "error": "Experiment process is no longer accepting input.", "state": public_run_state()}), 400
    is_discharge = prompt_action == "discharge"
    set_run_state(
        status="discharging" if is_discharge else "running",
        message="Battery discharge hover confirmed from GUI." if is_discharge else "Takeoff confirmed from GUI.",
        takeoff_confirmed=not is_discharge,
        ready_for_takeoff=False,
        ready_for_discharge=False,
        prompt_action="",
    )
    append_run_output("[GUI] Battery discharge hover confirmed." if is_discharge else "[GUI] Takeoff confirmed.")
    return jsonify({"ok": True, "state": public_run_state()})


@app.route("/experiment-run/skip-discharge", methods=["POST"])
def skip_discharge_hover():
    with RUN_LOCK:
        process = RUN_STATE.get("process")
        ready_for_discharge = RUN_STATE.get("ready_for_discharge")
        prompt_action = RUN_STATE.get("prompt_action")
    if not process or process.poll() is not None:
        return jsonify({"ok": False, "error": "No active experiment process.", "state": public_run_state()}), 400
    if not ready_for_discharge and prompt_action != "discharge":
        return jsonify({"ok": False, "error": "Experiment is not waiting for battery discharge confirmation.", "state": public_run_state()}), 400
    try:
        if process.stdin:
            process.stdin.write("skip\n")
            process.stdin.flush()
    except BrokenPipeError:
        return jsonify({"ok": False, "error": "Experiment process is no longer accepting input.", "state": public_run_state()}), 400
    set_run_state(
        status="preflight",
        message="Battery discharge hover skipped. Waiting for formal takeoff prompt.",
        ready_for_takeoff=False,
        ready_for_discharge=False,
        prompt_action="",
    )
    append_run_output("[GUI] Battery discharge hover skipped.")
    return jsonify({"ok": True, "state": public_run_state()})


@app.route("/experiment-run/stop", methods=["POST"])
def stop_experiment_run():
    with RUN_LOCK:
        process = RUN_STATE.get("process")
    if process and process.poll() is None:
        process.terminate()
        set_run_state(
            status="stopped",
            message="Experiment process was stopped from GUI.",
            ended_at=datetime.now().isoformat(timespec="seconds"),
        )
        append_run_output("[GUI] Experiment process stopped.")
    return jsonify({"ok": True, "state": public_run_state()})


@app.route("/file/<path:relpath>")
def file_detail(relpath):
    path = safe_relative_path(relpath)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        header, rows = csv_preview(path)
        analysis = analyze_csv(path)
        return render_template_string(
            FILE_TEMPLATE,
            path=path,
            relpath=relpath,
            file_type="csv",
            header=header,
            rows=rows,
            analysis=analysis,
            mime=None,
        )
    if suffix in IMAGE_EXTENSIONS:
        return render_template_string(
            FILE_TEMPLATE,
            path=path,
            relpath=relpath,
            file_type="image",
            header=[],
            rows=[],
            analysis={},
            mime=mimetypes.guess_type(path.name)[0],
        )
    return send_file(path, as_attachment=False)


@app.route("/raw/<path:relpath>")
def raw_file(relpath):
    path = safe_relative_path(relpath)
    return send_file(path, as_attachment=False)


@app.route("/files/delete/<path:relpath>", methods=["POST"])
def delete_file(relpath):
    path = safe_relative_path(relpath)
    try:
        path.unlink()
    except OSError as exc:
        abort(500, description=f"Could not delete file: {exc}")
    remove_empty_parent_dirs(path)
    remove_file_references(relpath)
    return redirect(request.form.get("next") or request.referrer or url_for("index"))


INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tello Experiment Lab</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #1d2430;
      --muted: #657286;
      --brand: #216c5f;
      --brand-2: #2f80a8;
      --warn: #9a5b18;
      --soft: #edf4f2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: #fff;
      padding: 18px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 22px; }
    h2 { font-size: 16px; }
    h3 { font-size: 14px; }
    .path { color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
    main {
      display: grid;
      grid-template-columns: minmax(340px, 0.9fr) minmax(420px, 1.1fr);
      gap: 18px;
      padding: 18px;
      max-width: 1500px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
    }
    .section-head {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .section-body { padding: 16px 18px; }
    .stats {
      display: grid;
      grid-template-columns: repeat(5, minmax(90px, 1fr));
      gap: 10px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .stat strong { display: block; font-size: 20px; }
    .stat span { color: var(--muted); font-size: 12px; }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .tab {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      color: var(--text);
      cursor: pointer;
    }
    .tab.active {
      background: var(--soft);
      border-color: #9cc7bd;
      color: var(--brand);
      font-weight: 650;
    }
    .file-list, .experiment-list {
      display: grid;
      gap: 10px;
      max-height: 68vh;
      overflow: auto;
      padding-right: 4px;
    }
    .file-list[hidden] { display: none; }
    .file-row, .experiment-row {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .file-row[hidden] { display: none; }
    .file-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }
    .file-name {
      color: var(--text);
      font-weight: 650;
      text-decoration: none;
      overflow-wrap: anywhere;
    }
    .file-meta, .small {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
      background: #fbfcfd;
    }
    .badge.coordination { color: var(--brand-2); border-color: #b7d7e4; }
    .badge.outlier { border-color:#d49a90; color:#9d3d31; background:#fff4f1; }
    .badge.battery { color: var(--warn); border-color: #e3c49e; }
    .badge.image { color: var(--brand); border-color: #a9d2c7; }
    details { margin-top: 10px; }
    summary { cursor: pointer; color: var(--muted); font-size: 13px; }
    .columns {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    form {
      display: grid;
      gap: 12px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      font: inherit;
      color: var(--text);
      background: #fff;
    }
    textarea { min-height: 72px; resize: vertical; }
    .formation-toolbar {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
    }
    .wind-toolbar {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .formation-option,
    .wind-option {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 8px;
      padding: 9px 8px;
      font-size: 13px;
    }
    .formation-option.active,
    .wind-option.active {
      background: var(--soft);
      border-color: #9cc7bd;
      color: var(--brand);
    }
    .formation-option.pending {
      color: var(--muted);
      background: #f7f8fa;
    }
    .mode-switch {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }
    .mode-switch button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
    }
    .mode-switch button.active {
      background: var(--soft);
      border-color: #9cc7bd;
      color: var(--brand);
    }
    .mode-panel[hidden] { display: none; }
    .mission-board {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
    }
    .mission-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(74px, 1fr));
      gap: 8px;
    }
    .mission-grid.column-layout {
      grid-template-columns: minmax(120px, 180px) !important;
      justify-content: center;
    }
    .pad-cell {
      min-height: 84px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      display: grid;
      grid-template-rows: auto 1fr 1fr;
      gap: 7px;
      opacity: 0.36;
    }
    .pad-cell[hidden] { display: none !important; }
    .pad-cell.active {
      opacity: 1;
      border-color: #8abeb2;
      background: #f3faf8;
      box-shadow: inset 0 0 0 1px #c4dfd8;
    }
    .pad-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .pad-id {
      color: var(--text);
      font-size: 18px;
      font-weight: 750;
    }
    .pad-cell input[type="text"],
    .pad-cell select {
      align-self: end;
      text-align: center;
      padding: 8px 6px;
      font-size: 14px;
      font-weight: 650;
    }
    .pad-cell:not(.active) input[type="text"],
    .pad-cell:not(.active) select {
      visibility: hidden;
    }
    .baseline-pad-cell {
      min-height: 74px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      display: grid;
      align-content: space-between;
      gap: 6px;
      opacity: 0.72;
      cursor: pointer;
      color: var(--muted);
    }
    .baseline-pad-cell.selected {
      opacity: 1;
      border-color: #8abeb2;
      background: #f3faf8;
      box-shadow: inset 0 0 0 1px #c4dfd8;
      color: var(--text);
    }
    .baseline-pad-cell.in-path {
      opacity: 1;
      border-color: #b7d7cd;
      background: #f8fcfb;
    }
    .baseline-pad-cell .path-index {
      min-height: 18px;
      font-size: 12px;
      color: var(--brand);
      font-weight: 700;
    }
    .arrow-toolbar {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .arrow-option {
      min-height: 48px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      font-size: 22px;
      line-height: 1;
    }
    .arrow-option.active {
      background: var(--soft);
      border-color: #9cc7bd;
      color: var(--brand);
    }
    .board-note {
      margin-top: 10px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .drone-grid {
      display: grid;
      grid-template-columns: 1fr 0.55fr 0.7fr;
      gap: 8px;
    }
    .drone-grid .head {
      color: var(--muted);
      font-size: 12px;
      padding: 0 2px;
    }
    button {
      border: 0;
      border-radius: 8px;
      background: var(--brand);
      color: white;
      padding: 10px 14px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button.secondary {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
    }
    button.danger {
      background: #a23b3b;
    }
    .icon-delete {
      width: 28px;
      height: 28px;
      border-radius: 999px;
      padding: 0;
      display: inline-grid;
      place-items: center;
      background: #fff;
      color: #a23b3b;
      border: 1px solid #d8aaa7;
      font-size: 16px;
      line-height: 1;
    }
    .experiment-row {
      display: grid;
      gap: 10px;
    }
    .experiment-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
    }
    .record-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .record-actions form {
      display: block;
    }
    .edit-panel {
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }
    .edit-panel summary {
      margin-bottom: 10px;
    }
    .edit-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .edit-drone-grid {
      display: grid;
      grid-template-columns: 0.8fr 0.8fr 0.7fr 1fr 0.7fr 0.7fr 0.7fr;
      gap: 8px;
      align-items: end;
    }
    .edit-drone-grid .head {
      color: var(--muted);
      font-size: 12px;
      padding: 0 2px;
    }
    .run-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
      display: grid;
      gap: 10px;
    }
    .filter-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
      display: grid;
      gap: 10px;
    }
    .filter-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .filter-group {
      display: grid;
      gap: 8px;
      align-content: start;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .filter-group-title {
      color: var(--text);
      font-size: 13px;
      font-weight: 700;
    }
    .filter-group summary {
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 11px;
      cursor: pointer;
      background: #fff;
    }
    .filter-group summary::-webkit-details-marker { display: none; }
    .filter-group summary::after {
      content: "+";
      display: inline-grid;
      place-items: center;
      inline-size: 20px;
      block-size: 20px;
      border: 1px solid var(--line);
      border-radius: 50%;
      color: var(--muted);
      font-weight: 700;
      flex: 0 0 auto;
    }
    .filter-group[open] summary {
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    .filter-group[open] summary::after { content: "-"; }
    .filter-summary {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .check-list {
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
      padding: 9px;
    }
    .check-pill {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #fff;
      font-size: 13px;
      cursor: pointer;
      user-select: none;
      min-height: 36px;
    }
    .check-pill:has(input:checked) {
      background: var(--soft);
      border-color: #9cc7bd;
      color: var(--brand);
      font-weight: 700;
    }
    .check-pill input {
      margin: 0;
      accent-color: var(--brand);
      inline-size: 16px;
      block-size: 16px;
      flex: 0 0 auto;
      order: 2;
    }
    .check-text {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .baseline-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .experiment-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .experiment-tags a {
      color: var(--brand);
      border: 1px solid #b7d7cd;
      background: #f3faf8;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      text-decoration: none;
    }
    .run-status {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .battery-monitor {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }
    .battery-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 9px;
      display: grid;
      gap: 7px;
      min-width: 0;
    }
    .battery-card .battery-top {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
    }
    .battery-card strong {
      font-size: 20px;
      line-height: 1;
      color: var(--text);
    }
    .battery-track {
      height: 10px;
      border-radius: 999px;
      background: #e8edf0;
      overflow: hidden;
      border: 1px solid #d8e0e4;
    }
    .battery-fill {
      height: 100%;
      width: 0%;
      background: #2f8f72;
      transition: width 0.25s ease, background 0.25s ease;
    }
    .battery-card[data-band="above"] .battery-fill { background: #2f80a8; }
    .battery-card[data-band="below"] .battery-fill { background: #c95b4d; }
    .battery-card[data-band="unknown"] .battery-fill { background: #a3adb5; }
    .battery-card[data-band="below"] { border-color: #e0aaa4; background: #fff7f5; }
    .battery-window-note {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      margin-top: 2px;
    }
    .terminal {
      min-height: 220px;
      max-height: 360px;
      overflow: auto;
      border: 1px solid #2d343f;
      border-radius: 0 0 8px 8px;
      background: #10151c;
      color: #dbe7ef;
      padding: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .terminal-shell {
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #2d343f;
      background: #10151c;
    }
    .terminal-titlebar {
      height: 34px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 0 10px;
      background: #202833;
      color: #c8d3dc;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      border-bottom: 1px solid #2d343f;
    }
    .terminal-dots {
      display: flex;
      gap: 6px;
      align-items: center;
      flex: 0 0 auto;
    }
    .terminal-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #66717f;
    }
    .terminal-dot.red { background: #d76060; }
    .terminal-dot.yellow { background: #d7aa4d; }
    .terminal-dot.green { background: #5abf79; }
    .terminal-title {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
      text-align: center;
    }
    .terminal-meta {
      color: #91a0ae;
      flex: 0 0 auto;
    }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(13, 19, 26, 0.48);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      z-index: 20;
    }
    .modal-backdrop.visible { display: flex; }
    .modal {
      width: min(440px, 100%);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 22px 60px rgba(16, 21, 28, 0.22);
      display: grid;
      gap: 12px;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, 1fr); }
      .formation-toolbar { grid-template-columns: repeat(2, 1fr); }
      .mission-grid { grid-template-columns: repeat(5, minmax(58px, 1fr)); }
      .edit-grid { grid-template-columns: 1fr; }
      .edit-drone-grid { grid-template-columns: 1fr 1fr; }
      .battery-monitor { grid-template-columns: 1fr; }
      .filter-grid { grid-template-columns: 1fr; }
      .baseline-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Tello Experiment Lab</h1>
      <div class="path">App: {{ base_dir }} · Data: {{ data_dir }}</div>
    </div>
    <div class="badge">Local GUI prototype</div>
  </header>

  <main>
    <section>
      <div class="section-head">
        <h2>Data Workspace</h2>
        <span class="small">CSV data and generated images from database/</span>
      </div>
      <div class="stats">
        <div class="stat"><strong>{{ stats.total_files }}</strong><span>files</span></div>
        <div class="stat"><strong>{{ stats.coordination }}</strong><span>coordination</span></div>
        <div class="stat"><strong>{{ stats.battery }}</strong><span>battery</span></div>
        <div class="stat"><strong>{{ stats.images }}</strong><span>images</span></div>
        <div class="stat"><strong>{{ stats.experiments }}</strong><span>experiments</span></div>
      </div>
      <div class="section-body">
        <div class="tabs" role="tablist">
          <button class="tab" type="button" data-filter="all">All</button>
          <button class="tab" type="button" data-filter="coordination">Coordination</button>
          <button class="tab" type="button" data-filter="battery">Battery</button>
          <button class="tab" type="button" data-filter="image">Images</button>
          <button class="tab" type="button" data-filter="script">Scripts</button>
        </div>
        <div class="filter-panel" style="margin-bottom:14px;">
          <div class="run-status">
            <div>
              <h3>Data Filters</h3>
              <div class="small">
                Showing {{ data_experiments|length }} matched experiment(s).
                Leave a list unselected to include all values for that condition.
              </div>
            </div>
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end;">
              <button class="secondary" type="button" data-toggle-filter-groups>Expand all</button>
              <a class="badge" href="{{ url_for('index') }}">Reset</a>
            </div>
          </div>
          <form method="get" action="{{ url_for('index') }}">
            <div class="filter-grid">
              <details class="filter-group" data-filter-group>
                <summary>
                  <span class="filter-group-title">Formation</span>
                  <span class="filter-summary">{% if selected_filters.formation %}{{ selected_filters.formation|join(', ') }}{% else %}All{% endif %}</span>
                </summary>
                <div class="check-list">
                  <label class="check-pill">
                    <input type="checkbox" data-filter-all {% if not selected_filters.formation %}checked{% endif %}>
                    <span class="check-text">All formations</span>
                  </label>
                  {% for formation in filter_options.formations %}
                    <label class="check-pill">
                      <input type="checkbox" name="formation" value="{{ formation }}" data-filter-option {% if formation in selected_filters.formation %}checked{% endif %}>
                      <span class="check-text">{{ formation }}</span>
                    </label>
                  {% endfor %}
                </div>
              </details>
              <details class="filter-group" data-filter-group>
                <summary>
                  <span class="filter-group-title">Wind Direction</span>
                  <span class="filter-summary">{% if selected_filters.wind_direction %}{{ selected_filters.wind_direction|join(', ') }}{% else %}All{% endif %}</span>
                </summary>
                <div class="check-list">
                  <label class="check-pill">
                    <input type="checkbox" data-filter-all {% if not selected_filters.wind_direction %}checked{% endif %}>
                    <span class="check-text">All wind directions</span>
                  </label>
                  {% for wind_direction in filter_options.wind_directions %}
                    <label class="check-pill">
                      <input type="checkbox" name="wind_direction" value="{{ wind_direction }}" data-filter-option {% if wind_direction in selected_filters.wind_direction %}checked{% endif %}>
                      <span class="check-text">{{ wind_direction }}</span>
                    </label>
                  {% endfor %}
                </div>
              </details>
              <details class="filter-group" data-filter-group>
                <summary>
                  <span class="filter-group-title">Wind Speed</span>
                  <span class="filter-summary">{% if selected_filters.wind_speed %}{{ selected_filters.wind_speed|join(', ') }}{% else %}All{% endif %}</span>
                </summary>
                <div class="check-list">
                  <label class="check-pill">
                    <input type="checkbox" data-filter-all {% if not selected_filters.wind_speed %}checked{% endif %}>
                    <span class="check-text">All wind speeds</span>
                  </label>
                  {% for wind_speed in filter_options.wind_speeds %}
                    <label class="check-pill">
                      <input type="checkbox" name="wind_speed" value="{{ wind_speed }}" data-filter-option {% if wind_speed in selected_filters.wind_speed %}checked{% endif %}>
                      <span class="check-text">{{ wind_speed }}</span>
                    </label>
                  {% endfor %}
                </div>
              </details>
            </div>
            <button type="submit">Apply filters</button>
          </form>
          <form method="post" action="{{ url_for('generate_filtered_condition_plots') }}">
            {% for formation in selected_filters.formation %}<input type="hidden" name="formation" value="{{ formation }}">{% endfor %}
            {% for wind_direction in selected_filters.wind_direction %}<input type="hidden" name="wind_direction" value="{{ wind_direction }}">{% endfor %}
            {% for wind_speed in selected_filters.wind_speed %}<input type="hidden" name="wind_speed" value="{{ wind_speed }}">{% endfor %}
            <button class="secondary" type="submit">Generate matched summaries</button>
          </form>
          <div>
            <h3>Matched Condition Groups</h3>
            <div class="experiment-tags" style="margin-top:8px;">
              {% for group in condition_groups %}
                <a href="{{ url_for('condition_detail', condition_key=group.condition_key) }}">
                  {{ group.condition_key }} ({{ group.included_trial_count }} used / {{ group.trial_count }} trials{% if group.outlier_count %}, {{ group.outlier_count }} outlier{% endif %})
                </a>
              {% else %}
                <span class="small">No experiments match the selected filters.</span>
              {% endfor %}
            </div>
          </div>
        </div>
        <div class="filter-panel" style="margin-bottom:14px;">
          <div class="run-status">
            <div>
              <h3>Baseline Data Filter</h3>
              <div class="small">
                Showing {{ baseline_runs|length }} matched baseline run(s) from {{ all_baseline_count }} total.
              </div>
            </div>
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end;">
              <button class="secondary" type="button" data-toggle-filter-groups>Expand all</button>
              <a class="badge" href="{{ url_for('index') }}">Reset</a>
            </div>
          </div>
          <form method="get" action="{{ url_for('index') }}">
            <div class="filter-grid">
              <details class="filter-group" data-filter-group>
                <summary>
                  <span class="filter-group-title">Drone</span>
                  <span class="filter-summary">{% if selected_baseline_filters.drone_number %}drone {{ selected_baseline_filters.drone_number|join(', drone ') }}{% else %}All{% endif %}</span>
                </summary>
                <div class="check-list">
                  <label class="check-pill">
                    <input type="checkbox" data-filter-all {% if not selected_baseline_filters.drone_number %}checked{% endif %}>
                    <span class="check-text">All drones</span>
                  </label>
                  {% for drone_number in baseline_filter_options.drones %}
                    <label class="check-pill">
                      <input type="checkbox" name="baseline_drone" value="{{ drone_number }}" data-filter-option {% if drone_number in selected_baseline_filters.drone_number %}checked{% endif %}>
                      <span class="check-text">drone {{ drone_number }}</span>
                    </label>
                  {% endfor %}
                </div>
              </details>
              <details class="filter-group" data-filter-group>
                <summary>
                  <span class="filter-group-title">Battery</span>
                  <span class="filter-summary">{% if selected_baseline_filters.battery_id %}{{ selected_baseline_filters.battery_id|join(', ') }}{% else %}All{% endif %}</span>
                </summary>
                <div class="check-list">
                  <label class="check-pill">
                    <input type="checkbox" data-filter-all {% if not selected_baseline_filters.battery_id %}checked{% endif %}>
                    <span class="check-text">All batteries</span>
                  </label>
                  {% for battery_id in baseline_filter_options.batteries %}
                    <label class="check-pill">
                      <input type="checkbox" name="baseline_battery" value="{{ battery_id }}" data-filter-option {% if battery_id in selected_baseline_filters.battery_id %}checked{% endif %}>
                      <span class="check-text">{{ battery_id }}</span>
                    </label>
                  {% endfor %}
                </div>
              </details>
              <details class="filter-group" data-filter-group>
                <summary>
                  <span class="filter-group-title">Mode</span>
                  <span class="filter-summary">{% if selected_baseline_filters.mode %}{{ selected_baseline_filters.mode|join(', ') }}{% else %}All{% endif %}</span>
                </summary>
                <div class="check-list">
                  <label class="check-pill">
                    <input type="checkbox" data-filter-all {% if not selected_baseline_filters.mode %}checked{% endif %}>
                    <span class="check-text">All modes</span>
                  </label>
                  {% for mode in baseline_filter_options.modes %}
                    <label class="check-pill">
                      <input type="checkbox" name="baseline_mode" value="{{ mode }}" data-filter-option {% if mode in selected_baseline_filters.mode %}checked{% endif %}>
                      <span class="check-text">{{ mode }}</span>
                    </label>
                  {% endfor %}
                </div>
              </details>
              <details class="filter-group" data-filter-group>
                <summary>
                  <span class="filter-group-title">Wind Level</span>
                  <span class="filter-summary">{% if selected_baseline_filters.wind_level %}{{ selected_baseline_filters.wind_level|join(', ') }}{% else %}All{% endif %}</span>
                </summary>
                <div class="check-list">
                  <label class="check-pill">
                    <input type="checkbox" data-filter-all {% if not selected_baseline_filters.wind_level %}checked{% endif %}>
                    <span class="check-text">All wind levels</span>
                  </label>
                  {% for wind_level in baseline_filter_options.wind_levels %}
                    <label class="check-pill">
                      <input type="checkbox" name="baseline_wind_level" value="{{ wind_level }}" data-filter-option {% if wind_level in selected_baseline_filters.wind_level %}checked{% endif %}>
                      <span class="check-text">{{ wind_level }}</span>
                    </label>
                  {% endfor %}
                </div>
              </details>
            </div>
            <button type="submit">Apply baseline filter</button>
          </form>
          <form method="post" action="{{ url_for('generate_baseline_summary') }}">
            {% for drone_number in selected_baseline_filters.drone_number %}<input type="hidden" name="baseline_drone" value="{{ drone_number }}">{% endfor %}
            {% for battery_id in selected_baseline_filters.battery_id %}<input type="hidden" name="baseline_battery" value="{{ battery_id }}">{% endfor %}
            {% for mode in selected_baseline_filters.mode %}<input type="hidden" name="baseline_mode" value="{{ mode }}">{% endfor %}
            {% for wind_level in selected_baseline_filters.wind_level %}<input type="hidden" name="baseline_wind_level" value="{{ wind_level }}">{% endfor %}
            <button class="secondary" type="submit">Generate baseline summary</button>
          </form>
          {% if baseline_summary_plot %}
            <div>
              <a class="file-name" href="{{ url_for('file_detail', relpath=baseline_summary_plot) }}">Open latest baseline summary plot</a>
              <img src="{{ url_for('raw_file', relpath=baseline_summary_plot) }}" alt="Baseline hover summary" style="width:100%;margin-top:10px;border:1px solid var(--line);border-radius:8px;background:#fff;">
            </div>
          {% endif %}
          {% for extra_plot in baseline_summary_extra_plots %}
            <div>
              <a class="file-name" href="{{ url_for('file_detail', relpath=extra_plot) }}">Open {{ extra_plot.rsplit('/', 1)[-1] }}</a>
              <img src="{{ url_for('raw_file', relpath=extra_plot) }}" alt="Baseline extra summary" style="width:100%;margin-top:10px;border:1px solid var(--line);border-radius:8px;background:#fff;">
            </div>
          {% endfor %}
          <div class="experiment-list" style="max-height:260px;">
            {% for run in baseline_runs %}
              <article class="experiment-row">
                <div class="experiment-title">
                  <div>
                    <h3>
                      {% if run.summary_relpath %}
                        <a class="file-name" href="{{ url_for('file_detail', relpath=run.summary_relpath) }}">{{ run.baseline_id }}</a>
                      {% else %}
                        <span class="file-name">{{ run.baseline_id }}</span>
                      {% endif %}
                    </h3>
                    <div class="small">drone {{ run.drone_number }} · {{ run.battery_id }} · {{ run.mode }}{% if run.wind_level %} · {{ run.wind_level }}{% endif %} · {{ run.mtime }}</div>
                  </div>
                  <span class="badge">{{ run.battery_start }}% -> {{ run.battery_end }}%</span>
                </div>
                <div class="small">
                  duration {{ run.duration_sec }}s · drop {{ run.battery_drop }}%{% if run.end_reason %} · {{ run.end_reason }}{% endif %}
                </div>
                {% if run.notes %}
                  <div class="small">{{ run.notes }}</div>
                {% endif %}
                <div class="record-actions">
                  {% if run.timeseries_relpath %}<a class="badge" href="{{ url_for('file_detail', relpath=run.timeseries_relpath) }}">timeseries</a>{% endif %}
                  {% if run.metadata_relpath %}<a class="badge" href="{{ url_for('file_detail', relpath=run.metadata_relpath) }}">metadata</a>{% endif %}
                  {% for plot in run.plots %}
                    <a class="badge image" href="{{ url_for('file_detail', relpath=plot) }}">plot</a>
                  {% endfor %}
                </div>
              </article>
            {% else %}
              <div class="small">No baseline runs match the selected filters.</div>
            {% endfor %}
          </div>
        </div>
        <div class="small" id="fileListHint">Select a condition group above to inspect its trials, or click a file category button to browse raw files.</div>
        <div class="file-list" id="workspaceFileList" hidden>
          {% for category, items in categories.items() %}
            {% for item in items %}
              <article class="file-row" data-category="{{ item.category }}">
                <div class="file-top">
                  <div>
                    {% if item.previewable %}
                      <a class="file-name" href="{{ url_for('file_detail', relpath=item.relpath) }}">{{ item.name }}</a>
                    {% else %}
                      <span class="file-name">{{ item.name }}</span>
                    {% endif %}
                    <div class="file-meta">{{ item.folder }} · {{ item.size }} · modified {{ item.mtime }}</div>
                  </div>
                  <div class="record-actions">
                    {% if item.is_csv %}
                      <form method="post" action="{{ url_for('generate_plots', relpath=item.relpath) }}">
                        <button class="secondary" type="submit">Generate plots</button>
                      </form>
                    {% endif %}
                    <form method="post"
                          action="{{ url_for('delete_file', relpath=item.relpath) }}"
                          onsubmit="return confirm('Delete {{ item.relpath }}? This cannot be undone.');">
                      <input type="hidden" name="next" value="{{ url_for('index') }}">
                      <button class="icon-delete" type="submit" title="Delete file" aria-label="Delete {{ item.name }}">×</button>
                    </form>
                    <span class="badge {{ item.category }}">{{ item.category }}</span>
                  </div>
                </div>
                {% if item.is_csv %}
                  <details>
                    <summary>{{ item.csv.row_count }} rows · {{ item.csv.trial_count }} trials</summary>
                    <div class="columns">
                      {% for formation in item.csv.formations %}
                        <span class="badge">{{ formation }}</span>
                      {% endfor %}
                      {% for trial_id in item.csv.trial_ids %}
                        <span class="badge">trial {{ trial_id }}</span>
                      {% endfor %}
                    </div>
                    {% if item.csv.battery_columns %}
                      <div class="small">Battery: {{ item.csv.battery_columns|join(', ') }}</div>
                    {% endif %}
                    {% if item.csv.coordination_columns %}
                      <div class="small">Coordination: {{ item.csv.coordination_columns|join(', ') }}</div>
                    {% endif %}
                  </details>
                {% endif %}
              </article>
            {% endfor %}
          {% endfor %}
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Experiment Area</h2>
        <span class="small">Create a run record before or after a flight</span>
      </div>
      <div class="section-body">
        <div class="mode-switch">
          <button class="area-mode-button active" type="button" data-area-mode="experiment">Experiment Area</button>
          <button class="area-mode-button" type="button" data-area-mode="baseline">Single Drone Baseline</button>
        </div>

        <div class="mode-panel" id="experimentAreaPanel">
        <form method="post" action="{{ url_for('create_experiment') }}">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <label>Experiment ID
              <input name="experiment_id" placeholder="e.g. 001 -> front_head_lv1_new_001">
            </label>
            <label>Selected Formation
              <input id="formationDisplay" value="front" readonly>
              <input id="formationInput" type="hidden" name="formation" value="front">
            </label>
          </div>
          <label>Status
            <select name="status">
              <option value="planned">planned</option>
              <option value="running">running</option>
              <option value="completed">completed</option>
              <option value="analysis">analysis</option>
            </select>
          </label>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <label>Wind Direction
              <input id="windDirectionDisplay" value="head wind" readonly>
              <input id="windDirectionInput" type="hidden" name="wind_direction" value="head wind">
            </label>
            <label>Wind Speed
              <select name="wind_speed">
                {% for wind_speed in wind_speed_options %}
                  <option value="{{ wind_speed }}">{{ wind_speed }}</option>
                {% endfor %}
              </select>
            </label>
          </div>
          <label>Inter-drone Distance
            <select id="interDroneDistanceInput" name="inter_drone_distance_cm">
              {% for distance_cm in inter_drone_distance_options_cm %}
                <option value="{{ distance_cm }}" {% if distance_cm == 50 %}selected{% endif %}>{{ distance_cm }} cm</option>
              {% endfor %}
            </select>
          </label>
          <div class="formation-toolbar wind-toolbar" style="margin-top:8px;">
            <button class="wind-option active" type="button" data-wind-direction="head wind">head wind</button>
            <button class="wind-option" type="button" data-wind-direction="tail wind">tail wind</button>
            <button class="wind-option" type="button" data-wind-direction="side wind">side wind</button>
          </div>
          <div>
            <h3>Run Experiment Board</h3>
            <div class="formation-toolbar" style="margin-top:8px;">
              <button class="formation-option active" type="button" data-formation="front">front</button>
              <button class="formation-option" type="button" data-formation="column">column</button>
              <button class="formation-option" type="button" data-formation="vee">vee</button>
              <button class="formation-option" type="button" data-formation="echalon">echalon</button>
              <button class="formation-option" type="button" data-formation="diamond">diamond</button>
            </div>
            <div class="mission-board" style="margin-top:10px;">
              <div class="mission-grid" id="missionGrid" style="grid-template-columns: repeat({{ max_mission_cols }}, minmax(74px, 1fr));">
                {% for display_row in range(max_mission_rows - 1, -1, -1) %}
                  {% for col in range(max_mission_cols) %}
                    {% set standard_pad = mission_pad_layouts.standard[col][display_row] if col < mission_pad_layouts.standard|length and display_row < mission_pad_layouts.standard[col]|length else '' %}
                    {% set column_pad = mission_pad_layouts.column[col][display_row] if col < mission_pad_layouts.column|length and display_row < mission_pad_layouts.column[col]|length else '' %}
                    {% set column_75_pad = mission_pad_layouts.column_75[col][display_row] if col < mission_pad_layouts.column_75|length and display_row < mission_pad_layouts.column_75[col]|length else '' %}
                    {% set vee_pad = mission_pad_layouts.vee[col][display_row] if col < mission_pad_layouts.vee|length and display_row < mission_pad_layouts.vee[col]|length else '' %}
                    {% set diamond_pad = mission_pad_layouts.diamond[col][display_row] if col < mission_pad_layouts.diamond|length and display_row < mission_pad_layouts.diamond[col]|length else '' %}
                    {% set pad_id = standard_pad %}
                    <div class="pad-cell"
                         data-col="{{ col }}"
                         data-row="{{ display_row }}"
                         data-pad="{{ pad_id }}"
                         data-standard-pad="{{ standard_pad }}"
                         data-column-pad="{{ column_pad }}"
                         data-column75-pad="{{ column_75_pad }}"
                         data-vee-pad="{{ vee_pad }}"
                         data-diamond-pad="{{ diamond_pad }}"
                         data-front-active="{{ 'true' if display_row == 0 else 'false' }}"
                         data-column-active="{{ 'true' if col == 0 else 'false' }}">
                      <div class="pad-title">
                        <span>pad</span>
                        <span class="pad-id">{{ pad_id }}</span>
                      </div>
                      <input type="hidden" name="pad_id_{{ col }}_{{ display_row }}" value="{{ pad_id }}">
                      <input type="hidden" name="pad_order_{{ col }}_{{ display_row }}" value="">
                      <input type="hidden" name="pad_role_{{ col }}_{{ display_row }}" value="">
                      <select name="pad_ip_{{ col }}_{{ display_row }}" disabled>
                        <option value="">drone</option>
                        {% for drone_number, ip_suffix in drone_options %}
                          <option value="{{ drone_number }}">{{ drone_number }}</option>
                        {% endfor %}
                      </select>
                      <select name="pad_battery_{{ col }}_{{ display_row }}" disabled>
                        <option value="">battery</option>
                        {% for battery_id in battery_options %}
                          <option value="{{ battery_id }}">{{ battery_id }}</option>
                        {% endfor %}
                      </select>
                    </div>
                  {% endfor %}
                {% endfor %}
              </div>
              <div class="board-note">
                <span class="small" id="frontFormationNote">front: bottom row, left to right = 1, 2, 3, 4, 5</span>
                <span class="small">column: left lane = 1, 2, 3, 4, 5, 6, 7, 8, 3, 4</span>
                <span class="small">drone #: 1->102, 2->109, 3->103, 4->106, 5->107</span>
                <span class="small">default battery order: B11, B10, B13, B14, B12</span>
              </div>
            </div>
          </div>
          <div>
            <h3>Mission Pad Layout</h3>
            <div class="small" id="experimentMissionLayoutText" style="margin-top:6px;">
              Columns left to right:
              {% for column in mission_pad_columns %}
                {{ column|join('-') }}{% if not loop.last %} · {% endif %}
              {% endfor %}
            </div>
          </div>
          <label>Notes
            <textarea name="notes" placeholder="Mission pads, battery condition, weather, abnormal behavior..."></textarea>
          </label>
          <button type="submit">Create experiment record</button>
        </form>
        </div>

        <div class="mode-panel" id="baselineAreaPanel" hidden>

        <div class="run-panel">
          <div class="run-status">
            <div>
              <h3>Single Drone Battery Baseline</h3>
              <div class="small">Test one drone with one battery before adding it to formal experiments. Data is saved under database/baselines/.</div>
            </div>
            <span class="badge">baseline mode</span>
          </div>
          <form id="baselineForm" method="post" action="{{ url_for('start_baseline') }}">
            <div class="baseline-grid">
              <label>Drone
                <select name="baseline_drone_number" required>
                  <option value="">drone</option>
                  {% for drone_number, ip_suffix in drone_options %}
                    <option value="{{ drone_number }}">{{ drone_number }}</option>
                  {% endfor %}
                </select>
              </label>
              <label>Battery
                <select name="baseline_battery_id" required>
                  <option value="">battery</option>
                  {% for battery_id in battery_options %}
                    <option value="{{ battery_id }}">{{ battery_id }}</option>
                  {% endfor %}
                </select>
              </label>
              <label>Baseline Test
                <select id="baselineModeInput" name="baseline_mode" required>
                  {% for mode, label in baseline_modes %}
                    <option value="{{ mode }}">{{ label }}</option>
                  {% endfor %}
                </select>
              </label>
              <label>Wind Level
                <select name="baseline_wind_level" required>
                  <option value="">level</option>
                  {% for wind_level, label in baseline_wind_levels %}
                    <option value="{{ wind_level }}">{{ label }}</option>
                  {% endfor %}
                </select>
              </label>
              <label>Flight Direction
                <input id="baselineDirectionDisplay" value="↑ up" readonly>
                <input id="baselineDirectionInput" type="hidden" name="baseline_direction" value="up">
              </label>
            </div>
            <div id="baselineMissionPadPanel">
              <h3>Baseline Mission Pad Board</h3>
              <div class="small" style="margin-top:6px;">Choose the start mission pad cell, then use the arrow buttons to set the pass-through direction.</div>
              <div class="mission-board" style="margin-top:10px;">
                <div class="mission-grid" id="baselineMissionGrid">
                  {% for display_row in range(mission_pad_columns[0]|length - 1, -1, -1) %}
                    {% for col in range(5) %}
                      {% set pad_id = mission_pad_columns[col][display_row] %}
                      <button class="baseline-pad-cell"
                              type="button"
                              data-col="{{ col }}"
                              data-row="{{ display_row }}"
                              data-pad="{{ pad_id }}">
                        <div class="pad-title">
                          <span>pad</span>
                          <span class="pad-id">{{ pad_id }}</span>
                        </div>
                        <div class="path-index"></div>
                      </button>
                    {% endfor %}
                  {% endfor %}
                </div>
                <div class="arrow-toolbar" style="margin-top:10px;">
                  <button class="arrow-option active" type="button" data-baseline-direction="up" title="Fly upward">↑</button>
                  <button class="arrow-option" type="button" data-baseline-direction="down" title="Fly downward">↓</button>
                </div>
                <div class="board-note">
                  <span class="small" id="baselinePathPreview">Select a start mission pad.</span>
                </div>
              </div>
            </div>
            <input id="baselineStartPadInput" type="hidden" name="baseline_start_pad" value="">
            <input id="baselineStartColInput" type="hidden" name="baseline_start_col" value="">
            <input id="baselineStartRowInput" type="hidden" name="baseline_start_row" value="">
            <label>Baseline Notes
              <textarea name="baseline_notes" placeholder="New battery batch, wind setup, room layout, abnormal behavior..."></textarea>
            </label>
            <button type="submit">Start baseline test</button>
          </form>
        </div>
        </div>

        <div style="height:1px;background:var(--line);margin:18px 0;"></div>

        <div class="run-panel">
          <div class="run-status">
            <div>
              <h3>Experiment Runner</h3>
              <div class="small" id="runMessage">{{ run_state.message or 'No experiment is running.' }}</div>
            </div>
            <div class="record-actions">
              <span class="badge" id="runStatus">{{ run_state.status }}</span>
              <button class="secondary" type="button" id="refreshRunButton">Refresh</button>
              <button class="danger" type="button" id="stopRunButton">Stop</button>
            </div>
          </div>
          <div class="battery-window-note">
            <div class="small">Live battery monitor · experiment window <strong id="batteryWindowLabel">{{ battery_window.high }}% - {{ battery_window.low }}%</strong></div>
            <div class="small">selected batteries: B02 · B04 · B06 · B07 · B10</div>
          </div>
          <div class="battery-monitor" id="batteryMonitor">
            <div class="small">Prepare an experiment to see live drone battery levels.</div>
          </div>
          <div class="terminal-shell">
            <div class="terminal-titlebar">
              <div class="terminal-dots">
                <span class="terminal-dot red"></span>
                <span class="terminal-dot yellow"></span>
                <span class="terminal-dot green"></span>
              </div>
              <div class="terminal-title" id="runTerminalTitle">
                {{ run_state.script or 'data_collector.py' }}{% if run_state.experiment_id %} --experiment-id {{ run_state.experiment_id }}{% endif %}
              </div>
              <div class="terminal-meta" id="runTerminalMeta">{{ run_state.started_at or 'idle' }}</div>
            </div>
            <div class="terminal" id="runTerminal">{% for line in run_state.output %}{{ line }}
{% else %}$ waiting for experiment command...
{% endfor %}</div>
          </div>
        </div>

        <div style="height:1px;background:var(--line);margin:18px 0;"></div>

        <div class="experiment-list">
          {% for exp in experiments %}
            <article class="experiment-row" id="{{ exp.experiment_id }}">
              <div class="experiment-title">
                <div>
              <h3>
                <a class="file-name" href="{{ url_for('experiment_detail', experiment_id=exp.experiment_id) }}">{{ exp.experiment_id }}</a>
                {% if exp.is_outlier %}<span class="badge outlier">outlier</span>{% endif %}
                {% for tag in exp.tags or [] %}<span class="badge">{{ tag }}</span>{% endfor %}
              </h3>
                  <div class="small">
                    {{ exp.formation or 'formation not set' }}
                    {% if exp.wind_direction or exp.wind_speed %}
                      · {{ exp.wind_direction or 'wind direction not set' }} · {{ exp.wind_speed or 'wind speed not set' }}
                    {% endif %}
                    · distance {{ exp.inter_drone_distance_cm or 50 }}cm
                    · {{ exp.created_at }}
                  </div>
                </div>
                <span class="badge">{{ exp.status }}</span>
              </div>
              {% if exp.drones %}
                <div class="small">
                  {% for drone in exp.drones %}
                    {{ drone.takeoff_order }}: {{ drone.ip }} {{ drone.role }}{% if drone.battery_id %} / {{ drone.battery_id }}{% endif %}{% if not loop.last %} · {% endif %}
                  {% endfor %}
                </div>
              {% endif %}
              {% if exp.notes %}
                <div class="small">{{ exp.notes }}</div>
              {% endif %}
              <details class="edit-panel">
                <summary>Edit experiment record</summary>
                <form method="post" action="{{ url_for('update_experiment', experiment_id=exp.experiment_id) }}">
                  <div class="edit-grid">
                    <label>Experiment ID
                      <input name="experiment_id" value="{{ exp.experiment_id }}">
                    </label>
                    <label>Formation
                      <select name="formation">
                        {% for formation in ['front', 'column', 'vee', 'echalon', 'diamond'] %}
                          <option value="{{ formation }}" {% if exp.formation == formation %}selected{% endif %}>{{ formation }}</option>
                        {% endfor %}
                      </select>
                    </label>
                    <label>Status
                      <select name="status">
                        {% for status in ['planned', 'running', 'completed', 'analysis'] %}
                          <option value="{{ status }}" {% if exp.status == status %}selected{% endif %}>{{ status }}</option>
                        {% endfor %}
                      </select>
                    </label>
                    <label>Wind Direction
                      <select name="wind_direction">
                        {% for wind_direction in ['head wind', 'tail wind', 'side wind'] %}
                          <option value="{{ wind_direction }}" {% if exp.wind_direction == wind_direction %}selected{% endif %}>{{ wind_direction }}</option>
                        {% endfor %}
                      </select>
                    </label>
                    <label>Wind Speed
                      <select name="wind_speed">
                        {% for wind_speed in wind_speed_options %}
                          <option value="{{ wind_speed }}" {% if exp.wind_speed == wind_speed %}selected{% endif %}>{{ wind_speed }}</option>
                        {% endfor %}
                      </select>
                    </label>
                    <label>Inter-drone Distance
                      <select name="inter_drone_distance_cm">
                        {% for distance_cm in inter_drone_distance_options_cm %}
                          <option value="{{ distance_cm }}" {% if (exp.inter_drone_distance_cm or 50)|int == distance_cm %}selected{% endif %}>{{ distance_cm }} cm</option>
                        {% endfor %}
                      </select>
                    </label>
                  </div>
                  <label>Notes
                    <textarea name="notes">{{ exp.notes or '' }}</textarea>
                  </label>
                  <div>
                    <h3>Editable Drone Numbers</h3>
                    <div class="edit-drone-grid" style="margin-top:8px;">
                      <div class="head">Drone #</div>
                      <div class="head">Battery</div>
                      <div class="head">Order</div>
                      <div class="head">Role</div>
                      <div class="head">Pad</div>
                      <div class="head">Grid col</div>
                      <div class="head">Grid row</div>
                      {% for idx in range(5) %}
                        {% set drone = exp.drones[idx] if exp.drones and idx < exp.drones|length else {} %}
                        <select name="edit_drone_number">
                          <option value="">drone</option>
                          {% for drone_number, ip_suffix in drone_options %}
                            <option value="{{ drone_number }}" {% if display_drone_number(drone) == drone_number %}selected{% endif %}>{{ drone_number }}</option>
                          {% endfor %}
                        </select>
                        <select name="edit_battery_id">
                          <option value="">battery</option>
                          {% for battery_id in battery_options %}
                            <option value="{{ battery_id }}" {% if display_battery_id(drone) == battery_id %}selected{% endif %}>{{ battery_id }}</option>
                          {% endfor %}
                        </select>
                        <input name="edit_takeoff_order" value="{{ drone.takeoff_order or '' }}" placeholder="{{ idx + 1 }}">
                        <input name="edit_drone_role" value="{{ drone.role or '' }}" placeholder="front_{{ idx + 1 }}">
                        <input name="edit_mission_pad" value="{{ drone.mission_pad or '' }}" placeholder="1">
                        <input name="edit_grid_column" value="{{ drone.grid_column or '' }}" placeholder="0">
                        <input name="edit_grid_row" value="{{ drone.grid_row or '' }}" placeholder="0">
                      {% endfor %}
                    </div>
                  </div>
                  <button type="submit">Save changes</button>
                </form>
              </details>
              <div class="record-actions">
                <form method="post" action="{{ url_for('toggle_experiment_outlier', experiment_id=exp.experiment_id) }}">
                  <input type="hidden" name="next" value="{{ url_for('index', selected=exp.experiment_id) }}">
                  <input type="hidden" name="is_outlier" value="{{ '0' if exp.is_outlier else '1' }}">
                  <button class="secondary" type="submit">{{ 'Normal' if exp.is_outlier else 'Outlier' }}</button>
                </form>
                <button class="secondary start-experiment-button"
                        type="button"
                        data-start-url="{{ url_for('start_experiment', experiment_id=exp.experiment_id) }}"
                        {% if exp.formation not in experiment_scripts %}disabled title="No runnable script is configured for this formation yet"{% endif %}>
                  Prepare experiment
                </button>
                <form method="post"
                      action="{{ url_for('delete_experiment', experiment_id=exp.experiment_id) }}"
                      onsubmit="return confirm('Delete experiment record {{ exp.experiment_id }}? Data files in database/ will not be deleted.');">
                  <button class="danger" type="submit">Delete record</button>
                </form>
              </div>
            </article>
          {% else %}
            <div class="small">No experiment records yet. Create one above to start grouping future data.</div>
          {% endfor %}
        </div>
      </div>
    </section>
  </main>

  <div class="modal-backdrop" id="takeoffModal" role="dialog" aria-modal="true">
    <div class="modal">
      <h2 id="takeoffModalTitle">Ready for takeoff</h2>
      <div class="small" id="takeoffModalBody">
        The script has completed preflight connection checks and is waiting at the takeoff prompt.
        Confirm only when the flight area is clear.
      </div>
      <div class="record-actions">
        <button type="button" id="confirmTakeoffButton">Confirm takeoff</button>
        <button class="secondary" type="button" id="skipDischargeButton" hidden>Skip discharge</button>
        <button class="secondary" type="button" id="closeTakeoffModalButton">Wait</button>
      </div>
    </div>
  </div>

  <script>
    const tabs = document.querySelectorAll(".tab");
    const rows = document.querySelectorAll(".file-row");
    const workspaceFileList = document.getElementById("workspaceFileList");
    const fileListHint = document.getElementById("fileListHint");
    const areaModeButtons = document.querySelectorAll(".area-mode-button");
    const experimentAreaPanel = document.getElementById("experimentAreaPanel");
    const baselineAreaPanel = document.getElementById("baselineAreaPanel");
    let activeFileFilter = null;
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const filter = tab.dataset.filter;
        if (activeFileFilter === filter) {
          activeFileFilter = null;
          tab.classList.remove("active");
          workspaceFileList.hidden = true;
          fileListHint.textContent = "Select a condition group above to inspect its trials, or click a file category button to browse raw files.";
          return;
        }
        activeFileFilter = filter;
        tabs.forEach((item) => item.classList.remove("active"));
        tab.classList.add("active");
        workspaceFileList.hidden = false;
        fileListHint.textContent = `Showing ${filter === "all" ? "all raw files" : filter + " files"}. Click the active category again to hide them.`;
        rows.forEach((row) => {
          row.hidden = filter !== "all" && row.dataset.category !== filter;
        });
      });
    });

    areaModeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const mode = button.dataset.areaMode;
        areaModeButtons.forEach((item) => item.classList.toggle("active", item === button));
        experimentAreaPanel.hidden = mode !== "experiment";
        baselineAreaPanel.hidden = mode !== "baseline";
      });
    });

    const formationInput = document.getElementById("formationInput");
    const formationDisplay = document.getElementById("formationDisplay");
    const formationButtons = document.querySelectorAll(".formation-option");
    const windDirectionInput = document.getElementById("windDirectionInput");
    const windDirectionDisplay = document.getElementById("windDirectionDisplay");
    const windDirectionButtons = document.querySelectorAll(".wind-option");
    const missionGrid = document.getElementById("missionGrid");
    const padCells = Array.from(document.querySelectorAll("#missionGrid .pad-cell"));
    const interDroneDistanceInput = document.getElementById("interDroneDistanceInput");
    const frontFormationNote = document.getElementById("frontFormationNote");
    const experimentMissionLayoutText = document.getElementById("experimentMissionLayoutText");
    const recommendedDroneOrder = ["1", "2", "3", "4", "5"];
    const recommendedBatteryOrder = ["B11", "B10", "B13", "B14", "B12"];
    const standardTopMissionRow = Math.max(...padCells.filter((cell) => cell.dataset.standardPad).map((cell) => Number(cell.dataset.row)));

    function isFrontTailWind() {
      return formationInput.value === "front" && windDirectionInput && windDirectionInput.value === "tail wind";
    }

    function isFrontHeadWind75cm() {
      return formationInput.value === "front" &&
             windDirectionInput &&
             windDirectionInput.value === "head wind" &&
             interDroneDistanceInput &&
             interDroneDistanceInput.value === "75";
    }

    function isEchalonFormation() {
      return formationInput.value === "echalon";
    }

    function isEchalonHeadWind75cm() {
      return isEchalonFormation() &&
             windDirectionInput &&
             windDirectionInput.value === "head wind" &&
             interDroneDistanceInput &&
             interDroneDistanceInput.value === "75";
    }

    function isVee75cm() {
      return formationInput.value === "vee" &&
             interDroneDistanceInput &&
             interDroneDistanceInput.value === "75";
    }

    function isFrontTailWind75cm() {
      return isFrontTailWind() && interDroneDistanceInput && interDroneDistanceInput.value === "75";
    }

    function isFrontReversePath() {
      return isFrontHeadWind75cm();
    }

    function isDiamondTailWind() {
      return formationInput.value === "diamond" && windDirectionInput && windDirectionInput.value === "tail wind";
    }

    function isColumnHeadWind() {
      return formationInput.value === "column" && windDirectionInput && windDirectionInput.value === "head wind";
    }

    function isColumn75cm() {
      return formationInput.value === "column" &&
             interDroneDistanceInput &&
             interDroneDistanceInput.value === "75";
    }

    function activeLayoutName(formation) {
      if (formation === "diamond") return "diamond";
      if (formation === "vee") return isVee75cm() ? "standard" : "vee";
      if (formation === "column") return isColumn75cm() ? "column75" : "column";
      return "standard";
    }

    function layoutPadForCell(cell, layoutName) {
      if (layoutName === "diamond") return cell.dataset.diamondPad;
      if (layoutName === "vee") return cell.dataset.veePad;
      if (layoutName === "column") return cell.dataset.columnPad;
      if (layoutName === "column75") return cell.dataset.column75Pad;
      return cell.dataset.standardPad;
    }

    function isFormationCell(cell, formation, layoutName) {
      const col = Number(cell.dataset.col);
      const row = Number(cell.dataset.row);
      const layoutPad = layoutPadForCell(cell, layoutName);
      if (!layoutPad) return false;
      if (formation === "front") return row === (isFrontReversePath() ? standardTopMissionRow : 0);
      if (formation === "echalon") return row === (isEchalonHeadWind75cm() ? standardTopMissionRow : 0);
      if (formation === "column") {
        if (isColumn75cm()) {
          return col === 0 && (isColumnHeadWind() ? row >= 4 && row <= 8 : row >= 0 && row <= 4);
        }
        return col === 0 && (isColumnHeadWind() ? row >= 5 && row <= 9 : row >= 0 && row <= 4);
      }
      if (formation === "vee") {
        return row === 0;
      }
      if (formation === "diamond") {
        const rowOffset = isDiamondTailWind() ? 5 : 0;
        return (row === rowOffset && col === 2) ||
               (row === rowOffset + 1 && col >= 1 && col <= 3) ||
               (row === rowOffset + 2 && col === 2);
      }
      return false;
    }

    function orderedActiveCells(formation, layoutName) {
      const activeCells = padCells.filter((cell) => isFormationCell(cell, formation, layoutName));
      if (formation === "front" || formation === "echalon") {
        return activeCells.sort((a, b) => Number(a.dataset.col) - Number(b.dataset.col));
      }
      if (formation === "column") {
        return activeCells.sort((a, b) => (
          isColumnHeadWind()
            ? Number(b.dataset.row) - Number(a.dataset.row)
            : Number(a.dataset.row) - Number(b.dataset.row)
        ));
      }
      if (formation === "vee") {
        return activeCells.sort((a, b) => (
          Number(a.dataset.row) - Number(b.dataset.row) ||
          Number(a.dataset.col) - Number(b.dataset.col)
        ));
      }
      if (formation === "diamond") {
        if (isDiamondTailWind()) {
          return activeCells.sort((a, b) => (
            Number(b.dataset.row) - Number(a.dataset.row) ||
            Number(a.dataset.col) - Number(b.dataset.col)
          ));
        }
        return activeCells.sort((a, b) => (
          Number(a.dataset.row) - Number(b.dataset.row) ||
          Number(a.dataset.col) - Number(b.dataset.col)
        ));
      }
      return activeCells;
    }

    function setFormation(formation) {
      formationInput.value = formation;
      formationDisplay.value = formation;
      const layoutName = activeLayoutName(formation);
      const isDiamond = formation === "diamond";
      if (missionGrid) {
        missionGrid.classList.toggle("column-layout", formation === "column");
        missionGrid.style.gridTemplateColumns = formation === "column"
          ? "minmax(120px, 180px)"
          : "repeat(5, minmax(74px, 1fr))";
      }
      if (frontFormationNote) {
        if (isDiamond) {
          frontFormationNote.textContent = isDiamondTailWind()
            ? "diamond + tail wind: starts at rows 6-8 in diamond shape; flies back 5 cells to rows 1-3."
            : "diamond: 3 columns x 8 rows; starts at row1 middle, row2 all three pads, row3 middle; each drone flies forward 5 cells.";
        } else if (formation === "vee") {
          frontFormationNote.textContent = isVee75cm()
            ? "vee + 75cm: first row pads are 1, 2, 3, 4, 5; columns use custom V origins and each lane keeps 50cm row spacing."
            : "vee: first row across five angled lanes = pads 1, 5, 3, 5, 1; adjacent drone spacing = 50 cm.";
        } else if (formation === "column") {
          frontFormationNote.textContent = isColumn75cm()
            ? (isColumnHeadWind()
              ? "column + head wind + 75cm: starts at pads 1, 8, 7, 6, 5 and flies 300cm back to 5, 4, 3, 2, 1."
              : "column + tail/side wind + 75cm: starts at pads 1, 2, 3, 4, 5 and flies 300cm to 5, 6, 7, 8, 1.")
            : (isColumnHeadWind()
              ? "column + head wind: starts at pads 4, 3, 8, 7, 6; flies back to 5, 4, 3, 2, 1."
              : "column + tail wind: starts at pads 1, 2, 3, 4, 5; flies forward to 6, 7, 8, 3, 4.");
        } else if (formation === "echalon") {
          frontFormationNote.textContent = isEchalonHeadWind75cm()
            ? "echalon + head wind + 75cm: top row, left to right = 6, 7, 8, 1, 2; flies along -Y to 1, 2, 3, 4, 5."
            : "echalon + tail/side wind + 75cm: bottom row, left to right = 1, 2, 3, 4, 5; flies along +Y to 6, 7, 8, 1, 2.";
        } else {
          frontFormationNote.textContent = isFrontHeadWind75cm()
            ? "front + head wind + 75cm: top row, left to right = 6, 7, 8, 1, 2; flies along -Y to 1, 2, 3, 4, 5; x spacing = 75cm."
            : isFrontTailWind75cm()
              ? "front + tail wind + 75cm: bottom row, left to right = 1, 2, 3, 4, 5; flies along +Y to 6, 7, 8, 1, 2; x spacing = 75cm."
              : "front: bottom row, left to right = 1, 2, 3, 4, 5";
        }
      }
      if (experimentMissionLayoutText) {
        if (formation === "vee") {
          experimentMissionLayoutText.textContent = isVee75cm()
            ? "Vee + 75cm uses front mission-pad order with custom V-shaped origins; all drones start from row 1 and each lane keeps 50cm row spacing."
            : "Columns left to right: 1-2-3-4-5-6 · 5-6-7-8-1-2 · 3-4-5-6-7-8 · 5-6-7-8-1-2 · 1-2-3-4-7-8";
        } else if (formation === "column") {
          experimentMissionLayoutText.textContent = isColumn75cm()
            ? "Column + 75cm uses the left lane only: 1-2-3-4-5-6-7-8-1, with 75cm pad spacing and 300cm flight distance."
            : "Column uses the left lane only: 1-2-3-4-5-6-7-8-3-4";
        } else if (isDiamond) {
          experimentMissionLayoutText.textContent = "Diamond uses the middle three columns: 1-2-3-4-5-6-7-8 · 2-3-4-5-6-7-8-1 · 3-4-5-6-7-8-1-2";
        } else if (formation === "echalon") {
          experimentMissionLayoutText.textContent = "Echalon uses front mission-pad order with custom 75cm diagonal origins; each lane keeps 50cm mission-pad row spacing.";
        } else {
          experimentMissionLayoutText.textContent = "Columns left to right: 1-2-3-4-5-6 · 2-3-4-5-6-7 · 3-4-5-6-7-8 · 4-5-6-7-8-1 · 5-6-7-8-1-2";
        }
      }
      formationButtons.forEach((button) => {
        button.classList.toggle("active", button.dataset.formation === formation);
      });

      padCells.forEach((cell) => {
        const layoutPad = layoutPadForCell(cell, layoutName);
        cell.hidden = !layoutPad && !isDiamond;
        cell.dataset.pad = layoutPad || "";
        const padLabel = cell.querySelector(".pad-id");
        const padHidden = cell.querySelector(`input[name="pad_id_${cell.dataset.col}_${cell.dataset.row}"]`);
        if (padLabel) padLabel.textContent = layoutPad || "";
        if (padHidden) padHidden.value = layoutPad || "";
        const input = cell.querySelector('select[name^="pad_ip_"]');
        const battery = cell.querySelector('select[name^="pad_battery_"]');
        const order = cell.querySelector(`input[name="pad_order_${cell.dataset.col}_${cell.dataset.row}"]`);
        const role = cell.querySelector(`input[name="pad_role_${cell.dataset.col}_${cell.dataset.row}"]`);
        cell.classList.remove("active");
        input.disabled = true;
        input.required = false;
        battery.disabled = true;
        battery.required = false;
        battery.value = "";
        order.value = "";
        role.value = "";
      });

      orderedActiveCells(formation, layoutName).forEach((cell, index) => {
        const input = cell.querySelector('select[name^="pad_ip_"]');
        const battery = cell.querySelector('select[name^="pad_battery_"]');
        const order = cell.querySelector(`input[name="pad_order_${cell.dataset.col}_${cell.dataset.row}"]`);
        const role = cell.querySelector(`input[name="pad_role_${cell.dataset.col}_${cell.dataset.row}"]`);
        cell.classList.add("active");
        input.disabled = false;
        input.required = true;
        input.value = recommendedDroneOrder[index] || "";
        battery.disabled = false;
        battery.required = true;
        battery.value = recommendedBatteryOrder[index] || "";
        order.value = String(index + 1);
        role.value = `${formation}_${index + 1}`;
      });
    }

    formationButtons.forEach((button) => {
      button.addEventListener("click", () => {
        setFormation(button.dataset.formation);
      });
    });
    windDirectionButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const windDirection = button.dataset.windDirection;
        windDirectionInput.value = windDirection;
        windDirectionDisplay.value = windDirection;
        windDirectionButtons.forEach((item) => {
          item.classList.toggle("active", item === button);
        });
        setFormation(formationInput.value);
      });
    });
    if (interDroneDistanceInput) {
      interDroneDistanceInput.addEventListener("change", () => {
        setFormation(formationInput.value);
      });
    }

    setFormation(formationInput.value || "front");

    const experimentForm = document.querySelector('form[action="{{ url_for("create_experiment") }}"]');
    if (experimentForm) {
      experimentForm.addEventListener("submit", (event) => {
        const selectedDrones = Array.from(experimentForm.querySelectorAll('select[name^="pad_ip_"]:not(:disabled)'))
          .map((select) => select.value)
          .filter(Boolean);
        const selectedBatteries = Array.from(experimentForm.querySelectorAll('select[name^="pad_battery_"]:not(:disabled)'))
          .map((select) => select.value)
          .filter(Boolean);
        if (selectedDrones.length !== new Set(selectedDrones).size) {
          event.preventDefault();
          alert("Each active mission pad must use a different drone.");
          return;
        }
        if (selectedBatteries.length !== new Set(selectedBatteries).size) {
          event.preventDefault();
          alert("Each active drone must use a different battery ID.");
        }
      });
    }

    const baselineCells = Array.from(document.querySelectorAll(".baseline-pad-cell"));
    const baselineModeInput = document.getElementById("baselineModeInput");
    const baselineMissionPadPanel = document.getElementById("baselineMissionPadPanel");
    const baselineDirectionInput = document.getElementById("baselineDirectionInput");
    const baselineDirectionDisplay = document.getElementById("baselineDirectionDisplay");
    const baselineStartPadInput = document.getElementById("baselineStartPadInput");
    const baselineStartColInput = document.getElementById("baselineStartColInput");
    const baselineStartRowInput = document.getElementById("baselineStartRowInput");
    const baselinePathPreview = document.getElementById("baselinePathPreview");
    const baselineArrowButtons = document.querySelectorAll(".arrow-option");
    let selectedBaselineCell = null;

    function isHoverBaselineMode() {
      return baselineModeInput && baselineModeInput.value === "hover";
    }

    function baselineCellAt(col, row) {
      return baselineCells.find((cell) => Number(cell.dataset.col) === col && Number(cell.dataset.row) === row);
    }

    function setBaselineDirection(direction) {
      baselineDirectionInput.value = direction;
      baselineDirectionDisplay.value = direction === "down" ? "↓ down" : "↑ up";
      baselineArrowButtons.forEach((item) => item.classList.toggle("active", item.dataset.baselineDirection === direction));
    }

    function baselineTargetRow() {
      if (!selectedBaselineCell) return null;
      const startRow = Number(selectedBaselineCell.dataset.row);
      const direction = baselineDirectionInput.value === "down" ? -1 : 1;
      return startRow + direction * 5;
    }

    function baselinePathFromSelection() {
      if (!selectedBaselineCell) return [];
      const col = Number(selectedBaselineCell.dataset.col);
      const startRow = Number(selectedBaselineCell.dataset.row);
      const direction = baselineDirectionInput.value === "down" ? -1 : 1;
      const targetRow = baselineTargetRow();
      const path = [];
      for (let row = startRow; row !== targetRow + direction; row += direction) {
        const cell = baselineCellAt(col, row);
        if (cell) path.push(cell);
      }
      return path;
    }

    function renderBaselinePath() {
      if (baselineMissionPadPanel) {
        baselineMissionPadPanel.hidden = isHoverBaselineMode();
      }
      baselineCells.forEach((cell) => {
        cell.classList.remove("selected", "in-path");
        const index = cell.querySelector(".path-index");
        if (index) index.textContent = "";
      });
      if (isHoverBaselineMode()) {
        baselineStartPadInput.value = "";
        baselineStartColInput.value = "";
        baselineStartRowInput.value = "";
        baselinePathPreview.textContent = "Hover baseline: no mission pad is required; it records from takeoff until battery reaches 10%.";
        return;
      }
      if (!selectedBaselineCell) {
        baselinePathPreview.textContent = "Select a start mission pad.";
        return;
      }
      const path = baselinePathFromSelection();
      path.forEach((cell, index) => {
        cell.classList.add(index === 0 ? "selected" : "in-path");
        const label = cell.querySelector(".path-index");
        if (label) label.textContent = index === 0 ? "start" : String(index + 1);
      });
      const pads = path.map((cell) => cell.dataset.pad);
      baselineStartPadInput.value = selectedBaselineCell.dataset.pad;
      baselineStartColInput.value = selectedBaselineCell.dataset.col;
      baselineStartRowInput.value = selectedBaselineCell.dataset.row;
      const targetRow = baselineTargetRow();
      if (targetRow < 0 || targetRow > topMissionRow) {
        baselinePathPreview.textContent = "Select the opposite edge row so the 250cm path stays on mission pads.";
        return;
      }
      baselinePathPreview.textContent = `250cm path: ${pads.join(" -> ")}`;
    }

    baselineCells.forEach((cell) => {
      cell.addEventListener("click", () => {
        selectedBaselineCell = cell;
        renderBaselinePath();
      });
    });

    baselineArrowButtons.forEach((button) => {
      button.addEventListener("click", () => {
        setBaselineDirection(button.dataset.baselineDirection);
        renderBaselinePath();
      });
    });
    if (baselineModeInput) {
      baselineModeInput.addEventListener("change", () => {
        if (baselineModeInput.value === "tail_forward_250") {
          setBaselineDirection("down");
        } else if (baselineModeInput.value === "head_forward_250") {
          setBaselineDirection("up");
        }
        renderBaselinePath();
      });
      renderBaselinePath();
    }

    const runStatus = document.getElementById("runStatus");
    const runMessage = document.getElementById("runMessage");
    const runTerminal = document.getElementById("runTerminal");
    const runTerminalTitle = document.getElementById("runTerminalTitle");
    const runTerminalMeta = document.getElementById("runTerminalMeta");
    const batteryMonitor = document.getElementById("batteryMonitor");
    const batteryWindowLabel = document.getElementById("batteryWindowLabel");
    const takeoffModal = document.getElementById("takeoffModal");
    const takeoffModalTitle = document.getElementById("takeoffModalTitle");
    const takeoffModalBody = document.getElementById("takeoffModalBody");
    const confirmTakeoffButton = document.getElementById("confirmTakeoffButton");
    const skipDischargeButton = document.getElementById("skipDischargeButton");
    const closeTakeoffModalButton = document.getElementById("closeTakeoffModalButton");
    const refreshRunButton = document.getElementById("refreshRunButton");
    const stopRunButton = document.getElementById("stopRunButton");
    const baselineForm = document.getElementById("baselineForm");
    let takeoffModalShownForRun = null;

    function batteryWindowFromState(state) {
      return (state.live_batteries && state.live_batteries.window) || { high: {{ battery_window.high }}, low: {{ battery_window.low }} };
    }

    function renderBatteryMonitor(liveBatteries) {
      const data = liveBatteries || {};
      const windowInfo = data.window || { high: {{ battery_window.high }}, low: {{ battery_window.low }} };
      batteryWindowLabel.textContent = `${windowInfo.high}% - ${windowInfo.low}%`;
      const drones = data.drones || [];
      batteryMonitor.innerHTML = "";
      if (!drones.length) {
        const empty = document.createElement("div");
        empty.className = "small";
        empty.textContent = "Prepare an experiment to see live drone battery levels.";
        batteryMonitor.appendChild(empty);
        return;
      }
      drones.forEach((drone) => {
        const card = document.createElement("div");
        card.className = "battery-card";
        card.dataset.band = drone.band || "unknown";

        const top = document.createElement("div");
        top.className = "battery-top";
        const label = document.createElement("span");
        label.className = "small";
        label.textContent = `Drone ${drone.drone_number || drone.takeoff_order || "-"} · ${drone.battery_id || "-"}`;
        const value = document.createElement("strong");
        value.textContent = drone.battery_percent_label || "--";
        top.append(label, value);

        const track = document.createElement("div");
        track.className = "battery-track";
        const fill = document.createElement("div");
        fill.className = "battery-fill";
        fill.style.width = `${drone.window_progress || 0}%`;
        track.appendChild(fill);

        const meta = document.createElement("div");
        meta.className = "small";
        const bandText = {
          above: "above experiment window",
          window: "inside experiment window",
          below: "below experiment window",
          unknown: "waiting for reading",
        }[drone.band || "unknown"];
        meta.textContent = `${bandText}${drone.phase ? " · " + drone.phase : ""}`;

        card.append(top, track, meta);
        batteryMonitor.appendChild(card);
      });
    }

    function renderRunState(state) {
      runStatus.textContent = state.status || "idle";
      runMessage.textContent = state.message || "No experiment is running.";
      runTerminalTitle.textContent = `${state.script || "data_collector.py"}${state.experiment_id ? " --experiment-id " + state.experiment_id : ""}`;
      runTerminalMeta.textContent = state.started_at || "idle";
      const lines = state.output || [];
      runTerminal.textContent = lines.length ? lines.join("\\n") : "$ waiting for experiment command...";
      runTerminal.scrollTop = runTerminal.scrollHeight;
      renderBatteryMonitor(state.live_batteries);
      const windowInfo = batteryWindowFromState(state);
      if (baselineForm && !["preflight", "ready_for_takeoff", "ready_for_discharge", "running", "discharging"].includes(state.status || "")) {
        const submitButton = baselineForm.querySelector('button[type="submit"]');
        if (submitButton) submitButton.disabled = false;
      }
      const promptKey = `${state.run_id || "none"}:${state.prompt_action || "takeoff"}`;
      if (state.ready_for_takeoff && takeoffModalShownForRun !== promptKey) {
        if (state.prompt_action === "discharge" || state.ready_for_discharge) {
          takeoffModalTitle.textContent = "Battery discharge needed";
          takeoffModalBody.textContent = state.script === "single_drone_baseline.py"
            ? `This baseline drone is above ${windowInfo.high}%. Start a hover discharge first, or skip discharge and continue to takeoff.`
            : `One or more drones are above ${windowInfo.high}%. You can hover-discharge them first, or skip discharge and continue to the formal takeoff confirmation.`;
          confirmTakeoffButton.textContent = "Start hover discharge";
          skipDischargeButton.hidden = false;
        } else {
          takeoffModalTitle.textContent = "Ready for takeoff";
          takeoffModalBody.textContent = state.script === "single_drone_baseline.py"
            ? "The single-drone baseline script is waiting at the takeoff prompt. Confirm only when the flight area is clear."
            : `All drones are inside the ${windowInfo.low}%-${windowInfo.high}% battery window and the script is waiting at the takeoff prompt. Confirm only when the flight area is clear.`;
          confirmTakeoffButton.textContent = "Confirm takeoff";
          skipDischargeButton.hidden = true;
        }
        takeoffModal.classList.add("visible");
        takeoffModalShownForRun = promptKey;
      }
    }

    async function fetchRunState() {
      const response = await fetch("{{ url_for('experiment_run_status') }}");
      const payload = await response.json();
      if (payload.state) renderRunState(payload.state);
      return payload.state;
    }

    async function postRunAction(url) {
      const response = await fetch(url, { method: "POST" });
      const payload = await response.json();
      if (payload.state) renderRunState(payload.state);
      if (!payload.ok) alert(payload.error || "Experiment action failed.");
      return payload;
    }

    document.querySelectorAll(".start-experiment-button").forEach((button) => {
      button.addEventListener("click", async () => {
        button.disabled = true;
        const payload = await postRunAction(button.dataset.startUrl);
        if (!payload.ok) button.disabled = false;
      });
    });
    if (baselineForm) {
      baselineForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!isHoverBaselineMode() && !baselineStartPadInput.value) {
          alert("Choose a start mission pad for the baseline test.");
          return;
        }
        if (!isHoverBaselineMode()) {
          const targetRow = baselineTargetRow();
          if (targetRow === null || targetRow < 0 || targetRow > topMissionRow) {
            alert("Choose a start pad on the opposite edge row so the single-drone baseline can fly exactly 250cm between mission pads.");
            return;
          }
        }
        const submitButton = baselineForm.querySelector('button[type="submit"]');
        submitButton.disabled = true;
        const response = await fetch(baselineForm.action, {
          method: "POST",
          body: new FormData(baselineForm),
        });
        const payload = await response.json();
        if (payload.state) renderRunState(payload.state);
        if (!payload.ok) {
          alert(payload.error || "Baseline start failed.");
          submitButton.disabled = false;
        }
      });
    }

    refreshRunButton.addEventListener("click", fetchRunState);
    stopRunButton.addEventListener("click", () => {
      if (confirm("Stop the current experiment process?")) {
        postRunAction("{{ url_for('stop_experiment_run') }}");
      }
    });
    confirmTakeoffButton.addEventListener("click", async () => {
      takeoffModal.classList.remove("visible");
      await postRunAction("{{ url_for('confirm_takeoff') }}");
    });
    skipDischargeButton.addEventListener("click", async () => {
      takeoffModal.classList.remove("visible");
      await postRunAction("{{ url_for('skip_discharge_hover') }}");
    });
    closeTakeoffModalButton.addEventListener("click", () => {
      takeoffModal.classList.remove("visible");
    });

    document.querySelectorAll("[data-filter-group]").forEach((group) => {
      const allBox = group.querySelector("[data-filter-all]");
      const optionBoxes = Array.from(group.querySelectorAll("[data-filter-option]"));
      const syncAllState = () => {
        if (!allBox) return;
        allBox.checked = !optionBoxes.some((box) => box.checked);
      };
      if (allBox) {
        allBox.addEventListener("change", () => {
          if (allBox.checked) {
            optionBoxes.forEach((box) => { box.checked = false; });
          } else {
            syncAllState();
          }
        });
      }
      optionBoxes.forEach((box) => {
        box.addEventListener("change", () => {
          if (box.checked && allBox) {
            allBox.checked = false;
          }
          syncAllState();
        });
      });
      syncAllState();
    });
    document.querySelectorAll("[data-toggle-filter-groups]").forEach((button) => {
      button.addEventListener("click", () => {
        const panel = button.closest(".filter-panel");
        const groups = Array.from(panel.querySelectorAll("[data-filter-group]"));
        const shouldOpen = groups.some((group) => !group.open);
        groups.forEach((group) => { group.open = shouldOpen; });
        button.textContent = shouldOpen ? "Collapse all" : "Expand all";
      });
    });

    setInterval(fetchRunState, 1500);
    fetchRunState();
  </script>
</body>
</html>
"""


FILE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ path.name }}</title>
  <style>
    :root { --bg:#f6f7f9; --panel:#fff; --line:#d8dde6; --text:#1d2430; --muted:#657286; --brand:#216c5f; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0; }
    header { background:#fff; border-bottom:1px solid var(--line); padding:18px 24px; display:flex; justify-content:space-between; gap:12px; align-items:center; }
    a { color:var(--brand); text-decoration:none; }
    main { padding:18px; max-width:1400px; margin:0 auto; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    .head { padding:16px 18px; border-bottom:1px solid var(--line); }
    .body { padding:16px 18px; overflow:auto; }
    h1 { font-size:20px; margin:0; overflow-wrap:anywhere; }
    .small { color:var(--muted); font-size:13px; line-height:1.5; }
    .badges { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
    .badge { border:1px solid var(--line); border-radius:999px; padding:3px 8px; color:var(--muted); font-size:12px; background:#fbfcfd; }
    table { border-collapse:collapse; width:100%; font-size:12px; min-width:900px; }
    th, td { border:1px solid var(--line); padding:7px 8px; text-align:left; white-space:nowrap; }
    th { background:#f0f3f5; position:sticky; top:0; z-index:1; }
    img { max-width:100%; height:auto; display:block; border:1px solid var(--line); border-radius:8px; background:#fff; }
    .actions { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    button { border:0; border-radius:8px; background:var(--brand); color:#fff; padding:9px 12px; font:inherit; font-weight:650; cursor:pointer; }
    button.danger { background:#a23b3b; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{{ path.name }}</h1>
      <div class="small">{{ path }}</div>
    </div>
    <div class="actions">
      <a href="{{ url_for('raw_file', relpath=relpath) }}">Open raw</a>
      <form method="post"
            action="{{ url_for('delete_file', relpath=relpath) }}"
            onsubmit="return confirm('Delete {{ relpath }}? This cannot be undone.');">
        <input type="hidden" name="next" value="{{ url_for('index') }}">
        <button class="danger" type="submit">Delete file</button>
      </form>
      <a href="{{ url_for('index') }}">Back</a>
    </div>
  </header>
  <main>
    <section>
      <div class="head">
        {% if file_type == 'csv' %}
          <div class="small">{{ analysis.row_count }} rows · {{ analysis.trial_count }} trials</div>
          <div class="badges">
            {% for formation in analysis.formations %}<span class="badge">{{ formation }}</span>{% endfor %}
            {% for drone in analysis.drones %}<span class="badge">{{ drone }}</span>{% endfor %}
          </div>
        {% else %}
          <div class="small">Generated image preview</div>
        {% endif %}
      </div>
      <div class="body">
        {% if file_type == 'image' %}
          <img src="{{ url_for('raw_file', relpath=relpath) }}" alt="{{ path.name }}">
        {% elif file_type == 'csv' %}
          <table>
            <thead>
              <tr>{% for col in header %}<th>{{ col }}</th>{% endfor %}</tr>
            </thead>
            <tbody>
              {% for row in rows %}
                <tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
              {% endfor %}
            </tbody>
          </table>
        {% endif %}
      </div>
    </section>
  </main>
</body>
</html>
"""


EXPERIMENT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ experiment.experiment_id }}</title>
  <style>
    :root { --bg:#f6f7f9; --panel:#fff; --line:#d8dde6; --text:#1d2430; --muted:#657286; --brand:#216c5f; --soft:#edf4f2; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0; }
    header { background:#fff; border-bottom:1px solid var(--line); padding:18px 24px; display:flex; justify-content:space-between; gap:12px; align-items:center; }
    a { color:var(--brand); text-decoration:none; }
    main { display:grid; grid-template-columns:minmax(360px,0.9fr) minmax(440px,1.1fr); gap:18px; padding:18px; max-width:1500px; margin:0 auto; }
    section { background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; min-width:0; }
    .head { padding:16px 18px; border-bottom:1px solid var(--line); }
    .body { padding:16px 18px; }
    h1,h2,h3 { margin:0; letter-spacing:0; }
    h1 { font-size:20px; }
    h2 { font-size:16px; }
    h3 { font-size:14px; }
    .small { color:var(--muted); font-size:13px; line-height:1.5; }
    .badge { display:inline-flex; border:1px solid var(--line); border-radius:999px; padding:3px 8px; color:var(--muted); font-size:12px; background:#fbfcfd; }
    .badge.outlier { border-color:#d49a90; color:#9d3d31; background:#fff4f1; }
    .mission-grid { display:grid; grid-template-columns:repeat(5, minmax(72px,1fr)); gap:8px; }
    .pad-cell { min-height:92px; border:1px solid var(--line); border-radius:8px; background:#fbfcfd; padding:8px; display:grid; gap:6px; opacity:.45; }
    .pad-cell.has-drone { opacity:1; background:#f3faf8; border-color:#8abeb2; }
    .pad-cell.selected { box-shadow: inset 0 0 0 2px var(--brand); }
    .pad-id { font-size:18px; font-weight:750; }
    .drone-link { display:block; color:var(--text); }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .metric { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfd; }
    .metric strong { display:block; font-size:20px; }
    .file-list { display:grid; gap:8px; }
    .file-row { border:1px solid var(--line); border-radius:8px; padding:10px; display:flex; justify-content:space-between; gap:10px; align-items:center; }
    .plots { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .plots img { width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; }
    button { border:0; border-radius:8px; background:var(--brand); color:#fff; padding:9px 12px; font:inherit; font-weight:650; cursor:pointer; }
    button.secondary { background:#657286; }
    @media (max-width: 980px) { main { grid-template-columns:1fr; } .plots { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{{ experiment.experiment_id }}</h1>
      <div class="small">
        {{ experiment.formation }} · {{ experiment.wind_direction }} · {{ experiment.wind_speed }} · distance {{ experiment.inter_drone_distance_cm or 50 }}cm
                {% if experiment.is_outlier %}<span class="badge outlier">outlier excluded from summary</span>{% endif %}
                {% for tag in experiment.tags or [] %}<span class="badge">{{ tag }}</span>{% endfor %}
      </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end;">
      <form method="post" action="{{ url_for('toggle_experiment_outlier', experiment_id=experiment.experiment_id) }}">
        <input type="hidden" name="next" value="{{ url_for('experiment_detail', experiment_id=experiment.experiment_id) }}">
        <input type="hidden" name="is_outlier" value="{{ '0' if experiment.is_outlier else '1' }}">
        <button class="secondary" type="submit">{{ 'Mark as normal' if experiment.is_outlier else 'Mark as outlier' }}</button>
      </form>
      <a href="{{ url_for('index') }}">Back</a>
    </div>
  </header>
  <main>
    <section>
      <div class="head">
        <h2>Mission Pad Drones</h2>
        <div class="small">Click a drone rectangle to inspect its battery and position archive.</div>
      </div>
      <div class="body">
        <div class="mission-grid" style="grid-template-columns:repeat({{ mission_pad_columns|length }}, minmax(72px,1fr));">
          {% for display_row in range(mission_pad_row_count - 1, -1, -1) %}
            {% for col in range(mission_pad_columns|length) %}
              {% set pad_id = mission_pad_columns[col][display_row] if display_row < mission_pad_columns[col]|length else '' %}
              {% set matched = namespace(drone=None) %}
              {% for drone in drone_cards %}
                {% if drone.grid_column|string == col|string and drone.grid_row|string == display_row|string %}
                  {% set matched.drone = drone %}
                {% endif %}
              {% endfor %}
              <div class="pad-cell {% if matched.drone %}has-drone{% endif %} {% if selected_drone and matched.drone and selected_drone.suffix == matched.drone.suffix %}selected{% endif %}">
                <div><span class="small">pad</span> <span class="pad-id">{{ pad_id }}</span></div>
                {% if matched.drone %}
                  <a class="drone-link" href="{{ url_for('experiment_detail', experiment_id=experiment.experiment_id, drone=matched.drone.suffix) }}">
                    <strong>{{ matched.drone.role }}</strong>
                    <div class="small">{{ matched.drone.ip }}</div>
                    <div class="small">battery {{ display_battery_id(matched.drone) or '-' }}</div>
                    <div class="small">order {{ matched.drone.takeoff_order }}</div>
                  </a>
                {% endif %}
              </div>
            {% endfor %}
          {% endfor %}
        </div>
      </div>
    </section>

    <section>
      <div class="head">
        <h2>{% if selected_drone %}{{ selected_drone.role }}{% else %}Experiment{% endif %} Details</h2>
      </div>
      <div class="body">
        {% if selected_drone %}
          <div class="grid">
            <div class="metric"><span class="small">Battery start</span><strong>{{ selected_drone.battery_row.battery_hover_start or '-' }}%</strong></div>
            <div class="metric"><span class="small">Battery end</span><strong>{{ selected_drone.battery_row.battery_hover_end or '-' }}%</strong></div>
            <div class="metric"><span class="small">Battery drop</span><strong>{{ selected_drone.battery_row.battery_drop or '-' }}%</strong></div>
            <div class="metric"><span class="small">Battery ID</span><strong>{{ display_battery_id(selected_drone) or '-' }}</strong></div>
            <div class="metric"><span class="small">Mission pad</span><strong>{{ selected_drone.mission_pad }}</strong></div>
          </div>
          <div style="height:12px;"></div>
          <div class="file-list">
            {% if selected_drone.coordination %}<div class="file-row"><a href="{{ url_for('file_detail', relpath=selected_drone.coordination) }}">Position data</a><span class="badge">coordination</span></div>{% endif %}
            {% if selected_drone.battery %}<div class="file-row"><a href="{{ url_for('file_detail', relpath=selected_drone.battery) }}">Battery data</a><span class="badge">battery</span></div>{% endif %}
          </div>
          {% if selected_drone.plot %}
            <div style="height:12px;"></div>
            <img style="width:100%;border:1px solid var(--line);border-radius:8px;" src="{{ url_for('raw_file', relpath=selected_drone.plot) }}" alt="{{ selected_drone.role }} plot">
          {% endif %}
        {% endif %}
      </div>
    </section>

    <section>
      <div class="head">
        <h2>All Five Drones</h2>
        <form method="post" action="{{ url_for('generate_plots', relpath=experiment.experiment_id ~ '/placeholder.csv') }}" style="margin-top:10px;">
          <button type="submit">Generate plots</button>
        </form>
      </div>
      <div class="body">
        <div class="grid">
          {% for row in archive.all_battery %}
            <div class="metric">
              <span class="small">{{ row.drone_name }} · {{ row.drone_ip }}{% if row.battery_id %} · {{ row.battery_id }}{% endif %}</span>
              <strong>{{ row.battery_drop or '-' }}%</strong>
              <div class="small">{{ row.battery_hover_start or '-' }}% -> {{ row.battery_hover_end or '-' }}%</div>
            </div>
          {% else %}
            <div class="small">No battery summary yet.</div>
          {% endfor %}
        </div>
      </div>
    </section>

    <section>
      <div class="head"><h2>Generated Images</h2></div>
      <div class="body">
        <div class="plots">
          {% for plot in archive.plots %}
            <a href="{{ url_for('file_detail', relpath=plot.relpath) }}"><img src="{{ url_for('raw_file', relpath=plot.relpath) }}" alt="{{ plot.name }}"></a>
          {% else %}
            <div class="small">No plots yet. Click Generate plots after data collection.</div>
          {% endfor %}
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""


CONDITION_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ condition.condition_key }}</title>
  <style>
    :root { --bg:#f6f7f9; --panel:#fff; --line:#d8dde6; --text:#1d2430; --muted:#657286; --brand:#216c5f; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0; }
    header { background:#fff; border-bottom:1px solid var(--line); padding:18px 24px; display:flex; justify-content:space-between; gap:12px; align-items:center; }
    main { padding:18px; max-width:1400px; margin:0 auto; display:grid; gap:18px; }
    section { background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    .head { padding:16px 18px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
    .body { padding:16px 18px; }
    h1,h2,h3 { margin:0; letter-spacing:0; }
    h1 { font-size:20px; }
    h2 { font-size:16px; }
    a { color:var(--brand); text-decoration:none; }
    .small { color:var(--muted); font-size:13px; line-height:1.5; }
    .trial-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }
    .trial-card { border:1px solid var(--line); border-radius:8px; padding:12px; background:#fbfcfd; display:grid; gap:8px; }
    .trial-card.outlier { border-color:#d49a90; background:#fff7f5; }
    .badge { display:inline-flex; border:1px solid var(--line); border-radius:999px; padding:3px 8px; color:var(--muted); font-size:12px; background:#fbfcfd; width:max-content; }
    .badge.outlier { border-color:#d49a90; color:#9d3d31; background:#fff4f1; }
    .plots { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
    .plots img { width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; }
    button { border:0; border-radius:8px; background:var(--brand); color:#fff; padding:9px 12px; font:inherit; font-weight:650; cursor:pointer; }
    @media (max-width: 900px) { .plots { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{{ condition.condition_key }}</h1>
      <div class="small">
        {{ condition.formation }} · {{ condition.wind_direction }} · {{ condition.wind_speed }} · distance {{ condition.inter_drone_distance_cm or 50 }}cm
        · {{ condition.included_trial_count }} used / {{ condition.trial_count }} trial(s)
        {% if condition.outlier_count %}· {{ condition.outlier_count }} outlier excluded{% endif %}
      </div>
    </div>
    <a href="{{ url_for('index') }}">Back</a>
  </header>
  <main>
    <section>
      <div class="head">
        <div>
          <h2>Trial Layer</h2>
          <div class="small">Each trial contains five-drone overall data plus per-drone data.</div>
        </div>
        <form method="post" action="{{ url_for('generate_condition_plots', condition_key=condition.condition_key) }}">
          <button type="submit">Generate condition summary plots</button>
        </form>
      </div>
      <div class="body">
        <div class="trial-grid">
          {% for trial in condition.trials %}
            <a class="trial-card {% if trial.is_outlier %}outlier{% endif %}" href="{{ url_for('experiment_detail', experiment_id=trial.experiment_id) }}">
              <h3>{{ trial.experiment_id }}</h3>
              {% if trial.is_outlier %}<span class="badge outlier">outlier excluded from summary</span>{% endif %}
              {% for tag in trial.tags or [] %}<span class="badge">{{ tag }}</span>{% endfor %}
              <div class="small">Trial {{ trial.short_id or trial.experiment_id.rsplit('_', 1)[-1] }}</div>
              <div class="small">{{ trial.status }} · {{ trial.created_at }}</div>
            </a>
          {% endfor %}
        </div>
      </div>
    </section>

    <section>
      <div class="head">
        <h2>Condition Summary Layer</h2>
        <div class="small">Generated from non-outlier trials in this condition group.</div>
      </div>
      <div class="body">
        <div class="plots">
          {% for plot in archive.plots %}
            <a href="{{ url_for('file_detail', relpath=plot.relpath) }}"><img src="{{ url_for('raw_file', relpath=plot.relpath) }}" alt="{{ plot.name }}"></a>
          {% else %}
            <div class="small">No summary plots yet. Generate them after at least one trial is archived; three trials is the intended minimum.</div>
          {% endfor %}
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
