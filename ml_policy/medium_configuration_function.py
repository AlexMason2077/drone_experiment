"""Three-input, read-only Medium configuration decision function.

SOC normalization and all scoring are PREPROCESSING in
output_py/build_medium_configuration_function.py. Runtime is a frozen lookup;
it does not accept SOC, translate curves, import a simulator, or control drones.
The fixed baseline is five equivalent Bideal batteries at 75%, a 250 cm / 25 s
segment, and the charging assumptions documented in the generated manifest.
"""
from __future__ import annotations

import argparse
import copy
import json
import operator
from functools import lru_cache
from pathlib import Path


RESULT_PATH = (Path(__file__).resolve().parents[1] / "analysis_results" /
               "medium_configuration_function_v1_20260909" / "decision_function.json")


def _integer(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer in {low}..{high}, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer in {low}..{high}") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _input_key(charging_pad_availability, wind_direction, wind_level):
    pads = _integer(charging_pad_availability, "charging_pad_availability", 1, 5)
    level = _integer(wind_level, "wind_level", 1, 2)
    if not isinstance(wind_direction, str):
        raise ValueError("wind_direction must be head, tail or side")
    wind = wind_direction.strip().lower().replace(" ", "").replace("_", "")
    wind = {"headwind": "head", "tailwind": "tail", "sidewind": "side"}.get(wind, wind)
    if wind not in {"head", "tail", "side"}:
        raise ValueError("wind_direction must be head, tail or side")
    return f"{pads}|{wind}|{level}"


@lru_cache(maxsize=2)
def _load(path, stamp):
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("schema_version") != 1 or model.get("kind") != "medium_configuration_lookup":
        raise ValueError("Unsupported Medium decision function artifact")
    expected = {f"{k}|{w}|{level}" for k in range(1, 6) for w in ("head", "side", "tail") for level in (1, 2)}
    if set(model["decisions"]) != expected:
        raise ValueError("Medium decision artifact does not contain all 30 supported input states")
    return model


def explain_medium_configuration(charging_pad_availability, wind_direction, wind_level):
    """Return the chosen configuration, computed times, gap and evidence caveats."""
    key = _input_key(charging_pad_availability, wind_direction, wind_level)
    stat = RESULT_PATH.stat()
    model = _load(RESULT_PATH, (stat.st_mtime_ns, stat.st_size))
    return copy.deepcopy(model["decisions"][key])


def select_medium_configuration(charging_pad_availability, wind_direction, wind_level):
    """F(k, wind, level) -> {formation, inter_drone_spacing_cm, position_assignment}.

    Returns the computed minimum-time configuration among available, non-excluded
    real-data candidates, NOT a guarantee of statistically significant dominance
    or the best configuration for arbitrary unequal Medium SOCs.
    """
    return explain_medium_configuration(charging_pad_availability, wind_direction, wind_level)["configuration"]


F = select_medium_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pads", type=int, required=True)
    parser.add_argument("--wind", required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    function = explain_medium_configuration if args.explain else select_medium_configuration
    print(json.dumps(function(args.pads, args.wind, args.level), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
