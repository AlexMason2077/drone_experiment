"""Paper-facing explanations for the unchanged, frozen Medium lookup.

This read-only layer adds low/high wind terminology, numerical reasons and
literature-informed hypotheses. It does not preprocess SOC or control drones.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from ml_policy import medium_configuration_function as baseline


EXPLANATIONS_PATH = (Path(__file__).resolve().parents[1] / "analysis_results" /
                     "medium_decision_interpretability_20260909" / "explanations.json")


@lru_cache(maxsize=2)
def _load_checked(path, stamp, baseline_path, baseline_stamp):
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("version") != 1 or artifact.get("kind") != "medium_decision_interpretation":
        raise ValueError("Unsupported Medium explanation artifact")
    frozen_bytes = baseline_path.read_bytes()
    if hashlib.sha256(frozen_bytes).hexdigest() != artifact["source_sha256"]["decision_function.json"]:
        raise ValueError("Frozen decisions changed; rebuild explanations before using them")
    frozen = json.loads(frozen_bytes)
    expected = {f"{k}|{w}|{l}" for k in range(1, 6)
                for w in ("head", "side", "tail") for l in (1, 2)}
    if set(artifact["decisions"]) != expected or set(frozen["decisions"]) != expected:
        raise ValueError("Expected all 30 Medium decisions")
    for key, result in artifact["decisions"].items():
        original = frozen["decisions"][key]
        if (result["configuration"] != original["configuration"] or
                not math.isclose(result["total_time_minutes"], original["total_time_minutes"],
                                 rel_tol=0, abs_tol=1e-9)):
            raise ValueError(f"Explanation disagrees with frozen decision: {key}")
        if not result.get("data_explanation_en") or not result.get("aerodynamic_evidence_status"):
            raise ValueError(f"Explanation lacks its evidence boundary: {key}")
    return artifact, frozen["assumptions"]


def select_medium_with_explanation(charging_pad_availability, wind_direction, wind_strength):
    """Return configuration + time + reasons for low/high wind at fixed Medium SOC.

    The common 75% Bideal starting SOC and 250 cm / 25 s segment are already
    preprocessed. This is not an optimizer for arbitrary live SOC inputs.
    """
    if not isinstance(wind_strength, str) or wind_strength.strip().lower() not in ("low", "high"):
        raise ValueError("wind_strength must be 'low' or 'high'")
    strength = wind_strength.strip().lower()
    key = baseline._input_key(charging_pad_availability, wind_direction,
                              {"low": 1, "high": 2}[strength])
    stat, base_stat = EXPLANATIONS_PATH.stat(), baseline.RESULT_PATH.stat()
    artifact, assumptions = _load_checked(
        EXPLANATIONS_PATH, (stat.st_mtime_ns, stat.st_size), baseline.RESULT_PATH,
        (base_stat.st_mtime_ns, base_stat.st_size))
    result = copy.deepcopy(artifact["decisions"][key])
    result["baseline_assumptions"] = copy.deepcopy(assumptions)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pads", type=int, required=True)
    parser.add_argument("--wind", required=True)
    parser.add_argument("--wind-strength", choices=("low", "high"), required=True)
    args = parser.parse_args()
    print(json.dumps(select_medium_with_explanation(args.pads, args.wind, args.wind_strength),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
