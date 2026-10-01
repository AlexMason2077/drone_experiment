"""Read-only raw replay; add audit and availability to this turn's export."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from process_50cm_stage_curves import (
    ROOT, MODEL, GRAIN, NAMES, BatteryNormalizer, sha, write_json, observed_hover_frames,
)
from recover_50cm_wind_stage_selection import DEFAULT_OUT, select_real_windows, replay_wls, LABEL_CORRECTION


def main(out):
    manifest = json.loads((out / "manifest.json").read_text())
    for rel, digest in manifest["source_hashes"].items():
        assert sha(ROOT / rel) == digest, rel
    normalizer = BatteryNormalizer.load(MODEL)
    wind = out / "wind_tunnel"
    rates = pd.read_csv(wind / "observed_run_stage_rates.csv")
    raw_errors, norm_errors, samples = [], [], []
    for source, group in rates.groupby("source_file", sort=False):
        raw = pd.read_csv(ROOT / source, low_memory=False)
        for drone, rows in group.groupby("drone_name", sort=False):
            g = raw[raw.drone_name.eq(drone)]
            new_selection = rows.selection_method.eq("real_hover_and_telemetry_verified_fault_windows").all()
            frame, _ = (select_real_windows(g, "_merged_" in source) if new_selection else
                         observed_hover_frames(g, "_merged_" in source))
            for r in rows.itertuples():
                i = NAMES.index(r.stage)
                own = normalizer.curve_for(r.battery_id, drone)
                up, lo = own.boundaries[i:i+2]
                part = frame[frame.raw_soc.between(lo, up)]
                actual = replay_wls(part)
                raw_errors.append(abs(actual-r.raw_rate_pp_min))
                norm_errors.append(abs(actual/own.rates_pp_min[i]*normalizer.reference.rates_pp_min[i]-r.normalized_rate_pp_min))
                samples.append(dict(condition_id=r.condition_id, position=r.position, stage=r.stage,
                    source_file=source, raw_replay_error_pp_min=raw_errors[-1],
                    normalized_replay_error_pp_min=norm_errors[-1],
                    actual_stage_sample_rows=len(part)))
    assert len(samples) == len(rates) and max(raw_errors) < 1e-8 and max(norm_errors) < 1e-8
    pd.DataFrame(samples).to_csv(out / "wind_tunnel/independent_all_stage_replay.csv", index=False)
    w = pd.read_csv(wind / "condition_position_stage_means.csv")
    f = pd.read_csv(out / "forward_250cm/condition_position_medium_means.csv")
    f["stage"] = "Medium"
    availability = w[GRAIN+["condition_id", "observed_run_count"]].rename(
        columns={"observed_run_count": "wind_observed_run_count"}).merge(
        f[GRAIN+["observed_run_count"]].rename(columns={"observed_run_count": "forward_observed_run_count"}),
        on=GRAIN, how="outer", validate="one_to_one")
    availability["both_sources_have_observed_rate"] = (availability.wind_observed_run_count.fillna(0).gt(0)
        & availability.forward_observed_run_count.fillna(0).gt(0))
    availability["cross_protocol_average_applied"] = False
    availability.to_csv(out / "source_stage_availability.csv", index=False)

    exp = "wind_tunnel_echalon_50_head_lv1_004"
    path = f"database/{exp}/{exp}_20260908_210136_all_coordination.csv"
    historical = subprocess.check_output(["git", "show", "055cfeef:"+path], cwd=ROOT)
    old = pd.read_csv(io.BytesIO(historical))
    current = pd.read_csv(ROOT / path)
    unchanged_telemetry = old[["drone_name", "elapsed_time", "battery"]].equals(
        current[["drone_name", "elapsed_time", "battery"]])
    assert unchanged_telemetry
    recovery = dict(experiment_id=exp, inspected_git_commit="055cfeef",
        earliest_version_rows=len(old), earliest_version_duration_s=float(old.elapsed_time.max()),
        current_rows=len(current), current_duration_s=float(current.elapsed_time.max()),
        telemetry_clock_and_SOC_identical_to_earliest_version=True,
        all_current_phases=current.phase.unique().tolist(),
        full_hover_copy_found_in_local_git_history=False,
        conclusion="Earliest saved version is already a short takeoff record; no evidence that a complete hover curve was later overwritten in local Git.")
    write_json(out / "echelon_004_local_history_check.json", recovery)
    images = sorted((wind / "figures").glob("*.png"))
    assert len(images) == 25
    for path in images:
        im = Image.open(path)
        assert im.size == (3300, 2040)
        im.verify()
    coverage = pd.read_csv(wind / "condition_coverage.csv")
    assert (coverage.analysis_status.eq("collected_has_stage_curves").sum() == 25
        and coverage.analysis_status.eq("excluded_safety_condition").sum() == 5)
    check = dict(status="passed_independent_replay_and_export_checks", all_observed_stage_rates_replayed=len(samples),
        raw_rate_replay_max_error_pp_min=max(raw_errors), normalized_rate_replay_max_error_pp_min=max(norm_errors),
        figures_3300x2040_verified=len(images), collection_grid_25_safe_plus_5_excluded=True,
        all_25_authored_figures_visually_inspected_in_contact_sheet=True,
        echelon_figure_full_size_visually_inspected=True,
        cross_protocol_average_applied=False, baseline_sha256=normalizer.sha256,
        stage_cells_with_both_observed_sources=int(availability.both_sources_have_observed_rate.sum()),
        simulated_samples_used=0, user_confirmed_wind_level_correction=LABEL_CORRECTION)
    write_json(out / "independent_validation.json", check)
    manifest["supplementary_audit_code"] = dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__))
    manifest["source_hashes"][str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    manifest["independent_all_stage_replay"] = check
    write_json(out / "manifest.json", manifest)
    readme = (out / "README.md").read_text()
    readme += f"\n## 独立复核\n\n全部{len(samples)}条实测阶段率已经从各自原CSV重新选择并独立最小二乘复算，原始率最大误差{max(raw_errors):.2g} pp/min。source_stage_availability.csv列出两个来源同条件同位置同阶段的观测支撑；共有{check['stage_cells_with_both_observed_sources']}个有双来源，未平均。echelon_004_local_history_check.json记录004在首次Git保存时已为6.062秒的起飞记录，时钟与SOC和当前相同。25张PNG均核验为3300×2040并检查图形排版。\n"
    (out / "README.md").write_text(readme, encoding="utf-8")
    checksums = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "file_checksums.json"}
    write_json(out / "file_checksums.json", checksums)
    # Refresh only the derived archive authored in this turn, never the prior package.
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, str(Path(out.name) / p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, digest in checksums.items():
            import hashlib
            assert hashlib.sha256(z.read(out.name+"/"+name)).hexdigest() == digest
    for rel, digest in manifest["source_hashes"].items():
        assert sha(ROOT / rel) == digest, rel
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    main(parser.parse_args().output.resolve())
