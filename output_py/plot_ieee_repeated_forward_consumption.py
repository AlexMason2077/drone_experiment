"""Fig. 4(b): observed forward-only consumption and independent run fits.

The three real repeated trials share condition, position, physical battery,
and the Medium SOC stage. Observed initial SOC values are 66%, 74%, and 74%.
No horizontal offsets or synthetic runs.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battery_normalization import BatteryNormalizer
from build_trajectory_cleaning_segments import prepare_run_groups, centered_rolling_median
from build_forward_motion_segments import build_forward_mask, durations_and_distance
from build_forward_discharge_rate_modeling_table import through_origin_fit

CASES = [("echalon_50_tail_lv1_new_002", "20260622_202226"),
         ("echalon_50_tail_lv1_new_003", "20260622_211505"),
         ("echalon_50_tail_lv1_new_004", "20260622_233125")]
DRONE = "drone_2"
MODEL = ROOT / "analysis_results/battery_normalization_extended_v3_20260909/model.json"
QC = ROOT / "db_copy_for_cleaning/_cleaning_admin/trajectory_qc"
OUT = ROOT / "analysis_outputs/methodology_figures"
STEM = "fig_forward_discharge_rate_repeated_runs"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    normalizer = BatteryNormalizer.load(MODEL)
    trajectory = pd.read_csv(QC / "trajectory_drone_segments.csv", dtype={"run_id": str})
    forward_summary = pd.read_csv(QC / "forward_motion_drone_segments.csv", dtype={"run_id": str})
    derived, summaries, source_records = [], [], []
    for n, (experiment, run_id) in enumerate(CASES, 1):
        source = ROOT / "db_copy_for_cleaning" / experiment / f"{experiment}_{run_id}_all_coordination.csv"
        raw = pd.read_csv(source, low_memory=False)
        item = prepare_run_groups(raw)[DRONE]
        group, times = item["group"], item["times"]
        seg = trajectory.loc[trajectory.run_id.eq(run_id) & trajectory.drone_name.eq(DRONE)].iloc[0]
        old = forward_summary.loc[forward_summary.run_id.eq(run_id) & forward_summary.drone_name.eq(DRONE)].iloc[0]
        assert set(group.battery_id.dropna()) == {"B10"}
        curve = normalizer.curve_for("B10", DRONE)
        progress = centered_rolling_median(item["relative"] @ item["run_direction"], 11)
        progress = progress * float(seg.trajectory_distance_calibration_factor)
        moving, inside, _ = build_forward_mask(times, progress, group.phase.fillna("").to_numpy(),
            float(seg.motion_onset_sec), float(seg.selected_250cm_end_sec), threshold_cm_s=2.0)
        forward_s, excluded_s, _ = durations_and_distance(times, np.maximum.accumulate(progress), moving, inside)
        assert np.isclose(forward_s, old.forward_movement_sec)
        battery = group.battery.to_numpy(float)
        dt = np.diff(times)
        valid = inside[:-1] & inside[1:] & np.isfinite(dt) & (dt >= 0)
        selected = valid & moving[:-1]
        ids = np.flatnonzero(selected)
        raw_drop = battery[:-1] - battery[1:]
        assert np.all(raw_drop[valid] >= 0)
        assert np.all((battery[inside] > curve.boundaries[2]) & (battery[inside] <= curve.boundaries[1]))
        reference = 75.0
        clock, consumption, actual_drop = [0.0], [0.0], [0.0]
        evidence = []
        for j in ids:
            h = curve.equivalent_seconds(battery[j], battery[j+1])
            reference = normalizer.reference.advance(reference, h)
            clock.append(clock[-1] + dt[j])
            consumption.append(75.0 - reference)
            actual_drop.append(actual_drop[-1] + raw_drop[j])
            evidence.append(dict(run=n, experiment=experiment, run_id=run_id, drone=DRONE,
                source_start_s=times[j], source_end_s=times[j+1], measured_soc_start=battery[j],
                measured_soc_end=battery[j+1], forward_time_s=clock[-1],
                cumulative_forward_reported_drop_pp=actual_drop[-1],
                battery_consumption_pp=consumption[-1]))
        x, y = np.array(clock), np.array(consumption)
        rate, r2 = through_origin_fit(x, y)
        assert np.isclose(x[-1], forward_s)
        scale = normalizer.reference.rates_pp_min[1] / curve.rates_pp_min[1]
        assert reference > normalizer.reference.boundaries[2]
        assert np.allclose(y, np.array(actual_drop)*scale)
        total_drop = float(raw_drop[valid].sum())
        excluded_drop = float(raw_drop[valid & ~moving[:-1]].sum())
        assert np.isclose(actual_drop[-1]+excluded_drop, total_drop)
        start_soc = float(battery[np.flatnonzero(inside)[0]])
        summaries.append(dict(run=n, experiment=experiment, run_id=run_id, drone=DRONE, battery="B10",
            measured_start_soc=start_soc, recorded_window_s=forward_s+excluded_s,
            forward_time_s=forward_s, excluded_nonforward_time_s=excluded_s,
            total_reported_drop_pp=total_drop, excluded_nonforward_reported_drop_pp=excluded_drop,
            retained_forward_reported_drop_pp=actual_drop[-1], battery_consumption_pp=y[-1],
            discharge_rate_pp_per_min=rate, fit_r_squared=r2))
        derived.append((x, y))
        pd.DataFrame(evidence).to_csv(OUT / f"{STEM}_run{n}_intervals.csv", index=False)
        source_records.append(dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))

    pd.DataFrame(summaries).to_csv(OUT / f"{STEM}_rates.csv", index=False)
    mean_rate = float(np.mean([v["discharge_rate_pp_per_min"] for v in summaries]))
    # Reconcile all displayed runs with the existing rate cohort.
    established = pd.read_csv(ROOT / "analysis_results/medium_forward_bideal_v3_20260909/run_drone_rates.csv", dtype={"run_id": str})
    for s in summaries:
        row = established.loc[established.run_id.eq(s["run_id"]) & established.drone_name.eq(DRONE)]
        assert len(row) == 1 and row.iloc[0].status == "included"
        assert np.isclose(row.iloc[0].discharge_rate_Bideal_pp_per_min, s["discharge_rate_pp_per_min"])

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                        "font.size": 8.5, "axes.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = ["#246B9A", "#26836A", "#D68A1C"]
    fig, ax = plt.subplots(figsize=(7.6, 3.45))
    fig.subplots_adjust(left=.10, right=.98, bottom=.18, top=.82)
    handles = []
    for (x, y), summary, color in zip(derived, summaries, colors):
        # Solid steps preserve the processed observations; only the averaged
        # discharge rate is displayed as a straight dashed line.
        ax.step(x, y, where="post", color=color, lw=1.7, alpha=.95, zorder=2)
        handles.append(Line2D([], [], color=color, lw=1.65,
            label=f"Run {summary['run']} (start {summary['measured_start_soc']:.0f}%)"))
    longest = max(x[-1] for x, _ in derived)
    ax.plot([0, longest], [0, mean_rate*longest/60], color="#9AA2AA", lw=1.4,
            ls=(0, (6, 3.5)), zorder=4)
    handles.append(Line2D([], [], color="#9AA2AA", lw=1.4, ls=(0,(6,3.5)), label="Mean discharge rate"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55,.985), ncol=4,
               frameon=False, columnspacing=1.1, handlelength=2.4, fontsize=7.8)
    fig.text(.015,.977,"(b)",ha="left",va="top",fontsize=10,fontweight="bold")
    x_upper = np.ceil(longest/5)*5
    y_upper = np.ceil((max(float(y.max()) for _, y in derived)+1.1)/2)*2
    ax.set(xlim=(0,x_upper),ylim=(0,y_upper),xlabel="Forward-flight time (s)",ylabel="Battery consumption (%)")
    ax.set_xticks(np.arange(0,x_upper+1,5))
    ax.set_yticks(np.arange(0,y_upper+1,2))
    ax.grid(color="#D9DEE3",lw=.55,alpha=.7)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    ax.text(.985,.96,f"Mean discharge rate = {mean_rate:.2f} percentage points/min",transform=ax.transAxes,
            ha="right",va="top",fontsize=8.2, bbox=dict(facecolor="white",edgecolor="none",alpha=.85,pad=1.5))
    fig.savefig(OUT / f"{STEM}.png",dpi=400)
    fig.savefig(OUT / f"{STEM}.pdf")
    plt.close(fig)
    caption = ("Estimation of the discharge rate from three repeated flights under the same condition, with reported starting SOC values of 66%, 74%, and 74%. "
               "Solid curves show battery consumption after normalization and exclusion of non-forward intervals. "
               "A straight line is fitted separately to each run. The single grey dashed line shows the mean of the three fitted slopes.")
    metadata = dict(condition="Echelon, 50 cm, tailwind Level 1, position 2", sources=source_records,
        normalizer=str(MODEL), normalizer_sha256=normalizer.sha256,
        processing="Retain the existing pipeline's forward sample-to-sample intervals. Remove both time and reported SOC changes in non-forward intervals. Integrate retained SOC drops through the individual battery curve and reference curve. All curves start at zero cumulative forward time and zero consumption.",
        normalization_reference_start=75, measured_start_soc=[s["measured_start_soc"] for s in summaries],
        fit="Separate origin-constrained least-squares fits per run, as in the current Medium-rate pipeline; equal arithmetic mean of run slopes, not a pooled fit.",
        mean_rate_pp_per_min=mean_rate, runs=summaries, caption=caption,
        notes=["Initial SOC labels are measured at the recorded analysis window, not an observed take-off event.",
               "These are independent repeated experiments, aligned to the start of retained forward time; not simultaneous flights.",
               "All three displayed runs are included in the existing Medium-rate cohort. Their independently recomputed slopes match the existing per-run table. No training rate tables are changed.",
               "A near-75% candidate with starting SOC of 72% was not selected because its swarm run was not in the training cohort (other drones had flat SOC traces).",
               "The third accepted repeat starts at 66%, not approximately 75%; initial SOC is disclosed without altering the data. The selected condition has three non-flat accepted records with more continuous SOC updates.",
               "The mean rate describes the three displayed examples; no broader representativeness is claimed.",
               "Integer SOC updates may lag the underlying consumption. Phase attribution follows the existing processing convention."])
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (OUT / f"{STEM}_caption.txt").write_text(caption+"\n")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Mean discharge rate: {mean_rate:.6f} pp/min")


if __name__ == "__main__":
    main()
