"""Build the executed companion notebook for the forward-only analysis."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "configuration_energy_analysis"
NOTEBOOK = OUT / "configuration_energy_forward_only_analysis.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Forward-Movement-Only Configuration Energy Analysis

## tl;dr

The revised analysis removes **all detected in-flight time without forward progression**, rather than subtracting only long confirmed waits.

- **Head wind, Level 1:** Front · 75 cm remains the strongest result.
- **Head wind, Level 2:** Vee · 75 cm remains provisional.
- **Side wind, Level 1:** the leader changes from Front · 50 cm to Front · 75 cm, but the gap is small.
- **Side wind, Level 2:** the leader changes from Front · 75 cm to Echelon · 50 cm; this is provisional because the leader has two runs.
- **Tail wind, Level 1:** Front · 50 cm remains narrowly ahead, with no clear separation from Echelon · 50 cm.
- **Tail wind, Level 2:** Column · 50 cm remains a one-run result and is unresolved.

The score is an estimated forward-only SOC energy proxy, not current/voltage-derived electrical energy."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

### Key Assumptions

The first trajectory-observed 250 cm is retained for every drone. Forward movement is defined as smoothed forward-progress velocity of at least 2 cm/s. Classification is performed separately inside each logged `node_segment_i_of_n` phase from `data_collector.py`, so release waits and end-of-segment synchronization holds cannot be bridged across segment boundaries. Gaps up to 1.0 s are bridged only within the same segment phase to tolerate mission-pad coordinate quantization, and movement islands shorter than 0.5 s are removed.

The dataset does not contain voltage or current. Therefore, forward-only energy is estimated by converting the selected-window SOC drop to battery-specific 75%–40% hover-equivalent seconds and subtracting the hover-baseline energy for every detected non-forward second. Values are clipped at zero because reported SOC is integer-valued."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import os
import sys

import pandas as pd
from IPython.display import Image, display

ROOT = next(
    path for path in [Path.cwd(), *Path.cwd().parents]
    if (path / "output_py" / "analyze_configuration_forward_only.py").exists()
)
OUT = ROOT / "analysis_outputs" / "configuration_energy_analysis"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/drone_mpl")
sys.path.insert(0, str(ROOT / "output_py"))

from build_forward_motion_segments import main as segment_forward_motion
from analyze_configuration_forward_only import main as run_forward_only_analysis
from validate_configuration_forward_only import main as validate_forward_only_analysis

segment_forward_motion()
run_forward_only_analysis()
validate_forward_only_analysis()"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """summary = json.loads((OUT / "forward_only_analysis_summary.json").read_text())
validation = json.loads((OUT / "forward_only_validation_checks.json").read_text())
drone = pd.read_csv(OUT / "forward_only_primary_drone_metrics.csv")

print(f"Primary runs: {summary['primary_run_count']}")
print(f"Drone records: {summary['primary_drone_row_count']}")
print(f"Median detected forward movement: {summary['median_forward_movement_sec']:.2f} s")
print(f"Median removed non-forward time: {summary['median_removed_nonforward_sec']:.2f} s")
print(f"Validation assessment: {validation['assessment']}")
display(drone[[
    "forward_movement_sec", "in_flight_nonforward_sec", "forward_movement_fraction",
    "detected_forward_distance_cm", "forward_only_hover_equivalent_sec"
]].describe().round(3))"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """leaders = pd.read_csv(OUT / "forward_only_condition_configuration_leaders.csv")
changes = pd.read_csv(OUT / "forward_only_leader_change_comparison.csv")
display(leaders[[
    "condition", "leading_configuration", "leader_forward_only_mean_sec", "leader_run_count",
    "runner_up_configuration", "gap_to_runner_up_pct", "variant_agreement_count_of_5",
    "evidence_label",
]].round(3))
display(changes[[
    "condition", "previous_wait_corrected_leader", "leading_configuration",
    "leader_changed_after_full_hover_removal",
]])
display(Image(filename=str(OUT / "forward_only_configuration_rankings_by_condition.png")))"""
        ),
        nbf.v4.new_markdown_cell("## Robustness Checks"),
        nbf.v4.new_code_cell(
            """display(leaders[[
    "condition", "leading_configuration", "forward_only_median_winner", "threshold_1_winner",
    "threshold_4_winner", "without_drone_5_winner", "confirmed_wait_only_winner",
    "direct_event_forward_winner", "raw_drop_winner",
]])

threshold = pd.read_csv(
    ROOT / "db_copy_for_cleaning" / "_cleaning_admin" / "trajectory_qc" /
    "forward_motion_threshold_sensitivity.csv"
)
primary_keys = drone[["experiment_directory", "run_id"]].drop_duplicates()
threshold = threshold.merge(primary_keys, on=["experiment_directory", "run_id"], how="inner")
display(threshold.groupby("threshold_cm_s")["forward_movement_sec"].describe().round(3))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. Full non-forward removal materially changes the side-wind rankings, showing that formation-induced waiting was a real confounder.
2. Front · 75 cm under Head wind · Level 1 remains the only comparatively stable leader with at least three runs and agreement across all primary robustness variants.
3. Side wind · Level 2 Echelon · 50 cm is movement-segmentation robust but still provisional: it has two runs, and raw/less-strict metrics favor Front · 75 cm.
4. Tail wind · Level 2 remains unresolved because its leading cell has one run.
5. The revised result should be reported as a modeled forward-only energy proxy. Direct electrical isolation would require synchronized voltage/current measurement."""
        ),
    ]
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
