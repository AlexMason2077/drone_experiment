"""Build the reproducible notebook for the configuration-energy analysis."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs" / "configuration_energy_analysis"
NOTEBOOK = OUT / "configuration_energy_analysis.ipynb"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Configuration Energy Comparison by Wind Condition

## tl;dr

The analysis compares **formation + inter-drone spacing** only within the same wind direction and wind level. The five-drone run is the analytical unit. Lower values are better.

- **Head wind, Level 1:** Front · 75 cm is the only comparatively stable leader.
- **Head wind, Level 2:** Vee · 75 cm leads, but the result is provisional.
- **Side wind, Level 1:** Front · 50 cm and Front · 75 cm are effectively tied; there is no clear winner.
- **Side wind, Level 2:** Front · 75 cm leads provisionally.
- **Tail wind, Level 1:** Front · 50 cm leads narrowly; Diamond · 75 cm and Echelon · 50 cm remain close.
- **Tail wind, Level 2:** No defensible winner yet. Column · 50 cm leads after wait correction but has only one run, while the raw and uncorrected metrics favor Echelon · 50 cm.

These are descriptive rankings, not causal claims or tests of statistical significance."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

Each valid run contains five drones that successfully completed the selected first 250 cm of forward flight. Test/preparation, no-wind, marked-outlier, incomplete, and primary-range-ineligible runs are excluded. Coordinate-frame jumps previously reviewed by the experimenter are retained after calibration.

For each drone, the reported battery drop is divided by that physical battery's fitted 75%–40% hover discharge rate. Only trajectory-confirmed stationary waiting is subtracted. The result is reported as **adjusted hover-equivalent seconds per drone per 250 cm**. It is a battery-normalized SOC proxy, not literal motor-on time or directly measured electrical energy."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import os
import sys

import pandas as pd
from IPython.display import Image, display

ROOT = next(
    path
    for path in [Path.cwd(), *Path.cwd().parents]
    if (path / "output_py" / "analyze_configuration_energy.py").exists()
)
OUT = ROOT / "analysis_outputs" / "configuration_energy_analysis"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/drone_mpl")
sys.path.insert(0, str(ROOT / "output_py"))

# Recreate all metrics, rankings, and the static figure from the cleaned data.
from analyze_configuration_energy import main as run_analysis
run_analysis()"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """summary = json.loads((OUT / "analysis_summary.json").read_text())
calibration = pd.read_csv(OUT / "hover_battery_calibration_75_40.csv")
coverage = pd.read_csv(OUT / "configuration_condition_coverage.csv")

print(f"Primary runs: {summary['primary_run_count']}")
print(f"Drone-level rows: {summary['primary_drone_row_count']}")
print(f"Observed condition-configuration cells: {summary['configuration_cell_count']} of 60 possible")
display(calibration.round(4))
display(coverage)"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """leaders = pd.read_csv(OUT / "condition_configuration_leaders.csv")
leader_columns = [
    "condition", "leading_configuration", "leader_adjusted_mean_sec", "leader_run_count",
    "runner_up_configuration", "gap_to_runner_up_pct", "bootstrap_winner_probability",
    "variant_agreement_count_of_5", "evidence_label",
]
display(leaders[leader_columns].round(3))
display(Image(filename=str(OUT / "configuration_rankings_by_condition.png")))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Robustness Checks

The primary ranking is compared with four alternatives: median instead of mean, raw reported SOC drop, normalized but uncorrected SOC drop, and an analysis excluding drone 5 (which also removes sensitivity to the B15-to-B12 replacement on that drone)."""
        ),
        nbf.v4.new_code_cell(
            """robustness_columns = [
    "condition", "leading_configuration", "adjusted_median_winner", "raw_drop_winner",
    "uncorrected_normalized_winner", "without_drone_5_winner",
    "variant_agreement_count_of_5", "evidence_label",
]
display(leaders[robustness_columns])"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. **Use Front · 75 cm as the current best-supported result only for head wind at Level 1.** It wins all five metric variants and has the largest credible margin among replicated cells.
2. **Treat the other five condition-level leaders as hypotheses for confirmation.** Side wind Level 1 is essentially tied, and Tail wind Level 2 lacks replication for its corrected leader.
3. **Do not interpret the score as direct energy in joules.** Tello reports integer battery percentage, producing zero-drop segments in 36 of 840 drone records.
4. **Complete the missing cells and rebalance replication before formal inference.** The current dataset covers 55 of 60 possible cells, with 1–6 runs per observed cell.
5. **Prioritize new runs for Tail wind Level 2 Column · 50 cm, Side wind Level 1 Front · 50/75 cm, and the closest alternatives in Head wind Level 2 and Side wind Level 2.**"""
        ),
    ]
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
