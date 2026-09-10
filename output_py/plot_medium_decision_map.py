"""Paper figure from the frozen decision lookup; no rate or optimizer edits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis_results/medium_configuration_function_v1_20260909/decision_function.json"
FIGURES = ROOT / "output/figures"
PDF_DIR = ROOT / "output/pdf"
PALETTE = {"front": "#DCEAF3", "echalon": "#F1E4C4", "column": "#DFE8DC",
           "vee": "#E9DFF0", "diamond": "#F4DFD9"}
NAMES = {"front": "Front", "echalon": "Echelon", "column": "Column", "vee": "Vee", "diamond": "Diamond"}


def map_records():
    model = json.loads(SOURCE.read_text())
    rows = []
    for wind in ("head", "side", "tail"):
        for level in (1, 2):
            for pads in range(1, 6):
                d = model["decisions"][f"{pads}|{wind}|{level}"]
                c = d["configuration"]
                rows.append(dict(wind_direction=wind, wind_strength="Low" if level == 1 else "High",
                                 charging_pads=pads, formation=NAMES[c["formation"]],
                                 formation_key=c["formation"], spacing_cm=c["inter_drone_spacing_cm"],
                                 total_time_minutes=d["total_time_minutes"]))
    assert len(rows) == 30
    return rows


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "text.color": "#20252B"})
    rows = map_records()
    # At 180 mm width the figure fits a two-column paper with 9-11 pt labels.
    fig = plt.figure(figsize=(180 / 25.4, 111 / 25.4), facecolor="white")
    ax = fig.add_axes([0.018, 0.02, 0.964, 0.96])
    ax.set_xlim(-1.52, 5.10)
    ax.set_ylim(-0.60, 7.52)
    ax.axis("off")
    ax.text(1.79, 7.24, "Medium-stage configuration decision map", ha="center", va="center",
            fontsize=13, fontweight="bold")
    ax.text(1.79, 6.84, "Common Bideal SOC: 75%  |  Segment: 2.5 m in 25 s", ha="center", va="center",
            fontsize=9.5, color="#4C545D")
    ax.text(2.5, 6.36, "Available charging pads", ha="center", va="center", fontsize=10.5)
    ax.text(-1.40, 5.91, "Wind direction", ha="left", va="center", fontsize=9)
    ax.text(-0.34, 5.91, "Level", ha="center", va="center", fontsize=9)
    for j in range(5):
        ax.text(j + 0.5, 5.91, str(j + 1), ha="center", va="center", fontsize=11, fontweight="bold")
    height = .89
    top = 5.59
    for i, (wind, level) in enumerate((w, l) for w in ("head", "side", "tail") for l in ("Low", "High")):
        y = top - (i + 1) * height
        ax.text(-.34, y + height / 2, level, ha="center", va="center", fontsize=10)
        for j in range(5):
            r = rows[i * 5 + j]
            assert (r["wind_direction"], r["wind_strength"], r["charging_pads"]) == (wind, level, j + 1)
            ax.add_patch(Rectangle((j, y), 1, height, facecolor=PALETTE[r["formation_key"]],
                                   edgecolor="white", linewidth=2))
            ax.text(j + .5, y + .57, r["formation"], ha="center", va="center", fontsize=10.5, fontweight="bold")
            ax.text(j + .5, y + .29, f"{r['spacing_cm']} cm", ha="center", va="center", fontsize=9.5)
    for group, wind in enumerate(("Headwind", "Sidewind", "Tailwind")):
        centre = top - (group * 2 + 1) * height
        ax.text(-1.40, centre, wind, ha="left", va="center", fontsize=10.5)
    for boundary in (0, 2, 4, 6):
        y = top - boundary * height
        ax.plot([-1.42, 5], [y, y], color="#75818B" if boundary in (0, 6) else "#B4BDC4", lw=.6)
    ax.text(1.79, -.22, "Each cell gives the minimum-total-time formation and inter-drone spacing.",
            ha="center", va="center", fontsize=9, color="#4C545D")
    stem = "medium_configuration_decision_map"
    for extension in ("svg", "png", "pdf"):
        output = (PDF_DIR if extension == "pdf" else FIGURES) / f"{stem}.{extension}"
        fig.savefig(output, dpi=600, facecolor="white", metadata={"Creator": "Matplotlib"} if extension == "pdf" else None)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIGURES / f"{stem}_data.csv", index=False)
    caption = (
        "Medium-stage configuration decision map for five drones. For each wind direction, wind level and number of available charging pads, "
        "the selected formation and inter-drone spacing minimize the fixed segment flight time plus the time until all five drones complete charging. "
        "Before applying the decision function, the empirical, battery-normalized Medium discharge profiles are aligned to a common Bideal SOC of 75% "
        "for a 2.5 m segment lasting 25 s. Low and high denote the two experimental wind settings. "
        "The comparison uses the observed, non-excluded configurations and the stated charging model; all available pads are identical and initially free. "
        "Formation is encoded by both text and colour."
    )
    (FIGURES / f"{stem}_caption.md").write_text(caption + "\n\n"
        "Use at full two-column width (180 mm). The PDF contains vector graphics and embedded text; the SVG retains editable text. "
        "PNG: 600 dpi. This map is the fixed-baseline decision function, not a neural-network prediction or an unequal-SOC policy.\n")
    (FIGURES / f"{stem}.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=\\textwidth]{medium_configuration_decision_map.pdf}\n"
        "\\caption{" + caption.replace("75%", "75\\%") + "}\n"
        "\\label{fig:medium-decision-map}\n\\end{figure*}\n")
    manifest = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    cells=len(rows), width_mm=180, height_mm=111, png_dpi=600,
                    encodings={"columns": "available charging pads, 1-5", "rows": "wind direction x low/high wind",
                               "colour": "categorical formation", "direct_text": "formation and spacing in cm"},
                    files=[str(PDF_DIR/f"{stem}.pdf"), str(FIGURES/f"{stem}.svg"), str(FIGURES/f"{stem}.png")])
    (FIGURES / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
