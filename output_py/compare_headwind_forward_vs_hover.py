"""Compare single-drone head-wind forward flight with hover at matched SOC."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "db_copy_for_cleaning" / "baselines"
OUTPUT = ROOT / "output_graph"
BATTERIES = ["B10", "B11", "B13", "B14", "B15"]
SOC_HIGH = 75.0
SOC_LOW = 40.0
COLORS = {"hover": "#1f77b4", "head_forward": "#d55e00"}


def crossing_time(df: pd.DataFrame, threshold: float) -> float | None:
    below = df[df.battery <= threshold]
    if below.empty:
        return None
    idx = below.index[0]
    if idx == df.index[0]:
        return float(df.loc[idx, "time_sec"])
    pos = df.index.get_loc(idx)
    prev_idx = df.index[pos - 1]
    left = df.loc[prev_idx]
    right = df.loc[idx]
    if right.battery == left.battery:
        return float(right.time_sec)
    ratio = (threshold - left.battery) / (right.battery - left.battery)
    return float(left.time_sec + ratio * (right.time_sec - left.time_sec))


def load_hover_segments() -> pd.DataFrame:
    summary = pd.read_csv(OUTPUT / "hover_battery_runs_summary.csv")
    summary = summary[(summary.status == "included") & summary.battery_id.isin(BATTERIES)]
    rows = []
    for item in summary.itertuples(index=False):
        path = ROOT / item.source_file
        df = pd.read_csv(path, low_memory=False)
        for col in ["node_elapsed_time", "hover_elapsed_time", "elapsed_time", "battery"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        time_candidates = [c for c in ["node_elapsed_time", "hover_elapsed_time", "elapsed_time"] if c in df]
        if not time_candidates:
            continue
        time_col = max(time_candidates, key=lambda c: df[c].max(skipna=True))
        x = df[[time_col, "battery"]].rename(columns={time_col: "time_sec"}).dropna()
        x = x[x.battery.between(0, 100)].sort_values("time_sec").drop_duplicates("time_sec")
        t_high = crossing_time(x, SOC_HIGH)
        t_low = crossing_time(x, SOC_LOW)
        if t_high is None or t_low is None or t_low <= t_high:
            continue
        duration = t_low - t_high
        rows.append({
            "mode": "hover", "battery_id": item.battery_id, "drone_name": item.drone_name,
            "run_id": item.run_id, "duration_sec": duration,
            "battery_drop": SOC_HIGH - SOC_LOW,
            "drop_rate_pct_per_min": (SOC_HIGH - SOC_LOW) / duration * 60,
            "status": "included_matched_soc", "source_file": item.source_file,
        })
    return pd.DataFrame(rows)


def load_head_forward_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    included, excluded = [], []
    for metadata_path in sorted(BASELINES.glob("**/*_metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("mode") != "head_forward_250" or metadata.get("battery_id") not in BATTERIES:
            continue
        prefix = metadata_path.name.removesuffix("_metadata.json")
        files = list(metadata_path.parent.glob(f"{prefix}*_all_battery.csv"))
        base = {
            "mode": "head_forward", "battery_id": metadata.get("battery_id"),
            "drone_name": metadata.get("drone_name"), "run_id": metadata.get("run_id"),
            "source_file": str(files[0].relative_to(ROOT)) if files else "",
        }
        if not files:
            excluded.append({**base, "status": "excluded_missing_summary", "reason": "missing all_battery.csv"})
            continue
        df = pd.read_csv(files[0], low_memory=False)
        if df.empty:
            excluded.append({**base, "status": "excluded_empty_summary", "reason": "empty all_battery.csv"})
            continue
        row = df.iloc[0]
        values = {name: pd.to_numeric(row.get(name), errors="coerce") for name in [
            "node_duration_sec", "battery_hover_start", "battery_hover_end", "battery_drop",
        ]}
        reasons = []
        if any(pd.isna(v) for v in values.values()): reasons.append("missing numeric field")
        elif not 20 <= values["node_duration_sec"] <= 90: reasons.append("duration outside 20-90 s")
        elif abs(values["battery_hover_start"] - values["battery_hover_end"] - values["battery_drop"]) > 0.01:
            reasons.append("battery drop mismatch")
        elif values["battery_drop"] < 0: reasons.append("negative battery drop")
        elif not (SOC_LOW <= values["battery_hover_end"] <= values["battery_hover_start"] <= SOC_HIGH):
            reasons.append(f"not fully inside {SOC_HIGH:g}-{SOC_LOW:g}% SOC window")
        if reasons:
            excluded.append({**base, **values, "status": "excluded", "reason": "; ".join(reasons)})
            continue
        duration = float(values["node_duration_sec"])
        drop = float(values["battery_drop"])
        included.append({
            **base, **values, "duration_sec": duration, "battery_drop": drop,
            "drop_rate_pct_per_min": drop / duration * 60,
            "status": "included_matched_soc",
        })
    return pd.DataFrame(included), pd.DataFrame(excluded)


def weighted_summary(all_runs: pd.DataFrame) -> pd.DataFrame:
    return all_runs.groupby(["battery_id", "mode"], as_index=False).apply(
        lambda x: pd.Series({
            "run_count": len(x),
            "total_duration_sec": x.duration_sec.sum(),
            "total_battery_drop": x.battery_drop.sum(),
            "weighted_drop_rate_pct_per_min": x.battery_drop.sum() / x.duration_sec.sum() * 60,
            "median_run_rate": x.drop_rate_pct_per_min.median(),
            "q25_run_rate": x.drop_rate_pct_per_min.quantile(0.25),
            "q75_run_rate": x.drop_rate_pct_per_min.quantile(0.75),
        }),
        include_groups=False,
    ).reset_index(drop=True)


def plot_comparison(all_runs: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), dpi=180, gridspec_kw={"width_ratios": [1, 1.35]})
    x = np.arange(len(BATTERIES))
    rates = {
        mode: summary[summary["mode"] == mode].set_index("battery_id").reindex(BATTERIES).weighted_drop_rate_pct_per_min
        for mode in ["hover", "head_forward"]
    }
    ax = axes[0]
    ax.plot(x, rates["hover"], marker="o", linewidth=2.2, markersize=7, color=COLORS["hover"], label="Hover")
    ax.plot(x, rates["head_forward"], marker="o", linewidth=2.2, markersize=7, color=COLORS["head_forward"], label="Head-wind forward")
    for idx in x:
        ax.plot([idx, idx], [rates["hover"].iloc[idx], rates["head_forward"].iloc[idx]], color="#c7ccd1", zorder=0)
    ax.set_xticks(x, BATTERIES)
    ax.set_ylabel("Battery drop rate (% points/min)")
    ax.set_title("A. Matched-SOC weighted rates", loc="left", weight="bold")
    ax.legend(frameon=False)

    ax = axes[1]
    head = all_runs[all_runs["mode"] == "head_forward"]
    data = [head[head.battery_id == b].drop_rate_pct_per_min.values for b in BATTERIES]
    boxes = ax.boxplot(data, positions=x, widths=0.55, patch_artist=True, showfliers=False)
    for box in boxes["boxes"]:
        box.set(facecolor="#f6c7aa", edgecolor=COLORS["head_forward"], linewidth=1.2)
    for element in ["whiskers", "caps", "medians"]:
        for line in boxes[element]: line.set(color="#7c4b2c", linewidth=1.2)
    rng = np.random.default_rng(42)
    for idx, values in enumerate(data):
        jitter = rng.uniform(-0.16, 0.16, len(values))
        ax.scatter(idx + jitter, values, s=14, color=COLORS["head_forward"], alpha=0.45, edgecolors="none")
    ax.scatter(x, rates["hover"], marker="D", s=55, color=COLORS["hover"], label="Hover matched-SOC rate", zorder=4)
    ax.set_xticks(x, BATTERIES)
    ax.set_ylabel("Run-level battery drop rate (% points/min)")
    ax.set_title("B. Head-forward run variation", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.grid(axis="y", color="#d9dee3", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.suptitle("Single-drone battery consumption: hover vs head-wind forward flight", x=0.06, ha="left", fontsize=16, weight="bold")
    fig.text(0.06, 0.92, "Comparison restricted to the common 75%–40% battery range; source: db_copy_for_cleaning/baselines", color="#59636e", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hover = load_hover_segments()
    head, excluded = load_head_forward_runs()
    all_runs = pd.concat([hover, head], ignore_index=True, sort=False)
    summary = weighted_summary(all_runs)
    comparison = summary.pivot(index="battery_id", columns="mode", values="weighted_drop_rate_pct_per_min").reset_index()
    comparison["absolute_increase_pct_points_per_min"] = comparison.head_forward - comparison.hover
    comparison["relative_increase_percent"] = comparison.absolute_increase_pct_points_per_min / comparison.hover * 100
    all_runs.to_csv(OUTPUT / "headwind_forward_vs_hover_clean_runs.csv", index=False)
    excluded.to_csv(OUTPUT / "headwind_forward_excluded_runs.csv", index=False)
    summary.to_csv(OUTPUT / "headwind_forward_vs_hover_summary.csv", index=False)
    comparison.to_csv(OUTPUT / "headwind_forward_vs_hover_battery_comparison.csv", index=False)
    plot_comparison(all_runs, summary, OUTPUT / "headwind_forward_vs_hover_comparison.png")
    print(summary.to_string(index=False))
    print(f"Included hover segments: {len(hover)}")
    print(f"Included head-forward runs: {len(head)}")
    print(f"Excluded head-forward records: {len(excluded)}")


if __name__ == "__main__":
    main()
