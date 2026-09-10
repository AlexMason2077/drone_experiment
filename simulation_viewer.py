"""Read-only browser for the export selected by simulation_data/CURRENT.json.

Chart contract (existing Flask/PNG surface, not a new analytical dashboard):
* SOC trend: five integer-step traces, raw logger simulation time, through each
  drone's first 20% sample. Shows stochastic discharge; not measured telemetry.
* Duration comparison: five zero-based bars, simulation start to first 20% sample.
* Stage comparison: grouped High/Medium/Low bars, raw and Bideal pp/min separately.
  Uses exported stage summaries; does not refit or impose common SOC boundaries.
Palette: explicit blue/gold/orange/olive/pink drone identities, plus line styles;
stage bars use explicit blue/gold/olive and hatching. PNGs are generated on demand
outside both the source export and real experiment registry. No flight imports.
"""

import csv
import gzip
import hashlib
import io
import json
import math
import re
import threading
from functools import lru_cache
from pathlib import Path

from flask import Blueprint, abort, redirect, render_template, request, send_file, url_for


PLOT_VERSION = "2"
PLOTS = {
    "battery": "Battery SOC — simulated",
    "duration": "Time to 20% — simulated",
    "stage_rates": "High / Medium / Low discharge rates — simulated",
}
STAGES = ("high", "medium", "low")
COLORS = ("#2878A5", "#B18A19", "#C76B31", "#76853B", "#BC6288")
PLOT_LOCK = threading.Lock()


class SimulationDataError(ValueError):
    pass


def contained_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise SimulationDataError("Invalid simulation file path.")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise SimulationDataError("Simulation file must stay inside its export directory.")
    return result


def signature(path):
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def is_true(value):
    return str(value).lower() == "true"


@lru_cache(maxsize=4)
def read_manifest(path, stamp):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    runs = {}
    for row in rows:
        run_id = row.get("experiment_id", "")
        if not re.fullmatch(r"SIM_[A-Za-z0-9_-]+", run_id) or not is_true(row.get("is_simulated")):
            raise SimulationDataError("Only explicitly simulated runs can appear in this browser.")
        if run_id in runs:
            raise SimulationDataError("Duplicate simulation ID in the active manifest.")
        relative = row.get("file", "")
        source = contained_file(path.parent, relative)
        if not relative.startswith("runs/") or not source.name.endswith(".csv.gz"):
            raise SimulationDataError("Expected a compressed simulation run CSV.")
        for field in ("condition_id", "formation", "spacing_cm", "wind_direction", "wind_level", "seed"):
            if not row.get(field):
                raise SimulationDataError(f"Simulation manifest is missing {field}.")
        row["rows"] = int(row["rows"])
        row["duration_s"] = float(row["duration_s"])
        runs[run_id] = row
    return runs


def current_catalog(simulation_root):
    pointer = simulation_root / "CURRENT.json"
    if not pointer.exists():
        return None, {}
    meta = json.loads(pointer.read_text(encoding="utf-8"))
    root = contained_file(simulation_root, meta.get("path"))
    manifest = contained_file(root, "run_manifest.csv")
    return root, read_manifest(manifest, signature(manifest))


@lru_cache(maxsize=8)
def read_run(path, stamp, run_id):
    """Read only one requested run; retain step changes, not millions of samples."""
    drones, preview, count = {}, [], 0
    with gzip.open(path, "rt", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        required = {"experiment_id", "is_simulated", "drone_name", "battery_id", "elapsed_time", "battery"}
        if not required.issubset(header):
            raise SimulationDataError("Simulation CSV is missing required columns.")
        for row in reader:
            count += 1
            if count <= 15:
                preview.append(row)
            if row["experiment_id"] != run_id or not is_true(row["is_simulated"]):
                raise SimulationDataError("CSV provenance does not match the simulated run.")
            name = row["drone_name"]
            if name not in {f"drone_{i}" for i in range(1, 6)}:
                raise SimulationDataError("Unexpected drone identity in simulation CSV.")
            soc, seconds = float(row["battery"]), float(row["elapsed_time"])
            if not (math.isfinite(seconds) and seconds >= 0 and soc.is_integer() and 20 <= soc <= 100):
                raise SimulationDataError("Invalid integer SOC or simulation time.")
            drone = drones.setdefault(name, {
                "name": name, "battery_id": row["battery_id"], "mission_pad": row.get("mission_pad", ""),
                "target_x": row.get("target_x", ""), "target_y": row.get("target_y", ""),
                "target_z": row.get("target_z", ""), "times": [], "soc": [], "end_s": None,
                "last_time": -1.0,
            })
            if seconds < drone["last_time"] or row["battery_id"] != drone["battery_id"]:
                raise SimulationDataError("Run time or battery assignment is inconsistent.")
            drone["last_time"] = seconds
            if drone["end_s"] is not None:
                continue
            if not drone["soc"] or drone["soc"][-1] != soc:
                drone["times"].append(seconds)
                drone["soc"].append(int(soc))
            if soc == 20:
                drone["end_s"] = seconds
    if len(drones) != 5 or any(d["end_s"] is None for d in drones.values()):
        raise SimulationDataError("Expected five complete simulated curves ending at 20%.")
    return {"drones": [drones[k] for k in sorted(drones)], "header": header, "preview": preview, "count": count}


@lru_cache(maxsize=4)
def read_rates(path, stamp):
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if not is_true(row.get("is_simulated")):
                raise SimulationDataError("Stage rates must be explicitly simulated.")
            row["position"] = int(row["position"])
            if row["stage"] not in STAGES or row["position"] not in range(1, 6):
                raise SimulationDataError("Unexpected stage or position in stage rates.")
            for key in ("raw_rate_pp_min", "bideal_rate_pp_min", "duration_s", "soc_start", "soc_end"):
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise SimulationDataError("Stage rates contain non-finite values.")
            if min(row["raw_rate_pp_min"], row["bideal_rate_pp_min"], row["duration_s"]) <= 0:
                raise SimulationDataError("Stage durations and discharge rates must be positive.")
            result.setdefault(row["experiment_id"], []).append(row)
    return result


def render_plots(run, data, rates):
    """Use non-interactive figures, independently of flight and real plotting code."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    result = {}
    subtitle = (f"{run['formation']} · {run['spacing_cm']} cm · {run['wind_direction']} wind Lv{run['wind_level']}"
                f" · seed {run['seed']}")
    labels = [f"D{i} / {d['battery_id']}" for i, d in enumerate(data["drones"], 1)]

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E1E5E9", linewidth=.7)
        ax.set_axisbelow(True)

    def finish(fig, key, title, note):
        fig.suptitle(title + "  |  SIMULATED", x=.08, y=.97, ha="left", fontsize=16, color="#1D2430")
        fig.text(.08, .91, subtitle, fontsize=10, color="#526174")
        fig.text(.08, .025, note, fontsize=9, color="#526174")
        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        result[key] = buf.getvalue()
        fig.clear()

    fig = Figure(figsize=(12, 6.6), dpi=140)
    ax = fig.add_axes([.08, .16, .88, .68])
    styles = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))
    for i, drone in enumerate(data["drones"]):
        ax.step(drone["times"], drone["soc"], where="post", color=COLORS[i],
                linestyle=styles[i], linewidth=1.9, label=labels[i])
    ax.set(xlabel="Elapsed simulation time from start (s)", ylabel="Battery SOC (%)", ylim=(15, 103), xlim=(0, None))
    ax.set_yticks([20, 40, 60, 80, 100])
    style(ax)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=10)
    finish(fig, "battery", "Battery discharge curves",
           "Source: simulated logger CSV (0.1 s sampling). Integer SOC; each trace stops at its first 20% sample.\n"
           "Initial plateau is retained; post-landing samples are not plotted. This is not measured flight telemetry.")

    fig = Figure(figsize=(12, 5.5), dpi=140)
    ax = fig.add_axes([.14, .18, .80, .64])
    times = [d["end_s"] for d in data["drones"]]
    bars = ax.barh(labels, times, color=COLORS, height=.6)
    ax.bar_label(bars, labels=[f"{v:.1f} s" for v in times], padding=6)
    ax.invert_yaxis()
    ax.set(xlabel="Time from simulation start to first 20% sample (s)", xlim=(0, max(times) * 1.17))
    style(ax)
    finish(fig, "duration", "Time to the landing threshold",
           "Individual 20% threshold times, not completion of landing. Takeoff/centering phases are model assumptions.")

    fig = Figure(figsize=(12, 8), dpi=140)
    lookup = {(r["position"], r["stage"]): r for r in rates}
    for panel, (field, label) in enumerate((("raw_rate_pp_min", "Raw battery"), ("bideal_rate_pp_min", "Bideal-normalized"))):
        ax = fig.add_axes([.08, .56 if panel == 0 else .16, .88, .27])
        for j, stage in enumerate(STAGES):
            values = [lookup[(i, stage)][field] for i in range(1, 6)]
            ax.bar([i + (j - 1) * .24 for i in range(5)], values, width=.23,
                   color=COLORS[(0, 1, 3)[j]], edgecolor="#394452", linewidth=.4,
                   hatch=("", "//", "..")[j], label=stage.title())
        ax.set_xticks(range(5), labels)
        ax.set(ylabel="SOC percentage points / min", ylim=(0, None))
        ax.set_title(label, loc="left", fontsize=12, pad=8)
        style(ax)
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), borderaxespad=0,
                  ncol=3, frameon=False, fontsize=9)
        ax.set_ylim(0, max(row[field] for row in rates) * 1.12)
    finish(fig, "stage_rates", "Stage discharge rates",
           "Source: simulated_stage_rates_long.csv. Battery-specific stage boundaries; initial plateau excluded.\n"
           "Raw and normalized scales are separate. Exact stage windows appear in the table on the run page.")
    return result


def create_simulation_blueprint(base_dir):
    bp = Blueprint("simulations", __name__, url_prefix="/simulations", template_folder="templates")
    simulation_root = Path(base_dir) / "simulation_data"
    cache_root = Path(base_dir) / "analysis_results" / "simulation_app_plots"

    @bp.errorhandler(SimulationDataError)
    def invalid_data(error):
        return render_template("simulations.html", error=str(error), run=None, runs=[], total=0,
                               conditions=0, filters={}, options={}, page=1, pages=1), 503

    def catalog():
        try:
            return current_catalog(simulation_root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SimulationDataError(f"Cannot read the active simulation export: {exc}") from exc

    def selected_run(run_id):
        root, runs = catalog()
        if run_id not in runs:
            abort(404, description="Simulation is not in the active manifest.")
        return root, runs[run_id]

    def sources(root, run):
        try:
            raw = contained_file(root, run["file"])
            rates = contained_file(root, "simulated_stage_rates_long.csv")
            stamp = (PLOT_VERSION, root.as_posix(), run, signature(raw), signature(rates))
            token = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:20]
            return raw, rates, contained_file(cache_root, run["experiment_id"] + "/" + token)
        except OSError as exc:
            raise SimulationDataError("A source file for this simulation is missing or unreadable.") from exc

    def run_data(raw, rate_file, run):
        try:
            data = read_run(raw, signature(raw), run["experiment_id"])
            rates = read_rates(rate_file, signature(rate_file)).get(run["experiment_id"], [])
            if data["count"] != run["rows"]:
                raise SimulationDataError("CSV row count does not match the manifest.")
            if len(rates) != 15 or {(r["position"], r["stage"]) for r in rates} != {(i, s) for i in range(1, 6) for s in STAGES}:
                raise SimulationDataError("Expected exactly 15 position/stage summaries.")
            if any(r["battery_id"] != data["drones"][r["position"] - 1]["battery_id"] for r in rates):
                raise SimulationDataError("Battery assignments differ between the logger and stage summaries.")
            rates = sorted(rates, key=lambda r: (r["position"], STAGES.index(r["stage"])))
            return data, rates
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SimulationDataError(f"Cannot read simulated curves: {exc}") from exc

    @bp.get("")
    @bp.get("/")
    def index():
        root, catalog_runs = catalog()
        all_runs = list(catalog_runs.values())
        fields = ("formation", "spacing_cm", "wind_direction", "wind_level")
        filters = {key: request.args.get(key, "").strip() for key in fields}
        if filters["formation"] == "echalon":
            filters["formation"] = "echelon"
        options = {key: sorted({r[key] for r in all_runs}) for key in fields}
        query = request.args.get("q", "").strip()
        runs = [r for r in all_runs if all(not val or r[key] == val for key, val in filters.items())
                and query.lower() in r["experiment_id"].lower()]
        runs.sort(key=lambda r: r["experiment_id"])
        pages = max(1, (len(runs) + 23) // 24)
        page = min(max(request.args.get("page", 1, type=int), 1), pages)

        def page_link(number):
            return url_for("simulations.index", **filters, q=query, page=number)

        return render_template("simulations.html", run=None, runs=runs[(page - 1) * 24:page * 24],
                               total=len(all_runs), conditions=len({r["condition_id"] for r in all_runs}),
                               matched=len(runs), filters=filters, options=options, query=query,
                               page=page, pages=pages, previous=page_link(page - 1), next=page_link(page + 1),
                               export_name=root.name if root else "No active export", error=None)

    @bp.get("/<run_id>")
    def detail(run_id):
        root, run = selected_run(run_id)
        raw, rate_file, cache = sources(root, run)
        data, rates = run_data(raw, rate_file, run)
        plots = [{"key": key, "title": title} for key, title in PLOTS.items() if (cache / f"{key}.png").is_file()]
        return render_template("simulations.html", run=run, data=data, rates=rates, plots=plots,
                               export_name=root.name, error=None)

    @bp.post("/<run_id>/plots")
    def generate(run_id):
        root, run = selected_run(run_id)
        raw, rate_file, cache = sources(root, run)
        data, rates = run_data(raw, rate_file, run)
        with PLOT_LOCK:
            if not all((cache / f"{key}.png").is_file() for key in PLOTS):
                images = render_plots(run, data, rates)
                cache.mkdir(parents=True, exist_ok=True)
                for key, content in images.items():
                    temporary = cache / f"{key}.tmp"
                    temporary.write_bytes(content)
                    temporary.replace(cache / f"{key}.png")
        return redirect(url_for("simulations.detail", run_id=run_id) + "#plots")

    @bp.get("/<run_id>/plots/<kind>.png")
    def plot(run_id, kind):
        if kind not in PLOTS:
            abort(404)
        root, run = selected_run(run_id)
        _, _, cache = sources(root, run)
        image = cache / f"{kind}.png"
        if not image.is_file():
            abort(404, description="Generate this simulation's plots first.")
        return send_file(image, mimetype="image/png", max_age=0)

    @bp.get("/<run_id>/download")
    def download(run_id):
        root, run = selected_run(run_id)
        raw, _, _ = sources(root, run)
        return send_file(raw, mimetype="application/gzip", as_attachment=True, download_name=raw.name)

    return bp
