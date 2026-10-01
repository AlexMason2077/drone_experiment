# Wind-tunnel simulation work plan

## Latest user revision (supersedes previous coverage targets)

Only conditions without usable real flight data may retain simulations. Partial-stage real data also counts as an observed condition. The previous all-allowed expansion has been cancelled. All exports and the non-deliverable prototype were pruned recoverably: 24 observed conditions and the four legacy safety cells excluded, 32 missing conditions × three realizations retained. Current export is `simulation_data/wind_tunnel_empirical_v1_20260909/`, 96 runs; old duplicate export folders must not be combined. See `simulation_cleanup_audit.json` for exact removed files and Trash recovery locations. No real data was changed.

This is an independent, explicitly synthetic simulation task. It does not authorize changing recorded experiments, registering simulated flights as observed, flying aircraft, replacing the frozen Bideal model, publishing, or pushing to GitHub.

## Requested outcome

Use existing battery baselines and wind-tunnel observations to build the most defensible available stochastic discharge simulator across High, Medium and Low. Preserve the logger's original columns, integer SOC, realistic temporal structure and per-drone landing at 20%. Missing conditions must carry larger uncertainty and remain labelled simulated, not additional experimental repetitions.

## Defaults if no further reply

- Scope: wind-tunnel hover only, five formations, 50/75 cm, head/tail/side, Lv1/Lv2; start at 100%, finish each drone at 20%.
- Keep the existing 2.5 m Medium table as a separate domain reference; do not silently transplant moving-flight rates into hover.
- Generate three seeded synthetic realizations for missing/incompletely covered combinations, plus bounded validation examples; keep bulk CSV files compressed in a separate simulation folder.
- Preserve original logger columns. Unknown telemetry fields remain empty unless the user requests an explicitly assumption-labelled telemetry emulator. Add unambiguous simulation identifiers/provenance fields.
- Do not force leader-highest or closer-always-more-consumption rankings. Compare those hypotheses with evidence; preserve uncertainty and contradictory observations.

## Work sequence

1. Inventory original records, stage coverage, battery assignments, telemetry quality and geometry; freeze input hashes.
2. Extract observed SOC step dwell times and stage windows, retaining partial stages and first-drop timing. Exclude synthetic/merged records and do not bridge fault/landing/telemetry gaps.
3. Fit and compare a baseline-only model with regularized condition-dependent models. Keep all samples from a flight together; additionally hold out complete configurations to evaluate missing-condition transfer.
4. Calibrate stochastic effects and stage correlations from training residuals. Model integer telemetry steps separately from continuous latent SOC; make uncertainty model assumptions explicit.
5. Generate labelled simulations only after validation, with target-condition coverage and uncertainty classifications. Do not claim missing conditions are experimentally verified.
6. Run provenance, reproducibility, integer/range/monotonicity, landing, geometry and data-leakage tests. Produce a report and runnable companion notebook.

## Questions asked while user was awake

1. Confirm wind-tunnel-only scope versus additionally simulating 2.5 m forward flight.
2. Whether unmodelled telemetry should remain blank or receive explicitly assumption-based synthetic values.

Continue with the above safe defaults if unanswered; do not interrupt the user for further optional questions.

## Completed outcome

The user confirmed wind-tunnel-only scope and requested non-uniform randomness and post-generation comparison. Unknown sensor telemetry was left blank under the stated default; no additional questions were asked.

- Extracted 8,306 eligible observed one-percentage-point intervals from 50 flight files across 27 labelled conditions (including no-wind references).
- Locked six-condition holdout before deterministic model selection; evaluated four candidate predictors. Selected conditional tree: holdout matched-segment time MAPE 21.07%, median APE 9.53%. This is not a full-flight or causal validation.
- Additional 75-to-50 transfer stress test: MAPE 27.12%, four front conditions only.
- Initial stochastic prototype failed plausibility checks because very short flight fragments could dominate whole-flight variability. Preserved the prototype as non-deliverable; restricted noise donors and used explicit development-data nominal dwell envelopes.
- Final export: 159 missing/partial-condition realizations plus 3 reference examples, 162 files, 4,231,980 logger rows, 64,800 integer SOC events. About 39 MB compressed.
- Verified all exports, 165 unchanged input hashes, 401 independently traced source intervals, reproducible seeds, geometry tests and 2,430 independently reconciled simulated stage-rate rows.
- Inspected notebook comparison PNGs. Native report validator and render tool passed; native UI layout was not independently captured. Notebook cells ran sequentially in Python, not through a Jupyter kernel because nbformat/nbclient are unavailable.

Deliverable data: `simulation_data/wind_tunnel_empirical_v1_20260909/`.
Methods, report, notebook and QA: `analysis_results/wind_tunnel_simulator_20260909/`.

All exports remain explicitly synthetic and separate from the experiment registry. No flights, source-data changes, frozen-model replacement, Git push or publication occurred.
