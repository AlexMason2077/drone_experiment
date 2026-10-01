# Medium configuration function — fixed preprocessed baseline

`F(charging_pad_availability, wind_direction, wind_level) -> configuration`

SOC alignment is preprocessing, NOT a runtime input or operation. This is an exact finite lookup, not a neural network.

Source: analysis_results/medium_forward_bideal_v3_with_b15_20260909. 55 real-data configuration rows, 275 scored candidates across 30 input states.

## Decision map

Times below are minutes for the 25-second flight plus whole-swarm charging completion.

| Wind | Level | 1 pad | 2 pads | 3 pads | 4 pads | 5 pads |
|---|---:|---|---|---|---|---|
| head | 1 | front 75 cm / 322.080 min | front 75 cm / 190.766 min | front 75 cm / 128.800 min | front 75 cm / 127.071 min | echalon 75 cm / 65.883 min |
| head | 2 | echalon 75 cm / 324.108 min | front 50 cm / 193.159 min | echalon 50 cm / 129.435 min | front 50 cm / 127.923 min | echalon 75 cm / 65.851 min |
| side | 1 | front 50 cm / 325.051 min | front 50 cm / 194.013 min | front 50 cm / 129.682 min | diamond 75 cm / 129.066 min | column 75 cm / 66.564 min |
| side | 2 | front 50 cm / 324.088 min | echalon 50 cm / 193.531 min | front 50 cm / 129.842 min | echalon 50 cm / 128.629 min | front 50 cm / 65.322 min |
| tail | 1 | vee 50 cm / 325.860 min | vee 50 cm / 193.628 min | diamond 75 cm / 130.515 min | vee 50 cm / 128.307 min | diamond 75 cm / 66.272 min |
| tail | 2 | column 50 cm / 323.332 min | column 50 cm / 191.996 min | column 50 cm / 128.778 min | column 50 cm / 127.449 min | echalon 50 cm / 66.210 min |

## Assumptions and limits

- **reference_soc_preprocessing**: 75
- **Bideal_middle_band**: [52.0, 82.0]
- **flight_distance_cm**: 250
- **flight_duration_seconds**: 25.0
- **batteries**: five equivalent Bideal batteries; not five physical batteries at merely equal displayed percentages
- **position_assignment**: identity retained; permutations are time-equivalent under equal SOC and identical charging assumptions
- **charging_model**: existing exponential model
- **charging_target_soc**: 99.0
- **zero_to_target_minutes**: 90.0
- **charging_pads**: 1..5 identical pads, all available on arrival, one drone per pad at a time, non-preemptive charging
- **charging_completion**: minimum time until every drone has completed charging; do not sum individual waiting times again
- **ground_wait**: no battery drain while waiting on the ground
- **switching**: not included; configuration is chosen before this segment
- **objective**: single standardized flight segment + subsequent whole-swarm charging completion; not a global multi-segment optimum
- **evidence**: unchanged equal-run mean Medium discharge rates; source short, partial, flat, and low-count warnings retained; no new filters
- **source_coverage**: observed real forward-flight candidates only; no simulated or invented missing configurations
- **total_time**: computed from empirical discharge rates and an assumed charging model, not directly measured total mission time

- Explicit collision exclusions: [('head', 2, 'column', 50), ('side', 2, 'column', 50), ('side', 2, 'diamond', 50), ('tail', 2, 'diamond', 50)]
- Missing safe real-data configurations: [('side', 1, 'column', 50)]
- Equal-run averaging occurs before nonlinear charging evaluation. This is a standardized mean-rate comparison, not the mean of measured trial charging times.
- A numerical winner, especially a tiny gap or low-count/flat-trace candidate, is not proof of statistically significant superiority.
- No forward experiment can be replaced by a wind-tunnel simulation in this function.

## Use

```python
from ml_policy.medium_configuration_function import select_medium_configuration
configuration = select_medium_configuration(2, 'head', 1)
```

Use `explain_medium_configuration` with the same three inputs for exact times, runner-up gap, individual SOCs and the charging schedule.
For reproducibility, the manifest records the source hashes and independent exhaustive-scheduling checks. Rankings retain every evaluated candidate.
