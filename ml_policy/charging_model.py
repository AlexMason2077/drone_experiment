"""Shared charging assumptions for Oracle labelling and online decisions."""

from __future__ import annotations

import math


FULLY_CHARGED_SOC = 99.0
ZERO_TO_FULLY_CHARGED_MINUTES = 90.0
DECISION_INTERVAL_SECONDS = 30.0


def charging_time_constant_minutes(
    *,
    fully_charged_soc: float = FULLY_CHARGED_SOC,
    zero_to_fully_charged_minutes: float = ZERO_TO_FULLY_CHARGED_MINUTES,
) -> float:
    """Return tau after anchoring 0% -> ``fully_charged_soc`` to 90 min."""

    if not 0.0 < fully_charged_soc < 100.0:
        raise ValueError("fully_charged_soc must be strictly between 0 and 100")
    if zero_to_fully_charged_minutes <= 0.0:
        raise ValueError("zero_to_fully_charged_minutes must be positive")
    target_fraction = fully_charged_soc / 100.0
    return zero_to_fully_charged_minutes / math.log(1.0 / (1.0 - target_fraction))


def exponential_charging_minutes(
    current_soc: float,
    *,
    fully_charged_soc: float = FULLY_CHARGED_SOC,
    zero_to_fully_charged_minutes: float = ZERO_TO_FULLY_CHARGED_MINUTES,
) -> float:
    """Estimate charging time from ``current_soc`` to fully charged.

    "Fully charged" is represented numerically by 99%, because the
    exponential model is asymptotic and is undefined at exactly 100%.
    ``tau`` is calibrated so that charging from 0% to 99% takes 90 minutes.
    """

    soc = float(current_soc)
    if not 0.0 <= soc <= 100.0:
        raise ValueError("current_soc must be between 0 and 100")
    if soc >= fully_charged_soc:
        return 0.0

    target_fraction = fully_charged_soc / 100.0
    current_fraction = soc / 100.0
    tau = charging_time_constant_minutes(
        fully_charged_soc=fully_charged_soc,
        zero_to_fully_charged_minutes=zero_to_fully_charged_minutes,
    )
    return tau * math.log(
        (1.0 - current_fraction) / (1.0 - target_fraction)
    )
