"""Online configuration controller for a five-drone swarm.

The controller knows only the wind condition that is observable *now*.  It
does not receive future wind conditions and does not plan a complete schedule
before take-off.  Every 30 seconds, the caller supplies the currently observed
wind, charging-pad availability K, battery levels, and remaining distance.
The controller then chooses a new configuration c = (f, p, d).

The mission begins at Node A with every drone at 100% and ends when every
drone has been fully charged at Node B.  The objective used at each online
decision is an estimate of reconfiguration time plus the Node-B charging
completion time implied by the candidate configuration.

This is a research draft.  The demonstration energy, charging, and switching
parameters are fictional and must later be calibrated from experimental data.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Mapping, Protocol, Sequence

from ml_policy.charging_model import (
    DECISION_INTERVAL_SECONDS,
    FULLY_CHARGED_SOC,
    ZERO_TO_FULLY_CHARGED_MINUTES,
    exponential_charging_minutes,
)


DroneId = str


@dataclass(frozen=True)
class WindCondition:
    """The wind condition currently observed by the swarm."""

    wind_direction: str
    wind_level: int

    def __post_init__(self) -> None:
        if not self.wind_direction.strip():
            raise ValueError("wind_direction cannot be empty")
        if self.wind_level <= 0:
            raise ValueError("wind_level must be positive")


@dataclass(frozen=True)
class Configuration:
    """A swarm configuration c = (formation, position, distance).

    ``drone_by_slot`` is the position assignment p.  Generic, formation-
    specific slot names are used because Column, Vee, Echelon, and Diamond do
    not necessarily share the same physical position labels.  Their exact
    geometry should later be read from the experiment controller.
    """

    formation: str
    drone_by_slot: tuple[DroneId, ...]
    distance_cm: int

    def position_mapping(self) -> dict[str, DroneId]:
        return {
            f"{self.formation}_slot_{index + 1}": drone
            for index, drone in enumerate(self.drone_by_slot)
        }


@dataclass(frozen=True)
class OnlineDecision:
    """One decision made with information available at that moment."""

    timestamp_seconds: float
    observed_condition: WindCondition
    charging_pad_availability: int
    measured_battery_before: tuple[float, ...]
    previous_configuration: Configuration | None
    selected_configuration: Configuration
    action: str
    evaluation_distance_m: float
    predicted_battery_drop: tuple[float, ...]
    projected_battery: tuple[float, ...]
    reconfiguration_seconds: float
    projected_charging_seconds: float
    online_score_seconds: float


@dataclass(frozen=True)
class MissionResult:
    """Observed mission result after the swarm arrives at Node B."""

    decision_history: tuple[OnlineDecision, ...]
    arrival_battery: tuple[float, ...]
    flight_seconds: float
    charging_seconds: float
    total_seconds: float
    charging_pad_loads_seconds: tuple[float, ...]


class EnergyModel(Protocol):
    """Replaceable energy-response model calibrated from cleaned data."""

    def predict_battery_drop(
        self,
        condition: WindCondition,
        configuration: Configuration,
        drone_ids: Sequence[DroneId],
        distance_m: float,
    ) -> tuple[float, ...]:
        """Return predicted percentage-point drops over ``distance_m``."""


class ChargingModel(Protocol):
    def seconds_to_full(self, current_soc: float) -> float:
        """Return charging time from ``current_soc`` to fully charged."""


@dataclass(frozen=True)
class ParametricEnergyModel:
    """Interpretable response model used by the first implementation.

    These coefficients describe energy responses.  They do not directly map a
    wind condition to a winning configuration.  Later, they can be fitted from
    the forward-motion-only data in ``db_copy_for_cleaning``.
    """

    base_drop_per_meter: float
    condition_factors: Mapping[tuple[str, int], float]
    formation_factors: Mapping[tuple[str, str], float]
    distance_factors: Mapping[int, float]
    slot_factors: Mapping[str, tuple[float, ...]]
    drone_factors: Mapping[DroneId, float]

    def predict_battery_drop(
        self,
        condition: WindCondition,
        configuration: Configuration,
        drone_ids: Sequence[DroneId],
        distance_m: float,
    ) -> tuple[float, ...]:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")

        direction = condition.wind_direction.lower()
        condition_factor = self.condition_factors.get(
            (direction, condition.wind_level),
            1.0,
        )
        formation_factor = self.formation_factors.get(
            (direction, configuration.formation.lower()),
            1.0,
        )
        distance_factor = self.distance_factors.get(
            configuration.distance_cm,
            1.0,
        )
        slot_factors = self.slot_factors.get(
            configuration.formation.lower(),
            (1.0,) * len(configuration.drone_by_slot),
        )
        if len(slot_factors) != len(configuration.drone_by_slot):
            raise ValueError(
                f"Expected {len(configuration.drone_by_slot)} slot factors for "
                f"{configuration.formation}; received {len(slot_factors)}"
            )

        drop_by_drone = {drone: 0.0 for drone in drone_ids}
        base_drop = self.base_drop_per_meter * distance_m
        for slot_index, drone in enumerate(configuration.drone_by_slot):
            drop_by_drone[drone] = (
                base_drop
                * condition_factor
                * formation_factor
                * distance_factor
                * slot_factors[slot_index]
                * self.drone_factors.get(drone, 1.0)
            )

        return tuple(drop_by_drone[drone] for drone in drone_ids)


@dataclass(frozen=True)
class LinearChargingModel:
    """Legacy approximation retained only for charging-model ablation."""

    percentage_points_per_minute: float

    def seconds_to_full(self, current_soc: float) -> float:
        if self.percentage_points_per_minute <= 0:
            raise ValueError("percentage_points_per_minute must be positive")
        return (
            60.0
            * max(0.0, 100.0 - current_soc)
            / self.percentage_points_per_minute
        )


@dataclass(frozen=True)
class ExponentialChargingModel:
    """Tello charging model: 0% to fully charged (99%) takes 90 minutes."""

    fully_charged_soc: float = FULLY_CHARGED_SOC
    zero_to_fully_charged_minutes: float = ZERO_TO_FULLY_CHARGED_MINUTES

    def seconds_to_full(self, current_soc: float) -> float:
        return 60.0 * exponential_charging_minutes(
            current_soc,
            fully_charged_soc=self.fully_charged_soc,
            zero_to_fully_charged_minutes=self.zero_to_fully_charged_minutes,
        )


@dataclass(frozen=True)
class ReconfigurationModel:
    """Estimated time and energy for an airborne configuration change."""

    formation_change_seconds: float = 5.0
    distance_change_seconds: float = 3.0
    seconds_per_changed_position: float = 0.8
    battery_drop_per_second: float = 0.01

    def estimate(
        self,
        current: Configuration | None,
        candidate: Configuration,
        drone_count: int,
    ) -> tuple[float, tuple[float, ...]]:
        # Choosing the initial configuration before take-off is not treated as
        # an airborne reconfiguration.
        if current is None:
            return 0.0, (0.0,) * drone_count

        seconds = 0.0
        if current.formation != candidate.formation:
            seconds += self.formation_change_seconds
        if current.distance_cm != candidate.distance_cm:
            seconds += self.distance_change_seconds

        changed_assignments = sum(
            old_drone != new_drone
            for old_drone, new_drone in zip(
                current.drone_by_slot,
                candidate.drone_by_slot,
            )
        )
        seconds += changed_assignments * self.seconds_per_changed_position
        transition_drop = (seconds * self.battery_drop_per_second,) * drone_count
        return seconds, transition_drop


def generate_candidate_configurations(
    drone_ids: Sequence[DroneId],
    formations: Iterable[str],
    distances_cm: Iterable[int],
) -> list[Configuration]:
    """Generate all feasible f, p, d combinations for the optimiser."""

    if len(set(drone_ids)) != len(drone_ids):
        raise ValueError("drone_ids must be unique")

    return [
        Configuration(
            formation=formation.lower(),
            drone_by_slot=tuple(position_assignment),
            distance_cm=distance_cm,
        )
        for formation in formations
        for distance_cm in distances_cm
        for position_assignment in permutations(drone_ids)
    ]


def _minimum_parallel_charging_time(
    charging_jobs_seconds: Sequence[float],
    pad_count: int,
) -> tuple[float, tuple[float, ...]]:
    """Exactly minimise the completion time of five charging jobs."""

    if pad_count <= 0:
        raise ValueError("charging_pad_availability must be at least 1")

    effective_pad_count = min(pad_count, len(charging_jobs_seconds))
    jobs = sorted(charging_jobs_seconds, reverse=True)
    loads = [0.0] * effective_pad_count
    best_makespan = float("inf")
    best_loads: tuple[float, ...] = ()

    def search(job_index: int) -> None:
        nonlocal best_makespan, best_loads
        if job_index == len(jobs):
            makespan = max(loads, default=0.0)
            if makespan < best_makespan:
                best_makespan = makespan
                best_loads = tuple(sorted(loads, reverse=True))
            return

        job = jobs[job_index]
        tried_loads: set[float] = set()
        for pad_index, current_load in enumerate(loads):
            rounded_load = round(current_load, 9)
            if rounded_load in tried_loads:
                continue
            tried_loads.add(rounded_load)

            new_load = current_load + job
            if new_load >= best_makespan:
                continue
            loads[pad_index] = new_load
            search(job_index + 1)
            loads[pad_index] = current_load

    search(0)
    return best_makespan, best_loads


class OnlineConfigurationController:
    """Fixed-interval online controller that never reads future conditions."""

    def __init__(
        self,
        *,
        charging_pad_availability: int,
        energy_model: EnergyModel,
        charging_model: ChargingModel,
        candidate_configurations: Sequence[Configuration],
        reconfiguration_model: ReconfigurationModel | None = None,
        drone_ids: Sequence[DroneId] = ("D1", "D2", "D3", "D4", "D5"),
        evaluation_distance_m: float = 1.0,
        decision_interval_seconds: float = DECISION_INTERVAL_SECONDS,
        minimum_soc: float = 20.0,
    ) -> None:
        if charging_pad_availability <= 0:
            raise ValueError("charging_pad_availability must be at least 1")
        if not candidate_configurations:
            raise ValueError("candidate_configurations cannot be empty")
        if evaluation_distance_m <= 0:
            raise ValueError("evaluation_distance_m must be positive")
        if decision_interval_seconds <= 0:
            raise ValueError("decision_interval_seconds must be positive")

        self.drone_ids = tuple(drone_ids)
        self.charging_pad_availability = charging_pad_availability
        self.energy_model = energy_model
        self.charging_model = charging_model
        self.candidate_configurations = tuple(candidate_configurations)
        self.reconfiguration_model = (
            reconfiguration_model or ReconfigurationModel()
        )
        self.evaluation_distance_m = evaluation_distance_m
        self.decision_interval_seconds = decision_interval_seconds
        self.minimum_soc = minimum_soc

        self._departure_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._current_battery: tuple[float, ...] | None = None
        self._current_configuration: Configuration | None = None
        self._history: list[OnlineDecision] = []

    @property
    def current_configuration(self) -> Configuration | None:
        return self._current_configuration

    @property
    def current_battery(self) -> tuple[float, ...] | None:
        return self._current_battery

    @property
    def decision_history(self) -> tuple[OnlineDecision, ...]:
        return tuple(self._history)

    @property
    def next_decision_timestamp(self) -> float | None:
        if self._last_timestamp is None:
            return None
        return self._last_timestamp + self.decision_interval_seconds

    def decision_due(self, timestamp_seconds: float) -> bool:
        next_timestamp = self.next_decision_timestamp
        return next_timestamp is not None and timestamp_seconds >= next_timestamp - 1e-9

    def start(
        self,
        initial_condition: WindCondition,
        *,
        timestamp_seconds: float = 0.0,
    ) -> OnlineDecision:
        """Start at Node A with all five batteries at 100%."""

        if self._departure_timestamp is not None:
            raise RuntimeError("The online controller has already been started")

        self._departure_timestamp = timestamp_seconds
        self._last_timestamp = timestamp_seconds
        self._current_battery = (100.0,) * len(self.drone_ids)
        return self._make_decision(initial_condition, timestamp_seconds)

    def on_environment_change(
        self,
        new_condition: WindCondition,
        *,
        measured_battery: Sequence[float],
        timestamp_seconds: float,
        charging_pad_availability: int | None = None,
    ) -> OnlineDecision:
        """Compatibility wrapper for the fixed-interval update method.

        New controller integrations should call :meth:`on_decision_interval`
        and explicitly refresh K and remaining distance every 30 seconds.
        """

        if charging_pad_availability is None:
            raise ValueError(
                "charging_pad_availability must be refreshed at every interval"
            )
        return self.on_decision_interval(
            observed_condition=new_condition,
            measured_battery=measured_battery,
            timestamp_seconds=timestamp_seconds,
            charging_pad_availability=charging_pad_availability,
            remaining_distance_m=self.evaluation_distance_m,
        )

    def on_decision_interval(
        self,
        observed_condition: WindCondition,
        *,
        measured_battery: Sequence[float],
        timestamp_seconds: float,
        charging_pad_availability: int,
        remaining_distance_m: float,
    ) -> OnlineDecision:
        """Refresh all observable inputs at one 30-second decision epoch."""

        self._require_started()
        self._validate_timestamp(timestamp_seconds)
        if not self.decision_due(timestamp_seconds):
            raise ValueError(
                f"Next decision is due at t={self.next_decision_timestamp:.1f}s"
            )
        self._current_battery = self._validate_battery(measured_battery)
        if charging_pad_availability <= 0:
            raise ValueError("charging_pad_availability must be at least 1")
        if remaining_distance_m <= 0:
            raise ValueError("remaining_distance_m must be positive before arrival")
        self.charging_pad_availability = charging_pad_availability
        self.evaluation_distance_m = remaining_distance_m

        self._last_timestamp = timestamp_seconds
        return self._make_decision(observed_condition, timestamp_seconds)

    def finish_at_node_b(
        self,
        *,
        arrival_battery: Sequence[float],
        arrival_timestamp_seconds: float,
        charging_pad_availability: int | None = None,
    ) -> MissionResult:
        """Finish the mission and calculate time until all drones are fully charged."""

        self._require_started()
        self._validate_timestamp(arrival_timestamp_seconds)
        battery = self._validate_battery(arrival_battery)

        if charging_pad_availability is not None:
            if charging_pad_availability <= 0:
                raise ValueError(
                    "charging_pad_availability must be at least 1"
                )
            self.charging_pad_availability = charging_pad_availability

        charging_seconds, pad_loads = self._charging_result(battery)
        assert self._departure_timestamp is not None
        flight_seconds = arrival_timestamp_seconds - self._departure_timestamp
        return MissionResult(
            decision_history=tuple(self._history),
            arrival_battery=battery,
            flight_seconds=flight_seconds,
            charging_seconds=charging_seconds,
            total_seconds=flight_seconds + charging_seconds,
            charging_pad_loads_seconds=pad_loads,
        )

    def _make_decision(
        self,
        condition: WindCondition,
        timestamp_seconds: float,
    ) -> OnlineDecision:
        assert self._current_battery is not None

        feasible: list[tuple[float, Configuration, tuple[float, ...],
                            tuple[float, ...], float, float]] = []
        for candidate in self.candidate_configurations:
            flight_drop = self.energy_model.predict_battery_drop(
                condition=condition,
                configuration=candidate,
                drone_ids=self.drone_ids,
                distance_m=self.evaluation_distance_m,
            )
            switch_seconds, switch_drop = self.reconfiguration_model.estimate(
                self._current_configuration,
                candidate,
                len(self.drone_ids),
            )
            predicted_drop = tuple(
                forward_drop + transition_drop
                for forward_drop, transition_drop in zip(
                    flight_drop,
                    switch_drop,
                )
            )
            projected_battery = tuple(
                soc - drop
                for soc, drop in zip(self._current_battery, predicted_drop)
            )
            if min(projected_battery) < self.minimum_soc:
                continue

            projected_charging_seconds, _ = self._charging_result(
                projected_battery
            )
            score = switch_seconds + projected_charging_seconds
            feasible.append(
                (
                    score,
                    candidate,
                    predicted_drop,
                    projected_battery,
                    switch_seconds,
                    projected_charging_seconds,
                )
            )

        if not feasible:
            raise RuntimeError(
                "No candidate configuration satisfies the minimum SOC constraint"
            )

        (
            score,
            selected,
            predicted_drop,
            projected_battery,
            switch_seconds,
            projected_charging_seconds,
        ) = min(feasible, key=lambda item: item[0])

        if self._current_configuration is None:
            action = "initialise"
        elif selected == self._current_configuration:
            action = "keep"
        else:
            action = "reconfigure"

        decision = OnlineDecision(
            timestamp_seconds=timestamp_seconds,
            observed_condition=condition,
            charging_pad_availability=self.charging_pad_availability,
            measured_battery_before=self._current_battery,
            previous_configuration=self._current_configuration,
            selected_configuration=selected,
            action=action,
            evaluation_distance_m=self.evaluation_distance_m,
            predicted_battery_drop=predicted_drop,
            projected_battery=projected_battery,
            reconfiguration_seconds=switch_seconds,
            projected_charging_seconds=projected_charging_seconds,
            online_score_seconds=score,
        )
        self._history.append(decision)
        self._current_configuration = selected
        return decision

    def _charging_result(
        self,
        battery: Sequence[float],
    ) -> tuple[float, tuple[float, ...]]:
        jobs = [
            self.charging_model.seconds_to_full(soc)
            for soc in battery
        ]
        return _minimum_parallel_charging_time(
            jobs,
            self.charging_pad_availability,
        )

    def _validate_battery(
        self,
        battery: Sequence[float],
    ) -> tuple[float, ...]:
        values = tuple(float(value) for value in battery)
        if len(values) != len(self.drone_ids):
            raise ValueError(
                f"Expected {len(self.drone_ids)} battery values; "
                f"received {len(values)}"
            )
        if any(value < 0 or value > 100 for value in values):
            raise ValueError("Every battery value must be between 0 and 100")
        return values

    def _validate_timestamp(self, timestamp_seconds: float) -> None:
        if self._last_timestamp is not None and timestamp_seconds < self._last_timestamp:
            raise ValueError("timestamp_seconds cannot move backwards")

    def _require_started(self) -> None:
        if self._departure_timestamp is None:
            raise RuntimeError("Call start() before updating or finishing the mission")


def build_demonstration_energy_model() -> ParametricEnergyModel:
    """Create fictional parameters solely to demonstrate controller flow."""

    return ParametricEnergyModel(
        base_drop_per_meter=3.5,
        condition_factors={
            ("head", 1): 1.15,
            ("head", 2): 1.35,
            ("side", 1): 1.08,
            ("side", 2): 1.25,
            ("tail", 1): 1.00,
            ("tail", 2): 1.12,
        },
        # These are fictional response coefficients, not fixed decisions.
        formation_factors={
            ("head", "vee"): 0.94,
            ("head", "column"): 1.02,
            ("head", "echelon"): 1.08,
            ("side", "vee"): 1.05,
            ("side", "column"): 1.08,
            ("side", "echelon"): 0.95,
            ("tail", "vee"): 1.00,
            ("tail", "column"): 0.97,
            ("tail", "echelon"): 1.02,
        },
        distance_factors={50: 1.03, 75: 0.98},
        slot_factors={
            "vee": (1.18, 1.02, 1.02, 0.92, 0.92),
            "column": (1.22, 1.04, 0.98, 0.93, 0.89),
            "echelon": (1.16, 1.08, 1.00, 0.94, 0.88),
        },
        drone_factors={
            "D1": 1.00,
            "D2": 1.01,
            "D3": 0.99,
            "D4": 1.02,
            "D5": 1.00,
        },
    )


def build_example_controller() -> OnlineConfigurationController:
    """Construct a runnable controller using demonstration parameters."""

    drone_ids = ("D1", "D2", "D3", "D4", "D5")
    candidates = generate_candidate_configurations(
        drone_ids=drone_ids,
        formations=("vee", "column", "echelon"),
        distances_cm=(50, 75),
    )
    return OnlineConfigurationController(
        charging_pad_availability=2,
        energy_model=build_demonstration_energy_model(),
        charging_model=ExponentialChargingModel(),
        candidate_configurations=candidates,
        drone_ids=drone_ids,
        evaluation_distance_m=1.0,
        minimum_soc=20.0,
    )


def print_decision(decision: OnlineDecision) -> None:
    condition = decision.observed_condition
    configuration = decision.selected_configuration
    print(
        f"t={decision.timestamp_seconds:.1f}s: observed "
        f"{condition.wind_direction} wind, level {condition.wind_level}"
    )
    print(f"  charging pads available: K={decision.charging_pad_availability}")
    print(f"  action: {decision.action}")
    print(
        f"  selected: formation={configuration.formation}, "
        f"distance={configuration.distance_cm} cm"
    )
    print(f"  position: {configuration.position_mapping()}")
    print(
        "  measured battery: "
        + ", ".join(
            f"{soc:.1f}%" for soc in decision.measured_battery_before
        )
    )


def online_example() -> MissionResult:
    """Feed observations one at a time; the controller cannot see the future."""

    controller = build_example_controller()

    # At Node A, only the initial wind condition is known.
    initial_decision = controller.start(
        WindCondition(wind_direction="head", wind_level=1),
        timestamp_seconds=0.0,
    )
    print_decision(initial_decision)

    # This call happens only after the swarm observes the environmental change.
    # The battery vector represents current telemetry, not an offline forecast.
    changed_decision = controller.on_decision_interval(
        WindCondition(wind_direction="side", wind_level=2),
        measured_battery=(96.0, 95.0, 96.0, 96.0, 97.0),
        timestamp_seconds=30.0,
        charging_pad_availability=3,
        remaining_distance_m=1.0,
    )
    print()
    print_decision(changed_decision)

    # Node B arrival telemetry and timestamp are supplied when arrival occurs.
    return controller.finish_at_node_b(
        arrival_battery=(90.0, 89.0, 90.0, 90.0, 91.0),
        arrival_timestamp_seconds=60.0,
    )


def print_mission_result(result: MissionResult) -> None:
    print("\nMission result")
    print(
        "  arrival battery: "
        + ", ".join(f"{soc:.1f}%" for soc in result.arrival_battery)
    )
    print(f"  A-to-B flight time: {result.flight_seconds:.1f} s")
    print(f"  Node-B charging time: {result.charging_seconds:.1f} s")
    print(f"  total time: {result.total_seconds:.1f} s")


if __name__ == "__main__":
    print_mission_result(online_example())
