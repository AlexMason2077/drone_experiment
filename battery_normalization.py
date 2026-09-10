"""Offline, opt-in SOC calibration. No drone/network/app imports or mutations.

Each physical battery has its own three SOC bands. Integrating dSOC/b_i(S)
gives equivalent baseline-hover seconds H; H/duration is a relative drain
factor, not a measured power ratio. A separate three-band reference advances
by H, splitting at its own boundaries. Values outside the model's declared
domain are rejected. Explicitly extended models retain the original calibration
range and mark normalization intervals that use extrapolated coefficients.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


def finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be numeric, not boolean')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be numeric') from exc
    if not math.isfinite(number):
        raise ValueError(f'{label} must be finite')
    return number


@dataclass(frozen=True)
class DischargeCurve:
    boundaries: tuple[float, ...]
    rates_pp_min: tuple[float, ...]

    def __post_init__(self):
        boundaries = tuple(finite(x, 'SOC boundary') for x in self.boundaries)
        rates = tuple(finite(x, 'discharge rate') for x in self.rates_pp_min)
        if len(boundaries) != 4 or len(rates) != 3:
            raise ValueError('Exactly three segments are required')
        if not 0 <= boundaries[-1] < boundaries[0] <= 100:
            raise ValueError('SOC domain must lie within 0..100')
        if any(a <= b for a, b in zip(boundaries, boundaries[1:])):
            raise ValueError('SOC boundaries must be strictly decreasing')
        if any(r <= 0 for r in rates):
            raise ValueError('Discharge rates must be positive')
        object.__setattr__(self, 'boundaries', boundaries)
        object.__setattr__(self, 'rates_pp_min', rates)

    def _soc(self, soc):
        s = finite(soc, 'SOC')
        if not self.boundaries[-1] <= s <= self.boundaries[0]:
            raise ValueError(f'SOC {s:g} outside declared model range {self.boundaries[-1]:g}..{self.boundaries[0]:g}; no automatic extrapolation')
        return s

    def rate_at(self, soc):
        """At an internal boundary choose the next, lower-SOC segment."""
        s = self._soc(soc)
        for upper, lower, rate in zip(self.boundaries, self.boundaries[1:], self.rates_pp_min):
            if lower < s <= upper:
                return rate
            if s == lower and s != self.boundaries[-1]:
                continue
        return self.rates_pp_min[-1]

    def equivalent_seconds(self, soc_start, soc_end):
        start, end = self._soc(soc_start), self._soc(soc_end)
        if end > start:
            raise ValueError('SOC increased; do not silently treat rebound/charging as discharge')
        return sum(max(0.0, min(start, upper)-max(end, lower))*60/rate
                   for upper, lower, rate in zip(self.boundaries, self.boundaries[1:], self.rates_pp_min))

    def advance(self, soc_start, equivalent_seconds):
        """Reference discharge for H seconds; raises instead of clipping below floor."""
        soc = self._soc(soc_start)
        remaining = finite(equivalent_seconds, 'equivalent seconds')
        if remaining < 0:
            raise ValueError('Equivalent seconds cannot be negative')
        capacity = self.equivalent_seconds(soc, self.boundaries[-1])
        if remaining > capacity + 1e-8:
            raise ValueError('Reference interval would cross calibrated SOC floor; shorten interval or choose a higher reference SOC')
        for upper, lower, rate in zip(self.boundaries, self.boundaries[1:], self.rates_pp_min):
            if not lower < soc <= upper:
                continue
            segment_seconds = (soc-lower)*60/rate
            if remaining <= segment_seconds + 1e-9:
                return max(lower, soc-remaining*rate/60)
            remaining -= segment_seconds
            soc = lower
        return soc

    def to_dict(self):
        return dict(boundaries_soc=list(self.boundaries), rates_pp_min=list(self.rates_pp_min))


class BatteryNormalizer:
    def __init__(self, model, model_sha256='unsaved'):
        if model.get('schema_version') != 1 or model.get('kind') != 'individual_battery_normalization_candidate':
            raise ValueError('Unsupported battery calibration model')
        self.version = str(model['version'])
        self.sha256 = model_sha256
        self.reference = DischargeCurve(tuple(model['reference']['boundaries_soc']), tuple(model['reference']['rates_pp_min']))
        domain = (self.reference.boundaries[-1], self.reference.boundaries[0])
        supported = tuple(finite(v, 'supported SOC boundary') for v in model.get('supported_soc_range', domain))
        calibrated = tuple(finite(v, 'calibrated SOC boundary') for v in model.get('calibrated_soc_range', domain))
        if supported != domain:
            raise ValueError('Supported SOC range must match curve endpoints')
        if len(calibrated) != 2 or not domain[0] <= calibrated[0] < calibrated[1] <= domain[1]:
            raise ValueError('Calibrated SOC range must lie within the declared model domain')
        self.calibrated_soc_range = calibrated
        self.curves, self.pairs = {}, {}
        if not model.get('batteries'):
            raise ValueError('No battery calibrations')
        for battery_id, item in model['batteries'].items():
            curve = DischargeCurve(tuple(item['boundaries_soc']), tuple(item['rates_pp_min']))
            if (curve.boundaries[0], curve.boundaries[-1]) != (self.reference.boundaries[0], self.reference.boundaries[-1]):
                raise ValueError('Battery and reference calibrated domains must agree')
            self.curves[battery_id] = curve
            self.pairs[battery_id] = str(item['drone_id'])

    @classmethod
    def load(cls, path):
        content = Path(path).read_bytes()
        return cls(json.loads(content), hashlib.sha256(content).hexdigest())

    def curve_for(self, battery_id, drone_id):
        if battery_id not in self.curves:
            raise ValueError(f'No calibration for battery {battery_id}')
        if drone_id != self.pairs[battery_id]:
            raise ValueError(f'Calibration pair mismatch: {battery_id} calibrated on {self.pairs[battery_id]}, not {drone_id}')
        return self.curves[battery_id]

    def relative_drain(self, battery_id, drone_id, soc_start, soc_end, duration_s):
        duration = finite(duration_s, 'duration_s')
        if duration <= 0:
            raise ValueError('duration_s must be positive')
        curve = self.curve_for(battery_id, drone_id)
        h = curve.equivalent_seconds(soc_start, soc_end)
        return h/duration

    def normalize_interval(self, battery_id, drone_id, soc_start, soc_end, duration_s, reference_soc):
        duration = finite(duration_s, 'duration_s')
        factor = self.relative_drain(battery_id, drone_id, soc_start, soc_end, duration)
        start, end = finite(soc_start, 'soc_start'), finite(soc_end, 'soc_end')
        reference_start = self.reference._soc(reference_soc)
        h = factor*duration
        reference_end = self.reference.advance(reference_start, h)
        calibrated_min, calibrated_max = self.calibrated_soc_range
        battery_extrapolated = start > calibrated_max or end < calibrated_min
        reference_extrapolated = reference_start > calibrated_max or reference_end < calibrated_min
        return dict(bn_version=self.version, bn_model_sha256=self.sha256,
            bn_kind='derived_Bideal_SOC_equivalent_not_measured_energy',
            bn_actual_soc_drop_pp=start-end, bn_actual_rate_pp_min=(start-end)*60/duration,
            bn_equivalent_hover_seconds=h, bn_relative_hover_factor=factor,
            bn_reference_soc_start=reference_start, bn_reference_soc_end=reference_end,
            bn_Bideal_drop_pp=reference_start-reference_end,
            bn_Bideal_average_rate_pp_min=(reference_start-reference_end)*60/duration,
            bn_reference_rate_at_start_pp_min=factor*self.reference.rate_at(reference_start),
            bn_battery_uses_extrapolation=battery_extrapolated,
            bn_reference_uses_extrapolation=reference_extrapolated,
            bn_uses_extrapolation=battery_extrapolated or reference_extrapolated,
            bn_calibrated_soc_min=calibrated_min, bn_calibrated_soc_max=calibrated_max)


def normalize_csv(model_path, input_path, output_path, reference_soc):
    """Append derived columns to a new file, preserving input rows and fields.

Each row is an independent comparison interval with the specified reference
start. Do not sum its equivalent drops as a continuous reference trajectory.
For a trajectory, feed each returned reference end into the next call.
"""
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path or output_path.exists():
        raise FileExistsError('Output must be a new file; original or existing files cannot be overwritten')
    normalizer = BatteryNormalizer.load(model_path)
    with input_path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {'battery_id', 'drone_id', 'soc_start', 'soc_end', 'duration_s'}
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError('CSV needs unique columns including: '+', '.join(sorted(required)))
        if any(f.startswith('bn_') for f in fields):
            raise ValueError('Input already has bn_ columns; refusing possible double normalization')
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f'CSV row {line} has excess cells')
            try:
                derived = normalizer.normalize_interval(row['battery_id'], row['drone_id'],
                    row['soc_start'], row['soc_end'], row['duration_s'], reference_soc)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'CSV row {line}: {exc}') from exc
            rows.append(dict(row, **derived))
    if not rows:
        raise ValueError('Input CSV contains no data')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return dict(rows=len(rows), output=str(output_path), model_version=normalizer.version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-soc', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize_csv(args.model,args.input,args.output,args.reference_soc),ensure_ascii=False))


if __name__ == '__main__':
    main()
