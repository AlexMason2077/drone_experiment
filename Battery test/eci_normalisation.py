"""
ECI: Energy Consumption Index — Within-Drone Baseline Normalisation
====================================================================

控制 SoH (State of Health) 异质性对 RQ1 测量的污染。

定义：
    r_i,0     = drone i 在 standardised solo-hover baseline 下的 discharge rate
    r_i,c     = drone i 在 condition c 下的 discharge rate
    ECI_i,c   = r_i,c / r_i,0

每架机做自己的 control：分子分母同电池、同温度、同时段，
SoH / virtual-full / capacity fade 在比值中抵消，剩下的差异即 condition effect。

重要保护：
    默认只使用 baseline/test 共同覆盖的可信 SoC 区间（默认 85%–40%）。
    例如一次测试是 96%→70%，另一次是 85%→60%，实际只比较 85%→70%。
    如果共同窗口过窄，函数会报错而不是给出误导性的 ECI。

用法（作为 module 或独立运行）：
    from eci_normalisation import compute_stable_rate, compute_eci

    # 单架机的 ECI
    result = compute_eci('drone03_baseline.csv', 'drone03_vee_5ms.csv')

    # 多架机汇总
    aggregate = aggregate_eci([result_for_each_drone, ...])
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

# ====================== 默认参数 ======================
TRIM_HEAD_S = 30.0     # 去掉起飞瞬态 (BMS recalibration spike, voltage settling)
TRIM_TAIL_S = 30.0     # 去掉降落前 sag-out (low-SoC voltage collapse)
TRUSTED_SOC_WINDOW = (85.0, 40.0)  # 默认只用 BMS 相对可信的 SoC 区间
SENSITIVITY_SOC_WINDOWS = (
    (90.0, 30.0),
    (85.0, 40.0),
    (80.0, 50.0),
)
RECOMMENDED_BASELINE_DURATION_S = 120.0
DEFAULT_REFERENCE_BATTERY_ID = '02'
MIN_WINDOW_POINTS = 50             # 原始窗口最小数据点数
MIN_REGRESSION_POINTS = 6          # 聚合后回归点数下限
MIN_SOC_SPAN_PCT = 8.0             # baseline/test 共同 SoC 区间至少覆盖的百分比
T_CRIT_95 = {          # 小样本 t-distribution 95% 临界值（df = n-1）
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


# ====================== 数据结构 ======================
@dataclass
class RateResult:
    rate_pct_per_min: float        # 放电速率 (%/min, positive)
    se: float                      # rate 的标准误
    n: int                         # 回归点数
    raw_n: int                     # 原始采样点数
    duration_s: float              # 分析窗口时长
    window: tuple[float, float]    # 实际使用的时间窗口 [t_start, t_end]
    soc_window: tuple[float, float] | None
    soc_span_pct: float
    battery_col: str


def _normalise_soc_window(
    soc_window: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if soc_window is None:
        return None
    hi, lo = float(max(soc_window)), float(min(soc_window))
    if hi <= lo:
        raise ValueError(f'非法 SoC window: {soc_window}')
    return hi, lo


def _available_soc_range(df: pd.DataFrame, battery_col: str) -> tuple[float, float]:
    values = pd.to_numeric(df[battery_col], errors='coerce').dropna()
    if values.empty:
        raise ValueError(f'{battery_col} 没有可用数值')
    return float(values.max()), float(values.min())


def discover_battery_logs(log_dir: str) -> dict[str, str]:
    """发现 discharge_logs/battery_XX_*.csv，按电池编号返回最新文件。"""
    discovered: dict[str, str] = {}
    if not os.path.isdir(log_dir):
        return discovered
    for name in sorted(os.listdir(log_dir)):
        match = re.match(r'battery_(\d+)_.*\.csv$', name)
        if not match:
            match = re.match(r'battery_test_B(\d+)_.*\.csv$', name)
        if not match:
            continue
        battery_id = match.group(1).zfill(2)
        discovered[battery_id] = os.path.join(log_dir, name)
    return dict(sorted(discovered.items(), key=lambda item: int(item[0])))


def _common_soc_window(
    df_base: pd.DataFrame,
    df_test: pd.DataFrame,
    requested: tuple[float, float] | None,
    battery_col: str,
    min_soc_span_pct: float,
) -> tuple[float, float] | None:
    requested_norm = _normalise_soc_window(requested)
    if requested_norm is None:
        return None

    req_hi, req_lo = requested_norm
    base_hi, base_lo = _available_soc_range(df_base, battery_col)
    test_hi, test_lo = _available_soc_range(df_test, battery_col)

    hi = min(req_hi, base_hi, test_hi)
    lo = max(req_lo, base_lo, test_lo)
    span = hi - lo
    if span < min_soc_span_pct:
        raise ValueError(
            'baseline/test 的共同可信 SoC 区间太窄，不能可靠计算 ECI: '
            f'common=[{hi:.1f}, {lo:.1f}]%, span={span:.1f}%, '
            f'min={min_soc_span_pct:.1f}%'
        )
    return hi, lo


def _aggregate_for_regression(
    t_w: np.ndarray,
    bat_w: np.ndarray,
    aggregate: str,
) -> tuple[np.ndarray, np.ndarray]:
    if aggregate == 'none':
        return t_w, bat_w
    if aggregate == 'battery_level':
        grouped = (
            pd.DataFrame({'elapsed_s': t_w, 'battery': bat_w})
            .groupby('battery', as_index=False)
            .agg(first_t=('elapsed_s', 'first'), last_t=('elapsed_s', 'last'))
            .sort_values('battery', ascending=False)
        )
        t_reg = ((grouped['first_t'] + grouped['last_t']) / 2.0).to_numpy(dtype=float)
        bat_reg = grouped['battery'].to_numpy(dtype=float)
        return t_reg, bat_reg
    if aggregate.startswith('block:'):
        block_s = float(aggregate.split(':', 1)[1])
        if block_s <= 0:
            raise ValueError(f'非法 block 聚合秒数: {aggregate}')
        block_id = np.floor((t_w - t_w.min()) / block_s).astype(int)
        grouped = (
            pd.DataFrame({'block': block_id, 'elapsed_s': t_w, 'battery': bat_w})
            .groupby('block', as_index=False)
            .agg(elapsed_s=('elapsed_s', 'mean'), battery=('battery', 'mean'))
            .sort_values('elapsed_s')
        )
        return (
            grouped['elapsed_s'].to_numpy(dtype=float),
            grouped['battery'].to_numpy(dtype=float),
        )
    raise ValueError(
        "aggregate 只能是 'battery_level', 'block:<seconds>' 或 'none'"
    )


@dataclass
class ECIResult:
    drone_id: str
    condition: str
    baseline: RateResult
    test: RateResult
    eci: float                     # ECI = test/baseline
    eci_se: float                  # ECI 的标准误（误差传播）
    eci_ci95: tuple[float, float]  # 95% 置信区间


# ====================== 核心函数 ======================
def compute_stable_rate(
    df: pd.DataFrame,
    trim_head_s: float = TRIM_HEAD_S,
    trim_tail_s: float = TRIM_TAIL_S,
    window: tuple[float, float] | None = None,
    soc_window: tuple[float, float] | None = TRUSTED_SOC_WINDOW,
    battery_col: str = 'battery_pct',
    time_col: str = 'elapsed_s',
    aggregate: str = 'battery_level',
    min_window_points: int = MIN_WINDOW_POINTS,
    min_regression_points: int = MIN_REGRESSION_POINTS,
) -> RateResult:
    """
    用线性回归估计 stable mid-region 的 discharge rate。

    默认会额外限制到可信 SoC 区间，并按每个 battery percentage level
    聚合后回归，避免 0.1s 高频重复采样把 CI 伪装得过窄。

    Parameters
    ----------
    df : DataFrame with 'elapsed_s' and battery_col columns
    trim_head_s, trim_tail_s : 头尾各裁掉的秒数（若 window=None）
    window : 显式指定 [t_start, t_end] 窗口，覆盖 trim 设置
    soc_window : 只使用 [upper, lower] 百分比窗口；None 表示不按 SoC 裁剪
    aggregate : 'battery_level', 'block:<seconds>' 或 'none'

    Returns
    -------
    RateResult
    """
    if not {time_col, battery_col}.issubset(df.columns):
        raise ValueError(f'CSV 缺少 {time_col} 或 {battery_col} 列')

    clean = (
        df[[time_col, battery_col]]
        .apply(pd.to_numeric, errors='coerce')
        .dropna()
        .sort_values(time_col)
    )
    if clean.empty:
        raise ValueError('没有可用的时间/电量数据')

    t = clean[time_col].to_numpy(dtype=float)
    bat = clean[battery_col].to_numpy(dtype=float)
    t_max = t[-1]

    # 决定分析窗口
    if window is not None:
        t_lo, t_hi = window
    else:
        # 短运行 (<2 min) 时按比例缩减 trim
        if t_max < trim_head_s + trim_tail_s + 30:
            trim_head_s = min(trim_head_s, 0.15 * t_max)
            trim_tail_s = min(trim_tail_s, 0.15 * t_max)
        t_lo, t_hi = trim_head_s, t_max - trim_tail_s

    mask = (t >= t_lo) & (t <= t_hi)
    soc_norm = _normalise_soc_window(soc_window)
    if soc_norm is not None:
        soc_hi, soc_lo = soc_norm
        mask &= (bat <= soc_hi) & (bat >= soc_lo)

    raw_n = int(mask.sum())
    if raw_n < min_window_points:
        raise ValueError(
            f'窗口内原始数据点过少 (n={raw_n} < {min_window_points})，'
            f'time_window=[{t_lo:.1f}, {t_hi:.1f}]s, '
            f'soc_window={soc_norm}'
        )

    t_w = t[mask]
    bat_w = bat[mask]
    soc_span = float(np.nanmax(bat_w) - np.nanmin(bat_w))
    t_reg, bat_reg = _aggregate_for_regression(t_w, bat_w, aggregate)
    n = int(len(t_reg))
    if n < min_regression_points:
        raise ValueError(
            f'聚合后回归点过少 (n={n} < {min_regression_points})，'
            f'可能 SoC 窗口太窄: soc_window={soc_norm}'
        )

    # 线性回归 bat = slope * t + intercept
    slope, intercept = np.polyfit(t_reg, bat_reg, 1)

    # 标准误：σ_residual / sqrt(Σ(t - t̄)²)
    residuals = bat_reg - (slope * t_reg + intercept)
    sigma_resid = np.sqrt(np.sum(residuals ** 2) / max(n - 2, 1))
    se_slope = sigma_resid / np.sqrt(np.sum((t_reg - t_reg.mean()) ** 2))

    rate = -slope * 60.0       # %/min, sign-flipped
    rate_se = abs(se_slope) * 60.0
    if rate <= 0:
        raise ValueError(f'估计得到非正放电速率: {rate:.3f} %/min')

    return RateResult(
        rate_pct_per_min=rate, se=rate_se, n=n, raw_n=raw_n,
        duration_s=float(t_w[-1] - t_w[0]), window=(float(t_w[0]), float(t_w[-1])),
        soc_window=soc_norm, soc_span_pct=soc_span, battery_col=battery_col,
    )


def compute_eci(
    baseline_csv: str,
    test_csv: str,
    drone_id: str = 'unknown',
    condition: str = 'unspecified',
    enforce_common_soc: bool = True,
    min_soc_span_pct: float = MIN_SOC_SPAN_PCT,
    **rate_kwargs,
) -> ECIResult:
    """
    计算 ECI = r_test / r_baseline，包含误差传播得到的 95% CI。

    默认 enforce_common_soc=True：
        先取 baseline/test 与 TRUSTED_SOC_WINDOW 的共同 SoC 区间。
        例如 baseline=96->70, test=85->60 时，只用 85->70。
        如果共同区间太窄，直接报错，避免把高电量虚标区混入比较。
        调用者传入 soc_window 时，该窗口会被解释为 requested trusted window，
        最终实际窗口仍然是 baseline ∩ test ∩ soc_window。

    误差传播（独立测量的比值）：
        (σ_ECI / ECI)² = (σ_test/r_test)² + (σ_base/r_base)²
    """
    df_base = pd.read_csv(baseline_csv)
    df_test = pd.read_csv(test_csv)

    battery_col = rate_kwargs.get('battery_col', 'battery_pct')
    requested_soc_window = rate_kwargs.pop('soc_window', TRUSTED_SOC_WINDOW)
    if enforce_common_soc:
        soc_window = _common_soc_window(
            df_base,
            df_test,
            requested_soc_window,
            battery_col=battery_col,
            min_soc_span_pct=min_soc_span_pct,
        )
    else:
        soc_window = requested_soc_window

    base_rate = compute_stable_rate(df_base, soc_window=soc_window, **rate_kwargs)
    test_rate = compute_stable_rate(df_test, soc_window=soc_window, **rate_kwargs)

    eci = test_rate.rate_pct_per_min / base_rate.rate_pct_per_min
    rel_se = np.sqrt(
        (test_rate.se / test_rate.rate_pct_per_min) ** 2
        + (base_rate.se / base_rate.rate_pct_per_min) ** 2
    )
    eci_se = eci * rel_se

    return ECIResult(
        drone_id=drone_id, condition=condition,
        baseline=base_rate, test=test_rate,
        eci=eci, eci_se=eci_se,
        eci_ci95=(eci - 1.96 * eci_se, eci + 1.96 * eci_se),
    )


def compute_eci_sensitivity(
    baseline_csv: str,
    test_csv: str,
    drone_id: str = 'unknown',
    condition: str = 'unspecified',
    soc_windows: Sequence[tuple[float, float]] = SENSITIVITY_SOC_WINDOWS,
    **rate_kwargs,
) -> list[dict]:
    """
    对多个 SoC 窗口重复计算 ECI，用于 method/results 里的 robustness check。

    每一行都强制使用 baseline ∩ test ∩ requested_window。如果某个窗口共同
    覆盖不足，会返回 ok=False 和错误原因，而不是中断整个 sensitivity run。
    """
    rows = []
    for requested_window in soc_windows:
        label = f'{max(requested_window):.0f}-{min(requested_window):.0f}%'
        try:
            result = compute_eci(
                baseline_csv,
                test_csv,
                drone_id=drone_id,
                condition=condition,
                soc_window=requested_window,
                **rate_kwargs,
            )
            rows.append({
                'ok': True,
                'requested_window': _normalise_soc_window(requested_window),
                'actual_window': result.baseline.soc_window,
                'r_base': result.baseline.rate_pct_per_min,
                'r_test': result.test.rate_pct_per_min,
                'eci': result.eci,
                'ci95': result.eci_ci95,
                'n_base': result.baseline.n,
                'n_test': result.test.n,
                'raw_n_base': result.baseline.raw_n,
                'raw_n_test': result.test.raw_n,
                'error': '',
            })
        except Exception as exc:
            rows.append({
                'ok': False,
                'requested_window': _normalise_soc_window(requested_window),
                'actual_window': None,
                'r_base': np.nan,
                'r_test': np.nan,
                'eci': np.nan,
                'ci95': (np.nan, np.nan),
                'n_base': 0,
                'n_test': 0,
                'raw_n_base': 0,
                'raw_n_test': 0,
                'error': f'{label}: {exc}',
            })
    return rows


def compute_battery_eci_against_reference(
    log_dir: str,
    reference_battery_id: str = DEFAULT_REFERENCE_BATTERY_ID,
    output_dir: str | None = None,
    output_prefix: str = 'eci_battery',
    **rate_kwargs,
) -> tuple[list[ECIResult], list[dict], list[dict]]:
    """
    对同一悬停条件下的多块电池做 apparent depletion ECI 分析。

    这不是正式 condition-vs-baseline 的 swarm ECI，而是用于量化
    inter-battery nuisance：每块电池的显示掉电速率 / reference 电池的显示掉电速率。
    """
    logs = discover_battery_logs(log_dir)
    if not logs:
        raise FileNotFoundError(f'No battery_XX_*.csv files found in {log_dir}')

    reference_battery_id = reference_battery_id.zfill(2)
    if reference_battery_id not in logs:
        reference_battery_id = next(iter(logs))

    reference_csv = logs[reference_battery_id]
    results: list[ECIResult] = []
    summary_rows: list[dict] = []
    sensitivity_rows: list[dict] = []

    summary_rows.append({
        'status': 'reference',
        'reference_battery_id': reference_battery_id,
        'test_battery_id': reference_battery_id,
        'actual_soc_window': f'{TRUSTED_SOC_WINDOW[0]:.0f}-{TRUSTED_SOC_WINDOW[1]:.0f}%',
        'r_reference_pct_per_min': np.nan,
        'r_test_pct_per_min': np.nan,
        'eci': 1.0,
        'ci95_low': 1.0,
        'ci95_high': 1.0,
        'n_reference': 0,
        'n_test': 0,
        'error': '',
    })

    for battery_id, test_csv in logs.items():
        if battery_id == reference_battery_id:
            continue
        condition = f'B{battery_id} vs B{reference_battery_id}'
        try:
            result = compute_eci(
                reference_csv,
                test_csv,
                drone_id=f'B{battery_id}',
                condition=condition,
                **rate_kwargs,
            )
            results.append(result)
            soc = result.baseline.soc_window
            summary_rows.append({
                'status': 'ok',
                'reference_battery_id': reference_battery_id,
                'test_battery_id': battery_id,
                'actual_soc_window': f'{soc[0]:.0f}-{soc[1]:.0f}%' if soc else 'none',
                'r_reference_pct_per_min': result.baseline.rate_pct_per_min,
                'r_test_pct_per_min': result.test.rate_pct_per_min,
                'eci': result.eci,
                'ci95_low': result.eci_ci95[0],
                'ci95_high': result.eci_ci95[1],
                'n_reference': result.baseline.n,
                'n_test': result.test.n,
                'error': '',
            })
        except Exception as exc:
            summary_rows.append({
                'status': 'rejected',
                'reference_battery_id': reference_battery_id,
                'test_battery_id': battery_id,
                'actual_soc_window': '',
                'r_reference_pct_per_min': np.nan,
                'r_test_pct_per_min': np.nan,
                'eci': np.nan,
                'ci95_low': np.nan,
                'ci95_high': np.nan,
                'n_reference': 0,
                'n_test': 0,
                'error': str(exc),
            })

        for row in compute_eci_sensitivity(
            reference_csv,
            test_csv,
            drone_id=f'B{battery_id}',
            condition=condition,
            **rate_kwargs,
        ):
            actual = row['actual_window']
            requested = row['requested_window']
            sensitivity_rows.append({
                'test_battery_id': battery_id,
                'reference_battery_id': reference_battery_id,
                'requested_window': f'{requested[0]:.0f}-{requested[1]:.0f}%',
                'actual_window': f'{actual[0]:.0f}-{actual[1]:.0f}%' if actual else '',
                'status': 'ok' if row['ok'] else 'rejected',
                'r_reference_pct_per_min': row['r_base'],
                'r_test_pct_per_min': row['r_test'],
                'eci': row['eci'],
                'ci95_low': row['ci95'][0],
                'ci95_high': row['ci95'][1],
                'n_reference': row['n_base'],
                'n_test': row['n_test'],
                'error': row['error'],
            })

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary_csv = os.path.join(output_dir, f'{output_prefix}_comparison.csv')
        sensitivity_csv = os.path.join(output_dir, f'{output_prefix}_sensitivity.csv')
        summary_json = os.path.join(output_dir, f'{output_prefix}_comparison.json')
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        pd.DataFrame(sensitivity_rows).to_csv(sensitivity_csv, index=False)
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'reference_battery_id': reference_battery_id,
                    'summary': summary_rows,
                    'sensitivity': sensitivity_rows,
                },
                f,
                indent=2,
            )

    return results, summary_rows, sensitivity_rows


def aggregate_eci(results: Sequence[ECIResult]) -> dict:
    """
    多架机的 ECI 汇总：mean ± 95% t-CI。
    """
    if not results:
        raise ValueError('results 为空')
    eci_values = np.array([r.eci for r in results])
    n = len(eci_values)
    mean = eci_values.mean()
    sd = eci_values.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    t_crit = T_CRIT_95.get(n - 1, 1.96)
    return {
        'n_drones': n,
        'mean': mean,
        'sd': sd,
        'se': se,
        'ci95': (mean - t_crit * se, mean + t_crit * se),
    }


# ====================== 表格 / 可视化 ======================
def format_table(results: Sequence[ECIResult]) -> str:
    """生成可读的 ECI 报表"""
    lines = []
    hdr = (f'{"drone":<8} {"condition":<22} {"SoC window":>13} '
           f'{"r_base":>10} {"r_test":>10} {"ECI":>8} {"95% CI":>20}')
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for r in results:
        ci_str = f'[{r.eci_ci95[0]:.3f}, {r.eci_ci95[1]:.3f}]'
        if r.baseline.soc_window is None:
            soc_str = 'none'
        else:
            soc_str = f'{r.baseline.soc_window[0]:.0f}-{r.baseline.soc_window[1]:.0f}%'
        lines.append(
            f'{r.drone_id:<8} {r.condition:<22} {soc_str:>13} '
            f'{r.baseline.rate_pct_per_min:>9.2f}* '
            f'{r.test.rate_pct_per_min:>9.2f}* '
            f'{r.eci:>7.3f} {ci_str:>20}'
        )
    lines.append('-' * len(hdr))
    lines.append('* in %/min')
    return '\n'.join(lines)


def format_sensitivity_table(rows: Sequence[dict]) -> str:
    """生成 SoC-window sensitivity analysis 表格。"""
    lines = []
    hdr = (f'{"requested":>10} {"actual":>10} {"r_base":>10} {"r_test":>10} '
           f'{"ECI":>8} {"95% CI":>20} {"n":>9} {"status":>10}')
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for row in rows:
        req = row['requested_window']
        req_str = f'{req[0]:.0f}-{req[1]:.0f}%'
        if not row['ok']:
            lines.append(
                f'{req_str:>10} {"--":>10} {"--":>10} {"--":>10} '
                f'{"--":>8} {"--":>20} {"--":>9} {"rejected":>10}'
            )
            lines.append(f'  reason: {row["error"]}')
            continue

        actual = row['actual_window']
        actual_str = f'{actual[0]:.0f}-{actual[1]:.0f}%'
        ci_str = f'[{row["ci95"][0]:.3f}, {row["ci95"][1]:.3f}]'
        n_str = f'{row["n_base"]}/{row["n_test"]}'
        lines.append(
            f'{req_str:>10} {actual_str:>10} '
            f'{row["r_base"]:>9.2f}* {row["r_test"]:>9.2f}* '
            f'{row["eci"]:>7.3f} {ci_str:>20} {n_str:>9} {"ok":>10}'
        )
    lines.append('-' * len(hdr))
    lines.append('* in %/min; n = aggregated regression points base/test')
    return '\n'.join(lines)


def format_battery_eci_table(rows: Sequence[dict]) -> str:
    """生成多电池 apparent depletion ECI 表格。"""
    lines = []
    hdr = (f'{"battery":>8} {"ref":>6} {"SoC window":>11} {"r_ref":>9} '
           f'{"r_test":>9} {"ECI":>8} {"95% CI":>20} {"n":>9} {"status":>10}')
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for row in rows:
        battery = f'B{row["test_battery_id"]}'
        ref = f'B{row["reference_battery_id"]}'
        if row['status'] == 'reference':
            lines.append(
                f'{battery:>8} {ref:>6} {"--":>11} {"--":>9} {"--":>9} '
                f'{1.0:>8.3f} {"[1.000, 1.000]":>20} {"--":>9} {"reference":>10}'
            )
            continue
        if row['status'] != 'ok':
            lines.append(
                f'{battery:>8} {ref:>6} {"--":>11} {"--":>9} {"--":>9} '
                f'{"--":>8} {"--":>20} {"--":>9} {"rejected":>10}'
            )
            lines.append(f'  reason: {row["error"]}')
            continue
        ci_str = f'[{row["ci95_low"]:.3f}, {row["ci95_high"]:.3f}]'
        n_str = f'{row["n_reference"]}/{row["n_test"]}'
        lines.append(
            f'{battery:>8} {ref:>6} {row["actual_soc_window"]:>11} '
            f'{row["r_reference_pct_per_min"]:>8.2f}* '
            f'{row["r_test_pct_per_min"]:>8.2f}* '
            f'{row["eci"]:>8.3f} {ci_str:>20} {n_str:>9} {"ok":>10}'
        )
    lines.append('-' * len(hdr))
    lines.append('* in %/min; ECI = battery apparent depletion rate / reference rate')
    return '\n'.join(lines)


def _plot_eci_with_pil(results: Sequence[ECIResult], outfile: str) -> None:
    """matplotlib 不可用时的轻量 PNG fallback。"""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1100, 620
    margin_l, margin_r, margin_t, margin_b = 90, 45, 70, 135
    image = Image.new('RGB', (width, height), (248, 248, 245))
    draw = ImageDraw.Draw(image)
    try:
        font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 28)
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 18)
        font_small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    except OSError:
        font_title = font = font_small = ImageFont.load_default()

    values = np.array([r.eci for r in results], dtype=float)
    errs = np.array([1.96 * r.eci_se for r in results], dtype=float)
    y_max = max(1.2, float(np.max(values + errs)) * 1.15)
    y_min = min(0.0, float(np.min(values - errs)) * 0.9)

    plot_l, plot_t = margin_l, margin_t
    plot_r, plot_b = width - margin_r, height - margin_b
    plot_w, plot_h = plot_r - plot_l, plot_b - plot_t

    def sx(idx: int) -> float:
        return plot_l + (idx + 0.5) * plot_w / len(values)

    def sy(value: float) -> float:
        return plot_b - (value - y_min) / (y_max - y_min) * plot_h

    draw.text((margin_l, 25), 'Per-Drone ECI (Condition rate / Baseline rate)',
              fill=(25, 25, 25), font=font_title)
    for i in range(6):
        y_val = y_min + i * (y_max - y_min) / 5
        y = sy(y_val)
        draw.line([(plot_l, y), (plot_r, y)], fill=(225, 225, 220), width=1)
        draw.text((20, y - 9), f'{y_val:.2f}', fill=(70, 70, 70), font=font_small)
    draw.line([(plot_l, plot_b), (plot_r, plot_b)], fill=(65, 65, 65), width=2)
    draw.line([(plot_l, plot_t), (plot_l, plot_b)], fill=(65, 65, 65), width=2)

    if y_min <= 1.0 <= y_max:
        y_ref = sy(1.0)
        draw.line([(plot_l, y_ref), (plot_r, y_ref)], fill=(120, 120, 120), width=2)
        draw.text((plot_r - 150, y_ref - 22), 'ECI = 1', fill=(80, 80, 80), font=font_small)

    bar_w = min(80, plot_w / max(len(values), 1) * 0.55)
    for idx, r in enumerate(results):
        x = sx(idx)
        y0, y1 = sy(0.0), sy(r.eci)
        draw.rectangle([x - bar_w / 2, y1, x + bar_w / 2, y0],
                       fill=(40, 116, 166), outline=(20, 20, 20))
        err_hi, err_lo = sy(r.eci + 1.96 * r.eci_se), sy(r.eci - 1.96 * r.eci_se)
        draw.line([(x, err_lo), (x, err_hi)], fill=(20, 20, 20), width=2)
        draw.line([(x - 12, err_hi), (x + 12, err_hi)], fill=(20, 20, 20), width=2)
        draw.line([(x - 12, err_lo), (x + 12, err_lo)], fill=(20, 20, 20), width=2)
        draw.text((x - 26, y1 - 24), f'{r.eci:.3f}', fill=(25, 25, 25), font=font_small)
        label = f'{r.drone_id}\n{r.condition}'
        for line_idx, text in enumerate(label.splitlines()):
            draw.text((x - 48, plot_b + 12 + line_idx * 20), text[:18],
                      fill=(25, 25, 25), font=font_small)

    image.save(outfile)
    print(f'[OK] saved {outfile} (PIL fallback)')


def plot_eci(results: Sequence[ECIResult], outfile: str = 'eci_summary.png'):
    """画 per-drone ECI bar chart with error bars + 1.0 reference line"""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _plot_eci_with_pil(results, outfile)
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f'{r.drone_id}\n{r.condition}' for r in results]
    values = [r.eci for r in results]
    errs = [1.96 * r.eci_se for r in results]
    x = np.arange(len(values))

    bars = ax.bar(x, values, yerr=errs, capsize=5,
                  color='#2874A6', alpha=0.75, edgecolor='black')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1,
               label='ECI = 1 (baseline reference)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Energy Consumption Index (ECI)')
    ax.set_title('Per-Drone ECI  (Condition rate / Baseline rate)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='best')

    # 数值标注
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=130, bbox_inches='tight')
    print(f'[OK] saved {outfile}')


# ====================== Demo / Self-test ======================
def demo():
    """
    用 discharge_logs 中的电池 CSV 做 ECI self-test。

    没有真实 condition 配对时，这里输出的是 inter-battery apparent
    depletion ECI：每块电池的显示掉电速率 / reference 电池的显示掉电速率。
    它用于量化 battery/BMS nuisance，不应解释为 formation/wind effect。

    实验 protocol 建议 baseline 至少记录 120 s；60 s 在 Tello 的整数 BMS
    百分比下可能只跨少数台阶，回归点数和 CI 会比较脆。
    """
    HERE = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(HERE, 'discharge_logs')
    OUT_DIR = os.path.join(HERE, 'analysis_outputs')

    results, summary_rows, sensitivity_rows = compute_battery_eci_against_reference(
        LOG_DIR,
        reference_battery_id=DEFAULT_REFERENCE_BATTERY_ID,
        output_dir=OUT_DIR,
    )

    if not summary_rows:
        return

    print('Battery apparent-depletion ECI:')
    print(format_battery_eci_table(summary_rows))
    print()
    print('SoC-window sensitivity:')
    print(pd.DataFrame(sensitivity_rows).to_string(index=False))

    if results:
        plot_eci(results, outfile=os.path.join(OUT_DIR, 'eci_battery_comparison.png'))


if __name__ == '__main__':
    demo()
