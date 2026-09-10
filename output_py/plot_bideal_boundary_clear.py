"""Readable static fallback after the user rejected the native overlays.

Contract: paired SOC-time small multiples, one battery per exported image;
left 75/40 and right 80/56, identical axes and original samples. A shared-scale
residual strip below each panel separates small differences from overlapping
curves. Black observations, orange old fit, blue new fit (hard two-root cap),
dash/marker distinctions, annotated anchors, no smoothing or altered data.
PNG/SVG exports are visually inspected; all five paired views are intentional
repetitions of the same curve-comparison question, not a broad report.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

from search_bideal_boundaries import ROOT, PREVIOUS, OUT, load_trace, prepare, fit, PAIRS

DEST = OUT / 'clear_comparison'
FONT = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({
    'font.family': font_manager.FontProperties(fname=FONT).get_name(),
    'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 12,
    'axes.unicode_minus': False, 'text.color': '#252A30',
    'axes.labelcolor': '#252A30', 'xtick.color': '#4A5058',
    'ytick.color': '#4A5058', 'axes.edgecolor': '#92969C',
    'svg.fonttype': 'path',
})
COLORS = ['#D16B32', '#2563A6']
BLACK = '#24272C'
MODELS = [(75, 40, 'original'), (80, 56, 'best')]
XLIM = (-10, 515)
YLIM = (14, 103)


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#E6E8EB', linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_xlim(*XLIM)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.tick_params(length=3)


def main_panel(ax, p, f, upper, lower, color, col, compact=False):
    t = p['t'] - p['t'][0]
    at = f['anchor_times'] - p['t'][0]
    levels = np.array([95, upper, lower, 20])
    style(ax)
    ax.set_ylim(*YLIM)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylabel('电量 SOC（%）')
    ax.plot(t, p['soc'], color=BLACK, lw=1.5, zorder=3)
    ax.fill_between(t, p['soc'], f['pred'], color=color, alpha=.13, zorder=1)
    ax.plot(at, levels, color=color, lw=2.4, ls=(0, (5, 3)), zorder=4)
    ax.scatter(at, levels, s=65, marker='o' if col == 0 else 'D',
               facecolor=color, edgecolor='white', linewidth=1.1, zorder=6)
    for j, (x, y) in enumerate(zip(at, levels)):
        if j == 0:
            xytext, ha = (12, 0), 'left'
        elif j == 3:
            xytext, ha = (-7, 17), 'right'
        elif j == 1:
            xytext, ha = (18, 18), 'left'
        else:
            xytext, ha = (0, 23), 'center'
        label = f'{y:.0f}% · {x:.1f}s' if j in [1, 2] else f'{y:.0f}%'
        ax.annotate(label, (x, y), xytext=xytext, textcoords='offset points',
                    ha=ha, va='bottom', fontsize=11 if compact else 12,
                    color=color if j in [1, 2] else '#5B626B',
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=.94),
                    arrowprops=dict(arrowstyle='-', color=color, lw=.8) if j in [1, 2] else None,
                    zorder=8)
    ax.set_title(f'{upper}% / {lower}%     RMSE = {f["rmse"]:.3f} 个百分点',
                 loc='left', color=color, pad=14, fontsize=13 if compact else 15)


def residual_panel(ax, p, f, color, residual_limit):
    t = p['t'] - p['t'][0]
    e = f['pred'] - p['soc']
    style(ax)
    ax.set_ylim(-residual_limit, residual_limit)
    ax.set_yticks([-residual_limit, 0, residual_limit])
    ax.axhline(0, color=BLACK, lw=1)
    ax.plot(t, e, color=color, lw=1.1)
    ax.fill_between(t, 0, e, color=color, alpha=.15)
    ax.set_ylabel('拟合值 − 实测值\n（百分点）', fontsize=11)
    ax.set_xlabel('从首次95%开始的悬停时间（秒）')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    sources = pd.read_csv(PREVIOUS / 'coefficients.csv').drop_duplicates(['battery_id', 'run_id']).sort_values('battery_id')
    old_results = pd.read_csv(OUT / 'comparison.csv')
    assert len(sources) == 5
    items = []
    for src in sources.itertuples():
        assert PAIRS[src.battery_id] == src.drone
        assert src.run_id != '20260906_165812'
        p = prepare(load_trace(src))  # original source hash verified
        fs = [fit(p, upper, lower) for upper, lower, _ in MODELS]
        for f, (u, l, model) in zip(fs, MODELS):
            old = old_results[(old_results.battery_id == src.battery_id) & (old_results.model == model)].iloc[0]
            assert abs(old.rmse_pp - f['rmse']) < 1e-10
            np.testing.assert_allclose(np.interp(f['anchor_times'], p['t'], p['soc']), [95, u, l, 20])
        items.append((src, p, fs))
    residual_limit = int(np.ceil(max(f['max_error'] for _, _, fs in items for f in fs) / 2) * 2)
    checks = []
    for src, p, fs in items:
        fig = plt.figure(figsize=(14.4, 8.2), facecolor='white')
        gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1], left=.075, right=.98,
                             bottom=.16, top=.77, wspace=.20, hspace=.13)
        fig.text(.075, .945, f'{src.battery_id} / {src.drone.replace("drone_", "D")}：两种分界线，左右对照', fontsize=23)
        fig.text(.075, .895, '2026-09-06 · 无风单机Hover · 相同实测数据 · 两侧及五块电池的坐标尺度一致', fontsize=12, color='#626872')
        handles = [Line2D([], [], color=BLACK, lw=1.5, label='黑线：真实SOC'),
                   Line2D([], [], color=COLORS[0], lw=2.4, ls='--', marker='o', label='橙色：75% / 40%'),
                   Line2D([], [], color=COLORS[1], lw=2.4, ls='--', marker='D', label='蓝色：80% / 56%')]
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.068, .866),
                   ncol=3, frameon=False, fontsize=12, columnspacing=2)
        for col, ((u, l, _), f) in enumerate(zip(MODELS, fs)):
            ax = fig.add_subplot(gs[0, col])
            main_panel(ax, p, f, u, l, COLORS[col], col)
            ax.tick_params(labelbottom=False)
            er = fig.add_subplot(gs[1, col])
            residual_panel(er, p, f, COLORS[col], residual_limit)
        fig.text(.075, .071, '标记点 = 95%起点、两个分界点、20%终点；每套彩色线均由这4个实测阈值点连接而成。', fontsize=11, color='#555D66')
        fig.text(.075, .038, '下方误差曲线越接近0越好。只改变分界线；未改变原始SOC、时间或拟合方法。', fontsize=11, color='#555D66')
        for ext in ['png', 'svg']:
            fig.savefig(DEST / f'{src.battery_id}_side_by_side.{ext}', dpi=180, facecolor='white')
        plt.close(fig)
        checks.append(dict(battery_id=src.battery_id, source=src.source, sha256=src.sha256,
            samples=len(p['t']), old_rmse=fs[0]['rmse'], new_rmse=fs[1]['rmse'],
            checked_anchors=8, xlim=XLIM, ylim=YLIM, residual_ylim=[-residual_limit, residual_limit]))

    # Compact overview is an optional companion, not the only readable export.
    fig, axes = plt.subplots(5, 2, figsize=(14.4, 22.0), facecolor='white')
    fig.subplots_adjust(left=.075, right=.98, top=.95, bottom=.05, wspace=.20, hspace=.50)
    fig.text(.075, .984, '五块电池：75% / 40% 与 80% / 56% 分界线对照', fontsize=23, va='top')
    fig.text(.075, .965, '黑线为实测SOC，橙/蓝虚线为三段近似，标记为计算端点。全部使用相同坐标。', fontsize=12, va='top', color='#626872')
    for row, (src, p, fs) in enumerate(items):
        for col, ((u, l, _), f) in enumerate(zip(MODELS, fs)):
            ax = axes[row, col]
            main_panel(ax, p, f, u, l, COLORS[col], col, compact=True)
            ax.set_title(f'{src.battery_id} / {src.drone.replace("drone_", "D")}  ·  {u}% / {l}%  ·  RMSE {f["rmse"]:.3f}',
                         loc='left', color=COLORS[col], pad=14, fontsize=13)
            ax.set_xlabel('从首次95%开始的时间（秒）')
    fig.text(.075, .015, '2026-09-06，无风单机Hover；B12采用D5，排除D2记录。范围95%→20%，不外推。', fontsize=11, color='#626872')
    fig.savefig(DEST / 'all_five_overview.png', dpi=160, facecolor='white')
    plt.close(fig)
    (DEST / 'validation.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    (DEST / 'README.md').write_text('''# 清晰版：左右对照

上一版原生交互叠加图被用户反馈为不清楚，改为静态PNG/SVG。每块电池各一张，左侧75%/40%，右侧80%/56%，同一实测曲线重复作为黑色参照。所有图时间轴0–500秒附近、电量轴14–103%、误差轴共同尺度。橙色圆点与蓝色菱形分别标出各模型的4个真实阈值端点，95%和20%不是新增分界线。

数据和上一轮完全相同：2026-09-06五块电池的无风单机Hover，B12仅用D5。取首次95%到首次20%范围，以首次95%为时间零点。不改原始时间，不平移SOC，不重做拟合。两套方法仍是端点连接，不是另外使用最小二乘回归。

下方展示拟合SOC减去实测SOC，正数表示模型算高了，负数表示模型算低了。RMSE使用全部原始样本，已逐个与上一轮结果核对。彩色阴影表示曲线之间的视觉差距，不是置信区间。

主交付是五张`Bxx_side_by_side.png`；同名SVG可以无限放大；`all_five_overview.png`为总览。模型v1、耗电率表、原始数据与飞行代码均不修改。
''')
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
