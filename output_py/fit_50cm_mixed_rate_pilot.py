"""One requested example only: Diamond / 50 cm / headwind / Level2.

Reuse existing 250-cm Medium coefficients unchanged. Fit the real wind-tunnel
record by each battery's own stages, average comparable normalized rates, and
only then compute durations. No batch processing, training or hardware imports.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from process_50cm_stage_curves import (
    ROOT, MODEL, FORWARD, NAMES, CELL, GRAIN, BatteryNormalizer, plt,
    sha, write_json, fit_blocks, observed_hover_frames, calibrated_rate,
    combined_means, complete_curves, plot_curve, exclusion_ids,
)

CID = "diamond_50_head_lv2"
EXPERIMENT = "wind_tunnel_diamond_50_head_lv2_002"
SOURCE = ROOT / "database" / EXPERIMENT / f"{EXPERIMENT}_20260928_232343_all_coordination.csv"
OUT = ROOT / "analysis_results/50cm_mixed_rate_pilot_20260929" / CID
PREVIOUS_PILOT = ROOT / "analysis_results/wind_tunnel_three_stage_rate_first_corrected_20260929" / EXPERIMENT / "stage_rates.csv"


def main():
    if OUT.exists():
        raise FileExistsError("Preserve existing pilot; choose a new output folder")
    assert CID not in exclusion_ids()
    normalizer=BatteryNormalizer.load(MODEL)
    forward_path=FORWARD/"run_drone_rates.csv"
    paths=[SOURCE,MODEL,forward_path,FORWARD/"discharge_rates_wide.csv",FORWARD/"manifest.json",
           PREVIOUS_PILOT,ROOT/"battery_normalization.py",ROOT/"TRAINING_EXCLUSIONS.md",
           Path(__file__),ROOT/"output_py/process_50cm_stage_curves.py"]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    forward_manifest=json.loads((FORWARD/"manifest.json").read_text())
    assert forward_manifest["normalization_model_sha256"]==normalizer.sha256
    raw=pd.read_csv(SOURCE,low_memory=False)
    for key,value in dict(experiment_id=EXPERIMENT,run_id="20260928_232343",formation="diamond",
                          wind_direction="head wind",wind_speed="Level2",inter_drone_distance_cm=50).items():
        assert set(raw[key])=={value}
    records=[];windows=[]
    for drone,g in raw.groupby("drone_name",sort=True):
        batteries=g.battery_id.dropna().astype(str).unique()
        assert len(batteries)==1
        battery=batteries[0]; own=normalizer.curve_for(battery,drone)
        frame,audit=observed_hover_frames(g)
        position=int(drone.split("_")[-1])
        windows.append(dict(position=position,battery_id=battery,window_details=json.dumps(audit)))
        assert audit[0]["own_20_percent_observed"]
        for i,(stage,upper,lower) in enumerate(zip(NAMES,own.boundaries,own.boundaries[1:])):
            fit=fit_blocks(frame[frame.raw_soc.between(lower,upper)])
            if fit is None:continue
            records.append(dict(condition_id=CID,formation="diamond",wind_direction="head",wind_level=2,spacing_cm=50,
                                 protocol="wind_tunnel",experiment_id=EXPERIMENT,run_id="20260928_232343",
                                 position=position,drone_name=drone,battery_id=battery,stage=stage,
                                 source_file=str(SOURCE.relative_to(ROOT)),
                                 **calibrated_rate(fit,battery,drone,i,normalizer)))
    wind=pd.DataFrame(records)
    old_pilot=pd.read_csv(PREVIOUS_PILOT)
    matched=wind.merge(old_pilot[["position","stage","raw_rate_pp_min","normalized_rate_pp_min"]],
                       on=["position","stage"],suffixes=("_new","_old"),validate="one_to_one")
    wind_error=float(np.max(np.abs(matched.normalized_rate_pp_min_new-matched.normalized_rate_pp_min_old)))
    assert wind_error<1e-8

    f=pd.read_csv(forward_path)
    f=f[f.formation.eq("diamond") & f.wind_direction.eq("head") & f.wind_level.eq(2) & f.inter_drone_spacing_cm.eq(50)]
    forward_audit=f.copy()
    f=f[f.status.eq("included")].copy()
    for _,r in f.iterrows():
        own=normalizer.curve_for(r.battery_id,r.drone_name)
        expected=float(r.raw_medium_discharge_rate_pp_min)/own.rates_pp_min[1]*normalizer.reference.rates_pp_min[1]
        assert np.isclose(expected,r.discharge_rate_Bideal_pp_per_min,rtol=1e-9,atol=1e-9)
        records.append(dict(condition_id=CID,formation="diamond",wind_direction="head",wind_level=2,spacing_cm=50,
                          protocol="forward_250cm",experiment_id=r.experiment_directory,run_id=r.run_id,
                          position=int(r.position),drone_name=r.drone_name,battery_id=r.battery_id,stage="Medium",
                          raw_rate_pp_min=float(r.raw_medium_discharge_rate_pp_min),
                          normalized_rate_pp_min=float(r.discharge_rate_Bideal_pp_per_min),
                          own_baseline_rate_pp_min=own.rates_pp_min[1],reference_baseline_rate_pp_min=normalizer.reference.rates_pp_min[1],
                          duration_s=float(r.medium_forward_duration_s),sample_count=int(r.selected_interval_count)+1,
                          source_file=str(forward_path.relative_to(ROOT)),
                          reused_forward_rate_unchanged=True,legacy_forward_qc_flags=r.qc_flags,
                          weak_support=bool(r.medium_forward_duration_s<20 or r.raw_medium_drop_pp<3)))
    measured=pd.DataFrame(records)
    source_means=measured.groupby(["protocol"]+GRAIN).agg(rate_pp_min=("normalized_rate_pp_min","mean"),
                                                        valid_run_count=("normalized_rate_pp_min","size")).reset_index()
    prior_wide=pd.read_csv(FORWARD/"discharge_rates_wide.csv")
    prior_wide=prior_wide[prior_wide.formation.eq("diamond") & prior_wide.wind_direction.eq("head") & prior_wide.wind_level.eq(2) & prior_wide.inter_drone_spacing_cm.eq(50)].iloc[0]
    forward_means=source_means[source_means.protocol.eq("forward_250cm")].set_index("position")
    forward_error=max(abs(forward_means.loc[p,"rate_pp_min"]-prior_wide[f"position_{p}_discharge_rate_Bideal_pp_per_min"]) for p in range(1,6))
    assert forward_error<1e-8
    observed=combined_means(measured,"equal_protocol")
    completed=complete_curves(observed,normalizer)
    assert completed.position.nunique()==5 and len(completed)==15
    assert completed.normalized_rate_pp_min.gt(0).all()
    mids=observed[observed.stage.eq("Medium")]
    average_error=float(np.max(np.abs(mids.normalized_rate_pp_min-(mids.wind_mean_pp_min+mids.forward_mean_pp_min)/2)))
    assert average_error<1e-12
    knots=[]
    for position,g in completed.groupby("position",sort=True):
        by=g.set_index("stage")
        time=np.r_[0.,np.cumsum([by.loc[s,"stage_duration_s"] for s in NAMES])]
        for i,(t,soc) in enumerate(zip(time,normalizer.reference.boundaries)):
            knots.append(dict(condition_id=CID,position=int(position),knot=i,time_s=float(t),normalized_soc=float(soc)))
    OUT.mkdir(parents=True)
    measured.to_csv(OUT/"measured_stage_rates.csv",index=False)
    wind.to_csv(OUT/"wind_stage_rates.csv",index=False)
    pd.DataFrame(windows).to_csv(OUT/"wind_window_audit.csv",index=False)
    forward_audit.to_csv(OUT/"reused_forward_records_audit.csv",index=False)
    source_means.to_csv(OUT/"source_stage_means.csv",index=False)
    observed.to_csv(OUT/"averaged_observed_stage_rates.csv",index=False)
    completed.to_csv(OUT/"curve_stage_rates.csv",index=False)
    pd.DataFrame(knots).to_csv(OUT/"curve_knots.csv",index=False)
    comparison=completed.pivot(index="position",columns="stage",values="normalized_rate_pp_min").reindex(columns=NAMES)
    comparison["reference_100_to_20_s"]=completed.groupby("position").stage_duration_s.sum()
    comparison.to_csv(OUT/"five_position_summary.csv")

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,ax=plt.subplots(figsize=(11,6.8));plot_curve(ax,completed,normalizer)
    for name,upper,lower in zip(NAMES,normalizer.reference.boundaries,normalizer.reference.boundaries[1:]):
        ax.text(.985,(upper+lower)/2,name,transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=11,color="#64757F",
                bbox=dict(facecolor="white",edgecolor="none",alpha=.9,pad=2))
    fig.suptitle("Diamond · 50 cm · Headwind · Level 2",x=.11,y=.966,ha="left",fontsize=17,fontweight="semibold")
    fig.text(.11,.912,"Normalized three-stage discharge curve · Average rates first, then calculate time",fontsize=10.5,color="#52636C")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.16),ncol=3,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.11,right=.96,bottom=.23,top=.855)
    for ext in ("png","pdf","svg"):fig.savefig(OUT/f"averaged_three_stage.{ext}",dpi=300,facecolor="white")
    plt.close(fig)
    validation=dict(only_one_condition=True,raw_inputs_unchanged=all(sha(ROOT/p)==h for p,h in hashes.items()),
                    wind_stage_rates_match_approved_pilot_max_error=wind_error,
                    forward_aggregate_matches_existing_coefficients_max_error=float(forward_error),
                    medium_average_max_error=average_error,all_curves_monotone=True,modeled_stage_rows=2,
                    initial_raw_diagnostic_figure_in_dataset=False,active_training_dataset_modified=False,
                    plotted_condition=CID)
    assert validation["raw_inputs_unchanged"]
    write_json(OUT/"validation.json",validation)
    write_json(OUT/"manifest.json",dict(status="single_condition_pilot_awaiting_user_review",condition_id=CID,
        wind_source_experiment=EXPERIMENT,source_hashes=hashes,normalization_model_version=normalizer.version,
        normalization_model_sha256=normalizer.sha256,reference=normalizer.reference.to_dict(),
        averaging_policy="per-protocol repetition means, then equal 50/50 of the two observed stage means",
        source_stage_support="Previously computed 250-cm coefficients cover Medium only. High and Low retain measured wind-tunnel rates where available.",
        missing_stage_assumption="P3/P4 High: same relative drain factor as that position's averaged Medium, multiplied by the reference High rate. These are modeled coefficients, not measurements.",
        zero_missing_time="Initial wind hover plateau removed; own-drone windows retained. Already processed forward rates are reused unchanged.",
        safety_conditions_excluded=sorted(exclusion_ids()),measured_stage_rows=len(measured),completed_curve_stage_rows=15,
        old_forward_refitted=False,bulk_processing_paused=True,active_dataset_updated=False,trained=False,
        mixed_protocol_interpretation="Descriptive user-requested average of static wind-tunnel hovering and 250-cm forward-flight rates; not an independently validated forward-flight model."))
    (OUT/"README.md").write_text("""# 单条件试处理：Diamond / 50 cm / Headwind / Level2

仅交付此例，等待用户看图确认后再继续批量。没有更新训练数据、激活模型、改动实机代码或原始记录。实测SOC对照图不进入结果集，原始 `_002` 实测数据保留。

## 平均方式

风洞各物理电池按自身High/Medium/Low阶段独立拟合，去掉各机hover开头不掉电的平台，保留各机至自身20%的时间。已算好的250cm v3耗电率直接复用，不重拟合。两类已在相同参考电池上，每类先算重复实验均值，再各占50%。

**现有250cm系数仅覆盖Medium**，因此本例只在中电量阶段做两类实测平均。High和Low有风洞实测的使用风洞率；P3/P4没有High观测，曲线中按该位置平均后Medium的相对耗电因子完成High。这两行明确记录为模型补全，不进入实测拟合表。

以P1为例，中电量风洞8.703709 pp/min，已有250cm均值5.934877 pp/min，平均7.319293 pp/min。统一参考Medium区间82→52，所以阶段时间=60×30/7.319293秒。先平均率，再计算时间；不对两个实验的原始时长做平均。

图中的横轴是统一参考电池从100%到20%的计算时间，和起始电量不同的原始实验时长含义不同。每个位置的三阶段率、四个折点以及复核信息均保留。PNG为3300×2040，PDF/SVG为矢量。

这是风洞悬停与250cm前进两种协议的描述性平均。旧参考电池模型保持candidate状态，本次未独立验证其物理有效性。P2/P5的风洞High各只有两档SOC，保留原试处理斜率并标记支撑较弱。其他飞机先落地后的本机hover仍按用户要求保留，因此不保证后段一直有五架同时悬停。

## 文件

- averaged_three_stage.png / pdf / svg：这一个条件的五位置标准化三段曲线。
- source_stage_means.csv：分别列出风洞/已有250cm均值与数量。
- averaged_observed_stage_rates.csv：可观测阶段平均；High/Low单来源未伪装成两类平均。
- curve_stage_rates.csv：图使用的15个阶段率及来源，P3/P4 High为模型完成。
- curve_knots.csv：100/82/52/20四个折点。
- measured_stage_rates.csv：只含真实观测拟合或已有处理系数。
- wind_window_audit.csv / reused_forward_records_audit.csv：窗口与前进复用依据。
- manifest.json / validation.json：来源校验、平均公式及单条件范围复核。
""")
    # A local companion, deliberately not represented as an executed notebook.
    notebook={"cells":[{"cell_type":"markdown","metadata":{},"source":["# Diamond 50cm Headwind Level2 单条件复核\n","中段双来源各50%；两行缺失High模型完成来源可查看。"]},
       {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["from pathlib import Path\nimport pandas as pd\nimport numpy as np\n",f"folder = Path({str(OUT)!r})\n", "rates = pd.read_csv(folder/'averaged_observed_stage_rates.csv')\nmedium = rates[rates.stage.eq('Medium')]\nassert np.allclose(medium.normalized_rate_pp_min, (medium.wind_mean_pp_min+medium.forward_mean_pp_min)/2)\ndisplay(pd.read_csv(folder/'five_position_summary.csv'))\n"]}],
       "metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},"language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
    write_json(OUT/"review_pilot.ipynb",notebook)
    receipt={"schemaVersion":1,"items":[{"id":"single-condition-averaged-stages","title":"Diamond 50cm：已有250cm耗电率与风洞阶段率平均",
      "assumptions":["P3/P4未观察到High，图中High沿用对应位置平均后Medium的相对耗电因子；这些是模型系数，不是实测样本。"],
      "queries":[{"id":"source-means","source":{"label":"Diamond 50cm 顶风 Level2 · 风洞 _002 / 既有250cm v3",
       "filters":["配置：Diamond / 50cm / Headwind / Level2","位置：P1–P5分别处理","风洞：去除初始不掉电平台，各机到自身20%","前进：复用已有v3中电量耗电率，保持系数不变"],
       "metricDefinitions":[{"id":"rate-mean","definition":"每个位置先取风洞与前进各自的重复实验均值，再将同阶段两类均值各取50%；只有一类有实测的阶段保留单来源率。"}],
       "caveats":["已有250cm耗电率只覆盖Medium，中段有两类实测平均，高低段不能声称都有双来源平均。","风洞悬停与前进是不同实验协议，本图是用户要求的描述性组合，未作新的实机验证。","横轴为统一参考电池100%至20%的计算时间，不等于不同起始SOC的原始实验时长。","P2/P5的High只有两档SOC，阶段斜率支撑较弱。"]},
       "rows":observed[["position","stage","wind_mean_pp_min","forward_mean_pp_min","normalized_rate_pp_min","wind_run_count","forward_run_count"]].replace({np.nan:None}).to_dict("records"),
       "columns":[{"field":"position","label":"位置"},{"field":"stage","label":"电池阶段"},{"field":"wind_mean_pp_min","label":"风洞率（pp/min）"},
                  {"field":"forward_mean_pp_min","label":"已有250cm率（pp/min）"},{"field":"normalized_rate_pp_min","label":"最终观测阶段率（pp/min）"},
                  {"field":"wind_run_count","label":"风洞组数"},{"field":"forward_run_count","label":"前进组数"}]}]}]}
    write_json(OUT/"answer_sources.json",receipt)
    print(comparison.round(6).to_string())
    print(json.dumps(validation,indent=2))
    print(str(OUT/"averaged_three_stage.png"))


if __name__=="__main__":main()
