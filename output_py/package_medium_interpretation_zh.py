"""Chinese edition of the complete reviewed report; numerical results unchanged."""
import copy
import hashlib
import json
from datetime import datetime, timezone

from output_py.interpret_medium_decisions import OUT


WIND = {"head": "逆风", "side": "侧风", "tail": "顺风"}


def condition(wind, strength):
    return ("低强度" if strength == "low" else "高强度") + WIND[wind]


def critical_text(r, prefix):
    return f"{r[prefix + '_critical_pad']}，充电 {r[prefix + '_critical_pad_seconds']/60:.2f} 分钟"


def comparison_zh(r):
    a, b = r["selected_configuration"], r["comparator_configuration"]
    sa, sb = r["selected_swarm_drop_pp"], r["comparator_swarm_drop_pp"]
    relation = "较低" if sa < sb else "较高"
    delta = abs(sa - sb) / sb * 100
    percent = f"{delta:.1f}%" if delta >= .05 else f"{delta:.3f}%"
    return (f"与 {b} 相比，{a} 的计算总时间减少 {r['gap_s']:.2f} 秒。"
            f"25 秒内五架无人机的标准化 SOC 消耗之和为 {sa:.3f} 个百分点，对方为 {sb:.3f} 个百分点，{relation} {percent}。"
            + ("这组比较中，总时间优势伴随着更低的全队标准化耗电。" if sa < sb else
               "这组比较中，总时间优势来自充电负担的分布，而不是全队耗电更少。")
            + f"最终完成充电的充电位分别承担 {critical_text(r, 'selected')}；对方承担 {critical_text(r, 'comparator')}。"
            + ("两者间距相同。" if r["selected_spacing_cm"] == r["comparator_spacing_cm"] else
               "这组比较同时改变了队形和间距；另附相同间距的比较表。"))


MECHANISMS = {
    ("head", "front"): "从布局看，Front 横向展开，避免无人机沿来流方向连续串列；相较 Column，少了连续的前后机尾流相遇，同时横向距离仍会改变相邻旋翼的来流。遮挡减少的阻力与尾流造成的推力变化需要一起考虑。",
    ("head", "echalon"): "Echelon 在前后错开的同时增加横向偏移，可使后方旋翼避开部分直接下洗流，并改变其接触尾流核心或边缘的方式。这会改变维持推力所需的电机输出以及各位置之间的负担分布。",
    ("side", "front"): "侧风下，Front 横排沿侧向来流形成前后遮挡关系；迎风侧无人机能为后方位置提供一定遮挡，同时也改变后方旋翼来流。因此既要看抗风需求，也要看尾流与位置修正带来的负担。",
    ("side", "echalon"): "Echelon 的斜向错位使相邻无人机在侧风方向和横向都存在偏移，与 Front 的直线遮挡不同；它改变各位置接收到的尾流及需要补偿的力和力矩，因而可能产生明显低耗电的位置。",
    ("side", "column"): "Column 纵向排布时，相对于侧风的横向分量，各机更接近并排受风，减少连续沿侧风方向的遮挡链与尾流串列作用。这种布局更适合从最不利位置的负担来分析，而不能只看全队平均耗电。",
    ("side", "diamond"): "Diamond 同时具有中心、前后和横向位置，遮挡与尾流相互作用分布在多个方向。某些位置较低的负担可以改善共享充电位的排程，但并不代表每个位置都更省电。",
    ("tail", "vee"): "顺风下，Vee 两臂相对于来流的前后关系与逆风不同，遮挡及尾流接触位置随之变化。这里还必须结合 P2 接近零的整数 SOC 差值理解：这个小数值参与计算，但不是该位置不消耗实际能量。",
    ("tail", "diamond"): "Diamond 将位置同时分布在前后和两侧，改变顺风下的遮挡与旋翼来流，使最耗电位置与低耗电位置的分布不同于 Vee 或 Column。判断其优势要看充电瓶颈是否降低。",
    ("tail", "column"): "Column 在纵向形成连续遮挡，也使各位置依次受到前方机体与尾流影响。遮挡与不均匀旋翼来流共同作用，可以形成部分位置负担低、个别位置负担高的分布；顺风时不能照搬逆风的领航机解释。",
    ("tail", "echalon"): "Echelon 的侧向错位改变顺风搬运的尾流与相邻旋翼相遇的位置。相比直线 Column，这种几何变化可以重新分配各机负担，尤其影响最大耗电率及最后完成充电的无人机。",
}


OVERVIEW = {
    ("head", "low"): "低强度逆风下，1–4 个充电位选择 Front 75 cm，5 个充电位改为 Echelon 75 cm。Front 的五个位置耗电率为 [1.639, 10.830, 0.967, 2.468, 7.344] Bideal 百分点/分钟，其中 P1、P3、P4 较低，利于减少共享充电位上的任务时间。但 P2 的 10.830 较高；五个充电位时，一机一位，低耗电位置不能再缩短 P2 自身的充电时间。Echelon 75 cm 将最大耗电率降至 8.395，因此取代 Front。一个具体对照是：Echelon 的全队标准化消耗之和反而较高，但五位充电的总时间比 Front 少 41.02 秒。",
    ("head", "high"): "高强度逆风下，Echelon 75 cm 在 1 个和 5 个充电位时最优，Front 50 cm 在 2 个和 4 个时最优，Echelon 50 cm 在 3 个时最优。Echelon 75 cm 的最大耗电率为 8.281，低于 Front 50 cm 的 11.067，因此有利于五位并行充电。Front 50 cm 的 P3、P5 分别只有 1.342、3.993，利于两个或四个充电位下组合较短的任务。三位充电时，Echelon 50 cm 的两组共享任务组合更有利。这不是风在不同充电位数量下改变了，而是排程目标对同一耗电分布的评价改变了。",
    ("side", "low"): "低强度侧风下，1–3 个充电位选择 Front 50 cm；4 个时选择 Diamond 75 cm；5 个时选择 Column 75 cm。Front 的 P3、P4 分别为 4.818、4.535，较短的任务帮助减少共享充电位负担。Diamond 的 P3、P5 为 3.499、5.545，这一短任务组合在四位充电时更有利。五位充电时，Column 将最大耗电率从 Front 的 10.910 降至 10.819，但总时间只少 1.50 秒，应表述为计算结果接近，而不是 Column 大幅领先。四位时 Diamond 相比 Front 也只快 5.92 秒。",
    ("side", "high"): "高强度侧风下，Front 50 cm 在 1、3、5 个充电位时最优，Echelon 50 cm 在 2、4 个时最优。Front 的各位置耗电率集中在 5.399–6.458，分布较均匀；Echelon 的 P3 低至 1.623，但 P4 达到 8.653。因此 Front 对限制最大负担有利，Echelon 则能利用一个很短的充电任务优化部分共享排程。两个充电位时，Echelon 比 Front 少 49.05 秒，尽管其全队标准化消耗之和略高。这是“平均或整体耗电”不能单独决定配置的直接例子。",
    ("tail", "low"): "低强度顺风下，1、2、4 个充电位选择 Vee 50 cm，3、5 个时选择 Diamond 75 cm。Vee 的 P2 耗电率约为 0.019，而 P4 达到 13.253，因此耗电分布非常不均匀；这个接近零的值来自短记录中的整数电量变化，不能解释为无人机几乎不用能量。Diamond 的最大耗电率约 9.770，五位充电时能降低最慢一架的负担。两者 25 秒的全队标准化消耗之和都是约 15.05 个百分点，几乎相同，却在不同充电位数量下各有优势，说明决定结果的主要区别在于五个任务如何组合。",
    ("tail", "high"): "高强度顺风下，Column 50 cm 在 1–4 个充电位时最优，5 个时改为 Echelon 50 cm。Column 的 P3、P4、P5 为 0.961、2.849、5.253，利于降低共享任务组合的时间；但 P1 为 12.579，是明显的高负担位置。五位充电时，Echelon 的最大耗电率降至 9.547，相比 Column 总时间减少 50.04 秒。需要区分这一对照与第二名：Echelon 相比 Diamond 75 cm 只快 0.68 秒，因此它与 Diamond 在当前计算中非常接近。",
}


def main():
    original_path = OUT / "artifact.json"
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    a = copy.deepcopy(original)
    source = json.loads((OUT / "explanations.json").read_text())
    refs = source["reviewed_literature"]
    cite = {r["id"]: f"[{r['id']}]({r['url']})" for r in refs}
    title = "Medium 阶段配置选择：中文分析"
    a["manifest"]["title"] = title
    a["manifest"]["generatedAt"] = a["snapshot"]["generatedAt"] = datetime.now(timezone.utc).isoformat()
    translated = {}
    driver = {1: "五架无人机充电时间的总和", 2: "共享充电位上三个较短任务的总时间",
              3: "优化分组后，两组双机充电任务中较长的一组", 4: "两个最短任务共享充电位的时间",
              5: "最长的单机充电时间"}
    for key, d in source["decisions"].items():
        k = d["input"]["charging_pad_availability"]
        w = d["input"]["wind_direction"]
        strength = d["input"]["wind_strength"]
        group = "、".join(f"P{i}" for i in d["bottleneck_positions"])
        text = (f"选择 **{d['paper_configuration']}**，计算总时间为 **{d['total_time_minutes']:.2f} 分钟**，"
                f"比第二名 {d['runner_up_configuration']} 少 {d['gap_to_runner_up_seconds']:.2f} 秒。"
                f"在这一充电位数量下，关键是{driver[k]}；最后完成的充电位承担 {group}。")
        text += "\n\n" + MECHANISMS[(w, d["configuration"]["formation"])]
        text += "\n\n" + comparison_zh(d["cross_formation_comparisons"][0])
        text += " " + " ".join(cite[r] for r in d["reference_ids"])
        translated[key] = dict(input=d["input"], configuration=d["configuration"],
                               total_time_minutes=d["total_time_minutes"], explanation_zh=text)

    bodies = {
        "title": "# " + title,
        "summary": "## 核心结论：最省电的配置，不一定让全队最快完成充电\n\n"
            "当前结果给出了 30 个条件下的配置选择。风向、风力、队形和间距共同对应不同的逐位置耗电率；充电位数量决定这些耗电差异如何转化为全队完成充电的时间。"
            "因此没有一种 formation 在所有条件下都最优，也不能只根据全队平均耗电或某一架无人机的耗电判断。\n\n"
            "在 120 组跨 formation 比较中，112 组的选择既使总时间更短，又具有更低的全队标准化 SOC 消耗；另有 8 组总时间更短，但全队标准化消耗更高。"
            "后者的优势在于充电负担的分布。本版仅将已完成的分析整理成中文，不改动原始数据、耗电率或最优配置。",
        "baseline": "## 比较前提：统一 Medium 起点后，再比较配置\n\n"
            "分析使用此前 250 cm 前进实验：134 次真实飞行中，有 630 条可用的无人机–飞行 Medium 记录，汇总为 55 组队形、间距与风况组合，每组包含五个位置的耗电率。"
            "不加入 wind tunnel 仿真数据。四组安全排除组合仍排除；低强度侧风的 Column 50 cm 因缺少合格真实数据也不参与。\n\n"
            "先按每块电池自己的 Medium 区间筛选并通过 Bideal 标准化，再平移到统一的 Bideal 75% 起点；这属于前置处理，不是函数输入。"
            "比较段固定 250 cm、25 秒，耗电率单位为 Bideal 百分点/分钟。全队标准化消耗指五架无人机 SOC 下降量之和，不是直接测量的焦耳或瓦时。\n\n"
            "总时间为固定段飞行时间加上全队完成充电的时间。充电沿用既有指数模型：0→99% 为 90 分钟，目标 99%；充电位相同且到达时可用，不中断充电，不计等待期间耗电和配置切换代价。"
            "因此报告里的总时间是该模型下的计算结果，不是现场实测充电时间，也不是整个多段任务的全局最短时间。",
        "labels": "## Low / High 表示风力，不表示电池阶段\n\n"
            "论文中用 low wind、high wind 分别对应原来的 Lv1、Lv2；这里写作低强度风、高强度风。它们是实验风扇档位，不换算成未测量的 m/s。"
            "电量区间称 Medium SOC。队形拼写用 Echelon，保留历史代码键 echalon。",
        "map": "## 充电位数量改变，最优配置也会改变\n\n"
            "下面每行是一种风况，每列为 1–5 个可用充电位。表中给出在现有安全且有真实数据的配置中，使计算总时间最短的 formation 与间距。"
            "同一行出现不同队形，并不是气动环境改变，而是同一耗电分布在不同充电排程下的效果改变。",
        "map_figure": "## 论文 Decision Map 与本版结果保持一致\n\n"
            "已有论文图覆盖同样的 30 个格子，横轴是充电位数量，纵轴是风向及 low/high 风力，每格直接标注队形和间距。"
            "PDF、SVG、600 dpi PNG 和英文图注仍保留。本次只增加中文解读，不改论文图中的决策。",
        "math": "## 函数如何从耗电率得到最短总时间\n\n"
            "对于候选配置 c，第 i 个位置的耗电率为 rᵢ，则到达 SOC 为 sᵢ = 75 − 25rᵢ/60。既有充电模型给出单机充电时间 qᵢ = [5400/ln(100)] ln(100−sᵢ)，单位为秒。"
            "把五个充电任务分配给 k 个充电位，使最晚完成的充电位尽早完成，再加上固定的 25 秒飞行时间。\n\n"
            "**F(充电位数量, 风向, 风力) = 计算总时间最短的 configuration。**\n\n"
            "在本数据的单机充电时间范围内，把五个任务从短到长排序为 q(1) 到 q(5)，可以这样理解：\n\n"
            "- 1 位：五个任务总和。\n- 2 位：三个最短任务合用一位，另外两个合用另一位；前一组决定完成时间。\n"
            "- 3 位：最长任务单独一位；另外四个优化配对，比较两组较长者。\n"
            "- 4 位：两个最短任务共享一位，其余各一位。\n- 5 位：所有无人机并行充电，最长单机任务决定完成时间。\n\n"
            "全部 275 个候选得分已用穷举分组核对。中间三个简化描述依赖当前任务时长接近的特点，不能直接用于任意不相等的实时 SOC。",
        "literature": "## 气动解释要连接到旋翼负担，再连接到耗电\n\n"
            "解释队形差异时，需要同时考虑机体受风、旋翼来流以及维持姿态和路径的修正，而不能只说“被前机挡住，所以更省电”。"
            "遮挡可能减少机体阻力，但前机尾流也会改变后机产生推力的效率，两种作用可能相互抵消。"
            "相关队形研究中，单独考虑阻力和加入旋翼气动力后的排名就不同。" + cite["R7"] + "\n\n"
            "单机风洞和飞行研究支持风、姿态与电机负担共同影响功耗；邻近飞行研究支持相对位置改变扰动力与控制需求。"
            "本文用这些机理解释测得的耗电分布，数值排名仍由自己的真实耗电率与充电计算决定，不照搬其他机型或固定翼 V 队形的结论。"
            + " ".join(cite[r] for r in ("R1", "R2", "R4", "R5")),
        "mechanism_load": "## 抗风既涉及推力，也涉及产生推力的效率\n\n"
            "在近似稳定、等高飞行中，无人机需同时支撑重量并平衡水平气动力。忽略机体竖直气动力时，有 T cosθ = mg、T sinθ = D。"
            "水平阻力增大需要更多总推力；但来流也会改变旋翼诱导功率，因此总功耗不是仅由阻力大小决定。" + cite["R1"] + " " + cite["R2"],
        "mechanism_wake": "## 前机尾流接触后机旋翼的位置很重要\n\n"
            "直接下洗流与尾流边缘的作用不同。旋翼台架研究中，串列下洗流会降低后方旋翼推力，而合适的斜向偏移可改善旋翼载荷。"
            "为了维持相同的必要推力，电机需要不同的转速和功率。这就是横向错位不仅仅是“拉远一点”的原因。" + cite["R10"],
        "mechanism_control": "## 纠偏动作也体现在最终耗电中\n\n"
            "不均匀的气动力和力矩会引起差动电机调节，用来维持姿态、位置和飞行路径。"
            "因此真实耗电率综合反映了推进和控制修正，不能把两架无人机的耗电差全部当成单一阻力差。" + cite["R2"] + " " + cite["R5"],
        "mechanism_direction": "## 顺风不能简单理解成逆风耗电的相反数\n\n"
            "相对空气速度等于地速减去风速。固定地速为 0.1 m/s 时，如果顺风速度超过地速，纵向相对来流会反向，无人机还需要避免被吹得过快。"
            "这会改变上下游关系以及尾流与相邻旋翼相遇的位置，所以不能把逆风下某个编号的“领航机”角色直接用于顺风。" + cite["R6"],
        "mechanism_spacing": "## 75 cm 不一定比 50 cm 更省电\n\n"
            "增大距离可能减轻直接尾流干扰，也可能削弱有利遮挡，或使旋翼离开较有利的来流区域。"
            "因此间距的作用取决于队形和风向，不能设置“间距更大必然更优”的统一规则。" + cite["R4"] + " " + cite["R7"],
        "geometry": "## P1–P5 是记录位置，不等于固定的气动角色\n\n"
            "本报告采用旧前进实验耗电表的 P1–P5，不把新 wind tunnel 的 Mission Pad 布局套回旧数据。"
            "不同队形的同一位置编号，未必都是最迎风或最受遮挡的位置。以下将数据比较与布局机理对应起来，但不把未单独测量的阻力、尾流功率和纠偏功率拆成额外实验数值。",
        "cross_formation_intro": "## 与其他队形比较：不能把时间优势全叫作省电优势\n\n"
            "30 个决策分别与其他四种 formation 的最佳可用间距比较，共 120 组；另有 99 组固定间距比较，缺失和安全排除的组合不补造。"
            "120 组中有 8 组，选中的配置全队标准化耗电更高，但总时间仍更短；这是充电负担分布的优势。这些比较复用了同一批耗电率，不是 120 次独立实验。\n\n"
            "例如低强度逆风、五个充电位：Echelon 75 cm 的全队 25 秒标准化消耗为 14.601 个百分点，Front 75 cm 为 9.687；"
            "然而前者最大位置耗电率更低，因此总时间快 41.02 秒。同一风况只有一个充电位时，Front 75 cm 又成为最优。",
        "uncertainty": "## 方法备注：部分配置的计算时间非常接近\n\n"
            "五个充电位时，低强度侧风第一名只比第二名快 1.50 秒，高强度顺风只快 0.68 秒。这些应描述为结果接近，不写成大幅领先。"
            "下图展示六种风况在五位充电时的第一、第二名时间差。\n\n"
            "已有逐次删除完整飞行的检查显示，30 个状态中有 25 个会在某次删除后改变选择；这反映了重复实验均值对排名的影响。详细检查与正文的物理解释分开呈现。",
        "bootstrap": "## 方法备注：以完整飞行为单位复核结果\n\n"
            "此前复核以整次飞行为单位联合重抽样五个位置，保留同次飞行内的关联。每种风况尝试 2,000 次，候选位置数据完整的有效次数为 1,162–1,863。"
            "表中展示原配置在有效重抽样中被保留的频率，以及逐次删除完整飞行后的保留次数。只有一次观测的位置不能凭重抽样增加独立信息；电池标定和空间风场误差未在其中传播。",
        "checks": "## 方法备注：充电模型检查与间距对照\n\n"
            "此前分别改用恒流充电到 99%、指数充电到 80% 和 90%，三组检查均保持当前 30 个配置选择，但绝对时间不同。"
            "这不把假定的充电曲线变成实测曲线。\n\n"
            "25 组具有两个间距数据的 formation–风况组合中，75 cm 的平均耗电率更低有 12 组，最大耗电率更低也有 12 组。"
            "因此当前数据并不呈现所有条件下 75 cm 都优于 50 cm 的规律。",
        "next": "## 当前进展怎样向老师汇报，之后怎样接入动态决策\n\n"
            "当前已经完成：从处理后的真实数据计算 Medium 逐位置耗电率，统一 Bideal 起点，建立三输入配置函数，并给出数据和气动机理相结合的理由。"
            "主要结论是：风决定逐位置负担，充电位数量决定哪种负担分布能最小化总时间。\n\n"
            "接下来若回到动态任务，则把当前五架无人机各自的 SOC 代入同一条“耗电率→到达 SOC→充电排程→总时间”计算链。"
            "本次固定起点的 decision map 不能直接替代不相等 SOC 下的选择。后续还需要明确实际充电曲线、配置切换时间以及跨段目标如何进入总时间；本中文版本不新增训练或修改这些假设。",
        "refs": "## 文献来源\n\n" + "\n\n".join(f"**{r['id']}**. {r['authors']} ({r['year']}). [{r['title']}]({r['url']}). {r['venue']}。" for r in refs),
    }
    for wind in ("head", "side", "tail"):
        for strength in ("low", "high"):
            prefix = f"{wind}_{strength}"
            bodies[prefix + "_intro"] = f"## {condition(wind, strength)}：为什么配置会这样变化\n\n" + OVERVIEW[(wind, strength)]
            for k in range(1, 6):
                key = f"{k}|{wind}|{1 if strength=='low' else 2}"
                bodies[f"{prefix}_{k}_reason"] = f"### {k} 个充电位：{source['decisions'][key]['paper_configuration']}\n\n" + translated[key]["explanation_zh"]
    used = set()
    for b in a["manifest"]["blocks"]:
        if b["type"] == "markdown":
            b["body"] = bodies[b["id"]]
            used.add(b["id"])
    assert used == set(bodies)
    labels = {"Wind":"风况", "Pads":"充电位数", "Configuration":"配置", "Total min":"总时间（分钟）", "Runner-up":"第二名", "Gap s":"领先时间（秒）",
              "Selected":"所选配置", "Comparator":"比较配置", "Selected drop pp":"所选消耗和（百分点）", "Comparator drop pp":"比较消耗和（百分点）",
              "Extra time s":"比较配置多耗时（秒）", "Selected P1-P5 rates":"所选 P1–P5 耗电率", "Comparator P1-P5 rates":"比较 P1–P5 耗电率",
              "Selected last pad":"所选最后完成组", "Comparator last pad":"比较最后完成组", "Comparative explanation":"中文比较说明",
              "Nominal winner":"原最优配置", "Bootstrap retained":"重抽样保留频率", "Valid draws":"有效次数", "LOO retained / valid":"删除检查保留／有效", "Min trials":"最少有效飞行数"}
    labels.update({f"{k} pad" + ("s" if k > 1 else ""):f"{k} 个充电位" for k in range(1, 6)})
    table_titles = {"Selected configuration":"配置决策表", "Computed decisions and margins":"配置及计算时间", "Other formations at their best available spacing":"其他队形采用自身最佳可用间距的比较",
                    "Other formations at matching spacing":"相同间距下的队形比较", "Decision stability audit":"重复实验复核"}
    for t in a["manifest"]["tables"]:
        t["title"] = table_titles[t["title"]]
        for c in t["columns"]:
            c["label"] = labels[c["label"]]
            if c["field"] == "explanation_en":
                c["field"] = "explanation_zh_full"
    for rows in a["snapshot"]["datasets"].values():
        for row in rows:
            if "wind_direction" in row and "wind_strength" in row:
                row["condition"] = condition(row["wind_direction"], row["wind_strength"])
            elif "condition" in row:
                level, wind = row["condition"].split()
                row["condition"] = condition(wind[:-4], level.lower())
            if "comparison_mode" in row:
                row["explanation_zh_full"] = comparison_zh(row)
            elif "key" in row and row["key"] in translated:
                row["explanation_zh_full"] = translated[row["key"]]["explanation_zh"]
    chart = a["manifest"]["charts"][0]
    chart.update(title="五个充电位时的配置时间差", subtitle="第一名比第二名快多少秒；数值较小表示两种配置接近")
    chart["encodings"]["x"]["label"] = "风况"
    chart["encodings"]["y"]["label"] = "总时间差（秒）"
    # Keep the English original and every analytical row/number, section and source.
    assert [b["id"] for b in a["manifest"]["blocks"]] == [b["id"] for b in original["manifest"]["blocks"]]
    assert a["sources"] == original["sources"]
    for key, rows in original["snapshot"]["datasets"].items():
        assert len(rows) == len(a["snapshot"]["datasets"][key])
        for before, after in zip(rows, a["snapshot"]["datasets"][key]):
            for field, value in before.items():
                if isinstance(value, (int, float, bool)):
                    assert after[field] == value, (key, field)
    (OUT / "artifact_zh.json").write_text(json.dumps(a, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (OUT / "explanations_zh.json").write_text(json.dumps(translated, ensure_ascii=False, indent=2) + "\n")
    text = []
    for b in a["manifest"]["blocks"]:
        if b["type"] == "markdown":
            text.extend([b["body"], ""])
        elif b["type"] == "table":
            t = next(t for t in a["manifest"]["tables"] if t["id"] == b["tableId"])
            if t["id"].startswith("cross_formation"):
                text.extend(["完整逐组数据见对应跨队形比较文件；原始数值不变。", ""])
                continue
            cols = t["columns"]
            text.append("| " + " | ".join(c["label"] for c in cols) + " |")
            text.append("| " + " | ".join("---" for c in cols) + " |")
            for row in a["snapshot"]["datasets"][t["dataset"]]:
                values = [f"{row[c['field']]:.3f}" if isinstance(row[c['field']], float) else str(row[c['field']]) for c in cols]
                text.append("| " + " | ".join(values) + " |")
            text.append("")
    (OUT / "thesis_explanations_zh.md").write_text("\n".join(text) + "\n")
    comparison_text = ["# 跨队形比较：中文解读", "", "同一风况和充电位数量下，每种其他队形采用自身最佳可用间距。耗电率和时间沿用已冻结结果。", ""]
    for key, d in source["decisions"].items():
        comparison_text.extend([f"## {condition(d['input']['wind_direction'], d['input']['wind_strength'])}，{d['input']['charging_pad_availability']} 个充电位：{d['paper_configuration']}", ""])
        for row in d["cross_formation_comparisons"]:
            comparison_text.extend([comparison_zh(row), "所选位置耗电率："+row["selected_rate_profile"]+"。比较配置位置耗电率："+row["comparator_rate_profile"]+"。单位均为 Bideal 百分点/分钟。", ""])
    (OUT / "cross_formation_comparisons_zh.md").write_text("\n".join(comparison_text) + "\n")
    assert original_path.read_bytes() == original_bytes
    plan = dict(language="zh-CN", surface="mcp-app", audience="technical", source_sha256=hashlib.sha256(original_bytes).hexdigest(),
                reporting_job="中文整理既有 Medium 配置结果，支持与导师讨论；不重新估计数据或改变结果",
                structure_mapping={"title":"title", "technical_summary":"summary", "findings":"map, six wind sections, cross_formation_intro", "scope":"baseline, labels", "methodology":"math, mechanism sections", "limitations":"geometry, uncertainty, bootstrap, checks", "next_steps_and_questions":"next"},
                chart_contract={"question":"五位充电时最优与次优配置差多少秒", "family":"comparison", "type":"single-series bar", "rows":6, "palette":"single blue root", "non_color":"category labels and values", "scope":"existing chart translated; no numerical changes", "table_rationale":"exact 30-cell decision lookup and detailed pairwise audit"},
                preserved=dict(blocks=len(a["manifest"]["blocks"]), tables=len(a["manifest"]["tables"]), sources=len(a["sources"]), datasets=len(a["snapshot"]["datasets"]), decisions=len(translated), numerical_values="unchanged"),
                english_original="preserved; independent Chinese edition")
    (OUT / "report_plan_zh.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(plan["preserved"], ensure_ascii=False))


if __name__ == "__main__":
    main()
