# 耗电率冻结版本：medium_bideal_v1_20260908

本版固定当前已确认的medium耗电率参考表，不重新拟合、不改原始实验、不重新训练模型，也不切换现有Online代码的数据路径。未来修改必须生成新版本，冻结脚本拒绝覆盖同名目录。

## 主文件

- `discharge_rates.csv`：55组条件配置 × 5个position，共275条Bideal耗电率，供查看。
- `rate_reference.json`：与当前训练参考表逐字节相同，供程序读取。
- `battery_calibration.csv`：冻结B10–B15共6个电池的原Bideal转换参数；没有B06参数，不能默认补1。
- `coverage.csv`：60组理论条件配置逐一列明available、unsafe_excluded、missing_data。
- `source_runs.csv`、`source_run_drone_rates.csv`：134次源实验、670条单机处理结果的快照，保留QC与元数据字段。
- `manifest.json`：单位、假设、源文件/冻结文件哈希及版本。

## 固定口径

来源为已经处理的medium前进实验，55个参考单元、134次五机实验。同风向/风力等级/formation/spacing/position，对源实验已拟合的Bideal耗电率按run等权平均。导出值直接取自既有参考表；重算均值只用于一致性检查。

单位`pp/min`表示每分钟下降多少电量百分点，不是相对百分比，也不是瓦特或瓦时。`pp/second`仅为除以60后的同值换算。wind_level是原有等级1/2，本版没有虚构对应m/s。

适用模型SOC区间40%–75%，采用当前确认的区间内线性假设；不自动外推到high/low。position是formation中的slot1..5，不是无人机编号或Mission Pad编号。charging pad数量不属于耗电率表的维度。

电池换算：`a = scale_physical_drop_to_Bideal`；`Bideal_drop = physical_drop × a`；某机处于位置p时，`physical_rate = frozen_Bideal_rate[p] / a_own_battery`。不同位置转用该耗电率仍是Bideal统一化后的建模假设，不代表实际完成过全部位置交换实验。

本版冻结的是放电参考及其电池转换依据，不包含充电模型或总时间标签。Bideal转换基准仍是既有75%→30%/36%的baseline拟合，未重新拟合为严格medium，范围差异明确保留。

## 安全与缺失

只排除这些明确的风况/队形/间距组合，不排除整个formation：head lv2 column50；side lv2 column50、diamond50；tail lv2 diamond50。

side lv1 column50当前没有已处理medium参考记录，保留missing_data，不当成不安全，也不填0。其他QC标记只作警告，按用户确认保留参与计算。短时间SOC拟合率为0不等于实际零耗电；有单次来源的单元，其跨run标准差为空而非0。

实验标定长度为250cm；原forward时间由逐机清理得到，本版没有将轨迹修改成共同25秒或假称实际速度严格一致。用耗电率推算后续飞行应保留这一限制，不能把wind tunnel悬停和本版前进数据混同。

## 如何保持冻结

Offline与Online后续应显式引用本版路径，不读取会自动更新的临时汇总。当前只交付快照，没有自动切换任何消费者。校验命令：

```text
python3 -m ml_policy.freeze_discharge_rates verify --directory frozen_data/discharge_rates/medium_bideal_v1_20260908
```

版本号不变时不应编辑这些文件；文件哈希能够检测变更，属于版本快照约定，并非操作系统级不可修改锁。若要增加数据、修改率值或模型适用范围，请生成v2。

数据验证技能用于核对所有率值与来源、Bideal系数和覆盖表；冻结表示固定可追溯的一版计算依据，不表示消除了测量不确定性或完成了独立实飞验证。
