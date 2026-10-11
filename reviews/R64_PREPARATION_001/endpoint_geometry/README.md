# R64 端点几何准备包

本目录只完成源码、协议草案和人工检查；没有登记或运行 R64 科学任务，没有读取科学数组、权重或 DEV/EVAL/CAL split。

`endpoint_geometry.py` 是纯 NumPy、无读写/随机数/拟合/前向/梯度的分析模块。唯一入口返回 `(summary, private_arrays)`；由主助手另行登记的 worker 保存结果。

```python
summary, arrays = analyze_endpoint_geometry(
    source_out_preADC, source_canonical, canonical_REAL,
    source_raw, REAL_raw, source_bins, REAL_bins, source_charge,
    statistics, directions, temporal_directions, matched_real_indices,
    REAL_control_canonical=optional_control,
)
```

输入约定：

- `source_out_preADC` 是原 `source_flow_core.endpoint` 的直接投影端点；`source_canonical` 是同一值经过实际 `canonical_adc_torch(Y, False)` 的结果。**本轮不以 R62 `traced_solve` / Heun 输出定义 Y。** 代码核对二阶段 ADC 一致性。
- 原 SOURCE/REAL 按保存顺序为 6593/21047 行、480 点。`source_raw` / `REAL_raw` 保留原 float32 值；`source_charge` 必须是原 SOURCE 经 float32→float64 后的 `q_np`。单位为原代码 V samples，不能传入 `1000*q`。
- `source_bins` / `REAL_bins` 是原始、未裁剪 raw-bin `[-1,0,...,8]`，不是网络条件 bin；所有十区都保留。worker 负责原 ID、顺序、TRAIN、raw-bin 与冻结输入的哈希绑定。模块验证人数、各区人数和原 q；它不能单凭数组内容验证身份。
- `statistics` 是 R63 冻结常数映射：`source_counts`, `REAL_counts`，旧 `V480/U480/DU479/D2U478_center/scale`（10×维数），新增 `temporal_center/scale`（10×31）；若有 `temporal_REAL_counts` 也会核对。这里不会重新估计任何中心、尺度或边界。
- `directions` 是四个 float32 `(480/480/479/478,64)` 固定方向；`temporal_directions` 是 float32 `(31,64)`。模块只验证，不归一化或重抽。worker 应让几何与梯度模块共享相同方向及端点语义。
- `matched_real_indices` 必須完整包含 `SCOPES=('all','region-1',...,'region8')`。每项为 `{'A': array, 'B': array}`，每个 array 是 `(32, 该 scope 原 SOURCE N)` 的**全局 REAL 行号**。由 root 预先固定、独立、有放回抽 A/B；重复父保留其每次出现的质量。区内抽样不可跨原 raw-bin。`all` 从全部 REAL 等权抽样。
- 可选 `REAL_control_canonical` 是全部 21047 REAL 经同一直接端点映射及生成 ADC 的独立控制；它不替换 `canonical_REAL`，后者一直是原 REAL 的规范 ADC 参照。

固定输出覆盖所有十区及全人口：四旧视图和 31 维的逐坐标 W1、固定 64 投影 W1、坐标均值/RMS/最大值、源原质量加权的 `full_window` / `coordinate_RMS` / `temporal_joint` 贡献；另保存未标准化坐标 W1、原/输出/REAL 电荷 min/10%/50%/90%/max、均值、SD、完整电荷 W1、preADC 与 ADC 电荷变化和下界分量。所有逐坐标、投影、32 个 REAL 底噪值及原抽样索引进入 `private_arrays`，不选择性保留结果。

`all` 的标准化统计按每父原 raw-bin 使用冻结常数后混合，它和原区域质量加权目标不同；两者明确分别输出。REAL 底噪是 A–B 同量经验差，SOURCE 主比较仍使用完整 REAL。32 次底噪不是独立采集置信区间，也不等于原训练 B=32 或每次出现重抽噪声的期望。

电荷界使用 `a=(-3×100, +1×300, 0×80)`、`c=abs(a)*scale`。把真实表示 `zbar` 重构到电压后计算名义误差

`Erep = mean_SOURCE sum abs(a)*abs(Y-mu-scale*zbarY) + mean_REAL sum abs(a)*abs(R-mu-scale*zbarR)`。

这使用实际表示偏差，不以理想单次 float32 舍入上界替代。所有运算仍是名义 float64；重构、均值、W1 与下界计算误差没有区间包络。令 `Dout=abs(mean q(Y)-mean q(R))`，得到 `LBmean=[Dout-Erep]+/(480*max(c))`、`LBRMS=[Dout-Erep]+/(sqrt(480)*norm(c))`。原源电荷形式另扣实际 `mean abs(q(Y)-q_source)`，并把 preADC 误差与 ADC 变化分别列出。区域合计只使用既有系数 `m_source*(LBmean/8 + LBRMS/4)`；不减损失、不改权重。

worker 应保存完整输出后核对 `summary['charge_bound_aggregation']['all_nominal_inequality_checks_pass']`；若为 false，结束该实现核验并保留失败回执，不随结果放宽容差。下界只描述本冻结端点的 V480 两个坐标贡献，不能被解释为整个目标、A–F 的不可行性或某项损失的因果/梯度主导性。

`artificial_check.py` 以 Windows 单核亲和、1 GiB 进程提交限制和 RSS 看门狗运行，使用 Python `-B`，缓存仅在本目录 `_tmp`。人工检查含 R62 不等 N W1 / 四视图原公式、R63 temporal31 / ADC 原公式的 CPU 值对照、重复父、非零电荷差下界、单位与误差扣除、源质量聚合及错误 raw-bin/电荷/ADC 拒绝。实际回执见 `ARTIFICIAL_CHECK.json` 与 `SOURCE_BINDING.json`；没有科学 GPU / 模型 / 梯度执行。

协议采用项与仍需 root 在科学解码前冻结的字段见 `PROTOCOL_DRAFT.json`。禁止把人工人数覆盖参数用于实际 R64 科学人口。
