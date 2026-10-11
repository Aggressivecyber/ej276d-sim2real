# R71 公共条件 Q-head 与电荷坐标 lifting 候选

实现只涉及已由根固定的小头和坐标接口。SOURCE/REAL 共同使用 480 维单位电荷全波形，MLP 为 float64 的 `480→16→SiLU→2`，最后一层权重和偏置为零。无域标记、父身份输入、REAL 旁路，也无数据读取、训练循环、科学损失或验收阈值。原 exp 增益式仅属 R70 设计记录，不是此处的第二候选。

## 固定公式与接口

采用原窗积分 `Q(x)=sum(x[100:400])-3 sum(x[:100])`。输入先取 `β=mean(x[:100])`、`u=(x-β)/Q0`，再以根提供的合法 TRAIN 冻结 480 维中心和正尺度标准化。完整输入接口为：

```python
head = PublicConditionalChargeHead(center480, scale480).double().to(device)
record = head(x480)  # SOURCE 与 REAL 同一调用
prepared = head.standardize_input(x480)  # 固定父输入可缓存一次
record2 = head.forward_features(prepared['features'], prepared['q_input'],
                                check_inputs=False)
```

`check_inputs=False` 仅用于此同一冻结标准化器已经核对且未改变的输入；输出有界性、非有限和斜率失败检查始终执行。MLP/缓冲区默认均为 float64，`.double()` 完整覆盖二者；无内部 float32 强制。模型创建使用调用者 Torch RNG，导入、前向和坐标函数不抽随机数。

令 `L=300`、`z=a(u)+exp(b(u)) [log(Q0)-log(L-Q0)]`，固定为 `Q*=L sigmoid(z)`。所有输入父必须满足 `0<Q0<L`；非有限、越界、预测数值饱和到任一轨或斜率溢出/下溢均整体失败，没有裁行、补 epsilon 或事后 Q clamp。零末层在实数运算下为恒等，float64 logistic 往返仍存在可测舍入误差，不宣称逐位相同。

记录独立预测 `Q*`、原 `Q0`、`a,b`、`logit_slope=exp(b)` 以及固定 u 的物理坐标斜率

`dQ*/dQ0|u = exp(b) Q*(L-Q*) / [Q0(L-Q0)]`。

另外 `dQ*/da=Q*(1-Q*/L)`，`dQ*/db=dQ*/da·exp(b)·logit(Q0/L)`。完整波形的反向包含 `Q0`、基线、单位波形和 MLP 的全部路径；固定 u 斜率为正不等于任意改变波形时总体 Q 顺序必然保持。

## 连续模板、迁移和共同末点

`interpolate_charge_statistics(Q*,mu8x480,scale8x480,edges9)` 使用原八个 logQ 区间中心、`log(1000Q*)` 上的连续线性插值，两端恒定延伸。冻结 μ 先去前 100 点均值并归一到单位 Q；插值后再次作相同坐标归一，仅去除代数上应为零的浮点偏差。尺度同时作正值线性插值。结点处函数值连续，导数为分段导数，恰在结点选择右侧区间。

`charge_coordinate_lift(x,Q*,m)=x+(Q*-Q0)m(Q*)` 保留全梯度，包括连续 μ 对 Q* 的依赖；理论上 `Q(lift)=Q*`、`dQ(lift)/dQ*=1`。`zero_charge_project(v,m)=v-Q(v)m` 用于后续零电荷噪声或残差。`common_charge_coordinates` 组合上述接口，返回 lift、连续 μ/scale、连续 Q 条件和输入/目标 raw/clipped 区号。

该阶段大波形残差固定为零。根的末点公共路径应为 `lift→共同噪声→原 Box(raw,Q*)→原 canonical ADC`。lift 本身允许越过幅度轨，最终由原 Box 约束 `[-0.5,0.5]`；不得将 Q* 重定义为末点积分，也不得又用原 Q0 作 Box 目标。末点分别记录原 Q0、预测 Q*、pre-ADC 实际积分、ADC 实际积分及四阶段区号；保留全部父 ID、顺序、事件权重和区外质量。区号裁到 0..7 仅为统计查表，raw 区外标记保存，Q 和连续条件均不裁剪。原九边界及最终边界属于区 7 的约定保持。

模板连续不自动使未来非零波形网络的硬分区 one-hot 连续。本轮波形残差为零；后续架构必须单独登记。ADC 是原离散量化，精确求导不存在；沿用 STE 时须明确其为优化近似，CPU 人工核对不能把它称为原 A–F 数值或科学验收通过。

## 噪声、合法 TRAIN 和证据边界

本候选不把 NumPy 旧 `innovation` 放进 Q-head 训练计算图，根已固定噪声不参加小头优化。末点全 480 点着色噪声需按连续 μ 作同一零 Q 投影；旧分区 `kappa2_b` 不能直接插值。根将以当前投影 `P=I-m aᵀ` 的 `P K Pᵀ` 前触发方差解析式求新归一，另做人工稠密协方差核对。这属于根的噪声绑定，不在本模块隐式决定。

SOURCE 与 REAL 的标准化、Q-head、lifting、噪声、Box 和 ADC 都须同一可调用路径。保持实测电荷分布仍需端点实测，不能由零初始化推定成功，更不是物理能量标定。共同全量 Q 损失、REAL 身份项、SOURCE 顺序项和 50,000 更新由根登记，此处不选择权重、学习率或科学阈值。GAP 方案重新拟合适用的合法 TRAIN 特征中心/尺度、μ、scale、噪声；不能复用含 held-out REAL 的 FULL 统计。未观测人口掩码表示不可识别，不能伪作零概率，不能宣称缺口概率质量已知。

## 实现检查和性能范围

`check_public_charge_head.run_checks()` 无文件 I/O，并恢复调用者 CPU RNG；仅人工数组检查 MLP 恒等舍入、公共/缓存入口一致、父独立、全部 480 输入连接、显式与完整输入梯度、模板连续、charge lifting、原区号约定和失败分支。`run_box_checks(Box,adc)` 另以调用者提供的确切封存 Box/ADC 定义核对人工接口；本地 launcher 只 AST 提取四个定义，不导入旧网络或读权重。

具体结果见 `ARTIFICIAL_CHECK.json`、`NATIVE_RESOURCE_RECEIPT.json` 和独立 `NATIVE_EXIT_CODE.txt`。证据仅为 native CPU 人工实现验证；未运行 CUDA、科学数组、完整波形网络、拟合或新的探针。源码与读入静态文件 SHA 见 `SHA256_MANIFEST.json`。

小头参数数目为 7,730，固定人口标准化可缓存；全量 float64 每次 27,640 父仍包含矩阵乘法、反向、完整 Q 损失的排序与优化。输出 finite/轨/斜率检查中的 Python bool 在 CUDA 会同步，需纳入根的完整工程测时。这里不能据参数少或 CPU 人工时间推断 50,000 更新一定数小时；根计划独立 10 warmup+40 timed 工程更新、丢弃工程状态、恢复同初态后再运行正式端点。
