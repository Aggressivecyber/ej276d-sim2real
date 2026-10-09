# R30：域别矩处理使实测对照也变得可分

本轮完成15次固定分类器拟合、0响应训练/生成。四个生成端点在全局中心化、观察Q区中心化及中心化加逐点尺度后仍高度可分；但REAL TRAIN–DEV原始BA .510也升至.954/.772/.814。因此本轮不能用“去均值/尺度后仍高BA”判定唯一物理瓶颈。R28仍为当前响应开发比较基线，完整响应验收未通过。

|保存端点/参照|原始u480|全局中心|Q区中心|Q区中心及尺度|
|---|---:|---:|---:|---:|
|FRESH_SELECTED|0.988|0.996|0.992|0.982|
|FRESH_FINAL|0.978|1.000|0.994|0.976|
|ENERGY_SELECTED|0.984|0.988|0.996|0.986|
|ENERGY_FINAL|0.992|0.994|0.990|0.986|
|TRAIN_REFERENCE|0.510|0.954|0.772|0.814|

这里是纯480维峰值归一化u；历史附加13标量的493维完整shape是另一视图，R28原.970不被本轮.978替换。每域固定600分类器训练、250留出，组交集0，旧RAW预测直接复用。500次区组区间条件于固定分类器，不含响应训练、矩估计重拟合或新采集不确定性。

R28末点Q中心加尺度相对RAW的ΔBA=-.002，95%区间[-.020921,.014515]；REAL–REAL同一变化为+.304，[.241435,.361562]。全局中心的REAL–REAL ΔBA=+.444，[.395624,.495182]。人工分域处理会改变可分性，均值估计、峰归一化几何及其他结构的贡献不能仅凭这些分类结果分离。

六域共同训练支持为bins1–7，bin8及区外/未知行都退回各自全局训练矩，不删除；共同留出描述区为bins1–6。所有域的全部2000行都通过原QC；两个历史阶段四端点共享源父顺序与417 pureEE/1518 pure-had/65 mixed组成。分类器训练均值和未触floor的标准差运算检查达到浮点精度，这只确认实现，不代表物理验收。

原始留出描述仍有实际差异：R28末点prompt均值残差RMS为.0839558，Q中心后.0233919；REAL–REAL对应为.00656324和.00911956。原始R28末点协方差相对REAL的Frobenius差为.394295，REAL–REAL为.179261。Q中心加尺度后两者为1.37087和.783109，R28与REAL有效秩为20.0491和37.7427。以上基于250条留出的高维样本描述，不是显著性检验、噪声真值或物理等效阈值；population曲线与holdout统计在阶段报告中分开列示。

两个科学进程均原生退出0，导入前实际worker PID/RSS握手成立，内存查询失败0；科学累计35.7090681秒，峰RSS419,291,136字节，CPU4。根实际查看全部8张最终图，布局可读，无需修图。没有新REAL、EVAL、CAL波形、G4、EM、SSH或安装；R28/R29模型与主输出保持原件。

下一轮R31已登记：用已知max(u)=1约束的固定逆均值解码器及唯一已有prompt100:150排除掩码，检验峰归一化是否参与上述人工可分性。12HGB、科学300秒、零响应生成；完整480指标继续保留。若去prompt后控制仍不合格，则关闭域别矩+C2ST的因果归因路线，不扩展掩码扫描。

[阶段报告](work/R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001/REPORT.md) · [分类结果图](work/R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001/figures/c2st_moment_views.png) · [留出均值](work/R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001/figures/holdout_mean_residuals.png) · [留出波动](work/R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001/figures/holdout_pointwise_SD.png) · [复现入口](work/R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001/REPRODUCTION.md) · [下一任务](tasks/R31_NORMALIZATION_ANCHOR_DIAGNOSTIC_001.md)。
