# R31：峰归一化泄露域身份，关闭矩处理分类的归因路线

在全局中心化后的完整480点上，仅比较两种逆均值恢复是否满足max(u)=1，不拟合模型，就能在R28、R29及REAL TRAIN–DEV三个比较中全部达到BA 1.000、覆盖100%、零弃权。真实逆变换误差均为0。这确认该域别处理包含可供解码的人工域身份，不能把处理后高BA唯一归因于物理噪声或时序结构；没有证明HGB实际只用了这一条规则。

固定删除已有prompt100:150后，保留430点、不重归一化、不重估矩，共12次HGB。结果为：

|固定端点/参照|原始430|全局中心430|Q区中心430|Q区中心及尺度430|
|---|---:|---:|---:|---:|
|FRESH_FINAL|0.988|0.992|0.986|0.978|
|ENERGY_FINAL|0.984|0.992|0.988|0.988|
|TRAIN_REFERENCE|0.536|0.662|0.710|0.690|

REAL–REAL原始430的BA .536，95%区组区间[.494525,.584701]包含.5；三种矩处理相对原始430的配对BA差区间分别为[.058164,.177216]、[.123829,.219721]、[.093347,.210985]，均不能完整落入预定[-.10,.10]容差。因此三种处理的控制可用性全部失败，本轮关闭域别矩+C2ST的因果归因路线，不扩展窗口、floor、样本量或种子扫描。容差为操作规则，不是物理等效界限。

R28在原始430仍为.988，表明完整不匹配并非仅靠删除该prompt窗口便消失；430的结果不能当完整波形验收。原完整480、R30全部矩参数和既有评价仍保留。四域各2000条原波形全部通过原QC、峰值floor均未激活、所有argmax位于固定prompt区。原600训练/250留出及组不变，所有行保留，共同条件报告bins1–6；高区低支持不强行给结论。500区组区间条件于固定分类器和现有行组，不含响应训练、矩估计重拟合或采集不确定性。

统计、绘图和一次纯布局修复共26.1183026秒，三个实际worker均native0，导入前PID/RSS握手成立、内存查询失败0，峰RSS 366,927,872字节。根实际查看全部6张最终图，修正后的配对差图标签可读，原图和修复回执保留。响应训练、生成、新REAL、EVAL、CAL波形、G4、EM、SSH和安装均0。

下一轮R32已登记：依据原始条件均值差，在R28方案上只增加固定系数1的TRAIN条件均值U项，保持完整480输入和原分布/内容损失。在恢复的服务器依次训练同平台原方案对照与干预各2950步，原J20不变，完整波形、PSD/宽度、噪声和源条件共同判定。该干预不使用R30分域变换，也不把每个事件推向同一模板；阶段预算和明确停止规则见任务。

[完整报告](work/R31_NORMALIZATION_ANCHOR_DIAGNOSTIC_001/REPORT.md) · [无拟合解码图](work/R31_NORMALIZATION_ANCHOR_DIAGNOSTIC_001/figures/inverse_mean_decoder.png) · [完整与430对比](work/R31_NORMALIZATION_ANCHOR_DIAGNOSTIC_001/figures/complete_and_masked_C2ST.png) · [控制容差](work/R31_NORMALIZATION_ANCHOR_DIAGNOSTIC_001/figures/control_tolerance.png) · [复现入口](work/R31_NORMALIZATION_ANCHOR_DIAGNOSTIC_001/REPRODUCTION.md) · [R32任务](tasks/R32_CONDITIONAL_MEAN_TRANSLATION_001.md)。

项目active，当前模型完整响应仍未通过。R31是诊断路线的负结果，不是新的生成模型。Git及D副本的实际完成状态分别见R31_PUBLICATION_RECEIPT.json与R31_BACKUP_RECEIPT.json。
