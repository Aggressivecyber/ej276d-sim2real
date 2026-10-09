# R33：完整波形局部改善，源结构与条件分布未共同通过

固定OT-CFM配方关闭。Independent与OT均降低了完整波形可分性，但仍高度区别于实测；OT未表现出相对独立耦合的可信完整响应增益，且条件PSD、低侧峰宽、基线噪声和纯had源条件结构未共同通过。

|同2950更新固定末点|BASE|Independent|OT|
|---|---:|---:|---:|
|完整shape493 BA|.978|.948|.952|
|纯u480 BA|.978|.952|.946|
|完整电压480 BA|.982|.954|.952|
|全人口PSD W1|.02306546|.02764015|.02174726|
|共同观察Q区1–7等权PSD W1|.03476580|.01470086|.04505234|
|低侧raw-width MAE，共同区2/3/6|.01053930|.01121538|.01393592|
|高侧raw-width MAE，共同区2/3/6|.01530337|.00315311|.01620770|
|baseline RMS W1 (V)|.00001936|.00002347|.00002269|
|ACF1 W1|.06423497|.08550216|.08501573|
|实际source-PSD到output-PSD Spearman|.418992|.138305|.408419|

OT减BASE的shape493/纯u480配对ΔBA为−.026/−.032，95%区间分别[−.050328,−.005874]/[−.055763,−.011859]。OT减Independent为+.004/−.006，两区间均跨0，完整shape点值也未更好。Independent相对BASE的两完整视图区间同样低于0。这些500次区组区间条件于固定分类器和历史组，不含训练种子或独立采集不确定性。

OT总体PSD W1下降5.72%，共同观察Q区PSD W1反而增加29.59%；低侧raw-width误差增加32.23%，噪声RMS/ACF1距离增加17.23%/32.35%。Independent的条件PSD/高侧宽度有收益，但全人口PSD距离更差，源PSD秩关联由.419降至.138。密度支持的共同区只有2/3/6；未收敛、成分未分离和支持不足完整保留，没有补拟合。两密度成分不是粒子真值。

按实际未量化NN输入的source-Q bins3/4，pure-had分别213/181父：BASE谷比例5.634%/.5525%，Independent18.310%/22.099%，OT13.615%/3.867%。OT增加7.981/3.315个百分点，超过本轮2个百分点限制。历史postADC源参考207/182另列，不用于主delta。谷内比例是统计诊断，不能直接称粒子分类错误。全部2000父记录、417 pureEE/1518 pure-had/65 mixed、顺序及权重保留，QC全通过、无ADC裁剪。

固定链为完整480点源→冻结R32 CONTROL_FINAL G→完整480维流→一次ADC，两臂只改变Q内临时统计耦合。每臂2950×32行、相同输入/时刻流和初始state，G/D/reader均零更新；仅TRAIN共用仿射常数，固定末点无DEV选模。主2000父中1583在流支持内，417支持外保留原G并明确标记。最终物理signal/electronic分解unknown，base分量和流增量仅作代数分解。

两臂32vs64步Heun的预定32父检查均通过，一个ADC LSB以内；发布仍32步。独立CPU从实际源和随机输入重放4000行，pre/postADC、原分量、流增量、mask/Qbin/ID逐值通过。实际20新HGB、16密度槽/14EM、主4000、joint2048、源探针1664，没有新增REAL/CAL/EVAL/G4。

科学累计101.219102/2400秒，真实worker峰RSS 2,027,515,904字节、GPU allocated 148,003,840字节。保留首次辅助模块缺失导致的6.356629秒导入失败；修复前没有拟合/生成。服务器1次RSS查询缺测有UTC，原因unknown。执行器实际查看11张图，根查看6张关键图；实际封存native0，262文件/212,433,266字节。正常传输成功，无迁移内容一致性测试。

R34已登记并交既有执行器：不重训，给两份冻结流增量施加同一个秩378欧氏投影，保留BASE前100点及100:150、150:400两段积分；检查完整形状收益、量化后源PSD、密度支持和固定边界一二阶差。该规则只保护BASE现有响应代理，不能修复其既有物理或分布偏差。科学600秒、本机CPU4/RSS4GiB，若失败即关闭该固定投影，不扫描窗口或平滑。项目仍active，完整响应尚未验收。

[完整报告](work/R33_RESIDUAL_OT_FLOW_001/REPORT.md) · [完整分类图](work/R33_RESIDUAL_OT_FLOW_001/results/figures/c2st.png) · [Q–PSD图](work/R33_RESIDUAL_OT_FLOW_001/results/figures/charge_psd.png) · [源保真图](work/R33_RESIDUAL_OT_FLOW_001/results/figures/source_preservation.png) · [峰宽图](work/R33_RESIDUAL_OT_FLOW_001/results/figures/energy_peak_width.png) · [复现入口](work/R33_RESIDUAL_OT_FLOW_001/REPRODUCTION.md) · [下一任务](tasks/R34_PROTECTED_FLOW_CORRECTION_001.md)。

实际私有Git与正常D副本完成情况分别见R33_PUBLICATION_RECEIPT.json和R33_BACKUP_RECEIPT.json。
