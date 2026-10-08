**R1：已有 G4 响应变换的共同相似性基线**

历史权重为0的输出 GEN_W0 改善了完整 PSD 分布的位置，但仍没有复现实测的整体响应。它的 PSD Wasserstein 距离由 SIM 的0.104627降至0.041618；固定独立 C2ST 在形状视图上的留出平衡准确率仍为99.6%。主要剩余差异是主峰/下降沿的固定时间结构、条件分布过窄及基线相关性。GEN_FIXED 几乎保持原SIM的PSD位置，这种旧工程行为不作为本项目的成功条件。

本轮使用三个版本相同的2000个G4父事件：SIM、旧权重10候选经有界/保基线包装的GEN_FIXED，以及旧权重0原始输出GEN_W0；实测为P1K已保存的完整2000 DEV。全部2000行均通过本轮共同QC，保留unknown/重叠区。生成版本与SIM父ID/顺序完全一致，历史REAL归一化行与P1K的2000行逐位一致。GEN_FIXED和GEN_W0还存在包装版本差别，因此本轮不把其所有差异唯一归因为保持项；严格历史原始权重对照已有P1I记录。这里是已历史暴露DEV上的诊断，未读取EVAL，也未训练新的响应模型、前向重跑或运行G4。

**共同尺度与总体结果**

时间间隔1 ns，保存480点；每行减前100点均值，Qfast积分100:150，Qtotal积分100:400，PSD=1−Qfast/Qtotal，电荷单位mV·ns。形状视图再除该保存窗自身正峰。SIM/GEN的有单位结果采用保存SIM父峰恢复尺度；GEN的V与电荷是保留父输入尺度的代理，并非独立学得或验证的实测幅度标定。REAL使用保存物理V。未作电荷匹配或重加权，整体谱差异与按各自电荷分区的条件结果同时保留。

|对象|PSD W1到REAL|PSD KS|总体固定切点FoM|低侧数/2000|HC gamma-like/neutron-like|unknown|
|---|---:|---:|---:|---:|---:|---:|
|SIM|0.104627|0.687000|undefined|7|0/845|1155|
|GEN_FIXED|0.103716|0.690000|undefined|6|0/747|1253|
|GEN_W0|0.041618|0.247000|0.689814|475|234/540|1226|
|REAL|—|—|0.702378|959|668/402|930|

GEN_W0总体FoM为0.689814，REAL为0.702378，但低侧分别23.75%与47.95%，所以FoM接近并不表示混合比例或完整分布接近。这里低/高侧是冻结CAL全局切点0.252960的操作性分区，不是独立粒子真值。FoM以切点两侧的均值和样本SD换算Gaussian FWHM；每侧至少20事件，unknown也参加计算。高置信参考来自冻结CAL各分区混合后验≥0.99，未新拟合。

![完整QC电荷—PSD分布](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/figures/charge_psd.png)

**条件PSD与支持**

冻结CAL电荷边界为2098.486、3703.778、6711.805 mV·ns。q4原参考未收敛，所有q4伪标签保持unknown，局部FoM保持undefined；q1实测没有HC neutron支持。无标签分布比较仍全部完成。

|区|SIM / FIXED / W0 / REAL支持数|W0 PSD W1 / KS|W0低侧比例 / REAL低侧比例|W0 FoM / REAL FoM|
|---|---|---:|---:|---:|
|q1|947/1020/1164/467|0.068478 / 0.470|0.323 / 0.713|0.545767 / 0.721288|
|q2|540/523/439/514|0.052238 / 0.430|0.221 / 0.523|2.087333 / 0.875107|
|q3|400/312/271/539|0.032406 / 0.374|0.258 / 0.451|1.983654 / 0.830593|
|q4|113/145/126/480|0.016176 / 0.204|undefined / undefined|undefined / undefined|

例如q2/q3的GEN_W0 FoM为2.087/1.984，REAL仅0.875/0.831；ECDF显示生成带仍非常窄。达到较高FoM不是本轮响应真实性目标。各区完整均值/SD、分位数、切点两侧位置/宽度/比例与支持数均在完整metrics中，q4无有效两峰参考时保留完整PSD统计。

![分电荷完整PSD累计分布](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/figures/conditional_psd.png)

**波形与涨落**

|对象|上升10–90%中位数 [ns]|上升时间IQR [ns]|基线RMS中位数 [mV; GEN代理]|基线ACF1中位数|
|---|---:|---:|---:|---:|
|SIM|4.531|0.184|0.624|-0.023|
|GEN_FIXED|6.435|1.210|0.407|0.201|
|GEN_W0|6.438|1.015|0.407|0.477|
|REAL|5.902|1.549|0.628|-0.031|

GEN_W0上升时间中位数接近REAL，但变化范围仍较小；条件中心波形出现多个固定时刻尖峰/振铃，REAL没有同样的中心纹理。GEN_W0的基线ACF1中位数0.477，REAL为−0.031，且生成基线RMS随电荷的关系明显不同。RMS和噪声谱使用同一保存窗前100点，ACF分母为全100点离差平方和；这些是现有窗口描述，不能单凭它们辨识完整电子学噪声源。均值/中位波形及10–90%带、原尺度/自身峰归一化两种视图均保存，未随机配对计算转换MSE。

![同电荷区原尺度与归一化中心波形及事件分位带](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/figures/conditional_waveforms.png)
![保存基线的RMS、自相关和单边噪声谱](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/figures/fluctuations.png)

**抽样参照与独立C2ST**

64次参照将REAL人工50记录组完整分到两侧后等量选事件；每侧911–1000事件，中位数977。总体REAL—REAL PSD W1中位数0.003832，5–95%分位0.002555–0.007200。在相同样本量比较中，GEN_W0—REAL的PSD W1中位数0.041339，仍远高于同采集抽样差异。64分位为描述参照，不是物理等效界或独立采集置信区间。

C2ST使用固定HistGradientBoosting：100轮、15叶、学习率0.06、最小叶20、L2=1，无搜索或早停。每域600训练、250留出；SIM及两个GEN以同一父ID分区，REAL及扰动视图以original_index//50组分区，所有视图共用训练/留出身份且交叉为0。完整480点归一化波形加13个形状/涨落特征；另一视图额外加入log电荷/峰/RMS。缺失特征仅用训练中位数填补。模型未使用GAN判别器。

|C2ST|形状留出BA [95%组bootstrap区间]|形状AUC|形状+幅度代理BA|
|---|---:|---:|---:|
|SIM_vs_REAL|0.996 [0.989, 1.000]|1.0000|0.998|
|GEN_FIXED_vs_REAL|0.998 [0.994, 1.000]|1.0000|0.998|
|GEN_W0_vs_REAL|0.996 [0.989, 1.000]|1.0000|0.996|
|REAL_vs_REAL|0.490 [0.448, 0.537]|0.4828|0.498|
|PERTURBED_REAL_vs_REAL|0.990 [0.980, 0.998]|0.9974|0.994|

REAL—REAL接近随机且区间包含0.5；预置对照在REAL上加入2%自身峰的80 ns尾项和1.5%自身峰的固定种子Gaussian噪声，BA为0.990，说明此固定评价器能检出这类非零响应变化。区间通过1000次留出父事件/人工组bootstrap计算，条件于本次已训练的评价器，未包含历史响应模型训练不确定性；人工组不是采集run。GEN_W0置换主峰段100:150后的BA下降0.500，而全形状BA为0.996，指向主峰/下降沿纹理的剩余域差异。块置换依赖特征相关性，是定位诊断，不是因果归属或额外训练选择。

![固定C2ST留出指标与区间](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/figures/c2st.png)

**G4关联及下一轮可实施的问题**

2000个SIM父记录含legacy neutron1683、gamma317；已有生产规则的gamma是非2112桶，未把它改称纯PDG22真值。已保存EE/hadron沉积、可见hadron分量、SIM声明能量及所有SIM→GEN ID关联。r_h与PSD的Spearman相关为SIM0.577、GEN_FIXED0.528、GEN_W00.704，支持响应仍与输入组成有关，不能由相关性证明逐事件物理标签正确。本轮复用了已有G4来源/处理链，未把波形相似性当作Geant4几何、源谱和沉积模型独立物理验证。

下一轮的单一问题建议为：保留同一G4 TRAIN父集合时，平滑条件均值响应加显式随机涨落能否同时改善主峰/下降沿、条件PSD宽度与基线谱，并降低这份固定C2ST的可分性？用EE/hadron沉积与可见能量代理组成条件c，以完整480点的平滑时间基或stride1轻量卷积学习均值mθ(t|c,xSIM)，另以独立z和小型条件噪声滤波器学习nφ(t|c,z)。幅度分支显式学习响应尺度，而不继续默认继承SIM父峰；保存父ID及沉积分量。均值的平滑约束抑制固定时刻纹理，随机分支承担RMS、ACF/谱和条件宽度；训练对齐完整QC总体的电荷—PSD及波形/噪声分布，unknown继续参与。旧0.15残差界和SIM FoM保持不作为目标，不用随机无配对MSE拟合变换。该建议尚未执行。

本地存在prepare_sim_data.py与响应配置入口，以及现成DEV2000及对应G4元数据。两份历史训练配置都指向远端sim_empirical_spectrum_v1.pickle；本轮检查的四个local data目录不存在，因此尚未绑定本地完整SIM TRAIN数组，不能把DEV2000改作TRAIN。ROOT已有来源记录的edep_nt_events.csv为4,463,813 B，下一轮可复用已有CSV及历史TRAIN父事件构造缓存，无需新跑G4。当前默认Python由ROOT确认torch2.10.0+cpu、无CUDA，建议使用现有4线程CPU预算训练轻量原型，不安装环境；本轮未读取此CSV、传输或训练。


**执行与交付**

首命令2026-10-08 08:55:53.493977 UTC，PID52564/tool3d208e。唯一数值监督PID36988、workerPID41284；09:09:52.648563至09:10:39.485242，墙钟46.8363 s，worker及launcher原生exit0，工具ac74b7→a68a66，stderr为空。数值累计46.8363 s/1200 s，4线程。首轮RSS由于Windows PMC结构漏字段未取得有效测量，明确记unknown而非0；修复后的监督器供后续使用，未为RSS重跑科学计算。纯OS查询已验证正确72B结构（15.88 MB仅属于该查询进程）。首轮执行代码保留source_run_001，当前launch.py是该单点监督修复版。

七张关键图均实际查看：全数据/坐标/图例清晰，没有裁切。重要原始NPZ版本hash与既存绑定匹配；必要检查只含共同定义、父/组用途分离、计数和图检。运行配置、源码、日志、输入清单、完整统计及C2ST留出预测均保留在本目录。新产物约数MiB，最终确切总量见RESULT_RECEIPT和FINAL_MANIFEST，远低于512MiB。

紧凑指标：[metrics.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/metrics.json)；完整统计：[完整metrics](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/metrics.json)；固定配置：[config.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/config.json)；计算入口：[evaluate.py](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/evaluate.py)；执行原生回执：[NATIVE_RECEIPT](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/NATIVE_RECEIPT.json)。另有[特征分布图](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/figures/response_features.png)和[父响应关联图](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R1_RESPONSE_SIMILARITY_BASELINE_001/run_001/results/figures/parent_response.png)。
