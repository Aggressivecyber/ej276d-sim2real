# R72：冻结实测电荷前置的完整波形全人口训练

本任务承接 active v5 目标以及用户“设置目标推进 如果全量在数个小时能跑完 就直接全量”的授权。R71 的完整 TRAIN 电荷组件和 Git/D 盘交付已核验，完整响应 A–F 尚未完成。本文是新候选的前瞻方法；实际注册时刻及源码、配置、输入 SHA 以 `reviews/R72_FULL_WAVEFORM_FULL_TRAIN_001/REGISTRATION.json` 为准。

## 唯一候选与原始人口

固定全部 6593 SOURCE、21047 REAL 合法 TRAIN 父事件，每次更新均使用原每父权重 1/N。保留父 ID、原始行序、h、原 Q、原分区、原支持外人口。TRAIN 文件、单位 Q 统计、色噪声及谷定义使用 `INPUT_BINDING.json` 原哈希。只读取这些原 TRAIN 输入以及 R71 唯一 update50000 的 Q 头；不调用会读取 DEV 的旧通用 loader。当前训练进程不读取 DEV/EVAL/CAL 波形。REAL 原始电荷和 canonical 波形始终为未改写参照。

SOURCE 和 REAL 共用 R71 原始完整480输入 Q 头及 lifting，冻结其参数。波形网络为 `482→128→128→480` 两层 SiLU 的直接残差，末层零初始化，网络参数/矩阵运算 float32；输入标准化、Q、lifting、色噪声和 Box 使用 float64。482 个公开输入为 lifted 波形去100点基线均值后除 Q* 的480坐标，加 logQ0、logQ*。标准化仅用合法原 REAL TRAIN 经公共前置后的均值/总体标准差，逐坐标 floor=1e-6。不使用域、ID、h或类别作为模型输入。128 是读取本轮科学数组前确定的唯一宽度；32 仅是未运行草案，因全100点基线变换的容量考虑在注册前废弃，没有架构对比。

位移 `d0=Q*·scale_condition·f` 还原为电压，再 `d=d0−mu_condition·Q(d0)` 作零Q投影。原 R71 completion 噪声创新仍依据 lifted 基线方差与 Q* 连续条件，公开输入 epsilon。`raw=lifted+noise+d`，按原64次 Box 约束到 Q*，转 float32 后走原两次数值 ADC。训练 ADC 使用既定 straight-through 反向。四阶段 Q/bins、源支持、位移/投影及独立 Q* 均保存。推理遇到新输入或源扰动时重算完整公共前置，不能使用父ID缓存。训练缓存只存同一已冻结前置对原合法输入的精确结果。

## 完整人口九项目标

所有权重为1，不按质量调权。统计仅从原 canonical REAL TRAIN 拟合，输出每次按自身 canonical Q 与原 frozen logQ edges 重建 raw bins `[-1,0,…,7,8]`。原始 bins 仅用于来源、谷和身份尺度，不充当输出条件。hard bin 和 ADC STE 的导数约定在此冻结：这是完整有限人口分布目标的约定梯度，不宣称离散 ADC/hard bin 是光滑函数。

五项分布损失分别对 SOURCE 和同处理 REAL 相对原 REAL 计算，然后两域平均。各维不同样本数的 W1 使用精确有限等权 quantile 积分，不插值近似分位网格。

1. `full_window`：保留 V480/U480/DU479/D2U478 的标准化坐标 marginal 与64个单位随机方向 sliced W1 的各半平均，再四种特征平均。U仍为去基线后除峰值 floor=.0025。各输出区按该域实际 count/N 加权，支持外区独立保留。
2. `coordinate_RMS`：同四种特征的逐坐标 W1 经向量L2/平方根维数，再按同一实际区质量与四特征平均。
3. `PSD`：`.5*(全局W1+输出raw bins0..6的等权平均W1)/.01`。全局量使用所有父事件；这项条件等权是原指标定义，不改变全局父权重。
4. `temporal_joint`：100点基线的 RMS、lag1..20 ACF、10段 rFFT 功率份额，共31维标准化 marginal 与64方向 sliced W1，各半平均并按当前区质量汇总。power floor1e-20，尺度 floor 保留旧配置。
5. `noise`：全局 baseline RMS/ACF1 精确 W1，经原 REAL TRAIN 总体标准差及固定 floor 标准化后平均。
6. `source_order`：仅 SOURCE；原 float32 存储区间精度可分辨的原 signed-PSD 顺序 hinge，分母仍是所有原 PSD 不同的有序父对权重总和，不能改成仅 resolved 对。等权精确扫线/Fenwick 的数值、严格端点和反向须与原全矩阵人工参照一致；无抽对、无省略父事件。可用现有 numba 编译相同整数算法，禁 fastmath；编译时间计入 setup。
7. `source_valley`：原名义 bin2/3 中精确 h==1 全母群的 smooth 谷概率，sigmoid smoothing=.002；超过其原 raw fraction+.02 的正超量除 .02 后平方，两区平均。h 保留原浮点值，不转整型。
8. `REAL_identity`：同处理 REAL 与该父原 canonical 波形配对，按该父原 REAL canonical bin 的 V尺度标准化后全坐标/全父 MSE。
9. `box_correction`：SOURCE、REAL 各自 `mean((preADC−raw)^2)` 的两域平均除 .5²。

若非零质量输出区缺 REAL 参照则明确失败；候选零质量的非必需区贡献严格零，不虚构样本。PSD 所需 bins0..6 任一候选或参照为空则失败。原两谷 h==1 母群为空也失败。非有限、非正 Q*、物理不可实现或超数值门直接失败，不删行、不插补为通过。

每步先保存全27640父同一 epsilon 下的 canonical/preADC/raw 端点，对九项分别重建图求六组独立端点叶梯度，float64累加。随后以原行序和同 epsilon 重放全部神经参数 VJP；轻量MLP的实际训练块固定27640，即完整N一次前向/反向，以减少重复64次Box的开销。通用累计代码在人工50行/13行块上验证尾块，并逐块核对端点逐位相等；参数梯度 float64 累加后一次转参数dtype、clip5、Adam更新。完整人工 direct graph 对照须证明值、梯度及一次 Adam 更新一致。实际显存仍受10GiB监督，无OOM后自动删父。不会采用旧 K4 抽取参数反向。

## 随机流、终点与耗时决策

配置绑定独立 SOURCE/REAL CUDA epsilon generator 和 PCG64 四特征/temporal 随机方向流。每步每父得到一份新的480维 float32标准正态 epsilon；两遍只在本步复用。每个 sliced 特征使用64个按列归一化的 Gaussian 方向。模型种子、各流及完整参数见 `config.json`。

固定唯一 shape update50000 末点，Adam(.9,.999)、clip5，500步从1e-6 warmup至2e-4，随后 cosine至2e-5。每5000步的恢复档只用于同一逻辑运行的故障恢复，不能按科学分数选取。

科学数组读取前完成 CPU/CUDA 人工模块、W1/顺序和完整端点/全VJP/Adam 检查。随后先执行10次暖机及40次计时的完整人口更新。工程50步只测运行成本，完整科学分数不用于选择；之后丢弃全部工程参数/Adam，恢复逐张量原始状态哈希、全部Torch/CUDA/PCG随机流以及空Adam。首编译、前置拟合和尾检预留均计入估计。

保守预计 `setup_elapsed + 1.25*p90完整步时*50000 + 1800秒尾检预留`。若≤14400秒，同一受控进程直接执行全部50000更新。若超过则保存 `COST_TOO_HIGH_FULL_MAIN_NOT_STARTED`，主训练0步；只证明当前实现不符“数小时直接全量”的成本条件，不能当作科学失败、缩人口或改终点继续跑。后续仅在新前瞻绑定下据成本推进，不恢复旧R65。

## 数值与后续评价

唯一末点做所有原 TRAIN 的 Q-only 零形状控制及 learned 输出，保存未删减的四阶段迁移、Q/PSD/噪声/九边界描述、源Q/PSD rank和原谷比例。以两域各最前64父和固定独立PCG epsilon作预登记独立 NumPy float64 完整Q头/128宽MLP/色噪声/Box/ADC参考：preADC RMS中位≤.25 LSB、P95≤1 LSB，Q*独立差与逐父实现差≤1e-5。报告真正整数ADC码差，不把float64电压array_equal命名为码相等。不虚构128/256积分器，本方法为非迭代直接残差。

重新加载唯一权重并重新生成相同PCG流，对前2000 SOURCE+2000 REAL TRAIN 检查 Q*/preADC/canonical逐位重放。该TRAIN4000只检验本轮组装与序列化；原最终独立held-out4000、完整固定A–C及通过后D/E仍须实际执行，不能用TRAIN改善替代。末点固定后将单独绑定当前v5的完整held-out评价适配，不用其结果回选本轮架构、权重或检查点。GAP必须独立重训两阶段Q与形状，不能借FULL权重或被隐藏REAL的统计；D须计入R71/R72真实开发暴露。

## 资源和交付

单科学作业，服务器 CPU0..11/RSS48GiB/GPU实际进程与allocator上限10GiB。启动前检查真实PID、GPU占用、可用内存和磁盘；原生PID握手在科学import之前，终止回执含真实退出码。本机全部辅助共享CPU4/RSS4GiB，各进程CPU1/RSS1GiB。所有新本地数据/源码/日志/缓存仅E/D，C仅入口及现有解释器。原始输入、旧v4/v5验收哈希和历史负结果不改。

交付需根复核原生结束、输入/输出/权重SHA、人口与损失计数，实际打开图验证，提供完整公开推理、复现/失败账本及私有Git远端SHA/D备份回执。仅局部电荷、成本测量或本任务TRAIN阶段完成不能将长期目标标记完成。

## 注册后的工程修复02

首个受控进程在CUDA人工W1检查中将torch.device("cuda")与实际Tensor设备cuda:0直接比较，触发类型检查错误；原生exit1，尚未读取科学数组、工程更新0、主训练0。完整初次源码、输入绑定、原生输出和哈希在reviews/R72_FULL_WAVEFORM_FULL_TRAIN_001/attempt_01_predata_device_alias/保存。修复仅将未指定序号的CUDA别名解析为当前设备；科学公式、架构、人口、随机种子、终点和cost规则不变。修复后新远端目录重新执行全部CPU/CUDA人工检查，通过后才读TRAIN。
