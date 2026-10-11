# R73 精确有限均匀 W1 解析反向候选

单一候选为 `exact_uniform_w1.UniformW1Cache`，可替换原类导入。构造参数 `(device,max_entries=32,max_cells=262144)`、`columns(a,b,*,b_sorted=False,check=True)`、`_plan(n,m)`、`receipt()` 均保留。候选不复制整体 loss，不改九项损失、分区/人口/尺度/分母/权重/随机数/模型，也不读科学数组。依赖仅既有 PyTorch 与标准库；人工检查另用既有 NumPy。

## 值与梯度契约

前向保留原稳定排序和原 LCM 单元计划，按相同 `ia/ib/mass`、相同 float64 表达式和原单元顺序计算：

`((sa[ia]-sb[ib]).abs()*mass[:,None]).sum(0)`。

没有网格、分位数近似、重采样、舍弃条件项或人口。`b_sorted=True` 与旧类一样信任调用者每列已排序；对可微的已排序 b 返回其输入顺序的梯度，调用者外部 sort 的反向仍由 PyTorch处理。

令排序后第 i 个 a 的区间为 A_i=[i/n,(i+1)/n]；L=#(b<a_i)、R=#(b<=a_i)，均由 float64 strict searchsorted 得到。原 abs 在相等处导数为0；区间 [0,L/m) 对 a_i 为正，[R/m,1] 为负，中间相等部分为0。因此：

`g_i=max(min(L/m,(i+1)/n)-i/n,0)-max((i+1)/n-max(R/m,i/n),0)`。

交换 a/b 得 b 梯度。重复值的每个排序位置分别使用自己的量化区间，保留稳定排序的逐 occurrence tie 梯度，不对重复值平均。最后用该列稳定排序的逆映射 scatter 回输入；每列索引为排列，无重复写入、无 scatter_add/浮点原子归约。各特征列再乘调用者的对应上游梯度。

实现用原 D=lcm(n,m)、s_a=D/n、s_b=D/m 在 int64 上算正负区间长度，最后一次转float64除D，避免接近 CDF 浮点数相减。全部项目人口的整数运算安全。理论上若D超过int64范围，则仍按同一归一化CDF式用float64计算；此无法在本资源下容纳的超大人口分支未人工验证，不影响本项目计数。

数学梯度等于原单元的有符号质量和。原程序的 cell mass 已各自float64舍入且反向逐 cell 乘上游/归约，解析式用一个有符号整数长度再除D，所以有限精度梯度不承诺按字节相同，Adam轨迹也不承诺逐位相同。这是计算实现迁移，不能描述为保持原R72浮点轨迹。值保持原表达式；tie 判定仍是精确值比较，没有新增tie容差。

## 保存、开销与边界

仅在 backward 实际需要时做searchsorted和解析区间运算；避免 no_grad 前向额外搜索。上下文保留两个已排序值及需要反向侧的排序索引，不保留 expanded-cell 的 autograd 图；这里只验证一阶训练，未验证高阶导数。

原反向包括 expanded-cell abs/mass 梯度与重复索引的 gather反向归约；候选改为每个所需侧两次搜索、整数区间计算和唯一scatter。CUDA每列二分搜索的随机访存、转置contiguous拷贝、额外kernel均有成本，保存的排序值也不能推断总显存下降。特别是marginal数百维和样本数变化时，可能没有收益；本机没有运行CUDA微基准或完整科学步，不能用CPU人工成本宣称真实加速。根需用全部九项/全人口/完整VJP/同步的工程测时判断，并以预登记完整端点成本规则决定后续执行。

## 已执行人工证据

最终 `ATTEMPT_002_ARTIFICIAL_CHECK.json` PASS，实际shell退出码0；1601组前向全部按字节相等，3138侧梯度比较通过。1521组 n,m=1..3、值取{-2,0,1}的穷举，使用独立Fraction精确单元积分证明CDF式，两侧解析梯度等于Fraction结果经一次float64转换。

另覆盖不等计数、多个特征/非连续布局、重复/相等/正负零、大平移和大小正尺度、上游含正/负/零/1e4/1e-8、两侧requires_grad所有组合、b_sorted真假、外部sort非叶链、共享a=b叶、check=False实际路径、计划/LRU/失败类型。人工6593/21047计数只用新生成数组，未读真实端点。

最大梯度绝对差1.3642420526593924e-12（测试含1e4上游）；最大差为本人工舍入界的0.013381322224934896倍。每侧采用 `16*eps64*(ceil(other_count/count)+3)*abs(upstream)/count`，用于验证舍入量级，绝非新的科学验收阈值。未把反向按字节一致作为声明。

人工入口 `check_exact_uniform_w1.run_checks(device='cpu',original_source=...)` 返回dict。只从指定封存源码AST提取原 `_tensor` 和 `UniformW1Cache`，不导入完整loss或其Numba路径；导入人工模块本身无文件读取/随机抽样。CUDA调用由根登记后自行执行，需显式传服务器原源码路径。

本机使用既有local_resource_guard，affinity mask2、1核、Torch及interop各1线程、硬RSS1GiB，全部缓存/临时目录E。最终native PID/实际RSS/墙钟在 `ATTEMPT_002_NATIVE_RESOURCE_RECEIPT.json`；首轮也PASS，因补check=False覆盖及去除纯检查scalar转换警告后重跑，首轮回执及当时checker源完整保留，不覆盖任何失败。没有CUDA/科学训练/权重读取/SSH/Git，未改原R72或根治理文件。

所有输入源SHA、候选SHA和实际native状态见同目录封存清单与完成回执；根拥有集成、完整objective/Adam人工核验、正式工程测时和科学决策。
