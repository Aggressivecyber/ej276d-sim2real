# R69：原 A 固定电荷–PSD 评价器的静态审阅

**原 A 尚未被证实普遍不可行。R47 已有更贴近原 A 的电荷一维留出证据，结论不确定；本轮新增拟合为0。** 根已决定不执行新的全 TRAIN 一维分类器：它不能单独决定成本hold或新波形候选。原 A–F、源Q/权重与R65退休保持。

下列证据路径相对 E:/GAN/research/sim2real_two_papers_20261008_0305。只读代码、报告和JSON；没有打开NPZ/PT/CSV、重算指标、加载科学模块或执行科学函数。

## 原门实际定义

| 项目 | 实际定义与文件行 |
|---|---|
| A 的通过条件 | SOURCE和同处理REAL分别在五视图的双侧95%分组bootstrap区间均完整位于[0.40,0.60]；RAW REAL–REAL同时报告。同一冻结方法还须满足B–F。PROJECT_ACCEPTANCE.md:15,19,26,36。 |
| 两维输入 | Qtotal_mVns、signed PSD两列，直接包含绝对电荷。Q=1000×(sum100:400−3sum0:100)，PSD=1−Qfast/Qtotal，Qfast窗100:150。没有Q分位数变换、条件重权或删除电荷列。R63 diagnostic_core.py:14–18；evaluation.py:83–87。 |
| 共同ADC表示 | 先按16383级重建整数码，再以float64电压计算特征；原SOURCE_INPUT为数值保源参考，不是A中的生成输出。R63 canonical.py:4–6；assessment.py:10–15；evaluate_saved.py:143–157。 |
| 行身份与数量 | 使用原R26保存phase0/phase1的三域选择，每域600行分类拟合、250行留出。SOURCE用域列j=0，对照REAL DEV用j=1；同处理REAL TRAIN-reference用j=2。后续不重新按新输出QC筛行或重分区，报告QC数量。R26 evaluate_endpoints.py:26–31,55–56；R63 evaluate_saved.py:35–47,69–85。 |
| 分类预处理 | 将两域600+600行合并；median imputer和StandardScaler只在分类训练行拟合，再变换250+250留出行。不拟合留出预处理。R63 diagnostic_core.py:107–112。这类正尺度仿射变换不消除有限、非恒定Q列里的阈值信息。 |
| HGB与预测 | max_iter=100、learning_rate=.06、max_leaf_nodes=15、min_samples_leaf=20、l2=1、early_stopping=false、random_state=20261008；candidate标签1、REAL标签0，p>=.5判candidate。新候选按这一固定设计新拟合HGB；“固定”不是跨候选复用同一个已训练HGB。R63 config_post.json:2–14；diagnostic_core.py:109,113–120；evaluate_saved.py:76–85。原R43同定义见其diagnostic_core.py:16,107–125、config.json:2–14。 |
| 区间与分组 | 500次以保存cluster为单位有放回重抽，按类内权重算BA，取2.5%/97.5%分位；不重新拟合HGB。SOURCE父组及REAL记录组训练/留出不相交；REAL为人工记录组、同一历史采集。R63 diagnostic_core.py:81–90,116,123；evaluate_saved.py:72–74,109–111；R68来源追查REPORT.md:25。 |
| 同处理REAL | 同一个响应方法作用于原2000 TRAIN_REFERENCE，再与固定2000 REAL DEV比较；REAL cluster统一R命名空间。不是新独立采集，也不是仅比较SOURCE输出与其自己的REAL教师。R63 evaluate_saved.py:69–74,146–159。 |

这里的“分类训练600行”属于固定评价人口内部的拟合行；不是生成器训练用的6593 SOURCE/21047 REAL全人口。本次没有解码私有身份/矩阵重做字节核验；实际600/250、两列、无缺失及原行保留由已封存结果JSON核对。

## 已有证据足以说什么

| 证据 | 封存值 | 能支持的结论 |
|---|---|---|
| R68全TRAIN容许区间证书 | KS下界34468748/138762871=0.248400366；完整有限经验样本阈值BA下界0.624200183 | 指定原q容差/ADC界类内存在不可消除的Q经验信号；不是原评价身份、固定HGB留出或其区间下界。R68 RESULT.json:33–88。 |
| R47原SOURCE charge-only | train BA .695；原600/250留出BA .586，95CI[.540830439,.626027758] | 原固定一维诊断不确定：点值未超过.60，但区间也没有完整落入验收带。R47 results/CHARGE_SOURCE.json:2–24,30–42。 |
| R47 LONG charge-only | BA .590，95CI[.543916856,.626281551] | 同样不确定；不是二维HGB的理论下界。R47 results/CHARGE_LONG.json:2–24；R47_REPORT.md:9–15。 |
| R47 RAW reference charge-only | BA .478，95CI[.423343588,.525233070] | 此固定一维对照未检出超限差异。R47 results/CHARGE_RAW_REFERENCE.json:5–24。 |
| R63原两维门 | SOURCE BA .700，95CI[.663403774,.736061000]；同处理REAL BA .488，95CI[.439633835,.532282160] | 确证该冻结候选SOURCE二维门失败、该REAL二维控制通过。不能约束未来候选重新拟合后的HGB。R63 results/C_COVERAGE_RESPONSE__Q_PSD_2D.json:2–33及REAL_CONTROL同名结果:2–33。 |

R47 numeric_worker.py:194–213与tasks/R47_POPULATION_NOISE_DIAGNOSTIC_001.md:15,24–26明确采用原身份、预处理、HGB参数和bootstrap。它没有枚举每源容许区间，也没有全6593/21047 TRAIN新分组，因此与根提出的新稳健一维证书不完全等价；但**“固定实用电荷分类器能否在留出发现差异”并非尚未做过的问题**。原2000的KS=.2685与该固定一维留出点值.586并存，直接说明不能把最佳阈值性能替换为实际有限分类器性能。

## 不能跳过的推理链

从R68推出原二维HGB必失败，仍缺：①全TRAIN质量与原评价600/250身份/权重之间的对应下界；②允许电荷漂移与每个候选实际训练/留出Q值的绑定；③在可改变PSD条件下，有限直方图树、正则、固定训练算法必检出该Q信号的保证；④固定预测在原分组bootstrap下区间必超线的保证。仅说明模型类可以表达Q阈值，不保证这次拟合会选到它；加入PSD列也不保证有限留出BA单调提高。

R63二维失败属于已实现模型的负结果。R47 uncertain不是A通过，也不是阻断证据。source/REAL解析幅度假设、触发/未记录零沉积/源谱、淬灭及独立标定仍有unknown（R68来源追查REPORT.md:5,9–25），不能把差异全归为响应优化不足或已证明的单位bug。

## 条件响应学习与最小下一证据

在原Q/权重不变时，依原完整SOURCE和Q条件学习零电荷的形状/PSD/噪声增量，是可检验的研究假设；允许它改善条件响应，**不保证**总体A通过。理想条件响应匹配仍保留SOURCE的Q边缘；若它被有效评价器检出，合理的波形学习不能靠改变PSD消除该边缘。需要同时检验原源秩/谷比例/扰动、九边界及同处理REAL，不能以重采样REAL波形、隐藏电荷或利用HGB盲区代替。R51_REPORT.md:3,7–15也显示REAL五视图通过与SOURCE总体失败可以并存。

可独立继续的最小证据是：复用R47与现有固定门收据；只用合法TRAIN形成一个明确处理非电荷条件残差的新机制假设，先固定其可检验预测与同起点/同成本对照，再由根决定是否登记。判定新响应候选有效最终仍须其原600/250两维及其余九个A区间、B/C和独立重放；一维诊断不能替代。新的全TRAIN一维HGB阳性不会证明原A必败，阴性不会抹掉R68下界，两者均不提供波形机制或成本收益证据；本次不执行，成本hold不据此解除。优化方向由根及其另一审阅者处理，本审核不扩展该方向。

现阶段没有已证实的总体科学阻塞，也没有因缺用户回答而必须停止上述独立准备的事项。只有要改变原Q目标、人口/权重或把无条件A改为条件验收时，才需要具体的前瞻性目标授权；若要声称物理标定或跨采集响应有效，则需要外部标定/源谱/采集证据。它们不是本次原历史开发A–F强加的新完成条件。
