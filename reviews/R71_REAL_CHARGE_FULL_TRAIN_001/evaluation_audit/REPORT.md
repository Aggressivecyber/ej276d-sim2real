# R71：全量训练的评价迁移静态审计

结论：可以在短工程测时支持数小时完成时直接选择全量，但 **R65 模板不能原样作为 v5 训练/评价**。关键是保留原人口与原来源轴，将目标条件改为实际 canonical 输出 Q，并让 Q* 贯穿公共响应路径。以下是最小接口迁移清单；没有新模型可行/不可行结论，也未审定尚未冻结的 R71 实现。

当前入口 CURRENT_ACCEPTANCE.json:2–7 已正确绑定 v5 的文件和 SHA。v5 的 A/B/D 数字与来源、边界、REAL 控制仍有效；最新“可数小时完成则直接全量”指令允许根选择全量执行，覆盖 v5.md:40 的先小试执行顺序，不改变科学目标，也不恢复已退休的 R65。

下文 F65=work/R65_FULL_EMPIRICAL_D_RESPONSE_001，F63=work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001，V5=PROJECT_ACCEPTANCE_REAL_CHARGE_V5.md。

## 1. 人口与执行方式分开绑定

- 合法完整 TRAIN 仍为 SOURCE6593/REAL21047。F65/train_final.py:358–410 从冻结 TRAIN 包读取并核对形状、SOURCE ID、原输入 bins、组成及全人口质量。F65/data_support.py:49–55 描述原 REAL 索引/内部验证排除与来源。不得把“全量”解释为并入原1599 SOURCE验证、599 REAL验证或 DEV/EVAL。新入口必须绑定原 SOURCE ID **及 REAL 原索引/顺序**；旧训练装载只显式读取 source_ids，不能靠行数替代 REAL 身份核验。
- 完整经验 SOURCE/REAL 两域各自原每父等权：全量分别 1/6593 与 1/21047。全人口整体损失不能先把每输入区平均到同质量；按原输入区抽样时，保留原区人口/采样概率的校正与重复次数。F65/distribution_adapter/distribution_gradient.py:244–250 已区别完整均匀质量与分层样本质量。
- DEV 保留原2000 SOURCE父 ID、顺序、组成、权重及原417支持外全部响应，另有原2000 REAL_REFERENCE。F65/evaluate_saved.py:143–164、generate_all.py:36–47 已绑定原身份。Q头改变输出条件，不改变这些父成员；原支持外与新输出支持外分别列账。每父重复生成不增加独立父事件数。
- “全量数据可用”“每次全人口目标”“完整梯度”是三件不同的事。R63每步32/区是原分层抽样（F63/train_final.py:91–101）；R65的 full-D 是6593输出与21047冻结参照的完整有限人口（F65/train_final.py:434、566–602）。R71应明确全量到底覆盖哪些项，特别是 REAL identity 项，不能仅凭载入了全数据宣称每项全梯度。

## 2. 原输入轴与输出轴的明确分工

| 项目 | R71 应使用的轴/质量 | 当前模板及必要迁移 |
|---|---|---|
| 采样、源来源/组成、原支持、源PSD精度区间 | 原输入 Q0、原 b0、原父质量 | 保留 F65/train_final.py:371–394 的原成员；不能按新 Q* 或 b_ADC 重新选父。 |
| SOURCE Q/PSD rank 与 purehad bins3/4 valley | 全原父配对 Q0→实际输出Q、PSD0→输出PSD；valley 成员仍为原名义 bin | F65/flow_diagnostics.py:93–104 与 F63/summarize_and_plot.py:61–74 已有原成员主门。原 preADC purehad 两分母213/181，与旧postADC参考207/182分别保留（F65/evaluate_saved.py:160–172）；输出自归属谷比例只作另表。 |
| PSD整体、全人口噪声、电荷边缘 | 实际 canonical 输出；原全人口权重 | 保留完整父分母。新增 Q CDF/KS/W1/counts/weights/outside mass 是描述量，不事后挑通过线（V5:19）。 |
| full_window、coordinate_RMS、temporal_joint、条件PSD | 输出 canonical Q→冻结 REAL 边界；目标 REAL 用其原 canonical Q | 旧 F65/R61_distribution_loss.py:77–88、F65/distribution_loss.py:64–75 按固定10×B槽位；完整适配器:208–217、276–321 仍按原 source_regions/source_mass。必须改成新输出成员/明确条件归约，不能直接改名；原SOURCE区质量不能冒充新的输出区质量。 |
| B条件W1/峰宽/噪声表 | 每域自身 canonical 输出Q；固定必需区、原REAL分母 | F65/evaluate_saved.py:25–32、113–138，endpoint_common.py:1–21 已基本使用自身输出Q，优先复用这部分；支持不足/拟合失败不能从跨区均值删掉。 |
| REAL identity 训练项 | 与原 REAL 输入逐行配对；参考标尺按原REAL条件冻结 | F65/R61_distribution_loss.py:100–101 的原输入条件归一化可以保持其参考角色，不能靠移动输出bin改变误差分母。该训练项不能替代完整同处理REAL五视图评价。 |
| Box correction、扰动、九边界 | 每原父完整响应；Q*为投影目标；原扰动/位置/对照不变 | Box residual不改人口质量；128扰动重新通过整个Q头+波形链，不能缓存未扰动Q*；九边界从真实输出电压重算。 |
| E的缺口 | 原固定CAL q3；按输出自身条件及原名义条件两套报告 | V5:34–36 为 REAL-only GAP，保留全部合法SOURCE。B/valley的8个logQ bins 与 E的CAL四区q3不是同一分区，不可合并。 |

条件训练的 hard canonical 归区、STE 或其他代理梯度必须如实定义；整数/NumPy归区没有自动变成可微选择。旧统计只有成员与 canonical REAL 参照逐一一致才可沿用，不能假设原输入bin与目标表示永远相同。若需要新统计，只用合法TRAIN并在首更新前固定；GAP不能借FULL缺区统计。

## 3. 最小具体适配清单

1. **独立 TRAIN loader/人口回执。** 复用冻结TRAIN包及合法索引绑定，保留父顺序与质量、验证排除、模型/常数/随机流SHA。工程测时不用 F65/data_support.inputs()：其:41–56 同时读取DEV并构建验证/DEV来源，不能作为干净TRAIN测时入口。
2. **独立 Q0/Q* 记录。** 公共 Q-head 从输入与冻结参数提前给出 Q*；输出保存 parent_ID、Q0、Q*、preADC Q、canonical Q、b0/b*/b_pre/b_ADC、原metadata/权重/随机流及原/输出support。Q*不是输出积分的别名（V5:10–12、26；json:73–85）。
3. **整链接 Q*。** F65/source_flow_core.py:57–84 的 guide、初态、零电荷残差、Box、数值误差目标，与 noise_core.py:35–45 的条件/零电荷噪声都需一致地迁移。若采用lifting，它应先建立Q*电荷坐标；不是只把 endpoint 最后的 bounded_charge(raw,q) 换参数。
4. **保护原字段。** innovation 接收lifting后的x*会把Q*/b*写入旧 source_q_Vns、source_Qbin_actual_preADC、inside_TRAIN_support。必须先保存原Q0/b0/support，并给目标噪声条件另设新键；否则原417计数、purehad分母及 generate_all.py:55 的128扰动选父会悄悄改变。
5. **分离来源与目标 layout。** 分布目标使用 canonical输出Q ragged成员；来源C项继续原b0。整体质量仍原每父权重；条件平均的区集合/归约显式绑定。空区/未知支持不能默认为零距离。共享明确的输出归区函数；旧 source_tables/association 的手写searchsorted与 evaluate_saved.qb 在最右边界处理有差异，需按冻结边界语义统一新输出表，保留原b0记录。
6. **版本化数值与门字段。** 新误差为 abs(Q_preADC−Q*)/abs(Q*)≤1e-5，同时检查f64轨迹与实际保存f32。旧 CHARGE_CONSERVATION_SUMMARY/source_q 分母、固定50000/原初始化SHA/旧训练审计不能改名后当成R71回执；新增Q头应有自己的初始化、参数数和冻结端点绑定。新A–C/阶段门从保存指标逐项重算；原六门中只有 SOURCE-Q 绝对守恒的对象被显式替换。
7. **完整公共控制与独立重放。** 同一Q头、lifting、噪声、波形、Box和ADC处理SOURCE/REAL，不加域标记/REAL旁路（V5:11）。原五视图、固定600/250身份、训练侧共同imputer/scaler和分组bootstrap保留，不能随新Q重抽HGB样本（F65/evaluate_saved.py:69–88；diagnostic_core.py:107–125）。4000主行独立进程从输入重算Q*及全链，随后才打开保存输出逐项比较（F65/replay_all.py:17–29）；不能拿保存Q*作重放输入。

源rank仍对BASE比较下降≤0.05，原名义valley相对BASE增加≤0.02；九边界限值 max(BASE,INDEPENDENT_CFM)+1/16383 V；原128扰动非相同比例≥0.95且preADC RMS中位>1LSB；同处理REAL的原配对点差≤.01/上界≤.10及最终五视图[.4,.6]仍要求（V5:19–21、30；F63/summarize_and_plot.py:63–98）。变Q本身不是撤销这些门的理由。

## 4. 工程短测时只决定执行路径

根可先固定模型、损失、合法数据、初态/种子、训练终点、精度参考与评价规则，再用隔离的TRAIN工程前向/反向测吞吐/峰值内存。测时不打开DEV/EVAL、不运行HGB/密度/完整响应门，也不按新Q/PSD改善、loss大小或候选分数挑架构/seed/检查点。记录真实计算；若做更新，记录为实际更新并按预声明规则处理，不伪装为0训练。

正式运行从规定初态和随机流开始；测时的模型、optimizer、梯度、buffer与RNG变化不得悄悄带入。预测时间应包含Q头、完整损失/排序、反向/重算、数值检查、4000重放和固定评价开销，列实测范围及预测误差。数小时可完成就直接全量；设备/chunk等执行选择不能改目标、采样权重或精度。由full-gradient改成抽样梯度属于训练定义变化，须在同一次R71方案绑定中明列，不包装成纯运行优化。无需额外科学小试选模型或再向用户逐步申请。

## 5. 必须失效/退出或记未验证的条件

- 人口/身份/顺序/权重改变，删支持外父、波形去重、复用REAL旁路/域标记，或DEV/EVAL/隐藏GAP统计进入拟合/选参：协议不成立。
- Q*非正/非有限、|Q*|>300的当前Box不可行、f64/f32残差>1e-5、非有限/新增无效/ADC裁剪，或通过裁Q/删父/改分母隐藏问题：候选失败；投影修正、rails和量化误差仍要单列，clip_count=0不能说明投影无失真。
- 只改最终投影而内部仍用旧Q条件，原字段被覆盖，目标layout仍原输入bin，或梯度/质量归约与所声明目标不一致：实现未通过，不能直接全量沿用。
- 未预绑定独立Q模块精度参考/完整波形N与2N，或数值门失败：不进入DEV科学评价。非迭代头不能虚构N/2N通过（V5:26–28；原numerical_check.py:24–43只核旧目标）。
- 原必需条件区不足40、密度失败、rank/valley/扰动/九边界/REAL控制或A/B任一失败：相应门失败；不得交集删区、改边界或挑较好检查点。D/E尚未执行或4000重放不一致则A–F未完成。
- 新工程测时只能形成成本选择证据，不能把R65退休/R69阴性改判，也不构成新总体科学阻塞。当前没有证明v5全量方法不可行；完成上述一次性迁移后可按用户授权推进。

范围回执：本审计只读取现有文本/JSON/源码并做静态AST/哈希；0科学数组读取、0科学函数/拟合/推理/SSH/Git，未修改根或冻结文件。真实全量耗时、梯度及全部数值/科学通过性由根与后续注册执行回执确认。来源SHA和本地资源/原生退出见同目录。
