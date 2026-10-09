# R10：组成配平未消除差距，固定源对照仍未学到完整分布

R10两项诊断已在服务器RTX 5070上完成。当前方案仍未达到完整实测响应相似；本轮保留负结果，停止于原登记范围，不自动启动R11。仅使用TRAIN/DEV，EVAL和新G4均为0。详见[阶段完整报告](work/R10_MATCHED_COMPOSITION_AND_TARGET_ONLY_001/REPORT.md)及[ROOT关键结果判断](reviews/ROOT_R10_KEY_RESULT_REVIEW_001.json)。

## 组成配平的结果

在REAL TRAIN上冻结电荷–PSD网格及权重后，DEV保留1857/2000条R3B和1879/2000条实测记录，有效样本量约1797和1851。共同支持内，完整shape判别BA从0.99578变为0.99376；加权减未加权的配对95%区间为[-0.00594,0]。实测对实测加权shape BA为0.52608，区间[0.48342,0.56917]；完整电压视图加权BA仍0.98724。这里BA越接近0.5越难区分，但0.5附近本身不构成等效证明。

这个配平方法没有明显缩小波形差距。坐标本身仍可区分：加权电荷–PSD判别BA为0.58792，区间[0.54625,0.63052]，logQ距离基本未降。因此不能说“已排除组成效应”，也不能把全部剩余差距归于高阶响应。这些权重是观测空间诊断，并非已识别的物理源谱修正。

## 固定源随机生成的结果

同一完整波形架构从原公共初态完成2000次G/D更新，内部TRAIN验证选择第1500步；2000次随机生成的完整shape、prompt和峰后判别BA均为1，完整电压BA为0.998。电荷W1为2141.37 mV·ns，PSD W1为0.02159，上升时间W1为1.988 ns。生成分布收缩成窄带，八个固定电荷区间中仅第5格能进入共有的两成分峰宽比较，不能把这一格代表完整电荷范围。

对2592条REAL TRAIN的最近邻检查没有发现完全复制，但生成shape的近邻距离也明显收缩；这不证明没有记忆。保存模型和实际随机输入已在服务器独立进程重放，2000条、960000个float32值一致；这检验固定推理，不等于跨环境重训验证。

原初态和固定验证输入可复用，但2000步训练采样合并摘要与历史R8C均未匹配；软件/设备及采样差异保留为限制，不能称严格配对。固定源还把全部事件差异交给随机分支，故此次失败不能唯一归因于G4来源或源条件。

## 后续判断与交付

优先方向是在新的有界任务中，先检查当前表示、训练目标和优化能否学好REAL TRAIN自身的总体分布，建立通过或停止标准，再恢复G4映射研究。此次不继续改权重、换模型、增加种子或打开EVAL。

主计算与独立重放合计97.34秒，其中训练71.65秒；含只读保存特征的绘图修订共98.56秒，完整费用见[资源回执](work/R10_MATCHED_COMPOSITION_AND_TARGET_ONLY_001/RESOURCE_RECEIPT.json)。共26个预定HGB，8个密度描述槽位中实际执行2次EM，其余支持不足保留；CPU最多4线程，峰RSS约2.85GiB，GPU峰分配约255.4MiB。

本次按用户要求直接传输，没有新增文件迁移一致性测试。报告、代码、聚合指标、关键图进入[私有GitHub仓库](https://github.com/Aggressivecyber/ej276d-sim2real)；模型、实际输入与逐事件载荷保存在E盘及D盘副本。准确提交和复制状态分别读取本地R10_PUBLICATION_RECEIPT.json和R10_BACKUP_RECEIPT.json。D盘复制完成不宣称逐文件字节一致性已验证。

- [复现说明](work/R10_MATCHED_COMPOSITION_AND_TARGET_ONLY_001/REPRODUCTION.md) · [交付索引](R10_DELIVERY_INDEX.json)
- [配平分类结果](work/R10_MATCHED_COMPOSITION_AND_TARGET_ONLY_001/results/figures/matching_c2st.png) · [共坐标完整电荷–PSD分布](work/R10_MATCHED_COMPOSITION_AND_TARGET_ONLY_001/results/figures/charge_psd_uniform.png) · [最近邻诊断](work/R10_MATCHED_COMPOSITION_AND_TARGET_ONLY_001/results/figures/nearest_neighbours.png)

既有R1–R9报告、冻结文件和交付版本保留。所有结论限于同一次采集、已查看DEV、当前配方及预算，唯一物理失配原因仍未确定。
