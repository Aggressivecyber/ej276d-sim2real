已完成两个最小接入文件，仅 AST 语法解析，未运行任何函数、人工 suite、模型或科学计算。旧 evaluator_candidate 和 root_summary_candidate 均保持原样。

安装 evaluate_saved_receipt_variant.py 时命名为 evaluate_saved.py；安装 summarize_and_plot.py 为同名。其余沿用已封存 evaluator 的模块、evaluation_io.py，以及旧 acceptance_gates.py / render_full_response.py。config_post.json 和 adapter config 既定接口无需新增选择或审批。正式流程为 generation、原定独立 held-out4000 replay、正式 evaluate_saved、receipt-only summary；每个入口仍由原生 owned supervisor 启动。

正式评价新增 FORMAL_EVALUATION_GATE_EVIDENCE.json 并在 EVALUATION_NUMERIC_RECEIPT.json 的 gate_evidence_file 绑定 path/SHA。population 的身份/均匀权重/完整处理、Qstar 正有限沿用正式评价已成功的检查；preADC 有限和越界采样数、零基线计数仅首次从已加载 pre32/既有 feature_packet 导出标量。f32 相对 Qstar 误差引用生成器实际 NumPy 预量化误差行字段。row_metadata_consistent 引用已有独立完整链4000 replay的逐字段 equality checks，缺失为 unknown；不额外重放元数据公式。mass 引用正式 charge_diagnostics 已计算的四轴/三张完整迁移，只有 JSON 结构与保存总量检查，无 CDF/KS/W1 或迁移重算。

汇总只消费正式评价 JSON、GENERATION_RECEIPT、REPLAY_RECEIPT、既有 TRAIN 数值 JSON（已嵌入正式 metrics）及原生回执；不再次 preflight_training、不读取原始波形/模型/统计 NPZ，不执行 Q/ADC/身份/电荷分布科学复算。旧绘图器仍读正式评价保存的派生 raw-voltage moment packet 作图。原 acceptance_gates 数字门、10 HGB / 16 density slots、完整2000+2000父事件与独立4000重放范围保留。未知零基线计数会产生 SUMMARY_INCOMPLETE/FAILURE，其他 unknown 不能伪造成 pass。receipt中明确 independent_root_recalculation=False。

本次 AST native 4605d6 exit0，mask4 CPU1 / RSS1GiB，峰值22773760B。此为生产源码准备，无候选质量或验收通过声明；不要求根增加人工检查或数值复算。
