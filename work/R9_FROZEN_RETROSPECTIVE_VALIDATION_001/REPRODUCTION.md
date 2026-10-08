# R9复现入口

本目录是已完成封存结果。直接运行会消费新的评价作业并改写run输出，不应在这个已封存目录再执行。必要复现应复制代码/config及已绑定输入引用到新的E/D工作目录，显式登记复现；不更改当前结果，也不据回顾性EVAL选新响应配方。

原入口为本目录cwd下`python -B launch.py`；launcher只启动`python -B run.py`一次。缓存TEMP/TMP为项目`.runtime-cache/tmp`，PYTHONDONTWRITEBYTECODE=1、PYTHONUTF8=1；数值环境OMP/OPENBLAS/MKL/NUMEXPR/BLIS/VECLIB线程为4。实际Python3.13.3、NumPy2.2.6、scikit-learn1.7.2。当前执行源SHA在READINESS，冻结响应模型/生成/实际随机输入SHA在CONTINUATION_FROZEN_METHOD和FROZEN_BINDING_CHECK，原22输入SHA在INPUTS。

已有R3B/R8/R8C生成和独立重放引用未修改；R9无响应推理或训练。run先核验READINESS/config/已知输入哈希，复用同R4主REAL位置、64真实组半分、CAL参考和TRAIN尺度，42固定分类器及500条件组bootstrap；密度结果从完全相同的已存波形拟合表复用并核对各区计数/直方图。本轮不复制/安装环境或读服务器。

完整本地载荷为metrics.json、results/DATA_ROLES.json、results/C2ST_SPLITS.json、results/C2ST_HOLDOUT_PREDICTIONS.npz、results/PER_EVENT_FEATURES.npz及results/REAL_REAL_REFERENCE.json。身份/逐次选择/预测不嵌入METRICS_COMPACT发布汇总。11图实际QA见FIGURE_QA。封存纯字节SHA清单排除其自身及FINAL_RECEIPT，由FINAL_RECEIPT绑定清单SHA；没有伪造自哈希。

原生回执的两个陈旧说明字段保留，机器可读更正见RECEIPT_FIELD_CLARIFICATION。本轮真实EVAL读取7600、有效7583、固定主2000；正确响应更新0。该更正不改变波形/指标或重跑科学。
