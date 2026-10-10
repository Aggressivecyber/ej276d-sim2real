# R45：长训练完成，固定32步方案数值门失败

服务器已从R43的2950步模型及Adam继续47050次更新，累计固定50000步。固定TRAIN探针全480维速度MSE由0.875450319降至0.837009741，降低4.391%；PSD速度RMSE由0.029373776降至0.023099292，降低21.361%。前者未达到预定10%描述标记。11次探针波动均保留，没有据此选择checkpoint，路径误差也不等于生成波形质量。

在原64条SOURCE加64条REAL的TRAIN输入及white上，32/64步preADC RMS差中位为 **0.422870993 LSB**，超过0.25门槛；p95为 **0.866297412 LSB**，满足1 LSB门槛。最大float32相对电荷误差约1.62e-8，满足1e-5。数值门总体失败，按预案在DEV前停止并关闭本固定32步配方。前五项分布/科学门均为not_evaluated，DEV生成、HGB/EM和4000条主输出重放均为0；不能据本轮训练推断完整响应已经改善。

载入SOURCE/REAL池为6593/21047，实际唯一父事件仍3752/18739。新增1505600行/域，加上历史94400行，共1600000次行出现；它们不是独立新事件。模型、480维MSE、优化器规则、常数和原经验耦合保持；新增t接续历史PCG64，white使用固定新种子，属于同一目标的额外Monte Carlo抽样。统计耦合不是逐事件物理真值。

实际GPU训练含探针与保存1443.364407秒，CPU12数值检查16.143875秒，本机图表3.650281秒，科学合计 **1463.158563/14400秒**。三次科学进程均native0，科学门失败与进程失败分开。历史R43训练76.152306秒另列。实际worker RSS峰2,042,970,112B，GPU allocated/reserved峰313,275,392/343,932,928B，进程显存采样峰629,145,600B；新服务器配额未触发。两次终态RSS查询缺测原因unknown，三次普通控制查询错误原样保留。

最终206文件、461,828,065B，封存工具81587a/PID56888/native0。五张最终图由执行器及根实际打开核对；权重、Adam、每5000步工程断点、随机状态和逐批记录保留在E盘/服务器。准备的DEV评价代码未执行。源电荷人口与总体联合分布目标的兼容性另有明确说明，不改变验收线或源权重。

下一独立问题为已冻结短/长模型的共同积分预算：预先固定64步候选及128步参考，保留0.25/1 LSB与1e-5容差。两组均通过才比较完整响应；不追加训练、不从中间末点选择、不继续搜索更大步数。R45本身保持关闭。该新配方只有正式登记后执行，完整模型目标仍未完成。

[完整报告](work/R45_EXTENDED_BUDGET_NEURAL_FLOW_001/REPORT.md) · [固定探针](work/R45_EXTENDED_BUDGET_NEURAL_FLOW_001/results/figures/fixed_TRAIN_probe.png) · [数值门](work/R45_EXTENDED_BUDGET_NEURAL_FLOW_001/results/figures/TRAIN_solver_gate.png) · [复现入口](work/R45_EXTENDED_BUDGET_NEURAL_FLOW_001/REPRODUCTION.md) · [目标兼容性说明](reviews/R45_GOAL_COMPATIBILITY_NOTE_001.md) · [R46协议](tasks/R46_FIXED_INTEGRATION_BUDGET_001.md)。Git推送与正常D副本以实际R45_PUBLICATION_RECEIPT.json、R45_BACKUP_RECEIPT.json为准。
