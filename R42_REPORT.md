# R42：完整源初态神经流在TRAIN数值检查处关闭

唯一2950步训练完成，但固定32步与64步参考的逐波形电压RMS差中位数为1.953997 ADC LSB、95分位为4.790319 LSB，超过预注册的.25/1 LSB。关闭固定R42；没有进入DEV，也没有HGB、EM或4000条主回放。五项科学分布门未评价，第六数值门失败。不能把此结果解释为完整波形分布改善或恶化，也未证明64步是真解。

模型由完整480点源波形和原1mV投影white启动，在共同REAL TRAIN单位电荷坐标中回归电荷切空间速度；新U-Net共1,371,841参数，未加载旧响应权重。训练2950×32行/域，原七项行/随机流匹配，2950批的PSD统计耦合均保留边缘行。载入6593/21047条，实际唯一消费3752/18739条；排序配对不是事件物理真值。

数值检查使用固定64 SOURCE+64 REAL TRAIN和首128 consumed-row white。32/64两解最大float32 preADC相对电荷误差分别1.33560e−8和1.42226e−8，满足1e−5限制，仍不能弥补积分精度失败。电荷约束、模型有限性与分布有效性是不同结论。417支持外主源的设计尚未执行，不能写成已验证。

三个实际科学进程均native0：训练75.791690秒、TRAIN检查52.046501秒、仅保存证据绘图3.549625秒，共131.387816/2700秒。真实worker峰RSS 2,188,296,192字节、GPU allocated峰341,479,936字节，CPU4；远端两次RSS查询缺测的原因unknown。native0表示正常退出，数值验收仍失败。准备时复制过DEV文件字节，但没有数值解码、生成或评价。新REAL/CAL/EVAL/G4均未读取。

四张TRAIN图均经执行器和根实际打开，布局可读。权重、公共常数、两套TRAIN数值解、输入/随机绑定、曲线和失败记录已封存；143文件、67,212,523字节。根仅读取保存结果，未重算波形。原兼容字段与实际物理语义、未执行autograd自检的静态构图声明均有明确说明。

源码核查显示：32步Heun只在j/32取时间，高频32/64/128的正弦/余弦特征退化为固定值；64步会改变其中32频率的取值。这是明确的特征混叠风险，但尚不能断言它主导此次误差。下一轮仅检验一次标量t输入，保持25输入槽和其他参数形状、训练流、seed过程、损失、2950更新及32步不变。新模型仍先通过相同TRAIN数值门；失败就停止这项时间表示干预，不扫描频率或solver。R42保持封存负结果，完整项目继续。

[完整执行报告](work/R42_SOURCE_INITIALIZED_NEURAL_FLOW_001/REPORT.md) · [数值误差图](work/R42_SOURCE_INITIALIZED_NEURAL_FLOW_001/results/figures/TRAIN_solver32_vs64.png) · [完整TRAIN波形](work/R42_SOURCE_INITIALIZED_NEURAL_FLOW_001/results/figures/TRAIN_solver_waveform_examples.png) · [复现入口](work/R42_SOURCE_INITIALIZED_NEURAL_FLOW_001/REPRODUCTION.md) · [R43协议](tasks/R43_SCALAR_TIME_NEURAL_FLOW_001.md)。

实际私有Git与正常D副本完成情况分别见R42_PUBLICATION_RECEIPT.json和R42_BACKUP_RECEIPT.json。
