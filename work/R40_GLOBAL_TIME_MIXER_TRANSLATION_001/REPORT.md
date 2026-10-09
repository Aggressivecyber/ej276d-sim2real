# R40 固定全段时间混合层：关闭此配方

唯一2950末点未通过完整响应共同门槛，六门结果为 **失败／失败／通过／失败／失败／通过**。完整 shape493 BA=.984、纯u480 BA=.982，相对冻结R32 BASE=.978的配对差分别+.006（95%CI [−.008504,.021652]）与+.004（[−.011299,.020007]）；两区间上界均不小于0，且远高于预定.934/.928点值限额。保存模型与4000条独立完整链重放通过，不能把可复现代替科学通过。关闭这一固定480→128→480单层配方，不据此声称所有非局部模型无效。

本轮保持完整480点G4派生源波形进入原网络，不是实测目标密度独立抽样。唯一结构在最终16通道style FiLM后、signal/electronic两个头前共享一次 `h + Linear(128,480)(GELU(Linear(480,128)(h)))`，新增123488参数；W2零初始化。时间混合思想参考[Tolstikhin等，MLP-Mixer](https://proceedings.neurips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html)，只使用跨时间MLP思想，不是完整MLP-Mixer复现。原网络已有全局归一化、扩张卷积和style，新增连接也增加参数/FLOP，不能将对比唯一归因于感受野。

完整R32 CONTROL配置、原初始化、STANDARDIZED D、所有原损失及fresh分位数L1恢复；没有R39 MMD、均值额外项或后置校准。2950×32的94400行、七项实际源/实测/NCE/随机流、8850组fresh方向均匹配冻结对照。实际暴露为3752唯一SOURCE与18739唯一REAL，而非载入6593/21047全部参与更新。G/D/reader各2950。64 SOURCE与64 REAL的旧/新三功能分量初始逐值相同，共256额外模型前向；原CPU/CUDA RNG未被新层初始化推进。初始J20=.6166301966，差0；组件核对在更新前，J20差断言实际在训练完成后执行，此顺序局限保留。内部选择1250的J20=.5569948554；实际2950末点J20=.6295024157，作为唯一DEV端点，没有用DEV换成1250。

第一更新W1权重/偏置梯度0为零W2初始化的预期，W2权重/偏置梯度2.111989975/.882978976。四个新增参数都与保存initial比较，变化有限且非零：

| 新模块参数 | final−initial L2 | 最大绝对变化 |
| --- | ---: | ---: |
| 0.weight | 2.9995385 | 0.0880984 |
| 0.bias | 0.2132163 | 0.0492223 |
| 2.weight | 3.6347871 | 0.1580299 |
| 2.bias | 0.2114742 | 0.0393114 |

两侧PSD峰位及宽度同时退化，不能用总体均值曲线或ACF单项改善作成功判据。下表所有PSD距离/宽度是同一冻结定义；峰宽拟合为密度成分描述，不是粒子标签真值。

| 指标 | 冻结 BASE | 本轮2950 | 相对变化 |
| --- | ---: | ---: | ---: |
| 总体 PSD W1 | 0.0230654614 | 0.0277634486 | +20.37% |
| 共同 Q bins1–7 等权 PSD W1 | 0.0347657952 | 0.0388366591 | +11.71% |
| 低侧峰位 MAE，bins2–7 | 0.00585717151 | 0.00772680524 | +31.92% |
| 高侧峰位 MAE，bins2–7 | 0.00265866336 | 0.00309835047 | +16.54% |
| 低侧 raw-width MAE，bins2–7 | 0.0103927218 | 0.0122486032 | +17.86% |
| 高侧 raw-width MAE，bins2–7 | 0.0130511744 | 0.0174877905 | +33.99% |
| 基线 RMS W1（V） | 1.93583857e-05 | 2.95591773e-05 | +52.69% |
| 基线 ACF1 W1 | 0.0642349714 | 0.0332918361 | -48.17% |

总体和共同Q PSD W1分别恶化20.37%/11.71%，低/高raw-width误差恶化17.86%/33.99%，峰位误差恶化31.92%/16.54%；基线RMS W1恶化52.69%，ACF1 W1改善48.17%。共同峰位/宽度区是bins2–7，BASE这六区均继续合格，没有缩交集掩盖新增失败。主候选bin8只有38条，`total_support_below40`，不宣称覆盖最高区。新REAL控制bin1/8各因`not_converged500`不支持；所有旧密度缺口保留在 [GATE_METRICS.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R40_GLOBAL_TIME_MIXER_TRANSLATION_001/GATE_METRICS.json)。

2000原源父ID、组成、顺序、权重全部保持，417个TRAIN支持外父也用新G，没有BASE回退、删行、新无效或ADC裁剪。actual preADC源-Q的pure-had bins3/4分别213/181父，谷比例BASE5.6338%/.5525%→本轮2.8169%/0%；历史postADC SOURCE_CLEAN的207/182父另外保存，不能混称。全人口源Q→输出Q Spearman .9997263→.9997787，源PSD→输出PSD .4189917→.4243149，源代理门通过，但相关性与跨域谷区不是逐事件粒子保真真值。源交换有作用，也不证明物理对应正确。

9项固定边界中3个二阶量失败：center99 W1=.215528mV>限.106272mV；center149=.749649>.742689mV；center150=.807848>.703809mV。其余6项通过。前128个支持内完整SOURCE输入150:400零积分1mV扰动保持原噪声；输出全部非相同、preADC全480 RMS中位1.140239mV，输入Qfast严格不变、Qtotal/PSD仅保存float32微小误差。扰动敏感性单项通过不抵消边界失败，不解释成物理增益。

同处理REAL控制完全复用原2000 TRAIN_REFERENCE及主z/white/style；旧BASE_REAL波形、10评价器/密度/回放引用封存R39，零重生成/重拟合。新减旧的预定三门如下，baseline点差+.016超过+.01，故控制门失败。V480及其它视图只描述。对RAW的绝对失真完整报告；随机G可能对已含噪REAL再加响应，不能预设RAW恒等。

| 同处理REAL视图 | BASE BA | 新BA | 配对差与95%CI | 预定门 |
| --- | ---: | ---: | --- | --- |
| shape493 | 0.970 | 0.968 | -0.002 [-0.01487, +0.01050] | 通过 |
| 纯 u480 | 0.964 | 0.972 | +0.008 [-0.00438, +0.02098] | 通过 |
| baseline | 0.658 | 0.674 | +0.016 [-0.03222, +0.07185] | 失败 |

最终新模块关闭描述只对固定前128行进行，preADC/信号/电子全480 RMS差中位分别.883872/.876522/.086752mV；全部非相同。它表明新增层在最终网络中实际作用于两个头，不是第二候选、纯方差消融或从头无模块训练反事实，不对它拟合HGB/EM。signal/electronic共享总归一化及交叉项，仅为功能分量，不宣称独立物理噪声。条件均值/中位尾部仍有固定时间起伏，DEV相关矩阵亦保留结构差异；prompt100:150包含峰和早期下降，不能全称纯上升沿。

服务器唯一训练真实PID77574、监督77573，GPU allocated峰283249664B，RSS峰2707943424B；训练记录115.998228s，含导入/封存的原生过程118.756857s。两次RSS查询gap保留22:27:54.529290/.629565时间戳，原因unknown，不将靠近退出当成已证实原因。其余四个本机科学过程原生0；科学累计212.281716118s/1800s，20新HGB、16描述槽/15实际EM，旧评价器零新拟合。新增参数/初始化256前向单列，FLOP未测量，不声称等算力。所有13张最终PNG已实际打开检查，布局/单位/支持缺口清晰。

本轮使用已暴露的同采集TRAIN/DEV与人工50记录组，区组bootstrap条件于固定分类器和这些组，不涵盖响应重训或独立采集不确定性。没有新EVAL/CAL波形、REAL采集、G4、安装或迁移内容测试。结论限于这一固定全段时间MLP未解决总体响应差距；不追加本轮变体或新拟合，下一问题由根依据完整证据登记。

完整聚合结果 [METRICS_COMPACT.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R40_GLOBAL_TIME_MIXER_TRANSLATION_001/METRICS_COMPACT.json)；全门槛 [GATE_METRICS.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R40_GLOBAL_TIME_MIXER_TRANSLATION_001/GATE_METRICS.json)；训练 [TRAINING_COMPACT.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R40_GLOBAL_TIME_MIXER_TRANSLATION_001/TRAINING_COMPACT.json)；重放 [REPLAY_RECEIPT.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R40_GLOBAL_TIME_MIXER_TRANSLATION_001/REPLAY_RECEIPT.json)；13图实际QA [FIGURE_QA.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R40_GLOBAL_TIME_MIXER_TRANSLATION_001/FIGURE_QA.json)；实际资源/原生链 [RESOURCE_RECEIPT.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R40_GLOBAL_TIME_MIXER_TRANSLATION_001/RESOURCE_RECEIPT.json)、[NATIVE_CHAIN.json](E:/GAN/research/sim2real_two_papers_20261008_0305/work/R40_GLOBAL_TIME_MIXER_TRANSLATION_001/NATIVE_CHAIN.json)。完整逐事件输入、预测、随机流、模型与输出保留E/D，不进入聚合发布版。
