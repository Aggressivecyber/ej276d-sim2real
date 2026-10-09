# R31 reproduction

本轮在 E/D 保存目录运行；旧阶段只读。实际 Python：`C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`。环境版本见 RESOURCE_RECEIPT.json；真实 science PID 在 RUN_HISTORY.json 与每个 NATIVE_RECEIPT 中。

输入路径如下。四域保存输出及 REAL/reference 已由本次唯一 numeric_worker 读取；报告封存不再解码数组。

- real: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R28_FRESH_PROJECTION_TRANSLATION_001\REAL_DEV_AND_REFERENCE_PRIVATE.npz`
- rows: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001\results\INPUT_ROWS_PRIVATE.npz`
- moments: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001\results\MOMENT_PARAMETERS_PRIVATE.npz`
- full480_metrics: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R30_CONDITIONAL_MOMENT_DIAGNOSTIC_001\CLASSIFIER_METRICS.json`
- FRESH_FINAL: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R28_FRESH_PROJECTION_TRANSLATION_001\FRESH_FINAL\generated.npz`
- ENERGY_FINAL: `E:\GAN\research\sim2real_two_papers_20261008_0305\work\R29_SLICED_ENERGY_TRANSLATION_001\ENERGY_FINAL\generated.npz`

配置、原列索引与矩参数引用在 config.json；未重新估计矩。固定600/250行、人工组、QC、原Qbin与N≥20矩支持均复用R30，条件描述N≥10。新私有预测/索引/解码误差在 results，原full480预测路径在配置。

运行顺序为 prepare.py → supervise.py numeric_worker.py → supervise.py render_figures.py → supervise.py render_layout_fix.py → assemble_delivery.py → seal.py；实际三次科学 native 链记录在资源回执。脚本的旧期限/单次标记禁止原目录再运行。未来复现须在获授权的新E/D目录复制源码与配置并重新登记期限，保留冻结数学/选序/参数，不覆盖本轮或旧阶段。

唯一掩码是0:100与150:480，先完整480变换再掩码。解码器只检验两套反变换均值后的max距1，1e−12差内弃权。12 HGB与500次固定分类器配对区组bootstrap仅执行一次。变换需要域身份，不能部署或推导新PSD。
