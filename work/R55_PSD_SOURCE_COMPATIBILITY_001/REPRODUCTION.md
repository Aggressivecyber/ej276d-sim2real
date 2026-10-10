# R55 复现范围

本目录是固定经验人口的必要条件诊断，不是新的响应模型或通过 A–F 的候选。实际结论以 `RESULT.json`、`LOWER_BOUND_CERTIFICATE.json` 和原生退出回执为准。

`INPUT_BINDING.json` 绑定原 R46 的 SOURCE 身份、完整输入、组成和冻结分区/谷定义，以及 R54 的共同 ADC 特征和结果。正式执行保留 2000 个 SOURCE 身份及其等权重、2000 个 REAL 特征、原始 purehad Q3/Q4 的 213/181。没有读取最终 EVAL 或新解码 CAL。

原始研究按 `config.json` 在新的隔离目录中，以 `linux_supervise_post.py numeric_worker.py` 运行一次，入口受 `.used` 与实际进程握手保护。若需重新执行，须先登记新的复现作业并使用独立目录，不能删除原标记或覆盖已封存结果。该入口会重新核对输入绑定，建立固定 LP，调用一次 `highs-ds`，再计算证书和图；参数扫描不属于本次复现。

可以只对保存的矩阵和对偶重算证书：

```text
python -B verify_certificate.py --matrix /data/R55/LP_PRIVATE.npz
```

这条命令不训练、不生成波形、不重新求解 LP，也不需要原始波形数据。它用 NumPy 读取实际保存的 float64 稀疏系数，再用 Python `Fraction` 对其二进制有理数作精确运算。证书以变量 [0,1]、不等式 Ax≤b、截断为非正的对偶 y 为前提，包含全部负对偶残差的保守修正。`LP_PRIVATE.npz` 和 `INPUT_SUMMARY_PRIVATE.npz` 属于私有数组，仅保存在授权服务器及 E/D 备份中。

原始 LP 的行系数向 float64 转换时，右端项按 [0,1] 上的最坏系数误差向外放宽；目标宽度向下舍入。CDF 变量代表固定区间的平均值，Jensen 放宽、可能的 ADC 分箱迁移、McCormick 放宽及源谷概率下界均见 `PROTOCOL.md` 和 `LP_CONSTRUCTION.json`。原本未加入的完整波形、密度、秩、判别器、噪声及 D/E 条件仍属于最终 A–F，不能用本 LP 的可行性替代。

科学进程限制为服务器 CPU12、作业树 RSS48GiB、一个作业。本轮不占 GPU，无计算时间配额。本机文件和临时文件只放 E/D；源代码可从 C 盘安装的 Python 解释器执行，但数据不得写入 C 盘。Git 只保存代码、协议、汇总、图和回执；矩阵、源身份及原始日志由 D 盘增量备份保存。
