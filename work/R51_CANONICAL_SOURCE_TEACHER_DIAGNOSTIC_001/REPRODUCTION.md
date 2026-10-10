# R51 复现

在独立目录放置本轮脚本、配置、SOURCE_INPUT_PRIVATE.npz与固定定义。config引用原TRAIN、R43关联和R50输入/预测/指标，按INPUT_BINDING及R50备份恢复这些依赖并更新路径。运行 `python -B linux_supervise_post.py numeric_worker.py`，得到固定35项结果、目标诊断、私有预测和关联、三图及原生回执。已运行目录不可清除.used标记重跑。新增响应训练0；服务器CPU12/RSS48GiB。复现是新的科学执行，需另记实际资源及原生结果；不得覆盖本轮。
