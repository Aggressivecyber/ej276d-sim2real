# R50 复现

在既有服务器Python环境下，把本目录的INPUT_PRIVATE.npz、config.json和三个Python脚本放在新的独立目录，运行 `python -B linux_supervise_post.py numeric_worker.py`。原已执行目录含.used标记，不能删除标记后重跑。输入来自INPUT_BINDING.json，原R49文件保持。固定24个HGB、原分组索引和种子，0响应训练。输出包含METRICS、DESCRIPTIVE_METRICS、三个图及原生进程回执。服务器CPU12/RSS48GiB；每次重跑须独立记录，不以重跑替代失败历史。
