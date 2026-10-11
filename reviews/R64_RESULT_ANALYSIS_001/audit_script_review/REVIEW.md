# R64 结果审计脚本静态复核

结论：发现项已由 root 修复，当前脚本未发现剩余静态阻塞。冻结 worker、统计模块、几何模块、配置及根封存清单均保持原哈希。

- 两处几何键名已由 `__sliced_W1` 改为生产端实际保存的 `__projection_W1`。
- 已补齐端点统计、名义电荷下界聚合与 RESULT 对照、NPZ 形状及 scope/view 顺序、job-tree RSS/12CPU 和本机峰值资源检查。
- 参数离散度改由保存向量的中心化偏差独立复算；端点原始二阶矩差仅作描述，有限恒等式使用直接均值误差、标量记录 MSE 和保存的 Welford 离散度，保留原容差及原 residual 括号。
- 九条 PCG64 流的消费顺序、形状、类型转换及 hash 链与 worker 一致。统计 bootstrap hash 的 `<i8` 与设计文件的 `<u2` 区别已正确处理。

改前 SHA：`b4a3f4d77bcebe578fc095367e1e7d01d6807632ed03d822eeb7508d8fbf2493`。
改后 SHA：`bc32340bb613640825c5258df5dadb85acf1eb69a4d571653fe946242a9afc65`。

本次只读取源码和已完成的 6 份 JSON；第 7 份 NPZ 未打开，其结构依据生产源码核对。未读取科学数组/权重，未运行 root 审计、人工测试、模型、损失或 VJP。静态检查进程实际固定 1 CPU、监测限额 1 GiB，观测峰值 30490624 字节，两次原生退出均为 0。

完整发现、修复行号和契约见 [REVIEW.json](REVIEW.json)；输入哈希及资源回执见 [STATIC_RECEIPT.json](STATIC_RECEIPT.json)。本回执不代替完整 32 组结果的实际数值审计；后续梯度分析等待下一条指令。

