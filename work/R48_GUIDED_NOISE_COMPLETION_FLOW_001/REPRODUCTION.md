# R48复现入口

本次实际结果：两臂64/128步TRAIN检查的p95均超过1 LSB，按协议关闭；DEV生成、HGB、EM及主输出独立重放均未执行。以下通过数值门后的入口仅为预备代码，不是本轮已执行证据。

本阶段仅复用已有输入，不读取新的G4、CAL或EVAL波形。配置中的历史绝对路径是必要输入引用，原目录保持只读。实际执行位置为`rylan@100.82.231.102:/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R48_REMOTE_20261010_073406_UTC`，Python为`/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`。环境实测记录见`SERVER_PREFLIGHT.json`。

在新独立目录恢复代码和配置路径后，按`calibrate.py`、`train_control.py`、`train_completion.py`、`check_models.py`的顺序执行。入口由`linux_supervise.py`启动，导入科学库前绑定真实worker PID/nonce并监测进程树RSS。原目录的`.used`标记防止重复提交；复现应创建新目录，不删除既有标记或覆盖原产物。两臂各从同一新初始化训练50000次更新，不加载历史响应权重或从中间点选优。

只在两臂128条TRAIN的64/128步检查均通过后，执行`generate_all.py`、`evaluate_saved.py`、`replay_all.py`。数值检查、生成和独立重放均为服务器CPU12、chunk64；固定Heun64候选，128步只作为TRAIN参考。`replay_all.py`加载保存的末点、共同核及原输入/白噪声，用独立进程核对每臂4000条主输出的pre/post ADC、身份和辅助字段，全部结果由`REPLAY_RECEIPT.json`记录。数值门未通过时，协议禁止DEV生成、分类拟合和主输出重放。

`summarize_and_plot.py`由`linux_supervise_post.py`监督，读取既存结果生成聚合指标和图；若数值门失败，只运行`render_train.py`。绘图不更新网络、不重新拟合分类器或密度。实际执行与失败成本见各入口的`NATIVE_RECEIPT.json`、`WORKER_RECEIPT.json`及`REMOTE_RUN_HISTORY.json`，最终状态见`REPORT.md`和`FINAL_RECEIPT.json`。

本轮四张TRAIN图已生成；`render_noise_display.py`在本机CPU4、`supervise_display.py`真实PID守护下仅从保存检查数组统一噪声规则图的坐标范围。原图另名保留，追加native与科学绘图费用见`LOCAL_DISPLAY_HISTORY.json`、`DISPLAY_CORRECTION_RECEIPT.json`和`NATIVE_CHAIN.json`。该显示步骤不调用模型。

输入配置引用R43保存的94400行source/matched_target/ordinal循环及R42/R43坐标常数；噪声校准仅用实际消费的unique matched_target。四条PCG64训练流分别为20261010481/482/483/484。源与目标的排序耦合是统计假设，不是真实事件配对。只有通过数值门后才进入预备生成分支，要求其输出保留全部2000源父事件和原组成/权重、支持外仍执行同一新流；本轮未进入该分支，没有这些输出。

控制与主候选共享完整guide、REAL自身身份路径、谱核、初始化、训练索引和随机流，只比较初态噪声幅度规则。相对R46同时改变了多个框架部分，不能将历史差异全部归因于噪声补全。100点方差和低5%阈值仅构成统计种子；功能分量、交叉协方差与源敏感性不等于物理噪声识别或正确逐事件映射。已有同采集DEV是开发评价，不能称为新的盲测。
