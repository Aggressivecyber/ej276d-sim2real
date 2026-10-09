本轮可复现入口与产物

执行在本机E盘，不需要SSH。训练使用 E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe；评价、独立重放、绘图使用 C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe（解释器只读，所有数据/缓存/临时文件写E盘stage/cache）。线程4；真实science PID导入前握手由supervise.py/owned_worker.py保存。

实际顺序：supervise.py train_gpu.py GPU_PYTHON → supervise.py evaluate_endpoints.py CPU_PYTHON → supervise.py replay_endpoints.py CPU_PYTHON → supervise.py render_figures.py CPU_PYTHON → supervise.py fix_c2st_layout.py CPU_PYTHON。入口的.used防止在原目录重复提交；复现应复制所需源、config及输入到新E/D目录并调整输出根，不删除原.used或覆盖旧结果。原绝对期限仅本次作业有效，复现需独立注册预算，不是直接解除旧作业守护。

config/PROTOCOL/PRETRAIN_FREEZE绑定训练源码和数据位置。R23 TRAIN_PRIVATE、VALIDATION_INPUTS、FINE_CONSTANTS、CRITIC_SCALER只读复用；原R23流、R26实际重排行载荷与状态由reorder_core读取并按原行绑定重建z/white/style。R21 step2000 G/reader与R18 INITIAL_CRITIC为共同起点。完整源波形继续作为网络输入，不退回目标域独立采样。

FRESH_FINE/TRANSLATOR.pt、STYLE_READER.pt、CRITIC.pt与OPTIMIZER_SELECTED.pt是2250所选同步状态；FINAL_*与OPTIMIZER_FINAL.pt为2950末点。FRESH_PROJECTION_STREAM.json保存2950次三矩阵摘要/范数界，FRESH_PROJECTION_FIRST_PRIVATE.npz、LAST及初末RNG状态用于原流核验；validation继承固定方向，fresh训练流不被验证消耗。

ACTUAL_GENERATION_INPUTS.npz、JOINT8_INPUTS_PRIVATE、PROBE_INPUTS_CORRECTED为保存的实际推理输入。FRESH_SELECTED/generated.npz与FRESH_FINAL/generated.npz保存各2000 pre/post ADC波形、两可加分量与source IDs。REPLAY_RECEIPT.json和replay_endpoints.py.NATIVE_RECEIPT.json证明一次独立CPU主生成重放4000条已执行、四数组及源ID一致；不重复完整评价。

旧R26四端点控制、R27实际投影空间/四块方向、原分类实际分组与预测按config路径只读复用；其中这些DEV方向已查看，不能作为新盲测。完整metrics/preds/父记录/权重留E/D；Git发布用METRICS_COMPACT与必要源码、报告、图、native摘要，不上传.private/逐事件载荷。最终交付清单由新stage字节SHA封存；不重审历史3GB源或做迁移一致性测试。
