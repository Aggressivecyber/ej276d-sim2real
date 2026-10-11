# R71 共同末点与独立 NumPy 参考审核

结论：经根修复后的受审版本没有发现当前路径的阻塞性错误。证据仅为静态推导与人工 native CPU 核对；未读科学数组/权重，未运行 CUDA、训练、完整波形网络或科学探针。绑定版本及实际资源见 `COMMON_RESPONSE_ARTIFICIAL_REVIEW.json` 的 sources/resources。

`common_response.py:14–23` 的谱归一正确。标准高斯经两次 ortho 实 FFT、乘 `sqrt(p)` 后，480 点循环平稳协方差为 `K_ij=irfft(p,480)[(i-j)%480]`；不需再乘 480，也没有漏计内部频率的 2 倍因子。令 `a=(-3×100,1×300,0×80)`，`P=I-m aᵀ`，`B` 为前 100 点选择后去均值，则目标严格是 `trace(B P K Pᵀ Bᵀ)/99`。代码的 base、交叉项 `-2(Bm)ᵀ(BKa)/99` 和二次项 `(aᵀKa)||Bm||²/99` 与此一致，分母 99 对应 unbiased 方差。四个人工谱的确定性 FFT 基向量协方差误差最高 `1.33e-15`，解析与稠密投影方差误差最高 `1.11e-15`。

`common_response.py:29–46` 保持独立预测 Q*→连续模板→lifting→同一零 Q 噪声投影→Box(raw,Q*)，没有脱离 Q* 的旧 Q 目标。连续 Q*→kappa 的自动微分与中央差分最大误差 `8.93e-13`。本轮噪声不参加小头优化；上述图核对不改变这一科学协议。

两个真实问题已由根修复并核实：原 `sqrt(amp2/kappa2)` 在 amp2=0 时能出现 `0·inf` 的 NaN 反向；当前 `common_response.py:40–43` 先将 inactive sqrt 输入换成 1 再置输出 0，前向与原式逐位相同、inactive 反向明确为 0，实际零幅度共同路径的头参数梯度全有限。原 NumPy 参考可接受 `exp(b)==0`，也可因 NaN epsilon 返回 NaN pre/ADC；当前参考输入/斜率/输出检查已拒绝这两个人工失败例。两者在已登记 no_grad 末点与有效状态下不改变数字。

`numpy_reference.py` 不导入 Torch，独立使用 NumPy MLP、稳定 sigmoid、模板插值、稠密 K 与显式 B，Box 自行执行 64 次二分，最后匹配原 float32 legacy ADC→float64 canonical codes。人工 tau2=0、0.002 两种场景中 Q* 一致至 `1e-12`，pre-ADC 和 ADC 数组逐位一致。此核对覆盖有效人工状态，不能替代科学人口的独立数值检查或 A–F 验收；ADC 离散边界附近的浮点误差仍应按原协议评估。

18 项人工检查通过，native exit 0；PID 68112、affinity mask 2、Torch intra/inter 均 1、硬 RSS 上限 1 GiB，实测峰值 RSS 230,912,000 B、峰值 private commit 268,963,840 B，耗时 5.083 秒。`float(error.max())` 的 requires-grad 警告仅来自人工图核对中的诊断标量转换，不改变保留的输出计算图；实际 generate 已为 no_grad。
