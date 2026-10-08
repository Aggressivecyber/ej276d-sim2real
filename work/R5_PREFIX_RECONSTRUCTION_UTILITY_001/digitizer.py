"""共享数字化仪模型:1 GS/s、14-bit、双极性 [-0.5, +0.5] V。

真实和仿真波形都过这个数字化仪,保证两域一致的采样率与量化精度,
CycleGAN 才能学到"sim→real"的形状差异而非数字化差异。
"""
import numpy as np

# 数字化仪规格
SAMPLE_RATE_MHZ = 1000.0     # 1 GS/s(1 ns/样本)
BITS = 14
V_MIN, V_MAX = -0.5, 0.5     # 量程 1.0 V,双极性
LEVELS = 2 ** BITS - 1       # 16384 级,LSB ≈ 61 μV


def digitize(wf_v: np.ndarray) -> np.ndarray:
    """把电压波形(V)量化到 14-bit ADC 再还原成 V(带量化台阶)。"""
    wf_v = np.asarray(wf_v, dtype=np.float64)
    adc = np.round((wf_v - V_MIN) / (V_MAX - V_MIN) * LEVELS)
    adc = np.clip(adc, 0, LEVELS)                       # 防越界削顶
    return (adc / LEVELS) * (V_MAX - V_MIN) + V_MIN
