from __future__ import annotations
import numpy as np
from digitizer import digitize
from noise_spectrum import spectrum_noise
TAU_EE = [13.0, 35.0, 270.0]

FRAC_EE = [0.70, 0.15, 0.15]

TAU_HAD = [13.0, 59.0, 460.0]

FRAC_HAD = [0.50, 0.10, 0.40]

RISE_NS = 5.0

YIELD = 8600.0

QUENCH = 0.15

CAL_A = 1.32506e-4

CAL_B = -0.03918

def pulse_rate(n_ee, n_had, t_ns, onset_ns):
    """光子率(ph/ns),带上升沿,onset 前为 0。双指数 (e^(−t/τd)−e^(−t/τr))/(τd−τr)。"""
    dt = np.maximum(t_ns - onset_ns, 0.0)
    mask = (t_ns - onset_ns) > 0.0
    r = np.zeros_like(t_ns)
    for tau, f in zip(TAU_EE, FRAC_EE):
        r += n_ee * f * (np.exp(-dt / tau) - np.exp(-dt / RISE_NS)) / (tau - RISE_NS)
    for tau, f in zip(TAU_HAD, FRAC_HAD):
        r += n_had * f * (np.exp(-dt / tau) - np.exp(-dt / RISE_NS)) / (tau - RISE_NS)
    return r * mask

def _normalize_waveform(wf: np.ndarray) -> np.ndarray:
    """基线扣除 + 峰值归一化到 ~[0, 1]。"""
    wf = np.asarray(wf, dtype=np.float32)
    base = wf[: max(1, len(wf) // 5)]  # 前 20% 当基线
    wf = wf - float(np.mean(base))     # 扣基线
    peak = float(np.max(wf))
    if peak > 0:
        wf = wf / peak                 # 峰值缩放到 1
    return wf.astype(np.float32)

def _window_waveform(wf: np.ndarray, tp0: int,
                     seq_len: int, trigger_index: int) -> np.ndarray:
    """以 tp0 为基准、trigger_index 为窗口内触发位置，切出长 seq_len 的窗口。

    越界部分用边缘值填充（mode='edge'），保证输出长度恒为 seq_len。
    """
    if not (0 <= trigger_index < seq_len):
        raise ValueError(f"trigger_index {trigger_index} out of [0, {seq_len})")
    left = int(tp0) - trigger_index    # 窗口左端在原波形里的下标
    right = left + seq_len
    src_left = max(0, left)
    src_right = min(wf.size, right)
    seg = wf[src_left:src_right]
    if seg.size == 0:
        raise ValueError("cannot window an empty waveform segment")
    pad_left = max(0, -left)           # 左侧不够则前补
    pad_right = max(0, right - wf.size)  # 右侧不够则后补
    if pad_left or pad_right:
        seg = np.pad(seg, (pad_left, pad_right), mode="edge")
    if seg.size < seq_len:             # 仍不足则尾补到 seq_len
        seg = np.pad(seg, (0, seq_len - seg.size), mode="edge")
    elif seg.size > seq_len:           # 过长则截断
        seg = seg[:seq_len]
    return np.ascontiguousarray(seg, dtype=np.float32)

def constant_fraction_tp0(wf: np.ndarray, fraction: float = 0.5) -> int:
    """触发下标：主峰相连上升支穿越峰值 fraction 处的样本。

    恒定分数定时（CFD）对 PMT/SiPM 闪烁脉冲比前沿阈值更稳健
    （消除了幅度依赖的走步）。
    """
    wf = np.asarray(wf, dtype=np.float64)
    if wf.size == 0:
        raise ValueError("empty waveform")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    peak = int(np.argmax(wf))
    peak_value = float(wf[peak])
    if not np.isfinite(peak_value) or peak_value <= 0:
        return peak
    thr = fraction * peak_value                  # 阈值 = fraction × 峰值
    # The final below-threshold sample locates the branch connected to the main
    # peak. A preceding isolated spike must not become the pulse's time origin.
    below = np.flatnonzero(wf[:peak] < thr)
    if below.size == 0:
        return peak
    return int(below[-1] + 1)
