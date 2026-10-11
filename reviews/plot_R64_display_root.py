"""Presentation-only repair: readable symmetric-log ticks, unchanged intervals."""
from pathlib import Path
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
import threading
import time

START = time.monotonic()
R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
cache = S/'cache/display_only'
cache.mkdir(parents=True, exist_ok=True)
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
for key in ('TMP', 'TEMP', 'TMPDIR', 'MPLCONFIGDIR'):
    os.environ[key] = str(cache)
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
handle = kernel.GetCurrentProcess()
assert kernel.SetProcessAffinityMask(handle, 1 << 2)


class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
        (key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
          'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]


getmem = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
getmem.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
resource = dict(PID=os.getpid(), CPU_affinity=[2], RSS_limit_bytes=512*1024**2, actual_peak_RSS_bytes=0)
done = threading.Event()


def probe():
    p = PMC()
    p.cb = ctypes.sizeof(p)
    assert getmem(handle, ctypes.byref(p), p.cb)
    resource['actual_peak_RSS_bytes'] = max(resource['actual_peak_RSS_bytes'], p.PeakWorkingSetSize)
    if p.WorkingSetSize > resource['RSS_limit_bytes']:
        os._exit(91)


def guard():
    while not done.wait(.1):
        try:
            probe()
        except BaseException:
            os._exit(92)


probe()
watcher = threading.Thread(target=guard, daemon=True)
watcher.start()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source_names = ['PARAMETER_GRADIENT_SUMMARY.json', 'ENDPOINT_GRADIENT_SUMMARY.json',
                'RESULT.json', 'render_diagnostic.py', 'figures/03_paired_error_intervals.png']
before = {name: sha(S/name) for name in source_names}
summaries = [json.loads((S/name).read_text('utf8')) for name in source_names[:2]]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 140, 'savefig.dpi': 170})
palette = ['#3066be', '#eb8b23', '#159b88', '#b64d6c', '#7255a6', '#2b2e34']
labels = ['Full window', 'PSD', 'Noise', 'Coord. RMS', 'Temporal', 'Total D']
fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.7))
display = []
for col, (stat, coordinate) in enumerate(zip(summaries, ['Parameter', 'Parent endpoint'])):
    ax = axes[col]
    means, low, high = [], [], []
    for name in stat['reported_order']:
        item = stat['paired'][name]
        means.append(item['mean_B_minus_A_squared_l2_error'])
        low.append(item['bootstrap_percentile_95_interval'][0])
        high.append(item['bootstrap_percentile_95_interval'][1])
    for j in range(6):
        ax.plot([low[j], high[j]], [j, j], color=palette[j], linewidth=3)
        ax.plot(means[j], j, 'o', color=palette[j], markersize=6)
    ax.axvline(0, color='#64748b', linewidth=1)
    nonzero = np.abs(np.r_[low, high, means])
    nonzero = nonzero[nonzero > 0]
    threshold = float(nonzero.min()) if len(nonzero) else 1.
    ax.set_xscale('symlog', linthresh=threshold)
    limits = (-1., 500.) if col == 0 else (-10., 500.)
    ticks = [-1., -.1, 0., .1, 1., 10., 100.] if col == 0 else [-10., -1., -.1, 0., .1, 1., 10., 100.]
    assert min(low) > limits[0] and max(high) < limits[1]
    ax.set_xticks(ticks, [format(value, 'g') for value in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(*limits)
    ax.set(title=coordinate+' gradient error: B minus A', yticks=np.arange(6), yticklabels=labels,
           xlabel='Squared Euclidean error change (symmetric log scale)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    display.append(dict(coordinate=coordinate, xlim=limits, ticks=ticks, original_linthresh_retained=threshold,
                        plotted_means=means, plotted_CI_low=low, plotted_CI_high=high))
fig.suptitle('R64 | Paired macro bootstrap, fixed 1,000 resamples', fontsize=16, fontweight='bold', y=.985)
fig.text(.5, .022, 'Total D parameter error is primary; components are descriptive. Negative intervals do not establish better training or A-F performance.',
         ha='center', va='bottom', fontsize=9, color='#465267')
fig.tight_layout(rect=[0, .075, 1, .94], h_pad=2.1, w_pad=2.)
path = S/'figures/03_paired_error_intervals.display.png'
assert not path.exists()
fig.savefig(path, facecolor='white')
plt.close(fig)
assert before == {name: sha(S/name) for name in source_names}
done.set()
watcher.join()
probe()
assert resource['actual_peak_RSS_bytes'] <= resource['RSS_limit_bytes']
receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), display_only=True,
    reason='Original parameter-panel symmetric-log labels near zero overlapped on actual opening.',
    changed='Explicit, spaced major tick labels, no minor ticks, and axis bounds retaining every interval.',
    original_files_unchanged=before, plotted_values_and_intervals=display, output=path.relative_to(S).as_posix(),
    output_sha256=sha(path), science_calls=0, new_fits=0, resource=resource, elapsed_s=time.monotonic()-START,
    root_visual_reopen_required=True)
(S/'DISPLAY_ONLY_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', 'utf8')
print(json.dumps(receipt, indent=2))
