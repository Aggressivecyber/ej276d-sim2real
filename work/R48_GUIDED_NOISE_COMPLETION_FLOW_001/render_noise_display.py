"""Display-only correction from already saved TRAIN numerical-check arrays."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, time, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
start = time.monotonic()
cfg = json.loads((root / 'config.json').read_text('utf8'))
out = root / 'results/figures'
original = out / 'TRAIN_initial_noise_rule.png'
preserved = out / 'TRAIN_initial_noise_rule.display_initial.png'
assert original.exists() and not preserved.exists()
shutil.copyfile(original, preserved)
colors = ['#2474ad', '#d45d1c']
labels = ['FIXED noise control', 'NOISE completion primary']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 180})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
for ax, arm, col, label in zip(axs, cfg['arms'], colors, labels):
    with np.load(root / arm / 'TRAIN_SOLVER_CHECK_PRIVATE.npz', allow_pickle=False) as z:
        v = z['input_variance_V2']
        a = z['innovation_amplitude_squared_V2']
    for domain, sl, marker in [('SOURCE', slice(0, 64), 'o'),
                               ('REAL', slice(64, 128), 'x')]:
        ax.scatter(np.sqrt(v[sl]) * 1000, np.sqrt(a[sl]) * 1000,
                   s=25, marker=marker, color=col, label=domain, alpha=.7)
    ax.set(xlabel='Input baseline sample SD (mV)',
           ylabel='Innovation amplitude SD (mV)', title=label, ylim=(0, .60))
    ax.legend(fontsize=8)
    ax.grid(alpha=.15)
fig.suptitle('Same fixed128 TRAIN rows; amplitude is not realized per-event noise SD')
fig.tight_layout()
fig.savefig(original, bbox_inches='tight')
plt.close(fig)
receipt = {'PID': os.getpid(), 'UTC': datetime.now(timezone.utc).isoformat(),
           'elapsed_s': time.monotonic() - start, 'display_only': True,
           'input': 'saved TRAIN_SOLVER_CHECK_PRIVATE per arm',
           'preserved_original': str(preserved.relative_to(root)),
           'final_figure': str(original.relative_to(root)),
           'common_y_limits_mV': [0, .60], 'common_x_axis': True,
           'new_neural_inference': 0, 'HGB_EM_updates': 0, 'DEV_read_or_generated': 0}
(root / 'DISPLAY_CORRECTION_RECEIPT.json').write_bytes(
    (json.dumps(receipt, indent=2) + '\n').encode('utf8'))
print(json.dumps(receipt), flush=True)
