"""Two display-only repairs from the existing compact statistics."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'results' / 'figures'
OLD = FIG / 'display_initial'
OLD.mkdir(exist_ok=True)
START = time.monotonic()
names = ['charge_only_C2ST.png', 'noise_by_charge.png']
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
for name in names:
    if (OLD / name).exists():
        raise RuntimeError('display repair already attempted; preserve originals')
    shutil.copyfile(FIG / name, OLD / name)
shutil.copyfile(ROOT / 'render_figures.py', ROOT / 'render_figures_display_initial.py')
compact = ROOT / 'METRICS_COMPACT.json'
compact_sha = digest(compact)
m = json.loads(compact.read_text('utf-8-sig'))
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
keys = ['CHARGE_SOURCE', 'CHARGE_LONG', 'CHARGE_RAW_REFERENCE']
labels = ['SOURCE input', 'LONG output', 'RAW reference']
for k, (metric, ci_key) in enumerate([
    ('balanced_accuracy', 'balanced_accuracy_95CI'), ('AUC', 'AUC_95CI')]):
    ax = axes[k]
    xx = np.arange(3)
    point = np.array([m['classifiers'][s][metric] for s in keys])
    ci = np.array([m['classifiers'][s]['heldout_CI'][ci_key] for s in keys])
    trainkey = 'training_balanced_accuracy' if k == 0 else 'training_AUC'
    train = [m['classifiers'][s][trainkey] for s in keys]
    ax.errorbar(xx, point,
        yerr=np.maximum(0, np.vstack([point-ci[:, 0], ci[:, 1]-point])),
        fmt='o', color='#266ca8', capsize=5, label='Holdout + 95% group CI')
    ax.plot(xx, train, 'x', color='#ab5d28', label='TRAIN score')
    ax.axhline(.5, color='gray', ls='--')
    ax.set_ylim(.38, .79)
    if k == 0:
        ax.axhspan(.4, .6, color='gray', alpha=.08)
        ax.axhline(.6, color='gray', ls=':')
    ax.set_xticks(xx, labels)
    ax.set_ylabel('Balanced accuracy' if k == 0 else 'AUC')
    ax.legend(loc='upper left' if k == 0 else 'lower left', fontsize=8)
    if k == 0:
        for i, s in enumerate(keys):
            text = ('uncertain' if m['classifiers'][s]['population_diagnostic']
                    == 'uncertain' else 'not detected')
            ax.text(i, .422, text, ha='center', fontsize=9)
fig.suptitle('Fixed charge-only diagnostic: 600 TRAIN + 250 holdout per domain\n'
             'CI crossing 0.60 is uncertain; not a Q-PSD evaluator lower bound')
fig.tight_layout()
fig.savefig(FIG / names[0], dpi=170, bbox_inches='tight')
plt.close(fig)

colors = {'RAW': '#30343b', 'INITIAL_POSTADC': '#c75a10',
          'SHORT': '#2878b5', 'LONG': '#8a4ca5', 'BASE': '#258570'}
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
for row, label in enumerate(['SOURCE', 'REAL_REFERENCE']):
    for step in ['RAW', 'INITIAL_POSTADC', 'SHORT', 'LONG', 'BASE']:
        if step not in m['baseline_paths'][label]:
            continue
        rows = m['baseline_paths'][label][step]['own_observed_Q_bins']
        for col, key in enumerate(['baseline_RMS_V', 'ACF1']):
            values = [r['fields'][key]['distance_vs_REAL_same_observed_Q']['W1']
                      for r in rows]
            v = np.array([np.nan if q is None else q for q in values])
            axes[row, col].plot(np.arange(1, 9), v*(1000 if col == 0 else 1),
                'o-', color=colors[step], label=step.replace('_', ' '), ms=4)
    for col in range(2):
        axes[row, col].set_title(label.replace('_', ' '))
        axes[row, col].set_xlabel('Own observed Q-bin')
        axes[row, col].set_ylabel('RMS W1 versus REAL [mV]' if col == 0
                                  else 'ACF1 W1 versus REAL')
        axes[row, col].set_xticks(np.arange(1, 9))
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .045),
           ncol=5, fontsize=10, frameon=False)
fig.text(.5, .015, 'SOURCE RAW ACF1 is undefined: zero baseline variance; '
         'its RMS curve remains shown.', ha='center', fontsize=10)
fig.suptitle('Fixed eight-bin noise descriptions: failed / undefined support retained\n'
             'Own-domain charge bins are not matched populations or causal interventions')
fig.tight_layout(rect=[0, .095, 1, .97])
fig.savefig(FIG / names[1], dpi=170, bbox_inches='tight')
plt.close(fig)
if digest(compact) != compact_sha:
    raise AssertionError('saved statistics changed during display-only repair')
receipt = {'PID': os.getpid(), 'UTC': datetime.now(timezone.utc).isoformat(),
           'elapsed_body_s': time.monotonic()-START, 'statistics_unchanged': True,
           'compact_sha256': compact_sha, 'new_HGB_EM_training_ODE_generation': 0,
           'original_script': 'render_figures_display_initial.py',
           'repairs': [{'figure': name, 'initial_sha256': digest(OLD/name),
                        'final_sha256': digest(FIG/name)} for name in names],
           'changes': ['AUC legend to lower-left blank region; uncertainty labels raised',
                       'noise shared legend outside axes; SOURCE RAW undefined ACF1 caption']}
(ROOT / 'DISPLAY_REPAIR_RECEIPT.json').write_bytes(
    (json.dumps(receipt, indent=2, allow_nan=False)+'\n').encode('utf-8'))
print(json.dumps(receipt), flush=True)
