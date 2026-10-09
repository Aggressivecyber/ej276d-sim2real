from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:
    os.environ[key] = '4'
import pathlib, json, datetime, hashlib, time, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
start = time.monotonic()
m = json.loads((ROOT/'metrics.json').read_text('utf8'))
cfg = json.loads((ROOT/'config.json').read_text('utf8'))
out = ROOT/'results/figures'
old_png = out/'c2st.layout_initial.png'
old_receipt = ROOT/'PLOT_RECEIPT.before_layout.json'
if old_png.exists() or old_receipt.exists():
    raise RuntimeError('layout correction already attempted; preserve existing files')
shutil.copyfile(out/'c2st.png', old_png)
shutil.copyfile(ROOT/'PLOT_RECEIPT.json', old_receipt)
names = ['HOMO_SELECTED','HOMO_FINAL','FRESH_SELECTED','FRESH_FINAL']
colors = ['#377eb8','#78b5da','#d95f02','#f4a460']
labels = ['Fixed selected2750','Fixed final2950','Fresh selected2250','Fresh final2950']
views = [*cfg['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480']
display = ['shape493','full\nvoltage','prompt\nlocal','postpeak\nlocal','tail\nlocal','baseline\nlocal','Q / PSD\n2D','pure u480','seen\n480 points','complement\n480 points']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':145})
fig, ax = plt.subplots(figsize=(13.5,4.2))
for j, n in enumerate(names):
    rows = [m['C2ST']['results'][n][v] if n.startswith('FRESH') else m['C2ST']['paired_vs_corresponding_R26']['FRESH_SELECTED' if n=='HOMO_SELECTED' else 'FRESH_FINAL'][v]['old_control_result'] for v in views]
    x = np.arange(len(views)) + (j-1.5)*.19
    ba = np.array([r['balanced_accuracy'] for r in rows])
    ci = np.array([r['heldout_CI']['balanced_accuracy_95CI'] for r in rows])
    ax.bar(x, ba, .18, color=colors[j], label=labels[j])
    ax.errorbar(x, ba, yerr=np.maximum(0,np.vstack([ba-ci[:,0],ci[:,1]-ba])), fmt='none', color='black', lw=.65, capsize=2)
ax.axhline(.5,color='gray',ls=':')
ax.set(ylim=(.4,1.02),ylabel='holdout balanced accuracy',title='Fixed rows/groups; seven historical and three pure480 views')
ax.set_xticks(np.arange(len(views)),display)
ax.legend(loc='lower right',fontsize=8,ncol=2)
fig.tight_layout()
fig.savefig(out/'c2st.png',bbox_inches='tight')
plt.close(fig)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt = json.loads(old_receipt.read_text('utf8'))
payload = (out/'c2st.png').read_bytes()
receipt['figures']['c2st.png'] = {'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
receipt['layout_correction'] = {'PID':os.getpid(),'UTC':now,'elapsed_s':time.monotonic()-start,'original_png':'results/figures/c2st.layout_initial.png','original_receipt':'PLOT_RECEIPT.before_layout.json','change':'short display labels and wider axes only; saved metric values unchanged','new_fits':0,'new_generation':0}
(ROOT/'PLOT_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n','utf8')
(ROOT/'LAYOUT_CORRECTION_RECEIPT.json').write_text(json.dumps(receipt['layout_correction'],indent=2,allow_nan=False)+'\n','utf8')
print(json.dumps(receipt['layout_correction']),flush=True)
