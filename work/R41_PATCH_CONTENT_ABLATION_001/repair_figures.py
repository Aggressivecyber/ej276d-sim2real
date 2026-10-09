"""Two display-only repairs from saved results; no fitted or generated values."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, time, hashlib, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/figures'
INITIAL=ROOT/'results/figures_initial'
START=time.monotonic()
A='NO_PATCH_NCE_FINAL'
NAMES=['source_preservation.png','training_no_patch_NCE.png']

def load(name):
    return json.loads((ROOT/name).read_text('utf8'))

def binding(p):
    b=p.read_bytes()
    return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

assert not (ROOT/'FIGURE_REPAIR_RECEIPT.json').exists()
INITIAL.mkdir(exist_ok=True)
old={}
for name in NAMES:
    assert not (INITIAL/name).exists()
    old[name]=binding(OUT/name)
    shutil.copyfile(OUT/name,INITIAL/name)

cfg=load('config.json');m=load('metrics.json');stats=load('TRAINING_COMPACT.json')
primary=m['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4']
colors={'BASE':'#888888','FULL_COVARIANCE_CALIBRATION':'#ce7d31',A:'#0867ad'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})

names=['BASE','FULL_COVARIANCE_CALIBRATION',A]
fig,axs=plt.subplots(1,4,figsize=(14.6,4.1));ticks=[];ticklabels=[]
for j,n in enumerate(names):
    for k,b in enumerate(['bin3','bin4']):
        row=primary[n][b];p=j*3+k;ticks.append(p);ticklabels.append(['BASE','COV','NO-NCE'][j]+str(k+3))
        axs[0].bar(p,row['output_PSD']['mean'],color=colors[n])
        axs[1].bar(p,row['output_PSD']['sample_sd'],color=colors[n])
        axs[2].bar(p,row['valley_fraction'],color=colors[n])
    r=m['source_association_actual_preADC_input'][n]['composition_and_PDG']['all']
    axs[3].bar(j-.15,r['source_Q_to_output_Q']['Spearman'],.28,color=colors[n],alpha=.45)
    axs[3].bar(j+.15,r['source_PSD_to_output_PSD']['Spearman'],.28,color=colors[n])
for ax,title in zip(axs[:3],['Pure-had PSD mean','Pure-had PSD sample SD','Frozen valley fraction']):
    ax.set_title(title);ax.set_xticks(ticks,ticklabels,rotation=45,fontsize=7);ax.grid(axis='y',alpha=.12)
axs[3].set(title='Actual source rank: light Q, dark PSD',ylim=(0,1.05))
axs[3].set_xticks(range(3),['BASE','COV','NO-NCE']);axs[3].grid(axis='y',alpha=.12)
fig.suptitle('Actual unADC source-Q bins3/4, N213/181; valley is not particle-error truth')
fig.tight_layout();fig.savefig(OUT/NAMES[0],bbox_inches='tight');plt.close(fig)

trace=[r for r in load(A+'/TRAINING_TRACE.json') if 'step'in r]
xx=[0]+[r['step']for r in trace]+[2950]
yy=[stats['step0_J20']]+[r['validation']['J']for r in trace]+[stats['final_J20']]
fig,axs=plt.subplots(2,2,figsize=(13,7.2))
axs[0,0].plot(xx,yy,'o-',ms=3,color=colors[A],label='fixed J20')
axs[0,0].axvline(stats['selected_step'],ls=':',color='gray',label=f"record-only selected{stats['selected_step']}")
axs[0,0].axvline(2950,ls='--',color=colors[A],label='sole DEV endpoint2950')
axs[0,0].set(title='Internal TRAIN validation including actual final',xlabel='update',ylabel='J20');axs[0,0].legend(fontsize=7)
for group in ['shape480','diff479','baseline100_V']:
    axs[0,1].plot([r['step']for r in trace],[r['fine_train_groups'][group]for r in trace],'o-',ms=2,label=group)
axs[0,1].set(title='Unchanged fresh-direction quantile losses',xlabel='update',ylabel='sampled training component');axs[0,1].legend(fontsize=7)
with np.load(ROOT/A/'GRADIENT_NORMS_PRIVATE.npz',allow_pickle=False)as z:
    gr=z['values'].copy()
for j,tag in enumerate(['G+reader','D']):
    axs[1,0].plot(np.arange(1,len(gr)+1),gr[:,j],lw=.5,alpha=.8,label=tag)
axs[1,0].axhline(cfg['gradient_clip'],ls=':',color='gray')
axs[1,0].set(title='Global preclip gradients; original clip5',xlabel='update',ylabel='L2 norm');axs[1,0].legend(fontsize=7)
for k in ['adversarial','identity','Q_relative','PSD_rank','domain_SW']:
    axs[1,1].plot([r['step']for r in trace],[cfg['loss_weights'][k]*r['components'][k]for r in trace],label=k,lw=1)
axs[1,1].set(title='Actual weighted components: NCE not evaluated',xlabel='update',ylabel='weighted loss');axs[1,1].legend(fontsize=6.5)
for ax in axs.flat:ax.grid(alpha=.12)
fig.text(.5,.006,'Three project heads: gradients None throughout; initial/final weights unchanged.',ha='center',va='bottom',fontsize=8)
fig.tight_layout(rect=(0,.035,1,1));fig.savefig(OUT/NAMES[1],bbox_inches='tight');plt.close(fig)

receipt={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
    'scope':'display only; no altered metrics or additional model fitting/generation',
    'preserved_initial_figures':'results/figures_initial','unchanged_executed_plot_source':'summarize_and_plot.py',
    'repairs':{'source_preservation.png':'source rank tick TIME corrected to NO-NCE',
        'training_no_patch_NCE.png':'project-head annotation moved to figure caption below axes'},
    'initial_figures':old,'final_figures':{n:binding(OUT/n)for n in NAMES},
    'HGB_EM_response_training_inference':0,'figure_values_from_saved_results_only':True,'visual_QA_pending':True}
(ROOT/'FIGURE_REPAIR_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n','utf8')
print(json.dumps(receipt),flush=True)
