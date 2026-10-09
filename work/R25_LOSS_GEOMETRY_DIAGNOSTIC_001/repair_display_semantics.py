import os,time,json,shutil
START=time.monotonic()
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostic_support import ROOT,write,log,check_output
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.dpi':120,'savefig.dpi':160})
b=json.loads((ROOT/'B_METRICS_COMPACT.json').read_text('utf8'));states=b['states'];terms=b['loss_names']
short=['adv','NCE','identity','Q rel','Q rank','PSD rank','domain SW','positive Q','fine','style L1'];labels=['initial (untrained D)','D1 final2950','D5 final2950'];colors=['#64748b','#2563eb','#ea580c'];d=ROOT/'figures'
for name in ['weighted_gradient_norms','negative_alignment_rates']:
    shutil.copy2(d/(name+'.png'),d/(name+'_display_initial.png'))
fig,ax=plt.subplots(figsize=(12,5))
for st,label,color in zip(states,labels,colors):
    med=np.array([st['terms'][t]['G_weighted_gradient_norm']['median'] for t in terms]);lo=np.array([st['terms'][t]['G_weighted_gradient_norm']['q05_q95'][0] for t in terms]);hi=np.array([st['terms'][t]['G_weighted_gradient_norm']['q05_q95'][1] for t in terms]);y=np.maximum(med,1e-8)
    ax.errorbar(np.arange(10),y,yerr=[y-np.maximum(lo,1e-8),np.maximum(hi,1e-8)-y],marker='o',capsize=3,label=label,color=color)
ax.set_yscale('log');ax.set_xticks(np.arange(10),short,rotation=25,ha='right');ax.set_ylabel('weighted G gradient L2 (median, 5–95% batch range)');ax.set_title('All zero norms / quantiles use display floor 1e-8; saved numerical values remain zero');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(d/'weighted_gradient_norms.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(12,5))
allzero=[]
for i,(st,label,color) in enumerate(zip(states,labels,colors)):
    y=np.asarray([st['terms'][t]['negative_dot_fraction_defined'] for t in terms],dtype='f8');pos=np.arange(10)+(i-1)*.25
    ax.bar(pos,y,width=.25,label=label,color=color)
    for j in np.flatnonzero(~np.isfinite(y)):ax.text(pos[j],.11,'n/a',color=color,rotation=90,ha='center',va='bottom',fontsize=9)
for j in range(10):
    if all(s['terms'][terms[j]]['negative_dot_fraction_defined'] is None for s in states):ax.text(j,.35,'zero norm',color='.35',ha='center',fontsize=8)
ax.set_xticks(np.arange(10),short,rotation=25,ha='right');ax.set_ylim(0,1.05);ax.set_ylabel('fraction of defined batches with g_term · g_total < 0');ax.set_title('Undefined zero-norm cases are n/a, not 0% negative alignment');ax.legend();fig.tight_layout();fig.savefig(d/'negative_alignment_rates.png',bbox_inches='tight');plt.close(fig)
rec={'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'updated':['weighted_gradient_norms.png','negative_alignment_rates.png'],'originals_preserved_suffix':'_display_initial.png','metric_changes':0,'new_fits_or_model_forward':0,'input':'Saved B_METRICS_COMPACT aggregate only'}
write('DISPLAY_REPAIR_RECEIPT.json',rec);check_output();log(rec)
