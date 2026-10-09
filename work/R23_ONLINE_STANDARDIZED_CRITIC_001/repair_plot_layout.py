from __future__ import annotations
import os,time,json,pathlib,datetime
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent; OUT=ROOT/'results/figures'; START=time.monotonic()
M=json.loads((ROOT/'metrics.json').read_text('utf8'));COL={'R21_GLOBAL_STYLE':'#9467bd','RAW':'#1f77b4','STANDARDIZED':'#e66101'}
def save(fig,name):fig.savefig(OUT/name,dpi=150,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
tags=list(M['source_probes']['RAW']['rows']);xx=np.arange(len(tags))
for off,n in [(-.18,'RAW'),(.18,'STANDARDIZED')]:
    rows=M['source_probes'][n]['rows'];axs[0,0].barh(xx+off,[rows[t]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for t in tags],height=.36,color=COL[n],label=n)
axs[0,0].set_yticks(xx,[t.replace('_',' ')for t in tags],fontsize=8);axs[0,0].set_xlabel('RMS against correct source/noise reference [mV]');axs[0,0].legend(fontsize=8)
for n in ['RAW','STANDARDIZED']:
    r=M['random_decomposition'][n]['groups']['joint8']['rows']
    for kind,ls in [('pure_EE','-'),('pure_had','--'),('mixed',':')]:
        a=[v['output_PSD']['sample_sd']for v in r if v['composition']==kind and v['output_PSD']['sample_sd']is not None];a=np.sort(a)
        if len(a):axs[0,1].step(a,np.arange(1,len(a)+1)/len(a),c=COL[n],ls=ls,label=n+' '+kind)
axs[0,1].set_xlabel('same-parent PSD SD, original joint8');axs[0,1].set_ylabel('CDF of parents');axs[0,1].legend(fontsize=7)
xx=np.arange(3)
for off,n in [(-.18,'RAW'),(.18,'STANDARDIZED')]:
    de=M['random_decomposition'][n]['groups']['joint8']['by_composition'];kinds=['pure_EE','pure_had','mixed']
    val=[de[k]['within_parent_waveform_RMS_V']['signal_pre_ADC']['full480']['mean']*1000 if k in de and de[k]['within_parent_waveform_RMS_V']['signal_pre_ADC']['full480']['mean']is not None else np.nan for k in kinds];axs[1,0].bar(xx+off,val,width=.36,color=COL[n],label=n)
axs[1,0].set_xticks(xx,kinds);axs[1,0].set_ylabel('signal branch parent RMS SD [mV]');axs[1,0].set_title('Functional signal is not conditional expectation')
for n in ['R21_GLOBAL_STYLE','RAW','STANDARDIZED']:
    p=M['source_composition']['primary_pure_had_source_bins3_4'][n];axs[1,1].plot([3,4],[p['bin3']['output_PSD']['mean'],p['bin4']['output_PSD']['mean']],'o-',c=COL[n],label=n)
    for b in [3,4]:
        frac=p['bin'+str(b)]['valley_fraction'];mean=p['bin'+str(b)]['output_PSD']['mean']
        if mean is not None:axs[1,1].annotate('valley '+(format(frac,'.1%')if frac is not None else'undefined'),(b,mean),xytext=(4,8 if n=='RAW'else -12),textcoords='offset points',fontsize=7,c=COL[n])
centers=[M['source_composition']['primary_pure_had_source_bins3_4'][n]['bin'+str(b)]['output_PSD']['mean']for n in ['R21_GLOBAL_STYLE','RAW','STANDARDIZED']for b in [3,4]]
axs[1,1].set_xlim(2.85,4.18);axs[1,1].set_ylim(min(centers)-.003,max(centers)+.003);axs[1,1].set_xticks([3,4]);axs[1,1].set_xlabel('prespecified pure-had source bins; N=207/182');axs[1,1].set_ylabel('mean output PSD');axs[1,1].legend(fontsize=7);save(fig,'source_controls_random.png')


r={'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'saved_metrics_only':True,'fit_or_generation':0,'reason':'margin for purehad valley annotations'}
(ROOT/'PLOT_LAYOUT_REPAIR_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n','utf8');print(json.dumps(r),flush=True)
