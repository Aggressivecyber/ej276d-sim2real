from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='4'
import pathlib,json,datetime,time
START=time.monotonic()
ROOT=pathlib.Path(__file__).resolve().parent
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def stats(a):
    a=np.asarray(a,dtype=np.float64)
    return {'N':int(len(a)),'mean':float(a.mean()),'median':float(np.median(a)),
            'p05':float(np.quantile(a,.05)),'p95':float(np.quantile(a,.95)),
            'min':float(a.min()),'max':float(a.max())} if len(a) else {'N':0,'undefined':True}

windows={'full480':[0,480],'baseline100':[0,100],'prompt100_150':[100,150],
         'postpeak150_220':[150,220],'tail220_400':[220,400]}
out={'definition':'For each of 128 parents, unbiased sample variance over its 8 saved post-ADC waveforms at each sample (denominator 7); sqrt of variance averaged over each stated window, in mV.',
     'parent_count_per_arm':128,'noise_draws_per_parent':8,'windows_halfopen':windows,
     'new_generation':0,'model_inference':0,'new_fit':0,'arms':{}}
for arm in ['CONT','FINE']:
    with np.load(ROOT/'results'/(arm+'_REPEAT_NOISE_PRIVATE.npz'),allow_pickle=False) as a:
        y=np.asarray(a['waveforms_V'],dtype=np.float64).reshape(128,8,480)
        parent=np.asarray(a['source_parent_index']).reshape(128,8)
        comp=np.asarray(a['parent_composition'])
        assert np.all(parent==parent[:,:1]) and len(np.unique(parent[:,0]))==128
        assert comp.shape==(128,) and np.isfinite(y).all()
        variance=np.var(y,axis=1,ddof=1)
    row={'whole_population':{},'by_composition':{}}
    for name,(left,right) in windows.items():
        v=np.sqrt(np.mean(variance[:,left:right],axis=1))*1000
        row['whole_population'][name]=stats(v)
        for c in ['pure_EE','pure_had','mixed']:
            row['by_composition'].setdefault(c,{})[name]=stats(v[comp==c])
    out['arms'][arm]=row
out['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
out['PID']=os.getpid()
(ROOT/'REPEAT_WAVEFORM_SUMMARY.json').write_text(json.dumps(out,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')

m=json.loads((ROOT/'metrics.json').read_text('utf8'))
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
fields=[('J','J20 = J19 + fine'),('J19_weighted','J19 weighted'),('fine_val','fine validation mean')]
for ax,(key,title) in zip(axes[0],fields):
    for arm,color in [('CONT','tab:orange'),('FINE','tab:red')]:
        t=m['training'][arm];trace=[r['validation'] for r in t['validation_trace']]
        ax.plot([v['step'] for v in trace],[v[key] for v in trace],label=arm,color=color)
        selected=t['selected_validation'];ax.scatter([selected['step']],[selected[key]],color=color,marker='*',s=110,zorder=5)
    ax.set_title(title);ax.set_xlabel('G/D updates');ax.set_ylabel('lower is better');ax.grid(alpha=.2);ax.legend()
for ax,key in zip(axes[1],['shape480','diff479','baseline100_V']):
    for arm,color in [('CONT','tab:orange'),('FINE','tab:red')]:
        t=m['training'][arm];trace=[r['validation'] for r in t['validation_trace']]
        ax.plot([v['step'] for v in trace],[v['fine_groups'][key] for v in trace],label=arm,color=color)
        selected=t['selected_validation'];ax.scatter([selected['step']],[selected['fine_groups'][key]],color=color,marker='*',s=110,zorder=5)
    ax.set_title(key+' projected SW');ax.set_xlabel('G/D updates');ax.set_ylabel('TRAIN-standardized units');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Frozen TRAIN constants; stars mark selection by common J20, not DEV')
fig.savefig(ROOT/'results/figures/fine_loss_trajectory.png',dpi=160)
plt.close(fig)
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'elapsed_s':time.monotonic()-START,'source':'existing R20 result copies only',
                  'new_fit':0,'new_generation':0,'saved_waveform_values_used':2*1024*480,
                  'summary':['REPEAT_WAVEFORM_SUMMARY.json','results/figures/fine_loss_trajectory.png']}),flush=True)
