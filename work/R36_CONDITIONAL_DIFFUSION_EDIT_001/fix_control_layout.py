from __future__ import annotations
import pathlib,json,time,os,hashlib,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();OUT=ROOT/'results/figures'
m=json.loads((ROOT/'metrics.json').read_text('utf8'));A='CONDITIONAL_DIFFUSION_EDIT';R='REAL_GUIDE_EDIT'
labels={'REAL':'REAL DEV','TRAIN_REFERENCE':'Raw REAL TRAIN ref',R:'REAL guide edit'}
colors={'REAL':'#111111','TRAIN_REFERENCE':'#398347',R:'#e67820',A:'#1565c0'}
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
raw=load(ROOT/'results/RAW_VOLTAGE_MOMENTS_PRIVATE.npz');features=load(ROOT/'results/FEATURES_PRIVATE.npz');gp=load(ROOT/'results/GUIDE_PERTURBATION_PRIVATE.npz');tt=np.arange(480)-120
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
fig,axs=plt.subplots(3,3,figsize=(13.8,10.2))
for j,(key,title)in enumerate([('mean_V','population mean'),('SD_V','population pointwise SD')]):
    for n in ['REAL','TRAIN_REFERENCE',R]:axs[0,j].plot(tt,raw[n+'__None__'+key]*1000,color=colors[n],ls='--'if n=='TRAIN_REFERENCE'else'-',label=labels[n],lw=1.2)
    axs[0,j].set(title='REAL guide control: '+title,xlabel='relative ns',ylabel='mV');axs[0,j].grid(alpha=.13)
for n in ['REAL','TRAIN_REFERENCE',R]:
    p=features[n+'__PSD'][features[n+'__valid']];x=np.sort(p[np.isfinite(p)]);axs[0,2].plot(x,np.arange(1,len(x)+1)/len(x),color=colors[n],ls='--'if n=='TRAIN_REFERENCE'else'-',lw=1.2)
axs[0,2].set(title='REAL guide control: full PSD',xlabel='signed PSD',ylabel='CDF');axs[0,0].legend(fontsize=7)
dv=gp['perturbed_output_preADC_V'].astype('f8')-gp['original_output_preADC_V'].astype('f8');dd=gp['actual_perturbed_guide_f32_V'].astype('f8')-gp['original_guide_V'].astype('f8')
axs[1,0].plot(tt,dd[0]*1000,color='#555',label='actual guide delta');axs[1,0].plot(tt,dv[0]*1000,color=colors[A],label='paired output delta');axs[1,0].set(title='First prespecified guide perturbation',xlabel='relative ns',ylabel='delta mV');axs[1,0].legend(fontsize=7)
rs=np.sqrt(np.mean(dv*dv,1));axs[1,1].hist(rs*1000,bins=18,color=colors[A]);axs[1,1].axvline(1000/16383,color='black',ls='--',label='1 ADC LSB');axs[1,1].set(title='Same c/bin/record noise, guide-only change',xlabel='preADC paired RMS (mV)',ylabel='parents');axs[1,1].legend(fontsize=7)
rows=m['source_probes'][A]['rows'];tags=list(rows)
short={'amplitude_x0.5':'amp\n×0.5','amplitude_x1.5':'amp\n×1.5','fixed_source_changed_noise':'fixed\nsource\nnew noise','same_input_same_noise':'same\nsource\nnoise','source_roll_no_fixed_points':'rolled\nsource'}
short.update({f'analytic_fast_slow_{v}':'fast\n'+v for v in ['0','0.25','0.5','0.75','1']})
axs[1,2].bar(range(len(tags)),[rows[k]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for k in tags],color=colors[A]);axs[1,2].set(title='Full source-chain probes',ylabel='RMS to correct reference (mV)');axs[1,2].set_xticks(range(len(tags)),[short[t]for t in tags],fontsize=7)
terms=['base_signal','base_electronic','base_sum_rounding','flow_increment_f64'];joint=m['random_decomposition'][A]['groups']['joint8']['by_composition']
for j,cat in enumerate(['pure_EE','pure_had','mixed']):
    r=joint[cat];cov=r['four_additive_algebraic_components_covariance_V2']['full480'];vals=[cov[t+'__'+t]['mean']for t in terms]+[2*sum(cov[terms[k]+'__'+terms[l]]['mean']for k in range(4)for l in range(k+1,4))];axs[2,j].bar(range(5),np.asarray(vals)*1e6,color=[colors[A]]*4+['#555']);axs[2,j].set(title=cat+f': joint8 ({r["parents"]} parents)',ylabel='variance contribution (mV²)');axs[2,j].set_xticks(range(5),['baseS','baseE','round','edit','all2cov'],fontsize=7);axs[2,j].axhline(0,color='gray',lw=.7)
fig.suptitle('REAL control, guide dependence and additive algebra; physical edit/noise split unknown');fig.tight_layout();path=OUT/'real_control_and_guide.png';fig.savefig(path,bbox_inches='tight');plt.close(fig)
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'new_fit_generation':0,'change':'shortened probe tick labels only; original preserved in layout_initial',
    'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(ROOT/'CONTROL_LAYOUT_REPAIR_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(json.dumps(receipt),flush=True)
