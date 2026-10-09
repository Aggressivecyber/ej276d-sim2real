from __future__ import annotations
import pathlib,json,time,datetime,hashlib,os,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/figures';START=time.monotonic()
A='SLICED_MONOTONE_TRANSPORT';colors={A:'#0867ad'}
m=json.loads((ROOT/'metrics.json').read_text('utf8'));stats=m['training'];tt=np.arange(480)-120
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
prior=OUT/'layout_initial';prior.mkdir(exist_ok=True)
for n in ['training_sliced_maps.png','guide_and_random_decomposition.png']:
    assert not(prior/n).exists();shutil.copyfile(OUT/n,prior/n)
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def save(fig,n):fig.savefig(OUT/n,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(13,7))
for k in range(8):
    rr=[r for r in stats['layer_summaries']if r['bin_id_1based']==k+1]
    x=np.arange(1,33);cc=plt.get_cmap('tab10')(k)
    axs[0,0].semilogy(x,[r['J_initial']for r in rr],lw=.8,color=cc,ls=':',alpha=.7)
    axs[0,0].semilogy(x,[r['J_final']for r in rr],lw=.9,color=cc,label=f'bin{k+1}')
    axs[0,1].semilogy(x,[r['full_projected_W2_squared_before']for r in rr],ls=':',lw=.8,color=cc,alpha=.7)
    axs[0,1].semilogy(x,[r['full_projected_W2_squared_after']for r in rr],lw=.9,color=cc)
    axs[1,0].semilogy(x,[r['minimum_x_knot_difference']for r in rr],lw=.8,color=cc,ls=':')
    axs[1,0].semilogy(x,[r['minimum_y_knot_difference']for r in rr],lw=.9,color=cc)
    axs[1,1].semilogy(x,[r['segment_slope_max']for r in rr],lw=.9,color=cc)
    axs[1,1].semilogy(x,[r['segment_slope_min']for r in rr],ls=':',lw=.8,color=cc)
axs[0,0].set(title='20-step direction ascent: dotted initial, solid final',ylabel='subset squared projected W2')
axs[0,0].legend(ncol=4,fontsize=7)
axs[0,1].set(title='Selected directions: dotted before, solid after map',ylabel='full-bin squared projected W2')
axs[1,0].set(title='Strict knot increments: dotted X, solid Y',ylabel='minimum adjacent knot difference')
axs[1,1].set(title='Scalar slopes: dotted minimum, solid maximum',ylabel='segment slope')
for ax in axs.flat:ax.set_xlabel('layer (1..32)');ax.grid(alpha=.12)
fig.suptitle('4096 scalar maps; marginal closure is not complete distribution matching')
fig.tight_layout();save(fig,'training_sliced_maps.png')

gp=load(ROOT/'results/GUIDE_PERTURBATION_PRIVATE.npz');dv=gp['perturbed_output_preADC_V'].astype('f8')-gp['original_output_preADC_V'].astype('f8');dd=gp['actual_perturbed_guide_f32_V'].astype('f8')-gp['original_guide_V'].astype('f8')
fig,axs=plt.subplots(2,3,figsize=(13.8,7))
axs[0,0].plot(tt,dd[0]*1000,color='#555',label='actual guide delta');axs[0,0].plot(tt,dv[0]*1000,color=colors[A],label='paired output delta');axs[0,0].set(title='First prespecified perturbation',xlabel='relative ns',ylabel='delta mV');axs[0,0].legend(fontsize=7)
axs[0,1].hist(np.sqrt(np.mean(dv*dv,1))*1000,bins=18,color=colors[A]);axs[0,1].axvline(1000/16383,color='black',ls='--');axs[0,1].set(title='128 fixed-bin guide-only changes',xlabel='preADC paired RMS (mV)',ylabel='parents')
rows=m['source_probes'][A]['rows'];tags=list(rows);axs[0,2].barh(range(len(tags)),[rows[k]['waveform_RMS_vs_correct_reference_V']['mean']*1000 for k in tags],color=colors[A]);axs[0,2].set(title='Full source-chain probes',xlabel='RMS to correct reference (mV)')
probe_labels=['Amp x0.5','Amp x1.5','Fast/slow 0','Fast/slow .25','Fast/slow .5','Fast/slow .75','Fast/slow 1','NewNoise','Same input/noise','Rolled source']
assert len(tags)==len(probe_labels)
axs[0,2].set_yticks(range(len(tags)),probe_labels,fontsize=7);axs[0,2].invert_yaxis()
terms=['base_signal','base_electronic','base_sum_rounding','flow_increment_f64'];joint=m['random_decomposition'][A]['groups']['joint8']['by_composition']
for j,cat in enumerate(['pure_EE','pure_had','mixed']):
    r=joint[cat];c=r['four_additive_algebraic_components_covariance_V2']['full480'];vals=[c[t+'__'+t]['mean']for t in terms]+[2*sum(c[terms[k]+'__'+terms[l]]['mean']for k in range(4)for l in range(k+1,4))]
    axs[1,j].bar(range(5),np.asarray(vals)*1e6,color=[colors[A]]*4+['#555']);axs[1,j].set(title=cat+f': joint8 ({r["parents"]} parents)',ylabel='variance contribution (mV²)')
    axs[1,j].set_xticks(range(5),['baseS','baseE','round','delta','all2cov'],fontsize=7);axs[1,j].axhline(0,color='gray',lw=.7)
fig.suptitle('Guide dependence and additive algebra; mapped physical branch split unknown');fig.tight_layout();save(fig,'guide_and_random_decomposition.png')
files={n:{'bytes':(OUT/n).stat().st_size,'sha256':hashlib.sha256((OUT/n).read_bytes()).hexdigest()}for n in ['training_sliced_maps.png','guide_and_random_decomposition.png']}
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':files,'reason':'same bin color for dotted/solid direction traces; source probe names explicitly Fast/slow, not particle fraction','original_figures_preserved':'results/figures/layout_initial','new_fit_generation':0,'saved_data_only':True}
(ROOT/'DISPLAY_REPAIR_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8');print(json.dumps(receipt),flush=True)
