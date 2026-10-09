from __future__ import annotations
import pathlib,json,time,datetime,hashlib,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluation as ev
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True,parents=True);START=time.monotonic()
CFG=json.loads((ROOT/'config.json').read_text('utf8'));m=json.loads((ROOT/'metrics.json').read_text('utf8'))
replay=json.loads((ROOT/'REPLAY_RECEIPT.json').read_text('utf8'));A='JOINT_KERNEL_FINAL'
controls=['BASE','INDEPENDENT_CFM','OT_CFM','PSD_ORDERED_CFM','CONDITIONAL_DIFFUSION_EDIT','FULL_COVARIANCE_CALIBRATION']
labels={'REAL':'REAL DEV','BASE':'Frozen BASE','INDEPENDENT_CFM':'Independent flow','OT_CFM':'OT flow',
    'PSD_ORDERED_CFM':'PSD ordered flow','CONDITIONAL_DIFFUSION_EDIT':'Diffusion edit','FULL_COVARIANCE_CALIBRATION':'Gaussian covariance map',A:'Joint-kernel G',
    'TRAIN_REFERENCE':'Raw REAL TRAIN ref','SOURCE_CLEAN':'Source ADC reference','SOURCE_INPUT':'Actual NN source'}
colors={'REAL':'#111111','BASE':'#888888','INDEPENDENT_CFM':'#86a9cc','OT_CFM':'#c6a98a',
    'PSD_ORDERED_CFM':'#ad80b7','CONDITIONAL_DIFFUSION_EDIT':'#d98628','FULL_COVARIANCE_CALIBRATION':'#ce7d31',A:'#0867ad','TRAIN_REFERENCE':'#398347',
    'SOURCE_CLEAN':'#6e839e','SOURCE_INPUT':'#6e839e','BASE_REAL_CONTROL':'#888888','JOINT_KERNEL_REAL_CONTROL':'#0867ad'}
labels.update(BASE_REAL_CONTROL='BASE G on REAL',JOINT_KERNEL_REAL_CONTROL='Joint-kernel G on REAL')
styles={'BASE':'--','TRAIN_REFERENCE':'--','INDEPENDENT_CFM':':','OT_CFM':':','PSD_ORDERED_CFM':':','CONDITIONAL_DIFFUSION_EDIT':':'}
def write(n,v):(ROOT/n).write_text(json.dumps(ev.safe(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in(z.files if keys is None else keys)}
def save(fig,n):fig.savefig(OUT/n,bbox_inches='tight');plt.close(fig)
def line(ax,n,x,y,**kw):ax.plot(x,y,color=colors[n],ls=styles.get(n,'-'),label=labels[n],lw=1.15,**kw)
def cdf(ax,n,x,scale=1):
    x=np.sort(x[np.isfinite(x)])*scale
    if len(x):line(ax,n,x,np.arange(1,len(x)+1)/len(x))
def measure(a,b):
    delta=a/b-1 if a is not None and b is not None and b>0 else (0. if a==b==0 else None)
    return {'candidate':a,'BASE':b,'relative_change':delta,'pass':delta is not None and delta<=.10}
features=load(ROOT/'results/FEATURES_PRIVATE.npz');waves=load(ROOT/'results/PLOT_WAVEFORMS_PRIVATE.npz')
psd=m['PSD_main'];width=m['energy_conditional_width'];edges=np.asarray(psd['edges_logQ']);tt=np.arange(480)-120
def obs(n):
    z=np.log(np.maximum(features[n+'__Qtotal_mVns'],1e-30));b=np.searchsorted(edges,z,side='right')-1;b[z==edges[-1]]=7;return b
views=[*CFG['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480'];full=['full_historical_shape','normalized_u480']

paired=m['C2ST']['paired'][A];cs=m['C2ST']['results']
stats=json.loads((ROOT/'TRAINING_COMPACT.json').read_text('utf8'));cal=json.loads((ROOT/'INITIAL_GRADIENT_SCALE_COMPACT.json').read_text('utf8'))
control_pair=json.loads((ROOT/'SAME_TREATED_REAL_CONTROL_PAIRED.json').read_text('utf8'))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})

old_dir=ROOT/'results/figures/layout_initial';old_dir.mkdir(exist_ok=True)
for n in ['c2st.png','training_joint_kernel.png','same_treated_REAL_controls.png']:
    p=OUT/n;q=old_dir/n
    if q.exists():raise RuntimeError('repair already attempted')
    q.write_bytes(p.read_bytes())
short=['shape493','V480','prompt','postpeak','tail','baseline','Q/PSD','pure u480','seen\n480 points','complement\n480 points']
fig,axs=plt.subplots(2,1,figsize=(14,8.3),gridspec_kw={'height_ratios':[1.4,1]})
for j,n in enumerate(['BASE','PSD_ORDERED_CFM','FULL_COVARIANCE_CALIBRATION',A]):
    y=np.array([cs[n][v]['balanced_accuracy']for v in views]);ci=np.array([cs[n][v]['heldout_CI']['balanced_accuracy_95CI']for v in views])
    axs[0].errorbar(np.arange(10)+(j-1.5)*.13,y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',ms=3,capsize=2,color=colors[n],label=labels[n])
y=[cs['RAW_REAL_REAL'][v]['balanced_accuracy'] if cs['RAW_REAL_REAL'][v]['balanced_accuracy']is not None else np.nan for v in views]
axs[0].plot(range(10),y,'s--',ms=3,lw=.8,color=colors['TRAIN_REFERENCE'],label='Raw REAL-REAL (Q/PSD unknown)')
y=np.array([paired[v]['BA_delta_first_minus_second']for v in views]);ci=np.array([paired[v]['BA_delta_95CI']for v in views])
axs[1].errorbar(range(10),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='o',ms=4,capsize=2,color=colors[A],label='Joint-kernel G minus BASE')
axs[0].axhline(.5,color='gray',ls=':');axs[0].set(ylim=(.43,1.02),ylabel='fixed holdout BA')
axs[1].axhline(0,color='gray',lw=.8);axs[1].set_ylabel('paired BA difference')
for ax in axs:ax.set_xticks(range(10),short);ax.grid(axis='y',alpha=.12)
axs[0].legend(fontsize=8,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.12))
axs[1].legend(fontsize=8,loc='upper left')
axs[0].set_title('Fixed 600/250 rows; 500 group resamples; lower is less distinguishable')
axs[1].set_title('Intervals condition on fitted classifiers; same acquisition, not physical equivalence')
fig.tight_layout();save(fig,'c2st.png')

trace=json.loads((ROOT/A/'TRAINING_TRACE.json').read_text('utf8'))
grams=load(ROOT/A/'MMD_GRAMS_BY_UPDATE_PRIVATE.npz')['values']
fig,axs=plt.subplots(2,2,figsize=(13,7.2))
trace=[r for r in trace if 'step'in r]
x=[0]+[r['step']for r in trace]+[2950];y=[stats['step0_J20']]+[r['validation']['J']for r in trace]+[stats['final_J20']]
axs[0,0].plot(x,y,'o-',ms=3,color=colors[A],label='fixed J20');axs[0,0].axvline(stats['selected_step'],ls=':',color='gray',label='record-only selected2750');axs[0,0].axvline(2950,ls='--',color=colors[A],label='sole DEV endpoint2950')
axs[0,0].annotate(f'final2950 J={stats["final_J20"]:.5f}',(2950,stats['final_J20']),xytext=(-90,-20),textcoords='offset points',fontsize=7,arrowprops={'arrowstyle':'-','color':'gray'})
axs[0,0].set(title='TRAIN internal validation; sole final endpoint',xlabel='update',ylabel='J20');axs[0,0].legend(fontsize=7)
for j,(title,c)in enumerate([('within G',2),('within REAL',3),('cross all',4)]):axs[0,1].plot(np.arange(1,2951),grams[:,c],lw=.35,alpha=.65,label=title)
axs[0,1].set(title='Saved Gram means; not separate feature losses',xlabel='update',ylabel='kernel mean');axs[0,1].legend(fontsize=7)
for col,title in [(0,'raw MMD U'),(1,'fixed c x MMD U')]:axs[1,0].plot(np.arange(1,2951),grams[:,col],lw=.4,label=title)
axs[1,0].axhline(0,color='gray',lw=.7);axs[1,0].set(title=f'Fixed c={stats["gradient_coefficient"]:.5f}; no truncation',xlabel='update',ylabel='loss');axs[1,0].legend(fontsize=7)
cr=cal['rows'];xx=np.arange(1,65)
axs[1,1].semilogy(xx,[r['old_grad_y_Frobenius_squared']for r in cr],lw=.9,label='old fresh fine grad-y squared')
axs[1,1].semilogy(xx,[r['new_grad_y_Frobenius_squared']for r in cr],lw=.9,label='raw joint MMD grad-y squared')
axs[1,1].semilogy(xx,[stats['gradient_coefficient']**2*r['new_grad_y_Frobenius_squared']for r in cr],lw=.9,ls=':',label='c-scaled joint grad-y squared')
axs[1,1].set(title='Once-only initial64 TRAIN batches; no parameter update',xlabel='original update ordinal',ylabel='Frobenius squared');axs[1,1].legend(fontsize=7)
for ax in axs.flat:ax.grid(alpha=.12)
fig.tight_layout();save(fig,'training_joint_kernel.png')

fig,axs=plt.subplots(2,2,figsize=(13,7.4))
cv=['full_historical_shape','normalized_u480','baseline_local','full_voltage']
shortcv=['shape493','pure u480','baseline','V480\n(descriptive)']
for j,n in enumerate(['RAW_REAL_REAL','BASE_REAL_CONTROL','JOINT_KERNEL_REAL_CONTROL']):
    color=colors['TRAIN_REFERENCE']if n=='RAW_REAL_REAL'else colors[n]
    yy=np.asarray([cs[n][v]['balanced_accuracy']for v in cv]);cc=np.asarray([cs[n][v]['heldout_CI']['balanced_accuracy_95CI']for v in cv])
    axs[0,0].errorbar(np.arange(4)+(j-1)*.13,yy,yerr=np.maximum(0,np.vstack([yy-cc[:,0],cc[:,1]-yy])),fmt='o',capsize=2,ms=4,color=color,label='RAW TRAIN-DEV'if n=='RAW_REAL_REAL'else labels[n])
axs[0,0].axhline(.5,color='gray',ls=':');axs[0,0].set(title='Same reference rows / stochastic processing',ylabel='fixed holdout BA',ylim=(.43,1.02));axs[0,0].set_xticks(range(4),shortcv);axs[0,0].legend(fontsize=7)
delta=np.asarray([control_pair[v]['BA_delta_first_minus_second']for v in cv]);cc=np.asarray([control_pair[v]['BA_delta_95CI']for v in cv])
axs[0,1].errorbar(range(4),delta,yerr=np.maximum(0,np.vstack([delta-cc[:,0],cc[:,1]-delta])),fmt='o',capsize=2,color=colors[A])
axs[0,1].hlines(.01,-.4,2.4,linestyles=':',color='red',label='first3: point limit +.01');axs[0,1].hlines(.10,-.4,2.4,linestyles='--',color='gray',label='first3: 95% upper +.10');axs[0,1].axvspan(2.6,3.4,color='#eeeeee',alpha=.4);axs[0,1].axhline(0,lw=.7,color='gray')
axs[0,1].set(title='New minus BASE same-treated REAL',ylabel='paired BA difference');axs[0,1].set_xticks(range(4),shortcv);axs[0,1].legend(fontsize=7)
for n in ['REAL','TRAIN_REFERENCE','BASE_REAL_CONTROL','JOINT_KERNEL_REAL_CONTROL']:
    cdf(axs[1,0],n,features[n+'__PSD'][features[n+'__valid']])
    cdf(axs[1,1],n,features[n+'__baseline_RMS_V'][features[n+'__valid']],1000)
axs[1,0].set(title='Complete valid populations',xlabel='signed PSD',ylabel='CDF');axs[1,0].legend(fontsize=7)
axs[1,1].set(title='REAL may receive added stochastic response',xlabel='baseline RMS (mV)',ylabel='CDF')
for ax in axs.flat:ax.grid(alpha=.12)
fig.suptitle('REAL processing controls; gates apply to shape/u/baseline only, V480 is descriptive');fig.tight_layout();save(fig,'same_treated_REAL_controls.png')


names=['c2st.png','training_joint_kernel.png','same_treated_REAL_controls.png']
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
    'new_fits_generation_updates':0,'only_display_repair':True,'saved_old_versions':'results/figures/layout_initial',
    'figures':{n:{'bytes':(OUT/n).stat().st_size,'sha256':hashlib.sha256((OUT/n).read_bytes()).hexdigest()}for n in names},
    'J2950_from_saved_TRAINING_COMPACT':stats['final_J20'],'V480_descriptive_only':True,'actual_visual_QA_pending':True}
write('FIGURE_REPAIR_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
