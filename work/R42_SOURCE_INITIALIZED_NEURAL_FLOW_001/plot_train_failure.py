"""Only stored TRAIN training/solver evidence; no DEV inference or evaluation."""
from pathlib import Path
import json,os,time,datetime,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/figures';OUT.mkdir(parents=True,exist_ok=True)
START=time.monotonic();A='SOURCE_NEURAL_FLOW_FINAL'
def load(n):return json.loads((ROOT/n).read_text('utf8'))
def save(fig,n):fig.savefig(OUT/n,bbox_inches='tight');plt.close(fig)
cfg=load('config.json');train=load(A+'/TRAINING_RECEIPT.json');check=load('TRAIN_SOLVER_CHECK.json')
assert not check['pass'] and check['DEV_read_or_generated']==0
with np.load(ROOT/A/'LOSS_BY_UPDATE_PRIVATE.npz',allow_pickle=False)as z:d=z['values'].copy()
with np.load(ROOT/'TRAIN_SOLVER_CHECK_PRIVATE.npz',allow_pickle=False)as z:
    x=z['input_V'].copy();a=z['solution32_f32_V'].copy();b=z['solution64_f32_V'].copy();err=z['RMS_difference_V'].copy()
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
blue='#0867ad';orange='#dd8833';gray='#666666';steps=np.arange(1,len(d)+1)
fig,axs=plt.subplots(2,2,figsize=(12.6,7.0))
axs[0,0].plot(steps,d[:,0],lw=.65,color=blue);axs[0,0].set(title='Only loss: full480 tangent velocity MSE',xlabel='update',ylabel='training batch MSE')
for k,tag in [(1,'target'),(2,'prediction')]:axs[0,1].plot(steps,d[:,k],lw=.6,label=tag)
axs[0,1].set(title='Standardized velocity RMS',xlabel='update',ylabel='RMS');axs[0,1].legend(fontsize=8)
axs[1,0].plot(steps,d[:,3],lw=.6,color=blue);axs[1,0].axhline(5,color=gray,ls=':');axs[1,0].set(title='Actual preclip gradient norm',xlabel='update',ylabel='L2 norm')
for k,tag in [(4,'path target'),(5,'network field')]:axs[1,1].plot(steps,d[:,k],lw=.6,label=tag)
axs[1,1].set(title='Unit-charge tangent residual',xlabel='update',ylabel='absolute q residual');axs[1,1].legend(fontsize=8)
for ax in axs.flat:ax.grid(alpha=.15)
fig.suptitle('R42: new1,371,841-parameter model, sole final2950; no old response weights')
fig.tight_layout();save(fig,'training_source_neural_flow.png')

fig,axs=plt.subplots(1,2,figsize=(12,4.3))
for sl,tag,color in [(slice(0,64),'TRAIN SOURCE first64',blue),(slice(64,128),'TRAIN REAL first64',orange)]:
    e=np.sort(err[sl]/check['LSB_V']);axs[0].plot(e,np.arange(1,65)/64,label=tag,color=color)
    axs[1].plot(np.arange(128)[sl],err[sl]/check['LSB_V'],'.',label=tag,color=color)
axs[0].axvline(.25,color=gray,ls=':',label='median limit .25LSB');axs[0].axvline(1,color=gray,ls='--',label='p95 limit1LSB')
axs[0].set(title='32/64-step TRAIN difference',xlabel='Per-row voltage RMS / ADC LSB',ylabel='CDF');axs[0].legend(fontsize=7.5)
axs[1].set(title=f"Fixed32 fails: median{check['RMS_median_LSB']:.3f}, p95{check['RMS_p95_LSB']:.3f}LSB",xlabel='fixed check row',ylabel='RMS / ADC LSB')
for ax in axs:ax.grid(alpha=.15)
fig.suptitle('64 steps is a numerical reference only; no DEV candidate or rescue')
fig.tight_layout();save(fig,'TRAIN_solver32_vs64.png')

fig,axs=plt.subplots(2,2,figsize=(13,7.2));tt=np.arange(480)-120
for ax,i in zip(axs.flat,[0,63,64,127]):
    tag='SOURCE'if i<64 else'REAL'
    ax.plot(tt,x[i]*1000,color=gray,lw=.9,label='TRAIN input')
    ax.plot(tt,a[i]*1000,color=blue,lw=.9,label='32Heun')
    ax.plot(tt,b[i]*1000,color=orange,lw=.7,ls='--',label='64Heun numeric ref')
    ax.set(title=f'{tag} fixed check row{i}: RMS{err[i]/check["LSB_V"]:.3f}LSB',xlabel='relative to saved input index120 (ns)',ylabel='raw mV',xlim=(-120,359));ax.grid(alpha=.15)
axs[0,0].legend(fontsize=8);fig.suptitle('First/last checked TRAIN rows; all480 points; no selection from DEV')
fig.tight_layout();save(fig,'TRAIN_solver_waveform_examples.png')

with np.load(ROOT/'UNIT_CHARGE_CONSTANTS.npz',allow_pickle=False)as z:mu=z['mu_unit_q'].copy();sd=z['raw_SD_unit_q'].copy();scale=z['scale_unit_q'].copy()
constants=load('UNIT_CHARGE_CONSTANTS.json')
fig,axs=plt.subplots(2,2,figsize=(12.6,7.2))
for j in range(8):
    axs[0,0].plot(tt,mu[j],lw=.85,label=f'Q{j+1}')
    axs[0,1].plot(tt,sd[j],lw=.85,label=f'Q{j+1}')
axs[0,0].set(title='Fixed REAL TRAIN unit-charge mean',xlabel='relative ns',ylabel='1/ns');axs[0,0].legend(ncol=4,fontsize=7)
axs[0,1].set(title='Fixed REAL TRAIN pointwise sample SD',xlabel='relative ns',ylabel='1/ns')
axs[1,0].bar(np.arange(8)+1,[r['floored_coordinates']for r in constants['bins']],color=blue)
axs[1,0].set(title='Fixed scale floor; no later tuning',xlabel='frozen Q bin',ylabel='floored coordinates of480')
vals=[check[k]['max_relative_q_error_f32']for k in ['solver32','solver64']]
axs[1,1].bar([0,1],vals,color=[blue,orange]);axs[1,1].set_xticks([0,1],['32Heun','64Heun reference'])
axs[1,1].axhline(1e-5,color=gray,ls='--',label='required max1e-5');axs[1,1].set_yscale('log');axs[1,1].set(title='Actual float32 preADC charge error',ylabel='maximum relative error');axs[1,1].legend(fontsize=7.5)
for ax in axs.flat:ax.grid(alpha=.15)
fig.suptitle('Common TRAIN coordinates and charge pass do not resolve solver-step failure')
fig.tight_layout();save(fig,'TRAIN_coordinates_charge.png')

files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sorted(OUT.glob('*.png'))}
r={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'figures':files,'figure_count':len(files),
   'inputs':'stored TRAIN trace,128 solver check rows and fixed TRAIN constants only','DEV_scientific_decode_generation':0,
   'response_fit_update':0,'HGB_EM':0,'additional_solver_calls':0,'all_plots_from_saved_results':True,'actual_visual_QA_pending':True}
(ROOT/'PLOT_FAILURE_RECEIPT.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n','utf8');print(json.dumps(r),flush=True)
