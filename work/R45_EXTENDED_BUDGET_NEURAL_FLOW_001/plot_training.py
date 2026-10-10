from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import json,os,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/figures';OUT.mkdir(parents=True,exist_ok=True)
START=time.monotonic();A='EXTENDED_FLOW_FINAL'
probe=json.loads((ROOT/'FIXED_PROBE_TRAJECTORY.json').read_text('utf8'))
receipt=json.loads((ROOT/'TRAINING_NUMERIC_RECEIPT.json').read_text('utf8'))
solver=json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))
steps=np.asarray([r['global_step'] for r in probe]);paths=[]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
def save(fig,name):
    p=OUT/(name+'.png');fig.savefig(p,bbox_inches='tight');plt.close(fig);paths.append(str(p))
fig,axes=plt.subplots(1,3,figsize=(15,4.2))
for ax,key,title in zip(axes,['MSE480','PSD_RMSE','PSD_signed_mean'],['Full480 path velocity MSE','PSD velocity RMSE','Signed PSD velocity residual']):
    ax.plot(steps,[r['overall'][key] for r in probe],'-o',color='#0867ad',ms=4)
    ax.set(xlabel='Total Adam updates',ylabel=key,title=title);ax.axvline(2950,color='#666',ls=':',lw=1)
axes[0].axhline(.9*probe[0]['overall']['MSE480'],color='#b45b30',ls='--',label='90% initial MSE marker')
axes[0].legend(fontsize=8);axes[2].axhline(0,color='#666',ls=':',lw=1)
fig.suptitle('Fixed 4096 TRAIN path states: same parents/t/white at every checkpoint; no model selection')
fig.tight_layout();save(fig,'fixed_TRAIN_probe')
fig,axes=plt.subplots(1,2,figsize=(12,4.5));colors=plt.cm.viridis(np.linspace(.08,.92,8))
for b,color in enumerate(colors):
    for ax,key in zip(axes,['MSE480','PSD_RMSE']):
        ax.plot(steps,[r['by_Q'][b][key] for r in probe],'-o',ms=3,lw=1,color=color,label='Q bin'+str(b+1))
for ax,title in zip(axes,['Full480 MSE per fixed charge bin','PSD velocity RMSE per fixed charge bin']):
    ax.set(xlabel='Total Adam updates',title=title)
axes[1].legend(fontsize=8,ncol=2);fig.suptitle('512 fixed path states per Q bin; not independent unseen-parent validation')
fig.tight_layout();save(fig,'probe_by_charge')
fig,axes=plt.subplots(1,3,figsize=(14,4.6))
initial=np.array([r['MSE480'] for r in probe[0]['by_Q_time']]).reshape(8,4)
final=np.array([r['MSE480'] for r in probe[-1]['by_Q_time']]).reshape(8,4)
lo=min(initial.min(),final.min());hi=max(initial.max(),final.max())
for ax,value,title in zip(axes,[initial,final,final/initial],['Inherited2950','Fixed50000','Final / initial MSE']):
    im=ax.imshow(value,aspect='auto',cmap='viridis',vmin=lo if title!='Final / initial MSE' else None,vmax=hi if title!='Final / initial MSE' else None)
    ax.set(title=title,xlabel='Path time interval',ylabel='Q bin');ax.set_xticks(range(4),['[0,.25)','.25-.5','.5-.75','.75-1']);ax.set_yticks(range(8),range(1,9))
    for b in range(8):
        for t in range(4):ax.text(t,b,f'{value[b,t]:.3f}',ha='center',va='center',fontsize=7,color='white' if value[b,t]<(lo+hi)/2 and title!='Final / initial MSE' else 'black')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
fig.suptitle('Same fixed TRAIN states; no confidence interval or ODE-error inference')
fig.tight_layout();save(fig,'probe_charge_time')
loss=np.load(ROOT/A/'LOSS_BY_UPDATE_PRIVATE.npy',mmap_mode='r',allow_pickle=False)
assert loss.shape==(47050,10)
fig,axes=plt.subplots(1,3,figsize=(15,4.3))
columns=[0,3,7];names=['Per-batch full480 velocity MSE','Pre-clip gradient norm','Per-batch PSD velocity RMSE']
for ax,column,title in zip(axes,columns,names):
    x=[];med=[];low=[];high=[]
    for start in range(0,len(loss),250):
        block=np.asarray(loss[start:min(start+250,len(loss)),column]);x.append(2951+start+(len(block)-1)/2)
        q=np.quantile(block,[.1,.5,.9]);low.append(q[0]);med.append(q[1]);high.append(q[2])
    ax.fill_between(x,low,high,color='#0867ad',alpha=.18,label='Within250-update10-90%')
    ax.plot(x,med,color='#0867ad',lw=1.2,label='Within250-update median')
    ax.set(xlabel='Total Adam updates',title=title)
axes[0].legend(fontsize=8);fig.suptitle('Fresh Monte Carlo t/white and repeated fixed parents; batch variation is not a convergence CI')
fig.tight_layout();save(fig,'extended_training_trace')
with np.load(ROOT/'TRAIN_SOLVER_CHECK_PRIVATE.npz',allow_pickle=False) as z:
    error=z['RMS_difference_V'].copy();x=z['input_V'].copy();a=z['solution32_f32_V'].copy();b=z['solution64_f32_V'].copy()
lsb=1/16383.;fig,axes=plt.subplots(1,3,figsize=(15,4.4))
axes[0].plot(np.sort(error/lsb),np.arange(1,len(error)+1)/len(error),color='#0867ad')
axes[0].axvline(.25,color='#b45b30',ls='--',label='Median limit .25LSB');axes[0].axvline(1,color='#c02323',ls=':',label='p95 limit1LSB')
axes[0].axhline(.5,color='#666',lw=.6);axes[0].axhline(.95,color='#666',lw=.6)
axes[0].set(xlabel='32 vs64 step RMS / ADC LSB',ylabel='Empirical CDF',title='Original128 TRAIN inputs/white');axes[0].legend(fontsize=7)
for ax,index,name in zip(axes[1:],[0,64],['Fixed SOURCE TRAIN row0','Fixed REAL TRAIN row0']):
    ax.plot(np.arange(480)-120,x[index]*1000,color='#666',lw=1,label='Input')
    ax.plot(np.arange(480)-120,a[index]*1000,color='#0867ad',lw=1,label='32Heun candidate')
    ax.plot(np.arange(480)-120,b[index]*1000,color='#d6862f',ls='--',lw=1,label='64Heun numerical reference')
    ax.set(xlabel='Time / ns, reference index120',ylabel='Voltage / mV',title=name);ax.legend(fontsize=7)
fig.suptitle('Pre-ADC TRAIN numerical gate: '+('PASS' if solver['pass'] else 'FAIL; DEV evaluation not executed'))
fig.tight_layout();save(fig,'TRAIN_solver_gate')
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
     'figures':paths,'new_response_forward_fit_or_HGB_EM':0,'plotting_saved_TRAIN_only':True,
     'scientific_distribution_evaluation_permitted':solver['pass'],'training_fixed50000_only':True}
(ROOT/'TRAIN_PLOT_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps(rec))
