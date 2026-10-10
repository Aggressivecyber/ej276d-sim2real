from pathlib import Path
import json,time,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(root,cfg,num):
    root=Path(root);out=root/'results/figures';out.mkdir(parents=True,exist_ok=True)
    arms=cfg['arms'];colors=['#2474ad','#d45d1c'];labels=['FIXED noise control','NOISE completion primary']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for arm,col,label in zip(arms,colors,labels):
        with np.load(Path(cfg['historical_training_root'])/arm/'LOSS_BY_UPDATE_PRIVATE.npz',allow_pickle=False)as z:x=z['values']
        assert x.shape==(50000,6)
        means=x[:,:3].reshape(50,1000,3).mean(1)
        for k,ax in enumerate(axs):ax.plot(np.arange(1,51)*1000,means[:,k],color=col,label=label,lw=1.6)
    for ax,title in zip(axs,['SOURCE velocity MSE','REAL identity velocity MSE','0.5 SOURCE + 0.5 identity']):
        ax.set(title=title,xlabel='Actual optimizer update',ylabel='1000-update block mean MSE',yscale='log');ax.grid(alpha=.15)
    axs[0].legend(fontsize=8);fig.suptitle('Historical R48 training: frozen 50000-step endpoints; R49 new training=0')
    fig.tight_layout();fig.savefig(out/'training_source_identity.png',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for j,(arm,col,label)in enumerate(zip(arms,colors,labels)):
        with np.load(root/arm/'TRAIN_SOLVER_CHECK_PRIVATE.npz',allow_pickle=False)as z:x=np.sort(z['RMS_difference_V']*16383.)
        axs[0].plot(x,np.arange(1,len(x)+1)/len(x),color=col,label=label)
        row=num['models'][arm];axs[1].scatter(j,row['preADC_relative_q_max'],s=55,color=col)
        axs[1].annotate(f"{row['preADC_relative_q_max']:.2g}",(j,row['preADC_relative_q_max']),xytext=(0,8),textcoords='offset points',ha='center')
    axs[0].axvline(.25,color='gray',ls=':');axs[0].axvline(1,color='gray',ls='--')
    axs[0].set(xlabel='TRAIN128 vs256 RMS difference (LSB)',ylabel='128-row empirical CDF',title='Median <=0.25; p95 <=1 LSB');axs[0].legend(fontsize=8)
    axs[1].axhline(1e-5,color='gray',ls='--');axs[1].set(yscale='log',ylabel='Maximum preADC relative q error',title='Candidate128 / reference256')
    axs[1].set_xticks([0,1],['FIXED','COMPLETION']);fig.tight_layout();fig.savefig(out/'TRAIN_numerical_check.png',bbox_inches='tight');plt.close(fig)
    cal=json.loads((root/'NOISE_CALIBRATION.json').read_text('utf8'))
    with np.load(root/'NOISE_CALIBRATION.npz',allow_pickle=False)as z:p=z['p480'];kap=z['kappa2'];tau=float(z['tau2'])
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    axs[0].plot(np.fft.rfftfreq(480)*1000,p,color='#2474ad');axs[0].set(xlabel='1 ns-grid frequency (MHz)',ylabel='Orthonormal coefficient power (V2)',title='Fixed interpolated TRAIN colored kernel; DC=0')
    axs[1].plot(np.arange(1,9),kap,'o-',color='#2474ad');axs[1].axhline(tau,color='#d45d1c',ls='--',label=f'tau2: 5th percentile, RMS={np.sqrt(tau)*1000:.4f} mV')
    axs[1].set(xlabel='Conditioning Q bin',ylabel='Projected baseline variance (V2)',title='kappa2 before per-bin unit-variance division');axs[1].legend(fontsize=8)
    fig.suptitle('Calibration uses unique consumed matched REAL TRAIN; no DEV/CAL/EVAL calibration')
    fig.tight_layout();fig.savefig(out/'TRAIN_noise_operator.png',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for j,(arm,col,label)in enumerate(zip(arms,colors,labels)):
        with np.load(root/arm/'TRAIN_SOLVER_CHECK_PRIVATE.npz',allow_pickle=False)as z:v=z['input_variance_V2'];a=z['innovation_amplitude_squared_V2']
        for domain,sl,marker in [('SOURCE',slice(0,64),'o'),('REAL',slice(64,128),'x')]:
            axs[j].scatter(np.sqrt(v[sl])*1000,np.sqrt(a[sl])*1000,s=25,marker=marker,label=domain,alpha=.7)
        axs[j].set(xlabel='Input baseline sample SD (mV)',ylabel='Innovation amplitude SD (mV)',title=label);axs[j].set_ylim(0,.60);axs[j].legend(fontsize=8);axs[j].grid(alpha=.15)
    fig.suptitle('Same fixed128 TRAIN rows; amplitude is not realized per-event noise SD')
    fig.tight_layout();fig.savefig(out/'TRAIN_initial_noise_rule.png',bbox_inches='tight');plt.close(fig)
    return sorted(p.name for p in out.glob('TRAIN*.png'))+['training_source_identity.png']

if __name__=='__main__':
    r=Path(__file__).resolve().parent;c=json.loads((r/'config.json').read_text('utf8'));n=json.loads((r/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))
    start=time.monotonic();files=run(r,c,n)
    (r/'TRAIN_PLOT_NUMERIC_RECEIPT.json').write_bytes((json.dumps({'PID':os.getpid(),'elapsed_s':time.monotonic()-start,'figures':files,
      'DEV_read_or_generated':0,'new_neural_inference':0,'new_HGB_or_EM':0},indent=2)+'\n').encode())
    print(json.dumps({'PID':os.getpid(),'figures':files,'elapsed_s':time.monotonic()-start}),flush=True)
