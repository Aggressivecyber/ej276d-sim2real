"""Fixed TRAIN diagnostics and independent complete-waveform numeric reference."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.stats import wasserstein_distance,spearmanr
from waveform_response import ShapeResidualNetwork,forward_raw
from waveform_numpy_reference import reference_numpy
from train_metrics import canonical,q,psd,bins,compare,migrations,summary

def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n','utf8')
@torch.no_grad()
def generate(shape,head,x,eps,mu,scale,edges,p480,tau2,chunk):
    pre=[];adc=[];meta={};device=next(shape.parameters()).device
    stats=[torch.as_tensor(v,device=device,dtype=torch.float64)for v in [mu,scale,edges,p480]]
    for start in range(0,len(x),chunk):
        v=torch.as_tensor(x[start:start+chunk],device=device,dtype=torch.float64)
        e=torch.as_tensor(eps[start:start+chunk],device=device,dtype=torch.float32)
        a,b,_,m=forward_raw(shape,head,v,e,*stats,tau2,straight_through=False)
        pre.append(a.cpu().numpy());adc.append(b.cpu().numpy())
        for k,value in m.items():
            if value.ndim>0:meta.setdefault(k,[]).append(value.cpu().numpy())
    return np.concatenate(pre),np.concatenate(adc),{k:np.concatenate(v)for k,v in meta.items()}

def postcheck(R,C,shape,head,initial,source,real,ids,h,mu,scale,edges,p480,tau2,valley):
    shape.eval();ref=canonical(real);original=source.astype('f8');original_q=q(original);original_p=psd(original);b0=bins(original_q,edges)
    epsS=np.random.Generator(np.random.PCG64(C['source_epsilon_seed'])).standard_normal(source.shape).astype('f4')
    epsR=np.random.Generator(np.random.PCG64(C['real_epsilon_seed'])).standard_normal(real.shape).astype('f4')
    np.savez_compressed(R/'POSTCHECK_EPSILON_PRIVATE.npz',SOURCE=epsS,REAL=epsR)
    control=ShapeResidualNetwork(initial['feature_center'],initial['feature_scale'],hidden=C['hidden']).to('cuda')
    control.load_state_dict(initial);control.eval();outputs={};metadata={};metrics={'SOURCE_INPUT':compare(original,ref,edges)}
    for label,network in [('ZERO_SHAPE',control),('LEARNED',shape)]:
        for domain,x,e in [('SOURCE',source,epsS),('REAL',real,epsR)]:
            pre,adc,meta=generate(network,head,x,e,mu,scale,edges,p480,tau2,C['generation_chunk'])
            name=label+'_'+domain;outputs[name]=(pre,adc);metadata[name]=meta
            parent={'source_parent_ID':ids,'source_h':h}if domain=='SOURCE'else{'REAL_original_TRAIN_row':np.arange(len(real))}
            np.savez_compressed(R/(name+'_PRIVATE.npz'),pre_ADC_V=pre,**parent,**meta)
            metrics[name]=compare(adc,ref,edges)
            metrics[name].update(Qstar_W1_mVns=float(wasserstein_distance(meta['Qstar']*1000,q(ref)*1000)),
              max_relative_Qstar_error_f64=float(meta['Q_relative_error_f64'].max()),max_relative_Qstar_error_f32=float(meta['Q_relative_error_f32'].max()),
              range_projection_fraction=float(meta['range_projection_active'].mean()),projection_RMS_V=summary(meta['projection_RMS_V']),
              same_public_input_path=True,Q_head_frozen=True)
            write(R/(name+'_MIGRATION.json'),migrations(meta))
            if domain=='SOURCE':
                p=psd(adc);metrics[name].update(source_Q_Spearman=float(spearmanr(original_q,q(adc)).statistic),source_PSD_Spearman=float(spearmanr(original_p,p).statistic))
                metrics[name]['purehad_original_bin_valley']={}
                for j in [2,3]:
                    use=(h==1)&(b0==j);d=valley['bins'][j];assert use.any()
                    metrics[name]['purehad_original_bin_valley'][str(j+1)]=dict(N=int(use.sum()),input_fraction=float(np.mean((original_p[use]>=d['low'])&(original_p[use]<=d['high']))),
                      output_fraction=float(np.mean((p[use]>=d['low'])&(p[use]<=d['high']))))
    qstate={k:v.detach().cpu().numpy()for k,v in head.state_dict().items()};sstate={k:v.detach().cpu().numpy()for k,v in shape.state_dict().items()}
    numerical={}
    for domain,x,e in [('SOURCE',source,epsS),('REAL',real,epsR)]:
        n=C['numeric_CPU_reference_rows_per_domain']
        pre,adc,raw,meta=reference_numpy(qstate,sstate,x[:n],e[:n],mu,scale,edges,p480,tau2)
        saved=outputs['LEARNED_'+domain];rms=np.sqrt(np.mean((saved[0][:n].astype('f8')-pre.astype('f8'))**2,1))*16383
        qt=meta['Qstar'];qerr=np.abs(qt-metadata['LEARNED_'+domain]['Qstar'][:n])/qt
        codes=lambda v:np.rint((v.astype('f8')+.5)*16383).astype('i4')
        delta=codes(adc)-codes(saved[1][:n]);numerical[domain]=dict(N=n,median_RMS_LSB=float(np.median(rms)),p95_RMS_LSB=float(np.quantile(rms,.95)),max_RMS_LSB=float(rms.max()),
          independent_Qstar_relative_error_max=float(qerr.max()),ADC_integer_code_equal=bool(np.all(delta==0)),ADC_differing_sample_N=int(np.count_nonzero(delta)),ADC_maxabs_code_difference=int(np.max(np.abs(delta))),
          pass_gate=bool(np.median(rms)<=C['numeric_median_LSB_max']and np.quantile(rms,.95)<=C['numeric_p95_LSB_max']and qerr.max()<=1e-5))
        np.savez_compressed(R/(domain+'_NUMERIC_REFERENCE_PRIVATE.npz'),reference_preADC=pre,torch_preADC=saved[0][:n],reference_Qstar=qt,RMS_difference_LSB=rms)
    reloaded=ShapeResidualNetwork(initial['feature_center'],initial['feature_scale'],hidden=C['hidden']).to('cuda')
    reloaded.load_state_dict(torch.load(R/'FINAL_SHAPE.pt',map_location='cuda',weights_only=True));reloaded.eval();replay={}
    for domain,x,seed in [('SOURCE',source,C['source_epsilon_seed']),('REAL',real,C['real_epsilon_seed'])]:
        n=2000;e=np.random.Generator(np.random.PCG64(seed)).standard_normal((n,480)).astype('f4')
        pre,adc,meta=generate(reloaded,head,x[:n],e,mu,scale,edges,p480,tau2,C['generation_chunk'])
        saved=outputs['LEARNED_'+domain];replay[domain]=dict(N=n,preADC_bitwise=bool(np.array_equal(pre,saved[0][:n])),
          canonical_ADC_bitwise=bool(np.array_equal(adc,saved[1][:n])),Qstar_bitwise=bool(np.array_equal(meta['Qstar'],metadata['LEARNED_'+domain]['Qstar'][:n])))
    write(R/'TRAIN_METRICS.json',dict(scope='Complete waveform TRAIN diagnostics; frozen held-out A-F still required',metrics=metrics))
    write(R/'NUMERICAL_REFERENCE.json',dict(reference='independent NumPy float64 Q head and complete128-width shape MLP, noise/Box/quantizers',results=numerical))
    write(R/'TRAIN_SERIALIZATION_REPLAY.json',dict(scope='TRAIN4000 serialization replay; does not replace independent held-out4000',results=replay))
    plot(R,outputs,original,ref)
    passed=all(v['pass_gate']for v in numerical.values())and all(v['preADC_bitwise']and v['canonical_ADC_bitwise']and v['Qstar_bitwise']for v in replay.values())
    receipt=dict(numerical_and_TRAIN_replay_pass=passed,TRAIN_rows_generated=2*(len(source)+len(real)),independent_numpy_rows=128,TRAIN_replay_rows=4000,
      heldout_arrays_read=False,final_A_to_F_pass=None,final_heldout_4000_replay_complete=False)
    write(R/'POSTCHECK_RECEIPT.json',receipt)
    if not passed:raise FloatingPointError('full-waveform numerical or TRAIN replay failure')
    return receipt

def plot(R,outputs,original,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    series={'REAL reference':reference,'Original SOURCE':original,'Q-only SOURCE':outputs['ZERO_SHAPE_SOURCE'][1],
      'LEARNED SOURCE':outputs['LEARNED_SOURCE'][1],'LEARNED REAL':outputs['LEARNED_REAL'][1]}
    for name,v in series.items():
        a=np.sort(q(v)*1000);ax[0,0].plot(a,np.arange(1,len(a)+1)/len(a),label=name,lw=1)
        ax[0,1].hist(psd(v),bins=np.linspace(-.1,.7,161),density=True,histtype='step',label=name)
        ax[1,0].plot(v.mean(0),label=name,lw=1)
        baseline=v[:,:100]-v[:,:100].mean(1,keepdims=True);a=np.sort(np.sqrt(np.mean(baseline**2,1)))
        ax[1,1].plot(a,np.arange(1,len(a)+1)/len(a),label=name,lw=1)
    ax[0,0].set(xscale='log',xlabel='Canonical charge [mV ns]',ylabel='Empirical CDF');ax[0,0].legend(fontsize=8)
    ax[0,1].set(xlabel='Signed PSD',ylabel='Density (plot range only)')
    ax[1,0].set(xlabel='Sample in complete480 window',ylabel='Mean voltage [V]')
    ax[1,1].set(xlabel='Baseline RMS [V]',ylabel='Empirical CDF')
    fig.suptitle('R72 complete-parent waveform training diagnostics; held-out A-F remains separate')
    fig.savefig(R/'FULL_WAVEFORM_TRAIN_DIAGNOSTICS.png',dpi=150);plt.close(fig)
    with np.load(R/'LOSS_BY_UPDATE_PRIVATE.npz',allow_pickle=False)as z:values=z['values'];columns=z['columns']
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for j in range(10):ax.plot(np.arange(1,len(values)+1),values[:,j],label=str(columns[j]),lw=.7)
    ax.set(yscale='symlog',xlabel='Complete-parent update',ylabel='Registered dimensionless objective');ax.legend(ncol=3,fontsize=8)
    fig.savefig(R/'FULL_TRAIN_LOSSES.png',dpi=150);plt.close(fig)
