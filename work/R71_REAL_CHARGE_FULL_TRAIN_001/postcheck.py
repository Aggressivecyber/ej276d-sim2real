"""Fixed TRAIN component diagnostics; no held-out A-F model acceptance."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.stats import wasserstein_distance,ks_2samp,spearmanr
from public_charge_head import PublicConditionalChargeHead
from common_response import generate
from numpy_reference import q,reference

def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
def canonical(v):return -.5+np.clip(np.rint((np.asarray(v,dtype='f8')+.5)*16383),0,16383)/16383
def psd(v):return 1-(v[:,100:150].sum(1)-.5*v[:,:100].sum(1))/q(v)
def bins(qs,edges):
    logq=np.log(1000*qs);b=np.searchsorted(edges,logq,side='right')-1;b[logq==edges[-1]]=7;return b
def rms(v):return np.sqrt(np.mean((v[:,:100]-v[:,:100].mean(1,keepdims=True))**2,1))
def acf(v):
    b=v[:,:100]-v[:,:100].mean(1,keepdims=True);power=np.mean(b*b,1)
    return np.mean(b[:,:-1]*b[:,1:],1)/np.maximum(power,1e-20)
def boundary(v):
    data={f'first_{i}_{j}':v[:,j]-v[:,i]for i,j in[(99,100),(149,150),(399,400)]}
    data.update({f'second_{i}':v[:,i-1]-2*v[:,i]+v[:,i+1]for i in[99,100,149,150,399,400]})
    return data
def summary(x):
    return dict(n=len(x),min=float(np.min(x)),median=float(np.median(x)),mean=float(np.mean(x)),max=float(np.max(x)))
def compare(v,ref,edges):
    qs,qr=q(v),q(ref);p,pr=psd(v),psd(ref);b,br=bins(qs,edges),bins(qr,edges)
    bv,bref=boundary(v),boundary(ref)
    by=[]
    for j in range(8):
        sx=b==j;rx=br==j
        by.append(dict(bin_1based=j+1,candidate_N=int(sx.sum()),reference_N=int(rx.sum()),
            PSD_W1=float(wasserstein_distance(p[sx],pr[rx]))if sx.any()and rx.any()else None))
    return dict(N=len(v),weight_sum=1.,Q_mVns=summary(qs*1000),Q_W1_mVns=float(wasserstein_distance(qs*1000,qr*1000)),
      logQ_W1=float(wasserstein_distance(np.log(qs),np.log(qr))),Q_KS=float(ks_2samp(qs,qr).statistic),
      output_bin_counts=[int((b==j).sum())for j in range(-1,9)],output_bin_weights=[float((b==j).mean())for j in range(-1,9)],
      outside_support_N=int(((b<0)|(b>7)).sum()),outside_support_mass=float(((b<0)|(b>7)).mean()),
      overall_PSD_W1=float(wasserstein_distance(p,pr)),output_condition_PSD=by,
      Q1_Q7_PSD_W1_mean=float(np.mean([row['PSD_W1']for row in by[:7]]))if all(row['PSD_W1']is not None for row in by[:7])else None,
      baseline_RMS_W1_V=float(wasserstein_distance(rms(v),rms(ref))),baseline_ACF1_W1=float(wasserstein_distance(acf(v),acf(ref))),
      boundary_W1_V={k:float(wasserstein_distance(bv[k],bref[k]))for k in bv})
def migrations(meta):
    names=['original_bin_raw','target_bin_raw','preADC_bin_raw','ADC_bin_raw']
    joint=np.column_stack([meta[n]for n in names]);rows,counts=np.unique(joint,axis=0,return_counts=True)
    return dict(coordinates=names,N=len(joint),rows=[dict(bins=row.tolist(),count=int(n),weight=float(n/len(joint)))for row,n in zip(rows,counts)])

def postcheck(R,C,model,initial_model,source,real,source_ids,source_h,epsS,epsR,mu,scale,edges,p480,tau2,valley):
    ref=canonical(real);original=np.asarray(source,dtype='f8');original_q=q(original);original_p=psd(original);b0=bins(original_q,edges)
    outputs={};metadata={};metrics={}
    metrics['SOURCE_INPUT']=compare(original,ref,edges)
    for label,head in [('IDENTITY_CONTROL',initial_model),('LEARNED',model)]:
        for domain,x,e in [('SOURCE',source,epsS),('REAL',real,epsR)]:
            pre,adc,meta=generate(head,x,e,mu,scale,edges,p480,tau2,C['generation_chunk'])
            name=label+'_'+domain
            identities={'source_parent_ID':source_ids,'source_h':source_h}if domain=='SOURCE'else{'REAL_original_TRAIN_row':np.arange(len(real))}
            np.savez_compressed(R/(name+'_PRIVATE.npz'),pre_ADC_V=pre,**identities,**meta)
            outputs[name]=(pre,adc);metadata[name]=meta
            metrics[name]=compare(adc,ref,edges)
            metrics[name].update(Q_target_summary=summary(meta['Qstar']),Q_target_W1_mVns=float(wasserstein_distance(meta['Qstar']*1000,q(ref)*1000)),
                max_relative_Qstar_error_f64=float(meta['Q_relative_error_f64'].max()),max_relative_Qstar_error_f32=float(meta['Q_relative_error_f32'].max()),
                range_projection_fraction=float(meta['range_projection_active'].mean()),projection_RMS_V=summary(meta['projection_RMS_V']),
                same_public_input_path=True,domain_marker_or_identity_bypass=False)
            write(R/(name+'_MIGRATION.json'),migrations(meta))
            if domain=='SOURCE':
                p=psd(adc);metrics[name].update(source_Q_Spearman=float(spearmanr(original_q,q(adc)).statistic),
                    source_PSD_Spearman=float(spearmanr(original_p,p).statistic))
                metrics[name]['purehad_original_bin_valley']={}
                for j in[2,3]:
                    use=(source_h==1)&(b0==j);definition=valley['bins'][j]
                    metrics[name]['purehad_original_bin_valley'][str(j+1)]=dict(N=int(use.sum()),
                      input_fraction=float(np.mean((original_p[use]>=definition['low'])&(original_p[use]<=definition['high'])))if use.any()else None,
                      output_fraction=float(np.mean((p[use]>=definition['low'])&(p[use]<=definition['high'])))if use.any()else None)
    state={k:v.detach().cpu().numpy()for k,v in model.state_dict().items()};numeric={}
    for domain,x,e in [('SOURCE',source,epsS),('REAL',real,epsR)]:
        n=C['numeric_CPU_reference_rows_per_domain'];pre,adc,qt=reference(state,x[:n],e[:n],mu,scale,edges,p480,tau2)
        observed=outputs['LEARNED_'+domain][0][:n].astype('f8');rerr=np.sqrt(np.mean((observed-pre.astype('f8'))**2,1))*16383
        qerr=np.abs(qt-metadata['LEARNED_'+domain]['Qstar'][:n])/qt
        numeric[domain]=dict(N=n,median_RMS_LSB=float(np.median(rerr)),p95_RMS_LSB=float(np.quantile(rerr,.95)),max_RMS_LSB=float(rerr.max()),
            independent_Qstar_relative_error_max=float(qerr.max()),ADC_code_equal=bool(np.array_equal(adc,outputs['LEARNED_'+domain][1][:n])),
            pass_gate=bool(np.median(rerr)<=C['numeric_median_LSB_max']and np.quantile(rerr,.95)<=C['numeric_p95_LSB_max']and qerr.max()<=1e-5))
    # Serialization/RNG replay on fixed first2000 TRAIN rows per domain.
    replay_model=PublicConditionalChargeHead(torch.tensor(state['feature_center'],device='cuda'),torch.tensor(state['feature_scale'],device='cuda'))
    replay_model.load_state_dict(torch.load(R/'FINAL_HEAD.pt',map_location='cuda',weights_only=True));replay={}
    for domain,x,seed in [('SOURCE',source,C['source_epsilon_seed']),('REAL',real,C['real_epsilon_seed'])]:
        n=2000;epsilon=np.random.Generator(np.random.PCG64(seed)).standard_normal((n,480)).astype('f4')
        pre,adc,meta=generate(replay_model,x[:n],epsilon,mu,scale,edges,p480,tau2,C['generation_chunk'])
        replay[domain]=dict(N=n,preADC_bitwise=bool(np.array_equal(pre,outputs['LEARNED_'+domain][0][:n])),
            canonical_ADC_bitwise=bool(np.array_equal(adc,outputs['LEARNED_'+domain][1][:n])),
            Qstar_bitwise=bool(np.array_equal(meta['Qstar'],metadata['LEARNED_'+domain]['Qstar'][:n])))
    write(R/'TRAIN_METRICS.json',dict(scope='TRAIN-only charge component; no A-F claims or new charge pass threshold',metrics=metrics))
    write(R/'NUMERICAL_REFERENCE.json',dict(reference='Independent NumPy float64 full head/lifting/dense-covariance noise/Box/ADC',results=numeric))
    write(R/'TRAIN_SERIALIZATION_REPLAY.json',dict(scope='TRAIN4000 component replay; final held-out independent4000 gate remains required',results=replay))
    plot(R,outputs,metadata,original,ref,edges)
    passed=all(v['pass_gate']for v in numeric.values())and all(v['preADC_bitwise']and v['canonical_ADC_bitwise']and v['Qstar_bitwise']for v in replay.values())
    answer=dict(numerical_and_component_replay_pass=passed,TRAIN_rows_generated=2*(len(source)+len(real)),
        independent_numpy_rows=128,TRAIN_replay_rows=4000,heldout_arrays_read=False,metrics_file='TRAIN_METRICS.json',
        final_A_to_F_pass=None,final_heldout_4000_replay_complete=False)
    write(R/'POSTCHECK_RECEIPT.json',answer)
    if not passed:raise FloatingPointError('registered component numerical/reference/replay failed')
    return answer

def plot(R,outputs,metadata,original,ref,edges):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    distributions={'REAL reference':ref,'SOURCE input':original,**{k:v[1]for k,v in outputs.items()if k in ['IDENTITY_CONTROL_SOURCE','LEARNED_SOURCE','LEARNED_REAL']}}
    for name,v in distributions.items():
        charge=np.sort(q(v)*1000);axes[0,0].plot(charge,np.arange(1,len(charge)+1)/len(charge),label=name,lw=1.4)
    axes[0,0].set(xscale='log',xlabel='Canonical/window charge [mV ns]',ylabel='Empirical CDF',title='Full TRAIN parent laws');axes[0,0].legend(fontsize=8)
    m=metadata['LEARNED_SOURCE'];axes[0,1].scatter(m['Q0']*1000,m['QADC']*1000,s=3,alpha=.35)
    lo=min(m['Q0'].min(),m['QADC'].min())*1000;hi=max(m['Q0'].max(),m['QADC'].max())*1000
    axes[0,1].plot([lo,hi],[lo,hi],color='black',lw=.7);axes[0,1].set(xscale='log',yscale='log',xlabel='Original parent charge [mV ns]',ylabel='Canonical output charge [mV ns]',title='Every SOURCE parent retained')
    for name,v in [('REAL reference',ref),('LEARNED SOURCE',outputs['LEARNED_SOURCE'][1]),('LEARNED REAL',outputs['LEARNED_REAL'][1])]:
        axes[1,0].hist(psd(v),bins=np.linspace(-.1,.7,161),density=True,histtype='step',label=name)
        axes[1,1].plot(v.mean(0),label=name,lw=1.3)
    axes[1,0].set(xlabel='Signed PSD',ylabel='Density (plot range only)',title='PSD diagnostic; all rows used in metrics');axes[1,0].legend(fontsize=8)
    axes[1,1].set(xlabel='Sample in fixed480 window',ylabel='Mean voltage [V]',title='Mean full waveform');axes[1,1].legend(fontsize=8)
    fig.suptitle('R71 full-TRAIN charge component — A–F not evaluated',fontsize=14)
    fig.savefig(R/'CHARGE_COMPONENT_DIAGNOSTICS.png',dpi=150);plt.close(fig)
    with np.load(R/'LOSS_BY_UPDATE_PRIVATE.npz',allow_pickle=False)as z:loss=z['values'].copy()
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for j,label in enumerate(['total','charge distribution','REAL identity','source order']):ax.plot(np.arange(1,len(loss)+1),loss[:,j],label=label,lw=.8)
    ax.set(xlabel='Full-parent optimizer update',ylabel='Registered dimensionless loss',yscale='symlog',title='Unique50000-update endpoint; no checkpoint selection');ax.legend()
    fig.savefig(R/'FULL_TRAIN_LOSSES.png',dpi=150);plt.close(fig)
