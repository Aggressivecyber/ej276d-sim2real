from pathlib import Path
from datetime import datetime,timezone
import json, numpy as np
from scipy.stats import wasserstein_distance
from noise_core import q_np,bins_for

def short_q(v):
 a=np.asarray(v,dtype='f8');return a[...,100:150].sum(-1)-.5*a[...,:100].sum(-1)
def basis():
 embed=np.eye(480,dtype='f8')[:,100:400]
 D=np.diff(np.pad(embed,((2,2),(0,0))),n=2,axis=0);K=D.T@D
 A=np.zeros((2,300));A[0,:50]=1;A[1,50:]=1
 Z=np.linalg.solve(K,A.T);B=Z@np.linalg.inv(A@Z)
 out=np.zeros((480,2));out[100:400]=B
 residual=float(np.max(np.abs(A@B-np.eye(2))))
 assert residual<1e-10 and np.isfinite(out).all() and not np.any(out[:100]) and not np.any(out[400:])
 return out,{'pulse_support':[100,400],'geometric_only':True,'data_fitted':False,'constraint_residual_max':residual,'penalty':'zero_extended_second_difference_squared','tuning_parameters':0}
def endpoint(real,source_q,B):
 r=np.asarray(real,dtype='f8');q=q_np(r);s=short_q(r);a=np.asarray(source_q)/q-1
 coefficients=np.stack((a*s,a*(q-s)),1)
 return r+coefficients@B.T
def summary(x):
 x=np.asarray(x,dtype='f8');assert np.isfinite(x).all()
 return {'N':int(x.size),'mean':float(x.mean()),'quantiles':np.quantile(x,[0,.05,.5,.95,1]).tolist(),'quantile_levels':[0,.05,.5,.95,1]}
def validate(root,cfg):
 root=Path(root);B,definition=basis();np.savez_compressed(root/'PULSE_BASIS.npz',B=B)
 with np.load(cfg['TRAIN_path'],allow_pickle=False) as z:source=z['source'].copy();real=z['target'].copy()
 with np.load(cfg['constants_path'],allow_pickle=False) as z:edges=z['edges_logQ'].copy()
 sb,sraw,sq=bins_for(source,edges);rb,rraw,rq=bins_for(real,edges);rs=short_q(real);rp=1-rs/rq
 rms=np.std(real[:,:100].astype('f8'),axis=1,ddof=1);reports={};example={};maxq=maxps=0.;all_pass=True
 for name,path in [('full_cycle',cfg['full_coupling_path']),('final_partial_cycle',cfg['partial_coupling_path'])]:
  with np.load(path,allow_pickle=False) as z:si=z['source'].astype('i8');mi=z['matched_target_joint'].astype('i8');old=z['matched_target_original'].astype('i8')
  assert np.array_equal(np.sort(mi),np.sort(old)) and np.array_equal(sb[si],rb[mi])
  relq=[];pd=[];rmsdelta=[];maxdelta=[];curvedelta=[];boundaries=[];new_clipped=0;baseline_tail_exact=True
  for lo in range(0,len(si),1024):
   sl=slice(lo,min(lo+1024,len(si)));rr=real[mi[sl]].astype('f8');t=endpoint(rr,sq[si[sl]],B);delta=t-rr
   q=q_np(t);psd=1-short_q(t)/q
   relq.append(np.abs(q-sq[si[sl]])/sq[si[sl]]);pd.append(np.abs(psd-rp[mi[sl]]))
   baseline_tail_exact &= np.array_equal(t[:,:100],rr[:,:100]) and np.array_equal(t[:,400:],rr[:,400:])
   rmsdelta.append(np.sqrt(np.mean(delta*delta,axis=1)));maxdelta.append(np.max(np.abs(delta),axis=1));curvedelta.append(np.max(np.abs(np.diff(delta,n=2,axis=1)),axis=1))
   boundaries.append(np.stack([delta[:,100]-delta[:,99],delta[:,150]-delta[:,149],delta[:,400]-delta[:,399]],1))
   new_clipped+=int(np.sum(np.any((t<-.5)|(t>.5),axis=1)&~np.any((rr<-.5)|(rr>.5),axis=1)))
   assert np.isfinite(t).all()
  relq=np.concatenate(relq);pd=np.concatenate(pd);rmsdelta=np.concatenate(rmsdelta);maxdelta=np.concatenate(maxdelta);curvedelta=np.concatenate(curvedelta);boundaries=np.concatenate(boundaries)
  maxq=max(maxq,float(relq.max()));maxps=max(maxps,float(pd.max()));rows=[]
  for b in range(8):
   ix=sb[si]==b;rr=rraw==b;inds=mi[ix];rat=sq[si[ix]]/rq[inds]
   assert np.array_equal(np.sort(mi[ix]),np.sort(old[ix]))
   noise=float(wasserstein_distance(rms[inds],rms[rr]));oldnoise=float(wasserstein_distance(rms[inds]*rat,rms[rr]))
   rows.append({'bin':b+1,'occurrences':int(ix.sum()),'target_PSD_W1_unchanged':float(wasserstein_distance(rp[inds],rp[rr])),
    'old_scaled_RMS_W1_V':oldnoise,'pulse_only_RMS_W1_V':noise,'unscaled_target_RMS_W1_V':noise,'baseline_noise_identity_exact':baseline_tail_exact,
    'waveform_correction_RMS_V':summary(rmsdelta[ix]),'max_sample_correction_V':summary(maxdelta[ix]),'max_second_difference_correction_V':summary(curvedelta[ix]),
    'max_absolute_fixed_boundary_first_correction_V':np.max(np.abs(boundaries[ix]),axis=0).tolist(),'source_target_charge_ratio':summary(rat)})
   if name=='full_cycle':
    pos=int(np.flatnonzero(ix)[0]);example[str(b+1)]={'ordinal':pos,'source_q':sq[si[pos]],'real':real[mi[pos]].astype('f8'),'whole_scaled':real[mi[pos]].astype('f8')*(sq[si[pos]]/rq[mi[pos]]),'pulse_only':endpoint(real[mi[pos]:mi[pos]+1],sq[si[pos]:si[pos]+1],B)[0]}
  passed=bool(relq.max()<=cfg['teacher_relative_Q_error_limit'] and pd.max()<=cfg['teacher_PSD_error_limit'] and baseline_tail_exact and new_clipped==0)
  reports[name]={'occurrences':len(si),'max_relative_q_error':float(relq.max()),'max_PSD_change':float(pd.max()),'baseline_tail_bitwise_equal':bool(baseline_tail_exact),'new_ADC_range_exceeded_rows':new_clipped,'all_finite':True,'passed':passed,'by_bin':rows}
  all_pass &= passed
  np.savez_compressed(root/(name+'_TEACHER_DESCRIPTIONS_PRIVATE.npz'),source=si,matched_target=mi,relative_q_error=relq,PSD_error=pd,correction_RMS_V=rmsdelta,max_correction_V=maxdelta,max_second_difference_correction_V=curvedelta,boundary_first_corrections_V=boundaries)
 out={'UTC':datetime.now(timezone.utc).isoformat(),'definition':definition,'blocks':reports,'passed':bool(all_pass),'max_relative_q_error':maxq,'max_PSD_change':maxps,'DEV_read':0,'EVAL_read':0,'CAL_decode':0,'response_training_started':False,'physical_correctness_claim':False}
 (root/'TEACHER_VALIDATION.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n','utf8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 f=root/'figures';f.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained');rows=reports['full_cycle']['by_bin'];x=np.arange(1,9)
 axes[0].plot(x,[v['old_scaled_RMS_W1_V']*1000 for v in rows],'o-',label='Whole-waveform charge scaling',color='#bd6b28')
 axes[0].plot(x,[v['pulse_only_RMS_W1_V']*1000 for v in rows],'o-',label='Pulse-only = unscaled target',color='#176b9c')
 axes[0].set(xlabel='Original Q bin',ylabel='Baseline RMS W1 [mV]',title='Fixed joint coupling; target endpoint change');axes[0].set_xticks(x);axes[0].legend(fontsize=8)
 axes[1].plot(B[:,0],label='Unit short-region integral');axes[1].plot(B[:,1],label='Unit long-region integral');axes[1].set(xlabel='Sample',ylabel='Pulse correction basis',title='Unique geometric minimum-curvature basis');axes[1].legend(fontsize=8)
 for ax in axes:ax.grid(alpha=.2)
 fig.savefig(f/'teacher_noise_and_basis.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(4,2,figsize=(12,10),layout='constrained')
 for ax,(b,e) in zip(axes.flat,example.items()):
  for k,label,color in [('real','Original measured target','#444444'),('whole_scaled','Previous whole scaling','#bd6b28'),('pulse_only','New pulse-only target','#176b9c')]:ax.plot(e[k]*1000,label=label,color=color,lw=.9)
  ax.set(title='Q'+b+' : first fixed occurrence '+str(e['ordinal']),xlabel='Sample',ylabel='Voltage [mV]');ax.grid(alpha=.2)
 axes.flat[0].legend(fontsize=7);fig.savefig(f/'teacher_fixed_examples.png',dpi=150);plt.close(fig)
 return out,B
