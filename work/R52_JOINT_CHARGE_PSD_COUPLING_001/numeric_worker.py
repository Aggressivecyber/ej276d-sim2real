from pathlib import Path
from datetime import datetime,timezone
import json,os,time,traceback,sys
import numpy as np,scipy
from scipy.stats import wasserstein_distance,spearmanr
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text());START=time.monotonic()
EDGES=np.asarray(json.loads((R/'Q_DEFINITION.json').read_text())['edges_logQ'])
def utc():return datetime.now(timezone.utc).isoformat()
def safe(x):
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,np.ndarray):return safe(x.tolist())
 if isinstance(x,np.generic):return safe(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(n,x):(R/n).write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n','utf8')
def load(p,keys):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def light(v):
 v=np.asarray(v,dtype='f8');x=v-v[:,:100].mean(1,keepdims=True);q=x[:,100:400].sum(1)*1000
 return {'Q':q,'PSD':1-x[:,100:150].sum(1)*1000/q,'RMS':x[:,:100].std(1,ddof=1)}
def bins(q):
 z=np.log(np.maximum(q,1e-30));b=np.searchsorted(EDGES,z,side='right')-1;b[z==EDGES[-1]]=7;return b
def summary(a):
 a=np.asarray(a);f=a[np.isfinite(a)]
 return {'N':len(a),'finite_N':len(f),'mean':float(f.mean()) if len(f) else None,'quantiles':np.quantile(f,[0,.05,.5,.95,1]) if len(f) else None,'quantile_levels':[0,.05,.5,.95,1]}
def w1(a,b):return float(wasserstein_distance(a,b)) if len(a) and len(b) else None
def parent_std(si,vals,size):
 n=np.bincount(si,minlength=size);su=np.bincount(si,weights=vals,minlength=size);ss=np.bincount(si,weights=vals*vals,minlength=size)
 mean=np.divide(su,n,out=np.full(size,np.nan),where=n>0);var=np.divide(ss-su*mean,n-1,out=np.full(size,np.nan),where=n>1)
 assert not np.any(var[np.isfinite(var)] < -1e-12)
 return n,np.sqrt(np.maximum(var,0))
def block(name,N,source,target,stream):
 si=stream['source'][:N].astype('i8');mi=stream['matched_target'][:N].astype('i8');ordinal=stream['original_row_ordinal'][:N]
 sbraw=bins(source['Q']);tbraw=bins(target['Q']);sb=np.clip(sbraw,0,7);tb=np.clip(tbraw,0,7)
 assert np.array_equal(stream['bin'][:N],sb[si]) and np.array_equal(sb[si],tb[mi]) and len(np.unique(ordinal))==N
 new=mi.copy();group_id=np.full(N,-1,dtype='i4');preserve=[]
 for b in range(8):
  ix=np.flatnonzero(sb[si]==b)
  osrc=ix[np.lexsort((ordinal[ix],si[ix],source['Q'][si[ix]]))]
  otgt=ix[np.lexsort((ordinal[ix],mi[ix],target['Q'][mi[ix]]))]
  for g,(sr,tr) in enumerate(zip(np.array_split(osrc,C['charge_quantile_groups']),np.array_split(otgt,C['charge_quantile_groups']))):
   assert len(sr)==len(tr)
   sr=sr[np.lexsort((ordinal[sr],si[sr],source['PSD'][si[sr]]))]
   tr=tr[np.lexsort((ordinal[tr],mi[tr],target['PSD'][mi[tr]]))]
   new[sr]=mi[tr];group_id[sr]=g
  assert np.array_equal(np.sort(mi[ix]),np.sort(new[ix]))
  assert np.array_equal(np.sort(target['PSD'][mi[ix]]),np.sort(target['PSD'][new[ix]]))
  preserve.append({'bin':b+1,'N':len(ix),'target_occurrence_multiset_equal':True,'target_PSD_distribution_exactly_equal':True})
 assert np.array_equal(np.sort(mi),np.sort(new)) and np.all(group_id>=0) and np.array_equal(tb[new],sb[si])
 pn,oldstd=parent_std(si,target['PSD'][mi],len(source['Q']));nn,newstd=parent_std(si,target['PSD'][new],len(source['Q']));assert np.array_equal(pn,nn)
 rows=[];hits=0
 for b in range(8):
  ix=sb[si]==b;rr=tbraw==b;parents=(sb==b)&(pn>=2)
  orig=float(np.median(oldstd[parents]));joint=float(np.median(newstd[parents]));ratio=joint/orig if orig>0 else None
  hit=bool(b in range(1,7) and parents.sum()>=C['min_repeated_parents'] and ratio is not None and ratio<=C['PSD_parent_SD_ratio_max']);hits+=int(hit)
  row={'bin':b+1,'occurrences':int(ix.sum()),'REAL_TRAIN_own_bin_N':int(rr.sum()),'repeated_source_parents':int(parents.sum()),
   'original_parent_target_PSD_SD':summary(oldstd[parents]),'joint_parent_target_PSD_SD':summary(newstd[parents]),'median_SD_ratio_joint_over_original':ratio,'PSD_direction_hit':hit}
  for tag,indices in [('original',mi),('joint',new)]:
   ti=indices[ix];scale=source['Q'][si[ix]]/target['Q'][ti]
   row[tag]={'source_to_target_Q_ratio':summary(scale),'target_PSD_W1':w1(target['PSD'][ti],target['PSD'][rr]),
    'unscaled_target_RMS_W1_V':w1(target['RMS'][ti],target['RMS'][rr]),'scaled_teacher_RMS_W1_V':w1(target['RMS'][ti]*scale,target['RMS'][rr]),
    'source_target_PSD_rank':float(spearmanr(source['PSD'][si[ix]],target['PSD'][ti]).statistic),
    'target_RMS_V':summary(target['RMS'][ti]),'scaled_teacher_RMS_V':summary(target['RMS'][ti]*scale)}
  assert abs(row['joint']['target_PSD_W1']-row['original']['target_PSD_W1'])<1e-14
  assert abs(row['joint']['unscaled_target_RMS_W1_V']-row['original']['unscaled_target_RMS_W1_V'])<1e-14
  row['noise_W1_limit_V']=row['joint']['unscaled_target_RMS_W1_V']+C['extra_RMS_W1_tolerance_V']
  row['noise_pass']=row['joint']['scaled_teacher_RMS_W1_V']<=row['noise_W1_limit_V'];rows.append(row)
 result={'block':name,'occurrences':N,'unique_source':len(np.unique(si)),'unique_target':len(np.unique(mi)),'source_order_unchanged':True,'per_bin_preservation':preserve,
  'target_occurrence_multiset_globally_equal':True,'by_bin':rows,'PSD_direction_hits_bins2_7':hits,'PSD_direction_supported':hits>=C['required_PSD_direction_hits_bins2_7'],
  'noise_all8_pass':all(r['noise_pass'] for r in rows),'noise_failed_bins':[r['bin'] for r in rows if not r['noise_pass']],
  'target_occurrences_changed_position':int(np.count_nonzero(new!=mi)),'physical_pairing_truth':False}
 result['joint_construction_pass']=result['PSD_direction_supported'] and result['noise_all8_pass']
 np.savez_compressed(R/(name+'_COUPLING_PRIVATE.npz'),source=si,matched_target_original=mi,matched_target_joint=new,original_row_ordinal=ordinal,condition_bin=sb[si],Q_quantile_group=group_id)
 np.savez_compressed(R/(name+'_PARENT_PRIVATE.npz'),source_Q_mVns=source['Q'],source_PSD=source['PSD'],source_bin=sb,parent_occurrences=pn,original_target_PSD_SD=oldstd,joint_target_PSD_SD=newstd)
 write(name+'_METRICS.json',result);return result
def plot(results):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=R/'figures';out.mkdir(exist_ok=True);fig,axes=plt.subplots(2,2,figsize=(12.0,8.2),layout='constrained');x=np.arange(1,9)
 for i,(name,result) in enumerate(results.items()):
  rows=result['by_bin'];left,right=axes[i]
  left.plot(x,[r['original']['scaled_teacher_RMS_W1_V']*1000 for r in rows],'o-',label='Original coupling',color='#bd6b28')
  left.plot(x,[r['joint']['scaled_teacher_RMS_W1_V']*1000 for r in rows],'o-',label='Joint Q / PSD coupling',color='#176b9c')
  left.plot(x,[r['noise_W1_limit_V']*1000 for r in rows],'--',label='Unscaled target + 0.1 LSB',color='#555555')
  left.set(title=name+' : target noise',ylabel='Baseline RMS W1 [mV]')
  right.plot(x,[r['median_SD_ratio_joint_over_original'] for r in rows],'o-',color='#176b9c',label='Joint / original median SD')
  right.axhline(.5,ls='--',color='#555555',label='Predeclared PSD criterion')
  right.set(title=name+' : parent target PSD',ylabel='Within-parent SD ratio',ylim=(0,max(.6,max(r['median_SD_ratio_joint_over_original'] for r in rows)*1.1)))
  for ax in [left,right]:ax.set_xlabel('Original Q bin');ax.set_xticks(x);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.savefig(out/'joint_target_checks.png',dpi=160);plt.close(fig)
def run():
 tr=load(C['TRAIN_path'],['source','target']);source,target=light(tr['source']),light(tr['target']);del tr
 assert np.all(source['Q']>0) and np.all(target['Q']>0)
 stream=load(C['original_stream'],['source','matched_target','original_row_ordinal','bin']);assert len(stream['source'])==94400
 results={name:block(name,N,source,target,stream) for name,N in C['blocks'].items()};plot(results)
 passed=all(r['joint_construction_pass'] for r in results.values())
 write('JOINT_DECISION.json',{'UTC':utc(),'all_blocks_pass':passed,'decision':'support_one_preregistered_joint_coupling_response_training' if passed else 'close_fixed_16_charge_quantile_coupling',
  'blocks':{k:{f:v[f] for f in ['occurrences','PSD_direction_hits_bins2_7','PSD_direction_supported','noise_all8_pass','noise_failed_bins','joint_construction_pass']} for k,v in results.items()},
  'new_training_started':False,'model_goal_complete':False,'group_count_scanned':False,'group_count':16,
  'training_exposure_rule':'16 repeats of full-cycle mapping, then independently constructed final-partial mapping; original target indices remain the REAL identity sequence.'})
 write('NUMERIC_RECEIPT.json',{'UTC':utc(),'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'new_response_training':0,'HGB_fits':0,'density_fits':0,'DEV_read':0,'CAL_decode':0,'EVAL_read':0,
  'all_blocks_pass':passed,'model_goal_complete':False,'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__}})
 print(json.dumps({'all_blocks_pass':passed,'blocks':{k:{f:v[f] for f in ['PSD_direction_hits_bins2_7','noise_all8_pass','noise_failed_bins']} for k,v in results.items()}}),flush=True)
try:run()
except BaseException as exc:
 write('FAILURE.json',{'UTC':utc(),'PID':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START});raise
