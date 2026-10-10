from pathlib import Path
from datetime import datetime,timezone
import json,os,time,traceback,sys
import numpy as np
import scipy,sklearn
from scipy.stats import wasserstein_distance,spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import feature_definitions as fd
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text());START=time.monotonic();COUNT=0
EDGES=np.asarray(json.loads((R/'Q_DEFINITION.json').read_text())['edges_logQ'])
VALLEY=json.loads((R/'VALLEY_DEFINITION.json').read_text())['bins']
def utc():return datetime.now(timezone.utc).isoformat()
def safe(x):
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,np.ndarray):return safe(x.tolist())
 if isinstance(x,np.generic):return safe(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(name,x):(R/name).write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n','utf8')
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def summary(x):
 a=np.asarray(x);a=a[np.isfinite(a)]
 return {'N':len(x),'finite_N':len(a),'mean':float(a.mean()) if len(a) else None,'quantile_levels':[0,.05,.5,.95,1],'quantiles':np.quantile(a,[0,.05,.5,.95,1]) if len(a) else None}
def codes(v):return np.clip(np.rint((np.asarray(v,dtype='f8')+.5)*16383),0,16383).astype('i4')
def canonical(v):return -.5+codes(v).astype('f8')/16383.
def bins(q):
 z=np.log(np.maximum(np.asarray(q),1e-30));b=np.searchsorted(EDGES,z,side='right')-1;b[z==EDGES[-1]]=7;return b
def light(v):
 v=np.asarray(v,dtype='f8');x=v-v[:,:100].mean(1,keepdims=True);q=x[:,100:400].sum(1)*1000.
 return {'Q':q,'PSD':1-x[:,100:150].sum(1)*1000/q,'RMS':x[:,:100].std(1,ddof=1)}
def w1(a,b):
 a=np.asarray(a);b=np.asarray(b);a=a[np.isfinite(a)];b=b[np.isfinite(b)]
 return float(wasserstein_distance(a,b)) if len(a) and len(b) else None
def matrices(d,kind):
 if kind=='normalized_u480':return d['x']/np.maximum(d['x'].max(1,keepdims=True),.0025)
 return fd.matrix(d,kind)

def classify(domains,data,fresh):
 global COUNT
 r50=Path(C['server_R50_root']);old=load(r50/'PREDICTIONS_PRIVATE.npz');oldm=json.loads((r50/'METRICS.json').read_text())
 results={};private={};boot={};groupsets={};pred={};meta={}
 y=np.r_[np.ones(250),np.zeros(250)];rt,re=data['phase0_1'],data['phase1_1']
 for name in C['domains']:
  source=name.endswith('_SOURCE');tr,te=(fresh['phase0_0'],fresh['phase1_0']) if source else (data['phase0_2'],data['phase1_2'])
  dg=np.arange(2000) if source else data['reference_groups'];rg=data['real_groups']
  assert not set(dg[tr])&set(dg[te]) and not set(rg[rt])&set(rg[re])
  if not source:assert not set(dg)&set(rg)
  group=np.r_[[('G4_SOURCE|' if source else 'R|')+str(i) for i in dg[te]],['R|'+str(i) for i in rg[re]]]
  _,inv=np.unique(group,return_inverse=True);ng=int(inv.max()+1);rng=np.random.default_rng(C['bootstrap_seed'])
  weights=np.stack([np.bincount(rng.integers(ng,size=ng),minlength=ng)[inv] for _ in range(C['bootstrap_repeats'])])
  n0=weights[:,y==0].sum(1);n1=weights[:,y==1].sum(1);assert (n0>0).all() and (n1>0).all()
  for kind in C['canonical_views']:
   slot=name+'__'+kind;clock=time.monotonic();reuse=not source and name in ['RAW_REFERENCE','R49_FIXED_REAL','R49_COMPLETION_REAL'] and kind=='full_voltage'
   a=matrices(domains[name],kind);b=matrices(domains['REAL_DEV'],kind)
   if reuse:
    tag={'RAW_REFERENCE':'RAW_REFERENCE','R49_FIXED_REAL':'FIXED_POST','R49_COMPLETION_REAL':'COMPLETION_POST'}[name]+'__canonical_ADC_voltage'
    assert np.array_equal(old['label'],y) and np.array_equal(old['cluster'],group)
    p=old[tag];origin={'reused':True,'source':str(r50/'METRICS.json'),'source_slot':tag}
   else:
    X=np.vstack([a[tr],b[rt]]);T=np.vstack([a[te],b[re]]);yt=np.r_[np.ones(len(tr)),np.zeros(len(rt))]
    X[~np.isfinite(X)]=np.nan;T[~np.isfinite(T)]=np.nan
    im=SimpleImputer(strategy='median',keep_empty_features=True);X=im.fit_transform(X);T=im.transform(T)
    scaler=StandardScaler();X=scaler.fit_transform(X);T=scaler.transform(T)
    clf=HistGradientBoostingClassifier(**C['C2ST']);COUNT+=1;clf.fit(X,yt);p=clf.predict_proba(T)[:,1];origin={'reused':False}
   bs=.5*(weights[:,(y==0)&(p<.5)].sum(1)/n0+weights[:,(y==1)&(p>=.5)].sum(1)/n1);ci=np.quantile(bs,[.025,.975])
   row={'case':name,'view':kind,'BA':float(balanced_accuracy_score(y,p>=.5)),'BA_95CI':ci,'AUC':float(roc_auc_score(y,p)),
    'features':a.shape[1],'CI_inside_original_band':bool(ci[0]>=.4 and ci[1]<=.6),'elapsed_s':time.monotonic()-clock,**origin}
   if reuse:assert row['BA']==oldm['results'][tag]['BA'] and np.array_equal(ci,oldm['results'][tag]['BA_95CI'])
   results[slot]=row;boot[slot]=bs;groupsets[slot]=group;pred[slot]=p
   private[slot+'__probability']=p;private[slot+'__label']=y;private[slot+'__cluster']=group
   write('PROGRESS.json',{'PID':os.getpid(),'UTC':utc(),'new_HGB_fits':COUNT,'total_results':len(results),'planned_results':35,'last_slot':slot,'elapsed_s':time.monotonic()-START})
   print(json.dumps({'new_fit_count':COUNT,'slot':slot,'BA':row['BA'],'CI':ci.tolist(),'reused':reuse}),flush=True)
 assert COUNT==32 and len(results)==35
 pairs=[]
 for name in ['R49_FIXED_SOURCE','R49_COMPLETION_SOURCE','R49_FIXED_REAL','R49_COMPLETION_REAL']:
  control='BASE_SOURCE' if name.endswith('_SOURCE') else 'BASE_REAL'
  for kind in C['canonical_views']:
   a=name+'__'+kind;b=control+'__'+kind;assert np.array_equal(groupsets[a],groupsets[b])
   pairs.append({'candidate':name,'control':control,'view':kind,'BA_delta':results[a]['BA']-results[b]['BA'],
     'paired_BA_95CI':np.quantile(boot[a]-boot[b],[.025,.975]),'conditional_on_fixed_classifiers':True})
 np.savez_compressed(R/'CANONICAL_PREDICTIONS_PRIVATE.npz',**private)
 write('CANONICAL_C2ST.json',{'new_fits':COUNT,'reused_results':3,'results':results,'paired':pairs,
  'same_original_selection':True,'original_thresholds_unchanged':True,'original_R49_judgment_rewritten':False})
 return results

def endpoint_stats(domains,fresh):
 source=fd.features(fresh['SOURCE_INPUT']);sb=bins(source['features']['Qtotal_mVns']);real=domains['REAL_DEV'];rb=bins(real['features']['Qtotal_mVns']);out={}
 assert [int(np.sum((fresh['composition']=='pure_had')&(sb==b))) for b in [2,3]]==[213,181]
 for name in ['BASE_SOURCE','R49_FIXED_SOURCE','R49_COMPLETION_SOURCE']:
  d=domains[name];db=bins(d['features']['Qtotal_mVns']);rows=[]
  for b in range(8):
   rr=real['valid']&(rb==b);nom=(sb==b)&d['valid'];own=(db==b)&d['valid'];de=VALLEY[b]
   had=(fresh['composition']=='pure_had')&(sb==b);hp=d['features']['PSD'][had&d['valid']]
   rows.append({'bin':b+1,'source_nominal_N':int((sb==b).sum()),'output_own_N':int(own.sum()),'REAL_own_N':int(rr.sum()),
    'PSD_W1_own_Q':w1(d['features']['PSD'][own],real['features']['PSD'][rr]),
    'PSD_W1_source_nominal_Q':w1(d['features']['PSD'][nom],real['features']['PSD'][rr]),
    'baseline_RMS_W1_own_Q':w1(d['features']['baseline_RMS_V'][own],real['features']['baseline_RMS_V'][rr]),
    'ACF1_W1_own_Q':w1(d['features']['ACF1'][own],real['features']['ACF1'][rr]),
    'pure_had_N':int(had.sum()),'pure_had_valid_N':len(hp),'pure_had_valley_fraction':float(np.mean((hp>=de['low'])&(hp<=de['high']))) if len(hp) and de['valid'] else None})
  fixed=[r['PSD_W1_own_Q'] for r in rows[:7]]
  out[name]={'N':len(d['V']),'QC_valid_N':int(d['valid'].sum()),'source_support_outside_N':int(np.sum((sb<0)|(sb>7))),
   'PSD_W1_overall':w1(d['features']['PSD'][d['valid']],real['features']['PSD'][real['valid']]),
   'fixed_Q1_7_equal_PSD_W1':float(np.mean(fixed)) if all(x is not None for x in fixed) else None,
   'baseline_RMS_W1_overall':w1(d['features']['baseline_RMS_V'][d['valid']],real['features']['baseline_RMS_V'][real['valid']]),
   'ACF1_W1_overall':w1(d['features']['ACF1'][d['valid']],real['features']['ACF1'][real['valid']]),
   'source_Q_rank':float(spearmanr(source['features']['Qtotal_mVns'],d['features']['Qtotal_mVns']).statistic),
   'source_PSD_rank':float(spearmanr(source['features']['PSD'],d['features']['PSD']).statistic),
   'all_Q_bins':rows,'conditional_density_widths':'R49 Q3 nonconvergence retained; no new density fit'}
 write('SOURCE_METRICS.json',out);return out

def teacher_diagnostic():
 tr=load(C['TRAIN_path'],['source','target']);st=load(C['original_stream'],['source','matched_target','bin','original_row_ordinal'])
 source,target=light(tr['source']),light(tr['target']);si=st['source'].astype('i8');mi=st['matched_target'].astype('i8');ordinal=st['original_row_ordinal'];N=len(si)
 assert N==94400 and len(np.unique(ordinal))==N and np.all(source['Q']>0) and np.all(target['Q']>0)
 sbraw=bins(source['Q']);tbraw=bins(target['Q']);sb=np.clip(sbraw,0,7);tb=np.clip(tbraw,0,7);new=mi.copy()
 preserve=[]
 for b in range(8):
  ix=np.flatnonzero(sb[si]==b);order_s=ix[np.lexsort((ordinal[ix],source['PSD'][si[ix]]))];order_t=ix[np.lexsort((ordinal[ix],target['PSD'][mi[ix]]))]
  new[order_s]=mi[order_t]
  same=bool(np.array_equal(np.sort(mi[ix]),np.sort(new[ix])));assert same
  preserve.append({'bin':b+1,'occurrences':len(ix),'target_occurrence_multiset_equal':same})
 assert np.array_equal(np.sort(mi),np.sort(new))
 def parent_stats(target_indices):
  vals=target['PSD'][target_indices];n=np.bincount(si,minlength=len(source['Q']));su=np.bincount(si,weights=vals,minlength=len(n));ss=np.bincount(si,weights=vals*vals,minlength=len(n))
  mean=np.divide(su,n,out=np.full(len(n),np.nan),where=n>0)
  var=np.divide(ss-su*mean,n-1,out=np.full(len(n),np.nan),where=n>1)
  if np.any(var[np.isfinite(var)] < -1e-12):raise FloatingPointError('negative parental sample variance beyond roundoff')
  std=np.sqrt(np.maximum(var,0));minimum=np.full(len(n),np.inf);maximum=np.full(len(n),-np.inf)
  np.minimum.at(minimum,si,vals);np.maximum.at(maximum,si,vals)
  return {'count':n,'mean':mean,'std':std,'min':minimum,'max':maximum}
 pa,pb=parent_stats(mi),parent_stats(new);rows=[];hits=0;max_q=max_ps=max_rms=0.
 # Verify the existing rescaled-target definition on all original occurrences, in small chunks.
 ratio=source['Q'][si]/target['Q'][mi]
 for lo in range(0,N,1024):
  ix=slice(lo,min(lo+1024,N));v=tr['target'][mi[ix]].astype('f8')*ratio[ix,None];f=light(v)
  max_q=max(max_q,float(np.max(np.abs(f['Q']-source['Q'][si[ix]])/source['Q'][si[ix]])))
  max_ps=max(max_ps,float(np.max(np.abs(f['PSD']-target['PSD'][mi[ix]]))))
  max_rms=max(max_rms,float(np.max(np.abs(f['RMS']-target['RMS'][mi[ix]]*ratio[ix]))))
 for b in range(8):
  ix=sb[si]==b;rr=tbraw==b;parents=(sb==b)&(pa['count']>=2);de=VALLEY[b]
  a=pa['std'][parents];bb=pb['std'][parents];ma=float(np.median(a)) if len(a) else None;mb=float(np.median(bb)) if len(bb) else None
  relative=mb/ma if ma is not None and ma>0 else None
  hit=bool(b in range(1,7) and len(a)>=20 and relative is not None and relative<=.5)
  hits+=int(hit)
  row={'bin':b+1,'occurrences':int(ix.sum()),'REAL_TRAIN_own_bin_N':int(rr.sum()),'repeated_source_parents':int(parents.sum()),
   'original_parent_target_PSD_std':summary(a),'global_parent_target_PSD_std':summary(bb),'median_std_ratio_global_over_original':relative,'direction_hit':hit,
   'original_parent_straddle_fraction':float(np.mean((pa['min'][parents]<de['low'])&(pa['max'][parents]>de['high']))) if len(a) and de['valid'] else None,
   'global_parent_straddle_fraction':float(np.mean((pb['min'][parents]<de['low'])&(pb['max'][parents]>de['high']))) if len(a) and de['valid'] else None}
  for tag,indices in [('original',mi),('global_rank',new)]:
   ti=indices[ix];scale=source['Q'][si[ix]]/target['Q'][ti]
   row[tag]={'source_to_target_charge_ratio':summary(scale),
    'teacher_PSD_W1_vs_REAL_TRAIN_own_Q':w1(target['PSD'][ti],target['PSD'][rr]),
    'unscaled_target_RMS_W1_vs_REAL_TRAIN_own_Q':w1(target['RMS'][ti],target['RMS'][rr]),
    'rescaled_teacher_RMS_W1_vs_REAL_TRAIN_own_Q':w1(target['RMS'][ti]*scale,target['RMS'][rr]),
    'teacher_RMS_scaled':summary(target['RMS'][ti]*scale),
    'source_target_PSD_rank':float(spearmanr(source['PSD'][si[ix]],target['PSD'][ti]).statistic) if ix.sum()>2 else None}
  rows.append(row)
 counts={'loaded_SOURCE':len(source['Q']),'loaded_REAL':len(target['Q']),'occurrences':N,'unique_SOURCE':len(np.unique(si)),'unique_REAL':len(np.unique(mi)),
  'original_bin_vs_actual_source_mismatch':int(np.count_nonzero(st['bin']!=sb[si])),
  'original_matched_actual_Q_bin_mismatch':int(np.count_nonzero(sb[si]!=tb[mi])),'global_matched_actual_Q_bin_mismatch':int(np.count_nonzero(sb[si]!=tb[new])),
  'SOURCE_all_TRAIN_below_above_range':[int(np.sum(sbraw<0)),int(np.sum(sbraw>7))],'REAL_all_TRAIN_below_above_range':[int(np.sum(tbraw<0)),int(np.sum(tbraw>7))]}
 result={'counts':counts,'preserved_per_condition_multisets':preserve,'target_occurrence_multiset_globally_equal':True,'source_order_and_occurrences_unchanged':True,
  'target_scaling_identity_checks':{'max_relative_q_error':max_q,'max_PSD_change':max_ps,'max_baseline_RMS_scaling_error_V':max_rms},
  'by_bin':rows,'direction_hits_bins2_7':hits,'required_hits':4,'direction_supported':hits>=4,
  'decision':'support_reducing_minibatch_coupling_conditional_dispersion' if hits>=4 else 'close_global_rank_conditional_dispersion_explanation',
  'target_noise_error_not_fixed_by_rank_alone':True,'physical_pairing_truth':False,'new_response_training':0}
 assert max_q<1e-10 and max_ps<1e-10 and max_rms<1e-12
 np.savez_compressed(R/'GLOBAL_RANK_COUPLING_PRIVATE.npz',source=si,matched_target_original=mi,matched_target_global=new,original_row_ordinal=ordinal,actual_condition_bin=sb[si])
 np.savez_compressed(R/'PARENT_TARGET_PSD_PRIVATE.npz',source_Q_mVns=source['Q'],source_PSD=source['PSD'],source_bin=sb,**{'original_'+k:v for k,v in pa.items()},**{'global_'+k:v for k,v in pb.items()})
 write('TEACHER_DIAGNOSTIC.json',result);return result

def plot(results,stats,teacher,domains):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=R/'figures';out.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'figure.dpi':140})
 values=np.array([[results[n+'__'+k]['BA'] for k in C['canonical_views']] for n in C['domains']])
 fig,ax=plt.subplots(figsize=(10.3,6.0));im=ax.imshow(values,vmin=.4,vmax=1,cmap='YlOrRd')
 for i,n in enumerate(C['domains']):
  for j,k in enumerate(C['canonical_views']):ax.text(j,i,f'{values[i,j]:.3f}'+(' *' if results[n+'__'+k]['CI_inside_original_band'] else ''),ha='center',va='center',color='black')
 ax.set(xticks=range(5),xticklabels=['Shape493','u480','Voltage480','Q / PSD','Baseline'],yticks=range(7),yticklabels=C['domains'],title='R51 common ADC representation: balanced accuracy\n* 95% group CI inside [0.40, 0.60]')
 fig.colorbar(im,ax=ax,label='Balanced accuracy');fig.tight_layout();fig.savefig(out/'canonical_five_views.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.2))
 for name,color in [('BASE_SOURCE','#444444'),('R49_FIXED_SOURCE','#bd6b28'),('R49_COMPLETION_SOURCE','#176b9c')]:
  rows=stats[name]['all_Q_bins']
  axes[0].plot(range(1,9),[r['PSD_W1_own_Q'] for r in rows],'o-',label=name,color=color)
  axes[1].plot(range(1,9),[r['baseline_RMS_W1_own_Q']*1000 if r['baseline_RMS_W1_own_Q'] is not None else np.nan for r in rows],'o-',label=name,color=color)
 axes[0].axhline(.01,color='gray',ls=':');axes[0].set(ylabel='Signed PSD W1',xlabel='Original Q bin',title='Output-own Q conditions')
 axes[1].set(ylabel='Baseline RMS W1 [mV]',xlabel='Original Q bin',title='Absolute noise discrepancy');axes[1].legend(fontsize=7)
 for ax in axes:ax.grid(alpha=.2);ax.set_xticks(range(1,9))
 fig.tight_layout();fig.savefig(out/'source_conditional_PSD_noise.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11.4,4.4));rows=teacher['by_bin'];x=np.arange(1,9)
 for key,label,color in [('original_parent_target_PSD_std','Original minibatch coupling','#bd6b28'),('global_parent_target_PSD_std','Global rank coupling','#176b9c')]:
  axes[0].plot(x,[r[key]['quantiles'][2] if r[key]['quantiles'] is not None else np.nan for r in rows],'o-',label=label,color=color)
 for key,label,color in [('unscaled_target_RMS_W1_vs_REAL_TRAIN_own_Q','Consumed target RMS','#444444'),('rescaled_teacher_RMS_W1_vs_REAL_TRAIN_own_Q','After source-Q rescaling','#bd6b28')]:
  axes[1].plot(x,[r['original'][key]*1000 if r['original'][key] is not None else np.nan for r in rows],'o-',label=label,color=color)
 axes[0].set(xlabel='Original Q bin',ylabel='Median within-parent target PSD SD',title='Same source and target occurrence marginals')
 axes[1].set(xlabel='Original Q bin',ylabel='Baseline RMS W1 [mV]',title='TRAIN target construction')
 for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xticks(x)
 fig.tight_layout();fig.savefig(out/'teacher_coupling_and_noise.png');plt.close(fig)

def run():
 r50=Path(C['server_R50_root']);data=load(r50/'INPUT_PRIVATE.npz');fresh=load(R/'SOURCE_INPUT_PRIVATE.npz')
 saved={n:fresh[n] for n in ['BASE_SOURCE','R49_FIXED_SOURCE','R49_COMPLETION_SOURCE','BASE_REAL']}
 saved.update(R49_FIXED_REAL=data['FIXED_POST'],R49_COMPLETION_REAL=data['COMPLETION_POST'],RAW_REFERENCE=data['reference_V'],REAL_DEV=data['real_V'])
 domains={};preserved={}
 for name,v in saved.items():
  cv=canonical(v);preserved[name]=bool(np.array_equal(codes(v),codes(cv)));assert preserved[name]
  domains[name]=fd.features(cv)
 write('REPRESENTATION_BINDING.json',{'code_preservation':preserved,'common_operator_all_domains':True,
  'feature_definition_source':'verbatim R49 AST functions','EVAL_read':0,'CAL_decode':0,'source_population_N':2000,'source_weights_unchanged':True,
  'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}})
 results=classify(domains,data,fresh);stats=endpoint_stats(domains,fresh);teacher=teacher_diagnostic();plot(results,stats,teacher,domains)
 write('NUMERIC_RECEIPT.json',{'UTC':utc(),'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'new_HGB_fits':COUNT,'reused_HGB_results':3,
  'teacher_occurrences':94400,'new_response_training_runs':0,'new_generation_rows':0,'density_fits':0,'EVAL_read':0,'CAL_decode':0,
  'global_coupling_direction_hits':teacher['direction_hits_bins2_7'],'global_coupling_direction_supported':teacher['direction_supported'],
  'model_goal_complete':False,'figures':sorted(str(p.relative_to(R)) for p in (R/'figures').glob('*.png'))})
try:run()
except BaseException as exc:
 write('FAILURE.json',{'UTC':utc(),'PID':os.getpid(),'new_HGB_attempts':COUNT,'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START})
 raise
