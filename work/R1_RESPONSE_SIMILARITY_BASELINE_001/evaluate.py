from __future__ import annotations
import os,pathlib,time,json,datetime,hashlib,io,traceback
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='4'
ROOT=pathlib.Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime_cache'/'matplotlib')
os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'runtime_cache'/'joblib')
START=time.monotonic()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from scipy.stats import spearmanr
from common_math import distances,posterior,fom
CFG=json.loads((ROOT/'config.json').read_text())
OUT=ROOT/os.environ.get('R1_RUN_DIR','run_001')/'results';OUT.mkdir(parents=True,exist_ok=False)
FIG=OUT/'figures';FIG.mkdir()
NAMES=['SIM','GEN_FIXED','GEN_W0','REAL'];COLORS={'SIM':'#2878b5','GEN_FIXED':'#eb8b23','GEN_W0':'#a33ea2','REAL':'#181818'}
INPUTS=[]

def log(msg):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),msg,flush=True)
def safe(x):
 if isinstance(x,np.ndarray):return safe(x.tolist())
 if isinstance(x,np.generic):return safe(x.item())
 if isinstance(x,float):return x if np.isfinite(x) else None
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [safe(v) for v in x]
 return x
def write_json(n,obj):(OUT/n).write_text(json.dumps(safe(obj),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def sha(b):return hashlib.sha256(b).hexdigest()
def hashint(s):return int.from_bytes(hashlib.sha256(str(s).encode()).digest()[:8],'big')
def order(ids,prefix):return np.argsort(np.array([hashint(prefix+'|'+str(s)) for s in ids],dtype=np.uint64),kind='stable')
def read_npz(path):
 p=pathlib.Path(path);b=p.read_bytes();INPUTS.append({'path':str(p),'bytes':len(b),'sha256':sha(b),'read_scope':'whole_existing_DEV2000_container'})
 return np.load(io.BytesIO(b),allow_pickle=False)
def read_prefix(path,dtype,count):
 p=pathlib.Path(path);size=np.dtype(dtype).itemsize*count
 with p.open('rb') as f:b=f.read(size)
 if len(b)!=size:raise ValueError('prefix length '+str(p))
 INPUTS.append({'path':str(p),'file_bytes':p.stat().st_size,'prefix_read_bytes':size,'prefix_sha256':sha(b),'read_scope':'first2000DEV;noTRAIN,CAL,EVAL waveform'})
 return np.frombuffer(b,dtype=dtype,count=count).copy()
def summary(v):
 a=np.asarray(v);x=a[np.isfinite(a)]
 return {'n':len(a),'defined_n':len(x),'undefined_n':len(a)-len(x),'mean':float(x.mean()) if len(x) else None,'sample_sd':float(x.std(ddof=1)) if len(x)>1 else None,'quantile_levels':[0,.05,.25,.5,.75,.95,1],'quantiles':np.quantile(x,[0,.05,.25,.5,.75,.95,1]).tolist() if len(x) else None}
def crossing(x,peak,frac):
 if peak<=100:return np.nan
 target=x[peak]*frac;s=x[100:peak+1];idx=np.flatnonzero((s[:-1]<=target)&(s[1:]>=target)&(s[1:]>s[:-1]))
 if not len(idx):return np.nan
 i=int(idx[-1])+100;return i+(target-x[i])/(x[i+1]-x[i])
def features(v):
 v=np.asarray(v,dtype=np.float64);b=v[:,:100].mean(axis=1);x=v-b[:,None];peak=x.max(axis=1);pi=x.argmax(axis=1)
 qt=x[:,100:400].sum(axis=1)*1000;qf=x[:,100:150].sum(axis=1)*1000;ps=np.divide(qf,qt,out=np.full(len(v),np.nan),where=qt>0);ps=1-ps
 sigma=x[:,:100].std(axis=1,ddof=1);shape=np.divide(x,peak[:,None],out=np.full_like(x,np.nan),where=peak[:,None]>0)
 rise=np.array([crossing(y,int(p),.9)-crossing(y,int(p),.1) for y,p in zip(x,pi)])
 tail=np.divide(x[:,200:400].sum(axis=1)*1000,qt,out=np.full(len(v),np.nan),where=qt>0)
 tailmean=np.divide(x[:,300:400].mean(axis=1),peak,out=np.full(len(v),np.nan),where=peak>0)
 den=(x[:,:100]**2).sum(axis=1)
 acf=np.column_stack([np.divide((x[:,k:100]*x[:,:100-k]).sum(axis=1),den,out=np.full(len(v),np.nan),where=den>0) for k in range(1,21)])
 valid=np.all(np.isfinite(v),axis=1)&(peak>0)&(qt>0)&np.isfinite(ps)
 rel=np.divide(sigma,peak,out=np.full(len(v),np.nan),where=peak>0)
 f={'PSD':ps,'Qtotal_mVns':qt,'peak_V':peak,'baseline_RMS_V':sigma,'baseline_RMS_over_peak':rel,'rise_10_90_ns':rise,'tail_charge_fraction':tail,'tail_mean_over_peak':tailmean,'ACF1':acf[:,0],'baseline_mean_V':b}
 return {'V':v,'x':x,'shape':shape,'acf':acf,'features':f,'valid':valid,'peak_index':pi}

def load():
 with read_npz(CFG['input_npz']['SIM_METADATA']) as z:
  metadata={k:z[k] for k in ['raw','real','sim_event_ids','real_event_ids','normalization_peak_V','source_label','sim_energy','edep_ee_MeV','edep_hadron_MeV','physical_r_h']}
 with read_npz(CFG['input_npz']['GEN_FIXED']) as z:
  fixed={k:z[k] for k in ['before','after','real','sim_event_ids','real_event_ids']}
 with read_npz(CFG['input_npz']['GEN_W0']) as z:
  zero={k:z[k] for k in ['before','after','real','sim_event_ids','real_event_ids']}
 for data in [fixed,zero]:
  for k in ['sim_event_ids','real_event_ids']:assert np.array_equal(data[k],metadata[k]),'parent ID/order conflict'
  assert np.array_equal(data['before'],metadata['raw']) and np.array_equal(data['real'],metadata['real']),'historical input version conflict'
 for k in ['raw','real']:assert metadata[k].shape==(2000,480)
 assert len(set(metadata['sim_event_ids'].tolist()))==2000 and len(set(metadata['real_event_ids'].tolist()))==2000
 prefix=pathlib.Path(CFG['REAL_prefix_dir']);identity=pathlib.Path(CFG['REAL_identity_dir'])
 real=read_prefix(prefix/'window_V_f32.bin','<f4',2000*480).reshape(2000,480)
 norm=read_prefix(prefix/'window_normalized_f32.bin','<f4',2000*480).reshape(2000,480)
 assert np.array_equal(norm,metadata['real']),'REAL/P1K DEV order or preprocessing conflict'
 source_qc=read_prefix(prefix/'QC_reason_u16.bin','<u2',2000)
 orig=read_prefix(identity/'selection_original_index.bin','<u4',2000)
 group=read_prefix(identity/'selection_group.bin','<u2',2000)
 assert np.array_equal(group,orig//50),'artificial group semantics'
 reference=json.loads((prefix/'STATISTICS.json').read_text())
 INPUTS.append({'path':str(prefix/'STATISTICS.json'),'bytes':(prefix/'STATISTICS.json').stat().st_size,'sha256':sha((prefix/'STATISTICS.json').read_bytes()),'read_scope':'frozen CAL reference JSON,no refit'})
 peak=metadata['normalization_peak_V'];assert peak.shape==(2000,) and np.all(np.isfinite(peak)&(peak>0))
 domains={'SIM':features(metadata['raw'].astype(float)*peak[:,None]),'GEN_FIXED':features(fixed['after'].astype(float)*peak[:,None]),'GEN_W0':features(zero['after'].astype(float)*peak[:,None]),'REAL':features(real)}
 domains['REAL']['valid']&=source_qc==0
 for name,d in domains.items():
  d['ids']=metadata['real_event_ids'] if name=='REAL' else metadata['sim_event_ids'];d['groups']=group.astype(int) if name=='REAL' else np.arange(2000);d['region']=np.searchsorted(reference['CAL_charge_quantile_edges_mV_ns'],d['features']['Qtotal_mVns'],side='right').astype('u1')
 np.savez_compressed(OUT/'association_and_features.npz',sim_event_ids=metadata['sim_event_ids'],real_event_ids=metadata['real_event_ids'],REAL_original_index=orig,REAL_artificial_group=group,SIM_source_label=metadata['source_label'],SIM_energy_declared=metadata['sim_energy'],edep_ee_MeV=metadata['edep_ee_MeV'],edep_hadron_MeV=metadata['edep_hadron_MeV'],SIM_normalization_peak_V=peak,physical_r_h=metadata['physical_r_h'],**{n+'__'+k:v for n,d in domains.items() for k,v in d['features'].items()},**{n+'__QC_valid':d['valid'] for n,d in domains.items()},**{n+'__charge_region':d['region'] for n,d in domains.items()})
 return domains,metadata,reference

def mask(d,r=None):return d['valid'] if r is None else d['valid']&(d['region']==r)
def characterize(domains,ref):
 stats={};comparisons={};real=domains['REAL']
 for name,d in domains.items():
  label=np.zeros(2000,dtype='u1');post=np.full((2000,2),np.nan)
  for r,slot in enumerate(ref['slots'][1:]):
   m=mask(d,r)
   if slot['valid'] and np.any(m):
    p,_=posterior(d['features']['PSD'][m],np.array(slot['weights']),np.array(slot['means']),np.array(slot['variances']));post[m]=p;winner=p.argmax(axis=1);high=p.max(axis=1)>=.99;label[m]=np.where(high,winner+1,0)
  blocks={}
  for r in [None,0,1,2,3]:
   key='overall' if r is None else 'q'+str(r+1);m=mask(d,r);mr=mask(real,r);slot=ref['slots'][0 if r is None else r+1];cut=slot['cut'] if slot['valid'] else None
   blocks[key]={'source_n':2000 if r is None else int(np.sum(d['region']==r)),'QC_n':int(m.sum()),'HC_gamma_like_n':int(np.sum(label[m]==1)),'HC_neutron_like_n':int(np.sum(label[m]==2)),'unknown_n':int(np.sum(label[m]==0)),'reference_valid':bool(slot['valid']),'frozen_reference_means':slot['means'] if slot['valid'] else None,'FoM':fom(d['features']['PSD'][m],cut),'features':{k:summary(v[m]) for k,v in d['features'].items()}}
   if name!='REAL':comparisons.setdefault(name,{})[key]={k:distances(v[m&np.isfinite(v)],real['features'][k][mr&np.isfinite(real['features'][k])]) for k,v in d['features'].items()}
  stats[name]={'QC_valid':int(d['valid'].sum()),'invalid':int((~d['valid']).sum()),'regions':blocks}
  d['label']=label;d['posterior']=post
 return stats,comparisons

def repeat_reference(real,domains):
 out={};rr_arrays={}
 for r in [None,0,1,2,3]:
  key='overall' if r is None else 'q'+str(r+1);rows=np.flatnonzero(mask(real,r));samples={k:[] for k in real['features']};counts=[];matched={n:{k:[] for k in samples} for n in NAMES if n!='REAL'}
  for rep in range(64):
   side=np.array([hashint(f'R1-RR-{rep}|{g}')%2 for g in real['groups']]);aa=rows[side[rows]==0];bb=rows[side[rows]==1];n=min(len(aa),len(bb));counts.append(n)
   aa=aa[order(real['ids'][aa],f'RR-A-{rep}')[:n]];bb=bb[order(real['ids'][bb],f'RR-B-{rep}')[:n]]
   for k,v in real['features'].items():samples[k].append(distances(v[aa][np.isfinite(v[aa])],v[bb][np.isfinite(v[bb])]))
   for name,d in domains.items():
    if name=='REAL':continue
    ii=np.flatnonzero(mask(d,r));m=min(n,len(ii));ii=ii[order(d['ids'][ii],f'RR-SIM-{rep}')[:m]];rr=bb[:m]
    for k,v in d['features'].items():matched[name][k].append(distances(v[ii][np.isfinite(v[ii])],real['features'][k][rr][np.isfinite(real['features'][k][rr])]))
  out[key]={'per_domain_sample_count':summary(np.array(counts)),'R_R':{k:{met:summary(np.array([v[met] if v[met] is not None else np.nan for v in vals])) for met in ['W1','KS']} for k,vals in samples.items()},'same_n_sim_gen_vs_REAL':{name:{k:{met:summary(np.array([v[met] if v[met] is not None else np.nan for v in vals])) for met in ['W1','KS']} for k,vals in dd.items()} for name,dd in matched.items()},'repeat_n':64,'confidence_interval':False,'physical_tolerance':False}
  rr_arrays[key]=counts
 return out

def c2st_matrix(d,view):
 f=d['features'];columns=['PSD','rise_10_90_ns','tail_charge_fraction','tail_mean_over_peak','baseline_RMS_over_peak'];scalar=np.column_stack([f[k] for k in columns]+[d['acf'][:,i] for i in range(8)])
 if view=='shape_plus_amplitude_proxy':
  scalar=np.column_stack([scalar,*[np.log(np.where(f[k]>0,f[k],np.nan)) for k in ['Qtotal_mVns','peak_V','baseline_RMS_V']]])
 a=np.column_stack([scalar,d['shape']]);a[~np.isfinite(a)]=np.nan;return a,scalar.shape[1]

def bootstrap(y,p,groups,seed):
 uniq,inv=np.unique(groups,return_inverse=True);rng=np.random.default_rng(seed);acc=[];auc=[]
 for _ in range(1000):
  mult=np.bincount(rng.integers(0,len(uniq),size=len(uniq)),minlength=len(uniq));weight=mult[inv]
  if np.sum(weight[y==0])==0 or np.sum(weight[y==1])==0:continue
  acc.append(.5*sum(np.sum(weight[(y==k)&((p>=.5)==k)])/np.sum(weight[y==k]) for k in [0,1]));auc.append(roc_auc_score(y,p,sample_weight=weight))
 return {'balanced_accuracy_95CI':np.quantile(acc,[.025,.975]).tolist(),'AUC_95CI':np.quantile(auc,[.025,.975]).tolist(),'bootstrap_clusters':len(uniq),'replicates':len(acc),'conditional_on_fitted_evaluator':True}

def run_c2st(domains):
 real=domains['REAL'];b=real['V'][:,:100].mean(axis=1);peak=real['features']['peak_V'];rng=np.random.default_rng(202610081);t=np.arange(480);tail=np.where(t>=120,np.exp(-np.maximum(t-120,0)/80),0)
 pert=features(real['V']+.02*peak[:,None]*tail[None,:]+.015*peak[:,None]*rng.standard_normal((2000,480)));pert['ids']=real['ids'];pert['groups']=real['groups'];pert['valid']&=real['valid']
 full={**domains,'PERTURBED_REAL':pert};splits={}
 for name,d in full.items():
  typ='REAL' if name in ['REAL','PERTURBED_REAL'] else 'SIM';keys=d['groups'] if typ=='REAL' else d['ids'];s=np.array([hashint('R1-SPLIT|'+typ+'|'+str(k))%10<7 for k in keys]);splits[name]=(np.flatnonzero(d['valid']&s),np.flatnonzero(d['valid']&~s))
 side=np.array([hashint('R1-RR-DOMAIN|'+str(g))%2 for g in real['groups']]);rr=[]
 for k in [0,1]:rr.append(tuple(ii[side[ii]==k] for ii in splits['REAL']))
 specs={n+'_vs_REAL':((n,splits[n]),('REAL',splits['REAL'])) for n in NAMES if n!='REAL'}
 specs['REAL_vs_REAL']=(('REAL',rr[0]),('REAL',rr[1]));specs['PERTURBED_REAL_vs_REAL']=(('PERTURBED_REAL',splits['PERTURBED_REAL']),('REAL',splits['REAL']))
 nt=min(CFG['C2ST']['train_cap_each_domain'],*(len(s[i]) for pair in specs.values() for _,s in pair for i in [0]));nv=min(CFG['C2ST']['test_cap_each_domain'],*(len(s[i]) for pair in specs.values() for _,s in pair for i in [1]));assert nt>=100 and nv>=100,'C2ST support too low'
 output={};saved={};proof={};params={k:CFG['C2ST'][k] for k in ['max_iter','learning_rate','max_leaf_nodes','min_samples_leaf','l2_regularization','early_stopping','random_state']}
 for title,pair in specs.items():
  selected=[]
  for name,s in pair:selected.append(tuple(ii[order(full[name]['ids'][ii],'R1-C2ST-SAMPLE')[:count]] for ii,count in zip(s,[nt,nv])))
  for (name,_),(tr,te) in zip(pair,selected):
   assert not set(full[name]['groups'][tr])&set(full[name]['groups'][te]),'source groups crossed TRAIN/HOLDOUT'
  # SIM and GEN use identical parent selection and source split by construction.
  proof[title]={'domain_names':[p[0] for p in pair],'train_indices':[s[0].tolist() for s in selected],'holdout_indices':[s[1].tolist() for s in selected],'parent_or_group_train_holdout_intersection':0}
  for view in CFG['C2ST']['views']:
   a,scalar=c2st_matrix(full[pair[0][0]],view);z,_=c2st_matrix(full[pair[1][0]],view);(at,av),(bt,bv)=selected;X=np.vstack([a[at],z[bt]]);T=np.vstack([a[av],z[bv]]);y=np.r_[np.ones(nt),np.zeros(nt)];yt=np.r_[np.ones(nv),np.zeros(nv)]
   imputer=SimpleImputer(strategy='median',keep_empty_features=True);X=imputer.fit_transform(X);T=imputer.transform(T);model=HistGradientBoostingClassifier(**params);fit_start=time.monotonic();model.fit(X,y);p=model.predict_proba(T)[:,1]
   groups=np.concatenate([np.array([('R|' if pair[j][0] in ['REAL','PERTURBED_REAL'] else 'S|')+str(g) for g in full[pair[j][0]]['groups'][selected[j][1]]]) for j in [0,1]])
   acc=balanced_accuracy_score(yt,p>=.5);result={'balanced_accuracy':float(acc),'AUC':float(roc_auc_score(yt,p)),'train_each_domain':nt,'holdout_each_domain':nv,'holdout_total':2*nv,'n_features':T.shape[1],'fit_s':time.monotonic()-fit_start,'heldout_CI':bootstrap(yt,p,groups,20261008),'chance':.5,'tuning':False}
   blocks={'scalar_PSD_rise_tail':list(range(4)),'relative_RMS_ACF':list(range(4,13)),'wave_baseline_0_100':list(range(scalar,scalar+100)),'wave_prompt_100_150':list(range(scalar+100,scalar+150)),'wave_tail_150_400':list(range(scalar+150,scalar+400)),'wave_end_400_480':list(range(scalar+400,scalar+480))}
   if scalar>13:blocks['absolute_charge_peak_RMS']=list(range(13,scalar))
   permutation={};rngp=np.random.default_rng(20261008)
   for block,ix in blocks.items():
    U=T.copy();sh=rngp.permutation(len(U));U[:,ix]=U[np.ix_(sh,ix)];permutation[block]=float(acc-balanced_accuracy_score(yt,model.predict(U)))
   result['heldout_permutation_block_accuracy_drop']=permutation;output.setdefault(title,{})[view]=result
   tag=title+'__'+view;saved[tag+'__probability']=p;saved[tag+'__label']=yt;saved[tag+'__cluster']=groups
   log(f'C2ST {title} {view}: BA={acc:.4f} AUC={result["AUC"]:.4f}; fit={result["fit_s"]:.2f}s')
 write_json('C2ST_SPLITS.json',proof);np.savez_compressed(OUT/'C2ST_HOLDOUT_PREDICTIONS.npz',**saved)
 return output

def savefig(fig,name):fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
def plots(domains,ref,stats,comparisons,c2st,metadata):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
 for name,ax in zip(NAMES,axes.ravel()):
  d=domains[name];m=mask(d);q=d['features']['Qtotal_mVns'][m];ps=d['features']['PSD'][m];ax.scatter(q,ps,c=COLORS[name],s=4,alpha=.25,rasterized=True);ax.set_xscale('log');ax.set_title(name+f' QC={m.sum()}/2000');ax.set_xlabel('Signed total charge [mV ns; GEN scale proxy]');ax.set_ylabel('Signed PSD = 1-Qfast/Qtotal')
  for edge in ref['CAL_charge_quantile_edges_mV_ns']:ax.axvline(edge,color='gray',lw=.6,ls='--')
 fig.suptitle('All QC-valid events, including overlap/unknown; no energy reweighting');fig.tight_layout();savefig(fig,'charge_psd')
 fig,axes=plt.subplots(2,2,figsize=(13,9))
 for r,ax in enumerate(axes.ravel()):
  for name,d in domains.items():
   x=np.sort(d['features']['PSD'][mask(d,r)]);ax.plot(x,np.arange(1,len(x)+1)/len(x) if len(x) else [],label=f'{name} n={len(x)}',color=COLORS[name])
  slot=ref['slots'][r+1]
  if slot['valid']:ax.axvline(slot['cut'],color='gray',ls='--',label='frozen CAL cut')
  ax.set_title(f'Charge q{r+1}; '+('reference valid' if slot['valid'] else 'reference failed: labels unknown'));ax.set_xlabel('Signed PSD');ax.set_ylabel('Full-sample empirical CDF');ax.legend(fontsize=8)
 fig.tight_layout();savefig(fig,'conditional_psd')
 fig,axes=plt.subplots(4,2,figsize=(14,16));wave_summary={};t=np.arange(480)-120
 for r in range(4):
  for j,kind in enumerate(['x','shape']):
   ax=axes[r,j]
   for name,d in domains.items():
    x=d[kind][mask(d,r)];n=len(x)
    if not n:continue
    qq=np.quantile(x,[.1,.5,.9],axis=0);mean=x.mean(axis=0);ax.fill_between(t,qq[0],qq[2],alpha=.08,color=COLORS[name]);ax.plot(t,qq[1],label=name+f' n={n}',color=COLORS[name]);ax.plot(t,mean,ls=':',lw=1,color=COLORS[name]);wave_summary[name+f'_q{r+1}_{kind}']=np.vstack([mean,qq])
   ax.set_title(f'q{r+1}: '+('baseline-subtracted voltage; GEN proxy' if j==0 else 'own-peak normalized shape'));ax.set_xlabel('Time relative to saved CFD alignment [ns]');ax.set_ylabel('V' if j==0 else 'Normalized amplitude');ax.legend(fontsize=8)
 fig.suptitle('Solid median, dotted mean, 10-90% event bands; own-charge regions');fig.tight_layout();savefig(fig,'conditional_waveforms');np.savez_compressed(OUT/'waveform_summary.npz',**wave_summary)
 fig,axes=plt.subplots(2,4,figsize=(17,8))
 keys=['PSD','Qtotal_mVns','peak_V','rise_10_90_ns','tail_charge_fraction','baseline_RMS_V','baseline_RMS_over_peak','ACF1']
 for k,ax in zip(keys,axes.ravel()):
  for name,d in domains.items():
   values=d['features'][k][mask(d)];x=np.sort(values[np.isfinite(values)]);ax.plot(x,np.arange(1,len(x)+1)/len(x) if len(x) else [],color=COLORS[name],label=name)
  ax.set_xlabel(k);ax.set_ylabel('Empirical CDF')
  if k in ['Qtotal_mVns','peak_V','baseline_RMS_V','baseline_RMS_over_peak']:ax.set_xscale('log')
 axes[0,0].legend(fontsize=8);fig.tight_layout();savefig(fig,'response_features')
 fig,axes=plt.subplots(2,3,figsize=(16,9));noise_saved={};freq=np.fft.rfftfreq(100,d=1e-9)/1e6
 for name,d in domains.items():
  x=d['x'][mask(d),:100];power=np.abs(np.fft.rfft(x,axis=1))**2/(1e9*100);power[:,1:-1]*=2;mean=power.mean(axis=0);noise_saved[name+'_PSD_V2_per_Hz']=mean;axes[0,0].loglog(freq[1:],mean[1:],color=COLORS[name],label=name)
  a=d['acf'][mask(d)];axes[0,1].plot(np.arange(1,21),np.nanmedian(a,axis=0),color=COLORS[name],label=name)
  f=d['features'];axes[0,2].scatter(f['Qtotal_mVns'][mask(d)],f['baseline_RMS_V'][mask(d)],s=3,alpha=.2,color=COLORS[name])
 axes[0,0].set_xlabel('Frequency [MHz]');axes[0,0].set_ylabel('Saved baseline periodogram [V²/Hz]');axes[0,0].legend(fontsize=8);axes[0,1].set_xlabel('Lag [ns]');axes[0,1].set_ylabel('Median ACF, full100 denominator');axes[0,1].legend(fontsize=8);axes[0,2].set_xscale('log');axes[0,2].set_yscale('log');axes[0,2].set_xlabel('Total charge [mV ns; GEN proxy]');axes[0,2].set_ylabel('Baseline RMS [V; GEN proxy]')
 for r,ax in enumerate(axes[1]):
  region=[0,1,2,3] if r==2 else [r]
  for name,d in domains.items():
   m=mask(d)&np.isin(d['region'],region);v=d['features']['baseline_RMS_over_peak'][m];v=np.sort(v[np.isfinite(v)]);ax.plot(v,np.arange(1,len(v)+1)/len(v) if len(v) else [],color=COLORS[name],label=name)
  ax.set_xscale('log');ax.set_xlabel('Baseline RMS / peak');ax.set_ylabel('CDF');ax.set_title('All regions' if r==2 else f'q{r+1}')
 fig.suptitle('Saved-window fluctuation diagnostics; not a standalone electronics-noise calibration');fig.tight_layout();savefig(fig,'fluctuations');np.savez_compressed(OUT/'noise_spectrum_summary.npz',frequency_MHz=freq,**noise_saved)
 fig,axes=plt.subplots(1,2,figsize=(13,5));titles=list(c2st);pos=np.arange(len(titles))
 for j,view in enumerate(CFG['C2ST']['views']):
  yy=[c2st[k][view]['balanced_accuracy'] for k in titles];ci=np.array([c2st[k][view]['heldout_CI']['balanced_accuracy_95CI'] for k in titles]);axes[j].errorbar(pos,yy,yerr=np.maximum(0,np.vstack([np.array(yy)-ci[:,0],ci[:,1]-np.array(yy)])),fmt='o',capsize=4);axes[j].axhline(.5,color='gray',ls='--');axes[j].set_xticks(pos,titles,rotation=25,ha='right');axes[j].set_ylim(.4,1.03);axes[j].set_ylabel('Heldout balanced accuracy');axes[j].set_title(view)
 fig.suptitle('Independent fixed evaluator, source-group holdout; cluster-bootstrap 95% intervals');fig.tight_layout();savefig(fig,'c2st')
 fig,axes=plt.subplots(1,3,figsize=(16,5));rh=metadata['physical_r_h'];ps=domains['SIM']['features']['PSD']
 for j,name in enumerate(['SIM','GEN_FIXED','GEN_W0']):
  d=domains[name];m=mask(d)&domains['SIM']['valid'];axes[j].scatter(rh[m],d['features']['PSD'][m],c=ps[m],cmap='viridis',s=5,alpha=.4);axes[j].set_title(name);axes[j].set_xlabel('Saved hadron visible fraction');axes[j].set_ylabel('Signed PSD')
 fig.suptitle('Identical G4 parents preserved; source label gamma = legacy non2112 bucket');fig.tight_layout();savefig(fig,'parent_response')

def main():
 log(f'R1 numeric worker PID={os.getpid()}');domains,metadata,ref=load();log('input identity + all2000 normalized REAL equality passed')
 stats,comparisons=characterize(domains,ref);rr=repeat_reference(domains['REAL'],domains);log('all distributions + 64 REAL group references complete')
 c2st=run_c2st(domains);log('C2ST complete')
 associations={}
 for name in ['SIM','GEN_FIXED','GEN_W0']:
  d=domains[name];m=mask(d);x=metadata['physical_r_h'][m];y=d['features']['PSD'][m];s=spearmanr(x,y);associations[name]={'r_h_PSD_Spearman':float(s.statistic),'n':int(m.sum()),'delta_PSD_from_same_SIM_parent':summary(d['features']['PSD'][m]-domains['SIM']['features']['PSD'][m]),'source_label_counts':{str(label):int(np.sum(metadata['source_label'][m]==label)) for label in np.unique(metadata['source_label'])}}
 metrics={'status':'complete','domains':stats,'domain_vs_REAL':comparisons,'REAL_group_reference':rr,'C2ST':c2st,'parent_association':associations,'CAL_charge_edges_mVns':ref['CAL_charge_quantile_edges_mV_ns'],'CAL_reference_valid':[s['valid'] for s in ref['slots']],'HC_reference_is_not_truth':True,'GEN_voltage_is_SIM_parent_scale_proxy':True,'all2000_parent_order_and_REAL_preprocessing_equality':True,'historically_exposed_DEV':True,'EVAL_read':0,'new_response_training':0,'new_G4':0,'SSH':0,'numeric_worker_PID':os.getpid(),'elapsed_s_before_plots':time.monotonic()-START}
 write_json('metrics.json',metrics);write_json('INPUT_MANIFEST.json',INPUTS);plots(domains,ref,stats,comparisons,c2st,metadata)
 metrics['elapsed_s']=time.monotonic()-START;write_json('metrics.json',metrics);log(f'completed in {metrics["elapsed_s"]:.3f}s')

if __name__=='__main__':
 try:main()
 except BaseException as e:
  write_json('FAILURE.json',{'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'error':repr(e),'traceback':traceback.format_exc()});raise
