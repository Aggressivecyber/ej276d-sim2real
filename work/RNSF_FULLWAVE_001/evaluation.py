from __future__ import annotations
import os,pathlib,time,json,datetime,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from scipy.stats import spearmanr
from common_math import distances,posterior,fom
OUT=None;FIG=None;CFG=None;NAMES=None;COLORS=None

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

def mask(d,r=None):return d['valid'] if r is None else d['valid']&(d['region']==r)

def characterize(domains,ref):
 stats={};comparisons={};real=domains['REAL']
 for name,d in domains.items():
  label=np.zeros(len(d['V']),dtype='u1');post=np.full((len(d['V']),2),np.nan)
  for r,slot in enumerate(ref['slots'][1:]):
   m=mask(d,r)
   if slot['valid'] and np.any(m):
    p,_=posterior(d['features']['PSD'][m],np.array(slot['weights']),np.array(slot['means']),np.array(slot['variances']));post[m]=p;winner=p.argmax(axis=1);high=p.max(axis=1)>=.99;label[m]=np.where(high,winner+1,0)
  blocks={}
  for r in [None,0,1,2,3]:
   key='overall' if r is None else 'q'+str(r+1);m=mask(d,r);mr=mask(real,r);slot=ref['slots'][0 if r is None else r+1];cut=slot['cut'] if slot['valid'] else None
   blocks[key]={'source_n':len(d['V']) if r is None else int(np.sum(d['region']==r)),'QC_n':int(m.sum()),'HC_gamma_like_n':int(np.sum(label[m]==1)),'HC_neutron_like_n':int(np.sum(label[m]==2)),'unknown_n':int(np.sum(label[m]==0)),'reference_valid':bool(slot['valid']),'frozen_reference_means':slot['means'] if slot['valid'] else None,'FoM':fom(d['features']['PSD'][m],cut),'features':{k:summary(v[m]) for k,v in d['features'].items()}}
   if name!='REAL':comparisons.setdefault(name,{})[key]={k:distances(v[m&np.isfinite(v)],real['features'][k][mr&np.isfinite(real['features'][k])]) for k,v in d['features'].items()}
  stats[name]={'QC_valid':int(d['valid'].sum()),'invalid':int((~d['valid']).sum()),'regions':blocks}
  d['label']=label;d['posterior']=post
 return stats,comparisons

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

def savefig(fig,name):fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)

def plots(domains,ref,stats,comparisons,c2st,metadata):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,3,figsize=(17,9),sharex=True,sharey=True)
 for name,ax in zip(NAMES,axes.ravel()):
  d=domains[name];m=mask(d);q=d['features']['Qtotal_mVns'][m];ps=d['features']['PSD'][m];ax.scatter(q,ps,c=COLORS[name],s=4,alpha=.25,rasterized=True);ax.set_xscale('log');ax.set_title(name+f' QC={m.sum()}/2000');ax.set_xlabel('Signed total charge [mV ns; historical GEN scale proxy]');ax.set_ylabel('Signed PSD = 1-Qfast/Qtotal')
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
   ax.set_title(f'q{r+1}: '+('baseline-subtracted voltage; historical GEN proxy' if j==0 else 'own-peak normalized shape'));ax.set_xlabel('Time relative to saved CFD alignment [ns]');ax.set_ylabel('V' if j==0 else 'Normalized amplitude');ax.legend(fontsize=8)
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
 axes[0,0].set_xlabel('Frequency [MHz]');axes[0,0].set_ylabel('Saved baseline periodogram [V²/Hz]');axes[0,0].legend(fontsize=8);axes[0,1].set_xlabel('Lag [ns]');axes[0,1].set_ylabel('Median ACF, full100 denominator');axes[0,1].legend(fontsize=8);axes[0,2].set_xscale('log');axes[0,2].set_yscale('log');axes[0,2].set_xlabel('Total charge [mV ns; GEN proxy]');axes[0,2].set_ylabel('Baseline RMS [V; historical GEN proxy]')
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
 fig,axes=plt.subplots(1,5,figsize=(21,5));rh=metadata['physical_r_h'];ps=domains['SIM']['features']['PSD']
 for j,name in enumerate(['SIM','GEN_W0','R2','R3','R3B']):
  d=domains[name];m=mask(d)&domains['SIM']['valid'];axes[j].scatter(rh[m],d['features']['PSD'][m],c=ps[m],cmap='viridis',s=5,alpha=.4);axes[j].set_title(name);axes[j].set_xlabel('Saved hadron visible fraction');axes[j].set_ylabel('Signed PSD')
 fig.suptitle('Identical G4 parents preserved; source label gamma = legacy non2112 bucket');fig.tight_layout();savefig(fig,'parent_response')

def run_new_c2st(candidate,real):
 full={'R3B':candidate,'REAL':real};selected=[];nt=600;nv=250
 for name,d in full.items():
  typ='REAL' if name=='REAL' else 'SIM';keys=d['groups'] if typ=='REAL' else d['ids']
  flag=np.array([hashint('R1-SPLIT|'+typ+'|'+str(k))%10<7 for k in keys])
  tr=np.flatnonzero(d['valid']&flag);te=np.flatnonzero(d['valid']&~flag)
  assert len(tr)>=nt and len(te)>=nv,'fixed R1 C2ST support unavailable'
  tr=tr[order(d['ids'][tr],'R1-C2ST-SAMPLE')[:nt]];te=te[order(d['ids'][te],'R1-C2ST-SAMPLE')[:nv]]
  assert not set(d['groups'][tr])&set(d['groups'][te]),'C2ST parent/group leakage'
  selected.append((tr,te))
 params={k:CFG['C2ST'][k] for k in ['max_iter','learning_rate','max_leaf_nodes','min_samples_leaf','l2_regularization','early_stopping','random_state']}
 output={};saved={};(at,av),(bt,bv)=selected
 write_json('C2ST_SPLITS.json',{'R3_train_indices':at.tolist(),'R3_holdout_indices':av.tolist(),'REAL_train_indices':bt.tolist(),'REAL_holdout_indices':bv.tolist(),'same_R1_hash_split_order_and_caps':True,'train_holdout_group_intersection':0})
 for view in CFG['C2ST']['views']:
  a,scalar=c2st_matrix(candidate,view);b,_=c2st_matrix(real,view)
  X=np.vstack([a[at],b[bt]]);T=np.vstack([a[av],b[bv]])
  y=np.r_[np.ones(nt),np.zeros(nt)];yt=np.r_[np.ones(nv),np.zeros(nv)]
  imputer=SimpleImputer(strategy='median',keep_empty_features=True)
  X=imputer.fit_transform(X);T=imputer.transform(T);model=HistGradientBoostingClassifier(**params)
  started=time.monotonic();model.fit(X,y);p=model.predict_proba(T)[:,1]
  groups=np.r_[np.array(['S|'+str(g) for g in candidate['groups'][av]]),np.array(['R|'+str(g) for g in real['groups'][bv]])]
  acc=balanced_accuracy_score(yt,p>=.5)
  blocks={'scalar_PSD_rise_tail':list(range(4)),'relative_RMS_ACF':list(range(4,13)),'wave_baseline_0_100':list(range(scalar,scalar+100)),'wave_prompt_100_150':list(range(scalar+100,scalar+150)),'wave_tail_150_400':list(range(scalar+150,scalar+400)),'wave_end_400_480':list(range(scalar+400,scalar+480))}
  if scalar>13:blocks['absolute_charge_peak_RMS']=list(range(13,scalar))
  rng=np.random.default_rng(20261008);perm={}
  for name,ix in blocks.items():
   U=T.copy();sh=rng.permutation(len(U));U[:,ix]=U[np.ix_(sh,ix)];perm[name]=float(acc-balanced_accuracy_score(yt,model.predict(U)))
  output[view]={'balanced_accuracy':float(acc),'AUC':float(roc_auc_score(yt,p)),'train_each_domain':nt,'holdout_each_domain':nv,'holdout_total':2*nv,'n_features':T.shape[1],'fit_s':time.monotonic()-started,'heldout_CI':bootstrap(yt,p,groups,20261008),'chance':.5,'tuning':False,'heldout_permutation_block_accuracy_drop':perm,'classifier_newly_fit':True,'response_training_loss':False}
  saved[view+'__probability']=p;saved[view+'__label']=yt;saved[view+'__cluster']=groups
  log(f'C2ST R3B {view}:BA={acc:.4f} AUC={output[view]["AUC"]:.4f}')
 np.savez_compressed(OUT/'C2ST_HOLDOUT_PREDICTIONS.npz',**saved)
 return output

def fixed_latent_condition_examples(model,c,q,meta):
 import torch
 ix=np.array([0,500,1000,1999]);g=torch.Generator().manual_seed(2026100825)
 z=torch.randn((1,8),generator=g).repeat(4,1);white=torch.randn((1,512),generator=g).repeat(4,1)
 with torch.no_grad():v,p=model(c[ix],q[ix],z,white)
 vv=v.numpy();f=features(vv)
 np.savez_compressed(OUT/'fixed_latent_condition_examples.npz',waveforms_V=vv,parent_indices=ix,event_ids=meta['sim_event_ids'][ix],condition=c[ix].numpy(),same_z=z.numpy(),same_white=white.numpy())
 return {'selection':'fixed parent positions0,500,1000,1999; identical z and white-noise draw','parent_indices':ix.tolist(),'conditions':c[ix].numpy(),'PSD':f['features']['PSD'],'Qtotal_mVns':f['features']['Qtotal_mVns'],'rise_10_90_ns':f['features']['rise_10_90_ns'],'waveform_max_pairwise_difference_V':float(np.max(np.ptp(vv,axis=0))),'not_a_truth_or_causal_identification_test':True}

def plot_stochastic_examples(domains,metadata,waves):
 ix=[0,500,1000,1999];fig,axes=plt.subplots(4,2,figsize=(14,13));t=np.arange(480)-120
 for row,i in enumerate(ix):
  for view,w in enumerate(waves):
   d=features(w[i:i+1]);axes[row,0].plot(t,d['x'][0],lw=1,label=f'R3B view{view+1}')
   axes[row,1].plot(t,d['shape'][0],lw=1,label=f'R3B view{view+1}')
  axes[row,1].plot(t,domains['SIM']['shape'][i],color='black',ls=':',label='same G4 input shape')
  for col in [0,1]:
   axes[row,col].set_xlim(-25,260);axes[row,col].set_xlabel('Time relative to saved CFD [ns]');axes[row,col].legend(fontsize=8)
  axes[row,0].set_ylabel('Baseline-subtracted V');axes[row,1].set_ylabel('Own-peak normalized')
  axes[row,0].set_title(f'Fixed parent index{i}; visible hadron fraction={metadata["physical_r_h"][i]:.3f}')
 fig.suptitle('Same G4 parents, three predeclared random response views; no selection by outcome');fig.tight_layout();savefig(fig,'stochastic_parent_examples')

def plot_condition_pairs(results,saved):
 fig,axes=plt.subplots(2,2,figsize=(13,8));t=np.arange(480)-120
 for row,pair in enumerate(results):
  f=features(saved[pair['kind']+'__V'])
  for i in range(2):
   label=f'Evis={pair["visible_energy_MeV"][i]:.5g}MeV, r_h={pair["r_h"][i]:.3f}'
   axes[row,0].plot(t,f['x'][i],label=label);axes[row,1].plot(t,f['shape'][i],label=label)
  for j in range(2):
   axes[row,j].set_xlim(-20,180);axes[row,j].set_xlabel('Time from saved CFD [ns]');axes[row,j].legend(fontsize=8)
   axes[row,j].set_title('Energy at same composition' if row==0 else 'Composition at matched source energy')
  axes[row,0].set_ylabel('Baseline-subtracted voltage [V]');axes[row,1].set_ylabel('Own-peak normalized amplitude')
 fig.suptitle('Input-selected G4 pairs with fixed z and white-noise draw; no output selection');fig.tight_layout();savefig(fig,'fixed_latent_condition_pairs')
