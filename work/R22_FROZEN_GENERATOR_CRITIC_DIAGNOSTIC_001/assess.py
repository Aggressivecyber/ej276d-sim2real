from __future__ import annotations
import os,sys,time,json
START=time.monotonic()
import numpy as np,scipy,sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from support import *
pool=npz(ROOT/'FIXED_POOL_PRIVATE.npz');pred=npz(ROOT/'CRITIC_SCORES_PRIVATE.npz');trace=json.loads((ROOT/'TRAINING_TRACE.json').read_text('utf8'));grad=npz(ROOT/'INPUT_GRADIENTS_PRIVATE.npz')
def score(g,r,wg=None,wr=None):
    g=np.asarray(g,dtype='f8');r=np.asarray(r,dtype='f8')
    wg=np.ones(len(g),'f8')if wg is None else np.asarray(wg,'f8');wr=np.ones(len(r),'f8')if wr is None else np.asarray(wr,'f8')
    if wg.sum()<=0 or wr.sum()<=0:return None
    wg=wg/wg.sum();wr=wr/wr.sum()
    y=np.r_[np.zeros(len(g),'u1'),np.ones(len(r),'u1')];p=np.r_[g,r]
    return {'balanced_MSE':float(.5*(np.dot(wg,g*g)+np.dot(wr,(r-1)**2))),
       'BA_threshold0.5':float(.5*(np.dot(wg,g<.5)+np.dot(wr,r>=.5))),
       'AUC_REAL1':float(roc_auc_score(y,p,sample_weight=np.r_[wg,wr])),
       'GEN_weighted_score_mean':float(np.dot(wg,g)),'REAL_weighted_score_mean':float(np.dot(wr,r)),
       'GEN_score_quantiles':np.quantile(g,[0,.05,.25,.5,.75,.95,1]),'REAL_score_quantiles':np.quantile(r,[0,.05,.25,.5,.75,.95,1])}
def evaluate(g,r,split):
    b=pool['gen_'+split+'_bin'];rb=pool['real_'+split+'_bin'];gw=pool['gen_'+split+'_weight'];rw=pool['real_'+split+'_weight']
    return {'GEN_n':len(g),'REAL_n':len(r),'unweighted':score(g,r),'equalbin_TILT_weighted':score(g,r,gw,rw),
      'bins':[{'bin':j+1,'GEN_n':int((b==j).sum()),'REAL_n':int((rb==j).sum()),'unweighted':score(g[b==j],r[rb==j]),'weighted':score(g[b==j],r[rb==j],gw[b==j],rw[rb==j])}for j in range(8)]}
metrics={'critics':{},'HGB':{},'definition':{'AUC_direction':'REAL=1 fixed; no sign inversion','BA_threshold':.5,'MSE':'balanced domains','HGB_fit_weight_scale':'each domain relativeweights sums1; concatenate and multiply Ntotal/2 giving global mean sample_weight1 for fixedL2=1','primary':'pre_ADC','pool':'fixed TRAIN-fit plus original internal_validation; noDEV/EVAL'},'training':trace}
for arm in ['RAW','STANDARDIZED']:
    metrics['critics'][arm]={}
    for ck in ['selected','final']:
        metrics['critics'][arm][ck]={}
        for rep in ['pre','post']:
            metrics['critics'][arm][ck][rep]={split:evaluate(pred[arm+'_'+ck+'_gen_'+split+'_'+rep],pred[arm+'_'+ck+'_real_'+split],split)for split in ['train','val']}
fit_count=0
for rep in ['pre','post']:
    x=np.concatenate([pool['gen_train_'+rep],pool['real_train']]);y=np.r_[np.zeros(len(pool['gen_train_'+rep]),'u1'),np.ones(len(pool['real_train']),'u1')]
    wt=np.r_[pool['gen_train_weight'],pool['real_train_weight']]*len(x)/2
    hgb=HistGradientBoostingClassifier(**CFG['HGB']);fit_count+=1;write('HGB_FIT_COUNTER.json',{'actual_fit_attempts':fit_count,'max':2,'current':rep,'UTC':utc()})
    hgb.fit(x,y,sample_weight=wt)
    result={}
    for split in ['train','val']:
        g=hgb.predict_proba(pool['gen_'+split+'_'+rep])[:,1];r=hgb.predict_proba(pool['real_'+split])[:,1]
        pred['HGB_'+rep+'_gen_'+split]=g;pred['HGB_'+rep+'_real_'+split]=r;result[split]=evaluate(g,r,split)
    metrics['HGB'][rep]=result
    import pickle
    with (ROOT/('HGB_'+rep+'.pkl')).open('wb')as f:pickle.dump(hgb,f)
    log({'HGB_rep':rep,'fit_attempts':fit_count,'val':result['val']['equalbin_TILT_weighted']})
    del x,hgb
np.savez_compressed(ROOT/'ALL_SCORES_PRIVATE.npz',**pred)
# Original source parents and real artificial groups are resampled globally;
# one shared multiplicity applies to every bin, preserving REAL group coupling.
gb=pool['gen_val_bin'];rb=pool['real_val_bin'];gw=pool['gen_val_weight'];rw=pool['real_val_weight']
rgroup=pool['real_val_original_index']//50;rg,ri=np.unique(rgroup,return_inverse=True)
rng=np.random.default_rng(CFG['seeds']['bootstrap'])
names=['RAW_pre','STANDARDIZED_pre','RAW_post','STANDARDIZED_post','HGB_pre','HGB_post'];boot={k:[]for k in names};diff=[];skipped=0
def pair(k):
    if k.startswith('HGB'):return pred[k+'_gen_val'],pred[k+'_real_val']
    arm,rep=k.rsplit('_',1);return pred[arm+'_selected_gen_val_'+rep],pred[arm+'_selected_real_val']
for repeat in range(500):
    gm=np.bincount(rng.integers(0,len(gb),len(gb)),minlength=len(gb))
    rm=np.bincount(rng.integers(0,len(rg),len(rg)),minlength=len(rg))[ri]
    wg=gw*gm;wr=rw*rm
    if any(wg[gb==b].sum()==0 or wr[rb==b].sum()==0 for b in range(8)):
        skipped+=1;continue
    for b in range(8):wg[gb==b]/=8*wg[gb==b].sum();wr[rb==b]/=8*wr[rb==b].sum()
    rows={}
    for k in names:
        g,r=pair(k);v=score(g,r,wg,wr);row=np.array([v['BA_threshold0.5'],v['AUC_REAL1'],v['balanced_MSE']]);boot[k].append(row);rows[k]=row
    diff.append(rows['STANDARDIZED_pre']-rows['RAW_pre'])
metrics['bootstrap']={'attempted':500,'accepted_all8bins_supported':500-skipped,'missing_bin_draws':skipped,'method':'source-parent and REAL50record-group global multiplicities shared acrossbins, re-normalize weights to1/8/bin/domain; conditional on complete8bin support',
 'intervals':{k:{'levels':[.025,.975],'columns':['BA','AUC','balanced_MSE'],'CI95':np.quantile(v,[.025,.975],axis=0)}for k,v in boot.items()},
 'STANDARDIZED_minus_RAW_pre':{'columns':['BA','AUC','balanced_MSE'],'CI95':np.quantile(diff,[.025,.975],axis=0)},
 'REAL_group_count':len(rg),'G4_parent_count':len(gb),'highest_source_val_bin_N':4,'fixed_fitted_classifier_only':True,'not_physical_independent_runs':True}
np.savez_compressed(ROOT/'BOOTSTRAP_PRIVATE.npz',**{k:np.asarray(v)for k,v in boot.items()},STANDARDIZED_minus_RAW_pre=np.asarray(diff))
regions={'all':(0,480),'baseline':(0,100),'prompt':(100,150),'postpeak':(150,220),'tail':(220,400),'end':(400,480)}
metrics['input_gradient']={'unit':'V^-1','objective':'mean over64 fixed val sourceparent rows of (D(gen)-1)^2; hence perrow gradient includes1/64 factor','indices_private':'INPUT_GRADIENTS_PRIVATE.npz','finite':{},'arms':{}}
for arm in ['RAW','STANDARDIZED']:
    a=grad[arm];metrics['input_gradient']['finite'][arm]=bool(np.isfinite(a).all());metrics['input_gradient']['arms'][arm]={}
    for k,(lo,hi)in regions.items():
        rms=np.sqrt(np.mean(a[:,lo:hi].astype('f8')**2,axis=1));metrics['input_gradient']['arms'][arm][k]={'mean_RMS':float(rms.mean()),'median_RMS':float(np.median(rms)),'q05_q95':np.quantile(rms,[.05,.95]),'n':len(rms)}
write('METRICS.json',metrics)
# No identity arrays, prediction rows or per-event selections in published aggregate.
write('METRICS_COMPACT.json',metrics)
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'RAW':'#c65b2b','STANDARDIZED':'#176cba','HGB_pre':'#329254','HGB_post':'#7f55a5'}
def save(fig,n):fig.tight_layout();fig.savefig(FIG/(n+'.png'),dpi=145,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.4))
for arm in ['RAW','STANDARDIZED']:
    t=trace[arm]['trace'];st=np.array([v['step']for v in t]);J=np.array([v['balanced_equalbin_MSE']for v in t]);axes[0].plot(st,J,label=arm,color=colors[arm]);axes[1].plot(st,J,label=arm,color=colors[arm]);axes[1].scatter(trace[arm]['selected_step'],trace[arm]['selected_J'],color=colors[arm],marker='x',s=70)
for ax in axes:ax.set_xlabel('D updates');ax.set_ylabel('Equal-bin validation balanced MSE');ax.axhline(.25,color='gray',ls=':');ax.legend()
axes[1].set_ylim(0,.28);save(fig,'training_trace')
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for arm in ['RAW','STANDARDIZED']:
    g,r=pair(arm+'_pre');axes[0].hist(g,bins=40,density=True,histtype='step',color=colors[arm],label=arm+' GEN');axes[0].hist(r,bins=40,density=True,histtype='step',ls='--',color=colors[arm],label=arm+' REAL')
    fpr=[];tpr=[]
    from sklearn.metrics import roc_curve
    aa,bb,_=roc_curve(np.r_[np.zeros(len(g)),np.ones(len(r))],np.r_[g,r],sample_weight=np.r_[gw,rw]);axes[1].plot(aa,bb,color=colors[arm],label=arm)
axes[0].axvline(.5,color='gray',ls=':');axes[0].set_xlabel('Unbounded LS critic score (REAL=1)');axes[0].set_ylabel('Unweighted density');axes[0].legend(fontsize=8)
axes[1].plot([0,1],[0,1],color='gray',ls=':');axes[1].set_xlabel('Weighted false positive rate');axes[1].set_ylabel('Weighted true positive rate');axes[1].legend();save(fig,'validation_scores')
fig,axes=plt.subplots(1,2,figsize=(12,4.4))
for j,k in enumerate(['BA_threshold0.5','AUC_REAL1']):
    for i,name in enumerate(names):
        g,r=pair(name);point=score(g,r,gw,rw)[k];ci=np.quantile(np.asarray(boot[name])[:,j],[.025,.975]);axes[j].errorbar(i,point,yerr=np.array([[point-ci[0]],[ci[1]-point]]).clip(0),fmt='o',capsize=4,color=colors.get(name.split('_')[0],colors.get(name,'gray')))
    axes[j].set_xticks(range(len(names)),names,rotation=25,ha='right');axes[j].axhline(.5,color='gray',ls=':');axes[j].set_ylabel(k);axes[j].set_ylim(.35,1.02)
fig.suptitle('Internal validation: fixed-classifier 95% group bootstrap');save(fig,'classifier_comparison')
fig,axes=plt.subplots(1,2,figsize=(13,4.7))
for name in ['RAW_pre','STANDARDIZED_pre','HGB_pre']:
    if name.startswith('HGB'):result=metrics['HGB']['pre']['val']
    else:arm=name.rsplit('_',1)[0];result=metrics['critics'][arm]['selected']['pre']['val']
    for j,key in enumerate(['BA_threshold0.5','AUC_REAL1']):axes[j].plot(range(1,9),[v['weighted'][key]for v in result['bins']],marker='o',label=name)
for ax in axes:ax.set_xticks(range(1,9));ax.set_xlabel('Frozen source-Q / observed REAL-Q bin');ax.set_ylim(.35,1.02);ax.legend();ax.axhline(.5,color='gray',ls=':')
axes[0].set_ylabel('Per-bin weighted BA');axes[1].set_ylabel('Per-bin weighted AUC');fig.suptitle('Source validation N=40,41,56,68,44,27,15,4; highest bin is weak');save(fig,'per_bin_classification')
fig,axes=plt.subplots(2,2,figsize=(13,8))
for split,row in [('train',0),('val',1)]:
    for dom,col in [('GEN',0),('REAL',1)]:
        k=('gen_'+split+'_pre')if dom=='GEN'else 'real_'+split
        xx=pool[k];binsx=pool[('gen'if dom=='GEN'else'real')+'_'+split+'_bin'];axes[row,col].scatter(np.arange(len(xx)),binsx+1,s=2,alpha=.15);axes[row,col].set_title(split+' '+dom+' n='+str(len(xx)));axes[row,col].set_ylabel('Frozen Q bin');axes[row,col].set_xlabel('Fixed pool position')
save(fig,'fixed_pool_support')
fig,axes=plt.subplots(1,3,figsize=(15,4.3))
for c in range(2):axes[0].plot(pool['mu'][c],label='channel'+str(c+1));axes[1].plot(pool['scale'][c],label='channel'+str(c+1))
axes[0].set_xlabel('Saved time index');axes[0].set_ylabel('REAL TRAIN coordinate mean');axes[0].legend();axes[1].set_xlabel('Saved time index');axes[1].set_ylabel('Fixed scale');axes[1].set_yscale('log');axes[1].legend()
keys=list(regions)
for arm in ['RAW','STANDARDIZED']:axes[2].plot(range(len(keys)),[metrics['input_gradient']['arms'][arm][k]['mean_RMS']for k in keys],marker='o',label=arm,color=colors[arm])
axes[2].set_xticks(range(len(keys)),keys,rotation=35);axes[2].set_yscale('log');axes[2].set_ylabel('Adversarial input gradient RMS [V^-1]');axes[2].legend();save(fig,'scaler_and_gradients')
fig,axes=plt.subplots(1,2,figsize=(12,4.3))
for arm in ['RAW','STANDARDIZED']:
    arr=metrics['critics'][arm]['selected'];values=[arr[r][s]['equalbin_TILT_weighted']['BA_threshold0.5']for s in ['train','val']for r in ['pre','post']];axes[0].plot(range(4),values,marker='o',label=arm)
    values=[arr[r][s]['equalbin_TILT_weighted']['balanced_MSE']for s in ['train','val']for r in ['pre','post']];axes[1].plot(range(4),values,marker='o',label=arm)
for ax in axes:ax.set_xticks(range(4),['fit pre','fit post','val pre','val post']);ax.legend()
axes[0].set_ylabel('Equal-bin weighted BA');axes[1].set_ylabel('Equal-bin balanced MSE');save(fig,'ADC_sensitivity')
write('ASSESSMENT_RECEIPT.json',{'PID':os.getpid(),'UTC':utc(),'elapsed_s':time.monotonic()-START,'python':sys.executable,'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'HGB_fit_attempts':fit_count,'newEM':0,'new_generation':0,'G_updates':0,'D_updates':0,'figures':7,'bootstrap_attempts':500,'bootstrap_accepted':500-skipped,'EVAL':0,'DEV':0,'CAL':0})
log({'phase':'complete','PID':os.getpid(),'HGB_fits':fit_count,'bootstrap_accepted':500-skipped,'elapsed_s':time.monotonic()-START})
