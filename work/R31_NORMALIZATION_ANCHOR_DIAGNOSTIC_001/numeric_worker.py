from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
from datetime import datetime,timezone
import json,time,traceback,platform
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from diagnostic_core import bootstrap
from paired_bootstrap import paired_bootstrap
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CFG=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic()
NAMES=CFG['domains'];COMPARE=['FRESH_FINAL','ENERGY_FINAL','TRAIN_REFERENCE'];VIEWS=CFG['views'];MASK=np.array(CFG['retained_indices'],'i8')
COUNTER={'HGB_attempts':0,'HGB_completed':0,'moment_reestimation':0,**CFG['prohibited_counts']}
def utc():return datetime.now(timezone.utc).isoformat()
def safe(v):
    if isinstance(v,dict):return {str(k):safe(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [safe(a) for a in v]
    if isinstance(v,np.ndarray):return safe(v.tolist())
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def write(n,v):(ROOT/n).write_text(json.dumps(safe(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def progress(s):
    if time.monotonic()-START>CFG['science_budget']-10:raise TimeoutError('science reserve')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(CFG['deadline_UTC']):raise TimeoutError('executor deadline')
    write('NUMERIC_PROGRESS.json',{'PID':os.getpid(),'UTC':utc(),'stage':s,'elapsed_s':time.monotonic()-START,'counters':COUNTER})
def quant(a):return np.quantile(a,[0,.5,.9,.99,1]).tolist() if len(a) else None
def build(v,name,rows,moments):
    v=np.asarray(v,'f8');s=v-v[:,:100].mean(1,keepdims=True);peak=s.max(1);argmax=np.argmax(s,axis=1)
    u=s/np.maximum(peak[:,None],CFG['normalization_peak_floor_V'])
    d={'u':u,'peak':peak,'argmax':argmax,'floor_active':peak<CFG['normalization_peak_floor_V'],
       'bin':rows[name+'__original_Qbin'],'q':rows[name+'__Q_mVns'],'valid':rows[name+'__QC'],'groups':rows[name+'__groups']}
    assert u.shape==(2000,480) and np.all(np.isfinite(u))
    prefix='REAL' if name=='REAL' else 'TRAIN_REFERENCE' if name=='TRAIN_REFERENCE' else 'G4_SOURCE'
    d['cluster']=np.array([prefix+'|'+str(g) for g in d['groups']])
    par={k:moments[name+'__'+k] for k in ['global_mean','global_SD','bin_mean','bin_SD','floor','support']}
    assert np.array_equal(par['support'],np.array([True]*7+[False]))
    d['params']=par
    m=np.repeat(par['global_mean'][None],2000,axis=0);sd=np.repeat(par['global_SD'][None],2000,axis=0)
    legal=(d['bin']>=0)&(d['bin']<8);good=np.zeros(2000,bool);good[legal]=par['support'][d['bin'][legal]]
    m[good]=par['bin_mean'][d['bin'][good]];sd[good]=par['bin_SD'][d['bin'][good]]
    d['views']={'RAW':u,'GLOBAL_CENTER':u-par['global_mean'],'Q_CENTER':u-m,'Q_CENTER_SCALE':(u-m)/np.maximum(sd,par['floor'])}
    d['fallback']=~good
    j=1 if name=='REAL' else 2 if name=='TRAIN_REFERENCE' else 0;tr=rows[f'phase0_{j}'].astype('i8');te=rows[f'phase1_{j}'].astype('i8')
    assert len(tr)==600 and len(te)==250 and d['valid'][np.r_[tr,te]].all() and not set(d['groups'][tr])&set(d['groups'][te])
    d['tr']=tr;d['te']=te
    return d
def geometry(domains):
    result={};private={}
    for n,d in domains.items():
        result[n]={}
        private.update({n+'__peak_V':d['peak'],n+'__argmax':d['argmax'],n+'__floor_active':d['floor_active'],n+'__max_u_error':np.abs(d['u'].max(1)-1)})
        for label,ix in [('population',np.arange(2000)),('train',d['tr']),('holdout',d['te'])]:
            err=np.abs(d['u'][ix].max(1)-1);floor=d['floor_active'][ix];a=d['argmax'][ix]
            result[n][label]={'N':len(ix),'QC_n':int(d['valid'][ix].sum()),'original_peak_V_quantiles':quant(d['peak'][ix]),'peak_floor_active_n':int(floor.sum()),
                'max_u_minus1_abs_quantiles':quant(err),'applicable_max_constraint_error':float(err[~floor].max()) if (~floor).any() else None,
                'argmax_inside_prompt100_150_n':int(((a>=100)&(a<150)).sum()),'argmax_outside_prompt_n':int(((a<100)|(a>=150)).sum()),
                'argmax_baseline_n':int((a<100).sum()),'argmax_postpeak_n':int(((a>=150)&(a<220)).sum()),'argmax_tail_n':int(((a>=220)&(a<400)).sum()),'argmax_late400_480_n':int((a>=400).sum()),
                'fallback_n':int(d['fallback'][ix].sum()),'removed_from_classification':0}
        private[n+'__argmax_histogram']=np.bincount(d['argmax'],minlength=480)
    np.savez_compressed(OUT/'GEOMETRY_PRIVATE.npz',**private)
    write('GEOMETRY_METRICS.json',result)
    return result
def decoder_score(y,p,abstain):
    n=[int((y==i).sum()) for i in [0,1]];non=[int(((y==i)&~abstain).sum()) for i in [0,1]];absn=[n[i]-non[i] for i in [0,1]]
    correct=[int(((y==i)&~abstain&(p==i)).sum()) for i in [0,1]]
    return {'classes_order':['REAL','candidate'],'class_N':n,'nonabstain_class_N':non,'abstain_class_N':absn,'correct_nonabstain_class_N':correct,
            'coverage':float((~abstain).mean()) if len(y) else None,'nonabstain_balanced_accuracy':.5*sum(correct[i]/non[i] for i in [0,1]) if min(non)>0 else None,
            'all_event_BA_worst_abstention_bound':.5*sum(correct[i]/n[i] for i in [0,1]) if min(n)>0 else None,
            'all_event_BA_best_abstention_bound':.5*sum((correct[i]+absn[i])/n[i] for i in [0,1]) if min(n)>0 else None}
def decoder(domains):
    result={};private={}
    for n in COMPARE:
        a=domains[n];r=domains['REAL'];w=np.vstack([a['views']['GLOBAL_CENTER'][a['te']],r['views']['GLOBAL_CENTER'][r['te']]])
        y=np.r_[np.ones(250,'i1'),np.zeros(250,'i1')]
        eA=np.abs((w+a['params']['global_mean']).max(1)-1);eB=np.abs((w+r['params']['global_mean']).max(1)-1)
        abstain=np.abs(eA-eB)<=CFG['decoder_abstain_error_difference_atol'];pred=(eA<eB).astype('i1');pred[abstain]=-1
        floor=np.r_[a['floor_active'][a['te']],r['floor_active'][r['te']]]
        true=np.where(y==1,eA,eB);wrong=np.where(y==1,eB,eA)
        result[n]={'no_fit':True,'error_difference_tolerance':1e-12,'all_rows':decoder_score(y,pred,abstain),'floor_inactive_applicable_rows':decoder_score(y[~floor],pred[~floor],abstain[~floor]),
            'floor_active_by_class_REAL_candidate':[int(((y==i)&floor).sum()) for i in [0,1]],'e_candidate_hypothesis_quantiles':quant(eA),'e_REAL_hypothesis_quantiles':quant(eB),
            'true_hypothesis_constraint_error_quantiles':quant(true),'wrong_hypothesis_constraint_error_quantiles':quant(wrong),'applicable_true_constraint_maxabs':float(true[~floor].max()) if (~floor).any() else None,
            'labels_only_for_scoring_not_prediction':True,'all_rows_retained':True}
        for key,value in {'error_candidate':eA,'error_REAL':eB,'label':y,'prediction':pred,'abstain':abstain,'peak_floor_active':floor}.items():private[n+'__'+key]=value
    np.savez_compressed(OUT/'DECODER_PRIVATE.npz',**private);write('DECODER_METRICS.json',result)
    return result
def pair(a,b):
    if not np.array_equal(a['label'],b['label']) or not np.array_equal(a['cluster'],b['cluster']):return {'defined':False,'reason':'different_rows_labels_or_groups'}
    return paired_bootstrap(a,b,CFG)
def fit(n,v,domains):
    a=domains[n];r=domains['REAL'];X=np.vstack([a['views'][v][a['tr']][:,MASK],r['views'][v][r['tr']][:,MASK]]);T=np.vstack([a['views'][v][a['te']][:,MASK],r['views'][v][r['te']][:,MASK]])
    y=np.r_[np.ones(600),np.zeros(600)];yt=np.r_[np.ones(250),np.zeros(250)]
    if X.shape!=(1200,430) or T.shape!=(500,430):raise ValueError('fixed dimensions')
    X[~np.isfinite(X)]=np.nan;T[~np.isfinite(T)]=np.nan
    imp=SimpleImputer(strategy='median',keep_empty_features=True);X=imp.fit_transform(X);T=imp.transform(T)
    scaler=StandardScaler();X=scaler.fit_transform(X);T=scaler.transform(T)
    h=HistGradientBoostingClassifier(**CFG['C2ST']);COUNTER['HGB_attempts']+=1;progress(n+' '+v+' fit')
    if COUNTER['HGB_attempts']>12:raise RuntimeError('12 fit cap')
    t=time.monotonic();h.fit(X,y);p=h.predict_proba(T)[:,1];pt=h.predict_proba(X)[:,1];COUNTER['HGB_completed']+=1
    pr={'probability':p,'label':yt,'cluster':np.r_[a['cluster'][a['te']],r['cluster'][r['te']]],'training_probability':pt,'training_label':y,
        'candidate_train':a['tr'],'candidate_holdout':a['te'],'REAL_train':r['tr'],'REAL_holdout':r['te'],
        'retained_indices':MASK,'imputer_statistics':imp.statistics_,'standardizer_mean':scaler.mean_,'standardizer_scale':scaler.scale_}
    np.savez_compressed(OUT/(n+'__'+v+'430_PREDICTIONS_PRIVATE.npz'),**pr)
    res={'balanced_accuracy':balanced_accuracy_score(yt,p>=.5),'AUC':roc_auc_score(yt,p),'training_balanced_accuracy':balanced_accuracy_score(y,pt>=.5),'training_AUC':roc_auc_score(y,pt),
         'features':430,'view':v,'train_each_domain':600,'holdout_each_domain':250,'heldout_CI':None,'fit_s':time.monotonic()-t,'train_only_imputer_and_scaler':True}
    write('results/C_'+n+'__'+v+'430.json',res)
    res['heldout_CI']=bootstrap(yt,p,pr['cluster'],CFG['bootstrap_repeats'],CFG['bootstrap_seed'])
    res['fit_and_bootstrap_s']=time.monotonic()-t
    return res,pr
def bybin(pr,b):
    ans=[]
    for k in range(8):
        use=b==k;y=pr['label'];p=pr['probability'];n0=int((use&(y==0)).sum());n1=int((use&(y==1)).sum());good=min(n0,n1)>=10
        ans.append({'bin_id':k+1,'REAL_n':n0,'candidate_n':n1,'qualified':good,'reason':None if good else 'a_domain_below10',
                    'balanced_accuracy':float(balanced_accuracy_score(y[use],p[use]>=.5)) if good else None})
    return ans
def classify(domains):
    old=json.loads(Path(CFG['inputs']['full480_metrics']).read_text('utf8'))['comparisons'];res={};regions={};common=np.ones(8,bool)
    for n in COMPARE:
        a=domains[n];r=domains['REAL'];qbin=np.r_[a['bin'][a['te']],r['bin'][r['te']]];preds={};res[n]={};regions[n]={}
        for v in VIEWS:
            rr,pr=fit(n,v,domains);preds[v]=pr
            p480=load(CFG['raw480_predictions'][n] if v=='RAW' else CFG['transform480_predictions'][n][v],['probability','label','cluster'])
            rr['full480_same_view_reused']=old[n][v];rr['paired430_minus480']=pair(pr,p480)
            if v!='RAW':rr['paired430_transformed_minus_RAW430']=pair(pr,preds['RAW'])
            else:rr['paired430_transformed_minus_RAW430']=None
            res[n][v]=rr;regions[n][v]=bybin(pr,qbin);common &= np.array([z['qualified'] for z in regions[n][v]])
            write('results/C_'+n+'__'+v+'430.json',rr);progress(n+' '+v+' persisted')
    for n in COMPARE:
        for v in VIEWS:
            vals=[z['balanced_accuracy'] for z in regions[n][v] if common[z['bin_id']-1]]
            res[n][v]['common_Qbin_equal_BA']=float(np.mean(vals)) if vals else None
    rr=res['TRAIN_REFERENCE'];ci=rr['RAW']['heldout_CI']['balanced_accuracy_95CI'];raw_ok=ci[0]<=.5<=ci[1];control={}
    for v in VIEWS[1:]:
        interval=rr[v]['paired430_transformed_minus_RAW430']['BA_delta_95CI'];delta_ok=interval[0]>=-.10 and interval[1]<=.10
        control[v]={'RAW430_CI_contains_0_5':raw_ok,'paired_delta_CI_within_minus_plus_0_10':delta_ok,'control_usable':raw_ok and delta_ok,
                    'paired_delta95CI':interval,'operational_not_physical_equivalence':True}
    result={'comparisons':res,'by_original_Qbin':regions,'common_Qbins_1based':np.flatnonzero(common)+1,'control_rule':control,
            'RAW430_REAL_REAL_CI_contains_0_5':raw_ok,'original480_results_retained':True,'HGB_attempts':COUNTER['HGB_attempts'],'no_regional_fits':True}
    write('CLASSIFIER_METRICS.json',result);return result
def run():
    progress('load only frozen saved inputs')
    write('NUMERIC_STARTED.json',{'PID':os.getpid(),'UTC':utc(),'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__})
    rows=load(CFG['inputs']['rows']);keys=[n+'__'+k for n in NAMES for k in ['global_mean','global_SD','bin_mean','bin_SD','floor','support']];moments=load(CFG['inputs']['moments'],keys)
    real=load(CFG['inputs']['real'],['real_V','reference_V']);domains={'REAL':build(real['real_V'],'REAL',rows,moments),'TRAIN_REFERENCE':build(real['reference_V'],'TRAIN_REFERENCE',rows,moments)}
    for n in CFG['candidates']:
        z=load(CFG['generated'][n],['waveforms_V','source_ids']);assert np.array_equal(z['source_ids'],rows['source_ids'])
        domains[n]=build(z['waveforms_V'],n,rows,moments)
    assert len(MASK)==430 and np.array_equal(MASK,np.r_[np.arange(100),np.arange(150,480)])
    geom=geometry(domains);dec=decoder(domains);classifications=classify(domains)
    result={'geometry':geom,'decoder':dec,'classification':classifications,'fixed_mask':{'retained_columns':430,'retained_intervals':[[0,100],[150,480]],'excluded_interval':[100,150],'no_renormalization_or_moment_reestimation':True},
            'parameter_source':CFG['inputs']['moments'],'full480_diagnostics':CFG['inputs']['full480_metrics'],
            'boundaries':{'masked_classification_not_complete_waveform_acceptance':True,'domain_identity_transform_not_deployable':True,'remaining_separability_not_unique_physical_cause':True,'same_acquisition_not_new_blind_test':True}}
    write('metrics.json',result);write('METRICS_COMPACT.json',result)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':utc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'domains_n':4,'saved_waveform_rows':8000,'all_rows_retained':True,'source_composition_by_reference':{'pure_EE':417,'pure_had':1518,'mixed':65},'exit_status':'success'})
    progress('complete');print(json.dumps({'PID':os.getpid(),'UTC':utc(),'success':True,'elapsed_s':time.monotonic()-START,'counters':COUNTER}),flush=True)
try:run()
except BaseException as e:
    write('NUMERIC_FAILURE.json',{'PID':os.getpid(),'UTC':utc(),'elapsed_s':time.monotonic()-START,'error':repr(e),'traceback':traceback.format_exc(),'counters':COUNTER,'failed_cost_retained':True});raise
