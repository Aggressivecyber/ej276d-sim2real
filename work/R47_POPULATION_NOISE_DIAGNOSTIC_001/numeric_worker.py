"""Five fixed population/noise diagnostics from existing arrays; no model imports."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,traceback,hashlib,platform
import numpy as np,sklearn,scipy
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import evaluation as ev,diagnostic_core as dc
from common_math import distances

R=Path(__file__).resolve().parent;OUT=R/'results';OUT.mkdir(exist_ok=True)
C=json.loads((R/'config.json').read_text());START=time.monotonic()
COUNT={'HGB_attempts':0,'HGB_completed':0,'response_training_updates':0,'ODE_calls':0,'EM_attempts':0,
       'response_generated_rows':0,'EVAL_read':0,'CAL_waveform_decode':0,'new_G4':0,'deterministic_ADC_feature_rows':0}
def utc():return datetime.now(timezone.utc).isoformat()
def write(n,x):
    p=R/n;p.parent.mkdir(exist_ok=True,parents=True)
    p.write_bytes((json.dumps(ev.safe(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def progress(stage):write('DIAGNOSTIC_PROGRESS.json',{'PID':os.getpid(),'UTC':utc(),'stage':stage,'counters':COUNT});write('COUNTERS.json',COUNT)
def load(key,keys):
    p=Path(C['inputs'][key])
    with np.load(p,allow_pickle=False)as z:
        absent=set(keys)-set(z.files)
        if absent:raise KeyError((key,'required saved keys absent',sorted(absent)))
        return{k:z[k].copy() for k in keys}
def adc(x):
    v=np.asarray(x,dtype=np.float32);clipped=np.count_nonzero((v<-.5)|(v>.5))
    out=(-.5+np.rint((np.clip(v,-.5,.5)+.5)*16383)/16383).astype(np.float32)
    COUNT['deterministic_ADC_feature_rows']+=len(v)
    return out,{'clip_count':int(clipped),'clip_row_count':int(np.any((v<-.5)|(v>.5),axis=1).sum()),
                'pre_ADC_finite':bool(np.isfinite(v).all()),'bits':14,'Vrange':[-.5,.5]}
def qb(q):
    q=np.asarray(q,dtype='f8');b=np.full(q.shape,-2,dtype='i2');use=np.isfinite(q)&(q>0)
    z=np.log(q[use]);b[use]=np.searchsorted(EDGES,z,side='right')-1;b[use & (q==np.exp(EDGES[-1]))]=7
    zz=np.log(np.maximum(q,1e-300));b[use & (zz==EDGES[-1])]=7
    return b
def data(v,ids,groups,namespace,original_qc=None):
    d=ev.features(v);d.update(ids=np.asarray(ids),groups=np.asarray(groups),cluster=np.array([namespace+'|'+str(g)for g in groups]))
    d['input_QC_valid']=np.asarray(original_qc,dtype=bool)if original_qc is not None else np.ones(len(v),bool)
    d['valid'] &= d['input_QC_valid'];d['qbin']=qb(d['features']['Qtotal_mVns'])
    return d
def q_counts(d):
    q=d['features']['Qtotal_mVns'];b=d['qbin'];v=d['valid']
    return {'total_rows':len(q),'QC_valid':int(v.sum()),'QC_invalid':int((~v).sum()),'nonfinite_Q':int((~np.isfinite(q)).sum()),
      'nonpositive_Q':int((np.isfinite(q)&(q<=0)).sum()),'below_lower_support':int((b==-1).sum()),'above_upper_support':int((b>=8).sum()),
      'nonfinite_or_nonpositive_Q_unassigned':int((b==-2).sum()),'bin_counts_all_rows':[int((b==i).sum())for i in range(8)],
      'bin_counts_QC':[int((v&(b==i)).sum())for i in range(8)],'all_original_rows_retained':True}
def paired_bootstrap(a,b):
    if not(np.array_equal(a['label'],b['label'])and np.array_equal(a['cluster'],b['cluster'])):
        return {'defined':False,'reason':'saved_reference_label_or_group_order_mismatch','new_reference_fit':False}
    y=a['label'];p=a['probability'];q=b['probability'];_,inv=np.unique(a['cluster'],return_inverse=True);n=int(inv.max()+1)
    rng=np.random.default_rng(C['bootstrap_seed']);ba=[];auc=[];_,tp=np.unique(p,return_inverse=True);_,tq=np.unique(q,return_inverse=True)
    def score(prob,tie,w):
        n0=w[y==0].sum();n1=w[y==1].sum()
        value=.5*(w[(y==0)&(prob<.5)].sum()/n0+w[(y==1)&(prob>=.5)].sum()/n1)
        neg=np.bincount(tie[y==0],weights=w[y==0],minlength=int(tie.max()+1));pos=np.bincount(tie[y==1],weights=w[y==1],minlength=int(tie.max()+1))
        return value,float(np.dot(pos,np.cumsum(neg)-.5*neg)/(n0*n1))
    pa=score(p,tp,np.ones(len(y)));pb=score(q,tq,np.ones(len(y)))
    for _ in range(C['bootstrap_repeats']):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[inv]
        if w[y==0].sum()==0 or w[y==1].sum()==0:continue
        aa=score(p,tp,w);bb=score(q,tq,w);ba.append(aa[0]-bb[0]);auc.append(aa[1]-bb[1])
    return {'defined':True,'BA_delta_first_minus_second':pa[0]-pb[0],'BA_delta_95CI':np.quantile(ba,[.025,.975]),
            'AUC_delta_first_minus_second':pa[1]-pb[1],'AUC_delta_95CI':np.quantile(auc,[.025,.975]),'valid_replicates':len(ba),
            'clusters':n,'paired_original_rows_groups':True,'conditional_on_fitted_classifiers':True,'response_training_uncertainty':False}
def finite_distance(a,b):return distances(a[np.isfinite(a)],b[np.isfinite(b)])
def baseline_description(d,real):
    out={'population':q_counts(d),'baseline_RMS_zero_rows':int((d['features']['baseline_RMS_V']==0).sum()),
         'baseline_ACF1_nonfinite_rows':int((~np.isfinite(d['features']['ACF1'])).sum()),'fields':{},'own_observed_Q_bins':[]}
    for k in ['baseline_RMS_V','ACF1']:
        a=d['features'][k];b=real['features'][k]
        out['fields'][k]={'all_rows':ev.summary(a),'QC_rows':ev.summary(a[d['valid']]),
                        'W1_KS_vs_REAL_DEV_all_finite_rows':finite_distance(a,b),
                        'W1_KS_vs_REAL_DEV_QC_rows':finite_distance(a[d['valid']],b[real['valid']])}
    for i in range(8):
        use=d['qbin']==i;ru=real['qbin']==i
        out['own_observed_Q_bins'].append({'bin_id_1based':i+1,'source_rows':int(use.sum()),'QC_rows':int((use&d['valid']).sum()),
          'REAL_rows':int(ru.sum()),'fields':{k:{'summary':ev.summary(d['features'][k][use]),
          'distance_vs_REAL_same_observed_Q':finite_distance(d['features'][k][use],real['features'][k][ru])}for k in ['baseline_RMS_V','ACF1']}})
    return out
def decompose(a,original,name):
    S=a['base_signal_V'].astype('f8');G=a['projected_initial_Gaussian_f64_V'];F=a['learned_flow_increment_f64_V'];RR=a['final_cast_rounding_f64_V']
    assert S.shape==(2000,480)and np.array_equal(a['base_signal_V'],np.asarray(original,dtype='f4')),(name,'input S identity')
    pre=a['pre_ADC_V'].astype('f8');calc=S+G+F+RR;err=float(np.max(np.abs(calc-pre)))
    initial=S+G;initial_err=float(np.max(np.abs(initial-a['initial_state_f64_V'])))
    assert err<=1e-12 and initial_err<=1e-12,(name,'component definition changed',err,initial_err)
    parts={'S':S[:,:100],'G':G[:,:100],'F':F[:,:100],'R':RR[:,:100]}
    centered={k:v-v.mean(1,keepdims=True)for k,v in parts.items()};terms={}
    for k in centered:terms['Var_'+k]=np.sum(centered[k]**2,axis=1)/99
    names=list(centered)
    for i,k in enumerate(names):
        for j in range(i+1,len(names)):
            other=names[j];terms['2Cov_'+k+'_'+other]=2*np.sum(centered[k]*centered[other],axis=1)/99
    centerpre=pre[:,:100]-pre[:,:100].mean(1,keepdims=True);actual=np.sum(centerpre**2,axis=1)/99
    vsum=sum(terms.values());v_error=float(np.max(np.abs(vsum-actual)))
    assert v_error<=1e-18,(name,'baseline variance algebra',v_error)
    terms.update(actual_preADC_baseline_variance_V2=actual,component_variance_sum_V2=vsum)
    np.savez_compressed(OUT/(name+'_BASELINE_VARIANCE_PRIVATE.npz'),**terms)
    row={'baseline_sample_denominator':99,'per_row_baseline_centering':True,'all_rows':2000,
      'component_identity_maxabs_error_V':err,'initial_identity_maxabs_error_V':initial_err,'variance_sum_maxabs_error_V2':v_error,
      'term_summary':{k:ev.summary(v)for k,v in terms.items()},'all_six_signed_cross_terms_retained':True,
      'physical_independence_or_causal_contribution':False,'initial_preADC_vs_saved_BASE_float32_maxabs_V':float(np.max(np.abs(initial-a.get('base_pre_ADC_V',initial))))}
    return initial,row

try:
    progress('read only existing saved arrays')
    defn=json.loads(Path(C['inputs']['width_definition']).read_text());EDGES=np.asarray(defn['edges_logQ'],dtype='f8');assert len(EDGES)==9
    cal=json.loads(Path(C['inputs']['CAL_metadata']).read_text())
    actual=load('actual',['source_waveforms_V','source_ids']);meta=load('metadata',['source_ids','composition'])
    assert np.array_equal(actual['source_ids'],meta['source_ids']);src=actual['source_waveforms_V'];ids=actual['source_ids'];comp=meta['composition']
    real=load('real',['real_V','real_ids','real_groups','real_QC_valid','reference_V','reference_ids','reference_groups'])
    saved=load('selection',[f'phase{p}_{j}'for p in[0,1]for j in[0,1,2]])
    selection={p:tuple(saved[f'phase{p}_{j}']for j in[0,1,2])for p in[0,1]}
    for p in [0,1]:
        for ix in selection[p]:assert len(ix)==(600 if p==0 else 250)and np.unique(ix).size==len(ix)and np.min(ix)>=0 and np.max(ix)<2000
    for j in range(3):assert not set(selection[0][j])&set(selection[1][j])
    srcADC,ad=adc(src)
    domains={'SOURCE_INPUT':data(src,ids,np.arange(2000),'G4_SOURCE'),
      'SOURCE_ONCE_ADC':data(srcADC,ids,np.arange(2000),'G4_SOURCE'),
      'REAL_DEV':data(real['real_V'],real['real_ids'],real['real_groups'],'R',real['real_QC_valid']),
      'REAL_RAW_REFERENCE':data(real['reference_V'],real['reference_ids'],real['reference_groups'],'R')}
    source_baseline={'RAW':domains['SOURCE_INPUT']};reference_baseline={'RAW':domains['REAL_RAW_REFERENCE']};components={};initials={};ADC={'SOURCE_ONCE_ADC':ad}
    endpoint_saved={}
    keys=['pre_ADC_V','waveforms_V','base_signal_V','projected_initial_Gaussian_f64_V','learned_flow_increment_f64_V','final_cast_rounding_f64_V','initial_state_f64_V','base_pre_ADC_V']
    identity={}
    for name in C['arms']:
        isreal=name.endswith('REAL_CONTROL');idkey='reference_ids'if isreal else'source_ids'
        a=load(name,keys+[idkey]);endpoint_saved[name]=a
        expected=real['reference_ids']if isreal else ids;assert np.array_equal(a[idkey],expected),(name,'saved endpoint ID order')
        original=real['reference_V']if isreal else src;initial,row=decompose(a,original,name);components[name]=row;initials[name]=initial
        groups=real['reference_groups']if isreal else np.arange(2000);ns='R'if isreal else'G4_SOURCE'
        domains[name]=data(a['waveforms_V'],expected,groups,ns)
        pre_name=name+'_PREADC';domains[pre_name]=data(a['pre_ADC_V'],expected,groups,ns)
        (reference_baseline if isreal else source_baseline)['SHORT'if name.startswith('SHORT')else'LONG']=domains[name]
        identity[name]={'input_ID_order_match':True,'input_S_exact_float32_match':True,'shape':[2000,480]}
        write('COMPONENT_DEFINITION_CHECK.json',components);progress('component '+name)
    common_initial={}
    for short,long,label,original,expected,groups,ns,target in[
      ('SHORT_2950_HEUN64','LONG_50000_HEUN64','SOURCE',src,ids,np.arange(2000),'G4_SOURCE',source_baseline),
      ('SHORT_2950_REAL_CONTROL','LONG_50000_REAL_CONTROL','REAL',real['reference_V'],real['reference_ids'],real['reference_groups'],'R',reference_baseline)]:
        a,b=endpoint_saved[short],endpoint_saved[long]
        common_initial[label]={'S_bitwise_equal':bool(np.array_equal(a['base_signal_V'],b['base_signal_V'])),
          'G_bitwise_equal':bool(np.array_equal(a['projected_initial_Gaussian_f64_V'],b['projected_initial_Gaussian_f64_V'])),
          'G_maxabs_difference_V':float(np.max(np.abs(a['projected_initial_Gaussian_f64_V']-b['projected_initial_Gaussian_f64_V']))),
          'initial_maxabs_difference_V':float(np.max(np.abs(initials[short]-initials[long])))}
        # Both initials are retained if unexpectedly unequal. No new inference or averaging.
        init=initials[short];post,aADC=adc(init);ADC[label+'_INITIAL']=aADC
        domains[label+'_INITIAL_PREADC']=data(init,expected,groups,ns);domains[label+'_INITIAL_POSTADC']=data(post,expected,groups,ns)
        target['INITIAL_PREADC']=domains[label+'_INITIAL_PREADC'];target['INITIAL_POSTADC']=domains[label+'_INITIAL_POSTADC']
        if not common_initial[label]['G_bitwise_equal']:
            lp,la=adc(initials[long]);ADC[label+'_LONG_INITIAL']=la
            domains[label+'_LONG_INITIAL_POSTADC']=data(lp,expected,groups,ns);target['LONG_INITIAL_POSTADC']=domains[label+'_LONG_INITIAL_POSTADC']
    for key,name,isreal,target in [('BASE','SOURCE_BASE',False,source_baseline),('BASE_REAL','REAL_BASE',True,reference_baseline)]:
        try:
            idkey='reference_ids'if isreal else'source_ids';a=load(key,['waveforms_V',idkey]);expected=real['reference_ids']if isreal else ids
            assert np.array_equal(a[idkey],expected),(name,'old BASE ID order')
            groups=real['reference_groups']if isreal else np.arange(2000)
            domains[name]=data(a['waveforms_V'],expected,groups,'R'if isreal else'G4_SOURCE');target['BASE']=domains[name]
        except(FileNotFoundError,KeyError)as e:identity[name]={'defined':False,'reason':repr(e)}
    inputcheck={'PID':os.getpid(),'UTC':utc(),'original_phase_rows_reused':True,'all_original_rows_retained':True,
     'selection_lengths':{f'phase{p}_{j}':len(v)for p,items in selection.items()for j,v in enumerate(items)},
     'endpoint_identity':identity,'short_long_initial_comparison':common_initial,'source_composition_counts':{str(k):int((comp==k).sum())for k in np.unique(comp)},
     'eight_charge_edges_logQ':EDGES,'CAL_metadata_read_only_definition':str(C['inputs']['CAL_metadata']),
     'CAL_waveform_decode':0,'inputs_and_keys':C['input_key_contract']}
    write('INPUT_IDENTITY_CHECK.json',inputcheck)
    populations={};population_names=['SOURCE_INPUT','SOURCE_ONCE_ADC','LONG_50000_HEUN64','REAL_RAW_REFERENCE','REAL_DEV']
    rd=domains['REAL_DEV'];chargeprivate={};sourcebin=domains['SOURCE_INPUT']['qbin']
    for name in population_names:
        d=domains[name];q=d['features']['Qtotal_mVns'];qr=rd['features']['Qtotal_mVns']
        populations[name]={'counts':q_counts(d),'Qtotal_mVns_all_rows':ev.summary(q),
          'W1_KS_vs_REAL_DEV_all_finite':finite_distance(q,qr),'W1_KS_vs_REAL_DEV_QC':finite_distance(q[d['valid']],qr[rd['valid']]),
          'composition_observed_Qbin':None,'composition_original_source_Qbin':None}
        chargeprivate[name+'__Qtotal_mVns']=q
        if name.startswith('SOURCE')or name=='LONG_50000_HEUN64':
            populations[name]['composition_observed_Qbin']={str(k):[int(((comp==k)&(d['qbin']==i)).sum())for i in range(8)]for k in np.unique(comp)}
            populations[name]['composition_original_source_Qbin']={str(k):[int(((comp==k)&(sourcebin==i)).sum())for i in range(8)]for k in np.unique(comp)}
    write('POPULATION_STATISTICS.json',{'domains':populations,'edges_logQ':EDGES,'all_finite_charge_rows_not_reweighted':True})
    np.savez_compressed(OUT/'POPULATION_FEATURES_PRIVATE.npz',**chargeprivate,source_original_Qbin=sourcebin,source_composition=comp,
                        source_ids=ids,real_ids=real['real_ids'],reference_ids=real['reference_ids'])
    baseline={'SOURCE':{},'REAL_REFERENCE':{}};baselineprivate={};pairdeltas={}
    for label,pool in [('SOURCE',source_baseline),('REAL_REFERENCE',reference_baseline)]:
        for step,d in pool.items():
            baseline[label][step]=baseline_description(d,rd)
            for k in ['baseline_RMS_V','ACF1']:baselineprivate[label+'__'+step+'__'+k]=d['features'][k]
        names=list(pool)
        for a,b in [('INITIAL_POSTADC','RAW'),('SHORT','INITIAL_POSTADC'),('LONG','INITIAL_POSTADC'),('LONG','SHORT')]:
            pairdeltas[label+'__'+a+'minus'+b]={k:ev.summary(pool[a]['features'][k]-pool[b]['features'][k])for k in ['baseline_RMS_V','ACF1']}
    for k in ['baseline_RMS_V','ACF1']:baselineprivate['REAL_DEV__'+k]=rd['features'][k]
    baseline['REAL_DEV']=baseline_description(rd,rd)
    write('BASELINE_PATH_STATISTICS.json',{'paths':baseline,'paired_row_changes':pairdeltas,'ADC':ADC,
      'initials':common_initial,'paired_row_changes_not_causal_intervention':True})
    np.savez_compressed(OUT/'BASELINE_FEATURES_PRIVATE.npz',**baselineprivate)
    original_matrix=dc.matrix
    dc.matrix=lambda d,v: d['features']['Qtotal_mVns'][:,None].copy()if v=='charge_only'else original_matrix(d,v)
    classifiers={};predictions={}
    jobs=[('CHARGE_SOURCE','SOURCE_INPUT',0,'charge_only'),('CHARGE_LONG','LONG_50000_HEUN64',0,'charge_only'),
          ('CHARGE_RAW_REFERENCE','REAL_RAW_REFERENCE',2,'charge_only'),('BASELINE_SOURCE_INITIAL','SOURCE_INITIAL_POSTADC',0,'baseline_local'),
          ('BASELINE_REAL_INITIAL','REAL_INITIAL_POSTADC',2,'baseline_local')]
    for slot,name,j,view in jobs:
        d=domains[name];tr,te=selection[0][j],selection[1][j];rt,re=selection[0][1],selection[1][1]
        assert not set(d['groups'][tr])&set(d['groups'][te]);assert not set(rd['groups'][rt])&set(rd['groups'][re])
        def persist(row,private):
            np.savez_compressed(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'),**private,train_candidate=tr,holdout_candidate=te,train_REAL=rt,holdout_REAL=re)
            write('results/'+slot+'.json',row);progress('saved '+slot)
        dc.PERSIST_HGB_CALLBACK=persist
        row,private=dc.fit_one(d,rd,(tr,te),(rt,re),view,C,COUNT);COUNT['HGB_completed']+=1
        row.update(candidate=name,view=view,fixed_R26_phase=j,train_QC_valid_candidate=int(d['valid'][tr].sum()),
         holdout_QC_valid_candidate=int(d['valid'][te].sum()),train_QC_valid_REAL=int(rd['valid'][rt].sum()),holdout_QC_valid_REAL=int(rd['valid'][re].sum()),
         train_candidate_groups=len(np.unique(d['groups'][tr])),holdout_candidate_groups=len(np.unique(d['groups'][te])),
         train_REAL_groups=len(np.unique(rd['groups'][rt])),holdout_REAL_groups=len(np.unique(rd['groups'][re])),QC_rows_not_dropped=True)
        if view=='charge_only':
            lo,hi=row['heldout_CI']['balanced_accuracy_95CI']
            row['population_diagnostic']=('strong_difference_fixed_diagnostic'if lo>.60 else'not_detected_excess_by_this_diagnostic'if lo>=.40 and hi<=.60 else'uncertain')
        persist(row,private);classifiers[slot]=row;predictions[slot]=private
        write('CLASSIFIER_DIAGNOSTICS.json',classifiers);progress('completed '+slot)
        print(utc(),slot,'BA',row['balanced_accuracy'],'CI',row['heldout_CI']['balanced_accuracy_95CI'],flush=True)
    # Reuse the historical RAW REAL–REAL baseline probabilities, without fitting a sixth classifier.
    paired={'defined':False,'reason':'not checked'};rawrow={}
    try:
        sel=json.loads(Path(C['inputs']['RAW_RR_selection']).read_text())['indices']
        matches={str((name,part)):bool(np.array_equal(sel[name][part],selection[p][j]))for name,part,p,j in[
          ('TRAIN_REFERENCE','train',0,2),('TRAIN_REFERENCE','holdout',1,2),('REAL','train',0,1),('REAL','holdout',1,1)]}
        assert all(matches.values()),'original RAW RR indices differ'
        prefix='native|baseline_local|REAL_TRAIN_vs_DEV__';a=load('RAW_RR_prediction',[prefix+k for k in ['probability','label','cluster']])
        old={k:a[prefix+k]for k in ['probability','label','cluster']}
        expected=np.r_[real['reference_groups'][selection[1][2]],real['real_groups'][selection[1][1]]]
        assert np.array_equal(np.array([str(s).split('|',1)[1]for s in old['cluster']]),np.array([str(g)for g in expected])),'original RAW RR group order differs'
        old['cluster']=np.array(['R|'+str(g)for g in expected]);pair=predictions['BASELINE_REAL_INITIAL']
        paired=paired_bootstrap(pair,old);paired.update(reference='saved_RAW_REAL_REAL_baseline_local',selection_match=matches,new_reference_fit=False)
        rawrow={'balanced_accuracy':float(balanced_accuracy_score(old['label'],old['probability']>=.5)),
          'AUC':float(roc_auc_score(old['label'],old['probability'])),'heldout_CI':dc.bootstrap(old['label'],old['probability'],old['cluster'],500,2026100871),'old_predictions_no_refit':True}
        np.savez_compressed(OUT/'RAW_RR_BASELINE_REUSED_PREDICTIONS_PRIVATE.npz',**old)
    except(FileNotFoundError,KeyError,AssertionError)as e:paired={'defined':False,'reason':repr(e),'new_reference_fit':False}
    oldm=json.loads(Path(C['inputs']['R46_metrics']).read_text())
    oldbaseline={n:oldm['C2ST']['results'][n]['baseline_local']for n in ['BASE','BASE_REAL_CONTROL','SHORT_2950_HEUN64','LONG_50000_HEUN64','SHORT_2950_REAL_CONTROL','LONG_50000_REAL_CONTROL']}
    compact={'stage':C['stage'],'PID':os.getpid(),'UTC':utc(),'counts':COUNT,'population':populations,
      'classifiers':classifiers,'baseline_paths':baseline,'paired_baseline_feature_changes':pairdeltas,
      'variance_decomposition':components,'component_initial_comparison':common_initial,'initial_ADC':ADC,
      'RAW_RR_baseline_reused':rawrow,'REAL_initial_minus_RAW_RR_paired':paired,'old_baseline_C2ST_reused':oldbaseline,
      'all_original_rows_retained':True,'no_model_candidate_created':True,'original_acceptance_unchanged':True,
      'functional_decomposition_not_independent_noise_or_causal_intervention':True,'edges_logQ':EDGES,
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__}}
    write('metrics.json',compact);write('METRICS_COMPACT.json',compact)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':utc(),'elapsed_body_s':time.monotonic()-START,'counts':COUNT,
      'original_input_ID_order_and_selection_checked':True,'component_sum_and_covariance_sum_checked':True,
      'no_model_inference_ODE_training_EM':True,'science_success':True,'output_bytes_so_far':sum(p.stat().st_size for p in R.rglob('*')if p.is_file())})
    progress('all five diagnostics completed')
    assert COUNT['HGB_attempts']==COUNT['HGB_completed']==5
    print(json.dumps(ev.safe({'PID':os.getpid(),'UTC':utc(),'counters':COUNT,'allfive_completed':True})),flush=True)
except BaseException as e:
    write('DIAGNOSTIC_FAILURE.json',{'PID':os.getpid(),'UTC':utc(),'error':repr(e),'traceback':traceback.format_exc(),'counters':COUNT,
      'saved_fits_not_to_be_repeated':True});progress('failure');raise
