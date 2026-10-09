from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='4'
import pathlib,json,io,time,datetime,traceback,signal,hashlib,platform
ROOT=pathlib.Path(__file__).resolve().parent; START=time.monotonic(); CFG=json.loads((ROOT/'config.json').read_text())
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
import numpy as np
import sklearn,scipy
import evaluation as ev
import diagnostic_core as dc
import assessment as aa
import energy_width as ew
import density as den
OUT=ROOT/'results';OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
COUNTER={'GMM_fit_attempts':0,'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'response_updates':0,'EVAL_read':0,'new_G4':0}
CACHE={};READS=[]
def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def write(name,obj):
    p=ROOT/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(ev.safe(obj),ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def remaining():
    deadline=datetime.datetime.fromisoformat(CFG['deadline_UTC']);return min(CFG['numeric_wall_limit_s']-(time.monotonic()-START),(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-60)
def alarm(s,f):raise TimeoutError('R11 fixed time budget exhausted')
signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,remaining()))
def read(key):
    if key not in CACHE:
        b=CFG['input_files'][key];p=pathlib.Path(CFG['remote_input_root'])/key
        with p.open('rb') as f:raw=f.read(b['bytes']+1)
        if len(raw)!=b['bytes']:raise RuntimeError('input size '+key)
        CACHE[key]=raw;READS.append({'key':key,'returned_bytes':len(raw),'original_version_sha256':b['sha256'],'input_or_migration_rehashed':False})
    return CACHE[key]
def inputs():
    real=np.frombuffer(read('REAL_V'),dtype='<f4',count=5200*480).reshape(5200,480).copy()
    qc=np.frombuffer(read('REAL_QC'),dtype='<u2',count=5200).copy();groups=np.frombuffer(read('REAL_groups'),dtype='<u2',count=5200).copy()
    idx=np.frombuffer(read('REAL_original_index'),dtype='<u4',count=5200).copy();roles=np.frombuffer(read('REAL_roles'),dtype='u1',count=5200)
    assert np.all(roles[:2000]==0) and np.all(roles[2000:]==1) and np.array_equal(groups,idx//50)
    vg=json.loads(read('TRAIN_roles'))['REAL_internal_validation_artificial_groups'];fit=(qc[2000:]==0)&~np.isin(groups[2000:],vg);val=(qc[2000:]==0)&np.isin(groups[2000:],vg)
    assert fit.sum()==2592 and val.sum()==599 and not set(groups[2000:][fit])&set(groups[2000:][val])
    assert not set(groups[:2000])&set(groups[2000:]);ref=json.loads(read('CAL_reference'));edges=np.array(json.loads(read('width_definition'))['edges_logQ']);selection=json.loads(read('R7_selections'))
    with np.load(io.BytesIO(read('SIM_identity')),allow_pickle=False) as z:rids=z['real_event_ids'].copy()
    with np.load(io.BytesIO(read('generated')),allow_pickle=False) as z:r3b=z['waveforms_V'].copy();sourceids=z['sim_event_ids'].copy()
    with np.load(pathlib.Path(CFG['remote_history_root'])/'generated.npz',allow_pickle=False) as z:oldgen=z['waveforms_V'].copy();oldids=z['draw_ids'].copy()
    oldmetrics=json.loads((pathlib.Path(CFG['remote_history_root'])/'metrics.json').read_text())
    idsfit=np.array(['RAW_ORIGINAL_INDEX|'+str(v) for v in idx[2000:][fit]]);pick=ev.order(idsfit,'R7-RR-TRAIN-2000')[:2000]
    assert np.array_equal(idx[2000:][fit][pick],selection['TRAIN_reference_original_indices'])
    domains={'REAL':aa.data(real[:2000],rids,groups[:2000],'REAL',ref,qc[:2000]),'R3B':aa.data(r3b,sourceids,np.arange(2000),'R3B',ref),'R10_FIXED':aa.data(oldgen,oldids,np.arange(2000),'R10DRAW',ref)}
    rfit=aa.data(real[2000:][fit],idsfit,groups[2000:][fit],'TRAIN',ref);rr=aa.data(real[2000:][fit][pick],idsfit[pick],groups[2000:][fit][pick],'TRAIN_REFERENCE',ref)
    facts={'REAL_DEV':2000,'REAL_TRAIN_source':3200,'REAL_TRAIN_QC':3191,'REAL_TRAIN_fit':2592,'REAL_TRAIN_val':599,'fit_val_artificial_groups_disjoint':True,'DEV_TRAIN_artificial_groups_disjoint':True,'REAL_REAL_reference':2000,'CAL_waveforms_decoded':0,'EVAL_read':0,'all_same_acquisition':True,'not_new_blind_validation':True,'no_G4_inputs_or_parent_correspondence_for_new_draws':True}
    write('DATA_ROLES.json',facts);np.savez_compressed(OUT/'ROLES_PRIVATE.npz',fit_original_index=idx[2000:][fit],val_original_index=idx[2000:][val],TRAIN_reference_original_index=idx[2000:][fit][pick])
    return real[2000:][fit].astype('f8'),real[2000:][val].astype('f8'),domains,rfit,rr,ref,edges,selection,oldmetrics
def nearest(domains,reference):
    out={'TRAIN_reference_N':len(reference['V']),'all_2000_queries_no_QC_resampling':True,'not_proof_of_no_memory':True};priv={};examples={}
    for view,key in [('V','V'),('own_shape','shape')]:
        R=np.asarray(reference[key],dtype='f8');rf=np.isfinite(R).all(axis=1);ri=np.flatnonzero(rf);R=R[rf];r2=(R*R).sum(axis=1)
        for name,d in domains.items():
            a=np.asarray(d[key],dtype='f8');ok=np.isfinite(a).all(axis=1);ix=np.flatnonzero(ok);ds=np.full(len(a),np.nan);nn=np.full(len(a),-1,dtype='i4');exact=np.zeros(len(a),bool)
            for start in range(0,len(ix),64):
                sl=ix[start:start+64];X=a[sl];sq=((X*X).sum(axis=1)[:,None]+r2[None,:]-2*X@R.T)/480;j=np.argmin(sq,axis=1);ds[sl]=np.sqrt(np.maximum(sq[np.arange(len(sl)),j],0));nn[sl]=ri[j];exact[sl]=np.array([np.array_equal(x,R[k]) for x,k in zip(X,j)])
            out.setdefault(view,{})[name]={'all_query_n':len(a),'defined_query_n':len(ix),'undefined_query_n':len(a)-len(ix),'TRAIN_reference_defined_n':len(R),'RMS_distance':ev.summary(ds),'exact_to_nearest_count':int(exact.sum()),'distance_le1e_6':int((ds<=1e-6).sum()),'distance_le1e_4':int((ds<=1e-4).sum()),'threshold_units':'V' if view=='V' else 'dimensionless','thresholds_descriptive':True}
            priv[name+'__'+view+'__distance']=ds;priv[name+'__'+view+'__neighbour_index']=nn
            chosen=ix[ev.order(d['ids'][ix],'R11-NN-example')[:3]];examples[name+'__'+view+'__query']=a[chosen];examples[name+'__'+view+'__neighbour']=reference[key][nn[chosen]];priv[name+'__'+view+'__example_query_index']=chosen
    np.savez_compressed(OUT/'NN_PRIVATE.npz',**priv);np.savez_compressed(OUT/'NN_EXAMPLES.npz',**examples);return out
def c2st(domains,rr,sel,names):
    sp={k:(np.array(v['train'],dtype='i4'),np.array(v['holdout'],dtype='i4')) for k,v in sel['indices'].items()};s={};counts={}
    for phase in range(2):
        g=sp['GEN'][phase];keep=np.logical_and.reduce([domains[n]['valid'][g] for n in names]);g=g[keep]
        r=sp['REAL'][phase];r=r[domains['REAL']['valid'][r]];t=sp['TRAIN_REFERENCE'][phase];t=t[rr['valid'][t]]
        n=min(len(g),len(r),len(t));s[phase]=(g[:n],r[:n],t[:n]);counts['train' if phase==0 else 'holdout']={'N_each':n,'available_GEN_common_QC':len(g),'REAL':len(r),'TRAIN_REFERENCE':len(t)}
    result={};pred={};support=counts['train']['N_each']>=CFG['C2ST_min_train'] and counts['holdout']['N_each']>=CFG['C2ST_min_holdout']
    if support:
        gs=(s[0][0],s[1][0]);rs=(s[0][1],s[1][1]);ts=(s[0][2],s[1][2])
        for n in names+['REAL_REAL']:
            d=rr if n=='REAL_REAL' else domains[n];chosen=ts if n=='REAL_REAL' else gs
            assert not set(d['groups'][chosen[0]])&set(d['groups'][chosen[1]])
            result[n]={}
            for view in CFG['C2ST_views']:
                result[n][view],p=dc.fit_one(d,domains['REAL'],chosen,rs,view,CFG,COUNTER)
                pred.update({n+'__'+view+'__'+k:v for k,v in p.items()});log(f'C2ST {n} {view} BA={result[n][view]["balanced_accuracy"]:.6f}')
    else:result={'unsupported':True,'reason':'fixed R7 buckets below150/75 after common QC'}
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**pred,**{'phase'+str(k)+'_'+str(j):v for k,a in s.items() for j,v in enumerate(a)})
    return {'counts':counts,'supported':support,'original_R7_positions_only':True,'same_train_and_holdout_counts_for_all_arms_and_RR':True,'RR_is_TRAIN_reference_vs_DEV_same_acquisition':True,'RR_not_equivalence_threshold':True,'results':result}
def widths(domains,names,old,edges):
    tables={'REAL':old['B']['width_tables']['REAL'],'R3B':old['B']['width_tables']['R3B'],'R10_FIXED':old['B']['width_tables']['FIXED_SOURCE_STOCHASTIC']}
    for name in names:
        COUNTER['density_description_slots']+=8
        if COUNTER['density_description_slots']>16:raise RuntimeError('description slots cap')
        d=domains[name];use=d['valid'];table=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges);COUNTER['density_EM_attempts']+=sum(r['fit_attempts'] for r in table['slices']);tables[name]=table
    errors=[]
    def compare(named,label):
        for c in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][c][field] is not None for n in ['REAL']+named)]
                errors.append({'scope':label,'component':c,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][c][field]-tables['REAL']['slices'][i]['components'][c][field]) for i in bins])) if bins else None for n in named}})
    compare(names,'two_new_arms_common_support');compare(['R3B','R10_FIXED']+names,'historical_and_new_common_support')
    return {'tables':tables,'edges_logQ':edges,'common_support_errors':errors,'old_REAL_R3B_R10_tables_reused_without_refit':True,'not_particle_truth_or_CAL_refit':True}
def run():
    log(f'R11 PID={os.getpid()} CPU4 sklearn={sklearn.__version__} NumPy={np.__version__}')
    fit,val,domains,rfit,rr,ref,edges,sel,old=inputs();model_results={};names=[];fitstart=time.monotonic()
    signal.setitimer(signal.ITIMER_REAL,max(.001,min(remaining(),CFG['fit_soft_limit_s'])))
    for index,name in enumerate(CFG['models']):
        log('fit representation '+name);rep,x,meta=den.representation(fit,name,CFG);folder=ROOT/name;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'representation.npz',**rep)
        rec={'TRAIN_fit':den.reconstruction(fit,rep,CFG['diagnostic_ranks']),'TRAIN_val':den.reconstruction(val,rep,CFG['diagnostic_ranks'])}
        gs,records,chosen,scores,checks=den.fit_models(val,rep,x,CFG,index,COUNTER,log)
        assert all(c['head_logpdf_formula_maxabs_nat']<=CFG['density_check_abs_nat'] for c in checks)
        for k,g in gs.items():np.savez_compressed(folder/f'GMM_{k}.npz',**g)
        np.savez_compressed(folder/'VALIDATION_SCORES_PRIVATE.npz',**{'K'+str(k):v for k,v in scores.items()})
        row={'representation':meta,'reconstruction':rec,'fits':records,'selected_components':chosen,'selection':'strict lowest complete599-val NLL among converged; exact ties smallerK','density_formula_checks':checks,'generation':None}
        if chosen is not None:
            row['selfcheck']=den.selfcheck(val,rep,gs[chosen],CFG);sample=den.sample_inputs(rep,gs[chosen],2000,CFG['seed']+5000+10*index);pre=den.invert_samples(rep,gs[chosen],sample,CFG['generation_chunk']);post,adc=den.adc(pre)
            ids=np.array([f'R11-{name}-DRAW|{j:06d}' for j in range(2000)]);np.savez_compressed(folder/'GENERATION_INPUTS.npz',**sample,draw_ids=ids);np.savez_compressed(folder/'generated.npz',pre_ADC_V=pre,waveforms_V=post,draw_ids=ids)
            COUNTER['main_generation_rows']+=2000;row['generation']={'draws':2000,'G4_parents':0,'seed':int(sample['seed']),'pre_ADC_payload_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest(),**adc}
            domains[name]=aa.data(post,ids,np.arange(2000),name+'-DRAW',ref);names.append(name)
        else:row['undefined_reason']='all fixed GMM candidates failed or nonconverged'
        model_results[name]=row;write('FIT_PROGRESS.json',{'models':model_results,'counters':COUNTER});log(name+' selected '+str(chosen))
    fitelapsed=time.monotonic()-fitstart;signal.setitimer(signal.ITIMER_REAL,max(.001,remaining()))
    stats,dist=ev.characterize(domains,ref);local={n:dc.native_statistics(d,ref) for n,d in domains.items()};cs=c2st(domains,rr,sel,names) if names else {'unsupported':True,'reason':'no eligible generated arm'}
    wd=widths(domains,names,old,edges);nn=nearest({n:domains[n] for n in ['REAL']+names},rfit)
    feature_arrays={}
    for n,d in domains.items():
        feature_arrays.update({n+'__'+k:v for k,v in {**d['features'],**d['extra']}.items()});feature_arrays[n+'__valid']=d['valid'];feature_arrays[n+'__region']=d['region']
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**feature_arrays)
    write('INPUTS_LOADED.json',READS);write('metrics.json',{'status':'complete','representations':model_results,'domains':stats,'domain_vs_REAL':dist,'local':local,'C2ST':cs,'energy_conditional_width':wd,'nearest_neighbours':nn,'counters':COUNTER,'fit_phase_s':fitelapsed,'elapsed_s':time.monotonic()-START,'versions':{'Python':platform.python_version(),'NumPy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__},'EVAL_read':0,'same_acquisition_DEV_not_blind':True,'target_density_models_not_G4_response':True,'continuous_density_not_exact_ADC_mass':True})
    from plotting import plots
    plots(domains,model_results,cs,wd,nn,ref,ROOT)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'fit_phase_s':fitelapsed,'status':'complete','counters':COUNTER,'threads':4,'EVAL_read':0});log('R11 complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:
    write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
