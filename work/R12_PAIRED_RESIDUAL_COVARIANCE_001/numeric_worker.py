from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import pathlib,json,io,time,datetime,traceback,signal,hashlib,platform
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text());os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
import numpy as np
import sklearn,scipy
import evaluation as ev
import diagnostic_core as dc
import assessment as aa
import energy_width as ew
import density as den
import covariance as cv
OUT=ROOT/'results';OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
COUNTER={'PCA_GMM_ridge_refits':0,'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'new_random_draws':0,'response_updates':0,'EVAL_read':0,'new_G4':0};CACHE={}
def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def alarm(a,b):raise TimeoutError('R12 numeric or seal wall limit')
signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(880,(datetime.datetime.fromisoformat(CFG['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-60)))
def read(k):
    if k not in CACHE:
        b=CFG['input_files'][k];p=pathlib.Path(CFG['remote_input_root'])/k
        with p.open('rb') as f:raw=f.read(b['bytes']+1)
        if len(raw)!=b['bytes']:raise RuntimeError('input size '+k)
        CACHE[k]=raw
    return CACHE[k]
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def inputs():
    v=np.frombuffer(read('REAL_V'),dtype='<f4',count=5200*480).reshape(5200,480).copy();qc=np.frombuffer(read('REAL_QC'),dtype='<u2',count=5200);g=np.frombuffer(read('REAL_groups'),dtype='<u2',count=5200);idx=np.frombuffer(read('REAL_original_index'),dtype='<u4',count=5200);roles=np.frombuffer(read('REAL_roles'),dtype='u1',count=5200)
    assert np.all(roles[:2000]==0) and np.all(roles[2000:]==1) and np.array_equal(g,idx//50)
    vg=json.loads(read('TRAIN_roles'))['REAL_internal_validation_artificial_groups'];fit=(qc[2000:]==0)&~np.isin(g[2000:],vg);val=(qc[2000:]==0)&np.isin(g[2000:],vg);assert fit.sum()==2592 and val.sum()==599
    assert not set(g[:2000])&set(g[2000:]) and not set(g[2000:][fit])&set(g[2000:][val]);ref=json.loads(read('CAL_reference'));edges=np.array(json.loads(read('width_definition'))['edges_logQ']);sel=json.loads(read('R7_selections'))
    with np.load(io.BytesIO(read('SIM_identity')),allow_pickle=False) as z:rids=z['real_event_ids'].copy()
    idsfit=np.array(['RAW_ORIGINAL_INDEX|'+str(i) for i in idx[2000:][fit]]);pick=ev.order(idsfit,'R7-RR-TRAIN-2000')[:2000];assert np.array_equal(idx[2000:][fit][pick],sel['TRAIN_reference_original_indices'])
    real=aa.data(v[:2000],rids,g[:2000],'REAL',ref,qc[:2000]);rfit=aa.data(v[2000:][fit],idsfit,g[2000:][fit],'TRAIN',ref);rr=aa.data(v[2000:][fit][pick],idsfit[pick],g[2000:][fit][pick],'TRAIN_REFERENCE',ref)
    old=json.loads((pathlib.Path(CFG['remote_R11_root'])/'metrics.json').read_text());write('DATA_ROLES.json',{'REAL_DEV':2000,'TRAIN_fit':2592,'TRAIN_val':599,'TRAIN_REFERENCE':2000,'same_acquisition_artificial_groups':True,'EVAL_read':0,'CAL_waveform_decode':0,'new_G4_parents':0,'old_R11_outputs_reused':True})
    return v[2000:][val].astype('f8'),{'REAL':real},rfit,rr,ref,edges,sel,old
def c2st(domains,names,rr,sel):
    sp={k:(np.array(v['train'],dtype='i4'),np.array(v['holdout'],dtype='i4')) for k,v in sel['indices'].items()};chosen={};counts={}
    for phase in [0,1]:
        g=sp['GEN'][phase];g=g[np.logical_and.reduce([domains[n]['valid'][g] for n in names])];r=sp['REAL'][phase];r=r[domains['REAL']['valid'][r]];t=sp['TRAIN_REFERENCE'][phase];t=t[rr['valid'][t]];n=min(len(g),len(r),len(t));chosen[phase]=(g[:n],r[:n],t[:n]);counts['train' if phase==0 else 'holdout']={'N_each':n,'candidate_common_valid':len(g),'REAL':len(r),'TRAIN_REFERENCE':len(t)}
    out={};pred={};paired={};private={};support=counts['train']['N_each']>=150 and counts['holdout']['N_each']>=75 and len(names)==4
    if support:
        gs=(chosen[0][0],chosen[1][0]);rs=(chosen[0][1],chosen[1][1]);ts=(chosen[0][2],chosen[1][2])
        for n in names+['REAL_REAL']:
            d=rr if n=='REAL_REAL' else domains[n];selection=ts if n=='REAL_REAL' else gs;assert not set(d['groups'][selection[0]])&set(d['groups'][selection[1]]);out[n]={};pred[n]={}
            for view in CFG['C2ST_views']:
                out[n][view],pred[n][view]=dc.fit_one(d,domains['REAL'],selection,rs,view,CFG,COUNTER);private.update({n+'__'+view+'__'+k:v for k,v in pred[n][view].items()});log(f'C2ST {n} {view} BA={out[n][view]["balanced_accuracy"]:.6f}')
        for new,spec in CFG['source_models'].items():paired[new]={view:aa.paired_bootstrap(pred[new][view],pred[spec['name']][view],CFG) for view in CFG['C2ST_views']}
    else:out={'unsupported':True,'reason':'four arms unavailable or fixed pool below150/75'}
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private,**{'phase'+str(k)+'_'+str(j):v for k,a in chosen.items() for j,v in enumerate(a)})
    return {'supported':support,'counts':counts,'results':out,'paired_new_minus_old':paired,'original_R7_pools_only':True,'same_N_for_all':True,'RR_is_same_acquisition_TRAIN_DEV_sampling_reference_not_equivalence':True}
def widths(domains,names,old,edges):
    tables={n:old['energy_conditional_width']['tables'][n] for n in ['REAL','R3B','RAW_PCA_GMM','FACTOR_PCA_GMM']};errors=[]
    for n in CFG['models']:
        if n not in domains:continue
        COUNTER['density_description_slots']+=8
        if COUNTER['density_description_slots']>16:raise RuntimeError('widthslot limit')
        d=domains[n];use=d['valid'];t=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges);COUNTER['density_EM_attempts']+=sum(r['fit_attempts'] for r in t['slices']);tables[n]=t
    sets=[('all_four',names)]+[(new+'_pair',[spec['name'],new]) for new,spec in CFG['source_models'].items() if new in domains]
    for label,ns in sets:
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in ['REAL']+ns)];errors.append({'scope':label,'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in bins])) if bins else None for n in ns}})
    return {'tables':tables,'edges_logQ':edges,'common_support_errors':errors,'old_tables_reused_without_fitting':True,'not_particle_truth':True}
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

def run():
    log(f'R12 PID={os.getpid()} CPU4');val,domains,rfit,rr,ref,edges,sel,old=inputs();models={};names=[];decompositions={}
    for newname,spec in CFG['source_models'].items():
        source=spec['name'];p=pathlib.Path(CFG['remote_R11_root'])/source;row=old['representations'][source];assert row['selected_components']==spec['selected_K'];rep=npz(p/'representation.npz');rep['kind']=str(rep['kind'].item());gmm=npz(p/f'GMM_{spec["selected_K"]}.npz');inp=npz(p/'GENERATION_INPUTS.npz');saved=npz(p/'generated.npz');assert np.array_equal(inp['draw_ids'],saved['draw_ids']);pairtype=source+'-PAIR'
        domains[source]=aa.data(saved['waveforms_V'],saved['draw_ids'],np.arange(2000),pairtype,ref);names.append(source)
        try:newrep,S,check=cv.recover(rep,CFG)
        except (ValueError,np.linalg.LinAlgError) as e:models[newname]={'status':'failed_covariance_recovery','error':repr(e)};continue
        folder=ROOT/newname;folder.mkdir(exist_ok=True);np.savez_compressed(folder/'representation.npz',**newrep);np.savez_compressed(folder/f'GMM_{spec["selected_K"]}.npz',**gmm);np.savez_compressed(folder/'GENERATION_INPUTS.npz',**inp);np.savez_compressed(folder/'EMPIRICAL_COVARIANCE.npz',S=S)
        unchanged={k:bool(np.array_equal(rep[k],newrep[k])) for k in rep if k not in ['tail_cov','tail_cholesky']};assert all(unchanged.values());check['all_non_covariance_parameters_unchanged']=unchanged
        oldval,oldscore=cv.validate(val,rep,gmm);newval,newscore=cv.validate(val,newrep,gmm);target=next(f['complete_validation_NLL'] for f in row['fits'] if f['components']==spec['selected_K']);assert abs(oldval['complete_NLL']-target)<1e-7
        mean,head,x=cv.fixed_mean(rep,gmm,inp,CFG['generation_chunk']);newmean,newhead,newx=cv.fixed_mean(newrep,gmm,inp,CFG['generation_chunk']);assert np.array_equal(mean,newmean) and np.array_equal(head,newhead) and np.array_equal(x,newx)
        pre=den.invert_samples(newrep,gmm,inp,CFG['generation_chunk']);post,adc=den.adc(pre);COUNTER['main_generation_rows']+=len(post);np.savez_compressed(folder/'generated.npz',pre_ADC_V=pre,waveforms_V=post,draw_ids=inp['draw_ids']);np.savez_compressed(folder/'VALIDATION_SCORES_PRIVATE.npz',old=oldscore,new=newscore)
        decomposition,private=cv.decomposition(rep,S,mean,saved['pre_ADC_V'],pre);np.savez_compressed(folder/'VARIANCE_PRIVATE.npz',**private);decompositions[newname]=decomposition
        models[newname]={'status':'complete','source':source,'selected_K_frozen':spec['selected_K'],'covariance_recovery':check,'validation_old':oldval,'validation_new':newval,'validation_NLL_new_minus_old':newval['complete_NLL']-oldval['complete_NLL'],'old_NLL_matches_R11':True,'pairing':{'all2000_actual_component_headwhite_tailwhite_reused':True,'same_head_bitwise':True,'same_conditional_mean_bitwise':True,'only_covariance_and_cholesky_changed':True,'migration_test':False},'generation':{'rows':2000,'pre_ADC_payload_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest(),**adc}}
        domains[newname]=aa.data(post,inp['draw_ids'],np.arange(2000),pairtype,ref);names.append(newname);write('MODEL_PROGRESS.json',models);log(newname+' recovered '+str(check['alpha']))
    stats,dist=ev.characterize(domains,ref);local={n:dc.native_statistics(d,ref) for n,d in domains.items()};cs=c2st(domains,names,rr,sel);wd=widths(domains,names,old,edges)
    nn=nearest({n:domains[n] for n in CFG['models'] if n in domains},rfit);nn['old_R11_and_REAL_reused']=old['nearest_neighbours']
    features={}
    for n,d in domains.items():
        features.update({n+'__'+k:v for k,v in {**d['features'],**d['extra']}.items()});features[n+'__valid']=d['valid'];features[n+'__region']=d['region']
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**features)
    metrics={'status':'complete' if len(names)==4 else 'incomplete_model_failure','models':models,'baseline_and_curvature_variance_decomposition':decompositions,'domains':stats,'domain_vs_REAL':dist,'local':local,'C2ST':cs,'energy_conditional_width':wd,'nearest_neighbours':nn,'counters':COUNTER,'versions':{'Python':platform.python_version(),'NumPy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__},'elapsed_s':time.monotonic()-START,'EVAL_read':0,'same_acquisition_development':True,'target_density_not_G4_mapping':True};write('metrics.json',metrics)
    from plotting import plots
    plots(domains,metrics,ref,ROOT)
    write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':metrics['status'],'counters':COUNTER,'threads':4});log('R12 complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
