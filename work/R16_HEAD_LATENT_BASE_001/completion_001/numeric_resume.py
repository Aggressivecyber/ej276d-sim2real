from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pathlib,json,io,time,datetime,traceback,signal,hashlib,platform
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();CFG=json.loads((ROOT/'config.json').read_text());os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
def alarm(a,b):raise TimeoutError('R16 numeric or seal wall limit')
signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(1760,(datetime.datetime.fromisoformat(CFG['deadline_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-90)))
import numpy as np
import torch,sklearn,scipy
import evaluation as ev
import diagnostic_core as dc
import assessment as aa
import energy_width as ew
import density as den
import conditional_density as cd
import residual_flow as rf
import head_base as hb
torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
COUNTER={'new_head_base_fit_attempts':0,'head_base_fitting_cumulative_s':0.,'new_network_fit_attempts':0,'network_updates':0,'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,'new_uniform_draws':0,'new_PCA_ridge_residual_fits':0,'EVAL_read':0,'new_G4':0,'raw_waveform_decode':0};CACHE={}

def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def write(n,v):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n')
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
    addition=npz(pathlib.Path(CFG['remote_R14_root'])/'TRAIN_ADDITION.npz');valid=addition['QC_reason_u16']==0
    assert len(addition['original_index'])==18500 and valid.sum()==18455
    train=np.concatenate([v[2000:][fit].astype('f8'),addition['waveforms_V'][valid].astype('f8')]);fitidx=np.concatenate([idx[2000:][fit],addition['original_index'][valid]])
    assert len(train)==21047 and len(np.unique(fitidx))==21047 and not np.isin(fitidx//50,vg).any()
    assert np.array_equal(fitidx,np.fromfile(pathlib.Path(CFG['remote_R14_root'])/'expanded_fit_original_index.bin',dtype='<u4'))
    write('DATA_ROLES.json',{'TRAIN_fit':21047,'TRAIN_val':599,'REAL_DEV':2000,'old_fit_reference':2592,'TRAIN_REFERENCE':2000,'groups':422,'unchanged_R14_order':True,'same_acquisition_artificial_groups':True,'EVAL_read':0,'CAL_waveform_numeric_decode':0,'raw_long_waveform_decode':0,'new_G4':0,'frozen_representation_fit_N':2592})
    return train,v[2000:][val].astype('f8'),{'REAL':real},rfit,rr,ref,edges,sel


def nearest(domains,reference,label):
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
    np.savez_compressed(OUT/('NN_'+label+'_PRIVATE.npz'),**priv);np.savez_compressed(OUT/('NN_'+label+'_EXAMPLES.npz'),**examples);return out
def widths(domains,old15,edges,names):
    tables={'REAL':old15['energy_conditional_width']['tables']['REAL'],'STANDARD':old15['energy_conditional_width']['tables']['RESIDUAL_FLOW']}
    for n in names:
        if n=='STANDARD':continue
        d=domains[n];use=d['valid'];COUNTER['density_description_slots']+=8
        tab=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges);COUNTER['density_EM_attempts']+=sum(r['fit_attempts'] for r in tab['slices']);tables[n]=tab
    errors=[]
    for scope,ns in [('all_candidates',names)]+[(n,['STANDARD',n]) for n in names if n!='STANDARD']:
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in ['REAL']+ns)]
                errors.append({'scope':scope,'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in bins])) if bins else None for n in ns}})
    return {'tables':tables,'edges_logQ':edges,'common_support_errors':errors,'old_REAL_and_STANDARD_reused_no_fit':True,'not_particle_truth':True}

def c2st_resume(domains,rr,sel,prior,first):
    old=npz(first/'results/C2ST_PRIVATE.npz');gs=(old['phase0_0'],old['phase1_0']);rs=(old['phase0_1'],old['phase1_1']);ts=(old['phase0_2'],old['phase1_2']);original={k:np.array(v['train'] if phase==0 else v['holdout']) for phase in [] for k,v in sel['indices'].items()}
    for phase in [0,1]:
        assert np.array_equal(gs[phase],np.array(sel['indices']['GEN']['train' if phase==0 else 'holdout']))
        assert np.array_equal(rs[phase],np.array(sel['indices']['REAL']['train' if phase==0 else 'holdout']))
        assert len(gs[phase])==(600 if phase==0 else 250)
        assert domains['GMM8']['valid'][gs[phase]].all() and domains['REAL']['valid'][rs[phase]].all()
    prior=dict(prior);out=dict(prior['results']);private=dict(old);pred={};out['GMM8']={}
    for view in CFG['C2ST_views']:
        out['GMM8'][view],pred[view]=dc.fit_one(domains['GMM8'],domains['REAL'],gs,rs,view,CFG,COUNTER);private.update({'GMM8__'+view+'__'+k:v for k,v in pred[view].items()});log(f'C2ST GMM8 {view} BA={out["GMM8"][view]["balanced_accuracy"]:.6f}')
    paired=dict(prior['paired_contrasts'])
    for second in ['STANDARD','GAUSSIAN']:
        paired['GMM8_minus_'+second]={view:aa.paired_bootstrap(pred[view],{k:old[second+'__'+view+'__'+k] for k in ['probability','label','cluster']},CFG) for view in CFG['C2ST_views']}
    np.savez_compressed(OUT/'C2ST_PRIVATE.npz',**private);prior.update(results=out,paired_contrasts=paired,all_32_current_environment_refit_if_all_arms_finite=True,engineering_completion={'old24_classifiers_reused':True,'additional8_fits':True,'exact_original600_250_positions_unchanged':True,'all_GMM_selected_QC_valid':True})
    return prior

def widths_resume(domains,prior,edges):
    tables=dict(prior['tables']);d=domains['GMM8'];use=d['valid'];COUNTER['density_description_slots']+=8;tab=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges);COUNTER['density_EM_attempts']+=sum(r['fit_attempts'] for r in tab['slices']);tables['GMM8']=tab;errors=[]
    for scope,ns in [('all_candidates',CFG['all_comparison_names'])]+[(n,['STANDARD',n]) for n in CFG['models']]:
        for k in [0,1]:
            for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
                bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in ['REAL']+ns)];errors.append({'scope':scope,'component':k,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in bins])) if bins else None for n in ns}})
    return {'tables':tables,'edges_logQ':edges,'common_support_errors':errors,'old_REAL_STANDARD_and_completed_Gaussian_reused_no_fit':True,'not_particle_truth':True}

def run():
    global COUNTER
    log(f'R16 completion PID={os.getpid()} CPUfloat64; one never-initialized GMM fit')
    first=pathlib.Path(CFG['remote_R16_first_root']);prior=json.loads((first/'metrics.json').read_text());COUNTER=dict(prior['counters']);COUNTER['GMM_parameter_validation_failure_attempts']=1;COUNTER['executed_head_base_fits']=1
    train_v,val_v,domains,rfit,rr,ref,edges,sel=inputs();old14=pathlib.Path(CFG['remote_R14_root']);old15=pathlib.Path(CFG['remote_R15_root']);rep=npz(old14/'representation.npz');rep['kind']=str(rep['kind'].item());inp=npz(first/'ACTUAL_GENERATION_INPUTS.npz');latent=npz(first/'results/LATENT_PRIVATE.npz');tz=latent['TRAIN_z'];vz=latent['VAL_z'];tld=latent['TRAIN_logdet'];vld=latent['VAL_logdet'];del val_v
    head=cd.Flow(CFG['head_seed']).double();head.load_state_dict(torch.load(old14/'FLOW_EXPOSURES.pt',map_location='cpu',weights_only=True));head.requires_grad_(False);head.eval();tail=cd.ConditionalTail().double();tail.load_state_dict(torch.load(old14/'TAIL_EXPOSURES.pt',map_location='cpu',weights_only=True));tail.requires_grad_(False);tail.eval();flow=rf.ResidualFlow(CFG['seed']).double();flow.load_state_dict(torch.load(old15/'RESIDUAL_FLOW.pt',map_location='cpu',weights_only=True));flow.requires_grad_(False);flow.eval()
    COUNTER['new_head_base_fit_attempts']+=1;COUNTER['executed_head_base_fits']+=1;begin=time.monotonic();prior_alarm=signal.getitimer(signal.ITIMER_REAL)[0];signal.setitimer(signal.ITIMER_REAL,min(prior_alarm,CFG['fitting_limit_s']))
    try:p,fit=hb.fit_gmm(tz,CFG)
    finally:COUNTER['head_base_fitting_cumulative_s']+=time.monotonic()-begin;signal.setitimer(signal.ITIMER_REAL,max(.001,prior_alarm-(time.monotonic()-begin)))
    fit['elapsed_s']=time.monotonic()-begin;fits=dict(prior['head_base_fits']);fits['GMM8']=fit;fits['original_parameter_validation_failure']=prior['head_base_fits']['GMM8'];write('HEAD_BASE_FITS.json',fits);np.savez_compressed(ROOT/'GMM8_BASE.npz',**p)
    from scipy.stats import multivariate_normal
    from scipy.special import logsumexp
    ix=np.array([0,199,399,598]);sample=vz[ix];a=den.gmm_logpdf(sample,p);b=logsumexp(np.stack([np.log(w)+multivariate_normal.logpdf(sample,mean=mu,cov=cov) for w,mu,cov in zip(p['weights'],p['means'],p['covariances'])]),axis=0);ma=float(np.max(np.abs(a-b)));checks=dict(prior['base_density_checks']);checks['checks']=checks['checks']+[{'base':'GMM8','validation_positions':ix.tolist(),'manual_vs_scipy_mixture_maxabs_nat':ma,'pass':ma<=1e-7}];checks['pass']=all(r['pass'] for r in checks['checks']);write('BASE_DENSITY_CHECKS.json',checks)
    if not checks['pass']:raise RuntimeError('GMM new-base formula check failed')
    scores=npz(first/'results/DENSITY_SCORES_PRIVATE.npz');validations=dict(prior['validation']);train_scores=dict(prior['training_scores'])
    for prefix,z,ld,rows in [('VAL',vz,vld,validations),('TRAIN',tz,tld,train_scores)]:
        lp=den.gmm_logpdf(z,p);hp=lp+ld;tailp=scores[prefix+'_GAUSSIAN__tail_logdensity'];jac=scores[prefix+'_GAUSSIAN__Jacobian'];complete=hp+tailp-jac
        if not np.isfinite(complete).all():raise FloatingPointError('GMM complete score nonfinite')
        rows['GMM8']={'all_rows':len(z),'finite_rows':len(z),'complete_NLL':float(-complete.mean()),'mean_base_logdensity':float(lp.mean()),'mean_head_flow_logdet':float(ld.mean()),'mean_head_std_logdensity':float(hp.mean()),'mean_tail_logdensity':float(tailp.mean()),'mean_logJacobian':float(jac.mean()),'arithmetic':'CPUfloat64; fixed R15 conditional tail and FACTOR terms'}
        scores.update({prefix+'_GMM8__'+k:v for k,v in {'complete_logdensity':complete,'head_logdensity':hp,'base_logdensity':lp,'head_flow_logdet':ld,'tail_logdensity':tailp,'Jacobian':jac}.items()})
    np.savez_compressed(OUT/'DENSITY_SCORES_PRIVATE.npz',**scores);del scores,latent,tz,vz,tld,vld
    eligible=['STANDARD','GAUSSIAN']+(['GMM8'] if fit['eligible'] else []);order={n:i for i,n in enumerate(CFG['all_comparison_names'])};selected=min(eligible,key=lambda n:(validations[n]['complete_NLL'],order[n]));selection={'rule':'minimum complete599 validation NLL; exact tie STANDARD,GAUSSIAN,GMM8','eligible':eligible,'selected_base':selected,'DEV_selection':False};gaps={n:{'TRAIN_N':21047,'val_N':599,'TRAIN_NLL':train_scores[n]['complete_NLL'],'val_NLL':validations[n]['complete_NLL'],'val_minus_TRAIN_NLL':validations[n]['complete_NLL']-train_scores[n]['complete_NLL']} for n in validations}
    for name,path in [('STANDARD',old15/'RESIDUAL_FLOW/generated.npz'),('GAUSSIAN',first/'GAUSSIAN/generated.npz')]:
        saved=npz(path);assert np.array_equal(saved['draw_ids'],inp['draw_ids']);domains[name]=aa.data(saved['waveforms_V'],inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref)
    xx,zz,comp=hb.sample_heads('GMM8',p,head,inp,inp['GMM8_uniform'],CFG['generation_chunk']);inp.update(GMM8__x=xx,GMM8__latent=zz,GMM8__component=comp);np.savez_compressed(ROOT/'ACTUAL_GENERATION_INPUTS.npz',**inp);pre=rf.generate(flow,tail,xx,rep,inp,CFG['generation_chunk']);post,adc=den.adc(pre);COUNTER['main_generation_rows']+=len(pre);folder=ROOT/'GMM8';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'generated.npz',pre_ADC_V=pre,waveforms_V=post,draw_ids=inp['draw_ids']);models=dict(prior['models']);models['GMM8']={'status':'complete' if np.isfinite(pre).all() else 'nonfinite_generation','base_eligible':fit['eligible'],'generation':{'rows':2000,**adc,'pre_ADC_payload_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest()}};domains['GMM8']=aa.data(post,inp['draw_ids'],np.arange(2000),'DENSITY_DRAW',ref)
    contribution=dict(prior['head_only_full_PSD']);newcont,newpriv=hb.psd_contribution({'GMM8':domains['GMM8']},rep);contribution.update(newcont);oldpriv=npz(first/'results/HEAD_PSD_PRIVATE.npz');oldpriv.update(newpriv);np.savez_compressed(OUT/'HEAD_PSD_PRIVATE.npz',**oldpriv);domains['REAL']['head34']=hb.encoded_heads(domains['REAL']['V'],rep)[0]
    newstats,newdist=ev.characterize({'REAL':domains['REAL'],'GMM8':domains['GMM8']},ref);stats=dict(prior['domains']);stats['GMM8']=newstats['GMM8'];dist=dict(prior['domain_vs_REAL']);dist.update(newdist);local=dict(prior['local']);local['GMM8']=dc.native_statistics(domains['GMM8'],ref);cs=c2st_resume(domains,rr,sel,prior['C2ST'],first);wd=widths_resume(domains,prior['energy_conditional_width'],edges)
    queries={'GMM8':domains['GMM8']};nn=dict(prior['nearest_neighbours']);newnn=nearest(queries,rfit,'ORIGINAL');baseline=train_v[:,:100].mean(axis=1);delt=train_v-baseline[:,None];peak=delt.max(axis=1);large_ref={'V':train_v,'shape':delt/peak[:,None]};newlarge=nearest(queries,large_ref,'EXPANDED')
    for key,newval in [('original2592',newnn),('expanded_fit',newlarge)]:
        merged=dict(nn[key]);merged={k:(dict(v) if isinstance(v,dict) else v) for k,v in merged.items()}
        for view in ['V','own_shape']:merged[view].update(newval[view])
        nn[key]=merged
    for label in ['ORIGINAL','EXPANDED']:
        for suffix in ['PRIVATE','EXAMPLES']:
            current=npz(OUT/('NN_'+label+'_'+suffix+'.npz'));current.update(npz(first/'results'/('NN_'+label+'_'+suffix+'.npz')));np.savez_compressed(OUT/('NN_'+label+'_'+suffix+'.npz'),**current)
    del large_ref,delt,train_v
    features=npz(first/'results/FEATURES_PRIVATE.npz');d=domains['GMM8'];features.update({'GMM8__'+k:v for k,v in {**d['features'],**d['extra']}.items()});features.update(GMM8__valid=d['valid'],GMM8__region=d['region']);np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**features)
    metrics=dict(prior);metrics.update(status='complete',models=models,head_base_fits=fits,base_selection=selection,validation=validations,training_scores=train_scores,train_validation_gap=gaps,head_only_full_PSD=contribution,base_density_checks=checks,domains=stats,domain_vs_REAL=dist,local=local,C2ST=cs,energy_conditional_width=wd,nearest_neighbours=nn,counters=COUNTER,elapsed_s=time.monotonic()-START,engineering_completion={'initial_parameter_validation_failure_preserved':True,'GMM_no_previous_initialization_or_EM':True,'seed_adapter':'RandomState(MT19937(SeedSequence(20261009161)))','initial_native_science_s':CFG['prior_numeric_cumulative_s'],'Gaussian_output_and_successful_replay_reused':True,'old24_classifiers_same_positions_reused':True});write('metrics.json',metrics)
    from plotting import plots
    plots(domains,metrics,ref,ROOT);write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':'complete','counters':COUNTER,'environment':metrics['environment'],'threads':4,'original_failed_receipts_preserved':True});log('R16 completion '+json.dumps(COUNTER))
try:run()
except BaseException as e:write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
