from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import pathlib,json,io,hashlib,time,datetime,traceback
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();cfg=json.loads((ROOT/'config.json').read_text());os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
import evaluation as ev
import diagnostic_core as dc
import assessment as aa
import matching as ma
import energy_width as ew
import fixed_training as ft
import training as tr
import cuda_runtime as cu
cu.configure(cfg['seed'])
OUT=ROOT/'results';OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
COUNTER={'HGB_attempts':0,'density_EM_attempts':0,'G_updates':0,'D_updates':0,'main_generation_rows':0,'EVAL_read':0,'SSH_scientific_queries':'remote execution authorized; source queries 0','new_G4':0};CACHE={};READS=[]
def log(x):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),x,flush=True)
def write(name,value):
    p=ROOT/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(ev.safe(value),ensure_ascii=False,allow_nan=False,indent=2)+'\n')
def read(k):
    if k not in CACHE:
        b=cfg['input_files'][k];p=ROOT/'inputs'/k if (ROOT/'inputs').exists() else pathlib.Path(b['path'])
        with p.open('rb') as f:v=f.read(b['bytes']+1)
        if len(v)!=b['bytes']:raise RuntimeError(k+' scientific input size')
        # User explicitly waived new source/destination migration hash audits.
        CACHE[k]=v;READS.append({'key':k,'loaded_bytes':len(v),'original_version_sha256':b['sha256'],'migration_sha_reverified':False})
    return CACHE[k]
def npz(k):return np.load(io.BytesIO(read(k)),allow_pickle=False)
def load(k,weights_only=True,raw=False):return read(k) if raw else torch.load(io.BytesIO(read(k)),map_location='cpu',weights_only=weights_only)
def inputs():
    with npz('G4_TRAIN_cache') as z:train=tuple(z[k].copy() for k in ['clean_expectation_V','condition','Qsource_Vns','internal_training','event_ids'])
    x,c,q,gt,gids=train;assert gt.sum()==6593 and len(gids)==8192 and len(set(gids))==8192
    with npz('generation_inputs') as z:devparents=z['sim_event_ids'].copy()
    with npz('generated') as z:devgen=z['waveforms_V'].copy();assert np.array_equal(devparents,z['sim_event_ids'])
    with npz('SIM_identity') as z:rids=z['real_event_ids'].copy();assert np.array_equal(devparents,z['sim_event_ids'])
    real=np.frombuffer(read('REAL_V'),dtype='<f4',count=5200*480).reshape(5200,480).copy();qc=np.frombuffer(read('REAL_QC'),dtype='<u2',count=5200).copy();groups=np.frombuffer(read('REAL_groups'),dtype='<u2',count=5200).copy();orig=np.frombuffer(read('REAL_original_index'),dtype='<u4',count=5200).copy();roles=np.frombuffer(read('REAL_roles'),dtype='u1',count=5200)
    assert np.all(roles[:2000]==0) and np.all(roles[2000:]==1) and np.array_equal(groups,orig//50)
    validgroups=json.loads(read('TRAIN_roles'))['REAL_internal_validation_artificial_groups'];fit=(qc[2000:]==0)&~np.isin(groups[2000:],validgroups);val=(qc[2000:]==0)&np.isin(groups[2000:],validgroups);assert fit.sum()==2592 and val.sum()==599
    assert not set(groups[2000:][fit])&set(groups[2000:][val]);assert not set(groups[:2000])&set(groups[2000:]);assert not set(gids)&set(devparents)
    ref=json.loads(read('CAL_reference'));edges=np.array(json.loads(read('width_definition'))['edges_logQ']);selection=json.loads(read('R7_selections'));old=json.loads(read('R7_metrics'))
    idsfit=np.array(['RAW_ORIGINAL_INDEX|'+str(v) for v in orig[2000:][fit]]);pick=ev.order(idsfit,'R7-RR-TRAIN-2000')[:2000];assert np.array_equal(orig[2000:][fit][pick],selection['TRAIN_reference_original_indices'])
    realfit=aa.data(real[2000:][fit],idsfit,groups[2000:][fit],'R',ref);realdev=aa.data(real[:2000],rids,groups[:2000],'R',ref,qc[:2000]);gend=aa.data(devgen,devparents,np.arange(2000),'G',ref);rr=aa.data(real[2000:][fit][pick],idsfit[pick],groups[2000:][fit][pick],'R',ref)
    with npz('R5_TRAIN_inputs') as z:synthids=z['parent_ids'].copy()
    with npz('R5_TRAIN_waveforms') as z:assert np.array_equal(synthids,z['parent_ids']);synth=z['R3B_V'].copy()
    assert set(synthids)<=set(gids);allowed=np.isin(synthids,gids[gt]);assert len(set(synthids))==2000
    genfit=aa.data(synth[allowed],synthids[allowed],np.arange(allowed.sum()),'G',ref)
    role={'REAL_DEV':2000,'REAL_TRAIN_source':3200,'REAL_TRAIN_QC':3191,'REAL_TRAIN_fit':2592,'REAL_TRAIN_val':599,'G4_TRAIN_all':8192,'G4_TRAIN_fit':6593,'G4_TRAIN_val':1599,'R5_synthetic_before_fit_filter':2000,'R5_synthetic_after_fit_filter':int(allowed.sum()),'R5_source_parent_unique':2000,'R5_source_subset_of_allowed_G4_TRAIN':True,'R5_run_invoked':False,'REAL_groups_never_independent_acquisitions':True,'EVAL_read':0,'CAL_waveforms_decoded':0,'historical_SIM_noise_prior_REAL_events':79793}
    write('DATA_ROLES.json',role);return train,real[2000:][fit],real[2000:][val],genfit,realfit,gend,realdev,rr,ref,edges,selection,old

def infer(model,x,c,q,noise,meta=False):
    waves=[];metadata={}
    with torch.no_grad():
        for i in range(0,len(x),cfg['generation_chunk']):
            s=slice(i,i+cfg['generation_chunk']);args=[torch.as_tensor(v[s],device=cu.DEVICE) for v in [x,c,q,*noise]]
            if meta:
                v,m=model(*args,with_meta=True)
                for k,t in m.items():metadata.setdefault(k,[]).append(t.detach().cpu().numpy())
            else:v=model(*args)
            waves.append(v.detach().cpu().numpy());cu.guard()
    return np.concatenate(waves),{k:np.concatenate(v) for k,v in metadata.items()}

def generate(model,source):
    x,c,q,donor=source;g=torch.Generator().manual_seed(cfg['generation_seeds'][0]);before=g.get_state().clone();noise=tr.draw_noise(2000,g);xx=np.repeat(x[None],2000,axis=0);cc=np.repeat(c[None],2000,axis=0);qq=np.repeat(q,2000);ids=np.array([f'R10-INDEPENDENT-DRAW|{i:06d}' for i in range(2000)])
    np.savez_compressed(ROOT/'GENERATION_INPUTS.npz',clean_V=xx,c4=cc,Qsource_Vns=qq,draw_ids=ids,donor_id=np.array(donor),**{k:v.numpy() for k,v in zip(['z_pulse','z_base','white_pulse','white_base'],noise)})
    torch.save({'seed':cfg['generation_seeds'][0],'before_draw':before,'after_draw':g.get_state()},ROOT/'GENERATION_RANDOM_STATES.pt');v,m=infer(model,xx,cc,qq,noise,True);COUNTER['main_generation_rows']+=2000
    np.savez_compressed(ROOT/'generated.npz',waveforms_V=v,draw_ids=ids,donor_id=np.array(donor),**{'main_'+k:t for k,t in m.items()})
    rec={'independent_random_draws':2000,'independent_G4_donor_parents':1,'padding_rows':int(m['padding'].sum()),'clip_rows':int((m['ADC_clip_count']>0).sum()),'clip_points':int(m['ADC_clip_count'].sum()),'invalid_full_peak_rows':int(m['invalid_full_peak'].sum()),'nonfinite_rows':int((~np.isfinite(v).all(axis=1)).sum()),'draw_id_distinct_from_donor':True,'CFD_ADC_same_existing_model':True}
    write('GENERATION_RECEIPT.json',rec);return v,ids,m,rec

def nearest(d,realdev,realfit):
    maskg=d['valid'];maskr=realdev['valid'];n=min(maskg.sum(),maskr.sum(),2000);ig=np.flatnonzero(maskg)[ev.order(d['ids'][maskg],'R10-NN-query')[:n]];ir=np.flatnonzero(maskr)[ev.order(realdev['ids'][maskr],'R10-NN-query')[:n]]
    out={'query_N_each':int(n),'TRAIN_reference_N':int(realfit['valid'].sum()),'candidate_QC':int(maskg.sum()),'REAL_DEV_QC':int(maskr.sum()),'no_reference_reduction':True,'not_proof_of_no_memory':True};private={};examples={};refidx=np.flatnonzero(realfit['valid'])
    for view,key in [('V','V'),('own_shape','shape')]:
        reference=np.asarray(realfit[key][refidx],dtype='f8');b2=(reference*reference).sum(axis=1)
        for name,domain,ix in [('FIXED',d,ig),('REAL_DEV',realdev,ir)]:
            query=np.asarray(domain[key][ix],dtype='f8');ds=[];ns=[]
            for j in range(0,len(query),64):
                a=query[j:j+64];sq=((a*a).sum(axis=1)[:,None]+b2[None,:]-2*a@reference.T)/480.;idx=np.argmin(sq,axis=1);ds.extend(np.sqrt(np.maximum(sq[np.arange(len(a)),idx],0)));ns.extend(idx)
            ds=np.array(ds);ns=np.array(ns);exact=np.array([np.array_equal(a,reference[k]) for a,k in zip(query,ns)])
            out.setdefault(view,{})[name]={'RMS_distance':ev.summary(ds),'exact_to_nearest_count':int(exact.sum()),'distance_le1e_6':int((ds<=1e-6).sum()),'distance_le1e_4':int((ds<=1e-4).sum()),'threshold_units':'V for V,dimensionless for own_shape','thresholds_descriptive_not_nonmemorization_tests':True}
            private[view+'_'+name+'_distance']=ds;private[view+'_'+name+'_query_indices']=ix;private[view+'_'+name+'_TRAIN_neighbour_index']=refidx[ns]
            chosen=ev.order(domain['ids'][ix],'R10-NN-fixed-example')[:3];examples[view+'_'+name+'_query']=query[chosen];examples[view+'_'+name+'_neighbour']=reference[ns[chosen]]
    np.savez_compressed(OUT/'NN_PRIVATE.npz',**private);np.savez_compressed(OUT/'NN_EXAMPLES.npz',**examples);return out

def run():
    log(f'R10 PID{os.getpid()} Torch{torch.__version__} NumPy{np.__version__} CUDA0 CPU4');values=inputs();train,fit,val,gfit,rfit,gd,rd,rr,ref,edges,selection,old=values
    A,private,apred=ma.analyze(gfit,rfit,gd,rd,rr,selection,cfg,COUNTER,old);write('A_RESULTS.json',A);np.savez_compressed(OUT/'A_WEIGHTS_SELECTIONS_PRIVATE.npz',**private);np.savez_compressed(OUT/'A_PREDICTIONS_PRIVATE.npz',**{k+'__'+j:v for k,p in apred.items() for j,v in p.items()});log(f'A complete HGB{COUNTER["HGB_attempts"]} overlap{A["plan"]["overlap_mass"]:.5f} ESSsupport{A["evaluation_support"]}')
    model,trec,source=ft.train(train,fit,val,cfg,ROOT,load,log,COUNTER,START);waves,draws,meta,grec=generate(model,source);d=aa.data(waves,draws,np.arange(2000),'DRAW',ref)
    domains={'REAL':rd,'FIXED_SOURCE_STOCHASTIC':d,'R3B':gd};historical=json.loads(read('old_metrics'))
    for n,key in [('FULL_WAVE_STOCHASTIC','old_FULL_waveforms'),('PULSE_STOCHASTIC_OFF','old_OFF_waveforms')]:
        with npz(key) as z:v=z['waveforms_V'].copy();assert np.array_equal(z['sim_event_ids'],gd['ids'])
        domains[n]=aa.data(v,gd['ids'],np.arange(2000),'G',ref)
    stats,dist=ev.characterize(domains,ref);local={n:dc.native_statistics(v,ref) for n,v in domains.items()};c2st={};preds={};sp={k:(np.array(v['train']),np.array(v['holdout'])) for k,v in selection['indices'].items()}
    for view in cfg['C2ST_views']:
        if not d['valid'][np.r_[sp['GEN'][0],sp['GEN'][1]]].all():c2st[view]={'supported':False,'reason':'fixed_original_R7_draw_positions_include_invalid_rows'};continue
        c2st[view],preds[view]=dc.fit_one(d,rd,sp['GEN'],sp['REAL'],view,cfg,COUNTER)
    np.savez_compressed(OUT/'B_PREDICTIONS_PRIVATE.npz',**{k+'__'+j:v for k,p in preds.items() for j,v in p.items()})
    original_fit=ew.fit_slice
    def counted_fit(*a,**kw):
        COUNTER['density_EM_attempts']+=1
        if COUNTER['density_EM_attempts']>8:raise RuntimeError('density cap')
        return original_fit(*a,**kw)
    ew.fit_slice=counted_fit
    widths={k:historical['energy_conditional_width']['tables'][k] for k in ['REAL','R3B','FULL_WAVE_STOCHASTIC','PULSE_STOCHASTIC_OFF']};use=d['valid'];widths['FIXED_SOURCE_STOCHASTIC']=ew.section(d['features']['PSD'][use],d['features']['Qtotal_mVns'][use],edges)
    errors=[]
    for comp in [0,1]:
        for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
            bins=[i for i in range(8) if all(t['slices'][i]['valid'] and len(t['slices'][i]['components'])==2 and t['slices'][i]['components'][comp][field] is not None for t in widths.values())]
            errors.append({'component':comp,'field':field,'common_bin_ids_1based':[i+1 for i in bins],'MAE':{n:float(np.mean([abs(t['slices'][i]['components'][comp][field]-widths['REAL']['slices'][i]['components'][comp][field]) for i in bins])) if bins else None for n,t in widths.items() if n!='REAL'}})
    nn=nearest(d,rd,rfit);features={}
    for n,v in domains.items():
        for k,z in {**v['features'],**v['extra']}.items():features[n+'__'+k]=z
        features[n+'__valid']=v['valid'];features[n+'__region']=v['region']
    np.savez_compressed(OUT/'FEATURES_PRIVATE.npz',**features)
    metrics={'status':'complete','A':A,'B':{'training':trec,'generation':grec,'domains':stats,'domain_vs_REAL':dist,'local':local,'C2ST':c2st,'width_tables':widths,'width_common_errors':errors,'width_edges_logQ':edges,'nearest_neighbours':nn,'historical_training_environment_and_update_counts_may_differ':True,'fixed_source_capacity_control_not_physical_mapping':True},'counters':COUNTER,'worker_PID':os.getpid(),'versions':{'Torch':torch.__version__,'NumPy':np.__version__},'GPU':cu.facts(),'elapsed_s':time.monotonic()-START,'EVAL_read':0,'same_acquisition_DEV_not_blind':True,'migration_not_bitwise_validated':True}
    write('metrics.json',metrics);write('INPUTS_LOADED.json',READS)
    from plotting import plots
    plots(A,domains,metrics,meta,ROOT);metrics['elapsed_s']=time.monotonic()-START;write('metrics.json',metrics);write('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':'complete','counters':COUNTER,'threads':4,'versions':metrics['versions'],'GPU':cu.facts(),'EVAL_read':0});log('R10 complete '+json.dumps(COUNTER))
try:run()
except BaseException as e:
    write('FAILURE_'+datetime.datetime.now().strftime('%H%M%S_%f')+'.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
