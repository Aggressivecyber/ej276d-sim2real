from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import pathlib,json,io,hashlib,time,datetime,traceback
ROOT=pathlib.Path(__file__).resolve().parent;START=time.monotonic();cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'cache/joblib')
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
import evaluation as ev
import diagnostic_core as dc
import training as tr
import assessment as aa
import energy_width as ew
from model import WholeResponse
from clean_source import clean_expectation
OUT=ROOT/'results';OUT.mkdir(exist_ok=True);FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
counter={'HGB_attempts':0,'density_EM_attempts':0,'response_updates':{},'generation_rows':0,'source_control_rows':0,'EVAL_read':0,'SSH':0,'new_G4':0}
CACHE={};INPUTS=[]

def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def write(n,x):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(ev.safe(x),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def read(k):
    if k not in CACHE:
        r=cfg['input_files'][k];p=pathlib.Path(r['path'])
        with p.open('rb') as f:b=f.read(r['bytes']+1)
        if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise RuntimeError(k+' binding drift')
        CACHE[k]=b;INPUTS.append({**r,'key':k,'scope':'whole bytes hash; selected members/prefix decode as DATA_ROLES'})
    return CACHE[k]
def npz(k):return np.load(io.BytesIO(read(k)),allow_pickle=False)

def load_inputs():
    with npz('generation_inputs') as z:source={k:z[k].copy() for k in z.files}
    ids=source['sim_event_ids'];assert len(ids)==2000
    source['clean_V'],computed_q=clean_expectation(source['edep_ee_MeV'],source['edep_hadron_MeV'])
    assert np.array_equal(computed_q,source['Qsource_Vns']),'DEV clean formula Q drift'
    with npz('SIM_identity') as z:
        rids=z['real_event_ids'].copy();assert np.array_equal(ids,z['sim_event_ids'])
        sim_V=z['raw'].astype('f8')*z['normalization_peak_V'][:,None]
        parent_peak=z['normalization_peak_V'].copy()
    with npz('GEN_W0') as z:
        assert np.array_equal(ids,z['sim_event_ids'])
        w0_V=z['after'].astype('f8')*parent_peak[:,None]
    with npz('generated') as z:
        assert np.array_equal(ids,z['sim_event_ids'])
        oldwaves=[z[k].copy() for k in ['waveforms_V','independent_view_2_V','independent_view_3_V']]
    with npz('G4_TRAIN_cache') as z:
        x=z['clean_expectation_V'].copy();c=z['condition'].copy();q=z['Qsource_Vns'].copy();gt=z['internal_training'].copy();gids=z['event_ids'].copy()
    assert x.shape==(8192,480) and c.shape==(8192,4) and not set(gids)&set(ids)
    assert np.array_equal(gt,np.array([ev.hashint('R2-G4-TRAIN-VALIDATION|'+str(s))%5!=0 for s in gids]))
    assert int(gt.sum())==6593
    real=np.frombuffer(read('REAL_V'),dtype='<f4',count=5200*480).reshape(5200,480).copy()
    qc=np.frombuffer(read('REAL_QC'),dtype='<u2',count=5200).copy();groups=np.frombuffer(read('REAL_groups'),dtype='<u2',count=5200).copy()
    original=np.frombuffer(read('REAL_original_index'),dtype='<u4',count=5200).copy();roles=np.frombuffer(read('REAL_roles'),dtype='u1',count=5200)
    assert np.all(roles[:2000]==0) and np.all(roles[2000:]==1) and np.array_equal(groups,original//50)
    validgroups=json.loads(read('TRAIN_roles'))['REAL_internal_validation_artificial_groups']
    fit=(qc[2000:]==0)&~np.isin(groups[2000:],validgroups);val=(qc[2000:]==0)&np.isin(groups[2000:],validgroups)
    assert fit.sum()==2592 and val.sum()==599 and not set(groups[2000:][fit])&set(groups[2000:][val])
    assert not set(groups[:2000])&set(groups[2000:])
    ref=json.loads(read('CAL_reference'));assert not ref['slots'][4]['valid']
    edges=np.array(json.loads(read('width_definition'))['edges_logQ'])
    selection=json.loads(read('R7_selections'));oldmetrics=json.loads(read('R7_metrics'))
    trainids=np.array(['RAW_ORIGINAL_INDEX|'+str(s) for s in original[2000:][fit]])
    pick=ev.order(trainids,'R7-RR-TRAIN-2000')[:2000];assert np.array_equal(original[2000:][fit][pick],selection['TRAIN_reference_original_indices'])
    write('results/DATA_ROLES.json',{'REAL_DEV':2000,'REAL_TRAIN_source':3200,'REAL_TRAIN_QC':3191,'REAL_fit':2592,'REAL_internal_val':599,
       'REAL_original_invalid_retained':int((qc[2000:]!=0).sum()),'G4_cache':8192,'G4_fit':int(gt.sum()),'G4_val':int((~gt).sum()),
       'G4_fit_val_DEV_identity_overlap':0,'REAL_artificial_group_fit_val_DEV_overlap':0,'REAL_decoded_prefix_rows':5200,
       'CAL_waveforms_not_decoded':True,'full_REAL_file_hash_bytes':cfg['input_files']['REAL_V']['bytes'],
       'DEV_clean_reconstructed_by_same_R2_formula':True,'G4_step_timing_not_available':True,'SIM_W0_V_units':'historical SIM parent-peak proxy; not independent calibration',
       'source_G4_conditions_never_used_as_REAL_label':True,'EVAL_read':0,'historical_REAL_noise_calibration_exposure_events':79793})
    return source,rids,oldwaves,sim_V,w0_V,(x,c,q,gt,gids),real,qc,groups,original,fit,val,ref,edges,selection,oldmetrics,pick,trainids

def generation_inputs(source):
    values=[];states={};save={'clean_V':source['clean_V'],'G4_condition':source['G4_condition'],'Qsource_Vns':source['Qsource_Vns'],'sim_event_ids':source['sim_event_ids']}
    for i,seed in enumerate(cfg['generation_seeds']):
        g=torch.Generator().manual_seed(seed);before=g.get_state().clone();noise=tr.draw_noise(2000,g);values.append(noise)
        states['view'+str(i+1)]={'seed':seed,'before':before,'after':g.get_state()}
        for key,v in zip(['z_pulse','z_base','white_pulse','white_base'],noise):save['view'+str(i+1)+'_'+key]=v.numpy()
    np.savez_compressed(ROOT/'GENERATION_INPUTS.npz',**save);torch.save(states,ROOT/'GENERATION_RANDOM_STATES.pt')
    return values

def inference(model,x,c,q,noise,metadata=False):
    allv=[];allp={}
    with torch.no_grad():
        for i in range(0,len(x),cfg['generation_chunk']):
            sl=slice(i,i+cfg['generation_chunk']);args=[torch.as_tensor(s[sl]) for s in [x,c,q,*noise]]
            if metadata:
                v,p=model(*args,with_meta=True)
                for k,t in p.items():allp.setdefault(k,[]).append(t.numpy())
            else:v=model(*args)
            allv.append(v.numpy())
    return np.concatenate(allv),{k:np.concatenate(v) for k,v in allp.items()}

def generate(name,model,source,inputs):
    waves=[];meta=[]
    for i,noise in enumerate(inputs):
        v,p=inference(model,source['clean_V'],source['G4_condition'],source['Qsource_Vns'],noise,True)
        waves.append(v);meta.append(p);counter['generation_rows']+=len(v)
    extra={k:meta[0][k] for k in meta[0]}
    np.savez_compressed(ROOT/'arms'/name/'generated.npz',waveforms_V=waves[0],independent_view_2_V=waves[1],independent_view_3_V=waves[2],sim_event_ids=source['sim_event_ids'],**{'main_'+k:v for k,v in extra.items()})
    receipt={'parents':2000,'views':3,'fixed_parent_order_preserved':True,'physical_V_not_parent_peak_scaled':True,'all_rows_preserved':True,
        'views_not_independent_events':True,'per_view':[{'padding_rows':int(p['padding'].sum()),'ADC_clip_rows':int((p['ADC_clip_count']>0).sum()),
          'ADC_clip_points':int(p['ADC_clip_count'].sum()),'invalid_full_peak_rows':int(p['invalid_full_peak'].sum()),
          'no_prepeak_halfcross_rows':int(p['no_prepeak_halfcross'].sum()),'nonfinite_window_rows':int((~np.isfinite(v).all(axis=1)).sum()),
          'nonfinite_raw_full_rows':int((~np.isfinite(p['raw_full_V']).all(axis=1)).sum())} for p,v in zip(meta,waves)],
        'pulse_process_enabled':model.pulse_enabled,'gain':ev.summary(meta[0]['gain']),
        'baseline_RMS_parameter_V':ev.summary(meta[0]['baseline_RMS_parameter_V']),
        'pulse_full_RMS_V':ev.summary(np.sqrt(np.mean(meta[0]['pulse_full_V']**2,axis=1))),
        'baseline_full_RMS_V':ev.summary(np.sqrt(np.mean(meta[0]['baseline_full_V']**2,axis=1))),
        'padding_rule':'existing gather clamped endpoint replication; rows flagged, no resampling'}
    write('arms/'+name+'/GENERATION_RECEIPT.json',receipt)
    if any(r['nonfinite_window_rows'] or r['nonfinite_raw_full_rows'] for r in receipt['per_view']):raise RuntimeError(name+' nonfinite output; saved failure, no recipe repair')
    return waves,meta,receipt

def source_controls(name,model,source,noise,train,main):
    x,c,q,gt,gids=train;ix=np.floor(np.arange(64)*1999/63).astype(int)
    perm=np.random.default_rng(cfg['source_permutation_seed']).permutation(64)
    donor=min(np.flatnonzero(gt),key=lambda i:hashlib.sha256(str(gids[i]).encode()).digest())
    save={'target_position':ix,'target_ids':source['sim_event_ids'][ix],'shuffle_donor_ids':source['sim_event_ids'][ix[perm]],
          'replacement_donor_id':np.array(str(gids[donor]))}
    ref=ev.features(main[ix]);out={}
    for label,xx,cc,qq in [('shuffle',source['clean_V'][ix[perm]],source['G4_condition'][ix[perm]],source['Qsource_Vns'][ix[perm]]),
                         ('replace_first_TRAIN_fit',np.repeat(x[donor:donor+1],64,axis=0),np.repeat(c[donor:donor+1],64,axis=0),np.repeat(q[donor:donor+1],64))]:
        v,_=inference(model,xx,cc,qq,[n[ix] for n in noise]);counter['source_control_rows']+=64
        feat=ev.features(v);equal=np.all(v==main[ix],axis=1);delta=v-main[ix]
        out[label]={'n':64,'all64_bitwise_unchanged':bool(equal.all()),'unchanged_n':int(equal.sum()),
            'waveform_L2_V':ev.summary(np.linalg.norm(delta,axis=1)),'RMS_delta_V':ev.summary(np.sqrt(np.mean(delta**2,axis=1))),
            'maxabs_delta_V':ev.summary(np.max(np.abs(delta),axis=1)),
            'feature_delta':{k:ev.summary(feat['features'][k]-ref['features'][k]) for k in ['Qtotal_mVns','PSD','peak_V']}}
        save[label+'_waveforms_V']=v
    out['ignores_source_under_both_controls']=all(r['all64_bitwise_unchanged'] for r in out.values())
    out['nonzero_sensitivity_is_not_physical_correspondence']=True;out['replacement_TRAIN_fit_id']=str(gids[donor])
    np.savez_compressed(ROOT/'arms'/name/'SOURCE_CONTROLS.npz',**save)
    write('arms/'+name+'/SOURCE_CONTROL_RECEIPT.json',out)
    return out

def widths(domains,old,r8,edges,viewfeatures):
    tables={'REAL':old['PSD_density_widths']['native']['REAL'],'R3B':old['PSD_density_widths']['native']['GEN']}
    for n in ['R8_JOINT','R8_JOINT_SIGNAL']:tables[n]=r8['energy_conditional_width']['tables'][n.replace('R8_','')]
    for name in cfg['arms']:
        d=domains[name];good=d['valid'];tables[name]=ew.section(d['features']['PSD'][good],d['features']['Qtotal_mVns'][good],edges)
        counter['density_EM_attempts']+=sum(s['fit_attempts'] for s in tables[name]['slices'])
    errors=[];names=list(tables)
    for k in [0,1]:
        for field in ['mean','sigma_PSD','detrended_sigma_PSD']:
            bins=[i for i in range(8) if all(tables[n]['slices'][i]['valid'] and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field] is not None for n in names)]
            errors.append({'component':k,'field':field,'common_qualified_bin_ids_1based':[i+1 for i in bins],
                 'mean_abs_error':{n:float(np.mean([abs(tables[n]['slices'][i]['components'][k][field]-tables['REAL']['slices'][i]['components'][k][field]) for i in bins])) if bins else None for n in names if n!='REAL'}})
    assert counter['density_EM_attempts']<=16
    return {'edges_logQ':edges,'edges_mVns':np.exp(edges),'tables':tables,'common_support_paired_errors':errors,
        'source_parent_three_view_width':{n:ew.parent_width(f,edges) for n,f in viewfeatures.items()},'old_fits_reused':True,'not_particle_truth':True}

def run():
    log(f'R8C PID={os.getpid()} Torch={torch.__version__} NumPy={np.__version__} CPU4')
    source,rids,oldwaves,sim,w0,train,real,qc,groups,original,fit,val,ref,edges,selection,oldmetrics,pick,trainids=load_inputs()
    past=json.loads((ROOT/'RUN_HISTORY.json').read_text()) if (ROOT/'RUN_HISTORY.json').exists() else []
    previous=sum(r['wall_s'] for r in past)
    models,training,paired=tr.paired_train(train,real[2000:][fit],real[2000:][val],edges,cfg,ROOT,log,previous)
    counter['response_updates']={n:{'G':r['optimizer_updates_G'],'D':r['optimizer_updates_D']} for n,r in training.items()}
    inputs=generation_inputs(source);waves={'R3B':oldwaves};metadata={};generation={};controls={}
    for n,model in models.items():
        waves[n],metadata[n],generation[n]=generate(n,model,source,inputs)
        controls[n]=source_controls(n,model,source,inputs[0],train,waves[n][0])
    assert counter['generation_rows']==12000 and counter['source_control_rows']==256
    # Old outputs are descriptive references, never targets or candidates for selection.
    r8=json.loads(read('R8_metrics'))
    for n in ['JOINT','JOINT_SIGNAL']:
        with npz('R8_'+n) as z:
            assert np.array_equal(z['sim_event_ids'],source['sim_event_ids'])
            waves['R8_'+n]=[z[k].copy() for k in ['waveforms_V','independent_view_2_V','independent_view_3_V']]
    domains={'REAL':aa.data(real[:2000],rids,groups[:2000],'R',ref,qc[:2000])}
    for n,v in {**{n:v[0] for n,v in waves.items()},'SIM':sim,'GEN_W0':w0}.items():domains[n]=aa.data(v,source['sim_event_ids'],np.arange(2000),'G',ref)
    domains['REAL']['CAL_edges']=ref['CAL_charge_quantile_edges_mV_ns']
    rr=aa.data(real[2000:][fit][pick],trainids[pick],groups[2000:][fit][pick],'R',ref)
    stats,dist=ev.characterize(domains,ref);extra_stats={n:dc.native_statistics(d,ref) for n,d in domains.items()}
    splits={n:(np.array(x['train']),np.array(x['holdout'])) for n,x in selection['indices'].items()}
    pred={};c2st={};delta={}
    with npz('R7_predictions') as z:oldpred={k:z[k].copy() for k in z.files if k.startswith('native|')}
    for n in cfg['arms']:
        fixedok=bool(domains[n]['valid'][np.r_[splits['GEN'][0],splits['GEN'][1]]].all())
        for view in cfg['C2ST_views']:
            if not fixedok:
                c2st[n+'|'+view]={'supported':False,'reason':'fixed_R7_subset_includes_invalid_candidate_rows','resampling':False};continue
            result,p=dc.fit_one(domains[n],domains['REAL'],splits['GEN'],splits['REAL'],view,cfg,counter)
            result['supported']=True;c2st[n+'|'+view]=result;pred[n+'|'+view]=p
            log(f'C2ST {n} {view}: BA={result["balanced_accuracy"]:.4f}')
    for view in cfg['C2ST_views']:
        key='native|'+view+'|GEN_vs_REAL';rp={k:oldpred[key+'__'+k] for k in ['probability','label','cluster']}
        c2st['R3B|'+view]=oldmetrics['C2ST'][key]
        c2st['REAL_TRAIN_vs_DEV|'+view]=oldmetrics['C2ST']['native|'+view+'|REAL_TRAIN_vs_DEV']
        c2st['positive_vs_REAL|'+view]=oldmetrics['C2ST']['native|'+view+'|positive_vs_REAL']
        for n in ['JOINT','JOINT_SIGNAL']:c2st['R8_'+n+'|'+view]=r8['C2ST'][n+'|'+view]
        a,b=cfg['arms']
        if a+'|'+view in pred and b+'|'+view in pred:delta[a+'_minus_'+b+'|'+view]=aa.paired_bootstrap(pred[a+'|'+view],pred[b+'|'+view],cfg)
        for n in cfg['arms']:
            if n+'|'+view in pred:delta[n+'_minus_R3B|'+view]=aa.paired_bootstrap(pred[n+'|'+view],rp,cfg)
    np.savez_compressed(OUT/'C2ST_HOLDOUT_PREDICTIONS.npz',**{n+'__'+k:v for n,d in pred.items() for k,v in d.items()})
    vf={n:[ev.features(x) for x in v] for n,v in waves.items()};width=widths(domains,oldmetrics,r8,edges,vf)
    same=aa.same_N(domains,rr);nominal,association,nominal_region=aa.source_nominal(domains,source['Qsource_Vns'],source['G4_condition'])
    pv=aa.parent_variation(vf)
    features={'nominal_source_region':nominal_region,'sim_event_ids':source['sim_event_ids'],'real_event_ids':rids,
        'REAL_original_index':original[:2000],'REAL_artificial_group':groups[:2000]}
    for n,d in domains.items():
        for k,v in {**d['features'],**d['extra']}.items():features[n+'__'+k]=v
        for k in ['valid','region','label']:features[n+'__'+k]=d[k]
    np.savez_compressed(OUT/'PER_EVENT_FEATURES.npz',**features)
    metrics={'status':'complete','question':'whole clean waveform conditional response, pulse stochastic head on/off',
        'training':training,'paired_training':paired,'domains':stats,'domain_vs_REAL':dist,'local_and_processing_statistics':extra_stats,
        'same_N_conditional_reference':same,'C2ST':c2st,'paired_C2ST_difference':delta,'energy_conditional_width':width,
        'source_nominal_condition':nominal,'source_association':association,'source_input_controls':controls,
        'three_view_within_parent_variation':pv,'generation':generation,'CAL_charge_edges_mVns':ref['CAL_charge_quantile_edges_mV_ns'],
        'CAL_reference_valid':[s['valid'] for s in ref['slots']],'q4_no_fallback':True,'unknown_all_preserved':True,
        'same_acquisition_historically_exposed_DEV':True,'REAL_REAL_definition':'sameN TRAIN-fit2000 vsDEV2000, same acquisition artificial groups',
        'source_spectrum_response_confounding':True,'unpaired_mapping_not_identifiable_per_event':True,'EVAL_read':0,'SSH':0,'new_G4':0,
        'CPU_threads':4,'counters':counter,'worker_PID':os.getpid(),'elapsed_before_plots_s':time.monotonic()-START}
    write('metrics.json',metrics);write('results/ENERGY_WIDTH_METRICS.json',width);write('INPUTS.json',INPUTS)
    from plotting import plots
    plots(domains,waves,metadata,metrics,source,ref,FIG)
    metrics['numeric_elapsed_s']=time.monotonic()-START;write('metrics.json',metrics)
    write('NUMERIC_RECEIPT.json',{'status':'complete','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_s':time.monotonic()-START,'Torch':torch.__version__,'NumPy':np.__version__,'threads':4,'counters':counter,
        'one_fixed_recipe_two_arms':True,'EVAL_read':0,'SSH':0})
    log('R8C complete '+json.dumps(counter))

try:run()
except BaseException as e:
    p=ROOT/('FAILURE_'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S_%f')+'.json')
    p.write_text(json.dumps(ev.safe({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
        'error':repr(e),'traceback':traceback.format_exc(),'counters':counter,'scientific_recipe_unchanged':True}),indent=2)+'\n');raise
