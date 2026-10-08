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
OUT=ROOT/'results';OUT.mkdir(exist_ok=False);FIG=OUT/'figures';FIG.mkdir()
counter={'HGB_attempts':0,'density_EM_attempts':0,'response_updates':{},'generation_rows':0,'EVAL_read':0,'SSH':0,'new_G4':0}
CACHE={};INPUTS=[]
def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def write(n,x):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(ev.safe(x),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def read(k):
    if k not in CACHE:
        r=cfg['input_files'][k];p=pathlib.Path(r['path'])
        with p.open('rb') as f:b=f.read(r['bytes']+1)
        if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise RuntimeError(k+' binding drift')
        CACHE[k]=b;INPUTS.append({**r,'key':k,'array_members_decoded':'see DATA_ROLES, not all hashed container members'})
    return CACHE[k]
def npz(k):return np.load(io.BytesIO(read(k)),allow_pickle=False)
def load_inputs():
    with npz('generation_inputs') as z:source={k:z[k].copy() for k in z.files}
    ids=source['sim_event_ids'];assert len(ids)==2000
    with npz('SIM_identity') as z:rids=z['real_event_ids'].copy();assert np.array_equal(ids,z['sim_event_ids'])
    with npz('generated') as z:
        assert np.array_equal(ids,z['sim_event_ids'])
        oldwaves=[z[k].copy() for k in ['waveforms_V','independent_view_2_V','independent_view_3_V']]
        oldlatent=z['latent_z'].copy()
    with npz('G4_TRAIN_cache') as z:c=z['condition'].copy();q=z['Qsource_Vns'].copy();gt=z['internal_training'].copy();gids=z['event_ids'].copy()
    assert len(c)==8192 and c.shape==(8192,4) and not set(gids)&set(ids)
    assert np.array_equal(gt,np.array([ev.hashint('R2-G4-TRAIN-VALIDATION|'+str(s))%5!=0 for s in gids]))
    real=np.frombuffer(read('REAL_V'),dtype='<f4',count=5200*480).reshape(5200,480).copy()
    qc=np.frombuffer(read('REAL_QC'),dtype='<u2',count=5200).copy();groups=np.frombuffer(read('REAL_groups'),dtype='<u2',count=5200).copy()
    original=np.frombuffer(read('REAL_original_index'),dtype='<u4',count=5200).copy();roles=np.frombuffer(read('REAL_roles'),dtype='u1',count=5200)
    assert np.all(roles[:2000]==0) and np.all(roles[2000:]==1) and np.array_equal(groups,original//50)
    prior_roles=json.loads(read('TRAIN_roles'));validgroups=prior_roles['REAL_internal_validation_artificial_groups']
    fit=(qc[2000:]==0)&~np.isin(groups[2000:],validgroups);val=(qc[2000:]==0)&np.isin(groups[2000:],validgroups)
    assert fit.sum()==2592 and val.sum()==599 and not set(groups[2000:][fit])&set(groups[2000:][val])
    ref=json.loads(read('CAL_reference'));assert not ref['slots'][4]['valid']
    definition=json.loads(read('width_definition'));edges=np.array(definition['edges_logQ'])
    selection=json.loads(read('R7_selections'));oldmetrics=json.loads(read('R7_metrics'))
    train_ids=np.array(['RAW_ORIGINAL_INDEX|'+str(s) for s in original[2000:][fit]])
    pick=ev.order(train_ids,'R7-RR-TRAIN-2000')[:2000];assert np.array_equal(original[2000:][fit][pick],selection['TRAIN_reference_original_indices'])
    roles_receipt={'REAL_DEV':2000,'REAL_TRAIN_source':3200,'REAL_TRAIN_QC':3191,'REAL_TRAIN_fit':2592,'REAL_TRAIN_validation':599,'REAL_TRAIN_invalid_preserved':int((qc[2000:]!=0).sum()),'G4_TRAIN_cached':len(c),'G4_internal_fit':int(gt.sum()),'G4_internal_validation':int((~gt).sum()),
        'role_group_intersections':{'fit_val':0,'fit_DEV':len(set(groups[2000:][fit])&set(groups[:2000])),'validation_DEV':len(set(groups[2000:][val])&set(groups[:2000]))},
        'REAL_decoded_prefix_rows':5200,'CAL_waveforms_not_decoded':True,'REAL_full_container_bytes_hashed':cfg['input_files']['REAL_V']['bytes'],'SIM_metadata_only_ids_decoded':True,'old_R3B_three_waveform_views_reused':True,'EVAL_read':0,'no_REAL_generator_input':True}
    write('results/DATA_ROLES.json',roles_receipt)
    return source,rids,oldwaves,oldlatent,(c,q,gt),real,qc,groups,original,fit,val,ref,edges,selection,oldmetrics,pick,train_ids

def generation_inputs(source,oldlatent):
    state=torch.load(io.BytesIO(read('generation_rng')),map_location='cpu',weights_only=True);g=torch.Generator();g.set_state(state['generator_before_draw'])
    z=torch.randn((2000,8),generator=g);w=torch.randn((2000,512),generator=g);assert np.array_equal(z.numpy(),oldlatent)
    sg=torch.Generator().manual_seed(cfg['signal_main_seed']);s=torch.randn((2000,512),generator=sg)
    inputs=[(z,w,s)];states={'main_R3B_original_state':state,'main_signal_before_draw':torch.Generator().manual_seed(cfg['signal_main_seed']).get_state(),'main_signal_after_draw':sg.get_state()}
    for seed in cfg['extra_view_seeds']:
        e=torch.Generator().manual_seed(seed);before=e.get_state();zz=torch.randn((2000,8),generator=e);ww=torch.randn((2000,512),generator=e);ss=torch.randn((2000,512),generator=e)
        inputs.append((zz,ww,ss));states[str(seed)]={'before_draw':before,'after_draw':e.get_state()}
    save={'G4_condition':source['G4_condition'],'Qsource_Vns':source['Qsource_Vns'],'sim_event_ids':source['sim_event_ids']}
    for i,(zz,ww,ss) in enumerate(inputs):
        for k,v in [('z',zz),('baseline_white',ww),('signal_white',ss)]:save['view'+str(i+1)+'_'+k]=v.numpy()
    np.savez_compressed(ROOT/'GENERATION_INPUTS.npz',**save);torch.save(states,ROOT/'GENERATION_RNG.pt')
    return inputs

def generate(name,model,source,inputs):
    c=torch.tensor(source['G4_condition']);q=torch.tensor(source['Qsource_Vns']);waves=[];params=[]
    with torch.no_grad():
        for view,(z,w,s) in enumerate(inputs):
            allv=[];allp={}
            for i in range(0,2000,cfg['generation_chunk']):
                v,p=model(c[i:i+256],q[i:i+256],z[i:i+256],w[i:i+256],s[i:i+256]);allv.append(v.numpy());counter['generation_rows']+=len(v)
                for k,x in p.items():allp.setdefault(k,[]).append(x.numpy())
            waves.append(np.concatenate(allv));params.append({k:np.concatenate(v) for k,v in allp.items()})
            assert np.isfinite(waves[-1]).all()
    extra={k:params[0][k] for k in params[0]};peak=params[0]['clean_peak_index'];delta=params[0]['signal_delta_V_512']
    before=np.arange(512)[None,:]<=peak[:,None];assert np.all(delta[before]==0)
    np.savez_compressed(ROOT/'arms'/name/'generated.npz',waveforms_V=waves[0],independent_view_2_V=waves[1],independent_view_3_V=waves[2],sim_event_ids=source['sim_event_ids'],**{'main_'+k:v for k,v in extra.items()})
    receipt={'same_2000_G4_parents':True,'views':3,'extra_views_not_independent_true_events':True,'padding_rows':int(params[0]['padding'].sum()),'ADC_clip_rows':int(params[0]['quantization_clip'].sum()),'signal_delta_before_and_at_clean_peak_exactly_zero':True,'signal_scale_V':ev.summary(params[0]['signal_scale_V']),'signal_delta_RMS_V':ev.summary(params[0]['signal_delta_RMS_V']),'signal_delta_mean_V':ev.summary(delta.mean(axis=1)),'event_phase_ns':ev.summary(params[0]['event_phase_ns']),'signal_kernel':model.signal_kernel().detach().numpy()}
    write('arms/'+name+'/GENERATION_RECEIPT.json',receipt)
    return waves,params,receipt

def run():
    log(f'R8 PID={os.getpid()} Torch={torch.__version__} NumPy={np.__version__} CPU4')
    source,rids,oldwaves,oldlatent,train,real,qc,groups,original,fit,val,ref,edges,selection,oldmetrics,pick,trainids=load_inputs()
    pack=torch.load(io.BytesIO(read('model')),map_location='cpu',weights_only=True)
    initial,constructor=tr.make_initial(pack,cfg,ROOT)
    target=tr.targets(real[2000:][fit],real[2000:][val],edges,cfg,ROOT)
    models={};training={};samplings={}
    for name in cfg['arms']:
        model,receipt,digests=tr.train_arm(name,initial,constructor,train,target,cfg,ROOT,log);models[name]=model;training[name]=receipt;samplings[name]=digests;counter['response_updates'][name]=receipt['optimizer_updates']
    common=min(map(len,samplings.values()));assert samplings['JOINT'][:common]==samplings['JOINT_SIGNAL'][:common]
    assert training['JOINT']['initial_parameter_sha256']==training['JOINT_SIGNAL']['initial_parameter_sha256']
    assert training['JOINT']['validation_input_sha256']==training['JOINT_SIGNAL']['validation_input_sha256']
    paired_training={'identical_initial_learnable_parameters_and_buffers':True,'identical_validation_inputs':True,'identical_source_real_z_baseline_signal_training_draws':True,'common_prefix_updates':common,'each_arm_updates':{k:len(v) for k,v in samplings.items()},'same_actual_update_count':len(samplings['JOINT'])==len(samplings['JOINT_SIGNAL']),'signal_white_drawn_even_if_disabled':True}
    write('PAIRED_TRAINING_RECEIPT.json',paired_training)
    ginputs=generation_inputs(source,oldlatent);waves={'R3B':oldwaves};metadata={};generation={}
    for name,model in models.items():waves[name],metadata[name],generation[name]=generate(name,model,source,ginputs)
    assert counter['generation_rows']==12000
    domains={'REAL':aa.data(real[:2000],rids,groups[:2000],'R',ref,qc[:2000])}
    for name in ['R3B','JOINT','JOINT_SIGNAL']:domains[name]=aa.data(waves[name][0],source['sim_event_ids'],np.arange(2000),'G',ref)
    domains['REAL']['CAL_edges']=ref['CAL_charge_quantile_edges_mV_ns']
    rr=aa.data(real[2000:][fit][pick],trainids[pick],groups[2000:][fit][pick],'R',ref)
    stats,dist=ev.characterize(domains,ref);extra_stats={n:dc.native_statistics(d,ref) for n,d in domains.items()}
    splits={n:(np.array(x['train']),np.array(x['holdout'])) for n,x in selection['indices'].items()}
    pred={};c2st={};paired={}
    with npz('R7_predictions') as z:oldpred={k:z[k].copy() for k in z.files if k.startswith('native|')}
    for name in cfg['arms']:
        d=domains[name]
        assert d['valid'][np.r_[splits['GEN'][0],splits['GEN'][1]]].all(),'new source C2ST fixed subset invalid; not reselecting'
        for view in cfg['C2ST_views']:
            result,p=dc.fit_one(d,domains['REAL'],splits['GEN'],splits['REAL'],view,cfg,counter);c2st[name+'|'+view]=result;pred[name+'|'+view]=p
            log(f'C2ST {name} {view}: BA={result["balanced_accuracy"]:.4f}')
    for view in cfg['C2ST_views']:
        key='native|'+view+'|GEN_vs_REAL';rp={k:oldpred[key+'__'+k] for k in ['probability','label','cluster']}
        c2st['R3B|'+view]=oldmetrics['C2ST'][key]
        c2st['REAL_TRAIN_vs_DEV|'+view]=oldmetrics['C2ST']['native|'+view+'|REAL_TRAIN_vs_DEV'];c2st['positive_vs_REAL|'+view]=oldmetrics['C2ST']['native|'+view+'|positive_vs_REAL']
        paired['JOINT_SIGNAL_minus_JOINT|'+view]=aa.paired_bootstrap(pred['JOINT_SIGNAL|'+view],pred['JOINT|'+view],cfg)
        for name in cfg['arms']:paired[name+'_minus_R3B|'+view]=aa.paired_bootstrap(pred[name+'|'+view],rp,cfg)
    assert counter['HGB_attempts']==12
    np.savez_compressed(OUT/'C2ST_HOLDOUT_PREDICTIONS.npz',**{n+'__'+k:v for n,d in pred.items() for k,v in d.items()})
    vf={n:[ev.features(x) for x in v] for n,v in waves.items()};width=aa.widths(domains,oldmetrics,edges,vf,counter)
    same=aa.same_N(domains,rr);nominal,associations,nominal_region=aa.source_nominal(domains,source['Qsource_Vns'],source['G4_condition']);pv=aa.parent_variation(vf)
    features_saved={'nominal_source_region':nominal_region,'sim_event_ids':source['sim_event_ids'],'real_event_ids':rids,'REAL_original_index':original[:2000],'REAL_artificial_group':groups[:2000]}
    for n,d in domains.items():
        for k,v in {**d['features'],**d['extra']}.items():features_saved[n+'__'+k]=v
        for k in ['valid','region','label']:features_saved[n+'__'+k]=d[k]
    np.savez_compressed(OUT/'PER_EVENT_FEATURES.npz',**features_saved)
    metrics={'status':'complete','question':'paired unsmoothed local joint loss with signal-envelope stochastic response on/off','training':training,'paired_training':paired_training,'domains':stats,'domain_vs_REAL':dist,'local_and_processing_statistics':extra_stats,'same_N_conditional_reference':same,'C2ST':c2st,'paired_C2ST_difference':paired,'energy_conditional_width':width,'source_nominal_condition':nominal,'source_association':associations,'three_view_within_parent_variation':pv,'generation':generation,
        'CAL_charge_edges_mVns':ref['CAL_charge_quantile_edges_mV_ns'],'CAL_reference_valid':[s['valid'] for s in ref['slots']],'q4_no_fallback':True,'unknown_all_preserved':True,'same_acquisition_historically_exposed_DEV':True,'R3B_extra_training_vs_mechanism_effect_not_identified':True,'R7_real_real_definition':'same-N TRAIN-fit2000 vsDEV2000, same acquisition artificial groups; not R4 halfEVAL','EVAL_read':0,'SSH':0,'new_G4':0,'CPU_threads':4,'counters':counter,'worker_PID':os.getpid(),'elapsed_before_plots_s':time.monotonic()-START}
    write('metrics.json',metrics);write('results/ENERGY_WIDTH_METRICS.json',width);write('INPUTS.json',INPUTS)
    from plotting import plots
    plots(domains,waves,metadata,metrics,source,ref,FIG)
    metrics['numeric_elapsed_s']=time.monotonic()-START;write('metrics.json',metrics)
    write('NUMERIC_RECEIPT.json',{'status':'complete','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'Torch':torch.__version__,'NumPy':np.__version__,'threads':4,'counters':counter,'one_response_recipe_two_fixed_arms':True,'EVAL_read':0,'SSH':0})
    log('R8 training/evaluation complete '+json.dumps(counter))

try:run()
except BaseException as e:
    write('FAILURE.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'error':repr(e),'traceback':traceback.format_exc(),'counters':counter,'no_recipe_change_on_engineering_repair':True});raise
