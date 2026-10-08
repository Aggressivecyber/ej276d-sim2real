from __future__ import annotations
import os,pathlib,sys,json,time,datetime,hashlib,io,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent.parent;OUT=ROOT/'results';FIG=OUT/'figures';OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl-cache');os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'joblib-cache');START=time.monotonic()
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
CFG=json.loads((ROOT/'config.json').read_text());torch.manual_seed(CFG['seed']);np.random.seed(CFG['seed'])
GLOBAL_TORCH_INITIAL=torch.get_rng_state().clone();GLOBAL_NUMPY_INITIAL=np.random.get_state()
import evaluation as ev
from training import train_arms
from source_condition import clean_expectation,condition
import gap_evaluation as ge
ev.OUT=OUT;ev.FIG=FIG;ev.CFG=CFG;INPUTS=[]
def convert(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().numpy()
    if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [convert(v) for v in x]
    return x
def write(p,x):
    p=pathlib.Path(p);p=p if p.is_absolute() else OUT/p
    p.write_text(json.dumps(ev.safe(convert(x)),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(p,expected=None):
    b=pathlib.Path(p).read_bytes();h=hashlib.sha256(b).hexdigest()
    if expected and h!=expected:raise RuntimeError('input drift '+str(p))
    INPUTS.append({'path':str(p),'bytes':len(b),'sha256':h});return np.load(io.BytesIO(b),allow_pickle=False)
def segment(p,dtype,first,n):
    dt=np.dtype(dtype)
    with pathlib.Path(p).open('rb') as f:f.seek(first*dt.itemsize);b=f.read(n*dt.itemsize)
    if len(b)!=n*dt.itemsize:raise RuntimeError('input segment incomplete')
    INPUTS.append({'path':str(p),'offset_bytes':first*dt.itemsize,'bytes':len(b),'segment_sha256':hashlib.sha256(b).hexdigest()});return np.frombuffer(b,dtype=dt).copy()
def log(s):ev.log(s)
def main():
    log(f'R6 PID{os.getpid()} numpy{np.__version__} torch{torch.__version__};two paired fixed curricula,CPU4 local')
    ready=json.loads((ROOT/'READINESS.json').read_text())
    for s in ready['execution_sources']:
        if sha(s['path'])!=s['sha256']:raise RuntimeError('source drift')
    torch.save({'master_seed':CFG['seed'],'torch_before_any_constructor':GLOBAL_TORCH_INITIAL,'NumPy_before_any_constructor':GLOBAL_NUMPY_INITIAL},ROOT/'INITIAL_GLOBAL_RANDOM_STATES.pt')
    k=BASE/'work/P1K_REAL_REFERENCE_AND_REPEATABILITY_001';data=k/'A2_ENGINEERING_001/run_001'
    real=segment(data/'window_V_f32.bin','<f4',2000*480,3200*480).reshape(3200,480);qc=segment(data/'QC_reason_u16.bin','<u2',2000,3200)
    groups=segment(k/'selection_group.bin','<u2',2000,3200);orig=segment(k/'selection_original_index.bin','<u4',2000,3200);roles=segment(k/'selection_role.bin','u1',2000,3200)
    assert np.all(roles==1) and np.array_equal(groups,orig//50)
    rd=ev.features(real);realgood=rd['valid']&(qc==0);vg=json.loads((BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/results/DATA_ROLES.json').read_text())['REAL_internal_validation_artificial_groups']
    rt=realgood&~np.isin(groups,vg);rv=realgood&np.isin(groups,vg);assert rt.sum()==2592 and rv.sum()==599 and not set(groups[rt])&set(groups[rv])
    cache=BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/data/G4_TRAIN_CACHE.npz'
    with load(cache,'117152db28a16fcfc9a83012cc863e32f2d73bdef77bbf7b5f8e977c3e64818b') as z:
        clean=z['clean_expectation_V'];c=z['condition'];q=z['Qsource_Vns'];gids=z['event_ids'];gt=z['internal_training'];gevents=z['source_event_id'];ee=z['edep_ee_MeV'];had=z['edep_hadron_MeV']
    gd=ev.features(clean);g_nominal=gd['features']['Qtotal_mVns'];assert gd['valid'].all() and gt.sum()==6593 and (~gt).sum()==1599
    lo,hi=CFG['gap_charge_mVns'];rgap=(rd['features']['Qtotal_mVns']>=lo)&(rd['features']['Qtotal_mVns']<hi);ggap=(g_nominal>=lo)&(g_nominal<hi)
    excluded={'FULL':{'REAL_fit_n':int(rt.sum()),'REAL_val_n':int(rv.sum()),'G4_fit_n':int(gt.sum()),'G4_val_n':int((~gt).sum())},
              'GAP':{'REAL_fit_n':int((rt&~rgap).sum()),'REAL_val_n':int((rv&~rgap).sum()),'G4_fit_n':int((gt&~ggap).sum()),'G4_val_n':int(((~gt)&~ggap).sum()),
              'removed_REAL_fit_original_index':orig[rt&rgap],'removed_REAL_validation_original_index':orig[rv&rgap],'allowed_REAL_fit_original_index':orig[rt&~rgap],'allowed_REAL_validation_original_index':orig[rv&~rgap],
              'removed_G4_fit_ids':gids[gt&ggap],'removed_G4_validation_ids':gids[(~gt)&ggap],'allowed_G4_fit_ids':gids[gt&~ggap],'allowed_G4_validation_ids':gids[(~gt)&~ggap],
              'removed_REAL_fit':int((rt&rgap).sum()),'removed_REAL_validation':int((rv&rgap).sum()),'removed_G4_fit':int((gt&ggap).sum()),'removed_G4_validation':int(((~gt)&ggap).sum()),
              'gap_fit_OR_validation_intersection_with_removed':0,'for_noise_scale_or_CDF_target_removed_events':0,'group_fit_validation_intersection':0}}
    assert not np.any(rgap&(rt&~rgap|rv&~rgap)) and not np.any(ggap&(gt&~ggap|~gt&~ggap))
    excluded['source_coverage']={arm:{'condition_fit_quantiles5_50_95':np.quantile(c[m],[.05,.5,.95],axis=0),'nominal_charge_fit_range': [float(g_nominal[m].min()),float(g_nominal[m].max())],
                                      'fit_below_gap':int((m&(g_nominal<lo)).sum()),'fit_in_gap':int((m&ggap).sum()),'fit_above_gap':int((m&(g_nominal>=hi)).sum()),'EE_only_fit':int((m&(c[:,1]<.05)).sum()),'hadron_only_fit':int((m&(c[:,1]>.95)).sum())}
                                   for arm,m in [('FULL',gt),('GAP',gt&~ggap)]}
    write('DATA_ROLES_AND_GAP_EXCLUSIONS.json',excluded)
    models,training,allowed,_,_=train_arms(real,rd['features']['Qtotal_mVns'],realgood,rt,rv,c,q,g_nominal,gt,CFG,ROOT,write,log)
    # Frozen same DEV2000 source inputs; no trained historical weight is loaded into the two new arms.
    p=pathlib.Path('E:/GAN/baseline_mean_fix_20261004_1125/inputs/evaluation_inputs.npz')
    with load(p,'29642e32877b6df3b5c1cdb10e33031474d8a95606a1cde0fc011dae61000c83') as z:
        meta={n:z[n] for n in ['sim_event_ids','edep_ee_MeV','edep_hadron_MeV','physical_r_h']}
    ids=meta['sim_event_ids'];assert not set(gids)&set(ids)
    dc,dq=clean_expectation(meta['edep_ee_MeV'],meta['edep_hadron_MeV']);cond,energy,h=condition(dc,meta['edep_ee_MeV'],meta['edep_hadron_MeV'])
    nominal=ev.features(dc)['features']['Qtotal_mVns'];ref=json.loads((data/'STATISTICS.json').read_text());nominal_region=np.searchsorted(ref['CAL_charge_quantile_edges_mV_ns'],nominal,side='right').astype('u1')
    gen=torch.Generator().manual_seed(CFG['generation_seed']);before=gen.get_state().clone();z=torch.randn((2000,8),generator=gen);white=torch.randn((2000,512),generator=gen)
    torch.save({'seed':CFG['generation_seed'],'before_draw':before,'after_draw':gen.get_state(),'global_after_training':torch.get_rng_state()},ROOT/'GENERATION_RANDOM_STATES.pt')
    np.savez_compressed(ROOT/'generation_inputs.npz',G4_condition=cond,Qsource_Vns=dq,nominal_clean_Qtotal_mVns=nominal,nominal_region=nominal_region,latent_z=z.numpy(),white=white.numpy(),sim_event_ids=ids)
    waves={};generation={}
    for name,model in models.items():
        ww=[];pm={}
        with torch.no_grad():
            for i in range(0,2000,256):
                v,m=model(torch.tensor(cond[i:i+256]),torch.tensor(dq[i:i+256]),z[i:i+256],white[i:i+256]);ww.append(v.numpy())
                for key,a in m.items():pm.setdefault(key,[]).append(a.numpy())
        v=np.concatenate(ww);pm={key:np.concatenate(a) for key,a in pm.items()};waves[name]=v
        np.savez_compressed(ROOT/(name+'_generated.npz'),waveforms_V=v,sim_event_ids=ids,**pm)
        generation[name]={'parent_N':2000,'padding':int(pm['padding'].sum()),'quantization_clip':int(pm['quantization_clip'].sum()),'seed':CFG['generation_seed'],'model_SHA256':sha(ROOT/name/'R3B/model.pt'),'waveform_npz_SHA256':sha(ROOT/(name+'_generated.npz'))}
    write('MODELS_AND_GENERATION_FROZEN_BEFORE_EVAL.json',{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),'task_SHA256':CFG['task_SHA256'],'config_SHA256':sha(ROOT/'config.json'),'both_models_and_main_generation_saved':True,'training':training,'generation':generation,'EVAL_values_opened_in_this_process':0,'no_EVAL_model_or_target_selection':True})
    ep=BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001/EVAL_EXTRACT.npz'
    with load(ep,'dc5dfa4e4f0ab3fb419d91a05edf1831e5fc918465f1aa0945c5ac8a23ea91d4') as z0:
        real_eval=ev.features(z0['window_V_f32']);real_eval['valid']&=z0['QC_reason_u16']==0;eo=z0['original_index'];assert real_eval['valid'].sum()==7583
    eid=json.loads((BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001/EVAL_IDENTITIES.json').read_text());assert np.array_equal(eo,eid['original_index'])
    domains={}
    frozen={pathlib.Path(s['path']):s for s in ready['frozen_bindings']}
    for name,stage in [('R2','R2_SMOOTH_STOCHASTIC_RESPONSE_001'),('R3','R3_FLEXIBLE_TEMPORAL_RESPONSE_001'),('R3B','R3B_ENERGY_CONDITIONAL_WIDTH_001')]:
        p=BASE/'work'/stage/'generated.npz';s=frozen[p]
        with load(p,s['sha256']) as z0:assert np.array_equal(z0['sim_event_ids'],ids);domains[name]=ev.features(z0['waveforms_V'])
    for name in ['FULL','GAP']:domains[name]=ev.features(waves[name])
    for name,d in domains.items():ge.assigned(d,ids,np.arange(2000),'SIM',ref)
    domains['REAL']=ge.assigned(real_eval,eid['source_id'],eo//50,'REAL',ref)
    descriptor=json.loads((BASE/'work/R3B_ENERGY_CONDITIONAL_WIDTH_001/results/DIAGNOSTIC_DEFINITION.json').read_text());edges=np.array(descriptor['edges_logQ'])
    evaluation=ge.evaluate(domains,ref,CFG,nominal_region,edges,write)
    metric={'status':'complete','training':training,'data_roles':excluded,'generation':generation,'evaluation':evaluation,'CAL_reference_valid':[s['valid'] for s in ref['slots']],
            'q4_CAL_failure_preserved':True,'retrospective_single_acquisition':True,'architectures_have_full_DEV_historical_exposure':True,'current_GAP_no_q3_fit_or_validation_or_statistics':True,
            'not_incident_energy_or_cross_condition_generalization':True,'no_paired_source_REAL_energy_calibration':True,'source_nominal_charge_is_saved480window100_400':True,'worker_PID':os.getpid(),'master_seed':CFG['seed'],'SSH':0,'new_G4':0,'numeric_elapsed_s':time.monotonic()-START}
    write('metrics.json',metric);write('INPUTS.json',INPUTS);(ROOT/'metrics.json').write_text((OUT/'metrics.json').read_text(),encoding='utf-8');log('R6 completed '+str(metric['numeric_elapsed_s'])+'s')
if __name__=='__main__':
    try:main()
    except BaseException as e:write('FAILURE.json',{'PID':os.getpid(),'numeric_elapsed_s':time.monotonic()-START,'error':repr(e),'traceback':traceback.format_exc()});raise
