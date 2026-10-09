from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,hashlib,copy
import numpy as np,torch
import evaluation as ev
from model import adc
from flow_runtime import Pipeline,infer
from flow_core import bins
from diffusion_core import physical_condition
import flow_diagnostics as fd

ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);CFG=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
COUNTER={'source_main_rows':0,'REAL_control_main_rows':0,'main_generation_rows':0,'repeat_noise_rows':0,'source_probe_rows':0,
         'extra_guide_probe_rows':0,'response_fit_or_update':0,'HGB_attempts':0,'density_EM_attempts':0,'EVAL_read':0,'CAL_waveform_decode':0}
def write(n,v):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(ev.safe(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in (z.files if keys is None else keys)}
def log(v):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),v,flush=True)
def progress(v):write('GENERATION_PROGRESS.json',{'PID':os.getpid(),'counters':COUNTER,'elapsed_s':time.monotonic()-START,**v});log(v)
def run():
    actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz',keys=['reference_V','reference_ids','reference_groups'])
    meta=load(ROOT/'DEV_SOURCE_METADATA.npz');probe=load(ROOT/'PROBE_INPUTS_CORRECTED.npz')
    source=actual['source_waveforms_V'];ids=actual['source_ids'];assert np.array_equal(ids,meta['source_ids'])
    master=np.random.Generator(np.random.PCG64(CFG['inference_seed']));initial=copy.deepcopy(master.bit_generator.state)
    seed_source=master.integers(0,np.iinfo(np.uint64).max,size=2000,dtype=np.uint64)
    seed_real=master.integers(0,np.iinfo(np.uint64).max,size=2000,dtype=np.uint64)
    seed_joint=master.integers(0,np.iinfo(np.uint64).max,size=1024,dtype=np.uint64)
    seed_probe=seed_source[probe['noise_origin_parent_index']]
    np.savez_compressed(ROOT/'ACTUAL_DIFFUSION_SEED_MAPS_PRIVATE.npz',SOURCE_MAIN=seed_source,REAL_GUIDE=seed_real,JOINT8=seed_joint,SOURCE_PROBES=seed_probe)
    write('INFERENCE_RANDOM_PROTOCOL.json',{'master_PCG64_seed':CFG['inference_seed'],'initial_state':initial,'final_state':master.bit_generator.state,
        'allocation_order':['SOURCE_MAIN2000','REAL_GUIDE2000','JOINT8_1024'],'probe_reuses_MAIN_noise_origin_seed':True,
        'NumPy':np.__version__,'record_normal_draw':'row-major normal f64(50,480) cast f32, initial then t50..2; t1 none',
        'record_seed_independent_of_chunk':True,'master_stream_uses_uint64_high_exclusive':int(np.iinfo(np.uint64).max)})
    name='CONDITIONAL_DIFFUSION_EDIT';pipeline=Pipeline(CFG,ROOT,name);pipeline.set_seeds(seed_source,'SOURCE_MAIN')
    progress({'phase':'source main2000 start'})
    pre,signal,electronic=infer(pipeline,source,actual['z'],actual['white'],actual['style'],CFG['generation_chunk']);detail={k:v.copy()for k,v in pipeline.details.items()}
    post,ad=adc(pre);COUNTER['source_main_rows']=2000;COUNTER['main_generation_rows']+=2000
    np.savez_compressed(ROOT/name/'generated.npz',pre_ADC_V=pre,waveforms_V=post,source_ids=ids,**detail)
    old_base=load(pathlib.Path(CFG['R32_root'])/'CONTROL_FINAL/generated.npz',keys=['pre_ADC_V','source_ids'])
    same=bool(np.array_equal(detail['base_pre_ADC_V'],old_base['pre_ADC_V']));assert same and np.array_equal(ids,old_base['source_ids'])
    generations={name:{'rows':2000,'all_original_source_ids_preserved':True,'ADC':ad,'edit_applied_N':int(detail['flow_applied'].sum()),
        'unsupported_source_Q_base_fallback_N':int((~detail['flow_applied']).sum()),'base_preADC_matches_R32_bitwise':same,
        'namespace':'SOURCE_MAIN','pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),
        'weight_file':CFG['endpoint_weight_paths'][name],'base_G_weight_file':CFG['base_G'],'t_start':50,'reverse_steps':50,
        'source_clean_full480_input':True,'physical_final_signal_electronic_split':'unknown'}}
    write('GENERATION_RECEIPT.json',{'domains':generations,'counter':COUNTER});progress({'phase':'source main2000 saved','elapsed_s':time.monotonic()-START})
    rname='REAL_GUIDE_EDIT';(ROOT/rname).mkdir(exist_ok=False);guide=real['reference_V'];b=bins(guide,pipeline.edges)
    pipeline.set_seeds(seed_real,'REAL_GUIDE');rpre=pipeline.edit(guide,b,chunk=CFG['generation_chunk']);rdetail={k:v.copy()for k,v in pipeline.details.items()}
    rpost,rad=adc(rpre);rpost[~rdetail['flow_applied']]=guide[~rdetail['flow_applied']]
    rdetail.update(base_signal_V=guide.copy(),base_electronic_V=np.zeros_like(guide),base_sum_rounding_f64_V=np.zeros(guide.shape,'f8'))
    COUNTER['REAL_control_main_rows']=2000;COUNTER['main_generation_rows']+=2000
    np.savez_compressed(ROOT/rname/'generated.npz',pre_ADC_V=rpre,waveforms_V=rpost,reference_ids=real['reference_ids'],reference_groups=real['reference_groups'],**rdetail)
    generations[rname]={'rows':2000,'seen_TRAIN_REFERENCE_control':True,'reference_order_groups_preserved':True,
        'edit_applied_N':int(rdetail['flow_applied'].sum()),'unsupported_reference_Q_original_record_fallback_N':int((~rdetail['flow_applied']).sum()),
        'unsupported_postADC_exact_original_reference':bool(np.array_equal(rpost[~rdetail['flow_applied']],guide[~rdetail['flow_applied']])),
        'ADC':{**rad,'actual_inside_support_ADC_rows':int(rdetail['flow_applied'].sum()),'outside_support_original_ADC_record_unchanged':True},
        'namespace':'REAL_GUIDE','pre_ADC_sha256':hashlib.sha256(rpre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(rpost.tobytes()).hexdigest(),
        't_start':50,'reverse_steps':50,'no_new_REAL_library':True,'physical_split_unknown':True}
    write('GENERATION_RECEIPT.json',{'domains':generations,'counter':COUNTER});progress({'phase':'REAL_GUIDE2000 saved'})
    pipeline.set_seeds(seed_probe,'SOURCE_PROBES_noise_origin_MAIN')
    pp=fd.probes(pipeline,post,actual['style'],name,ROOT,OUT,CFG,pipeline.edges,COUNTER)
    write('SOURCE_PROBES_SUMMARY.json',{name:pp});progress({'phase':'source832 probes saved'})
    pipeline.set_seeds(seed_joint,'JOINT8')
    jj=fd.joint(pipeline,source,meta['composition'],name,ROOT,OUT,CFG,pipeline.edges,json.loads((ROOT/'VALLEY_DEFINITION.json').read_text('utf8')),COUNTER)
    write('JOINT8_SUMMARY.json',{name:jj});progress({'phase':'joint1024 saved'})
    pick=np.flatnonzero(detail['flow_applied'])[:128];assert len(pick)==128
    j=np.arange(250,dtype='f8');pert=np.sin(2*np.pi*j/249.);pert-=pert.mean();pert*=.001/np.sqrt(np.mean(pert**2))
    delta=np.zeros((128,480),'f8');delta[:,150:400]=pert
    g0=detail['base_pre_ADC_V'][pick];g1=(g0.astype('f8')+delta).astype('f4');fixed_c=detail['condition_raw_logQ_PSD'][pick]
    perturbed=pipeline.edit(g1,detail['source_Qbin_actual_preADC'][pick],seeds=seed_source[pick],fixed_condition_raw=fixed_c,chunk=CFG['generation_chunk'])
    pd={k:v.copy()for k,v in pipeline.details.items()};dif=perturbed.astype('f8')-pre[pick].astype('f8')
    q0=physical_condition(g0);q1=physical_condition(g1);qf=physical_condition(g0.astype('f8')+delta)
    rs=np.sqrt(np.mean(dif*dif,1));identical=np.all(perturbed==pre[pick],axis=1)
    np.savez_compressed(OUT/'GUIDE_PERTURBATION_PRIVATE.npz',parent_index=pick,source_ids=ids[pick],original_guide_V=g0,
        actual_perturbed_guide_f32_V=g1,designed_perturbation_f64_V=delta,original_output_preADC_V=pre[pick],perturbed_output_preADC_V=perturbed,
        fixed_condition_raw=fixed_c,original_guide_physical_condition=q0,actual_perturbed_guide_physical_condition=q1,
        intended_f64_guide_physical_condition=qf,original_noise_sha256=detail['actual_diffusion_noise_sha256'][pick],**pd)
    gp={'rows':128,'selection':'first128 supported original source order','fixed_c_bitwise':bool(np.array_equal(pd['condition_raw_logQ_PSD'],fixed_c)),
        'same_actual_noise_hashes':bool(np.array_equal(pd['actual_diffusion_noise_sha256'],detail['actual_diffusion_noise_sha256'][pick])),
        'designed_segment_mean_V':float(pert.mean()),'designed_segment_RMS_V':float(np.sqrt(np.mean(pert**2))),
        'actual_f32_Qtotal_mVns_change':ev.summary(np.exp(q1[:,0])-np.exp(q0[:,0])),
        'actual_f32_PSD_change':ev.summary(q1[:,1]-q0[:,1]),'intended_f64_Qtotal_mVns_change':ev.summary(np.exp(qf[:,0])-np.exp(q0[:,0])),
        'intended_f64_PSD_change':ev.summary(qf[:,1]-q0[:,1]),'preADC_output_RMS_V':ev.summary(rs),
        'output_nonidentical_fraction':float(1.-identical.mean()),'ADC_LSB_V':1/16383,
        'pass':bool((1.-identical.mean())>=.95 and np.median(rs)>1/16383),'effect_not_physical_truth':True}
    assert gp['fixed_c_bitwise']and gp['same_actual_noise_hashes'];COUNTER['extra_guide_probe_rows']=128
    write('GUIDE_PERTURBATION_SUMMARY.json',gp)
    write('GENERATION_RECEIPT.json',{'domains':generations,'counter':COUNTER,'elapsed_s':time.monotonic()-START,'NumPy':np.__version__,'torch':torch.__version__,
        'exact_main_and_auxiliary_preADC_saved':True,'flow_named_fields_are_compatibility_aliases_for_diffusion_edit':True})
    assert COUNTER['main_generation_rows']==4000 and COUNTER['repeat_noise_rows']==1024 and COUNTER['source_probe_rows']==832 and COUNTER['extra_guide_probe_rows']==128
    write('GENERATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete'})
    progress({'phase':'generation complete'})
try:run()
except BaseException as e:
    write('GENERATION_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
