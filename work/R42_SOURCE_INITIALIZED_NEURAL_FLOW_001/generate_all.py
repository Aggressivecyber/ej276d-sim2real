from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,hashlib
import numpy as np,torch
import evaluation as ev
from model import adc
from flow_runtime import Pipeline,infer
import flow_diagnostics as fd
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CFG=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic();ARM='SOURCE_NEURAL_FLOW_FINAL'
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
COUNTER={'source_main_rows':0,'REAL_control_rows':0,'main_generation_rows':0,'repeat_noise_rows':0,'source_probe_rows':0,'extra_guide_probe_rows':0,'response_fit_or_update':0,'HGB_attempts':0,'density_EM_attempts':0,'EVAL_read':0,'CAL_waveform_decode':0}
def write(n,v):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(ev.safe(v),indent=2,allow_nan=False)+'\n','utf8')
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def progress(phase):
    r={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'counters':COUNTER,'elapsed_s':time.monotonic()-START}
    write('GENERATION_PROGRESS.json',r);print(json.dumps(r),flush=True)
def charges(x):
    s=x.astype('f8')-x.astype('f8')[:,:100].mean(1,keepdims=True);qf=s[:,100:150].sum(1)*1000.;qt=s[:,100:400].sum(1)*1000.;return qf,qt,1-qf/qt
def run():
    check=json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'));assert check['pass'], 'fixed32 solver failed; no DEV generation'
    actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');meta=load(ROOT/'DEV_SOURCE_METADATA.npz');real=load(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz',keys=['reference_V','reference_ids','reference_groups'])
    source=actual['source_waveforms_V'];ids=actual['source_ids'];assert np.array_equal(ids,meta['source_ids']);generations={}
    pipeline=Pipeline(CFG,ROOT,ARM);progress('candidate main2000')
    pre,signal,electronic=infer(pipeline,source,actual['z'],actual['white'],actual['style'],CFG['generation_chunk']);detail={k:v.copy()for k,v in pipeline.details.items()};post,ad=adc(pre)
    np.savez_compressed(ROOT/ARM/'generated.npz',pre_ADC_V=pre,waveforms_V=post,source_ids=ids,**detail)
    support=detail['flow_applied'];COUNTER['source_main_rows']=2000;COUNTER['main_generation_rows']=2000
    generations[ARM]={'rows':2000,'all_source_ID_order_weights_and_composition_preserved':True,'ADC':ad,'standalone_flow_applied_N':2000,'G_applied_N':2000,'G_applied_N_compatibility_means_new_standalone_flow':True,'TRAIN_support_N':int(support.sum()),'TRAIN_support_outside_N':int((~support).sum()),'flow_applied_field_is_TRAIN_support_only':True,'BASE_fallback_N':0,'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),'flow_file':CFG['endpoint_weight_paths'][ARM],'source_clean_full480_input':True,'old_G_used':False,'charge_tangent_projection_inside_velocity':True,'component_labels_functional_not_physical_truth':True}
    write('GENERATION_RECEIPT.json',{'domains':generations,'counter':COUNTER});progress('candidate saved')
    pp=fd.probes(pipeline,post,actual['style'],ARM,ROOT,OUT,CFG,pipeline.edges,COUNTER);write('SOURCE_PROBES_SUMMARY.json',{ARM:pp})
    jj=fd.joint(pipeline,source,meta['composition'],ARM,ROOT,OUT,CFG,pipeline.edges,json.loads((ROOT/'VALLEY_DEFINITION.json').read_text('utf8')),COUNTER);write('JOINT8_SUMMARY.json',{ARM:jj})
    pick=np.flatnonzero(support)[:128];assert len(pick)==128
    j=np.arange(250,dtype='f8');pert=np.sin(2*np.pi*j/249.);pert-=pert.mean();pert*=.001/np.sqrt(np.mean(pert**2));delta=np.zeros((128,480),'f8');delta[:,150:400]=pert
    s0=source[pick];s1=(s0.astype('f8')+delta).astype('f4')
    changed,_,_=infer(pipeline,s1,actual['z'][pick],actual['white'][pick],actual['style'][pick],CFG['generation_chunk']);pd={k:v.copy()for k,v in pipeline.details.items()}
    dv=changed.astype('f8')-pre[pick].astype('f8');rs=np.sqrt(np.mean(dv*dv,1));identical=np.all(changed==pre[pick],axis=1)
    qf0,qt0,p0=charges(s0);qf1,qt1,p1=charges(s1)
    np.savez_compressed(OUT/'GUIDE_PERTURBATION_PRIVATE.npz',parent_index=pick,source_ids=ids[pick],original_guide_V=s0,actual_perturbed_guide_f32_V=s1,designed_perturbation_f64_V=delta,original_output_preADC_V=pre[pick],perturbed_output_preADC_V=changed,fixed_source_Qbin=detail['source_Qbin_actual_preADC'][pick],**pd)
    gp={'rows':128,'probe_kind':'fullSOURCE_input_to_standalone_flow not postG guide','compatibility_guide_named_payload_is_SOURCE':True,'selection':'first128 original TRAIN-supported source order','source_original_bin_fixed_for_description':True,'z_white_style_original_rows':True,'actual_perturbed_bin_change_N':int(np.sum(pd['source_Qbin_actual_preADC']!=detail['source_Qbin_actual_preADC'][pick])),
        'designed_segment_mean_V':float(pert.mean()),'designed_segment_RMS_V':float(np.sqrt(np.mean(pert**2))),'actual_f32_Qfast_mVns_change':ev.summary(qf1-qf0),'actual_f32_Qtotal_mVns_change':ev.summary(qt1-qt0),'actual_f32_PSD_change':ev.summary(p1-p0),'Qfast_exact_unchanged':bool(np.array_equal(qf0,qf1)),
        'preADC_output_RMS_V':ev.summary(rs),'output_nonidentical_fraction':float(1-identical.mean()),'ADC_LSB_V':1/16383,'pass':bool(1-identical.mean()>=.95 and np.median(rs)>1/16383),'effect_not_physical_truth':True}
    COUNTER['extra_guide_probe_rows']=128;write('GUIDE_PERTURBATION_SUMMARY.json',gp);write('GUIDE_PERTURBATION_COMPLETE_SUMMARY.json',gp)
    for name in ['SOURCE_NEURAL_FLOW_REAL_CONTROL']:
        ctrl=Pipeline(CFG,ROOT,name);path=ROOT/name;path.mkdir(exist_ok=True)
        cp,_,_=infer(ctrl,real['reference_V'],actual['z'],actual['white'],actual['style'],CFG['generation_chunk']);cd=ctrl.details;cpost,cad=adc(cp)
        np.savez_compressed(path/'generated.npz',pre_ADC_V=cp,waveforms_V=cpost,reference_ids=real['reference_ids'],reference_groups=real['reference_groups'],**cd)
        generations[name]={'rows':2000,'ADC':cad,'flow_file':CFG['endpoint_weight_paths'][name],'reference_ID_order_groups_preserved':True,'same_exact_main_random_arrays':True,'no_G4_composition_or_PDG_assignment':True,'stochastic_control_not_identity_claim':True,'pre_ADC_sha256':hashlib.sha256(cp.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(cpost.tobytes()).hexdigest()}
        COUNTER['REAL_control_rows']+=2000;COUNTER['main_generation_rows']+=2000;progress(name+' saved')
    old=json.loads((pathlib.Path(CFG['R39_root'])/'GENERATION_RECEIPT.json').read_text('utf8'))
    generations['BASE_REAL_CONTROL']={**old['domains']['BASE_REAL_CONTROL'],'reused_from':CFG['old_BASE_REAL_generated'],'new_inference_rows':0,'new_fits':0,'old_independent_replay_from':str(pathlib.Path(CFG['R39_root'])/'REPLAY_RECEIPT.json')}
    write('REAL_CONTROL_INPUT_BINDING.json',{'reference_array_from':str(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz'),'random_arrays_from':str(ROOT/'ACTUAL_GENERATION_INPUTS.npz'),'rows':2000,'reference_waveform_sha256':hashlib.sha256(real['reference_V'].tobytes()).hexdigest(),'reference_ids_sha256':hashlib.sha256(real['reference_ids'].tobytes()).hexdigest(),'random_sha256':{k:hashlib.sha256(actual[k].tobytes()).hexdigest()for k in ['z','white','style']},'RAW_identity_expected':False,'new_REAL_acquisition':0})
    write('GENERATION_RECEIPT.json',{'domains':generations,'counter':COUNTER,'elapsed_s':time.monotonic()-START,'NumPy':np.__version__,'torch':torch.__version__,'exact_main_and_auxiliary_preADC_saved':True,'functional_component_semantics':{'base_signal':'actual full SOURCE input','base_electronic':'projected initial Gaussian, not electronics','base_sum_rounding':'float32 initial sum residue','flow_increment':'learned full480 flow plus final cast residual; not independent signal'},'unused_random_inputs':['z','style'],'field_forward_rows_main_source_plus_auxiliary':pipeline.total_field_rows,'field_forward_rows_REAL_control':ctrl.total_field_rows})
    assert COUNTER['main_generation_rows']==4000 and COUNTER['repeat_noise_rows']==1024 and COUNTER['source_probe_rows']==832 and COUNTER['extra_guide_probe_rows']==128
    write('GENERATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete'});progress('complete')
try:run()
except BaseException as e:
    write('GENERATION_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
