from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,hashlib
import numpy as np,torch
import evaluation as ev
from model import adc
from flow_runtime import Pipeline,infer
import flow_diagnostics as fd
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CFG=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
COUNTER={'source_main_rows':0,'main_generation_rows':0,'repeat_noise_rows':0,'source_probe_rows':0,
    'extra_guide_probe_rows':0,'response_fit_or_update':0,'HGB_attempts':0,'density_EM_attempts':0,'EVAL_read':0,'CAL_waveform_decode':0}
def write(n,v):
    p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(ev.safe(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def progress(phase):
    v={'PID':os.getpid(),'phase':phase,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'counters':COUNTER,'elapsed_s':time.monotonic()-START}
    write('GENERATION_PROGRESS.json',v);print(json.dumps(v),flush=True)
def charges(x):
    s=x.astype('f8')-x.astype('f8')[:,:100].mean(1,keepdims=True)
    qf=s[:,100:150].sum(1)*1000.;qt=s[:,100:400].sum(1)*1000.
    return qf,qt,1-qf/qt
def run():
    actual=load(ROOT/'ACTUAL_GENERATION_INPUTS.npz');meta=load(ROOT/'DEV_SOURCE_METADATA.npz')
    source=actual['source_waveforms_V'];ids=actual['source_ids'];assert np.array_equal(ids,meta['source_ids'])
    name='FULL_COVARIANCE_CALIBRATION';pipeline=Pipeline(CFG,ROOT,name)
    progress('source main2000 start')
    pre,signal,electronic=infer(pipeline,source,actual['z'],actual['white'],actual['style'],CFG['generation_chunk'])
    detail={k:v.copy() for k,v in pipeline.details.items()};post,ad=adc(pre)
    COUNTER['source_main_rows']=2000;COUNTER['main_generation_rows']=2000
    np.savez_compressed(ROOT/name/'generated.npz',pre_ADC_V=pre,waveforms_V=post,source_ids=ids,**detail)
    old=load(pathlib.Path(CFG['R32_root'])/'CONTROL_FINAL/generated.npz',keys=['pre_ADC_V','source_ids'])
    same=np.array_equal(detail['base_pre_ADC_V'],old['pre_ADC_V']) and np.array_equal(ids,old['source_ids']);assert same
    generations={name:{'rows':2000,'all_original_source_ids_preserved':True,'ADC':ad,
        'edit_applied_N':int(detail['flow_applied'].sum()),'unsupported_source_Q_base_fallback_N':int((~detail['flow_applied']).sum()),
        'base_preADC_matches_R32_bitwise':bool(same),'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),
        'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),'map_file':CFG['endpoint_weight_paths'][name],
        'base_G_weight_file':CFG['base_G'],'source_clean_full480_input':True,
        'mapping':'one TRAIN-fitted full480 Gaussian Monge map per original sourceQ bin; support outside base unchanged',
        'physical_final_signal_electronic_split':'unknown'}}
    write('GENERATION_RECEIPT.json',{'domains':generations,'counter':COUNTER});progress('source main2000 saved')
    pp=fd.probes(pipeline,post,actual['style'],name,ROOT,OUT,CFG,pipeline.edges,COUNTER)
    write('SOURCE_PROBES_SUMMARY.json',{name:pp});progress('source832 probes saved')
    jj=fd.joint(pipeline,source,meta['composition'],name,ROOT,OUT,CFG,pipeline.edges,
        json.loads((ROOT/'VALLEY_DEFINITION.json').read_text('utf8')),COUNTER)
    write('JOINT8_SUMMARY.json',{name:jj});progress('joint1024 saved')
    pick=np.flatnonzero(detail['flow_applied'])[:128];assert len(pick)==128
    j=np.arange(250,dtype='f8');pert=np.sin(2*np.pi*j/249.);pert-=pert.mean();pert*=.001/np.sqrt(np.mean(pert**2))
    delta=np.zeros((128,480),'f8');delta[:,150:400]=pert
    g0=detail['base_pre_ADC_V'][pick];g1=(g0.astype('f8')+delta).astype('f4');fixed_b=detail['source_Qbin_actual_preADC'][pick]
    perturbed=pipeline.edit(g1,fixed_b);pd={k:v.copy() for k,v in pipeline.details.items()}
    dif=perturbed.astype('f8')-pre[pick].astype('f8');rs=np.sqrt(np.mean(dif*dif,1));identical=np.all(perturbed==pre[pick],axis=1)
    qf0,qt0,p0=charges(g0);qf1,qt1,p1=charges(g1);qfi,qti,pi=charges(g0.astype('f8')+delta)
    np.savez_compressed(OUT/'GUIDE_PERTURBATION_PRIVATE.npz',parent_index=pick,source_ids=ids[pick],original_guide_V=g0,
        actual_perturbed_guide_f32_V=g1,designed_perturbation_f64_V=delta,original_output_preADC_V=pre[pick],
        perturbed_output_preADC_V=perturbed,fixed_source_Qbin=fixed_b,**pd)
    gp={'rows':128,'selection':'first128 supported original source order','fixed_source_bin_bitwise':bool(np.array_equal(pd['source_Qbin_actual_preADC'],fixed_b)),
        'original_source_and_G_noise_preserved':True,'designed_segment_mean_V':float(pert.mean()),'designed_segment_RMS_V':float(np.sqrt(np.mean(pert**2))),
        'actual_f32_Qfast_mVns_change':ev.summary(qf1-qf0),'actual_f32_Qtotal_mVns_change':ev.summary(qt1-qt0),
        'actual_f32_PSD_change':ev.summary(p1-p0),'intended_f64_Qfast_mVns_change':ev.summary(qfi-qf0),
        'intended_f64_Qtotal_mVns_change':ev.summary(qti-qt0),'intended_f64_PSD_change':ev.summary(pi-p0),
        'Qfast_exact_unchanged':bool(np.array_equal(qf0,qf1)),'preADC_output_RMS_V':ev.summary(rs),
        'output_nonidentical_fraction':float(1-identical.mean()),'ADC_LSB_V':1/16383,
        'pass':bool(1-identical.mean()>=.95 and np.median(rs)>1/16383),'effect_not_physical_truth':True}
    assert gp['fixed_source_bin_bitwise'];COUNTER['extra_guide_probe_rows']=128
    write('GUIDE_PERTURBATION_SUMMARY.json',gp);write('GUIDE_PERTURBATION_COMPLETE_SUMMARY.json',gp)
    write('GENERATION_RECEIPT.json',{'domains':generations,'counter':COUNTER,'elapsed_s':time.monotonic()-START,
        'NumPy':np.__version__,'torch':torch.__version__,'exact_main_and_auxiliary_preADC_saved':True,
        'flow_named_fields_are_compatibility_aliases_for_calibration_delta':True,'new_real_guide_controls':0})
    assert COUNTER['main_generation_rows']==2000 and COUNTER['repeat_noise_rows']==1024 and COUNTER['source_probe_rows']==832 and COUNTER['extra_guide_probe_rows']==128
    write('GENERATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete'});progress('generation complete')
try:run()
except BaseException as e:
    write('GENERATION_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
