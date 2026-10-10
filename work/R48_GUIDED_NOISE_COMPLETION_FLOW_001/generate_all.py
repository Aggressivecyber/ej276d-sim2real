from __future__ import annotations
import os,time,pathlib,json,datetime,traceback,hashlib
import numpy as np,torch
import evaluation as ev
from model import adc
from flow_runtime import Pipeline,infer
import flow_diagnostics as fd
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CFG=json.loads((ROOT/'config_cpu.json').read_text('utf8'));START=time.monotonic()
torch.set_num_threads(12);torch.set_num_interop_threads(12);torch.use_deterministic_algorithms(True)
COUNTER={'source_main_rows':0,'REAL_control_rows':0,'main_generation_rows':0,'repeat_noise_rows':0,'source_probe_rows':0,
 'extra_guide_probe_rows':0,'response_fit_or_update':0,'HGB_attempts':0,'density_EM_attempts':0,'EVAL_read':0,'CAL_waveform_decode':0}
def write(n,v):
 p=ROOT/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes((json.dumps(ev.safe(v),indent=2,allow_nan=False)+'\n').encode())
def load(p,keys=None):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def progress(phase):
 r={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'counters':COUNTER,'elapsed_s':time.monotonic()-START}
 write('GENERATION_PROGRESS.json',r);print(json.dumps(r),flush=True)
def charges(x):
 s=x.astype('f8')-x.astype('f8')[:,:100].mean(1,keepdims=True);qf=s[:,100:150].sum(1)*1000.;qt=s[:,100:400].sum(1)*1000.;return qf,qt,1-qf/qt
def initial_description(detail):
 fields=['input_variance100_V2','innovation_amplitude_squared_V2','initial_variance100_actual_V2','innovation_variance100_actual_V2',
  'preADC_charge_relative_error_f64','preADC_charge_relative_error_f32']
 result={'N':len(detail['conditioning_Qbin']),'noise_completion_triggered_N':int(detail['noise_completion_triggered'].sum()),
  'all_rows_same_universal_operator':True,'expected_variance_not_eventwise_equality':True}
 result['population']={k:ev.summary(detail[k]) for k in fields}
 result['by_actual_conditioning_bin']=[{'bin_id_1based':b+1,'N':int((detail['conditioning_Qbin']==b).sum()),
  'noise_completion_triggered_N':int(detail['noise_completion_triggered'][detail['conditioning_Qbin']==b].sum()),
  'fields':{k:ev.summary(detail[k][detail['conditioning_Qbin']==b])for k in fields}}for b in range(8)]
 return result
def run():
 assert json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))['both_pass']
 actual=load(pathlib.Path(CFG['frozen_eval_input_root'])/'ACTUAL_GENERATION_INPUTS.npz');meta=load(pathlib.Path(CFG['frozen_eval_input_root'])/'DEV_SOURCE_METADATA.npz')
 real=load(pathlib.Path(CFG['frozen_eval_input_root'])/'REAL_DEV_AND_REFERENCE_PRIVATE.npz',keys=['reference_V','reference_ids','reference_groups'])
 source=actual['source_waveforms_V'];ids=actual['source_ids'];assert np.array_equal(ids,meta['source_ids'])
 generations={};probes={};joint={};gps={};field_rows={};initial={};identity={}
 for arm,control in zip(CFG['arms'],['GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_COMPLETION_REAL_CONTROL']):
  pipeline=Pipeline(CFG,ROOT,arm);(ROOT/arm).mkdir(exist_ok=True);progress(arm+' SOURCE2000')
  pre,_,_=infer(pipeline,source,actual['z'],actual['white'],actual['style'],64);detail={k:v.copy()for k,v in pipeline.details.items()};post,ad=adc(pre)
  np.savez_compressed(ROOT/arm/'generated.npz',pre_ADC_V=pre,waveforms_V=post,source_ids=ids,**detail)
  support=detail['inside_TRAIN_support'];COUNTER['source_main_rows']+=2000;COUNTER['main_generation_rows']+=2000
  generations[arm]={'rows':2000,'all_source_ID_order_weights_and_composition_preserved':True,'ADC':ad,
   'standalone_flow_applied_N':2000,'G_applied_N':2000,'G_applied_N_compatibility_means_new_standalone_flow':True,
   'TRAIN_support_N':int(support.sum()),'TRAIN_support_outside_N':int((~support).sum()),'BASE_fallback_N':0,
   'pre_ADC_sha256':hashlib.sha256(pre.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),
   'flow_file':CFG['endpoint_weight_paths'][arm],'source_clean_full480_input':True,'old_G_used':False,
   'charge_tangent_projection_inside_velocity':True,'component_labels_functional_not_physical_truth':True,'solver_steps':64,'chunk':64}
  write('GENERATION_RECEIPT.json',{'domains':generations,'counter':COUNTER});progress(arm+' SOURCE saved')
  initial[arm]=initial_description(detail);write('INITIAL_NOISE_DESCRIPTIONS.json',initial)
  probes[arm]=fd.probes(pipeline,post,actual['style'],arm,ROOT,OUT,CFG,pipeline.edges,COUNTER);write('SOURCE_PROBES_SUMMARY.json',probes);progress(arm+' probes saved')
  joint[arm]=fd.joint(pipeline,source,meta['composition'],arm,ROOT,OUT,CFG,pipeline.edges,json.loads((pathlib.Path(CFG['frozen_eval_input_root'])/'VALLEY_DEFINITION.json').read_text('utf8')),COUNTER);write('JOINT8_SUMMARY.json',joint);progress(arm+' joint saved')
  pick=np.flatnonzero(support)[:128];assert len(pick)==128
  j=np.arange(250,dtype='f8');pert=np.sin(2*np.pi*j/249.);pert-=pert.mean();pert*=.001/np.sqrt(np.mean(pert**2));delta=np.zeros((128,480),'f8');delta[:,150:400]=pert
  s0=source[pick];s1=(s0.astype('f8')+delta).astype('f4')
  changed,_,_=infer(pipeline,s1,actual['z'][pick],actual['white'][pick],actual['style'][pick],64);pd={k:v.copy()for k,v in pipeline.details.items()}
  dv=changed.astype('f8')-pre[pick].astype('f8');rs=np.sqrt(np.mean(dv*dv,1));identical=np.all(changed==pre[pick],axis=1)
  qf0,qt0,p0=charges(s0);qf1,qt1,p1=charges(s1)
  np.savez_compressed(OUT/(arm+'_GUIDE_PERTURBATION_PRIVATE.npz'),parent_index=pick,source_ids=ids[pick],original_guide_V=s0,
   actual_perturbed_guide_f32_V=s1,designed_perturbation_f64_V=delta,original_output_preADC_V=pre[pick],perturbed_output_preADC_V=changed,
   fixed_source_Qbin=detail['source_Qbin_actual_preADC'][pick],**pd)
  gps[arm]={'rows':128,'probe_kind':'fullSOURCE input to standalone flow, not postG guide','selection':'first128 original TRAIN-supported source order',
   'source_original_bin_fixed_for_description':True,'z_white_style_original_rows':True,
   'actual_perturbed_bin_change_N':int(np.sum(pd['source_Qbin_actual_preADC']!=detail['source_Qbin_actual_preADC'][pick])),
   'designed_segment_mean_V':float(pert.mean()),'designed_segment_RMS_V':float(np.sqrt(np.mean(pert**2))),
   'actual_f32_Qfast_mVns_change':ev.summary(qf1-qf0),'actual_f32_Qtotal_mVns_change':ev.summary(qt1-qt0),
   'actual_f32_PSD_change':ev.summary(p1-p0),'Qfast_exact_unchanged':bool(np.array_equal(qf0,qf1)),
   'preADC_output_RMS_V':ev.summary(rs),'output_nonidentical_fraction':float(1-identical.mean()),'ADC_LSB_V':1/16383,
   'pass':bool(1-identical.mean()>=.95 and np.median(rs)>1/16383),'effect_not_physical_truth':True}
  COUNTER['extra_guide_probe_rows']+=128;write('GUIDE_PERTURBATION_SUMMARY.json',gps);write('GUIDE_PERTURBATION_COMPLETE_SUMMARY.json',gps)
  field_rows[arm]=pipeline.total_field_rows;progress(arm+' complete SOURCE set')
  ctrl=Pipeline(CFG,ROOT,control);path=ROOT/control;path.mkdir(exist_ok=True)
  cp,_,_=infer(ctrl,real['reference_V'],actual['z'],actual['white'],actual['style'],64);cd=ctrl.details;cpost,cad=adc(cp)
  np.savez_compressed(path/'generated.npz',pre_ADC_V=cp,waveforms_V=cpost,reference_ids=real['reference_ids'],reference_groups=real['reference_groups'],**cd)
  initial[control]=initial_description(cd);write('INITIAL_NOISE_DESCRIPTIONS.json',initial)
  before=ev.features(real['reference_V']);after=ev.features(cpost)
  identity[control]={'N':2000,'paired_same_REAL_rows':True,'claim_of_RAW_identity':False,
   'postADC_waveform_RMS_vs_original_V':ev.summary(np.sqrt(np.mean((cpost.astype('f8')-real['reference_V'].astype('f8'))**2,axis=1))),
   'PSD_change':ev.summary(after['features']['PSD']-before['features']['PSD']),
   'Qtotal_mVns_change':ev.summary(after['features']['Qtotal_mVns']-before['features']['Qtotal_mVns']),
   'baseline_RMS_V_change':ev.summary(after['features']['baseline_RMS_V']-before['features']['baseline_RMS_V']) if 'baseline_RMS_V' in after['features'] else None}
  write('REAL_IDENTITY_DESCRIPTIONS.json',identity)
  generations[control]={'rows':2000,'ADC':cad,'flow_file':CFG['endpoint_weight_paths'][control],'reference_ID_order_groups_preserved':True,
   'same_exact_main_random_arrays':True,'no_G4_composition_or_PDG_assignment':True,'stochastic_control_not_identity_claim':True,
   'pre_ADC_sha256':hashlib.sha256(cp.tobytes()).hexdigest(),'post_ADC_sha256':hashlib.sha256(cpost.tobytes()).hexdigest(),'solver_steps':64,'chunk':64}
  COUNTER['REAL_control_rows']+=2000;COUNTER['main_generation_rows']+=2000;field_rows[control]=ctrl.total_field_rows
  write('GENERATION_RECEIPT.json',{'domains':generations,'counter':COUNTER});progress(control+' saved')
 old=json.loads((pathlib.Path(CFG['R39_root'])/'GENERATION_RECEIPT.json').read_text('utf8'))
 generations['BASE_REAL_CONTROL']={**old['domains']['BASE_REAL_CONTROL'],'reused_from':CFG['old_BASE_REAL_generated'],'new_inference_rows':0,'new_fits':0,
  'old_independent_replay_from':str(pathlib.Path(CFG['R39_root'])/'REPLAY_RECEIPT.json')}
 write('REAL_CONTROL_INPUT_BINDING.json',{'reference_array_from':str(pathlib.Path(CFG['frozen_eval_input_root'])/'REAL_DEV_AND_REFERENCE_PRIVATE.npz'),'random_arrays_from':str(pathlib.Path(CFG['frozen_eval_input_root'])/'ACTUAL_GENERATION_INPUTS.npz'),
  'rows':2000,'reference_waveform_sha256':hashlib.sha256(real['reference_V'].tobytes()).hexdigest(),'reference_ids_sha256':hashlib.sha256(real['reference_ids'].tobytes()).hexdigest(),
  'random_sha256':{k:hashlib.sha256(actual[k].tobytes()).hexdigest()for k in ['z','white','style']},'RAW_identity_expected':False,'new_REAL_acquisition':0})
 write('GENERATION_RECEIPT.json',{'domains':generations,'counter':COUNTER,'elapsed_s':time.monotonic()-START,'NumPy':np.__version__,'torch':torch.__version__,
  'exact_main_and_auxiliary_preADC_saved':True,'functional_component_semantics':{'source_input':'fixed complete guide','innovation':'projected colored seed, not physical electronics','flow_plus_final_cast':'total learned correction not independent noise'},
  'unused_random_inputs':['z','style'],'field_forward_rows_by_arm':field_rows,'all_generation_rows':sum(COUNTER[k]for k in ['main_generation_rows','repeat_noise_rows','source_probe_rows','extra_guide_probe_rows'])})
 assert [COUNTER[k]for k in ['main_generation_rows','repeat_noise_rows','source_probe_rows','extra_guide_probe_rows']]==[8000,2048,1664,256]
 write('GENERATION_NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counters':COUNTER,'status':'complete'});progress('complete11968')
try:run()
except BaseException as e:
 write('GENERATION_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,'counters':COUNTER});raise
