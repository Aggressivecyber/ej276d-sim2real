"""Synthetic saved-record checks, with independent negative perturbations."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,ctypes,copy,hashlib,time,threading
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001';START=time.monotonic()
T=R/'reviews/R63_POST_AUDIT_ARTIFICIAL_001';T.mkdir(exist_ok=False);assert T.drive.upper()in ['E:','D:']
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[key]='4'
(T/'cache').mkdir();os.environ.update(TMP=str(T/'cache'),TEMP=str(T/'cache'),TMPDIR=str(T/'cache'))
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
handle=kernel.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t();assert kernel.GetProcessAffinityMask(handle,ctypes.byref(allowed),ctypes.byref(system))
cpus=[i for i in range(ctypes.sizeof(ctypes.c_size_t)*8)if allowed.value&(1<<i)][:4];assert kernel.SetProcessAffinityMask(handle,sum(1<<i for i in cpus))
class PMC(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t)for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
psapi=ctypes.WinDLL('psapi',use_last_error=True);psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
stop=threading.Event();peak=0
def monitor():
 global peak
 while not stop.wait(.1):
  mi=PMC();mi.cb=ctypes.sizeof(mi)
  if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(mi),mi.cb):os._exit(138)
  peak=max(peak,mi.PeakWorkingSetSize)
  if peak>4*1024**3:os._exit(137)
threading.Thread(target=monitor,daemon=True).start();sys.path.insert(0,str(S))
import numpy as np
from coverage_audit import coverage_extension_audit,reconstruct_temporal_stream
def write(p,v):p.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=T/'candidate/ARTIFICIAL';p.mkdir(parents=True);h=T/'historical/PRECISION_RESOLVED_RESPONSE';h.mkdir(parents=True)
seed=20261011631;stream='baseline_projection_directions';updates=9
old_streams={f'stream_{i}':str(i)*64 for i in range(6)}
old_keys=['full_window','PSD','source_order','source_valley','noise','REAL_identity','box_correction']
cfg={'arm':'ARTIFICIAL','historical_R61_root':str(h.parent),'input_paths':{'distribution_statistics':str(h.parent/'constants.npz')},
 'expected_R61_initial_state_SHA':'synthetic_initial','expected_R61_stream_hashes':old_streams,'PCG64_seeds':{stream:seed},'updates':updates,'REAL_N':55,
 'sole_scientific_change':'artificial coverage extension','source_order_rule':'artificial fixed rule','feature_names':['V480','U480','DU479','D2U478'],
 'loss_weights':{k:1. for k in [*old_keys,'coordinate_RMS','temporal_joint']},'temporal_scale_floor':[1/16383/10]+[.01]*30}
digest,newstate=reconstruct_temporal_stream(seed,updates)
receipt={'initial_state_SHA':cfg['expected_R61_initial_state_SHA'],'stream_hashes':{**copy.deepcopy(old_streams),stream:digest},
 'objective_extension_integrity':{k:True for k in ['same_R61_initial_state','all_six_consumed_stream_hashes_equal_R61','all_parent_exposures_equal_R61','original_six_final_random_states_equal_R61','frozen_R61_distribution_statistics_reused','seventh_direction_stream_recorded_independently']},
 'precision_order_trace_rows':updates,'source_order_rule':cfg['source_order_rule'],'coverage_trace_rows':updates,'temporal_direction_calls':updates}
receipt['objective_extension_integrity']['sole_change']=cfg['sole_scientific_change']
write(h/'TRAINING_RECEIPT.json',{'initial_state_SHA':receipt['initial_state_SHA'],'stream_hashes':old_streams})
fields=['source_ids','source_counts','REAL_reference_counts','REAL_identity_counts','source_bin0','REAL_bin0','source_h','source_original_psd','source_original_q'];arrays={k:np.array([1.,2.,3.])for k in fields}
for a in [p,h]:np.savez_compressed(a/'PARENT_EXPOSURES_PRIVATE.npz',**arrays)
oldstates={k:{'example_state':17}for k in old_streams};states={**oldstates,stream:newstate};write(h/'FINAL_RANDOM_STATES.json',oldstates);write(p/'FINAL_RANDOM_STATES.json',states)
np.savez_compressed(h.parent/'constants.npz',REAL_counts=np.arange(1,11))
(p.parent/'TRAIN_DISTRIBUTION_CONSTANTS.npz').write_bytes((h.parent/'constants.npz').read_bytes())
values=np.tile(np.array([3.,1.,.9,.45]),(updates,1));names=['original_order_loss','omitted_order_loss','original_nonzero_pair_mass','resolved_pair_fraction']
np.savez_compressed(p/'PRECISION_ORDER_BY_UPDATE_PRIVATE.npz',columns=np.array(names),values=values)
columns=['total_loss',*cfg['loss_weights']];losses=np.ones((updates,len(columns)))
losses[:,columns.index('source_order')]=2.;losses[:,columns.index('coordinate_RMS')]=2.;losses[:,columns.index('temporal_joint')]=2.;losses[:,0]=12.
cn=['original_seven_total','coordinate_RMS','temporal_joint','temporal_marginal','temporal_sliced',*[k+'_coordinate_RMS'for k in cfg['feature_names']]]
coverage=np.tile(np.array([8.,2.,2.,1.,3.,2.,2.,2.,2.]),(updates,1))
np.savez_compressed(p/'COVERAGE_BY_UPDATE_PRIVATE.npz',columns=np.array(cn),values=coverage)
for d in [p.parent,h.parent]:np.savez_compressed(d/'SOURCE_PSD_PRECISION_PRIVATE.npz',source_ids=arrays['source_ids'],source_original_psd=arrays['source_original_psd'],PSD_interval_low=arrays['source_original_psd']-.01,PSD_interval_high=arrays['source_original_psd']+.01)
write(p.parent/'SOURCE_ORDER_PRECISION_DEFINITION.json',{'interval_rule':cfg['source_order_rule'],'all_original_denominator_pairs_retained':True,'new_threshold_fit':False})
center=np.zeros((10,31));scale=np.ones((10,31))
np.savez_compressed(p.parent/'TRAIN_TEMPORAL_CONSTANTS.npz',temporal_center=center,temporal_scale=scale,temporal_REAL_counts=np.arange(1,11),temporal_zero_power_counts=np.zeros(10,dtype='i8'))
meta={'center':center.tolist(),'scale':scale.tolist(),'fitted_REAL_TRAIN_rows':55,'new_statistics_SHA256':sha(p.parent/'TRAIN_TEMPORAL_CONSTANTS.npz'),'old_statistics_SHA256':sha(p.parent/'TRAIN_DISTRIBUTION_CONSTANTS.npz'),'fitted_before_first_update':True,'scientific_response_fit':False,'teacher_fit':False}
write(p.parent/'TRAIN_TEMPORAL_CONSTANTS.json',meta)
run=lambda rec=receipt,loss=losses,config=cfg:coverage_extension_audit(p.parent,config,rec,loss,columns)
baseline=run();assert baseline['pass'],baseline
cases={}
def rejected(label,result):
 assert not result['pass'],label
 cases[label]=[k for k,v in result['checks'].items()if not v]
for label,key in [('initial_state','initial_state_SHA'),('original_stream','stream_hashes'),('new_stream','stream_hashes')]:
 rec=copy.deepcopy(receipt)
 if key=='stream_hashes':rec[key][stream if label=='new_stream'else'stream_0']='f'*64
 else:rec[key]='changed'
 rejected(label,run(rec))
changed=copy.deepcopy(arrays);changed['source_counts'][0]+=1;np.savez_compressed(p/'PARENT_EXPOSURES_PRIVATE.npz',**changed);rejected('parent_exposure',run());np.savez_compressed(p/'PARENT_EXPOSURES_PRIVATE.npz',**arrays)
changed=copy.deepcopy(states);changed[stream]['state']['state']+=1;write(p/'FINAL_RANDOM_STATES.json',changed);rejected('new_RNG_state',run());write(p/'FINAL_RANDOM_STATES.json',states)
changed=coverage.copy();changed[0,0]+=1;np.savez_compressed(p/'COVERAGE_BY_UPDATE_PRIVATE.npz',columns=np.array(cn),values=changed);rejected('old_seven_decomposition',run())
changed=coverage.copy();changed[0,3]+=1;np.savez_compressed(p/'COVERAGE_BY_UPDATE_PRIVATE.npz',columns=np.array(cn),values=changed);rejected('temporal_half_sum',run())
changed=coverage.copy();changed[0,5]+=1;np.savez_compressed(p/'COVERAGE_BY_UPDATE_PRIVATE.npz',columns=np.array(cn),values=changed);rejected('coordinate_four_view_mean',run())
np.savez_compressed(p/'COVERAGE_BY_UPDATE_PRIVATE.npz',columns=np.array(cn),values=coverage)
changed=copy.deepcopy(meta);changed['fitted_before_first_update']=False;write(p.parent/'TRAIN_TEMPORAL_CONSTANTS.json',changed);rejected('post_update_fit_metadata',run());write(p.parent/'TRAIN_TEMPORAL_CONSTANTS.json',meta)
changed=copy.deepcopy(cfg);changed['loss_weights']['temporal_joint']=2.;rejected('outer_loss_weight',run(config=changed))
changed=losses.copy();changed[:,columns.index('source_order')]+=.1;rejected('precision_order_decomposition',run(loss=changed))
assert run()['pass']
record={'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'CPU_affinity':cpus,'peak_RSS_bytes':peak,'elapsed_s':time.monotonic()-START,'pass':True,'scientific_data_read':0,'scientific_forward_or_fit':0,
 'positive_case_all_checks_pass':True,'independent_negative_perturbations_rejected':cases,'files_are_synthetic_under_E':True,
 'synthetic_updates':updates,'real_stage_still_requires_exactly_50000_updates':True,'coverage_audit_SHA256':sha(S/'coverage_audit.py')}
write(S/'POST_AUDIT_ARTIFICIAL_CHECK.json',record);write(T/'RECEIPT.json',record);stop.set();print(json.dumps(record,indent=2))
