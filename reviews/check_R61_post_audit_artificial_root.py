"""Artificial positive and independently perturbed negative cases; no science input."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,ctypes,copy,hashlib
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
T=R/'reviews/R61_POST_AUDIT_ARTIFICIAL_001';T.mkdir(exist_ok=False)
assert T.drive.upper()in ['E:','D:']
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[key]='4'
(T/'cache').mkdir();os.environ.update(TMP=str(T/'cache'),TEMP=str(T/'cache'),TMPDIR=str(T/'cache'))
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
hnd=kernel.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(hnd,ctypes.byref(allowed),ctypes.byref(system))
cpus=[i for i in range(ctypes.sizeof(ctypes.c_size_t)*8)if allowed.value&(1<<i)][:4]
assert kernel.SetProcessAffinityMask(hnd,sum(1<<i for i in cpus))
sys.path.insert(0,str(S))
import numpy as np
from precision_audit import precision_single_factor_audit

def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n','utf8')
p=T/'candidate/ARTIFICIAL';p.mkdir(parents=True);h=T/'historical/DIRECT_DISTRIBUTION_RESPONSE';h.mkdir(parents=True)
cfg={'arm':'ARTIFICIAL','historical_R59_root':str(h.parent),'input_paths':{'distribution_statistics':str(h.parent/'constants.npz')},
    'expected_R59_initial_state_SHA':'synthetic_initial','expected_R59_stream_hashes':{f'stream_{i}':str(i)*64 for i in range(6)},
    'sole_scientific_change':'artificial example','source_order_rule':'artificial fixed rule'}
receipt={'initial_state_SHA':cfg['expected_R59_initial_state_SHA'],'stream_hashes':copy.deepcopy(cfg['expected_R59_stream_hashes']),
    'single_factor_integrity':{k:True for k in ['same_R59_initial_state','all_six_consumed_stream_hashes_equal_R59','all_parent_exposures_equal_R59','final_random_states_equal_R59','frozen_R59_distribution_statistics_reused']},
    'precision_order_trace_rows':50000,'source_order_rule':cfg['source_order_rule']}
receipt['single_factor_integrity']['sole_change']=cfg['sole_scientific_change']
write(h/'TRAINING_RECEIPT.json',receipt)
fields=['source_ids','source_counts','REAL_reference_counts','REAL_identity_counts','source_bin0','REAL_bin0','source_h','source_original_psd','source_original_q']
arrays={k:np.array([1.,2.,3.]) for k in fields}
for a in [p,h]:
    np.savez_compressed(a/'PARENT_EXPOSURES_PRIVATE.npz',**arrays)
    write(a/'FINAL_RANDOM_STATES.json',{'example_state':17})
for target in [p.parent/'TRAIN_DISTRIBUTION_CONSTANTS.npz',h.parent/'constants.npz']:target.write_bytes(b'artificial identical frozen constants')
names=['original_order_loss','omitted_order_loss','original_nonzero_pair_mass','resolved_pair_fraction']
values=np.tile(np.array([3.,1.,.9,.45]),(50000,1));losses=np.full((50000,1),2.)
np.savez_compressed(p/'PRECISION_ORDER_BY_UPDATE_PRIVATE.npz',columns=np.array(names),values=values)
np.savez_compressed(p.parent/'SOURCE_PSD_PRECISION_PRIVATE.npz',source_ids=arrays['source_ids'],source_original_psd=arrays['source_original_psd'],PSD_interval_low=arrays['source_original_psd']-.01,PSD_interval_high=arrays['source_original_psd']+.01)
write(p.parent/'SOURCE_ORDER_PRECISION_DEFINITION.json',{'interval_rule':cfg['source_order_rule'],'all_original_denominator_pairs_retained':True,'new_threshold_fit':False})
run=lambda rec=receipt,loss=losses:precision_single_factor_audit(p.parent,cfg,rec,loss,['source_order'])
baseline=run();assert baseline['pass'],baseline
cases={}
for label,field in [('initial_state','initial_state_SHA'),('consumed_stream','stream_hashes')]:
    rec=copy.deepcopy(receipt)
    if field=='stream_hashes':rec[field]['stream_0']='f'*64
    else:rec[field]='changed'
    result=run(rec);assert not result['pass'];cases[label]=[k for k,v in result['checks'].items()if not v]
changed=copy.deepcopy(arrays);changed['source_counts'][0]+=1;np.savez_compressed(p/'PARENT_EXPOSURES_PRIVATE.npz',**changed)
result=run();assert not result['pass'];cases['parent_exposure']=[k for k,v in result['checks'].items()if not v]
np.savez_compressed(p/'PARENT_EXPOSURES_PRIVATE.npz',**arrays)
write(p/'FINAL_RANDOM_STATES.json',{'example_state':18});result=run();assert not result['pass'];cases['RNG_state']=[k for k,v in result['checks'].items()if not v]
write(p/'FINAL_RANDOM_STATES.json',{'example_state':17})
(p.parent/'TRAIN_DISTRIBUTION_CONSTANTS.npz').write_bytes(b'changed artificial constants');result=run();assert not result['pass'];cases['constants']=[k for k,v in result['checks'].items()if not v]
(p.parent/'TRAIN_DISTRIBUTION_CONSTANTS.npz').write_bytes(b'artificial identical frozen constants')
result=run(loss=np.full((50000,1),2.1));assert not result['pass'];cases['decomposition']=[k for k,v in result['checks'].items()if not v]
assert run()['pass']
record={'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'CPU_affinity':cpus,'pass':True,
    'scientific_data_read':0,'scientific_forward_or_fit':0,'positive_case_all_checks_pass':True,
    'independent_negative_perturbations_rejected':cases,'files_are_synthetic_under_E':True,
    'precision_audit_SHA256':hashlib.sha256((S/'precision_audit.py').read_bytes()).hexdigest()}
write(S/'POST_AUDIT_ARTIFICIAL_CHECK.json',record);write(T/'RECEIPT.json',record)
print(json.dumps(record,indent=2))
