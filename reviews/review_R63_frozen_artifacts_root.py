from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,hashlib,ctypes
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001'
assert S.drive.upper()in ['E:','D:']
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[key]='4'
for n in ['tmp','torch','matplotlib']:(S/'cache'/n).mkdir(parents=True,exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMP=str(S/'cache/tmp'),TEMP=str(S/'cache/tmp'),
    TMPDIR=str(S/'cache/tmp'),TORCH_HOME=str(S/'cache/torch'),MPLCONFIGDIR=str(S/'cache/matplotlib'))
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
handle=kernel.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle,ctypes.byref(allowed),ctypes.byref(system))
cpus=[i for i in range(ctypes.sizeof(ctypes.c_size_t)*8)if allowed.value&(1<<i)][:4]
assert kernel.SetProcessAffinityMask(handle,sum(1<<i for i in cpus))
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(4)
sys.path.insert(0,str(S))
from network import ResponseNetwork
from post_audit import training_audit
def read(n):return json.loads((S/n).read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
C=read('config.json');arm=C['arm'];train=read(arm+'/TRAINING_RECEIPT.json')
assert read('LOCAL_PIPELINE_CONTROL.json')['status']=='registered_pipeline_complete_pending_root_QA'
assert not read('LIVE_OBSERVATION.json')['actual_process']['exists']
assert read('REPLAY_RECEIPT.json')['pass'] and read('REPLAY_RECEIPT.json')['rows']==4000
weight=S/arm/'FINAL_FLOW.pt';state=torch.load(weight,map_location='cpu',weights_only=True)
h=hashlib.sha256()
for k,v in sorted(state.items()):
    assert isinstance(v,torch.Tensor)and torch.isfinite(v).all(),k
    h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
assert h.hexdigest()==train['final_state_SHA']
model=ResponseNetwork();model.load_state_dict(state,strict=True)
assert sum(p.numel()for p in model.parameters())==train['network_parameter_count']
binding=read('INPUT_BINDING.json')['inputs']
assert sha(S/'UNIT_CHARGE_CONSTANTS.npz')==binding['constants_path']['sha256']
assert sha(S/'NOISE_CALIBRATION.npz')==binding['noise_calibration_path']['sha256']
protected=read('POSTPROCESS_PREPARATION.json')['training_files_and_protocol_acceptance_untouched']
for path,digest in protected.items():assert sha(Path(path))==digest,path
post=read('POSTPROCESS_FILE_BINDING.json')['bound_SHA256']
for n,digest in post.items():assert sha(S/n)==digest,n
P=read('config_post.json')
assert len(set(P['endpoint_weight_paths'].values()))==1
H=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
local_cfg=dict(P);local_cfg['historical_R61_root']=str(H)
local_cfg['input_paths']=dict(P['input_paths'],distribution_statistics=str(H/'TRAIN_DISTRIBUTION_CONSTANTS.npz'))
local_audit=training_audit(S,local_cfg,train)
assert local_audit['pass'],local_audit['checks']
assert local_audit['checks']==read('TRAINING_FROZEN_AUDIT.json')['checks']
assert train['initial_state_SHA']==C['expected_R61_initial_state_SHA']
assert {k:train['stream_hashes'][k]for k in C['expected_R61_stream_hashes']}==C['expected_R61_stream_hashes']
gen=read('GENERATION_RECEIPT.json')
assert sum(gen['actual_network_rows_by_arm'].values())==5984
assert sum(gen['field_forward_rows_by_arm'].values())==5984*256
class PMC(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t)for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
psapi=ctypes.WinDLL('psapi',use_last_error=True);psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
memory=PMC();memory.cb=ctypes.sizeof(memory)
assert psapi.GetProcessMemoryInfo(handle,ctypes.byref(memory),memory.cb)
assert memory.PeakWorkingSetSize<=4*1024**3
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'pass':True,
    'actual_weight_tensor_SHA_matches_training':h.hexdigest(),'weight_file_SHA256':sha(weight),
    'strict_native_Windows_CPU_model_load':True,'torch':torch.__version__,'NumPy':np.__version__,
    'CPU_affinity':cpus,'worker_peak_RSS_bytes':memory.PeakWorkingSetSize,
    'frozen_priors_training_code_protocol_acceptance_and_postcode_match':True,
    'SOURCE_REAL_use_one_weight_file':True,'actual_generation_rows_verified':5984,
    'independent_GPU_replay_rows_verified':4000,'new_forward_rows':0,'new_fits_or_updates':0,
    'local_CPU_waveform_inference_executed':False,'model_goal_complete':False,
    'independent_local_saved_TRAIN_record_audit':local_audit,'remote_and_local_training_audit_checks_equal':True}
(S/'FROZEN_ARTIFACT_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt,indent=2))
