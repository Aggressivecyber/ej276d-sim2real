"""Independent saved-artifact arithmetic and binding review; no model forward."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,hashlib,time,ctypes,copy
START=time.monotonic()
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[n]='4'
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
handle=kernel.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle,ctypes.byref(allowed),ctypes.byref(system))
cpus=[i for i in range(64)if allowed.value&(1<<i)][:4];assert kernel.SetProcessAffinityMask(handle,sum(1<<i for i in cpus))
import numpy as np
from scipy.stats import wasserstein_distance,spearmanr
R=Path(__file__).resolve().parents[1];ID='R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001';S=R/'work'/ID;H=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
read=lambda p:json.loads(p.read_text('utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda n:{k:v.copy()for k,v in np.load(S/n,allow_pickle=False).items()}
def close(a,b):assert np.allclose(a,b,atol=1e-11,rtol=1e-10),(a,b)
B=read(S/'INPUT_BINDING.json');C=read(S/'config.json');result=read(S/'RESULT.json');native=read(S/'diagnose_train.py.NATIVE_RECEIPT.json');checks={}
assert native['actual_worker_native_exit']==native['worker_self_receipt']['native_exit']==0 and native['termination_reason']is None and native['handshake_before_imports']
assert not read(S/'LIVE_OBSERVATION.json')['actual_process']['exists']
assert all(v['native_exit']==0 for v in read(S/'FETCH_RECEIPT.json')['events'])
assert result['frozen_weight_before_SHA']==result['frozen_weight_after_SHA']==B['frozen_weight_tensor_SHA']
new=read(S/'ROOT_IMPLEMENTATION_REVIEW.json');old=read(S/'repairs/ADC_CHECK_001/ROOT_IMPLEMENTATION_REVIEW.json')
assert all(sha(S/v['name'])==v['sha256']for v in new['files'])
assert [v['name']for v in new['files']if v['sha256']!=next(z['sha256']for z in old['files']if z['name']==v['name'])]==['ARTIFICIAL_CHECK_LOCAL.json','artificial_check.py']
assert all(sha(Path(v['path']))==v['sha256']for v in B['original_acceptance'])
assert all(sha(S/name)==sha(H/name)==v['sha256']for name,v in B['frozen_R61_helpers'].items())
for key,v in B['inputs'].items():
    if key!='TRAIN_path':assert sha(H/v['path'].split(C['historical_R61_root']+'/',1)[1])==v['sha256'],key
failed=read(S/'repairs/ADC_CHECK_001/diagnose_train.py.NATIVE_RECEIPT.json');assert failed['actual_worker_native_exit']==1
assert read(S/'repairs/ADC_CHECK_001/DIAGNOSTIC_FAILURE.json')['counter']['neural_forward_rows_completed']==0
for name in ['ARTIFICIAL_CHECK_LOCAL.json','ARTIFICIAL_CHECK_SERVER.json','ARTIFICIAL_CHECK_CUDA.json']:assert read(S/name)['pass']
adc=read(S/'ARTIFICIAL_CHECK_CUDA.json')['checks']['CPU_numpy_vs_selected_torch_backend_ADC']
assert all(v['integer_codes_bitwise_equal']and v['float64_max_absolute_difference_V']<=np.finfo('f8').eps for v in adc.values())
assert read(S/'ARTIFICIAL_CHECK_CUDA.json')['checks']['traced_frozen_solve_bitwise']['pass']
checks['frozen_bindings_unchanged_and_pre_TRAIN_assertion_failure_preserved']=True
access=read(S/'DATA_ACCESS_RECEIPT.json');assert access['decoded_fields']==['source','target','source_ids','h','source_bin','target_bin']
cnt=result['counter'];assert cnt['full_SOURCE_rows_completed']==6593 and cnt['full_REAL_rows_completed']==21047
assert cnt['neural_forward_rows_completed']==27640 and cnt['neural_forward_calls_completed']==433 and cnt['mathematical_Heun_field_rows']==7075840 and cnt['saved_endpoint_batches_completed']==32
assert all(cnt[k]==0 for k in ['response_training_runs','response_updates','HGB_fits','density_fits','teacher_waveforms','autograd_calls','DEV_read','EVAL_read','CAL_decode'])
checks['native_complete_registered_counts_and_TRAIN_whitelist']=True
path=read(S/'PATH_SUMMARY.json');base=read(S/'BASELINE_JOINT_SUMMARY.json');loss=read(S/'LOSS_COVERAGE_SUMMARY.json');psd=read(S/'PSD_SUMMARY.json');projection=read(S/'PROJECTION_SUMMARY.json')
p=load('PATH_STATISTICS_PRIVATE.npz');b=load('BASELINE_JOINT_PRIVATE.npz');l=load('LOSS_COVERAGE_PRIVATE.npz');s=load('PER_PARENT_PRIVATE.npz')
assert hashlib.sha256(s['source_ids'].tobytes()).hexdigest()==B['source_parent_ID_SHA']
regions=list(range(-1,9));features=['V480','U480','DU479','D2U478'];sb=s['source_raw_region'];rb=s['REAL_raw_region']
assert len(sb)==6593 and len(rb)==21047 and set(sb)==set(rb)==set(regions)
scount=np.array([(sb==r).sum()for r in regions]);rcount=np.array([(rb==r).sum()for r in regions]);mass=scount/scount.sum()
assert all(np.isfinite(v).all()for data in [p,b,l]for v in data.values())
assert len(path['rows'])==110
for row in path['rows']:
    for name in features:
        vector=p['_'.join([row['domain'],row['stage'],row['scope'],name,'W1'])]
        assert len(vector)=={'V480':480,'U480':480,'DU479':479,'D2U478':478}[name]and np.min(vector)>=0
        close(vector.mean(),row['feature_mean_W1'][name])
        if name=='V480':
            for lo,hi in [(0,100),(100,150),(150,400),(400,480)]:close(vector[lo:hi].mean(),row['voltage_segment_mean_W1'][str(lo)+':'+str(hi)])
    n=6593 if row['domain']=='SOURCE'else 21047
    if row['original_region']is not None:n=int((sb==row['original_region']).sum())if row['domain']=='SOURCE'else int((rb==row['original_region']).sum())
    assert row['N']==n
find=lambda st:next(v for v in path['rows']if v['domain']=='SOURCE'and v['stage']==st and v['scope']=='all')
delta=np.array(find('neural_raw')['boundary_W1_V'])-np.array(find('noise_initial')['boundary_W1_V'])
flags=(delta[[1,5,6]]>1/16383).tolist();marker=path['neural_boundary_increase'];assert marker['flags']==flags and marker['activated']==(sum(flags)>=2)
checks['all_path_coordinate_reductions_segment_summaries_and_boundary_marker']=True
assert len(base['rows'])==11
for row in base['rows']:
    scope=row['scope'];r=row['original_region'];si=np.arange(6593)if r is None else np.flatnonzero(sb==r);ri=np.arange(21047)if r is None else np.flatnonzero(rb==r)
    ref=b[scope+'_REAL_reference_covariance_V2'];den=np.linalg.norm(ref);assert den>0
    se=np.linalg.norm(b[scope+'_SOURCE_covariance_V2']-ref)/den;ce=np.linalg.norm(b[scope+'_REAL_control_covariance_V2']-ref)/den
    close(se,row['SOURCE_covariance_relative_Frobenius']);close(ce,row['REAL_control_covariance_relative_Frobenius'])
    for domain,index,key in [('SOURCE',si,'SOURCE_baseline_feature_W1'),('REAL_control',ri,'REAL_control_baseline_feature_W1')]:
        expected=[wasserstein_distance(b[domain+'_baseline_features'][index,j],b['REAL_reference_baseline_features'][ri,j])for j in range(31)]
        close(expected,row[key])
    null=row['empirical_reference_errors'];assert len(null)==32 and min(null)>=0
    close(np.quantile(null,[0,.025,.5,.975,1]),row['empirical_reference_quantiles'])
    assert row['SOURCE_exceeds_all_32_references_and_REAL_control']==bool(se>max(null)and se>ce)
    for name in ['SOURCE','REAL_control','REAL_reference']:
        matrix=b[scope+'_'+name+'_covariance_V2'];close(matrix,matrix.T);assert np.min(np.diag(matrix))>=0
flagged=sum(v['SOURCE_exceeds_all_32_references_and_REAL_control']for v in base['rows']if v['original_region']in range(8))
assert base['baseline_joint_residual']['regions']==flagged and base['baseline_joint_residual']['activated']==(flagged>=6)
checks['covariance_residuals_saved_feature_W1_reference_quantiles_and_marker']=True
close(l['source_original_region_mass'],mass)
assert l['full_tables'].shape==(3,10,4,2)and l['batch_tables'].shape==(32,3,10,4,2)
close((l['batch_tables'].mean((3,4))*mass[None,None,:]).sum(2),l['batch_total_losses'])
close((l['batch_tables'].mean(4)*mass[None,None,:,None]).sum(2),l['batch_feature_losses'])
for i,name in enumerate(['SOURCE_REAL','REAL_REAL','REAL_control_REAL']):
    row=loss['comparisons'][name];full=l['full_tables'][i];total=l['batch_total_losses'][:,i]
    close((full.mean((1,2))*mass).sum(),row['full_empirical_loss']);close((full.mean(2)*mass[:,None]).sum(0),row['full_per_feature'])
    close((full.mean((1,2))*mass),row['full_region_contributions']);close(total,row['batch_losses'])
    close(total.mean(),row['batch_loss_summary']['mean']);close(total.std(ddof=1),row['batch_loss_summary']['SD'])
assert np.count_nonzero(l['full_tables'][1])==0
checks['all_original_objective_weights_full_tables_and_32_batch_reductions']=True
for row in psd['regions']:
    r=row['region'];si=sb==r;ri=rb==r
    close(wasserstein_distance(s['source_output_psd'][si],s['REAL_reference_psd'][ri]),row['SOURCE_W1'])
    close(wasserstein_distance(s['REAL_output_psd'][ri],s['REAL_reference_psd'][ri]),row['REAL_control_W1'])
close(wasserstein_distance(s['source_output_psd'],s['REAL_reference_psd']),psd['overall_SOURCE_W1'])
close(np.mean([v['SOURCE_W1']for v in psd['regions'][1:8]]),psd['common_Q1_Q7_mean_W1'])
close(spearmanr(s['source_original_psd'],s['source_output_psd']).statistic,psd['source_PSD_rank'])
close(spearmanr(s['source_original_q'],s['source_output_q']).statistic,psd['source_Q_rank'])
for v in psd['valleys']:
    ix=(sb==v['region'])&(s['source_h']==1);lo,hi=v['bounds'];assert int(ix.sum())==v['N']
    for name,arr in [('original',s['source_original_psd']),('output',s['source_output_psd'])]:
        close(np.mean((arr[ix]>=lo)&(arr[ix]<=hi)),v[name+'_valley_mass'])
        close(np.mean(np.abs(arr[ix]-lo)<=.004),v[name+'_low_edge_mass']);close(np.mean(np.abs(arr[ix]-hi)<=.004),v[name+'_high_edge_mass'])
for name,N in [('SOURCE',6593),('REAL_control',21047)]:
    v=projection[name];assert v['rows']==N and v['range_active_N']==int(s[name+'_range_active'].sum())
    assert v['rail_samples_total']==int(s[name+'_rail_samples'].sum())
    assert np.max(s[name+'_preADC_relative_Q_error_f32'])<=1e-5
    for key in ['delta_RMS_V','projection_RMS_V','preADC_relative_Q_error_f32','preADC_relative_Q_error_f64']:
        close(s[name+'_'+key].mean(),v[key]['mean']);close(s[name+'_'+key].max(),v[key]['max'])
checks['all_PSD_valley_edge_source_rank_and_projection_summaries']=True
streams={k:np.random.Generator(np.random.PCG64(seed))for k,seed in C['PCG64_seeds'].items()};chains={k:bytes(32)for k in streams}
initial=read(S/'INITIAL_RANDOM_STATES.json');final=read(S/'FINAL_RANDOM_STATES.json');receipt=read(S/'RANDOM_STREAM_RECEIPT.json')
assert {k:v.bit_generator.state for k,v in streams.items()}==initial
def consume(k,a):chains[k]=hashlib.sha256(chains[k]+np.ascontiguousarray(a).tobytes()).digest()
for key,N in [('full_SOURCE_epsilon',6593),('full_REAL_epsilon',21047)]:
    a=streams[key].standard_normal((N,480)).astype('<f4');consume(key,a);del a
for i,dim in enumerate([480,480,479,478]):
    a=streams['full_feature_directions'].standard_normal((dim,64));a/=np.linalg.norm(a,axis=0,keepdims=True);a=a.astype('<f4')
    assert np.array_equal(a,l['directions_'+str(i)]);consume('full_feature_directions',a)
for key,bins in [('batch_SOURCE_rows',sb),('batch_REAL_A_rows',rb),('batch_REAL_B_rows',rb),('batch_REAL_control_rows',rb)]:
    groups=[np.flatnonzero(bins==r)for r in regions]
    for batch in range(32):
        a=np.stack([g[streams[key].integers(0,len(g),32)]for g in groups]).astype('<u2')
        assert np.array_equal(a,l[key][batch]);consume(key,a)
for row in base['rows']:
    region=row['original_region'];ri=np.arange(21047)if region is None else np.flatnonzero(rb==region)
    for k in range(32):
        a=ri[streams['covariance_reference_rows'].integers(0,len(ri),row['SOURCE_N'])].astype('<u2')
        assert np.array_equal(a,b[row['scope']+'_reference_indices'][k]);consume('covariance_reference_rows',a)
assert {k:v.bit_generator.state for k,v in streams.items()}==final
assert {k:v.hex()for k,v in chains.items()}==receipt['chains']
assert all(chains[k].hex()==v for k,v in C['expected_full_stream_hashes'].items())
checks['all_eight_consumption_streams_and_actual_indices_reconstructed']=True
assert result['diagnostic_markers']=={'neural_boundary_increase':path['neural_boundary_increase'],'baseline_joint_residual':base['baseline_joint_residual']}
for fig in result['figures']:assert sha(S/fig['path'])==fig['sha256']and(S/fig['path']).stat().st_size==fig['bytes']
assert len(result['figures'])==8
class PMC(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t)for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
getmem=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo;getmem.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
mem=PMC();mem.cb=ctypes.sizeof(mem);assert getmem(handle,ctypes.byref(mem),mem.cb)and mem.PeakWorkingSetSize<=4*1024**3
out={'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'CPU_affinity':cpus,'pass':True,'checks':checks,'elapsed_s':time.monotonic()-START,'peak_RSS_bytes':mem.PeakWorkingSetSize,
 'new_model_forwards':0,'raw_TRAIN_waveforms_decoded':0,'DEV_EVAL_CAL_decoded':0,'training_updates':0,'figures_integrity_verified':8,
 'scope':'Saved artifact arithmetic, stream reconstruction, bound executed code and native receipts; not a new full-waveform replay.'}
(S/'FROZEN_ARTIFACT_REVIEW.json').write_bytes((json.dumps(out,indent=2,allow_nan=False)+'\n').encode());print(json.dumps(out,indent=2))
