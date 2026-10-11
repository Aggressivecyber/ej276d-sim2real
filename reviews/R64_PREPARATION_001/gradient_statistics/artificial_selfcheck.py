"""Artificial NumPy-only checks, including virtual full parameter dimension.

No scientific arrays, weights, forward pass, gradient, RNG or optimizer. The
bootstrap index table below is an explicitly deterministic TEST fixture; the
scientific worker supplies its independently registered table to the module.
"""
from pathlib import Path
import os
import sys
import json
import time
import ctypes
import threading
import hashlib
import traceback

ROOT=Path(__file__).resolve().parent
assert ROOT.drive.upper() in ('E:','D:')
assert sys.dont_write_bytecode,'Use -B'
START=time.monotonic()
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
(ROOT/'cache/tmp').mkdir(parents=True,exist_ok=True)
os.environ.update(TEMP=str(ROOT/'cache/tmp'),TMP=str(ROOT/'cache/tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),PYTHONDONTWRITEBYTECODE='1')
TOL=json.loads((ROOT/'FROZEN_TOLERANCES.json').read_text('utf8'))
TOL_SHA=hashlib.sha256((ROOT/'FROZEN_TOLERANCES.json').read_bytes()).hexdigest()
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
handle=kernel.GetCurrentProcess();allowed=ctypes.c_size_t();system=ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle,ctypes.byref(allowed),ctypes.byref(system))
cpu=next(i for i in range(ctypes.sizeof(ctypes.c_size_t)*8) if allowed.value&(1<<i))
assert kernel.SetProcessAffinityMask(handle,1<<cpu)

class PMC(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t)for k in
        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
         'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]

psapi=ctypes.WinDLL('psapi',use_last_error=True)
psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
peak=0;stop=threading.Event()

def resource():
    global peak
    m=PMC();m.cb=ctypes.sizeof(m)
    if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(m),m.cb):os._exit(138)
    peak=max(peak,m.PeakWorkingSetSize)
    if peak>TOL['RSS_limit_bytes']:os._exit(137)
    return peak

def monitor():
    while not stop.wait(.1):resource()

threading.Thread(target=monitor,daemon=True).start()

def stage(name):
    print(json.dumps({'stage':name,'elapsed_seconds':round(time.monotonic()-START,3),'peak_RSS_bytes':resource()}),flush=True)

try:
    import numpy as np
    import gradient_statistics as gs
    assert gs.DECOMPOSITION_ATOL==TOL['finite_sample_decomposition']['atol']
    assert gs.DECOMPOSITION_RTOL==TOL['finite_sample_decomposition']['rtol']
    tests={}

    def close(name,a,b,rule='scalar_oracle'):
        aa,bb=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
        assert aa.shape==bb.shape,(name,aa.shape,bb.shape)
        delta=np.abs(aa-bb);bound=TOL[rule]['atol']+TOL[rule]['rtol']*np.abs(bb)
        result={'passed':bool(np.all(delta<=bound)),'max_abs_error':float(delta.max(initial=0.)),
                'reference_absmax':float(np.abs(bb).max(initial=0.)),'atol':TOL[rule]['atol'],'rtol':TOL[rule]['rtol']}
        tests[name]=result
        assert result['passed'],(name,result)

    def reject(name,call,exception=(ValueError,TypeError,RuntimeError,ArithmeticError)):
        try:call()
        except exception as exc:
            tests[name]={'passed':True,'error_type':type(exc).__name__,'message':str(exc)}
        else:raise AssertionError(name+' was not rejected')

    p=257
    x=np.arange(p,dtype=np.float64)[None,:]
    term=np.arange(5,dtype=np.float64)[:,None]
    ref=.1*(term+1)+np.sin((x+1)*(.019+.003*term))
    macro=np.arange(32,dtype=np.float64)[:,None,None]
    perturb=(.017*(macro-13)+.04*np.cos(macro*.71+term[None,:,:]))*np.cos(x[None,:,:]*.043+.3*term[None,:,:])
    A=ref[None,:,:]+perturb
    B=ref[None,:,:]+.25*perturb
    ii=np.arange(1000,dtype=np.int64)[:,None];jj=np.arange(32,dtype=np.int64)[None,:]
    indices=((ii*ii+3*ii*jj+7*jj*jj+11*(ii//7)+jj//3)%32).astype(np.int64)
    assert any(len(np.unique(row))<32 for row in indices)
    fixture_sha=hashlib.sha256()
    for v in (ref,A,B,indices):fixture_sha.update(str(v.shape).encode());fixture_sha.update(v.dtype.str.encode());fixture_sha.update(v.tobytes())
    result=gs.summarize_gradients(ref,A,B,indices,feature_chunk=41)
    assert result.summary['all_finite_sample_decompositions_passed']
    rr=np.concatenate([ref,ref.sum(axis=0,keepdims=True)],axis=0)
    aa=np.concatenate([A,A.sum(axis=1,keepdims=True)],axis=1)
    bb=np.concatenate([B,B.sum(axis=1,keepdims=True)],axis=1)
    direct_errors={};direct_dispersion={}
    for label,array,mean in (('A',aa,result.mean_A),('B',bb,result.mean_B)):
        direct_mean=array.mean(axis=0)
        close('direct_mean_'+label,mean,direct_mean,'mean_vector_oracle')
        errors=np.sum((array-rr[None,:,:])**2,axis=2)
        norm_sq=np.sum(array**2,axis=2)
        dots=np.sum(array*rr[None,:,:],axis=2)
        dispersion=np.mean(np.sum((array-direct_mean[None,:,:])**2,axis=2),axis=0)
        direct_errors[label]=errors;direct_dispersion[label]=dispersion
        for j,name in enumerate(gs.ALL_NAMES):
            row=result.summary['arms'][label][name]
            close('MSE_direct_'+label+'_'+name,row['mean_squared_l2_error'],errors[:,j].mean())
            close('dispersion_direct_'+label+'_'+name,row['empirical_dispersion_mean_squared_l2'],dispersion[j])
            close('mean_discrepancy_'+label+'_'+name,row['mean_gradient']['squared_l2_error'],np.sum((direct_mean[j]-rr[j])**2))
            close('norm_dot_macro_'+label+'_'+name,
                  np.array([[z[label][name]['squared_norm'],z[label][name]['dot_reference']] for z in result.summary['macro_records']]),
                  np.stack([norm_sq[:,j],dots[:,j]],axis=1))
            expected_cos=dots[:,j]/np.sqrt(norm_sq[:,j])/np.linalg.norm(rr[j])
            close('cos_macro_'+label+'_'+name,[z[label][name]['cosine_reference'] for z in result.summary['macro_records']],expected_cos)
    changes=direct_errors['B']-direct_errors['A']
    boot=np.array([[sum(float(changes[k,j]) for k in row)/32 for j in range(6)] for row in indices])
    close('paired_shared_indices_all_bootstrap_replicates',result.bootstrap_error_changes,boot,'bootstrap_oracle')
    for j,name in enumerate(gs.ALL_NAMES):
        close('paired_mean_'+name,result.summary['paired'][name]['mean_B_minus_A_squared_l2_error'],changes[:,j].mean())
        close('paired_linear_interval_'+name,result.summary['paired'][name]['bootstrap_percentile_95_interval'],np.quantile(boot[:,j],[.025,.975],method='linear'),'bootstrap_oracle')
        assert result.summary['paired'][name]['mean_B_minus_A_squared_l2_error']<0
    tests['B_closer_than_A_for_every_component_and_total']={'passed':True}
    stage('direct_arithmetic_decomposition_and_shared_paired_bootstrap_passed')

    streaming=gs.GradientAccumulator(ref,indices,feature_chunk=41,coordinate_label='endpoint_artificial')
    compact=[];sum_A=np.zeros_like(ref);sum_B=np.zeros_like(ref)
    for macro_id in range(32):
        streaming.update(A[macro_id],B[macro_id])
        compact.append(gs.macro_gradient_record(ref,A[macro_id],B[macro_id],feature_chunk=41))
        sum_A+=A[macro_id];sum_B+=B[macro_id]
    streamed=streaming.finalize()
    sufficient=gs.summarize_sufficient_statistics(ref,sum_A,sum_B,compact,indices,coordinate_label='endpoint_artificial',feature_chunk=41)
    close('stream_vs_array_mean_A',streamed.mean_A,result.mean_A,'same_api_same_order')
    close('stream_vs_array_mean_B',streamed.mean_B,result.mean_B,'same_api_same_order')
    close('stream_vs_array_bootstrap',streamed.bootstrap_error_changes,result.bootstrap_error_changes,'same_api_same_order')
    close('sufficient_vs_array_mean_A',sufficient.mean_A,result.mean_A,'mean_vector_oracle')
    close('sufficient_vs_array_mean_B',sufficient.mean_B,result.mean_B,'mean_vector_oracle')
    close('sufficient_vs_array_bootstrap',sufficient.bootstrap_error_changes,result.bootstrap_error_changes,'same_api_same_order')
    for label in ('A','B'):
        for name in gs.ALL_NAMES:
            close('sufficient_dispersion_'+label+'_'+name,sufficient.summary['arms'][label][name]['empirical_dispersion_mean_squared_l2'],
                  result.summary['arms'][label][name]['empirical_dispersion_mean_squared_l2'])
    tests['sufficient_stats_decomposition_all_passed']={'passed':sufficient.summary['all_finite_sample_decompositions_passed']}
    assert not sufficient.mean_A.flags.writeable and not result.bootstrap_error_changes.flags.writeable
    reject('finalized_accumulator_rejects_update',lambda:streaming.update(A[0],B[0]))
    stage('streaming_and_online_sum_plus_scalar_record_APIs_passed')

    identical=gs.summarize_gradients(ref,A,A,indices,feature_chunk=41)
    for name in gs.ALL_NAMES:
        assert identical.summary['paired'][name]['mean_B_minus_A_squared_l2_error']==0
        assert identical.summary['paired'][name]['bootstrap_percentile_95_interval']==[0.,0.]
        assert all(z['paired'][name]['squared_l2_A_minus_B']==0 for z in identical.summary['macro_records'])
    tests['A_equals_B_exact_zero_paired_changes_and_CI']={'passed':True}
    zero=np.zeros_like(ref);allzero=np.zeros_like(A)
    zero_ref=gs.summarize_gradients(zero,A,B,indices,feature_chunk=41)
    zero_g=gs.summarize_gradients(ref,allzero,allzero,indices,feature_chunk=41)
    zero_both=gs.summarize_gradients(zero,allzero,allzero,indices,feature_chunk=41)
    for name in gs.ALL_NAMES:
        for label in ('A','B'):
            assert zero_ref.summary['arms'][label][name]['relative_mean_squared_l2_error'] is None
            assert zero_ref.summary['arms'][label][name]['mean_gradient']['cosine_reference'] is None
            assert zero_ref.summary['arms'][label][name]['mean_squared_l2_error']>0
            assert zero_g.summary['arms'][label][name]['mean_norm']==0
            assert zero_g.summary['arms'][label][name]['mean_cosine_reference'] is None
            close('zero_gradient_absolute_error_'+label+'_'+name,zero_g.summary['arms'][label][name]['mean_squared_l2_error'],np.sum(rr[gs.ALL_NAMES.index(name)]**2))
            assert zero_both.summary['arms'][label][name]['mean_squared_l2_error']==0
            assert zero_both.summary['arms'][label][name]['relative_mean_squared_l2_error'] is None
        assert zero_both.summary['paired'][name]['bootstrap_percentile_95_interval']==[0.,0.]
    assert len(zero_ref.summary['macro_records'])==32
    tests['zero_reference_and_zero_gradient_nulls_without_row_dropping']={'passed':True}
    mixed=A.copy();mixed[0]=0
    mixed_result=gs.summarize_gradients(ref,mixed,B,indices,feature_chunk=41)
    assert mixed_result.summary['arms']['A']['full_window']['mean_cosine_reference'] is None
    assert mixed_result.summary['arms']['A']['full_window']['defined_cosine_count']==31
    tests['mixed_undefined_cosine_keeps_macro_and_marks_mean_null']={'passed':True}
    # Total vector sum must precede squared norms: equal and opposite component
    # perturbations cancel completely, despite nonzero component errors.
    cancel=np.zeros_like(A);cancel[:,0,:]=.75;cancel[:,1,:]=-.75
    cancelling=gs.summarize_gradients(zero,cancel,allzero,indices,feature_chunk=41)
    assert cancelling.summary['arms']['A']['total_D']['mean_squared_l2_error']==0
    assert cancelling.summary['arms']['A']['full_window']['mean_squared_l2_error']>0
    tests['total_D_vector_sum_before_squared_norm']={'passed':True}
    stage('equal_arms_zero_denominators_and_vector_sum_total_passed')

    reject('wrong_ref_shape_rejected',lambda:gs.GradientAccumulator(ref[:4],indices))
    reject('wrong_A_macro_count_rejected',lambda:gs.summarize_gradients(ref,A[:31],B,indices))
    reject('wrong_B_parameter_dimension_rejected',lambda:gs.summarize_gradients(ref,A,B[:,:,:-1],indices))
    reject('wrong_bootstrap_shape_rejected',lambda:gs.GradientAccumulator(ref,indices[:999]))
    reject('noninteger_bootstrap_rejected',lambda:gs.GradientAccumulator(ref,indices.astype(np.float64)))
    invalid=indices.copy();invalid[0,0]=32
    reject('out_of_range_bootstrap_rejected',lambda:gs.GradientAccumulator(ref,invalid))
    incomplete=gs.GradientAccumulator(ref,indices)
    reject('incomplete_32_macro_run_rejected',incomplete.finalize)
    reject('wrong_macro_shape_before_mutation',lambda:incomplete.update(A[0,:,:-1],B[0]))
    assert incomplete.count==0
    nonfinite=A[0].copy();nonfinite[0,-1]=np.nan
    reject('nonfinite_before_mutation',lambda:incomplete.update(nonfinite,B[0]))
    assert incomplete.count==0
    wrong_records=[dict(z) for z in compact];wrong_records[0]=dict(wrong_records[0],reported_order=list(reversed(gs.ALL_NAMES)))
    reject('wrong_scalar_record_term_order_rejected',lambda:gs.summarize_sufficient_statistics(ref,sum_A,sum_B,wrong_records,indices))
    wrong_records=[dict(z) for z in compact];wrong_records[0]=dict(wrong_records[0],reference_squared_norm=[123456.]*6)
    reject('wrong_scalar_record_reference_binding_rejected',lambda:gs.summarize_sufficient_statistics(ref,sum_A,sum_B,wrong_records,indices))
    tests['module_contains_no_random_number_generation']={'passed':not any(s in (ROOT/'gradient_statistics.py').read_text('utf8') for s in ('np.random','random.'))}
    assert tests['module_contains_no_random_number_generation']['passed']
    stage('shape_reference_and_index_negative_controls_passed')

    # Artificial full parameter dimension with broadcast INPUT VIEWS only.
    # The accumulator still materializes its real [6,P] reference/means and
    # processes all 32 macros, providing a native memory/runtime receipt.
    big_p=1369889
    big_ref=np.broadcast_to(np.zeros((5,1),dtype=np.float64),(5,big_p))
    big_A=np.broadcast_to(np.ones((32,5,1),dtype=np.float64),(32,5,big_p))
    big_B=np.broadcast_to(np.full((32,5,1),.5,dtype=np.float64),(32,5,big_p))
    probe_start=time.monotonic()
    big=gs.summarize_gradients(big_ref,big_A,big_B,indices,coordinate_label='full_parameter_dimension_artificial')
    close('full_dimension_A_component_MSE',big.summary['arms']['A']['PSD']['mean_squared_l2_error'],float(big_p))
    close('full_dimension_total_vector_MSE',big.summary['arms']['A']['total_D']['mean_squared_l2_error'],25.*big_p)
    assert big.summary['arms']['A']['total_D']['mean_norm']>5
    assert big.summary['arms']['A']['total_D']['empirical_dispersion_mean_squared_l2']==0
    assert big.mean_A.shape==(6,big_p)
    probe={'dimension':big_p,'macro_pairs':32,'broadcast_inputs_are_artificial_views':True,
           'input_logical_A_B_bytes':int(big_A.nbytes+big_B.nbytes),
           'module_persistent_numeric_bytes_formula':144*big_p,'elapsed_seconds':time.monotonic()-probe_start,
           'native_peak_RSS_bytes_after_probe':resource(),'all_decompositions_passed':big.summary['all_finite_sample_decompositions_passed']}
    tests['full_dimension_norm_is_raw_not_clipped']={'passed':True,'total_D_norm':big.summary['arms']['A']['total_D']['mean_norm']}
    stage('full_1369889_parameter_dimension_32_macro_memory_probe_passed')

    # ROOT is reviews/R64_PREPARATION_001/gradient_statistics; parents[2] is project.
    protocol=ROOT.parents[2]/'tasks/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001.md'
    sources=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sorted(ROOT.glob('*.py'))]
    receipt={
        'status':'ARTIFICIAL_NUMPY_GRADIENT_STATISTICS_PASSED_NOT_SCIENTIFIC_RESULTS',
        'PID':os.getpid(),'python':sys.version,'numpy':np.__version__,'CPU_affinity':[cpu],
        'thread_environment':{k:os.environ[k]for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')},
        'RSS_limit_bytes':TOL['RSS_limit_bytes'],'peak_RSS_bytes':resource(),'elapsed_seconds':time.monotonic()-START,
        'tolerance_sha256_before_run':TOL_SHA,'fixture_arrays_sha256':fixture_sha.hexdigest(),
        'artificial_bootstrap_indices_sha256':result.summary['bootstrap_indices_sha256'],
        'bootstrap_indices_generated_by_module':False,'bootstrap_test_indices_are_deterministic_artificial_formula':True,
        'tests':tests,'test_count':len(tests),'all_tests_passed':all(t['passed']for t in tests.values()),
        'full_parameter_dimension_resource_probe':probe,'source_bindings':sources,
        'protocol_source':{'path':str(protocol),'sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'section':'Reported gradient statistics'},
        'scientific_array_reads':0,'scientific_weight_reads':0,'scientific_forward_runs':0,'scientific_gradient_runs':0,
        'optimizer_updates':0,'new_random_draws':0,'root_metadata_changes':0,'git_operations':0,
        'limitations':['Only artificial NumPy vectors were used. This does not report a scientific R64 outcome.','The full-size memory probe materializes real accumulator state but uses broadcast artificial input views; storage of caller-owned real parameter arrays remains external.','The supplied scientific bootstrap table must be the registered macro-level paired table, reused without redrawing.','Online-sum sufficient-statistic dispersion uses second-moment subtraction; arithmetic cancellation is exposed and decomposition checked, with Welford streaming available.'],
    }
    assert receipt['all_tests_passed']
    (ROOT/'ARTIFICIAL_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (ROOT/'EXAMPLE_ARTIFICIAL_SUMMARY.json').write_text(json.dumps(result.summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:receipt[k]for k in ('status','test_count','peak_RSS_bytes','elapsed_seconds')}),flush=True)
except BaseException as exc:
    failure={'status':'ARTIFICIAL_FAILED','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
             'elapsed_seconds':time.monotonic()-START,'peak_RSS_bytes':resource(),'tolerance_sha256_before_run':TOL_SHA}
    (ROOT/'ARTIFICIAL_FAILURE.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf8')
    print(json.dumps(failure),flush=True)
    raise
finally:stop.set()
