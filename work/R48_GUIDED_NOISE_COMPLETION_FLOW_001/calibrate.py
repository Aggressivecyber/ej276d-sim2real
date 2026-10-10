from pathlib import Path
from datetime import datetime,timezone
import os,time,json,hashlib
START=time.monotonic();R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'))
import numpy as np,torch
from noise_core import q_np,variance100,kernel,bins_for,summary
from source_flow_core import formula_check
torch.set_num_threads(12);torch.set_num_interop_threads(12)
def write(n,v):(R/n).write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
counter={'noise_calibration_attempts':1,'response_training_runs':0,'response_updates':0}
try:
    with np.load(C['TRAIN_path'],allow_pickle=False) as z:source=z['source'].copy();real=z['target'].copy();train_keys=list(z.files)
    with np.load(C['constants_path'],allow_pickle=False) as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
    with np.load(C['original_stream'],allow_pickle=False) as z:
        si=z['source'].astype('i8');mi=z['matched_target'].astype('i8');oldbin=z['bin'].astype('i8');ordinal=z['original_row_ordinal'].copy()
    assert si.shape==mi.shape==oldbin.shape==ordinal.shape==(94400,)
    assert si.min()>=0 and si.max()<len(source) and mi.min()>=0 and mi.max()<len(real)
    assert np.isfinite(mu).all() and np.isfinite(scale).all() and (scale>0).all() and np.isfinite(edges).all()
    assert np.all(np.diff(edges)>0) and np.max(np.abs(q_np(mu)-1))<1e-10
    us,si_inverse=np.unique(si,return_inverse=True);unique,mi_inverse=np.unique(mi,return_inverse=True)
    sb,sraw,sq=bins_for(source[us],edges);rb,rraw,rq=bins_for(real[unique],edges)
    e=real[unique,:100].astype('f8');e-=e.mean(1,keepdims=True)
    vi=(e*e).sum(1)/99.;tau2=float(np.quantile(vi,.05,method='linear'))
    if not np.isfinite(tau2) or tau2<=0:raise FloatingPointError('tau2 invalid; fixed recipe stops')
    p100=(np.abs(np.fft.rfft(e,axis=1,norm='ortho'))**2).mean(0);p100[0]=0.
    p480=np.interp(np.arange(241)/480.,np.arange(51)/100.,p100);p480[0]=0.
    assert np.isfinite(p480).all() and (p480>=0).all()
    basis=np.eye(480,dtype='f8');k=kernel(basis,p480);kappa=[];normalizations=[]
    for b in range(8):
        kp=k-mu[b]*q_np(k)[:,None];kap=float(variance100(kp).sum())
        if not np.isfinite(kap) or kap<=0:raise FloatingPointError(('kappa2 invalid',b,kap))
        kappa.append(kap);normalizations.append({'bin':b,'unit_expected_sample_variance':float(variance100(kp/np.sqrt(kap)).sum()),
            'projected_q_basis_maxabs':float(np.abs(q_np(kp)).max())})
    kappa=np.asarray(kappa,dtype='f8')
    np.savez(R/'NOISE_CALIBRATION.npz',tau2=np.asarray(tau2),p100=p100,p480=p480,kappa2=kappa)
    check=formula_check();check['unit_kernel_variance_checks']=normalizations
    assert all(abs(r['unit_expected_sample_variance']-1)<1e-10 for r in normalizations)
    write('FORMULA_CHECK.json',check)
    payload={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
      'calibration_unique_matched_REAL':int(len(unique)),'calibration_unique_index_SHA':hashlib.sha256(unique.astype('<i8').tobytes()).hexdigest(),
      'TRAIN_loaded_source_REAL':[len(source),len(real)],'TRAIN_container_keys':train_keys,
      'source_actual_consumed_unique':int(len(np.unique(si))),'reference_stream_rows':94400,'stream_ordinal_SHA':hashlib.sha256(ordinal.tobytes()).hexdigest(),
      'calibration_only_unique_matched_target':True,'tau2_V2':tau2,'tau_RMS_mV':1000*np.sqrt(tau2),'quantile':.05,'method':'linear',
      'REAL_variance_V2':summary(vi),'SOURCE_consumed_variance_V2':summary(variance100(source[np.unique(si)])),
      'kappa2':kappa.tolist(),'mu_unit_charge_residual_max':float(np.max(np.abs(q_np(mu)-1))),
      'actual_conditioning_bin_checks':{'SOURCE_mismatch_with_original_bin':int(np.count_nonzero(sb[si_inverse]!=oldbin)),
           'matched_REAL_mismatch_with_original_bin':int(np.count_nonzero(rb[mi_inverse]!=oldbin))},
      'finite_positive_charge_gate_only_actual_unique_consumed':True,
      'REAL_unique_completion_trigger_count':int(np.count_nonzero(vi<tau2)),
      'periodogram_norm':'ortho complex magnitude squared; no one-sided doubling; DC0',
      'floor_scan_or_spectrum_floor':False,'projected_covariance_shape_not_equal_to_interpolated_spectrum':True,
      'counter':counter,'new_DEV_or_CAL_or_EVAL':0}
    write('NOISE_CALIBRATION.json',payload);print(json.dumps(payload),flush=True)
except BaseException as e:
    write('CALIBRATION_FAILURE.json',{'PID':os.getpid(),'error':repr(e),'counter':counter,'elapsed_s':time.monotonic()-START});raise
