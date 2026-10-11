"""Small artificial-only value/gradient checks; no actual science arrays/models."""
from __future__ import annotations
import sys, os, pathlib, json, hashlib, time, ctypes, importlib.util
sys.dont_write_bytecode = True
OUT = pathlib.Path(__file__).resolve().parent
GUARD = None
LIMITS = None
if __name__ == "__main__" and os.name == "nt":
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    assert k.SetProcessAffinityMask(k.GetCurrentProcess(), 4)
    path = OUT.parents[2] / "reviews/R65_PREPARATION_001/training_audit_adapter/local_resource_guard.py"
    spec = importlib.util.spec_from_file_location("r72_guard", path)
    GUARD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(GUARD)
    LIMITS = GUARD.apply(OUT)
if __name__ == "__main__":
    task_numba_cache = OUT / "runtime_cache" / "numba"
    task_numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(task_numba_cache)

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from full_waveform_loss import (FullWaveformLoss, UniformW1Cache, PrecisionOrderPlan,
     _PrecisionOrder, NUMBA_AVAILABLE, TERM_NAMES, D_TERMS, signed_psd, charge)
DTYPE = torch.float64

CONFIG = {
 "feature_scale_floor": {"V480":1/16383,"U480":.001,"DU479":.001,"D2U478":.001},
 "noise_scale_floor": [.1/16383,.01],
 "temporal_scale_floor": [.1/16383]+[.01]*30,
 "temporal_power_floor":1e-20, "PSD_loss_scale":.01,
 "valley_smoothing":.002,"valley_delta":.02,
 "loss_weights":{name:1.0 for name in TERM_NAMES}
}


def dense_w1(a, b):
    """Independent TOY NxM CDF overlap only; never used by production code."""
    n,m=len(a),len(b)
    a=a[:,None] if a.ndim==1 else a
    b=b[:,None] if b.ndim==1 else b
    start_a=torch.arange(n,device=a.device,dtype=DTYPE)/n
    stop_a=(torch.arange(n,device=a.device,dtype=DTYPE)+1)/n
    start_b=torch.arange(m,device=b.device,dtype=DTYPE)/m
    stop_b=(torch.arange(m,device=b.device,dtype=DTYPE)+1)/m
    overlap=(torch.minimum(stop_a[:,None],stop_b[None,:])-
             torch.maximum(start_a[:,None],start_b[None,:])).clamp_min(0)
    sa=torch.sort(a,dim=0,stable=True).values
    sb=torch.sort(b,dim=0,stable=True).values
    return (overlap[:,:,None]*(sa[:,None,:]-sb[None,:,:]).abs()).sum((0,1))


def dense_order(y,p,lo,hi):
    n=len(y);ds=p[:,None]-p[None,:];dy=y[:,None]-y[None,:]
    weight=(ds!=0).to(DTYPE)/(n*n)
    resolved=(lo[:,None]>hi[None,:])|(hi[:,None]<lo[None,:])
    return (weight*resolved*torch.relu(-torch.sign(ds)*dy)).sum()/weight.sum().clamp_min(1e-30)/.01


def make_waveforms(per_region):
    q=[.6,.95,1.5,2.2,3.2,4.6,6.4,8.6,11.5,16.]
    rows=[]
    t=np.arange(100)
    for region,amount in enumerate(q):
        for j in range(per_region):
            v=np.zeros(480,dtype=np.float64)
            v[:100]=.001*(1+.08*j)*np.sin(2*np.pi*(1+j)*t/100)
            p=.09+.027*region+.045*j
            v[100:150]=amount*(1-p)/50
            v[150:400]=amount*p/250
            v[400:]=.0002*np.sin(np.arange(80)*.15+j)
            rows.append(v)
    return np.stack(rows)


def fixture(device="cpu", reference_override=None):
    sc=torch.as_tensor(make_waveforms(2),device=device,dtype=DTYPE)
    ref=torch.as_tensor(make_waveforms(3),device=device,dtype=DTYPE)
    if reference_override is not None:ref=reference_override
    edges=torch.log(torch.tensor([.8,1.2,1.8,2.7,4.,5.7,7.7,10.3,14.],
                                 device=device,dtype=DTYPE)*1000)
    p=signed_psd(sc).detach()
    original_bins=torch.tensor(np.repeat(np.arange(-1,9),2),device=device,dtype=torch.int64)
    h=torch.ones(len(sc),device=device,dtype=DTYPE)
    h[6]=.5  # Retain mixed h rather than casting it to integer.
    bounds=torch.tensor([[.12,.19],[.15,.22]],device=device,dtype=DTYPE)
    obj=FullWaveformLoss(ref,edges,p,p-.011,p+.011,original_bins,h,bounds,CONFIG)
    rng=np.random.default_rng(72014)
    directions=[]
    for dim in (480,480,479,478):
        d=rng.standard_normal((dim,64));d/=np.linalg.norm(d,axis=0)
        directions.append(torch.as_tensor(d,device=device,dtype=DTYPE))
    d=rng.standard_normal((31,64));d/=np.linalg.norm(d,axis=0)
    td=torch.as_tensor(d,device=device,dtype=DTYPE)
    return obj,sc,ref,directions,td


def get_grad(value, leaves):
    result=torch.autograd.grad(value,leaves,allow_unused=True)
    return tuple(torch.zeros_like(x) if g is None else g for x,g in zip(leaves,result))


def run_checks(*, require_cuda=False, require_numba=False):
    start=time.perf_counter();checks=[];max_w1_value=max_w1_gradient=max_scipy=0.
    max_order_value=max_order_gradient=0.
    if require_cuda and not torch.cuda.is_available():raise RuntimeError("registered CUDA artificial check unavailable")
    if require_numba and not NUMBA_AVAILABLE:raise RuntimeError("registered compiled order check unavailable")
    devices=["cpu"]+(["cuda"]if torch.cuda.is_available()else[])
    compiled_first_seconds=None
    for device in devices:
        cache=UniformW1Cache(device,max_entries=2,max_cells=20)
        laws=[
          ([1.,2.,4.],[.5,1.5,2.5,3.5,5.]),
          ([2.],[1.,3.,4.,5.]),
          ([1.,2.,2.,4.],[1.,1.,3.,3.]),
          ([1.,1.,3.,3.],[1.,3.]),
          ([.5,2.5],[1.,1.,4.]),
          ([1.,2.,4.,5.],[3.])
        ]
        for raw_a,raw_b in laws:
            a=torch.tensor(np.column_stack((raw_a,np.square(raw_a))),device=device,dtype=DTYPE,requires_grad=True)
            b=torch.tensor(np.column_stack((raw_b,np.square(raw_b))),device=device,dtype=DTYPE,requires_grad=True)
            actual=cache.columns(a,b);expected=dense_w1(a,b)
            max_w1_value=max(max_w1_value,float((actual-expected).abs().max().detach()))
            ga=get_grad(actual.sum(),(a,b));ge=get_grad(expected.sum(),(a,b))
            for x,y in zip(ga,ge):
                max_w1_gradient=max(max_w1_gradient,float((x-y).abs().max()))
                torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            for j in range(2):
                scipy=wasserstein_distance(a.detach().cpu().numpy()[:,j],b.detach().cpu().numpy()[:,j])
                max_scipy=max(max_scipy,abs(float(actual[j].detach())-scipy))
                assert abs(float(actual[j].detach())-scipy)<1e-12
        a=torch.linspace(1.,3.,35,device=device,dtype=DTYPE)[:,None]
        b=torch.linspace(.5,4.,41,device=device,dtype=DTYPE)[:,None]
        cache.columns(a,b)
        assert len(cache._cache)<=2 and cache.cached_cells<=20
        checks.append({"device":device,"case":"uniform_W1_unequal_repeats_values_gradients_scipy_LRU","pass":True})

        order_cases=[
          ([.1,.1,.2,.3,.4,.4],[.095,.095,.195,.295,.395,.395],
           [.105,.105,.205,.305,.405,.405],[.35,.1,.3,.15,.45,.15]),
          ([.1,.15,.3,.3,.4],[.05,.09,.21,.26,.36],
           [.2,.18,.35,.34,.5],[.5,.4,.3,.1,.2]),
          ([.1,.2,.3],[0.,.15,.2],[.15,.3,.4],[.4,.25,.4]),
          ([.2,.2,.2],[.1,.12,.15],[.3,.35,.4],[.5,.1,.2]),
          ([.1,.2,.3],[.09,.19,.29],[.11,.21,.31],[.2,.2,.2])
        ]
        rng=np.random.default_rng(72015)
        rp=np.round(rng.uniform(-.2,.7,29),1)
        order_cases.append((rp,rp-.015,rp+.015,np.round(rng.uniform(-.1,.6,29),2)))
        for p,lo,hi,y0 in order_cases:
            plan=PrecisionOrderPlan(p,lo,hi)
            y=torch.tensor(y0,device=device,dtype=DTYPE,requires_grad=True)
            pt,lot,hit=[torch.tensor(v,device=device,dtype=DTYPE)for v in(p,lo,hi)]
            tick=time.perf_counter()
            actual=_PrecisionOrder.apply(y,plan,.01)
            if NUMBA_AVAILABLE and compiled_first_seconds is None:
                compiled_first_seconds=time.perf_counter()-tick
            expected=dense_order(y,pt,lot,hit)
            max_order_value=max(max_order_value,float((actual-expected).abs().detach()))
            ga,=get_grad(actual,(y,));ge,=get_grad(expected,(y,))
            max_order_gradient=max(max_order_gradient,float((ga-ge).abs().max()))
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(ga,ge,rtol=1e-12,atol=1e-12)
            pure_v,pure_g=plan.value_gradient(np.asarray(y0),.01,backend="python")
            np.testing.assert_allclose(pure_v,float(actual.detach()),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(pure_g,ga.detach().cpu().numpy(),rtol=1e-12,atol=1e-12)
            if NUMBA_AVAILABLE:
                compiled_v,compiled_g=plan.value_gradient(np.asarray(y0),.01,backend="numba")
                assert compiled_v==pure_v
                np.testing.assert_array_equal(compiled_g,pure_g)
        checks.append({"device":device,"case":"order_vs_old_NxN_full_denominator_mask_touching_ties_and_pure_compiled","pass":True})

        obj,sc,ref,directions,td=fixture(device)
        source=sc.clone();source[:,150:220]+=.0001
        real=ref.clone();real[:,220:300]-=.00003
        source_raw=source.clone();source_raw[:,100:150]+=.0004
        real_raw=real.clone();real_raw[:,100:150]-=.0002
        leaves=[source.detach().requires_grad_(),real.detach().requires_grad_(),
                source.float().detach().requires_grad_(),real.float().detach().requires_grad_(),
                source_raw.detach().requires_grad_(),real_raw.detach().requires_grad_()]
        total,terms,diagnostic=obj(*leaves,directions,td)
        direct_grad=get_grad(total,tuple(leaves))
        accum=[torch.zeros_like(v)for v in leaves];sum_value=0.
        for name,value,_ in obj.iter_terms(*leaves,directions,td):
            sum_value+=float(value.detach())
            for j,g in enumerate(get_grad(value,tuple(leaves))):accum[j]+=g
        assert abs(sum_value-float(total.detach()))<1e-10
        for a,b in zip(accum,direct_grad):torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-9)
        assert set(terms)==set(TERM_NAMES)
        assert all(v.dtype==DTYPE and v.ndim==0 for v in terms.values())
        assert diagnostic["SOURCE_bin_counts"]==[2]*10
        assert abs(sum(diagnostic["SOURCE_bin_mass"])-1)<1e-15

        # Same distribution of unprocessed REAL in a new row order still has
        # zero D terms, while paired identity remains sensitive to original IDs.
        frozen_args=(sc,ref,sc.float(),ref.float(),sc,ref,directions,td)
        context=obj._context(*frozen_args)
        for name in D_TERMS:
            if name in ("full_window","coordinate_RMS","temporal_joint"):
                mode="temporal"if name=="temporal_joint"else name
                dr=td if mode=="temporal"else directions
                zero=obj._conditional_features(ref,obj.reference_bins,obj._reference_counts,dr,mode)
            elif name=="PSD":zero=obj._domain_PSD(ref,obj.reference_bins)
            else:zero=obj._domain_noise(ref)
            assert float(zero.detach())==0.,name
        _,control_terms,_=obj(*frozen_args)
        assert float(control_terms["REAL_identity"].detach())==0.
        perm=torch.arange(len(ref)-1,-1,-1,device=device)
        shuffled=ref[perm]
        _,shuffled_terms,_=obj(sc,shuffled,sc.float(),shuffled.float(),sc,shuffled,directions,td)
        for name in D_TERMS:torch.testing.assert_close(shuffled_terms[name],control_terms[name],rtol=1e-12,atol=1e-12)
        assert float(shuffled_terms["REAL_identity"].detach())>0.
        manual=((shuffled-obj.reference)/obj._identity_scale).square().mean()
        torch.testing.assert_close(shuffled_terms["REAL_identity"],manual,rtol=0,atol=0)

        # Move exactly one original outside-support parent to current bin0;
        # all parents and raw support-outside weight remain present.
        migrated=sc.clone()
        migrated[0]*=1.5
        _,_,dm=obj(migrated,ref,migrated.float(),ref.float(),migrated,ref,directions,td)
        assert dm["SOURCE_bin_counts"]==[1,3]+[2]*8
        assert dm["SOURCE_bin_mass"][0]==1/len(sc) and dm["SOURCE_bin_mass"][-1]==2/len(sc)
        assert sum(dm["SOURCE_bin_counts"])==len(sc)
        assert obj.stats_serializable()["source_original_bin_counts"]==[2]*10

        empty=sc.clone();empty[2:4]*=1.6
        try:obj(empty,ref,empty.float(),ref.float(),empty,ref,directions,td)
        except ValueError as error:assert "PSD bin 0 empty"in str(error)
        else:raise AssertionError("empty mandatory output PSD bin accepted")
        no_outside=sc.clone();no_outside[:2]*=1.5
        _,_,outside=obj(no_outside,ref,no_outside.float(),ref.float(),no_outside,ref,directions,td)
        assert outside["SOURCE_bin_counts"][0]==0 and outside["SOURCE_bin_mass"][0]==0.
        ref_no_outside=ref[3:]
        obj2,sc2,ref2,d2,td2=fixture(device,reference_override=ref_no_outside)
        try:obj2(sc2,ref2,sc2.float(),ref2.float(),sc2,ref2,d2,td2)
        except ValueError as error:assert "empty REAL reference"in str(error)
        else:raise AssertionError("occupied support-outside with empty reference accepted")
        checks.append({"device":device,"case":"nine_term_direct_vs_iter_six_leaf_gradients_dynamic_bins_outside_mass_empty_fail_identity","pass":True})

        # Independent small public endpoint map checks full versus leaf-gradient
        # accumulation plus ALL-parent VJP. No optimizer/model fitting.
        pattern=torch.zeros_like(sc);pattern[:,150:220]=.00002
        rpattern=torch.zeros_like(ref);rpattern[:,220:300]=-.00001
        def endpoints(w):
            s=sc+w[0]*pattern
            r=ref+w[0]*rpattern
            return (s,r,(s+w[1]*pattern).float(),(r+w[1]*rpattern).float(),
                    s+2*w[1]*pattern,r+2*w[1]*rpattern)
        w=torch.tensor([.3,.2],device=device,dtype=DTYPE,requires_grad=True)
        outputs=endpoints(w)
        all_value,_,_=obj(*outputs,directions,td)
        direct_w,=get_grad(all_value,(w,))
        detached=[v.detach().requires_grad_()for v in outputs]
        leaf_grad=[torch.zeros_like(v,dtype=DTYPE)for v in detached]
        values=0.
        for name,value,_ in obj.iter_terms(*detached,directions,td):
            values+=float(value.detach())
            for j,g in enumerate(get_grad(value,tuple(detached))):leaf_grad[j]+=g.to(DTYPE)
        w2=w.detach().clone().requires_grad_()
        replay_outputs=endpoints(w2)
        torch.autograd.backward(replay_outputs,
                               [g.to(v.dtype)for g,v in zip(leaf_grad,replay_outputs)])
        torch.testing.assert_close(w2.grad,direct_w,rtol=1e-10,atol=1e-10)
        assert abs(values-float(all_value.detach()))<1e-10
        json.dumps(obj.stats_serializable(),allow_nan=False)
        checks.append({"device":device,"case":"complete_leaf_plus_all_parent_VJP_matches_direct_value_gradient_stats_serializable","pass":True})

    # Exact scalar algorithm artificial-size timing only, NOT a model forecast.
    n=6593
    p=np.linspace(.05,.55,n);low=p-1e-7;high=p+1e-7
    y=.3+.1*np.sin(np.arange(n)*.017)
    plan=PrecisionOrderPlan(p,low,high)
    tick=time.perf_counter();v,g=plan.value_gradient(y,.01);native_order_seconds=time.perf_counter()-tick
    assert np.isfinite(v)and np.isfinite(g).all()
    tick=time.perf_counter();pv,pg=plan.value_gradient(y,.01,backend="python");pure_seconds=time.perf_counter()-tick
    assert pv==v;np.testing.assert_array_equal(pg,g)
    checks.append({"case":"artificial6593_full_order_gradient_no_pair_matrix","pass":True})
    return {"status":"PASS_ARTIFICIAL_ONLY","checks":checks,"CPU":"passed",
      "CUDA":"passed"if torch.cuda.is_available()else"unknown_no_local_CUDA_available",
      "numba_available":NUMBA_AVAILABLE,"order_backend":plan.backend,
      "compiled_first_call_seconds_included_in_setup":compiled_first_seconds,
      "artificial6593_order_backend_seconds":native_order_seconds,
      "artificial6593_order_python_seconds":pure_seconds,
      "order_timing_scope":"artificial scalar order only; no full-training forecast",
      "max_W1_dense_value_abs_error":max_w1_value,
      "max_W1_dense_gradient_abs_error":max_w1_gradient,
      "max_W1_scipy_abs_error":max_scipy,
      "max_order_old_dense_value_abs_error":max_order_value,
      "max_order_old_dense_gradient_abs_error":max_order_gradient,
      "elapsed_checks_seconds_including_first_JIT":time.perf_counter()-start,
      "scientific_array_reads":0,"actual_scientific_model_fit_or_inference":0,
      "torch_version":torch.__version__,"numpy_version":np.__version__}


if __name__ == "__main__":
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result=run_checks()
    result["resource_limits"]=LIMITS
    result["resource_memory"]=GUARD.memory()if GUARD else None
    result["source_SHA256"]={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                            for name in("full_waveform_loss.py","check_full_waveform_loss.py")}
    (OUT/"ARTIFICIAL_CHECK_RECEIPT.json").write_text(
       json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
