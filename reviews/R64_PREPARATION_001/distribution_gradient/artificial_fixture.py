"""One artificial 480-sample/ten-region fixture. No scientific inputs or weights.

Run only with -B. All logs/caches/results stay beside this script on E:.
The toy network is nonzero; endpoint/Box/ADC/features are byte-bound R63 code.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys
import json
import time
import hashlib
import math
import ctypes
import threading
import traceback

ROOT = Path(__file__).resolve().parent
assert ROOT.drive.upper() in ('E:', 'D:')
assert sys.dont_write_bytecode, 'Launch with -B'
START = time.monotonic()
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
for name in ('tmp', 'torch', 'matplotlib', 'xdg'):
    (ROOT / 'cache' / name).mkdir(parents=True, exist_ok=True)
os.environ.update(TEMP=str(ROOT/'cache/tmp'), TMP=str(ROOT/'cache/tmp'),
                  TORCH_HOME=str(ROOT/'cache/torch'), MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
                  XDG_CACHE_HOME=str(ROOT/'cache/xdg'), PYTHONDONTWRITEBYTECODE='1')
TOL = json.loads((ROOT/'FROZEN_TOLERANCES.json').read_text('utf8'))
TOL_SHA = hashlib.sha256((ROOT/'FROZEN_TOLERANCES.json').read_bytes()).hexdigest()
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
handle = kernel.GetCurrentProcess()
allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
assert kernel.GetProcessAffinityMask(handle, ctypes.byref(allowed), ctypes.byref(system))
cpu = next(i for i in range(ctypes.sizeof(ctypes.c_size_t)*8) if allowed.value & (1<<i))
assert kernel.SetProcessAffinityMask(handle, 1<<cpu)


class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [(k, ctypes.c_size_t) for k in
        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
         'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]


psapi = ctypes.WinDLL('psapi', use_last_error=True)
psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
stop = threading.Event()
peak = 0


def resources():
    global peak
    m = PMC(); m.cb = ctypes.sizeof(m)
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(m), m.cb):
        os._exit(138)
    peak = max(peak, m.PeakWorkingSetSize)
    if peak > TOL['RSS_limit_bytes']:
        os._exit(137)
    return peak


def monitor():
    while not stop.wait(.1):
        resources()


threading.Thread(target=monitor, daemon=True).start()


def stage(name):
    print(json.dumps({'stage': name, 'elapsed_s': round(time.monotonic()-START, 3), 'peak_RSS_bytes': resources()}), flush=True)


try:
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    import distribution_gradient as dg
    from original_bindings import R63
    from weighted_empirical_w1 import FrozenMass

    TESTS = {}

    def compare(name, actual, expected, rule):
        a = torch.as_tensor(actual, dtype=torch.float64).detach()
        b = torch.as_tensor(expected, dtype=torch.float64).detach()
        assert a.shape == b.shape, (name, a.shape, b.shape)
        atol, rtol = TOL[rule]['atol'], TOL[rule]['rtol']
        delta = (a-b).abs()
        passed = bool((delta <= atol + rtol*b.abs()).all())
        result = {'max_abs_error': float(delta.max()), 'reference_absmax': float(b.abs().max()),
                  'atol': atol, 'rtol': rtol, 'passed': passed}
        TESTS[name] = result
        assert passed, (name, result)
        return result

    source_counts = np.array([3,4,5,2,6,3,4,5,2,4], dtype=np.int64)
    real_counts = np.array([2,5,3,6,4,2,5,3,6,4], dtype=np.int64)
    source_regions = np.repeat(dg.CATEGORIES, source_counts)
    real_regions = np.repeat(dg.CATEGORIES, real_counts)
    t = np.arange(480, dtype=np.float64)

    def waveform(region, parent, real=False):
        if region == -1:
            code = np.full(480, 8192, dtype=np.float64)
            code[100:150], code[150:400] = 10100, 8750
            return -.5 + code/16383
        j = parent if parent < 2 else parent % 2
        base = .0008*np.sin(t*(.11+.013*region)+j) + .00035*np.cos(t*.19+region)
        pulse = np.zeros(480)
        amp = .11+.012*region+.004*j + (.005 if real else 0.)
        tau = 27.+2.5*region+1.5*j + (4. if real else 0.)
        pulse[100:400] = amp*np.exp(-np.arange(300)/tau)
        return (base+pulse).astype(np.float32).astype(np.float64)

    source_np = np.stack([waveform(int(r), j) for r, n in zip(dg.CATEGORIES, source_counts) for j in range(n)])
    real_np = np.stack([waveform(int(r), j, True) for r, n in zip(dg.CATEGORIES, real_counts) for j in range(n)])
    source = torch.from_numpy(source_np)
    real = torch.from_numpy(real_np)
    q = R63.core.q_torch(source)
    bins = torch.from_numpy(np.clip(source_regions, 0, 7))
    condition = torch.from_numpy(source_regions.astype(np.float64))
    mu = torch.zeros((8,480), dtype=torch.float64); mu[:,100:400] = 1/300
    scale = torch.full((8,480), .012, dtype=torch.float64)
    noise = .0004*torch.sin(torch.arange(len(source), dtype=torch.float64)[:,None]+torch.arange(480)[None,:]*.131)
    noise = R63.core.project(noise, mu[bins])
    noise[condition < 0] = 0

    class ToyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.coefficients = torch.nn.Parameter(torch.tensor([.08,.035,.40,.018], dtype=torch.float32))
            basis = torch.sin(torch.arange(480, dtype=torch.float32)*.041)
            basis[450], basis[451] = 28., -24.
            self.register_buffer('basis', basis)

        def forward(self, noisy, b, guide, h):
            y, g, hh = noisy.float(), guide.float(), h.float()[:,None]
            c = self.coefficients
            return (hh >= 0).float() * (c[0]*y+c[1]*g+c[2]*self.basis[None,:]+c[3]*(b.float()[:,None]+hh))

    net = ToyNetwork().train()
    # These exact slices are used by BOTH graph-retaining and no-grad/replay arms.
    schedule = tuple((k, min(k+7, len(source))) for k in range(0, len(source), 7))

    def endpoint_chunk(a, b):
        return R63.core.endpoint(net, source[a:b], noise[a:b], q[a:b], bins[a:b], condition[a:b], mu, scale)

    def endpoint_all():
        pairs = [endpoint_chunk(a,b) for a,b in schedule]
        return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])

    cfg = json.loads((ROOT/'r63_original/config.json').read_text('utf8'))
    stats = {'source_counts':source_counts, 'REAL_counts':real_counts,
             'noise_scale': np.array([.0012,.2]), 'valley_bounds':np.array([[.2,.7],[.2,.7]]),
             'source_h_one_fraction':np.ones(2), 'source_raw_valley_fraction':np.zeros(2)}
    for name, dim, value in zip(dg.NAMES, (480,480,479,478), (.15,.5,.25,.25)):
        stats[name+'_center'] = np.zeros((10,dim))
        stats[name+'_scale'] = value*(1+.08*np.arange(10)[:,None])*(1+.05*np.sin(np.arange(dim)[None,:]*.03))
    stats['temporal_center'] = np.zeros((10,31))
    stats['temporal_scale'] = np.tile(np.r_[.005,np.full(30,.35)], (10,1))*(1+.04*np.arange(10)[:,None])
    rng = np.random.Generator(np.random.PCG64(640032128))
    directions = []
    for dim in (480,480,479,478):
        p = rng.standard_normal((dim,64)); p /= np.linalg.norm(p,axis=0,keepdims=True)
        directions.append(p.astype(np.float32))
    tp = rng.standard_normal((31,64)); tp /= np.linalg.norm(tp,axis=0,keepdims=True)
    tp = tp.astype(np.float32)
    objective = dg.FrozenDistributionObjective(stats,cfg,directions,tp,source_regions,real_regions)
    full = dg.ParentLayout.full_reference(source_regions,real_regions)
    sslots = np.stack([rng.choice(np.flatnonzero(source_regions==r),size=(4,32),replace=True) for r in dg.CATEGORIES])
    rslots = np.stack([rng.choice(np.flatnonzero(real_regions==r),size=(4,32),replace=True) for r in dg.CATEGORIES])
    alayouts, blayout = dg.paired_layouts(sslots,rslots,source_regions,real_regions)
    fixture_hash = hashlib.sha256()
    for a in (source_np,real_np,source_counts,real_counts,noise.numpy(),mu.numpy(),scale.numpy(),q.numpy(),bins.numpy(),condition.numpy(),
              net.coefficients.detach().numpy(),net.basis.numpy(),*(np.asarray(stats[k]) for k in sorted(stats)),*directions,tp,sslots,rslots):
        fixture_hash.update(str(a.shape).encode()); fixture_hash.update(str(a.dtype).encode()); fixture_hash.update(a.tobytes())
    with torch.no_grad():
        cached_y,cached_raw = endpoint_all()
    assert bool((cached_y.abs()==.5).any()), 'Fixture must exercise active Box rails'
    assert bool((cached_y.abs()<.5).any()), 'Fixture must retain free coordinates'
    assert bool((q>0).all())
    stage('fixture_constructed_only_from_artificial_constants')

    # Original R63 32-per-region sorting objective: all nine pieces are finite,
    # but only the five D pieces enter numerical/algorithm-gradient comparison.
    original = R63.loss.DistributionLoss(stats,cfg,torch.device('cpu'))
    layout = alayouts[0]
    sy, ry = list(layout.source_global), list(layout.real_global)
    identity_ids = np.arange(64)%len(real)
    ident = R63.core.canonical_REAL_torch(real[identity_ids])
    ident_bins = torch.from_numpy(real_regions[identity_ids])
    source_psd = R63.old_loss.psd(source)[sy]
    source_h = torch.ones(320,dtype=torch.int64)
    equal_leaf = cached_y.clone().requires_grad_(True)
    output = torch.cat([equal_leaf[sy],ident])
    raw = torch.cat([cached_raw[sy],ident])
    _,old,old_diag = original(output,raw,real[ry],real[identity_ids],ident_bins,source_psd,
                              source_h,directions,source_psd-1e-8,source_psd+1e-8,tp)
    old_values = {name:float(value.detach()) for name,value in old.items()}
    old_gradients = {name:torch.autograd.grad(old[name],equal_leaf,retain_graph=True)[0].detach() for name in dg.D_NAMES}
    prepared = objective.endpoint_gradients(cached_y,real,layout)
    for name in dg.D_NAMES:
        compare('R63_equal32_value_'+name,prepared['values'][name],old_values[name],'equal_R63_f32_vs_f64_value')
        compare('R63_equal32_gradient_'+name,prepared['gradients'][name],old_gradients[name],'equal_R63_f32_vs_f64_endpoint_gradient')
        assert float(torch.linalg.vector_norm(prepared['gradients'][name])) > TOL['minimum_nonzero_gradient_norm']
    del output, old, old_gradients, equal_leaf
    stage('original_R63_equal32_values_and_algorithm_gradients_passed')

    # Independent dense CDF-rectangle oracle. It intersects EVERY sorted
    # SOURCE/REAL quantile interval pair on an exact integer product grid;
    # it never calls quantile_plan or the preparation's linear-time merge.
    def dense_oracle_columns(a,b,ma,mb):
        aa,bb = a.detach().double().numpy(),b.detach().double().numpy()
        ww,gg,hh = [],[],[]
        for j in range(aa.shape[1]):
            ix,iy = np.argsort(aa[:,j],kind='stable'),np.argsort(bb[:,j],kind='stable')
            ax=[ma.ticks[int(i)]*mb.denominator for i in ix]
            by=[mb.ticks[int(i)]*ma.denominator for i in iy]
            cx=[0]; cy=[0]
            for v in ax: cx.append(cx[-1]+v)
            for v in by: cy.append(cy[-1]+v)
            denominator = ma.denominator*mb.denominator
            overlap = np.array([[max(0,min(cx[i+1],cy[k+1])-max(cx[i],cy[k]))/denominator
                                 for k in range(len(iy))] for i in range(len(ix))])
            delta=aa[ix,j,None]-bb[iy,j][None,:]
            ww.append(math.fsum(float(z) for z in (overlap*np.abs(delta)).reshape(-1)))
            gx=np.zeros(len(ix));gy=np.zeros(len(iy))
            gx[ix]=(overlap*np.sign(delta)).sum(1);gy[iy]=-(overlap*np.sign(delta)).sum(0)
            gg.append(gx);hh.append(gy)
        ga=torch.from_numpy(np.stack(gg,axis=1));gb=torch.from_numpy(np.stack(hh,axis=1))
        # Independent explicit first-order selected-branch surrogate.
        return torch.tensor(ww,dtype=torch.float64)+((a-a.detach()).double()*ga).sum(0)+((b-b.detach()).double()*gb).sum(0)

    def oracle_full(y):
        v=R63.core.canonical_adc_torch(y,True); ref=R63.core.canonical_REAL_torch(real)
        sm=source_counts/source_counts.sum()
        full_terms=[];rms_terms=[]
        uniform=lambda n:FrozenMass.from_normalized(np.full(n,1/n),label='independent_uniform')
        for name,fy,fr,p in zip(dg.NAMES,R63.old_loss.feature_vectors(v),R63.old_loss.feature_vectors(ref),directions):
            f=[];c=[]
            for k,r in enumerate(dg.CATEGORIES):
                si=np.flatnonzero(source_regions==r);ri=np.flatnonzero(real_regions==r)
                a=((fy[si]-torch.tensor(stats[name+'_center'][k]))/torch.tensor(stats[name+'_scale'][k])).float()
                b=((fr[ri]-torch.tensor(stats[name+'_center'][k]))/torch.tensor(stats[name+'_scale'][k])).float()
                ma,mb=uniform(len(si)),uniform(len(ri))
                w=dense_oracle_columns(a,b,ma,mb)
                sp=dense_oracle_columns(a@torch.from_numpy(p),b@torch.from_numpy(p),ma,mb)
                f.append(.5*(w.mean()+sp.mean()));c.append(torch.linalg.vector_norm(w)/math.sqrt(len(w)))
            full_terms.append(sum(sm[k]*f[k] for k in range(10)))
            rms_terms.append(sum(sm[k]*c[k] for k in range(10)))
        yp,rp=R63.old_loss.psd(v),R63.old_loss.psd(ref)
        overall=dense_oracle_columns(yp[:,None],rp[:,None],uniform(len(yp)),uniform(len(rp)))[0]
        cond=[]
        for r in range(7):
            si=np.flatnonzero(source_regions==r);ri=np.flatnonzero(real_regions==r)
            cond.append(dense_oracle_columns(yp[si,None],rp[ri,None],uniform(len(si)),uniform(len(ri)))[0])
        nn=dense_oracle_columns(R63.old_loss.noise_features(v),R63.old_loss.noise_features(ref),uniform(len(v)),uniform(len(ref)))
        tf,tr=R63.loss.temporal_features(v),R63.loss.temporal_features(ref)
        temporal=[]
        for k,r in enumerate(dg.CATEGORIES):
            si=np.flatnonzero(source_regions==r);ri=np.flatnonzero(real_regions==r)
            a=((tf[si]-torch.tensor(stats['temporal_center'][k]))/torch.tensor(stats['temporal_scale'][k])).float()
            b=((tr[ri]-torch.tensor(stats['temporal_center'][k]))/torch.tensor(stats['temporal_scale'][k])).float()
            ma,mb=uniform(len(si)),uniform(len(ri))
            m=dense_oracle_columns(a,b,ma,mb)
            p=dense_oracle_columns(a@torch.from_numpy(tp),b@torch.from_numpy(tp),ma,mb)
            temporal.append(.5*(m.mean()+p.mean()))
        return dict(full_window=torch.stack(full_terms).mean(),coordinate_RMS=torch.stack(rms_terms).mean(),
                    PSD=.5*(overall+torch.stack(cond).mean())/.01,
                    noise=(nn/torch.tensor(stats['noise_scale'])).mean(),
                    temporal_joint=sum(sm[k]*temporal[k] for k in range(10)))

    full_result=objective.endpoint_gradients(cached_y,real,full)
    generic_objective=dg.FrozenDistributionObjective(stats,cfg,directions,tp,source_regions,real_regions,uniform_fastpath=False)
    generic_full=generic_objective.endpoint_gradients(cached_y,real,full)
    for name in dg.D_NAMES:
        compare('uniform_fastpath_vs_generic_full_value_'+name,full_result['values'][name],generic_full['values'][name],'unequal_oracle_value')
        compare('uniform_fastpath_vs_generic_full_gradient_'+name,full_result['gradients'][name],generic_full['gradients'][name],'unequal_oracle_endpoint_gradient')
    generic_equal=generic_objective.endpoint_gradients(cached_y,real,layout)
    for name in dg.D_NAMES:
        compare('uniform_fastpath_vs_generic_equal32_value_'+name,prepared['values'][name],generic_equal['values'][name],'unequal_oracle_value')
        compare('uniform_fastpath_vs_generic_equal32_gradient_'+name,prepared['gradients'][name],generic_equal['gradients'][name],'unequal_oracle_endpoint_gradient')
    stage('uniform_integer_plan_fastpath_vs_generic_full_and_equal32_passed')
    oracle_leaf=cached_y.clone().requires_grad_(True)
    oracle=oracle_full(oracle_leaf)
    for name in dg.D_NAMES:
        g=torch.autograd.grad(oracle[name],oracle_leaf,retain_graph=True)[0].detach()
        compare('full_unequal_oracle_value_'+name,full_result['values'][name],oracle[name],'unequal_oracle_value')
        compare('full_unequal_oracle_gradient_'+name,full_result['gradients'][name],g,'unequal_oracle_endpoint_gradient')
    del oracle,oracle_leaf
    assert bool((full_result['diagnostics']['coordinate_RMS_by_region'][:,0]==0).all())
    assert bool((full_result['gradients']['coordinate_RMS'][source_regions==-1]==0).all())
    TESTS['zero_RMS_exact_zero_value_and_selected_gradient']={'passed':True,'zero_region':-1}
    stage('full_unequal_counts_dense_CDF_oracle_and_zero_RMS_passed')

    # Duplicate-parent pullback: explicitly independent occurrence leaves,
    # then scatter_add vs direct parent gather. The original 32 sorting loss
    # supplies the occurrence-level objective independently of this module.
    occurrence=cached_y[sy].clone().requires_grad_(True)
    old_result=original(torch.cat([occurrence,ident]),raw,real[ry],real[identity_ids],ident_bins,
                        source_psd,source_h,directions,source_psd-1e-8,source_psd+1e-8,tp)[1]
    occurrence_total=sum(old_result[name] for name in dg.D_NAMES)
    gslot=torch.autograd.grad(occurrence_total,occurrence)[0]
    scatter=torch.zeros_like(cached_y).index_add_(0,torch.tensor(sy),gslot)
    # The original-vs-f64 reduction difference uses the declared f32 rule;
    # direct gather/scatter identity is checked separately at exact f64 law.
    compare('duplicates_original_slot_scatter_vs_preparation',prepared['total_gradient'],scatter,'equal_R63_f32_vs_f64_endpoint_gradient')
    expanded_regions=source_regions[sy]
    expanded_real_regions=real_regions[ry]
    expanded_stats=dict(stats,source_counts=np.full(10,32),REAL_counts=np.full(10,32))
    # For the direct scatter check retain the original region masses by
    # evaluating occurrence leaves through one parent graph expansion.
    duplicate_leaf=cached_y.clone().requires_grad_(True)
    detached_occurrence=duplicate_leaf[sy].detach().requires_grad_(True)
    exact_parent_result=objective(duplicate_leaf,real,layout)
    exact_g=torch.autograd.grad(exact_parent_result.total,duplicate_leaf)[0]
    original_column_w1=dg.column_w1
    # Kernel duplicate check uses the global PSD/noise masses for every slot;
    # full objective parent equivalence was checked above through original loss.
    ma=objective._mass(layout,'SOURCE');mb=objective._mass(layout,'REAL')
    psg=R63.old_loss.psd(R63.core.canonical_adc_torch(detached_occurrence,True))
    pr=R63.old_loss.psd(R63.core.canonical_REAL_torch(real[ry]))
    scalar=dg.column_w1(psg[:,None],pr[:,None],ma,mb)[0]
    gocc=torch.autograd.grad(scalar,detached_occurrence)[0]
    pl=cached_y.clone().requires_grad_(True)
    val=dg.column_w1(R63.old_loss.psd(R63.core.canonical_adc_torch(pl,True))[sy,None],pr[:,None],ma,mb)[0]
    gparent=torch.autograd.grad(val,pl)[0]
    compare('duplicate_exact_global_mass_scatter',torch.zeros_like(cached_y).index_add_(0,torch.tensor(sy),gocc),gparent,'duplicate_scatter_gradient')
    assert max(np.bincount(np.asarray(sy),minlength=len(source)))>1
    stage('duplicate_parent_gradient_accumulation_passed')

    # Both arms use exactly the same occurrence IDs/noise and region-major row
    # order. C is never constructed for A/B, and no network reforward is used.
    a_results=[prepared]+[objective.endpoint_gradients(cached_y,real,l) for l in alayouts[1:]]
    b_result=objective.endpoint_gradients(cached_y,real,blayout)
    generic_b=generic_objective.endpoint_gradients(cached_y,real,blayout)
    for name in dg.D_NAMES:
        compare('uniform_fastpath_vs_generic_equal128_value_'+name,b_result['values'][name],generic_b['values'][name],'unequal_oracle_value')
        compare('uniform_fastpath_vs_generic_equal128_gradient_'+name,b_result['gradients'][name],generic_b['gradients'][name],'unequal_oracle_endpoint_gradient')
    a_values={name:sum(x['values'][name] for x in a_results)/4 for name in dg.D_NAMES}
    a_gradients={name:sum(x['gradients'][name] for x in a_results)/4 for name in dg.D_NAMES}
    for k in range(10):
        assert blayout.source[k]==tuple(int(i) for i in sslots[k].reshape(-1))
        assert blayout.real[k]==tuple(int(i) for i in rslots[k].reshape(-1))
    try:
        wrong_s=sslots.transpose(1,0,2).reshape(10,128)
        wrong_r=rslots.transpose(1,0,2).reshape(10,128)
        dg.ParentLayout.region_major(wrong_s,wrong_r,source_regions,real_regions)
        raise AssertionError('Incorrect microbatch-major layout was accepted')
    except ValueError as exc:
        TESTS['wrong_microbatch_major_layout_rejected']={'passed':True,'error':str(exc)}
    try:
        badstats=dict(stats,REAL_counts=source_counts)
        dg.FrozenDistributionObjective(badstats,cfg,directions,tp,source_regions,real_regions)
        raise AssertionError('Wrong population masses were accepted')
    except ValueError as exc:
        TESTS['wrong_population_mass_binding_rejected']={'passed':True,'error':str(exc)}
    # Deliberately bypass validation ONLY inside this negative-control subclass.
    class WrongMass(dg.FrozenDistributionObjective):
        def _mass(self,layout,domain):
            if domain=='SOURCE':return super()._mass(layout,domain)
            weights=[self.source_mass[k]/len(layout.real[k]) for k in range(10) for _ in layout.real[k]]
            return FrozenMass.from_normalized(weights,label='DELIBERATE_WRONG_REAL_USES_SOURCE')
    wrong=WrongMass(stats,cfg,directions,tp,source_regions,real_regions).endpoint_gradients(cached_y,real,layout,terms=('PSD','noise'))
    wrong_gap=max(abs(wrong['values'][k]-prepared['values'][k]) for k in ('PSD','noise'))
    wrong_grad_gap=max(float((wrong['gradients'][k]-prepared['gradients'][k]).abs().max()) for k in ('PSD','noise'))
    assert wrong_gap>TOL['negative_wrong_mass_min_value_difference']
    assert wrong_grad_gap>TOL['negative_wrong_mass_min_gradient_difference']
    TESTS['wrong_global_mass_detectably_changes_values_and_gradients']={'passed':True,'max_value_gap':wrong_gap,'max_gradient_gap':wrong_grad_gap}
    for wrong_terms in ((),('PSD','PSD'),('source_order',)):
        try:
            objective.endpoint_gradients(cached_y,real,layout,terms=wrong_terms)
            raise AssertionError('Invalid or C term selection accepted')
        except ValueError:
            pass
    TESTS['empty_duplicate_and_C_gradient_requests_rejected']={'passed':True}
    stage('A_four32_B128_region_major_and_negative_controls_passed')

    # One graph reference over the SAME small chunk schedule, versus two-pass
    # cached endpoint gradients injected at out. Original Box and STE once each.
    graph_y,graph_raw=endpoint_all()
    graph=objective(graph_y,real,full)
    graph_parameter={}
    graph_endpoint={}
    for name in dg.D_NAMES:
        gg=torch.autograd.grad(graph.pieces[name],(graph_y,net.coefficients),retain_graph=True)
        graph_endpoint[name],graph_parameter[name]=gg[0].detach(),gg[1].detach()
    compare('same_schedule_firstpass_vs_singlegraph_Y',cached_y,graph_y,'same_schedule_two_pass_endpoint')
    del graph,graph_y,graph_raw
    replay_parameter={name:torch.zeros_like(net.coefficients) for name in dg.D_NAMES}
    replay_checks=[]
    for a,b in schedule:
        y,raw2=endpoint_chunk(a,b)
        assert torch.equal(y.detach(),cached_y[a:b]), 'Projected endpoint replay drift'
        assert torch.equal(raw2.detach(),cached_raw[a:b]), 'Raw endpoint replay drift'
        assert torch.equal(R63.core.canonical_adc_torch(y.detach()),R63.core.canonical_adc_torch(cached_y[a:b])), 'ADC replay drift'
        assert torch.equal((y.detach()>-.5)&(y.detach()<.5),(cached_y[a:b]>-.5)&(cached_y[a:b]<.5)), 'Box active set drift'
        for j,name in enumerate(dg.D_NAMES):
            replay_parameter[name]+=torch.autograd.grad(y,net.coefficients,grad_outputs=full_result['gradients'][name][a:b],retain_graph=j<len(dg.D_NAMES)-1)[0].detach()
        replay_checks.append({'start':a,'stop':b,'Y_raw_codes_free_bitwise_equal':True})
    for name in dg.D_NAMES:
        compare('two_pass_parent_endpoint_gradient_'+name,full_result['gradients'][name],graph_endpoint[name],'same_schedule_two_pass_endpoint')
        compare('two_pass_parameter_gradient_'+name,replay_parameter[name],graph_parameter[name],'same_schedule_two_pass_parameter_gradient')
        assert float(torch.linalg.vector_norm(replay_parameter[name]))>TOL['minimum_nonzero_gradient_norm']
    compare('two_pass_total_parameter_gradient',sum(replay_parameter.values()),sum(graph_parameter.values()),'same_schedule_two_pass_parameter_gradient')
    stage('original_endpoint_Box_STE_two_pass_VJP_passed')

    # At least two distinct parents collide at exact canonical ADC values.
    canonical=R63.core.canonical_adc_torch(cached_y)
    duplicate_waveforms=int(len(canonical)-len(torch.unique(canonical,dim=0)))
    assert duplicate_waveforms>0
    receipt={
        'status':'ARTIFICIAL_PREPARATION_PASSED_NOT_SCIENTIFIC_REGISTRATION_OR_PASS',
        'PID':os.getpid(),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
        'CPU_affinity':[cpu],'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'RSS_limit_bytes':TOL['RSS_limit_bytes'],'peak_RSS_bytes':resources(),'elapsed_seconds':time.monotonic()-START,
        'tolerance_sha256_before_run':TOL_SHA,'fixture_arrays_sha256':fixture_hash.hexdigest(),
        'fixture':{'waveform_dimension':480,'regions':list(dg.CATEGORIES),'source_counts':source_counts.tolist(),'REAL_counts':real_counts.tolist(),
                   'all_original_artificial_parents_in_F':True,'independent_population_masses':True,'same_parent_repeated_slots':True,
                   'SOURCE_unique_parents':len(source),'REAL_unique_parents':len(real),'ADC_duplicate_parent_waveforms':duplicate_waveforms,
                   'active_Box_rail_coordinates':int((cached_y.abs()==.5).sum()),'exact_zero_RMS_region':-1,'toy_network_nonzero_coefficients':net.coefficients.detach().tolist()},
        'tests':TESTS,'original_nine_piece_values_32':old_values,'D_F_values':full_result['values'],'D_A_four32_values':a_values,'D_B128_values':b_result['values'],
        'D_F_parent_gradient_norms':{k:float(torch.linalg.vector_norm(v)) for k,v in full_result['gradients'].items()},
        'D_F_parameter_gradients':{k:v.tolist() for k,v in replay_parameter.items()},
        'D_A_B_endpoint_gradient_distance':{k:float(torch.linalg.vector_norm(a_gradients[k]-b_result['gradients'][k])) for k in dg.D_NAMES},
        'same_schedule_replay_checks':replay_checks,
        'source_bindings':json.loads((ROOT/'SOURCE_BINDINGS.json').read_text('utf-8-sig')),
        'implementation_files':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(ROOT.glob('*.py'))],
        'scientific_arrays_opened':0,'scientific_weights_loaded':0,'DEV_EVAL_CAL_reads':0,'scientific_forward_or_gradient_runs':0,'optimizer_updates':0,
        'real_R63_ResponseNetwork_executed':False,'original_endpoint_Box_STE_feature_code_executed_with_artificial_data':True,
        'C_terms_in_primary_gradient_comparison':False,
        'limitations':['Toy network only: real R63 attention/kernel replay not tested here.','Fixed per-parent noise and directions define a conditional finite empirical law, not original independent-occurrence noise/fresh-direction expectation.','Algorithm gradient uses canonical STE, stable original-row tie branch, abs(0)=0 and exact zero-RMS branch; no unique nonsmooth derivative or hard-ADC finite-difference equivalence is claimed.','CPU only; first-order gradients only; no scientific job registered or launched.'],
    }
    (ROOT/'ARTIFICIAL_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf8')
    stage('receipt_written')
    print(json.dumps({'status':receipt['status'],'tests':len(TESTS),'peak_RSS_bytes':receipt['peak_RSS_bytes'],'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
except BaseException as exc:
    failure={'status':'ARTIFICIAL_FAILED_NOT_SCIENTIFIC_RUN','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
             'elapsed_seconds':time.monotonic()-START,'peak_RSS_bytes':resources(),'tolerance_sha256_before_run':TOL_SHA}
    (ROOT/'ARTIFICIAL_FAILURE.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf8')
    print(json.dumps(failure),flush=True)
    raise
finally:
    stop.set()
