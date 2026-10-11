"""Portable, synthetic-only actual-network R65 joint D+C update check.

The caller owns resource guards, thread/backend settings, -B, logs and writes.
Importing this file never constructs a model, reads data, or executes a test.
run_check is intentionally NOT executed by the local preparation validator.

The three public gradient/update functions below are the same functions exercised
by the fixture and are reusable by the root-owned trainer. They have no data I/O,
model construction, random sampling, extra averaging or implicit zero_grad.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
D_NAMES = ("full_window", "PSD", "noise", "coordinate_RMS", "temporal_joint")
C_NAMES = ("source_order", "source_valley", "REAL_identity", "box_correction")


def _flatten_legacy(gradients, parameters):
    import torch
    flat = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None or gradient.shape != parameter.shape or gradient.dtype != parameter.dtype:
            raise RuntimeError("Missing, reshaped or differently typed parameter gradient")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Nonfinite parameter gradient")
        flat.append(gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float64))
    return torch.cat(flat)

def _flatten_gradients_cpu_f64(gradients, parameters):
    """Bit-preserving fast path for homogeneous real f32/f64 gradients.

    All elements retain original parameter order. A same-dtype cat does no
    reduction. f32 -> f64 is exact for finite IEEE values. Mixed devices/dtypes
    and other dtypes fall back to the old operation sequence. CUDA subnormal
    behavior must still be tested against the original fused .to(cpu, f64).
    """
    import torch
    gradients, parameters = tuple(gradients), tuple(parameters)
    if len(gradients) != len(parameters) or not parameters:
        raise ValueError("Every supplied parameter must have one gradient")
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None or gradient.shape != parameter.shape or gradient.dtype != parameter.dtype:
            raise RuntimeError("Missing, reshaped or differently typed parameter gradient")
    first = gradients[0]
    if first.dtype not in (torch.float32, torch.float64) or any(
        g.device != first.device or g.dtype != first.dtype or g.layout != torch.strided
        for g in gradients
    ):
        return _flatten_legacy(gradients, parameters)
    packed = torch.cat([g.detach().reshape(-1) for g in gradients])
    # Explicitly copy native dtype first; cast on CPU, never reduce on GPU.
    native_cpu = packed.to(device="cpu")
    result = native_cpu.to(dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Nonfinite parameter gradient")
    return result

def parameter_vjp_cpu_f64(output, parameters, gradient):
    """One VJP at projected pre-ADC output, returned in original parameter order."""
    import torch
    parameters = tuple(parameters)
    if output.shape != gradient.shape or output.dtype != torch.float64 or gradient.dtype != torch.float64:
        raise ValueError("Output and GY must be matching float64 tensors")
    if output.device != gradient.device or gradient.requires_grad:
        raise ValueError("Detached GY must be on the replay output device")
    values = torch.autograd.grad(output, parameters, grad_outputs=gradient,
                                 create_graph=False, retain_graph=False, allow_unused=False)
    return _flatten_gradients_cpu_f64(values, parameters)


def scalar_parameter_gradient_cpu_f64(scalar, parameters, *, retain_graph=False):
    """One ordinary scalar reverse pass, used for the unchanged original C graph."""
    import torch
    parameters = tuple(parameters)
    if scalar.ndim != 0:
        raise ValueError("Scalar objective required")
    values = torch.autograd.grad(scalar, parameters, create_graph=False,
                                 retain_graph=retain_graph, allow_unused=False)
    return _flatten_gradients_cpu_f64(values, parameters)


def install_cpu_f64_clip_step(parameters, gradient_cpu_f64, optimizer, *, clip=5.0, capture=False):
    """Install one CPU f64 sum, cast once per parameter, clip once, step once.

    Caller must have zeroed gradients to None and supplied parameters in the same
    order used to form the vector. The optimizer and its LR/state remain owned by
    the caller. capture=True adds detached CPU f64 gradient vectors for checks.
    """
    import torch
    parameters = tuple(parameters)
    if gradient_cpu_f64.device.type != "cpu" or gradient_cpu_f64.dtype != torch.float64 or gradient_cpu_f64.ndim != 1:
        raise ValueError("Combined gradient must be a flat CPU float64 vector")
    if gradient_cpu_f64.numel() != sum(p.numel() for p in parameters) or not bool(torch.isfinite(gradient_cpu_f64).all()):
        raise ValueError("Combined gradient length or finiteness mismatch")
    if not parameters or any(p.grad is not None for p in parameters):
        raise RuntimeError("Caller must zero_grad(set_to_none=True) before installation")
    if clip != 5.0:
        raise ValueError("Selected original clipping threshold is 5")
    optimizer_parameters = tuple(p for group in optimizer.param_groups for p in group["params"])
    if len(optimizer_parameters) != len(parameters) or any(a is not b for a,b in zip(optimizer_parameters, parameters)):
        raise ValueError("Optimizer parameter order must match the accumulated vector")
    offset = 0
    for p in parameters:
        size = p.numel()
        p.grad = gradient_cpu_f64[offset:offset+size].reshape(p.shape).to(device=p.device, dtype=p.dtype)
        offset += size
    before = _flatten_gradients_cpu_f64((p.grad for p in parameters), parameters) if capture else None
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    after = _flatten_gradients_cpu_f64((p.grad for p in parameters), parameters) if capture else None
    optimizer.step()
    clip_coefficient = float(torch.clamp(clip/(norm+1e-6), max=1.0).detach().cpu())
    receipt = {"pre_clip_norm": float(norm.detach().cpu()), "clip_threshold": clip,
               "did_clip": clip_coefficient < 1.0, "clip_coefficient": clip_coefficient,
               "one_parameter_dtype_cast": True, "cast_calls": 1,
               "parameter_tensor_cast_calls": len(parameters), "clip_calls": 1,
               "optimizer_calls": 1, "optimizer_step_calls": 1}
    if capture:
        receipt.update(before_clip=before, after_clip=after)
    return receipt


class ArtificialNetworkCheckFailure(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(receipt.get("error_type", "Failure")+": "+receipt.get("error", ""))


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _array_hash(np, values):
    h = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        h.update(name.encode("utf8")); h.update(str(value.shape).encode()); h.update(value.dtype.str.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def _tensor_hash(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        v = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tuple(v.shape)).encode()); h.update(str(v.dtype).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def _verified_bundle(objective_adapter_root):
    binding_path = ROOT / "SOURCE_BINDING.json"
    binding_bytes = binding_path.read_bytes()
    binding = json.loads(binding_bytes.decode("utf-8-sig"))
    if _sha_bytes(Path(__file__).read_bytes()) != binding["helper_sha256"]:
        raise RuntimeError("Network helper source differs from its preparation binding")
    own_values = {}
    for name, expected in binding["own_inputs"].items():
        if name not in ("FROZEN_TOLERANCES.json", "FIXTURE_SPEC.json"):
            raise ValueError("Unexpected own input")
        content = (ROOT/name).read_bytes()
        if _sha_bytes(content) != expected:
            raise RuntimeError("Frozen own input mismatch: "+name)
        own_values[name] = json.loads(content.decode("utf-8-sig"))
    root = Path(objective_adapter_root).resolve(strict=True)
    actual = []
    for record in binding["objective_files"]:
        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix not in (".py", ".json"):
            raise ValueError("Invalid explicit definition-file binding")
        path = (root/relative).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("Definition file escapes objective adapter root")
        digest = _sha_bytes(path.read_bytes())
        if digest != record["sha256"]:
            raise RuntimeError("Bound objective source mismatch: "+record["file"])
        actual.append({"file": record["file"], "sha256": digest})
    return root, own_values["FROZEN_TOLERANCES.json"], own_values["FIXTURE_SPEC.json"], actual, _sha_bytes(binding_bytes)


@contextmanager
def _definition_module(name, path):
    absent = object()
    previous = sys.modules.get(name, absent)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is absent:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _fixture(np, R63, spec, variant):
    setting = spec[variant]
    categories = np.asarray(spec["region_categories"], dtype="i8")
    counts_s = np.asarray(setting["source_counts"], dtype="i8")
    counts_r = np.asarray(setting["REAL_counts"], dtype="i8")
    t = np.arange(480, dtype="f8")
    edges = np.log(1000*np.asarray(spec["calibration"]["q_edges"], dtype="f8"))
    mu = np.zeros((8, 480), dtype="f8"); mu[:,100:400] = 1/300.
    scale = np.full((8,480), spec["calibration"]["unit_charge_scale"], dtype="f8")
    seeds = spec["seeds"]
    streams = {k: np.random.Generator(np.random.PCG64(v)) for k,v in seeds.items()}

    def population(counts, real=False):
        rows, labels, h = [], [], []
        for r, count in zip(categories, counts):
            for j in range(int(count)):
                q_target = (1.5 if r == -1 else 10.5 if r == 8 else 2.5+int(r))*(1+.003*j/max(int(count),1))
                tau = 55.+.8*(int(r)+1)+.05*j+(3. if real else 0.)
                pulse = np.where((t>=100)&(t<400), np.exp(-(t-100)/tau), 0.)
                pulse = pulse*(q_target/R63.noise.q_np(pulse))
                baseline = .0003*np.sin(t/7.+.31*j+.17*(int(r)+1)+(0.2 if real else 0.))
                x = pulse+baseline
                x += mu[int(np.clip(r,0,7))]*(q_target-R63.noise.q_np(x))
                rows.append(x.astype("<f4")); labels.append(int(r)); h.append(int(j%2==0))
        rows, labels, h = np.stack(rows), np.asarray(labels,dtype="i8"), np.asarray(h,dtype="i8")
        if not real:
            rows[1] = rows[0]
            rows[1,220] = np.nextafter(rows[1,220], np.float32(np.inf))
        permutation = streams["original_parent_permutations"].permutation(len(rows))
        return rows[permutation], labels[permutation], h[permutation], permutation

    source, sraw, source_h, sp_order = population(counts_s)
    real, rraw, _, rp_order = population(counts_r, real=True)
    sb, sb_raw, sq = R63.noise.bins_for(source, edges)
    rb, rb_raw, rq = R63.noise.bins_for(real, edges)
    if not np.array_equal(sb_raw,sraw) or not np.array_equal(rb_raw,rraw):
        raise AssertionError("Synthetic original raw regions disagree with original bins_for")
    sh = R63.noise.continuous_logq(sq,sb,edges)
    rh = R63.noise.continuous_logq(rq,rb,edges)
    source_psd = 1-(source.astype("f8")[:,100:150].sum(1)-.5*source.astype("f8")[:,:100].sum(1))/sq
    low, high = R63.precision.rounding_psd_interval(source)
    groups_s = [np.flatnonzero(sraw==r) for r in categories]
    groups_r = [np.flatnonzero(rraw==r) for r in categories]
    si = np.concatenate([g[streams["C_SOURCE_rows"].integers(0,len(g),32)] for g in groups_s]).astype("i8")
    ri = np.concatenate([g[streams["C_REAL_reference_rows"].integers(0,len(g),32)] for g in groups_r]).astype("i8")
    ii = streams["C_REAL_identity_rows"].integers(0,len(real),64).astype("i8")
    eps_d = streams["D_parent_epsilon"].standard_normal(source.shape).astype("<f4")
    eps_c = streams["C_SOURCE_occurrence_epsilon"].standard_normal((320,480)).astype("<f4")
    eps_i = streams["C_identity_occurrence_epsilon"].standard_normal((64,480)).astype("<f4")
    eps_d[0] *= 110.; eps_c[0] *= 110.; eps_i[0] *= 110.
    p = np.ones(241,dtype="f8"); kappa2 = np.ones(8,dtype="f8"); tau2=spec["calibration"]["noise_tau2"]
    schedule = [tuple(v) for v in setting["schedule"]]
    nd = np.empty(source.shape,dtype="f8")
    for lo_,hi_ in schedule:
        nd[lo_:hi_], _ = R63.noise.innovation(source[lo_:hi_],eps_d[lo_:hi_],mu,edges,p,kappa2,tau2,True)
    nc, _ = R63.noise.innovation(source[si],eps_c,mu,edges,p,kappa2,tau2,True)
    ni, _ = R63.noise.innovation(real[ii],eps_i,mu,edges,p,kappa2,tau2,True)
    directions=[]
    for d in (480,480,479,478):
        direction=streams["full_window_directions"].standard_normal((d,64))
        direction/=np.linalg.norm(direction,axis=0,keepdims=True)
        directions.append(direction.astype("<f4"))
    td=streams["temporal_directions"].standard_normal((31,64))
    td=(td/np.linalg.norm(td,axis=0,keepdims=True)).astype("<f4")
    stats={"source_counts":counts_s,"REAL_counts":counts_r,"noise_scale":np.array([.007,.2],dtype="f8")}
    for name,d,sc in zip(("V480","U480","DU479","D2U478"),(480,480,479,478),(.05,.25,.03,.03)):
        stats[name+"_center"]=np.zeros((10,d),dtype="f8")
        stats[name+"_scale"]=np.full((10,d),sc,dtype="f8")
    stats["temporal_center"]=np.zeros((10,31),dtype="f8")
    stats["temporal_scale"]=np.tile(np.array([.007]+[.2]*20+[.1]*10,dtype="f8"),(10,1))
    bounds=[]; fractions=[]; raw_fractions=[]
    for region in (2,3):
        parents=(sraw==region); selected=parents&(source_h==1)
        a=float(source_psd[selected].max())+.003
        bounds.append([a,a+.035]); fractions.append(float(selected.sum()/parents.sum()))
        raw_fractions.append(float(np.mean((source_psd[selected]>=a)&(source_psd[selected]<=a+.035))))
    stats["valley_bounds"]=np.asarray(bounds,dtype="f8")
    stats["source_h_one_fraction"]=np.asarray(fractions,dtype="f8")
    stats["source_raw_valley_fraction"]=np.asarray(raw_fractions,dtype="f8")
    cfg={"batch_per_region":32,"projection_directions":64,"temporal_projection_directions":64,
         "temporal_feature_count":31,"temporal_power_floor":1e-20,"PSD_loss_scale":.01,
         "valley_delta":.02,"valley_smoothing":.002,
         "loss_weights":{k:1. for k in ("full_window","PSD","source_order","source_valley","noise","REAL_identity","box_correction","coordinate_RMS","temporal_joint")}}
    arrays={"source":source,"REAL":real,"source_raw_bins":sraw,"REAL_raw_bins":rraw,"source_h":source_h,
            "source_original_order":sp_order,"REAL_original_order":rp_order,"source_psd":source_psd,
            "interval_low":low,"interval_high":high,"epsilon_D":eps_d,"epsilon_C_SOURCE":eps_c,"epsilon_C_identity":eps_i,
            "innovation_D":nd,"innovation_C_SOURCE":nc,"innovation_C_identity":ni,"C_SOURCE_rows":si,"C_REAL_reference_rows":ri,
            "C_identity_rows":ii,"mu":mu,"scale":scale,"edges":edges,"temporal_directions":td}
    arrays.update({"direction_"+str(i):v for i,v in enumerate(directions)})
    arrays.update({"statistics_"+k:v for k,v in stats.items()})
    return dict(source=source,real=real,sraw=sraw,rraw=rraw,source_h=source_h,source_psd=source_psd,
                source_low=low,source_high=high,mu=mu,scale=scale,stats=stats,cfg=cfg,directions=directions,td=td,
                D=dict(x=source.astype("f8"),noise=nd,q=sq,b=sb,h=sh),
                C=dict(x=np.concatenate((source[si],real[ii])).astype("f8"),noise=np.concatenate((nc,ni)),
                       q=np.concatenate((sq[si],rq[ii])),b=np.concatenate((sb[si],rb[ii])),h=np.concatenate((sh[si],rh[ii]))),
                si=si,ri=ri,ii=ii,schedule=schedule,arrays=arrays,
                fixture_SHA256=_array_hash(np,arrays),cfg_SHA256=_sha_bytes(json.dumps(cfg,sort_keys=True).encode()))


def _tensor_inputs(torch, fixture, device):
    T=lambda a,d=torch.float64:torch.as_tensor(a,device=device,dtype=d)
    data={k:{name:T(value,torch.int64 if name=="b" else torch.float64) for name,value in fixture[k].items()} for k in ("D","C")}
    data["mu"],data["scale"]=T(fixture["mu"]),T(fixture["scale"])
    data["reference"]=T(fixture["real"])
    si,ri,ii=fixture["si"],fixture["ri"],fixture["ii"]
    data["C_args"]=(T(fixture["real"][ri]),T(fixture["real"][ii]),T(fixture["rraw"][ii],torch.int64),
                    T(fixture["source_psd"][si]),T(fixture["source_h"][si],torch.int64),fixture["directions"],
                    T(fixture["source_low"][si]),T(fixture["source_high"][si]),fixture["td"])
    return data


def _forward(R63,model,data,branch,sl):
    a=data[branch]
    return R63.core.endpoint(model,a["x"][sl],a["noise"][sl],a["q"][sl],a["b"][sl],a["h"][sl],data["mu"],data["scale"])


def _endpoint_snapshot(torch,R63,out,raw):
    with torch.no_grad():
        canonical=R63.core.canonical_adc_torch(out,False)
        return {"output":out.detach().cpu().clone(),"raw":raw.detach().cpu().clone(),
                "codes":torch.round((canonical+.5)*16383).to(torch.int64).cpu(),
                "free":((out>-.5)&(out<.5)).detach().cpu()}


def _check_endpoint(torch,R63,out,raw,expected,label,checks,lo=None,hi=None):
    current=_endpoint_snapshot(torch,R63,out,raw)
    values={key:torch.equal(value,expected[key]) for key,value in current.items()}
    checks.append({"name":label,"start":lo,"stop":hi,**values,"passed":all(values.values())})
    if not all(values.values()):
        raise AssertionError(label+": endpoint/raw/codes/free replay mismatch "+str(values))
    return current


def _gradient_compare(torch,reference,candidate,tol,label,checks):
    if reference.shape!=candidate.shape or reference.dtype!=torch.float64 or candidate.dtype!=torch.float64 or reference.device.type!="cpu" or candidate.device.type!="cpu":
        raise ValueError("Gradient comparison requires equal CPU float64 vectors")
    delta=candidate-reference
    norm=float(torch.linalg.vector_norm(reference));err=float(torch.linalg.vector_norm(delta))
    maximum=float(delta.abs().max());refmax=float(reference.abs().max())
    relative=err/norm if norm>0 else float("inf")
    allowance=tol["max_absolute"]+tol["max_relative_to_reference_max"]*refmax
    passed=bool(torch.isfinite(candidate).all() and norm>tol["reference_norm_strictly_above"] and relative<=tol["relative_L2"] and maximum<=allowance)
    row={"name":label,"reference_L2":norm,"error_L2":err,"relative_L2_error":relative,
         "max_absolute_error":maximum,"reference_max_absolute":refmax,"allowed_max_absolute_error":allowance,"passed":passed}
    checks.append(row)
    if not passed:
        raise AssertionError(label+": "+str(row))
    return row


def _C_signal_checks(torch,R63,pieces,diagnostics,out,raw,tol,checks,label):
    signal={}
    for name in C_NAMES:
        value=float(pieces[name].detach().cpu())
        grad,=torch.autograd.grad(pieces[name],out,retain_graph=True,allow_unused=False)
        norm=float(torch.linalg.vector_norm(grad.detach().double()).cpu())
        passed=value>tol["positive_piece_value_strictly_above"] and norm>tol["nonzero_C_piece_endpoint_gradient_norm_strictly_above"]
        signal[name]={"value":value,"endpoint_gradient_L2":norm,"passed":passed}
        if not passed:
            checks.append({"name":label+"_nonzero_C_signals","pieces":signal,"passed":False})
            raise AssertionError(label+": synthetic "+name+" did not exercise a nonzero signal")
    gout,graw=torch.autograd.grad(pieces["box_correction"],(out,raw),retain_graph=True,allow_unused=False)
    outnorm=float(torch.linalg.vector_norm(gout.detach().double()).cpu())
    rawnorm=float(torch.linalg.vector_norm(graw.detach().double()).cpu())
    if outnorm<=tol["nonzero_box_output_gradient_norm_strictly_above"] or rawnorm<=tol["nonzero_box_raw_gradient_norm_strictly_above"]:
        raise AssertionError("Original C box raw/output paths were not both exercised")
    old=float(diagnostics["original_order_loss"].cpu());omitted=float(diagnostics["omitted_order_loss"].cpu())
    current=signal["source_order"]["value"]
    residual=abs(old-omitted-current)
    allowed=tol["source_order_decomposition_atol"]+tol["source_order_decomposition_rtol"]*abs(old)
    fraction=float(diagnostics["resolved_pair_fraction"].cpu())
    denominator=float(diagnostics["original_nonzero_pair_mass"].cpu())
    if residual>allowed or not 0<fraction<1 or denominator<=0:
        raise AssertionError("Original C order denominator/overlap fixture failed")
    answer={"name":label+"_nonzero_C_signals","pieces":signal,"box_output_gradient_L2":outnorm,"box_raw_gradient_L2":rawnorm,
            "source_order_original_nonzero_pair_mass":denominator,"source_order_resolved_pair_fraction":fraction,
            "source_order_decomposition_residual":residual,"passed":True}
    checks.append(answer)
    return answer


def _optimizer_snapshot(torch,model,optimizer):
    result={"parameters":{},"exp_avg":{},"exp_avg_sq":{},"steps":{}}
    for name,p in model.named_parameters():
        state=optimizer.state[p]
        if set(state)!={"step","exp_avg","exp_avg_sq"}:
            raise AssertionError("Unexpected Adam state fields: "+name)
        result["parameters"][name]=p.detach().cpu().double().clone()
        result["exp_avg"][name]=state["exp_avg"].detach().cpu().double().clone()
        result["exp_avg_sq"][name]=state["exp_avg_sq"].detach().cpu().double().clone()
        result["steps"][name]=float(state["step"].detach().cpu()) if isinstance(state["step"],torch.Tensor) else float(state["step"])
    return result


def _perform_artificial_update(torch,model,gradient,tol,checks,label,ledger):
    opts=tol["optimizer"]
    optimizer=torch.optim.Adam(model.parameters(),lr=opts["learning_rate"],betas=tuple(opts["betas"]),
                               eps=opts["epsilon"],weight_decay=opts["weight_decay"])
    optimizer.zero_grad(set_to_none=True)
    attempt={"arm":label,"status":"CAST_CLIP_STEP_CALL_STARTED"}
    ledger.setdefault("synthetic_update_attempts",[]).append(attempt)
    receipt=install_cpu_f64_clip_step(tuple(model.parameters()),gradient,optimizer,clip=opts["gradient_clip"],capture=True)
    ledger["synthetic_optimizer_steps"]+=1
    attempt.update(status="ONE_OPTIMIZER_STEP_RETURNED",**{k:v for k,v in receipt.items() if k not in ("before_clip","after_clip")})
    if tol["signal"]["require_clipping_active"] and receipt["pre_clip_norm"]<=opts["gradient_clip"]:
        raise AssertionError("Synthetic fixture did not exercise clipping")
    state=_optimizer_snapshot(torch,model,optimizer)
    if not all(v==1 for v in state["steps"].values()):
        raise AssertionError("Artificial Adam must advance every parameter exactly once")
    checks.append({"name":label+"_one_clipped_artificial_Adam_step","pre_clip_norm":receipt["pre_clip_norm"],
                   "clip_threshold":opts["gradient_clip"],"parameter_state_count":len(state["steps"]),"step_counters_all_one":True,
                    "did_clip":receipt["did_clip"],"cast_calls":receipt["cast_calls"],
                    "parameter_tensor_cast_calls":receipt["parameter_tensor_cast_calls"],
                    "clip_calls":receipt["clip_calls"],"optimizer_calls":receipt["optimizer_calls"],
                    "optimizer_step_calls":receipt["optimizer_step_calls"],"passed":True})
    return receipt,state


def _direct_arm(torch,dg,R63,initial,fixture,data,device,tol,checks,ledger):
    model=copy.deepcopy(initial).to(device).train()
    parameters=tuple(model.parameters());before=_tensor_hash(model.state_dict())
    objective_D=dg.FrozenDistributionObjective(fixture["stats"],fixture["cfg"],fixture["directions"],fixture["td"],
                fixture["sraw"],fixture["rraw"],feature_chunk=64,device=device)
    objective_C=R63.loss.DistributionLoss(fixture["stats"],fixture["cfg"],device)
    layout=dg.ParentLayout.full_reference(fixture["sraw"],fixture["rraw"])
    parts=[];raw_parts=[]
    for lo,hi in fixture["schedule"]:
        out,raw=_forward(R63,model,data,"D",slice(lo,hi));parts.append(out);raw_parts.append(raw)
    y,raw_y=torch.cat(parts),torch.cat(raw_parts)
    full=objective_D(y,data["reference"],layout,terms=D_NAMES)
    gy,=torch.autograd.grad(full.total,y,retain_graph=True)
    gy=gy.detach().cpu().clone()
    out_C,raw_C=_forward(R63,model,data,"C",slice(None))
    nine,all_pieces,diagnostics=objective_C(out_C,raw_C,*data["C_args"])
    pieces_C={k:all_pieces[k] for k in C_NAMES}
    total_C=sum(fixture["cfg"]["loss_weights"][k]*pieces_C[k] for k in C_NAMES)
    signal=_C_signal_checks(torch,R63,pieces_C,diagnostics,out_C,raw_C,tol["signal"],checks,"direct")
    direct_D=scalar_parameter_gradient_cpu_f64(full.total,parameters,retain_graph=True)
    direct_C=scalar_parameter_gradient_cpu_f64(total_C,parameters,retain_graph=True)
    total=full.total+total_C
    combined=scalar_parameter_gradient_cpu_f64(total,parameters)
    if _tensor_hash(model.state_dict())!=before or any(p.grad is not None for p in parameters):
        raise AssertionError("Direct comparison changed theta before the artificial update")
    endpoint_D=_endpoint_snapshot(torch,R63,y,raw_y)
    endpoint_C=_endpoint_snapshot(torch,R63,out_C,raw_C)
    values={"D":float(full.total.detach().cpu()),"C":float(total_C.detach().cpu()),"D_plus_C":float(total.detach().cpu()),
            "D_pieces":{k:float(v.detach().cpu()) for k,v in full.pieces.items()},
            "C_pieces":{k:float(v.detach().cpu()) for k,v in pieces_C.items()},
            "discarded_micro_D":{k:float(all_pieces[k].detach().cpu()) for k in D_NAMES},
            "original_micro_nine_total":float(nine.detach().cpu())}
    del parts,raw_parts,y,raw_y,full,total
    update,state=_perform_artificial_update(torch,model,combined,tol,checks,"direct",ledger)
    answer=dict(initial_state_SHA256=before,D_gradient=direct_D,C_gradient=direct_C,total_gradient=combined,
                GY=gy,endpoint_D=endpoint_D,endpoint_C=endpoint_C,values=values,signal=signal,update=update,optimizer=state,
                after_state_SHA256=_tensor_hash(model.state_dict()))
    return answer


def _replayed_arm(torch,dg,R63,adapter,initial,fixture,data,device,tol,checks,direct,ledger):
    model=copy.deepcopy(initial).to(device).train()
    parameters=tuple(model.parameters());before=_tensor_hash(model.state_dict())
    if before!=direct["initial_state_SHA256"]:
        raise AssertionError("Arms must share the same synthetic initial state")
    objective_D=dg.FrozenDistributionObjective(fixture["stats"],fixture["cfg"],fixture["directions"],fixture["td"],
                 fixture["sraw"],fixture["rraw"],feature_chunk=64,device=device)
    objective_C=R63.loss.DistributionLoss(fixture["stats"],fixture["cfg"],device)
    layout=dg.ParentLayout.full_reference(fixture["sraw"],fixture["rraw"])
    cached=[]
    with torch.no_grad():
        for lo,hi in fixture["schedule"]:
            out,raw=_forward(R63,model,data,"D",slice(lo,hi))
            expected={k:v[lo:hi] for k,v in direct["endpoint_D"].items()}
            _check_endpoint(torch,R63,out,raw,expected,"collection_vs_direct",checks,lo,hi)
            cached.append(out.detach())
    cache=torch.cat(cached)
    full=adapter.joint_total_D(objective_D,cache,data["reference"],layout)
    gy=full["GY"]
    if not torch.equal(gy.detach().cpu(),direct["GY"]):
        raise AssertionError("Joint endpoint leaf derivative differs despite identical scalar order")
    if float(full["total"].cpu())!=direct["values"]["D"] or {k:float(v.cpu()) for k,v in full["pieces"].items()}!=direct["values"]["D_pieces"]:
        raise AssertionError("Joint D scalar/pieces changed")
    d_gradient=torch.zeros(sum(p.numel() for p in parameters),dtype=torch.float64,device="cpu")
    for lo,hi in fixture["schedule"]:
        out,raw=_forward(R63,model,data,"D",slice(lo,hi))
        expected={k:v[lo:hi] for k,v in direct["endpoint_D"].items()}
        _check_endpoint(torch,R63,out,raw,expected,"replay_vs_direct",checks,lo,hi)
        d_gradient+=parameter_vjp_cpu_f64(out,parameters,gy[lo:hi])
        del out,raw
    out_C,raw_C=_forward(R63,model,data,"C",slice(None))
    _check_endpoint(torch,R63,out_C,raw_C,direct["endpoint_C"],"original_C384_replay",checks)
    result_C=adapter.original_C_microbatch(objective_C,out_C,raw_C,*data["C_args"])
    signal=_C_signal_checks(torch,R63,result_C["pieces"],result_C["diagnostics"],out_C,raw_C,tol["signal"],checks,"replay")
    c_values={k:float(v.detach().cpu()) for k,v in result_C["pieces"].items()}
    if c_values!=direct["values"]["C_pieces"] or float(result_C["total"].detach().cpu())!=direct["values"]["C"]:
        raise AssertionError("Original retained C values changed")
    if {k:float(v.cpu()) for k,v in result_C["discarded_D_values"].items()}!=direct["values"]["discarded_micro_D"]:
        raise AssertionError("Original computed/discarded microbatch D changed")
    c_gradient=scalar_parameter_gradient_cpu_f64(result_C["total"],parameters)
    combined=d_gradient+c_gradient
    if _tensor_hash(model.state_dict())!=before or any(p.grad is not None for p in parameters):
        raise AssertionError("Two-pass accumulation changed theta before the artificial update")
    _gradient_compare(torch,direct["D_gradient"],d_gradient,tol["gradient"],"D_full_VJP_cpu64_accumulation",checks)
    _gradient_compare(torch,direct["C_gradient"],c_gradient,tol["gradient"],"original_C_full_raw_output_graph",checks)
    _gradient_compare(torch,direct["total_gradient"],combined,tol["gradient"],"combined_D_C_before_single_cast",checks)
    update,state=_perform_artificial_update(torch,model,combined,tol,checks,"replay",ledger)
    _gradient_compare(torch,direct["update"]["before_clip"],update["before_clip"],tol["gradient"],"combined_D_C_after_single_cast",checks)
    _gradient_compare(torch,direct["update"]["after_clip"],update["after_clip"],tol["gradient"],"combined_D_C_after_one_clip",checks)
    return dict(initial_state_SHA256=before,D_gradient=d_gradient,C_gradient=c_gradient,total_gradient=combined,
                update=update,optimizer=state,signal=signal,after_state_SHA256=_tensor_hash(model.state_dict()))


def _Adam_formula_check(torch,initial,arm,tol,checks,label):
    opts=tol["optimizer"];bounds=tol["Adam_formula_and_state"]
    eps32=torch.finfo(torch.float32).eps;tiny32=torch.finfo(torch.float32).tiny
    beta1,beta2=opts["betas"];lr=opts["learning_rate"];epsilon=opts["epsilon"]
    results={"parameters":{},"exp_avg":{},"exp_avg_sq":{}}
    allowances={k:{} for k in results}
    rows=[];offset=0;changed=0
    for name,p in initial.named_parameters():
        size=p.numel();g=arm["update"]["after_clip"][offset:offset+size].reshape(p.shape);offset+=size
        p0=p.detach().cpu().double()
        update=lr*g/(g.abs()+epsilon)
        ideals={"parameters":p0-update,"exp_avg":(1-beta1)*g,"exp_avg_sq":(1-beta2)*g.square()}
        row={"name":name,"elements":size}
        if not bool((g!=0).any()) and tol["signal"]["require_each_parameter_tensor_nonzero_gradient"]:
            raise AssertionError("Synthetic check does not exercise parameter tensor: "+name)
        for kind,ideal in ideals.items():
            if kind=="parameters":
                rounded=ideal.to(torch.float32)
                up=torch.nextafter(rounded,torch.full_like(rounded,float("inf")))
                down=torch.nextafter(rounded,torch.full_like(rounded,float("-inf")))
                ulp=torch.maximum((up-rounded).abs(),(rounded-down).abs()).double()
                allowance=bounds["parameter_ulp_multiplier"]*ulp+bounds["float32_epsilon_multiplier"]*eps32*update.abs()+bounds["float32_tiny_multiplier"]*tiny32
            else:
                allowance=bounds["float32_epsilon_multiplier"]*eps32*ideal.abs()+bounds["float32_tiny_multiplier"]*tiny32
            actual=arm["optimizer"][kind][name]
            delta=(actual-ideal).abs()
            if not bool(torch.isfinite(actual).all() and (delta<=allowance).all()):
                raise AssertionError(label+" first-step Adam "+kind+" formula mismatch: "+name)
            results[kind][name]=ideal;allowances[kind][name]=allowance
            row[kind+"_max_formula_error"]=float(delta.max())
            row[kind+"_max_allowed_error"]=float(allowance.max())
        changed+=int((arm["optimizer"]["parameters"][name]!=p0).sum())
        rows.append(row)
    if changed==0 or not all(v==1 for v in arm["optimizer"]["steps"].values()):
        raise AssertionError("Artificial Adam update did not change parameters exactly once")
    checks.append({"name":label+"_all_parameter_Adam_formula","parameter_tensors":len(rows),"parameter_elements":offset,
                   "changed_parameter_elements":changed,"every_named_parameter_included":True,"passed":True})
    return results,allowances,rows


def _Adam_paired_check(torch,initial,direct,replayed,tol,checks):
    ideal_a,allow_a,rows_a=_Adam_formula_check(torch,initial,direct,tol,checks,"direct")
    ideal_b,allow_b,rows_b=_Adam_formula_check(torch,initial,replayed,tol,checks,"replay")
    output=[]
    for kind in ("parameters","exp_avg","exp_avg_sq"):
        raw_max=residual_max=allowed_max=0.;norm2=0.
        for name,_ in initial.named_parameters():
            actual=replayed["optimizer"][kind][name]-direct["optimizer"][kind][name]
            predicted=ideal_b[kind][name]-ideal_a[kind][name]
            allowance=allow_a[kind][name]+allow_b[kind][name]
            residual=(actual-predicted).abs()
            if not bool((residual<=allowance).all()):
                raise AssertionError("Paired sensitivity-aware Adam "+kind+" mismatch: "+name)
            raw_max=max(raw_max,float(actual.abs().max()));residual_max=max(residual_max,float(residual.max()))
            allowed_max=max(allowed_max,float(allowance.max()));norm2+=float(actual.square().sum())
        row={"name":"paired_Adam_"+kind,"raw_difference_L2":norm2**.5,"raw_difference_max_absolute":raw_max,
             "formula_difference_residual_max":residual_max,"max_rounding_allowance":allowed_max,"passed":True,
             "interpretation":"Sensitivity-aware first-step formula check after tight full-gradient comparison; not bitwise equality"}
        checks.append(row);output.append(row)
    return {"direct_all_parameter_rows":rows_a,"replay_all_parameter_rows":rows_b,"paired":output}


def _runtime_preconditions(torch,device):
    if not sys.dont_write_bytecode:
        raise RuntimeError("Launch owning process with python -B")
    if torch.get_default_dtype()!=torch.float32 or str(torch.get_default_device())!="cpu":
        raise RuntimeError("Original float32 default dtype and CPU construction device required")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Ordinary autograd is required")
    if not torch.are_deterministic_algorithms_enabled() or torch.backends.mha.get_fastpath_enabled():
        raise RuntimeError("Caller must enable deterministic algorithms and disable MHA fastpath")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic:
        raise RuntimeError("Caller must install the original deterministic/no-TF32/no-benchmark backend policy")
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("Autocast would change the original numerical path")
    if device.type not in ("cpu","cuda") or (device.type=="cuda" and not torch.cuda.is_available()):
        raise ValueError("Available CPU or CUDA device required")


def run_check(device="cpu", *, objective_adapter_root=None):
    """Run only under the root's supervised server; return a JSON-compatible receipt.

    CPU and CUDA both include the original full384 C network graph. The local
    1GiB preparation process must not execute this callable. On failure the
    exception carries .receipt; this module never writes a result file.
    """
    import numpy as np
    import torch
    started=time.monotonic();checks=[];receipt={"passed":False,"checks":checks,"artificial_only":True,
          "scientific_data_or_weight_files_read":0,"scientific_updates":0,"synthetic_optimizer_steps":0}
    before_state=None;portable=None;dev=torch.device(device)
    if dev.type=="cuda" and dev.index is None:
        dev=torch.device("cuda",torch.cuda.current_device())
    if objective_adapter_root is None:
        objective_adapter_root=ROOT.parent/"objective_adapter"
    try:
        _runtime_preconditions(torch,dev)
        root,tol,spec,manifest,binding_sha=_verified_bundle(objective_adapter_root)
        with _definition_module("_r65_network_bound_portable",root/"bound_r64"/"portable_fixture.py") as portable:
            before_state=portable._state_fingerprint(torch,np,dev)
            with _definition_module("_r65_network_objective_adapter",root/"objective_adapter.py") as adapter:
                with portable._adapter_modules(root/"bound_r64") as (dg,R63,_):
                    with torch.enable_grad():
                        fixture=_fixture(np,R63,spec,dev.type)
                        with torch.random.fork_rng(devices=[]):
                            torch.random.default_generator.manual_seed(spec["model_seed"])
                            initial=R63.network.ResponseNetwork().train()
                        head_rng=np.random.Generator(np.random.PCG64(spec["seeds"]["head"]))
                        with torch.no_grad():
                            head=head_rng.standard_normal(tuple(initial.final.weight.shape)).astype("<f4")*spec["nonzero_head_weight_SD"]
                            bias=head_rng.standard_normal(tuple(initial.final.bias.shape)).astype("<f4")*spec["nonzero_head_bias_SD"]
                            initial.final.weight.copy_(torch.from_numpy(head));initial.final.bias.copy_(torch.from_numpy(bias))
                        if sum(p.numel() for p in initial.parameters())!=spec["model_parameter_count"]:
                            raise AssertionError("Original architecture parameter count changed")
                        initial_sha=_tensor_hash(initial.state_dict())
                        data=_tensor_inputs(torch,fixture,dev)
                        receipt.update(device=str(dev),torch=torch.__version__,numpy=np.__version__,
                            source_binding_sha256=binding_sha,definition_sources=manifest,
                            tolerances_sha256=_sha_bytes((ROOT/"FROZEN_TOLERANCES.json").read_bytes()),
                            fixture_spec_sha256=_sha_bytes((ROOT/"FIXTURE_SPEC.json").read_bytes()),
                            fixture_arrays_sha256=fixture["fixture_SHA256"],synthetic_config_sha256=fixture["cfg_SHA256"],
                            initial_synthetic_state_sha256=initial_sha,fixture=spec[dev.type],
                            all_parameter_count=spec["model_parameter_count"],C_neural_rows=384,
                            D_epsilon_sha256=_array_hash(np,{"epsilon":fixture["arrays"]["epsilon_D"]}),
                            C_SOURCE_epsilon_sha256=_array_hash(np,{"epsilon":fixture["arrays"]["epsilon_C_SOURCE"]}),
                            C_identity_epsilon_sha256=_array_hash(np,{"epsilon":fixture["arrays"]["epsilon_C_identity"]}),
                            C_source_duplicate_occurrences=int(len(fixture["si"])-len(np.unique(fixture["si"]))),
                            frozen_tolerances=tol,training_mode=True,MHA_fastpath=False)
                        direct=_direct_arm(torch,dg,R63,initial,fixture,data,dev,tol,checks,receipt)
                        gc.collect()
                        replayed=_replayed_arm(torch,dg,R63,adapter,initial,fixture,data,dev,tol,checks,direct,receipt)
                        adam=_Adam_paired_check(torch,initial,direct,replayed,tol,checks)
                        for label,endpoint,q in (("D",direct["endpoint_D"],data["D"]["q"].cpu()),("C",direct["endpoint_C"],data["C"]["q"].cpu())):
                            output=endpoint["output"]
                            rail_count=int((~endpoint["free"]).sum())
                            error=float(((R63.core.q_torch(output)-q).abs()/q).max())
                            error32=float(((R63.core.q_torch(output.float().double())-q).abs()/q).max())
                            if rail_count<tol["signal"]["minimum_active_Box_samples"] or max(error,error32)>tol["signal"]["charge_relative_error_max"] or bool((output.abs()>.5).any()):
                                raise AssertionError("Synthetic "+label+" active-Box/charge fixture check failed")
                            checks.append({"name":label+"_active_Box_and_charge","active_rail_samples":rail_count,
                                           "max_relative_charge_f64":error,"max_relative_charge_f32":error32,"passed":True})
                        if _tensor_hash(initial.state_dict())!=initial_sha:
                            raise AssertionError("Synthetic template model was modified")
                        receipt.update(passed=True,status="SYNTHETIC_JOINT_D_C_NETWORK_UPDATE_PASSED",values=direct["values"],
                            Adam_comparison=adam,reference_post_update_sha256=direct["after_state_SHA256"],
                            replay_post_update_sha256=replayed["after_state_SHA256"],
                            reference_clip_norm=direct["update"]["pre_clip_norm"],replay_clip_norm=replayed["update"]["pre_clip_norm"],
                            two_pass_D_parameter_VJPs=len(fixture["schedule"]),two_pass_C_scalar_reverse_passes=1,
                            reference_single_scalar="joint full D scalar + sum of four original C pieces",
                            comparison_boundary="Same inputs/state/device/chunks; reordered float32 parameter gradients use frozen R64 tolerance. No R64 separately differentiated five-piece bitwise claim.",
                            scientific_training_updates=0)
            after_state=portable._state_fingerprint(torch,np,dev)
            receipt.update(caller_state_before=before_state,caller_state_after=after_state,caller_state_preserved=before_state==after_state)
            if before_state!=after_state:
                raise AssertionError("Artificial check altered caller RNG/backend/thread state")
        _,_,_,after_manifest,after_binding=_verified_bundle(root)
        if after_manifest!=manifest or after_binding!=binding_sha:
            raise AssertionError("Definition/tolerance binding changed during artificial execution")
        receipt.update(elapsed_seconds=time.monotonic()-started,definition_sources_unchanged=True,
                       compact={"passed":True,"device":str(dev),"checks":len(checks),"synthetic_optimizer_steps":2,
                                "scientific_updates":0,"all_parameters":spec["model_parameter_count"]})
        return receipt
    except BaseException as exc:
        receipt.update(passed=False,status="SYNTHETIC_JOINT_D_C_NETWORK_UPDATE_FAILED",error_type=type(exc).__name__,error=str(exc),
                       traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        if before_state is not None and portable is not None:
            receipt["caller_state_after_failure"]=portable._state_fingerprint(torch,np,dev)
            receipt["caller_state_preserved"]=before_state==receipt["caller_state_after_failure"]
        raise ArtificialNetworkCheckFailure(receipt) from exc


if __name__=="__main__":
    raise RuntimeError("Import run_check under the owned server supervisor; local preparation must only run static_check.py")
