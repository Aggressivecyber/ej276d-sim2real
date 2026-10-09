from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import pathlib, sys, json, time, datetime, signal, traceback, platform, resource, argparse
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'deps'))
os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
os.environ['JOBLIB_TEMP_FOLDER'] = str(ROOT/'tmp')
import numpy as np
import torch
import scipy, sklearn
from flow_model import make_density, make_transform, symmetric_loss, engineering_checks
from r3b_model import SmoothResponse
import evaluation as ev

CFG = json.loads((ROOT/'config.json').read_text())
OUT = ROOT/'results'
OUT.mkdir(exist_ok=True)
START = time.monotonic()
DEVICE = torch.device('cuda:0')
COUNTERS = {'EVAL_read':0,'new_G4':0,'source_updates':0,'real_updates':0,'transport_updates':0,
            'R3B_rows':0,'transport_rows':0,'HGB_attempts':0,'density_description_slots':0}
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(CFG['gpu_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)


def write(path, obj):
    target = ROOT/path
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix+'.tmp')
    temp.write_text(json.dumps(ev.safe(obj), ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    temp.replace(target)


def log(message):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), message, flush=True)


def guard():
    if time.monotonic()-START > CFG['numeric_wall_limit_s']-60:
        raise TimeoutError('numeric budget exhausted; retain all current states')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if rss > CFG['rss_limit_bytes']:
        raise MemoryError('process RSS limit exceeded')


def record_budget(signum, frame):
    raise TimeoutError('hard numeric time limit')


def read_real():
    base = pathlib.Path(CFG['remote_input_root'])
    v = np.fromfile(base/'REAL_V', dtype='<f4', count=5200*480).reshape(5200,480)
    qc = np.fromfile(base/'REAL_QC', dtype='<u2', count=5200)
    groups = np.fromfile(base/'REAL_groups', dtype='<u2', count=5200)
    original = np.fromfile(base/'REAL_original_index', dtype='<u4', count=5200)
    roles = np.fromfile(base/'REAL_roles', dtype='u1', count=5200)
    vg = json.loads((base/'TRAIN_roles').read_text())['REAL_internal_validation_artificial_groups']
    fit = (qc[2000:]==0)&~np.isin(groups[2000:],vg)
    val = (qc[2000:]==0)&np.isin(groups[2000:],vg)
    assert fit.sum()==2592 and val.sum()==599
    assert np.all(roles[:2000]==0) and np.all(roles[2000:]==1)
    assert np.array_equal(groups, original//50)
    assert not set(groups[:2000])&set(groups[2000:])
    assert not set(groups[2000:][fit])&set(groups[2000:][val])
    return v, qc, groups, original, fit, val


def load_r3b():
    # This is the project's own explicitly selected R3B checkpoint, with numpy
    # constructor arrays in its trusted legacy serialization.
    pack = torch.load(ROOT/'inputs/r3b_model.pt', map_location='cpu', weights_only=False)
    m = SmoothResponse(**pack['constructor'])
    m.load_state_dict(pack['state_dict'])
    m.eval()
    m.requires_grad_(False)
    return m


def r3b_predict(model, c, q, z, white):
    out, meta = [], {}
    with torch.no_grad():
        for first in range(0,len(c),256):
            stop = first+256
            v,m = model(torch.as_tensor(c[first:stop]), torch.as_tensor(q[first:stop]),
                        torch.as_tensor(z[first:stop]), torch.as_tensor(white[first:stop]))
            out.append(v.numpy())
            for key in ['padding','quantization_clip']:
                meta.setdefault(key,[]).append(m[key].numpy())
    COUNTERS['R3B_rows'] += len(c)
    return np.concatenate(out), {k:int(np.concatenate(v).sum()) for k,v in meta.items()}


def random_inputs(n, seed):
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(n,8,generator=rng).numpy(), torch.randn(n,512,generator=rng).numpy()


def prepare_training():
    v,qc,groups,original,rf,rv = read_real()
    with np.load(pathlib.Path(CFG['remote_input_root'])/'G4_TRAIN_cache', allow_pickle=False) as z:
        c=z['condition'].copy(); q=z['Qsource_Vns'].copy(); gf=z['internal_training'].copy(); ids=z['event_ids'].copy()
    with np.load(ROOT/'inputs/r3b_dev_inputs.npz',allow_pickle=False) as z:
        dev_ids=z['sim_event_ids'].copy()
    assert gf.sum()==6593 and (~gf).sum()==1599 and len(ids)==8192
    assert not set(ids)&set(dev_ids)
    torch.manual_seed(CFG['seed'])
    model = load_r3b()
    latent,white = random_inputs(len(c),CFG['source_generation_seed'])
    source,meta = r3b_predict(model,c,q,latent,white)
    realfit,realval = v[2000:][rf],v[2000:][rv]
    center = realfit.astype('f8').mean(axis=0).astype('f4')
    sd = realfit.astype('f8').std(axis=0,ddof=1)
    scale = np.maximum(sd, CFG['scale_floor_V']).astype('f4')
    np.savez_compressed(OUT/'TRAIN_PRIVATE.npz',source=source,G4_fit=gf,source_ids=ids,
                        realfit=realfit,realval=realval,center=center,scale=scale,
                        real_fit_index=original[2000:][rf],real_val_index=original[2000:][rv])
    np.savez_compressed(OUT/'SOURCE_TRAIN_RANDOM_PRIVATE.npz',latent=latent,white=white)
    write('DATA_ROLES.json',{'REAL_fit':len(realfit),'REAL_val':len(realval),'REAL_DEV':2000,
          'G4_fit':int(gf.sum()),'G4_val':int((~gf).sum()),'G4_DEV':2000,
          'fit_val_and_DEV_source_groups_disjoint':True,'all_real_from_same_acquisition':True,
          'DEV_historically_exposed_not_blind':True,'EVAL_read':0,'new_G4':0,
          'scale_floored_coordinates':int((sd<CFG['scale_floor_V']).sum()),'source_training_generation':meta})
    return source[gf], source[~gf], realfit,realval,center,scale


def normalize(v, center, scale):
    return torch.as_tensor((v-center)/scale, device=DEVICE)


def fixed_validation(v, center, scale, seed):
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-CFG['jitter_halfwidth_V'],CFG['jitter_halfwidth_V'],v.shape).astype('f4')
    return normalize(v+jitter,center,scale)


def density_validation(model, x):
    model.eval()
    total=0.
    with torch.no_grad():
        for first in range(0,len(x),CFG['validation_chunk']):
            a=x[first:first+CFG['validation_chunk']]
            total += float((-model.log_prob(a)).sum())
    return total/len(x)/480


def save_fit(name, model, optimizer, records, selected, update, started):
    torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state(),
                'selected_update':selected,'last_update':update,'model_config':CFG['model']},OUT/(name+'_last.pt'))
    write('results/'+name+'_training.json',{'name':name,'selected_update':selected,
          'actual_updates':update,'elapsed_s':time.monotonic()-started,'trace':records,
          'selection':'minimum all-record TRAIN-validation nats per dimension only',
          'early_stopping_patience_checks':CFG['patience_checks']})


def fit_density(name, fit, val, scale, seed):
    torch.manual_seed(seed)
    model=make_density(CFG['model']).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
    rng=torch.Generator(device=DEVICE).manual_seed(seed+1)
    scales=torch.as_tensor(scale,device=DEVICE)
    started=time.monotonic(); initial=density_validation(model,val)
    if not np.isfinite(initial):raise FloatingPointError(name+' initial likelihood')
    best=initial;selected=0;stale=0;update=0
    records=[{'step':0,'validation_nats_per_dim':initial,'elapsed_s':0.}]
    torch.save(model.state_dict(),OUT/(name+'_best.pt'))
    log(f'{name} initial validation {initial:.6f}')
    for step in range(1,CFG['density_max_updates']+1):
        guard()
        if time.monotonic()-started>=CFG['density_stage_s']-15:break
        model.train()
        ix=torch.randint(len(fit),(CFG['batch_size'],),device=DEVICE,generator=rng)
        jitter=(torch.rand((len(ix),480),device=DEVICE,generator=rng)*2-1)*CFG['jitter_halfwidth_V']/scales
        x=fit[ix]+jitter
        optimizer.zero_grad(set_to_none=True)
        loss=-model.log_prob(x).mean()/480
        if not bool(torch.isfinite(loss)):raise FloatingPointError(name+' nonfinite loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['gradient_norm'],error_if_nonfinite=True)
        optimizer.step();update=step;COUNTERS[name+'_updates']=step
        if step%CFG['validation_interval']==0 or step==CFG['density_max_updates']:
            value=density_validation(model,val)
            records.append({'step':step,'train_nats_per_dim':float(loss.detach()),
                            'validation_nats_per_dim':value,'grad_norm_before_clip':float(norm),'elapsed_s':time.monotonic()-started})
            if not np.isfinite(value):raise FloatingPointError(name+' nonfinite validation')
            if value<best:
                best=value;selected=step;stale=0;torch.save(model.state_dict(),OUT/(name+'_best.pt'))
            else:stale+=1
            save_fit(name,model,optimizer,records,selected,update,started)
            write('PROGRESS.json',{'phase':name,'step':step,'selected':selected,'validation':value,
                                  'elapsed_s':time.monotonic()-START,'counters':COUNTERS})
            log(f'{name} step={step} val={value:.6f} best={best:.6f}')
            if stale>=CFG['patience_checks']:break
    save_fit(name,model,optimizer,records,selected,update,started)
    model.load_state_dict(torch.load(OUT/(name+'_best.pt'),map_location=DEVICE,weights_only=True))
    model.eval();model.requires_grad_(False)
    return model


def transfer_validation(transform, source, real, sv, rv):
    transform.eval();a=b=0.
    with torch.no_grad():
        for first in range(0,len(sv),CFG['validation_chunk']):
            x=sv[first:first+CFG['validation_chunk']];tx,ld=transform(x)
            a+=float((-(real.log_prob(tx)+ld)).sum())
        for first in range(0,len(rv),CFG['validation_chunk']):
            y=rv[first:first+CFG['validation_chunk']];iy,ld=transform.inverse(y)
            b+=float((-(source.log_prob(iy)+ld)).sum())
    return .5*(a/len(sv)+b/len(rv))/480


def fit_transfer(source,real,sf,rf,sv,rv,scale):
    torch.manual_seed(CFG['seed']+30)
    model=make_transform(CFG['model']).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
    rng=torch.Generator(device=DEVICE).manual_seed(CFG['seed']+31)
    scales=torch.as_tensor(scale,device=DEVICE)
    started=time.monotonic();best=transfer_validation(model,source,real,sv,rv)
    selected=0;stale=0;update=0
    records=[{'step':0,'validation_nats_per_dim':best,'elapsed_s':0.}]
    torch.save(model.state_dict(),OUT/'transport_best.pt')
    log(f'transport initial validation {best:.6f}')
    for step in range(1,CFG['transport_max_updates']+1):
        guard()
        if time.monotonic()-started>=CFG['transport_stage_s']-20:break
        if time.monotonic()-START>=CFG['numeric_wall_limit_s']-CFG['evaluation_reserve_s']:break
        model.train();n=CFG['batch_size']
        ix=torch.randint(len(sf),(n,),device=DEVICE,generator=rng)
        iy=torch.randint(len(rf),(n,),device=DEVICE,generator=rng)
        jitter=(torch.rand((2,n,480),device=DEVICE,generator=rng)*2-1)*CFG['jitter_halfwidth_V']/scales
        x=sf[ix]+jitter[0];y=rf[iy]+jitter[1]
        optimizer.zero_grad(set_to_none=True)
        loss,a,b=symmetric_loss(model,source,real,x,y)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('transport nonfinite loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['gradient_norm'],error_if_nonfinite=True)
        optimizer.step();update=step;COUNTERS['transport_updates']=step
        if step%CFG['validation_interval']==0 or step==CFG['transport_max_updates']:
            value=transfer_validation(model,source,real,sv,rv)
            if not np.isfinite(value):raise FloatingPointError('transport nonfinite validation')
            records.append({'step':step,'train_nats_per_dim':float(loss.detach()),
                            'forward_term':float(a.detach()),'inverse_term':float(b.detach()),
                            'validation_nats_per_dim':value,'grad_norm_before_clip':float(norm),'elapsed_s':time.monotonic()-started})
            if value<best:
                best=value;selected=step;stale=0;torch.save(model.state_dict(),OUT/'transport_best.pt')
            else:stale+=1
            save_fit('transport',model,optimizer,records,selected,update,started)
            write('PROGRESS.json',{'phase':'transport','step':step,'selected':selected,'validation':value,
                                  'elapsed_s':time.monotonic()-START,'counters':COUNTERS})
            log(f'transport step={step} val={value:.6f} best={best:.6f}')
            if stale>=CFG['patience_checks']:break
    save_fit('transport',model,optimizer,records,selected,update,started)
    model.load_state_dict(torch.load(OUT/'transport_best.pt',map_location=DEVICE,weights_only=True))
    model.eval();model.requires_grad_(False)
    return model


def transform_voltage(model, v, center, scale, jitter):
    output=[]
    with torch.no_grad():
        for first in range(0,len(v),CFG['generation_chunk']):
            sl=slice(first,first+CFG['generation_chunk'])
            x=normalize(v[sl]+jitter[sl],center,scale)
            a,_=model(x)
            output.append(a.cpu().numpy()*scale+center)
    pre=np.concatenate(output).astype('f4')
    if not np.isfinite(pre).all():raise FloatingPointError('nonfinite generated voltage')
    raw=np.rint((pre.astype('f8')+.5)*16383.)
    post=(raw.clip(0,16383)/16383.-.5).astype('f4')
    COUNTERS['transport_rows']+=len(v)
    return pre,post,{'ADC_clipped_values':int(((raw<0)|(raw>16383)).sum()),
                    'ADC_clipped_rows':int(((raw<0)|(raw>16383)).any(axis=1).sum()),'draws':len(v)}


def generate(model,center,scale):
    with np.load(ROOT/'inputs/r3b_dev_inputs.npz',allow_pickle=False) as f:
        c=f['G4_condition'].copy();q=f['Qsource_Vns'].copy();ids=f['sim_event_ids'].copy()
    z,w=random_inputs(2000,CFG['dev_generation_seed'])
    frozen=load_r3b()
    baseline,meta=r3b_predict(frozen,c,q,z,w)
    rng=np.random.default_rng(CFG['seed']+40)
    jitter=rng.uniform(-CFG['jitter_halfwidth_V'],CFG['jitter_halfwidth_V'],baseline.shape).astype('f4')
    pre,post,adc=transform_voltage(model,baseline,center,scale,jitter)
    np.savez_compressed(OUT/'GENERATION_INPUTS_PRIVATE.npz',condition=c,qsource=q,ids=ids,z=z,white=w,jitter=jitter)
    np.savez_compressed(OUT/'generated.npz',R3B_V=baseline,pre_ADC_V=pre,NSF_V=post,ids=ids)
    ix=np.floor(np.arange(64)*1999/63).astype(int)
    perm=np.random.default_rng(CFG['seed']+41).permutation(64)
    xp,mp=r3b_predict(frozen,c[ix][perm],q[ix][perm],z[ix],w[ix])
    pp,yp,ap=transform_voltage(model,xp,center,scale,jitter[ix])
    zr,wr=random_inputs(64,CFG['seed']+42)
    xr,mr=r3b_predict(frozen,c[ix],q[ix],zr,wr)
    jr=rng.uniform(-CFG['jitter_halfwidth_V'],CFG['jitter_halfwidth_V'],xr.shape).astype('f4')
    pr,yr,ar=transform_voltage(model,xr,center,scale,jr)
    np.savez_compressed(OUT/'SENSITIVITY_PRIVATE.npz',indices=ix,permutation=perm,R3B_permuted=xp,NSF_permuted=yp,
                        R3B_fresh=xr,NSF_fresh=yr,z_fresh=zr,white_fresh=wr,jitter_fresh=jr)
    write('GENERATION_RECEIPT.json',{'main_R3B':meta,'main_NSF':adc,'permuted_NSF':ap,'fresh_NSF':ar,
          'source_sensitivity_rows':64,'fresh_noise_rows':64,'new_independent_parents':0,
          'no_rejection_or_CFD_recrop':True,'main_parent_count':len(ids)})


def main(mode):
    signal.signal(signal.SIGALRM,record_budget)
    signal.setitimer(signal.ITIMER_REAL,CFG['numeric_wall_limit_s'])
    log(f'RNSF {mode} pid={os.getpid()} device={torch.cuda.get_device_name(0)}')
    if mode=='smoke':
        write('ENGINEERING_CHECKS.json',engineering_checks(CFG,DEVICE))
    elif mode=='train':
        write('ENGINEERING_CHECKS.json',engineering_checks(CFG,DEVICE))
        sf,sv,rf,rv,center,scale=prepare_training()
        validation_source=fixed_validation(sv,center,scale,CFG['seed']+11)
        validation_real=fixed_validation(rv,center,scale,CFG['seed']+21)
        sf=normalize(sf,center,scale);rf=normalize(rf,center,scale)
        source=fit_density('source',sf,validation_source,scale,CFG['seed']+10)
        real=fit_density('real',rf,validation_real,scale,CFG['seed']+20)
        transport=fit_transfer(source,real,sf,rf,validation_source,validation_real,scale)
        np.savez_compressed(OUT/'standardization.npz',center=center,scale=scale)
        write('MODEL_FROZEN.json',{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'checkpoint':'results/transport_best.pt','DEV_used_for_training_or_selection':False,
              'counters':COUNTERS})
        # Scientific success is assessed only after all selection is frozen.
        del source,real,sf,rf,validation_source,validation_real
        torch.cuda.empty_cache()
        generate(transport,center,scale)
        from assess_trial import evaluate
        evaluate(CFG,ROOT,COUNTERS)
    elif mode=='replay':
        with np.load(OUT/'standardization.npz') as z:center=z['center'];scale=z['scale']
        with np.load(OUT/'GENERATION_INPUTS_PRIVATE.npz',allow_pickle=False) as z:inputs={k:z[k] for k in z.files}
        with np.load(OUT/'generated.npz',allow_pickle=False) as z:expected={k:z[k] for k in z.files}
        m=make_transform(CFG['model']).to(DEVICE)
        m.load_state_dict(torch.load(OUT/'transport_best.pt',map_location=DEVICE,weights_only=True));m.eval()
        source,_=r3b_predict(load_r3b(),inputs['condition'],inputs['qsource'],inputs['z'],inputs['white'])
        pre,post,_=transform_voltage(m,source,center,scale,inputs['jitter'])
        error=float(np.max(np.abs(pre-expected['pre_ADC_V'])))
        result={'R3B_exact':bool(np.array_equal(source,expected['R3B_V'])),
                'pre_ADC_maxabs_V':error,'pre_ADC_atol_V':1e-7,
                'post_ADC_exact':bool(np.array_equal(post,expected['NSF_V'])),
                'all2000_end_to_end_replayed':True,'values':int(post.size),'metric_refits':0}
        write('REPLAY_RECEIPT.json',result)
        assert result['R3B_exact'] and error<=1e-7 and result['post_ADC_exact'],result
    else:
        from assess_trial import evaluate
        evaluate(CFG,ROOT,COUNTERS)
    torch.cuda.synchronize()
    write('NATIVE_'+mode.upper()+'.json',{'mode':mode,'PID':os.getpid(),'status':'complete',
          'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
          'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),
          'peak_GPU_reserved_bytes':torch.cuda.max_memory_reserved(),'counters':COUNTERS,
          'versions':{'Python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                      'scipy':scipy.__version__,'sklearn':sklearn.__version__,'nflows':'0.14'},
          'device':torch.cuda.get_device_name(0)})
    log(mode+' complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','train','replay','evaluate'])
    mode=parser.parse_args().mode
    try:main(mode)
    except BaseException as exc:
        write('FAILURE_'+mode+'_'+datetime.datetime.now().strftime('%H%M%S')+'.json',{
              'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,
              'PID':os.getpid(),'counters':COUNTERS})
        raise
