from __future__ import annotations
import pathlib,json,hashlib,copy,time,datetime,threading,os,math
import numpy as np
import torch
from model import Response,observables,distribution_loss
import local_joint as lj
import energy_width as ew
import evaluation as ev

def digest_tensors(values):
    h=hashlib.sha256()
    for v in values:h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

def parameters_equal(model,state):return all(torch.equal(p.detach(),state[n]) for n,p in model.named_parameters())

def make_initial(pack,cfg,root):
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    before=torch.get_rng_state().clone()
    constructor=pack['constructor'];model=Response(**constructor,signal_enabled=False)
    status=model.load_state_dict(pack['state_dict'],strict=False)
    assert set(status.missing_keys)=={'signal_w','signal_b','signal_a'} and not status.unexpected_keys
    assert all(torch.equal(model.state_dict()[k],v) for k,v in pack['state_dict'].items())
    state=copy.deepcopy(model.state_dict())
    torch.save({'state_dict':state,'constructor':constructor,'config':cfg,'external_R3B_sha256':cfg['input_files']['model']['sha256']},root/'COMMON_INITIAL_MODEL.pt')
    torch.save({'before_first_constructor':before,'after_first_constructor':torch.get_rng_state(),'seed':cfg['seed'],'numpy_state':np.random.get_state()[1]},root/'INITIAL_RANDOM_STATES.pt')
    return state,constructor

def targets(real_fit,real_val,edges,cfg,root):
    fit=torch.tensor(real_fit);val=torch.tensor(real_val)
    fs,_,_=observables(fit)
    scale=(torch.quantile(fs,.75,dim=0)-torch.quantile(fs,.25,dim=0)).clamp_min(torch.tensor([.05,.5,.5,1.,.05,.02,.2,.005]))
    g=torch.Generator().manual_seed(cfg['old_projection_seed']);projection=torch.randn((480,32),generator=g)
    projection=torch.nn.functional.avg_pool1d(projection.T[:,None,:],5,stride=1,padding=2).squeeze(1).T
    projection/=torch.linalg.vector_norm(projection,dim=0)[None,:];projection[:100]*=.1
    centers=torch.tensor((edges[:-1]+edges[1:])/2,dtype=torch.float32);width=float(edges[1]-edges[0])
    tf=ew.conditional_targets(fit,centers,width);tv=ew.conditional_targets(val,centers,width)
    local=lj.fit_targets(fit,cfg['joint_projection_seed'])
    torch.save({'scale':scale,'old_projection':projection,'conditional_fit':tf,'conditional_validation':tv,'local':local},root/'TRAIN_TARGETS.pt')
    compact={'fit_n':len(fit),'validation_n':len(val),'scale':scale.numpy(),'edges_logQ':edges,'conditional_fit':tf,'conditional_validation':tv,
             'local_regions':[{'region':r['region'],'vector_dimension':len(r['mean']),'block_dims':r['block_dims'],'floor_affected_count':r['floor_affected_count'],'floor_affected_per_block':r['floor_affected_per_block'],'fit_n':r['fit_n'],'projection_sha256':digest_tensors([r['projection']]),'mean':r['mean'].numpy(),'raw_IQR':r['raw_IQR'].numpy(),'IQR':r['IQR'].numpy()} for r in local],
             'TRAIN_only_scale_and_conditional_targets':True,'no_DEV_or_EVAL_selection':True}
    (root/'TARGET_STATISTICS.json').write_text(json.dumps(ev.safe(compact),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    return fit,val,scale,projection,tf,tv,local

def train_arm(name,state,constructor,data,target,cfg,root,log):
    enabled=name=='JOINT_SIGNAL';path=root/'arms'/name;path.mkdir(parents=True,exist_ok=False);(path/'checkpoints').mkdir()
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    model=Response(**constructor,signal_enabled=enabled);model.load_state_dict(state,strict=True);assert parameters_equal(model,state)
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    pack=lambda sd,step:{'state_dict':sd,'constructor':constructor,'signal_enabled':enabled,'best_update':step,'config':cfg,'external_R3B_sha256':cfg['input_files']['model']['sha256']}
    torch.save(pack(copy.deepcopy(model.state_dict()),0),path/'initial_model.pt')
    fit,val,scale,projection,tf,tv,local=target;c,q,gt=data
    ct=torch.tensor(c[gt]);qt=torch.tensor(q[gt]);cv=torch.tensor(c[~gt]);qv=torch.tensor(q[~gt])
    gen=torch.Generator().manual_seed(cfg['seed']);vgen=torch.Generator().manual_seed(cfg['validation_seed']);sg=torch.Generator().manual_seed(cfg['signal_validation_seed'])
    nv=cfg['validation_n'];vi=torch.randperm(len(cv),generator=vgen)[:nv];ri=torch.randperm(len(val),generator=vgen)[:nv]
    vz=torch.randn((nv,8),generator=vgen);vw=torch.randn((nv,512),generator=vgen);vs=torch.randn((nv,512),generator=sg)
    torch.save({'training_before_draw':gen.get_state(),'validation_after_draw':vgen.get_state(),'signal_validation_after_draw':sg.get_state(),'global_after_constructor':torch.get_rng_state(),'seed_before_constructor':cfg['seed']},path/'INITIAL_RNG.pt')
    np.savez_compressed(path/'VALIDATION_INPUTS.npz',source_internal_validation_indices=vi.numpy(),REAL_validation_indices=ri.numpy(),z=vz.numpy(),baseline_white=vw.numpy(),signal_white=vs.numpy())
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    arm_start=time.monotonic();trace=[];sample_digests=[];updates=0
    def wall_abort():
        (path/'HARD_WALL_FAILURE.json').write_text(json.dumps({'PID':os.getpid(),'arm':name,'wall_limit_s':cfg['training_wall_s_per_arm'],'actual_training_wall_s':time.monotonic()-arm_start,'updates':updates,'status':'native_hard_timer_abort'})+'\n',encoding='utf-8')
        os._exit(124)
    timer=threading.Timer(cfg['training_wall_s_per_arm'],wall_abort);timer.daemon=True;timer.start()
    def validation():
        model.eval()
        with torch.no_grad():
            fake,_=model(cv[vi],qv[vi],vz,vw,vs)
            loss,parts=distribution_loss(fake,val[ri],scale,projection)
            extra,cp=ew.conditional_loss(fake,tv);new,jp=lj.loss(fake,val[ri],local)
            loss=loss+extra+new;parts.update(cp);parts.update(jp);parts['new_joint']=new
        model.train()
        if not torch.isfinite(loss):raise FloatingPointError('validation nonfinite')
        return float(loss),{k:float(v) for k,v in parts.items()}
    def keep_record(record):
        trace.append(record)
        (path/'TRAINING_TRACE.json').write_text(json.dumps(trace,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        torch.save(pack(copy.deepcopy(model.state_dict()),record['update']),path/'checkpoints'/('update_'+str(record['update']).zfill(6)+'.pt'))
    try:
        best,vp=validation();beststep=0;beststate=copy.deepcopy(model.state_dict());keep_record({'update':0,'validation':best,'validation_parts':vp,'training_elapsed_s':time.monotonic()-arm_start})
        log(f'{name} initial validation={best:.6f}; G4fit={len(ct)} REALfit={len(fit)} val={len(val)}')
        for step in range(1,cfg['max_updates_per_arm']+1):
            if time.monotonic()-arm_start>=cfg['training_wall_s_per_arm']-5:log(name+' training wall reserve reached');break
            bs=cfg['batch_size'];ii=torch.randint(len(ct),(bs,),generator=gen);jj=torch.randint(len(fit),(bs,),generator=gen)
            z=torch.randn((bs,8),generator=gen);white=torch.randn((bs,512),generator=gen);signal=torch.randn((bs,512),generator=gen)
            sample_digests.append(digest_tensors([ii,jj,z,white,signal]))
            fake,m=model(ct[ii],qt[ii],z,white,signal)
            loss,parts=distribution_loss(fake,fit[jj],scale,projection)
            fs,_,_=observables(fake);ch=ct[ii,1]-ct[ii,1].mean();ps=fs[:,0]-fs[:,0].mean();composition_cov=(ch*ps).mean()
            conditional=.2*torch.relu(.01-composition_cov)
            smooth=.002*model.last_time_coefficients.diff(n=2,dim=1).square().mean()
            regular=.0001*sum(p.square().mean() for p in list(model.mean.parameters())+list(model.random.parameters()))
            extra,cp=ew.conditional_loss(fake,tf);new,jp=lj.loss(fake,fit[jj],local)
            total=loss+conditional+smooth+regular+extra+new
            if not torch.isfinite(total):raise FloatingPointError(name+' nonfinite total')
            opt.zero_grad(set_to_none=True);total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError(name+' nonfinite gradient')
            if not enabled:assert all(p.grad is None for n,p in model.named_parameters() if n.startswith('signal_'))
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();updates=step
            if step%100==0:
                vl,vp=validation();record={'update':step,'training_loss':float(total.detach()),'training_old_parts':{k:float(v.detach()) for k,v in parts.items()},'training_conditional_parts':{k:float(v.detach()) for k,v in cp.items()},'training_joint_parts':{k:float(v.detach()) for k,v in jp.items()},'new_joint':float(new.detach()),'composition_cov':float(composition_cov.detach()),'validation':vl,'validation_parts':vp,'training_elapsed_s':time.monotonic()-arm_start}
                keep_record(record)
                if vl<best:best,beststep,beststate=vl,step,copy.deepcopy(model.state_dict())
                log(f'{name} update={step} loss={record["training_loss"]:.5f} validation={vl:.5f} best={beststep} wall={record["training_elapsed_s"]:.1f}s')
        if updates%100:
            vl,vp=validation();keep_record({'update':updates,'validation':vl,'validation_parts':vp,'wall_stop':True,'training_elapsed_s':time.monotonic()-arm_start})
            if vl<best:best,beststep,beststate=vl,updates,copy.deepcopy(model.state_dict())
        torch.save(pack(copy.deepcopy(model.state_dict()),updates),path/'final_model.pt')
        torch.save({'optimizer':opt.state_dict(),'training_final':gen.get_state(),'validation_after_draw':vgen.get_state(),'signal_validation_after_draw':sg.get_state(),'global_final':torch.get_rng_state(),'sample_digests':sample_digests},path/'FINAL_RNG_OPTIMIZER.pt')
        (path/'SAMPLING_DIGESTS.json').write_text(json.dumps(sample_digests,indent=2)+'\n',encoding='utf-8')
        model.load_state_dict(beststate);model.eval();torch.save(pack(beststate,beststep),path/'model.pt')
        for n,v in model.state_dict().items():assert torch.isfinite(v).all(),n
        if not enabled:assert all(torch.equal(model.state_dict()[n],state[n]) for n in ['signal_b','signal_w','signal_a'])
        elapsed=time.monotonic()-arm_start;assert elapsed<cfg['training_wall_s_per_arm']
        receipt={'arm':name,'signal_enabled':enabled,'optimizer_updates':updates,'selected_update':beststep,'initial_validation':trace[0]['validation'],'best_validation':best,'training_wall_s':elapsed,'initial_all_parameters_and_buffers_equal':True,'initial_parameter_sha256':digest_tensors([state[n] for n,_ in model.named_parameters()]),'validation_input_sha256':digest_tensors([vi,ri,vz,vw,vs]),'paired_training_prefix_digests':len(sample_digests),'new_disabled_parameters_no_gradient_and_unchanged':not enabled,'signal_w':model.signal_w.detach().numpy(),'signal_b':float(model.signal_b.detach()),'signal_a':model.signal_a.detach().numpy(),'signal_FIR_L2_kernel':model.signal_kernel().detach().numpy(),'all_parameters_finite':True,'no_DEV_selection':True,'explicit_seed_before_model_constructor':cfg['seed'],'validation_sample_n':nv,'source_fit_n':len(ct),'REAL_fit_n':len(fit),'REAL_validation_n':len(val)}
        (path/'TRAINING_RECEIPT.json').write_text(json.dumps(ev.safe(receipt),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        return model,receipt,sample_digests
    finally:timer.cancel()
