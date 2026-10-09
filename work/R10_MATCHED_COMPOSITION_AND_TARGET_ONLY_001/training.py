from __future__ import annotations
import copy,hashlib,json,math,pathlib,time,os
import numpy as np
import torch
from torch.nn import functional as F
from model import WholeResponse,Critic,observables,train_initialization
import energy_width as ew
import evaluation as ev

def digest(values):
    h=hashlib.sha256()
    for v in values:
        if isinstance(v,torch.Tensor):v=v.detach().contiguous().numpy()
        h.update(np.asarray(v).tobytes())
    return h.hexdigest()

def state_digest(state):return digest([state[k] for k in sorted(state)])

def joint_blocks(v):
    scalar,shape,_=observables(v)
    return [scalar[:,:5],v/.1,shape,torch.diff(v,dim=1)/.1,torch.diff(v,n=2,dim=1)/.1]

def baseline_blocks(v):
    b=v[:,:100]-v[:,:100].mean(dim=1,keepdim=True);den=(b*b).sum(dim=1)
    rms=b.std(dim=1)
    acf=torch.stack([(b[:,k:]*b[:,:-k]).sum(dim=1)/den.clamp_min(1e-12) for k in range(1,9)],dim=1)
    power=torch.fft.rfft(b,dim=1).abs().square()[:,1:]
    power=power/power.sum(dim=1,keepdim=True).clamp_min(1e-12)
    scalar,_,_=observables(v)
    return [rms.clamp_min(1e-7).log()[:,None],acf,power,scalar[:,[1,2]]]

def scaled_vector(blocks,definition):
    return torch.cat([(b-r['mean'])/r['scale']/math.sqrt(b.shape[1]) for b,r in zip(blocks,definition)],dim=1)

def make_scales(blocks,floors):
    out=[]
    for b,floor in zip(blocks,floors):
        raw=torch.quantile(b,.75,dim=0)-torch.quantile(b,.25,dim=0)
        floor=torch.as_tensor(floor,dtype=b.dtype);scale=torch.maximum(raw,floor)
        out.append({'mean':b.mean(dim=0),'raw_IQR':raw,'scale':scale,'floor':floor,
                    'dim':b.shape[1],'floor_affected_coordinates':int((raw<floor).sum())})
    return out

def projected_loss(a,b,projection):
    aa=(a@projection).sort(dim=0).values;bb=(b@projection).sort(dim=0).values
    if len(bb)!=len(aa):
        pos=torch.linspace(0,len(bb)-1,len(aa),device=bb.device);lo=pos.floor().long();hi=pos.ceil().long();t=(pos-lo)[:,None]
        bb=bb[lo]*(1-t)+bb[hi]*t
    return (aa-bb).abs().mean()

def stats_json(x):
    if isinstance(x,torch.Tensor):return x.detach().numpy().tolist()
    if isinstance(x,dict):return {k:stats_json(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [stats_json(v) for v in x]
    return x

def make_targets(realfit,realval,edges,cfg,root):
    fit=torch.tensor(realfit);val=torch.tensor(realval)
    jd=make_scales(joint_blocks(fit),[[.05,.5,.5,1.,.05],.005,.02,.005,.005])
    bd=make_scales(baseline_blocks(fit),[.2,.02,.005,[.5,.5]])
    gen=torch.Generator().manual_seed(cfg['projection_seed'])
    jp=torch.randn((sum(s['dim'] for s in jd),128),generator=gen);jp/=torch.linalg.vector_norm(jp,dim=0)[None,:]
    bp=torch.randn((sum(s['dim'] for s in bd),128),generator=gen);bp/=torch.linalg.vector_norm(bp,dim=0)[None,:]
    centers=torch.tensor((edges[:-1]+edges[1:])/2,dtype=torch.float32);width=float(edges[1]-edges[0])
    target={'joint_definition':jd,'baseline_definition':bd,'joint_projection':jp,'baseline_projection':bp,
            'fit_CDF':ew.conditional_targets(fit,centers,width),'val_CDF':ew.conditional_targets(val,centers,width),
            'fit_baseline':scaled_vector(baseline_blocks(fit),bd),'val_baseline':scaled_vector(baseline_blocks(val),bd)}
    torch.save(target,root/'TRAIN_TARGETS.pt')
    (root/'TARGET_STATISTICS.json').write_text(json.dumps(stats_json({**target,'fit_n':len(fit),'val_n':len(val),
        'projection_sha256':digest([jp,bp]),'only_TRAIN_fit_scales_and_initialization':True}),allow_nan=False)+'\n')
    return fit,val,target

def distribution(v,real,target,validation=False):
    sw=projected_loss(scaled_vector(joint_blocks(v),target['joint_definition']),
        scaled_vector(joint_blocks(real),target['joint_definition']),target['joint_projection'])
    _,cond=ew.conditional_loss(v,target['val_CDF' if validation else 'fit_CDF'])
    base=projected_loss(scaled_vector(baseline_blocks(v),target['baseline_definition']),
        target['val_baseline' if validation else 'fit_baseline'],target['baseline_projection'])
    value=sw+2*cond['conditional_CDF']+.5*cond['conditional_occupancy']+.5*base
    return value,{'SW':sw,'CDF':cond['conditional_CDF'],'occupancy':cond['conditional_occupancy'],'baseline':base}

def draw_noise(n,gen):
    return [torch.randn((n,16),generator=gen),torch.randn((n,4),generator=gen),
            torch.randn((n,512),generator=gen),torch.randn((n,512),generator=gen)]

def paired_train(train,realfit,realval,edges,cfg,root,log,previous_numeric_s=0):
    x,c,q,gt,gids=train
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    before=torch.get_rng_state().clone()
    constructor=train_initialization(realfit,x[gt],q[gt])
    commonG=WholeResponse(**constructor);commonD=Critic()
    gs=copy.deepcopy(commonG.state_dict());ds=copy.deepcopy(commonD.state_dict())
    torch.save({'G':gs,'D':ds,'constructor':constructor,'seed':cfg['seed']},root/'COMMON_INITIAL_STATE.pt')
    torch.save({'before_first_constructor':before,'after_common_G_D_constructor':torch.get_rng_state(),
        'numpy_state':np.random.get_state()[1],'seed':cfg['seed']},root/'INITIAL_RANDOM_STATES.pt')
    fit,val,target=make_targets(realfit,realval,edges,cfg,root)
    xt=torch.tensor(x);ct=torch.tensor(c);qt=torch.tensor(q)
    ii=torch.tensor(np.flatnonzero(gt));iv=torch.tensor(np.flatnonzero(~gt))
    vg=torch.Generator().manual_seed(cfg['validation_seed']);vbefore=vg.get_state().clone()
    vi=iv[torch.randperm(len(iv),generator=vg)[:cfg['validation_n']]]
    vj=torch.randperm(len(val),generator=vg)[:cfg['validation_n']]
    vn=draw_noise(cfg['validation_n'],vg)
    torch.save({'source_index':vi,'real_validation_index':vj,'z_pulse':vn[0],'z_base':vn[1],
        'white_pulse':vn[2],'white_base':vn[3],'RNG_before':vbefore,'RNG_after':vg.get_state()},root/'VALIDATION_INPUTS.pt')
    validation_hash=digest([vi,vj,*vn])
    models={};critics={};optim={};track={};best={};paths={}
    for name in cfg['arms']:
        models[name]=copy.deepcopy(commonG);models[name].pulse_enabled=name=='FULL_WAVE_STOCHASTIC'
        critics[name]=copy.deepcopy(commonD)
        assert state_digest(models[name].state_dict())==state_digest(gs)
        assert state_digest(critics[name].state_dict())==state_digest(ds)
        optim[name]=(torch.optim.Adam(models[name].parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['Adam_betas']),weight_decay=cfg['weight_decay']),
                     torch.optim.Adam(critics[name].parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['Adam_betas']),weight_decay=cfg['weight_decay']))
        p=root/'arms'/name;p.mkdir(parents=True,exist_ok=False);(p/'checkpoints').mkdir();paths[name]=p
        track[name]=[];best[name]=(float('inf'),0,None)
    del commonG,commonD
    gen=torch.Generator().manual_seed(cfg['seed']+1);rngbefore=gen.get_state().clone()
    samples=[];clock=time.monotonic();stop_reason='max_updates';step=0
    def checkpoint(name,step,final=False):
        model=models[name];model.eval()
        with torch.no_grad():
            values=[]
            for at in range(0,len(vi),cfg['generation_chunk']):
                z=[s[at:at+cfg['generation_chunk']] for s in vn]
                values.append(model(xt[vi[at:at+cfg['generation_chunk']]],ct[vi[at:at+cfg['generation_chunk']]],qt[vi[at:at+cfg['generation_chunk']]],*z))
            fake=torch.cat(values);value,parts=distribution(fake,val[vj],target,True)
            row={'step':step,'score':float(value),'parts':{k:float(v) for k,v in parts.items()},'training_elapsed_s':time.monotonic()-clock}
            if not math.isfinite(row['score']):raise RuntimeError(name+' nonfinite validation')
        if value<best[name][0]:best[name]=(float(value),step,copy.deepcopy(model.state_dict()))
        track[name].append(row)
        torch.save({'state_dict':model.state_dict(),'critic':critics[name].state_dict(),'G_optimizer':optim[name][0].state_dict(),
            'D_optimizer':optim[name][1].state_dict(),'training_RNG':gen.get_state(),'torch_RNG':torch.get_rng_state(),
            'constructor':constructor,'pulse_enabled':model.pulse_enabled,'step':step,'validation':row},
            paths[name]/('final_state.pt' if final else 'checkpoints/'+f'{step:04d}.pt'))
        log(f'{name} checkpoint {step} val={value:.6f} SW={parts["SW"]:.6f} CDF={parts["CDF"]:.6f}')
        model.train()
    for name in cfg['arms']:checkpoint(name,0)
    row_times=[];last_loss={}
    while step<cfg['max_updates_per_arm']:
        elapsed=time.monotonic()-clock+previous_numeric_s
        reserve=max(cfg['training_finalization_reserve_s'],3*np.mean(row_times[-10:]) if row_times else 30)
        if elapsed>=cfg['training_soft_limit_s']-reserve:stop_reason='training_soft_wall_budget_complete_round';break
        rc=time.monotonic()
        a=ii[torch.randint(len(ii),(cfg['batch_size'],),generator=gen)]
        b=torch.randint(len(fit),(cfg['batch_size'],),generator=gen);noise=draw_noise(cfg['batch_size'],gen)
        sha=digest([a,b,*noise]);samples.append({'step':step+1,'source_real_noise_sha256':sha})
        for name in cfg['arms']:
            model=models[name];critic=critics[name];go,do=optim[name];model.train();critic.train()
            for p in critic.parameters():p.requires_grad_(True)
            do.zero_grad(set_to_none=True)
            with torch.no_grad():fake=model(xt[a],ct[a],qt[a],*noise)
            dloss=F.relu(1-critic(fit[b])).mean()+F.relu(1+critic(fake)).mean()
            if not torch.isfinite(dloss):raise RuntimeError(name+' nonfinite D loss')
            dloss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg['gradient_norm_max']);do.step()
            critic.eval()
            for p in critic.parameters():p.requires_grad_(False)
            go.zero_grad(set_to_none=True);fake=model(xt[a],ct[a],qt[a],*noise)
            dist,parts=distribution(fake,fit[b],target);adv=-critic(fake).mean();loss=dist+adv
            if not torch.isfinite(loss):raise RuntimeError(name+' nonfinite G loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_norm_max']);go.step()
            last_loss[name]={'G_total':float(loss.detach()),'D_hinge':float(dloss.detach()),'adv':float(adv.detach()),**{k:float(v.detach()) for k,v in parts.items()}}
            del fake,loss,dist,parts,adv,dloss
        step+=1;row_times.append(time.monotonic()-rc)
        if step%25==0:log(f'common round {step} elapsed={time.monotonic()-clock:.1f}s losses={last_loss}')
        if step%cfg['validation_interval']==0:
            for name in cfg['arms']:checkpoint(name,step)
    # The budget stop never creates an extra validation candidate between fixed intervals.
    for name in cfg['arms']:
        model=models[name]
        torch.save({'state_dict':model.state_dict(),'critic':critics[name].state_dict(),'G_optimizer':optim[name][0].state_dict(),
            'D_optimizer':optim[name][1].state_dict(),'training_RNG':gen.get_state(),'torch_RNG':torch.get_rng_state(),
            'constructor':constructor,'pulse_enabled':model.pulse_enabled,'step':step},paths[name]/'final_state.pt')
        model.load_state_dict(best[name][2]);model.eval()
        torch.save({'state_dict':model.state_dict(),'constructor':constructor,'pulse_enabled':model.pulse_enabled,
            'selected_step':best[name][1],'selected_score':best[name][0],'config':cfg},paths[name]/'model.pt')
        receipt={'optimizer_updates_G':step,'optimizer_updates_D':step,'selected_step':best[name][1],
            'best_TRAIN_validation_score':best[name][0],'validation_trace':track[name],'stop_reason':stop_reason,
            'initial_G_state_sha256':state_digest(gs),'initial_D_state_sha256':state_digest(ds),'validation_input_sha256':validation_hash,
            'shared_sampling_rows':len(samples),'last_losses':last_loss[name],'constructor_TRAIN_fit_only':constructor,
            'actual_2000_updates':step==2000,'no_DEV_or_EVAL_selection':True}
        (paths[name]/'TRAINING_RECEIPT.json').write_text(json.dumps(receipt,allow_nan=False,indent=2)+'\n')
    torch.save({'before_first_draw':rngbefore,'after_final_round':gen.get_state(),'torch_after_training':torch.get_rng_state()},root/'TRAINING_RANDOM_STATES.pt')
    (root/'TRAINING_SAMPLE_DIGESTS.json').write_text(json.dumps(samples,indent=2)+'\n')
    receipt={'common_completed_rounds':step,'same_G_D_updates_per_arm':True,'same_initial_G_parameters_buffers':True,'same_initial_D_parameters_buffers':True,
        'same_indices_z_pulse_z_base_white_pulse_white_base':True,'validation_input_sha256':validation_hash,
        'sample_sequence_sha256':hashlib.sha256(json.dumps(samples,separators=(',',':')).encode()).hexdigest(),
        'training_elapsed_s':time.monotonic()-clock,'previous_numeric_s_charged_to_training_limit':previous_numeric_s,'stop_reason':stop_reason,
        'pulse_noise_drawn_for_off_arm':True,'cfD_and_crossing_indices_nondifferentiable':True,'interpolated_cross_values_piecewise_differentiable':True,
        'ADC_gradient':'STE training; exact rounding inference','training_clamps':'same saved observable floor guards; scientific metrics report true QC without those guards'}
    (root/'PAIRED_TRAINING_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    training={n:json.loads((paths[n]/'TRAINING_RECEIPT.json').read_text()) for n in cfg['arms']}
    return models,training,receipt
