from __future__ import annotations
import copy,hashlib,json,math,time
import numpy as np
import torch
from torch.nn import functional as F
from model import WholeResponse,Critic
import training as tr
import cuda_runtime as cu

def train(data,fit,val,cfg,root,load,log,counter,worker_start):
    x,c,q,gt,gids=data;donor=min(np.flatnonzero(gt),key=lambda i:hashlib.sha256(str(gids[i]).encode()).digest())
    np.savez_compressed(root/'FIXED_SOURCE.npz',clean_V=x[donor],c4=c[donor],Qsource_Vns=q[donor],donor_id=np.array(str(gids[donor])),donor_index=np.array(donor))
    initial=load('initial_state');targets=cu.transfer(load('saved_targets'));valid=load('saved_validation');oldrng=load('training_rng');initialrng=load('initial_rng',False)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    model=WholeResponse(**initial['constructor'],pulse_enabled=True);critic=Critic();model.load_state_dict(initial['G']);critic.load_state_dict(initial['D']);model.to(cu.DEVICE);critic.to(cu.DEVICE)
    torch.set_rng_state(initialrng['after_common_G_D_constructor'])
    gen=torch.Generator();gen.set_state(oldrng['before_first_draw']);before=gen.get_state().clone()
    go=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['Adam_betas']),weight_decay=cfg['weight_decay']);do=torch.optim.Adam(critic.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['Adam_betas']),weight_decay=cfg['weight_decay'])
    tx=torch.tensor(x[donor:donor+1],device=cu.DEVICE);tc=torch.tensor(c[donor:donor+1],device=cu.DEVICE);tq=torch.tensor(q[donor:donor+1],device=cu.DEVICE);fit=torch.tensor(fit,device=cu.DEVICE);val=torch.tensor(val,device=cu.DEVICE);ii=torch.tensor(np.flatnonzero(gt));vi=valid['source_index'];vj=valid['real_validation_index'];vn=[valid[k] for k in ['z_pulse','z_base','white_pulse','white_base']]
    vn_cpu=vn;vn=cu.transfer(vn);vj_device=vj.to(cu.DEVICE)
    old_samples=json.loads(load('old_sample_digests',raw=True));oldreceipt=json.loads(load('old_receipt',raw=True))
    path=root/'arm';path.mkdir(exist_ok=True);(path/'checkpoints').mkdir(exist_ok=True);trace=[];samples=[];loss_trace=[];best=(math.inf,0,None);step=0;clock=time.monotonic();losses={};stop='max_updates'
    def fixed(n):return tx.expand(n,-1),tc.expand(n,-1),tq.expand(n)
    def checkpoint(step):
        nonlocal best
        model.eval()
        with torch.no_grad():
            vv=[]
            for j in range(0,len(vi),cfg['generation_chunk']):
                no=[v[j:j+cfg['generation_chunk']] for v in vn];vv.append(model(*fixed(len(no[0])),*no))
            value,parts=tr.distribution(torch.cat(vv),val[vj_device],targets,True)
        row={'step':step,'score':float(value),'parts':{k:float(v) for k,v in parts.items()},'elapsed_s':time.monotonic()-clock};trace.append(row)
        if not math.isfinite(row['score']):raise RuntimeError('nonfinite fixed-source validation')
        if row['score']<best[0]:best=(row['score'],step,copy.deepcopy(model.state_dict()))
        torch.save({'G':model.state_dict(),'D':critic.state_dict(),'G_optimizer':go.state_dict(),'D_optimizer':do.state_dict(),'training_RNG':gen.get_state(),'global_Torch_RNG':torch.get_rng_state(),'step':step,'validation':row},path/'checkpoints'/f'{step:04d}.pt')
        log(f'FIXED checkpoint{step} score={value:.6f}');model.train()
    checkpoint(0);roundtimes=[]
    while step<2000:
        reserve=max(30,3*np.mean(roundtimes[-10:]) if roundtimes else 30)
        if time.monotonic()-worker_start>=cfg['training_soft_limit_s']-reserve:stop='training_soft_budget_including_prior_worker_time';break
        tick=time.monotonic();a=ii[torch.randint(len(ii),(cfg['batch_size'],),generator=gen)];b=torch.randint(len(fit),(cfg['batch_size'],),generator=gen);noise=tr.draw_noise(cfg['batch_size'],gen)
        sha=tr.digest([a,b,*noise]);expected=old_samples[step]['source_real_noise_sha256'] if step<len(old_samples) else None
        samples.append({'step':step+1,'sampled_original_source_target_noise_sha256':sha,'old_R8C_sample_sha256':expected,'matches_old_if_available':sha==expected if expected is not None else None})
        noise=cu.transfer(noise);b=b.to(cu.DEVICE)
        for p in critic.parameters():p.requires_grad_(True)
        model.train();critic.train();do.zero_grad(set_to_none=True)
        with torch.no_grad():fake=model(*fixed(cfg['batch_size']),*noise)
        dl=F.relu(1-critic(fit[b])).mean()+F.relu(1+critic(fake)).mean()
        if not torch.isfinite(dl):raise RuntimeError('nonfinite D loss')
        dl.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg['gradient_norm_max']);do.step();counter['D_updates']+=1
        critic.eval()
        for p in critic.parameters():p.requires_grad_(False)
        go.zero_grad(set_to_none=True);fake=model(*fixed(cfg['batch_size']),*noise);dist,parts=tr.distribution(fake,fit[b],targets);adv=-critic(fake).mean();loss=dist+adv
        if not torch.isfinite(loss):raise RuntimeError('nonfinite G loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_norm_max']);go.step();counter['G_updates']+=1
        losses={'G':float(loss.detach()),'D':float(dl.detach()),'adv':float(adv.detach()),**{k:float(v.detach()) for k,v in parts.items()}}
        step+=1;cu.guard();roundtimes.append(time.monotonic()-tick)
        if step%100==0:checkpoint(step)
        if step%25==0:
            loss_trace.append({'step':step,'elapsed_s':time.monotonic()-clock,**losses});log(f'FIXED step{step} elapsed={time.monotonic()-clock:.1f} losses={losses}')
    torch.save({'G':model.state_dict(),'D':critic.state_dict(),'G_optimizer':go.state_dict(),'D_optimizer':do.state_dict(),'training_RNG':gen.get_state(),'global_Torch_RNG':torch.get_rng_state(),'step':step},path/'final_state.pt')
    model.load_state_dict(best[2]);model.eval();torch.save({'state_dict':model.state_dict(),'constructor':initial['constructor'],'pulse_enabled':True,'selected_step':best[1],'selected_score':best[0]},path/'model.pt')
    torch.save({'before_first_draw':before,'after_training':gen.get_state(),'global_Torch_after':torch.get_rng_state()},root/'TRAINING_RANDOM_STATES.pt')
    (root/'TRAINING_SAMPLE_DIGESTS.json').write_text(json.dumps(samples,indent=2)+'\n')
    initial_hash={'G':tr.state_digest(initial['G']),'D':tr.state_digest(initial['D'])}
    receipt={'G_updates':step,'D_updates':step,'selected_step':best[1],'selected_score':best[0],'validation_trace':trace,'training_elapsed_s':time.monotonic()-clock,'stop_reason':stop,'max_updates':2000,'soft_budget_s':1100,'fixed_source_donor_id':str(gids[donor]),'fixed_source_donor_index':int(donor),'source_ids_sampled_then_discarded':True,'sample_digest_available':sum(s['old_R8C_sample_sha256'] is not None for s in samples),'sample_digest_matches':sum(s['matches_old_if_available'] is True for s in samples),'initial_state_sha256':initial_hash,'old_validation_input_sha256':tr.digest([vi,vj,*vn_cpu]),'expected_old_validation_input_sha256':oldreceipt['validation_input_sha256'],'uses_original_TRAIN_target_statistics':True,'TRAIN_only_selection':True,'DEV_or_EVAL_selection':False,'last_losses':losses,'historical_contrast_not_guaranteed_optimizer_bitwise_pair':True,'fixed_pulse_branch_not_zero_conditional_mean':True}
    receipt['loss_trace']=loss_trace;receipt['execution_environment']=cu.facts();receipt['CPU_private_RNG_inputs_retained']=True
    (root/'TRAINING_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return model,receipt,(x[donor],c[donor],q[donor],str(gids[donor]))
