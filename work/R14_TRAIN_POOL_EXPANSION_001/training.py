from __future__ import annotations
import math,time,json
import numpy as np
import torch
from conditional_density import Flow,ConditionalTail,setup,clone_state,tail_log_prob

def train_expanded(kind,tx_np,te_np,vx_np,ve_np,rep,cfg,root,device,total_used,log,counter):
    seed=cfg['head_seed'] if kind=='head' else cfg['tail_seed'];rng=setup(seed,device);start=time.monotonic()
    U=min(cfg['maximum_updates'],100*math.ceil((2000*len(tx_np)/2592)/100))
    model=Flow(seed) if kind=='head' else ConditionalTail()
    old=torch.load(__import__('pathlib').Path(cfg['remote_R13_root'])/(kind.upper()+'_INITIAL_STATE.pt'),map_location='cpu',weights_only=True)
    model.load_state_dict(old['model'],strict=True);torch.set_rng_state(old['torch_cpu_rng'])
    if device.type=='cuda' and old['torch_cuda_rng']:torch.cuda.set_rng_state_all(old['torch_cuda_rng'])
    torch.save(old,root/(kind.upper()+'_INITIAL_STATE.pt'))
    model=model.to(device=device,dtype=torch.float32);opt=torch.optim.Adam(model.parameters(),lr=cfg['train_lr'],weight_decay=cfg['train_weight_decay'])
    tx=torch.as_tensor(tx_np,dtype=torch.float32,device=device);te=torch.as_tensor(te_np,dtype=torch.float32,device=device);vx=torch.as_tensor(vx_np,dtype=torch.float32,device=device);ve=torch.as_tensor(ve_np,dtype=torch.float32,device=device)
    logdetL=float(np.log(np.diag(rep['tail_cholesky'])).sum());trace=[];best={'UPDATES':None,'EXPOSURES':None};states={};steps={};updates=0;loss_last=None;reason='planned_U_reached'
    def validate(step):
        nonlocal best,states,steps
        with torch.no_grad():lp=model.log_prob(vx) if kind=='head' else tail_log_prob(model,vx,ve,logdetL);score=float(-lp.mean().item())
        if not np.isfinite(score):raise FloatingPointError(kind+' validation nonfinite')
        trace.append({'step':step,'all_validation_n':599,'NLL_nat_event':score,'train_loss_per_dimension_last':loss_last,'elapsed_s':time.monotonic()-start})
        for budget in ['UPDATES','EXPOSURES']:
            if budget=='UPDATES' and step>2000:continue
            if best[budget] is None or score<best[budget]:best[budget]=score;steps[budget]=step;states[budget]=clone_state(model)
        if step%1000==0 or step<=300:log(f'{kind} step={step}/{U} valNLL={score:.7f} best={steps}')
    validate(0)
    for step in range(1,U+1):
        if time.monotonic()-start>=min(cfg['training_network_limit_s'],cfg['training_total_limit_s']-total_used):reason='training_wall_budget';break
        ix=rng.integers(len(tx),size=cfg['train_batch']);bx=tx[ix];opt.zero_grad(set_to_none=True)
        lp=model.log_prob(bx) if kind=='head' else tail_log_prob(model,bx,te[ix],logdetL);loss=-lp.mean()/(34 if kind=='head' else 446)
        if not torch.isfinite(loss):raise FloatingPointError(kind+' training loss nonfinite')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train_gradient_clip'])
        if not torch.isfinite(norm):raise FloatingPointError(kind+' gradient nonfinite')
        opt.step();updates=step;counter[kind+'_updates']=step;loss_last=float(loss.detach().cpu())
        if device.type=='cuda' and torch.cuda.max_memory_allocated()>cfg['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
        if step%100==0:validate(step)
    torch.save(clone_state(model),root/(kind.upper()+'_FINAL_STATE.pt'))
    results={}
    for budget in states:
        result=(Flow(seed) if kind=='head' else ConditionalTail()).double();result.load_state_dict(states[budget]);result.eval();results[budget]=result
        torch.save(result.state_dict(),root/(('FLOW_' if kind=='head' else 'TAIL_')+budget+'.pt'))
    torch.save({'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'batch_rng':rng.bit_generator.state},root/(kind.upper()+'_FINAL_RANDOM_STATE.pt'))
    record={'kind':kind,'seed':seed,'training_N':len(tx_np),'actual_updates':updates,'planned_equal_updates':2000,'planned_equal_exposure_U':U,'equal_updates_complete':updates>=2000,'equal_exposures_complete':updates>=U,'selected_steps':steps,'selected_NLL':best,'trace':trace,'initial_state_source':cfg['remote_R13_root']+'/'+kind.upper()+'_INITIAL_STATE.pt','same_initial_parameters':True,'same_minibatch_sequence_as_R13':False,'continuous_optimizer_path':True,'no_restart_scheduler':True,'batch':128,'lr':cfg['train_lr'],'weight_decay':cfg['train_weight_decay'],'gradient_clip':5,'validation_arithmetic':'float32 '+str(device),'original_small_mean_draws_per_record':2000*128/2592,'large_equal_updates_mean_draws_per_record':min(updates,2000)*128/len(tx_np),'large_actual_mean_draws_per_record':updates*128/len(tx_np),'elapsed_s':time.monotonic()-start,'stop_reason':reason,'selection':'step0/every100 minimum full599 NLL; exact ties earliest, no DEV selection'}
    (root/(kind.upper()+'_TRAINING.json')).write_text(json.dumps(record,indent=2)+'\n')
    del model,opt,tx,te,vx,ve
    if device.type=='cuda':torch.cuda.empty_cache()
    return results,record
