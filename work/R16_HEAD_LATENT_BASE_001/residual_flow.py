from __future__ import annotations
import json, math, time, copy
import numpy as np
import torch
from torch import nn
from conditional_density import clone_state

class ConditionalCoupling(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        p=torch.as_tensor(permutation,dtype=torch.long)
        self.register_buffer('permutation',p)
        self.register_buffer('inverse_permutation',torch.argsort(p))
        self.net=nn.Sequential(nn.Linear(257,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,446))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def parameters_at(self,a,x):
        shift,raw=self.net(torch.cat([a,x],dim=1)).chunk(2,dim=1)
        return shift,3*torch.tanh(raw/3)

    def forward(self,y,x):
        yp=y.index_select(1,self.permutation);a,b=yp.chunk(2,dim=1)
        shift,s=self.parameters_at(a,x)
        out=torch.cat([a,b*torch.exp(s)+shift],dim=1)
        return out.index_select(1,self.inverse_permutation),s.sum(dim=1)

    def inverse(self,z,x):
        zp=z.index_select(1,self.permutation);a,b=zp.chunk(2,dim=1)
        shift,s=self.parameters_at(a,x)
        out=torch.cat([a,(b-shift)*torch.exp(-s)],dim=1)
        return out.index_select(1,self.inverse_permutation),-s.sum(dim=1)

class ResidualFlow(nn.Module):
    def __init__(self,seed=20261009151):
        super().__init__();rng=np.random.default_rng(seed)
        self.layers=nn.ModuleList([ConditionalCoupling(rng.permutation(446)) for _ in range(8)])

    def forward(self,y,x):
        ld=torch.zeros(len(y),dtype=y.dtype,device=y.device)
        for layer in self.layers:y,d=layer(y,x);ld=ld+d
        return y,ld

    def inverse(self,z,x):
        ld=torch.zeros(len(z),dtype=z.dtype,device=z.device)
        for layer in reversed(self.layers):z,d=layer.inverse(z,x);ld=ld+d
        return z,ld

    def log_prob(self,y,x):
        z,ld=self(y,x)
        return -.5*(446*math.log(2*math.pi)+(z*z).sum(dim=1))+ld

def tail_parts(flow,tail,x,e,logdetL):
    m,ell=tail(x);y=(e-m)*torch.exp(-ell)
    lp=flow.log_prob(y,x)
    return lp-ell.sum(dim=1)-logdetL,lp,ell.sum(dim=1),y

def full_scores(flow,head,tail,x,e,h,rep,chunk=512):
    logdetL=float(np.log(np.diag(rep['tail_cholesky'])).sum());parts={k:[] for k in ['head_logdensity','tail_logdensity','flow_std_logdensity','sum_ell']}
    for start in range(0,len(x),chunk):
        sl=slice(start,start+chunk);xx=torch.from_numpy(x[sl]);ee=torch.from_numpy(e[sl])
        with torch.no_grad():
            hp=head.log_prob(xx);tp,fp,ell,_=tail_parts(flow,tail,xx,ee,logdetL)
        for k,a in zip(parts,[hp,tp,fp,ell]):parts[k].append(a.numpy())
    p={k:np.concatenate(a) for k,a in parts.items()}
    p['Jacobian']=rep['head_log_scale']+33*h[:,0]+rep['constant_logdet'];p['complete_logdensity']=p['head_logdensity']+p['tail_logdensity']-p['Jacobian']
    if not all(np.isfinite(a).all() for a in p.values()):raise FloatingPointError('full density contains nonfinite')
    summary={'all_rows':len(x),'legal_rows':len(x),'illegal_rows':0,'complete_NLL':float(-p['complete_logdensity'].mean()),'mean_head_std_logdensity':float(p['head_logdensity'].mean()),'mean_tail_logdensity':float(p['tail_logdensity'].mean()),'mean_flow_std_logdensity':float(p['flow_std_logdensity'].mean()),'mean_sum_ell':float(p['sum_ell'].mean()),'logdet_L':logdetL,'mean_logJacobian':float(p['Jacobian'].mean()),'arithmetic':'CPU float64; selected checkpoint not reselected','tail_density_formula':'logp_flow - sum(ell) - logdetL'}
    return summary,p

def generate(flow,tail,x,rep,inputs,chunk=128):
    out=[]
    for start in range(0,len(x),chunk):
        sl=slice(start,start+chunk);xx=x[sl];h=xx*rep['head_scale']+rep['head_mean'];white=inputs['tail_white'][sl]
        with torch.no_grad():
            t=torch.from_numpy(xx);y,_=flow.inverse(torch.from_numpy(white),t);m,ell=tail(t);innovation=m.numpy()+np.exp(ell.numpy())*y.numpy()
        r=rep['tail_intercept']+xx@rep['tail_B']+innovation@rep['tail_cholesky'].T
        with np.errstate(over='ignore',invalid='ignore'):
            q=np.exp(h[:,0]);v=h[:,1,None]+q[:,None]*(rep['mean']+h[:,2:]@rep['basis'][:,:32].T)+r@rep['basis'][:,32:].T
        out.append(v)
    return np.concatenate(out)

def train(model,tail,tx_np,te_np,vx_np,ve_np,rep,cfg,root,device,counter,log):
    start=time.monotonic();rng=np.random.default_rng(cfg['batch_seed'])
    initial=clone_state(model)
    torch.save({'model':initial,'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'batch_rng':rng.bit_generator.state},root/'RESIDUAL_INITIAL_STATE.pt')
    (root/'FLOW_PERMUTATIONS.json').write_text(json.dumps({'seed':cfg['permutation_seed'],'permutations':[l.permutation.tolist() for l in model.layers],'undo_permutation_each_layer':True},indent=2)+'\n')
    model=model.to(device=device,dtype=torch.float32)
    fixed_tail=copy.deepcopy(tail).to(device=device,dtype=torch.float32);fixed_tail.requires_grad_(False);fixed_tail.eval()
    tx=torch.as_tensor(tx_np,dtype=torch.float32,device=device);vx=torch.as_tensor(vx_np,dtype=torch.float32,device=device)
    with torch.no_grad():
        te=torch.as_tensor(te_np,dtype=torch.float32,device=device);ve=torch.as_tensor(ve_np,dtype=torch.float32,device=device)
        tm,tell=fixed_tail(tx);vm,vell=fixed_tail(vx)
        ty=(te-tm)*torch.exp(-tell);vy=(ve-vm)*torch.exp(-vell);vscale=vell.sum(dim=1)
    del te,ve,tm,tell,vm,vell,fixed_tail
    logdetL=float(np.log(np.diag(rep['tail_cholesky'])).sum());opt=torch.optim.Adam(model.parameters(),lr=cfg['train_lr'],weight_decay=cfg['train_weight_decay'])
    trace=[];best=None;best_step=None;best_state=None;updates=0;last_loss=None;reason='max_updates'
    def validate(step):
        nonlocal best,best_step,best_state
        with torch.no_grad():
            lp=model.log_prob(vy,vx);score=float(-(lp-vscale-logdetL).mean().item());z,_=model(vy,vx)
        if not np.isfinite(score):raise FloatingPointError('residual validation nonfinite')
        row={'step':step,'all_validation_n':599,'conditional_NLL_nat_event':score,'flow_base_NLL_nat_event':float(-lp.mean().item()),'train_loss_per_dimension_last':last_loss,'elapsed_s':time.monotonic()-start,'validation_z_maxabs':float(z.abs().max().item())};trace.append(row)
        if best is None or score<best:best=score;best_step=step;best_state=clone_state(model)
        if step%1000==0 or step<=300:log(f'residual step={step} conditional valNLL={score:.8f} best={best_step}')
    validate(0)
    for step in range(1,cfg['maximum_updates']+1):
        if time.monotonic()-start>=cfg['training_limit_s']:reason='training_wall_budget';break
        ix=rng.integers(len(tx),size=cfg['train_batch']);opt.zero_grad(set_to_none=True)
        lp=model.log_prob(ty[ix],tx[ix]);loss=-lp.mean()/446
        if not torch.isfinite(loss):raise FloatingPointError('residual training nonfinite')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train_gradient_clip'])
        if not torch.isfinite(norm):raise FloatingPointError('residual gradient nonfinite')
        opt.step();updates=step;counter['residual_updates']=step;last_loss=float(loss.detach().cpu())
        if device.type=='cuda' and torch.cuda.max_memory_allocated()>cfg['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
        if step%cfg['train_validation_every']==0:validate(step)
    torch.save(clone_state(model),root/'RESIDUAL_FINAL_STATE.pt')
    torch.save(opt.state_dict(),root/'OPTIMIZER_FINAL_STATE.pt')
    torch.save({'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'batch_rng':rng.bit_generator.state},root/'FINAL_RANDOM_STATES.pt')
    model.load_state_dict(best_state);result=model.to(device='cpu',dtype=torch.float64).eval();torch.save(result.state_dict(),root/'RESIDUAL_FLOW.pt')
    record={'kind':'conditional_residual_flow','seed':cfg['seed'],'training_N':len(tx_np),'actual_updates':updates,'planned_updates':cfg['maximum_updates'],'budget_complete':updates>=cfg['maximum_updates'],'selected_step':best_step,'selected_conditional_NLL':best,'trace':trace,'elapsed_s':time.monotonic()-start,'stop_reason':reason,'optimizer':'Adam','batch':128,'lr':cfg['train_lr'],'weight_decay':cfg['train_weight_decay'],'gradient_clip':5,'scheduler_or_restart':False,'frozen_upstream_no_optimizer_gradients':True,'selection':'minimum all599 conditional NLL step0/every100, exact tie earliest','validation_arithmetic':'float32 '+str(device),'coordinate_precomputation':'float32 frozen Gaussian tail(x), no gradients','new_random_generation_draws':0}
    (root/'TRAINING.json').write_text(json.dumps(record,indent=2)+'\n')
    del opt,tx,ty,vx,vy,vscale
    if device.type=='cuda':torch.cuda.empty_cache()
    return result,record

def mathematical_checks(model,tail,vx,ve):
    ix=np.array([0,199,399,598]);x=torch.from_numpy(vx[ix]);e=torch.from_numpy(ve[ix])
    with torch.no_grad():
        m,ell=tail(x);y=(e-m)*torch.exp(-ell);z,ld=model(y,x);back,ild=model.inverse(z,x)
    inverse=float((back-y).abs().max());cancel=float((ld+ild).abs().max())
    xx=torch.from_numpy(vx[0:1]);ee=torch.from_numpy(ve[0:1])
    with torch.no_grad():m,ell=tail(xx);yy=((ee-m)*torch.exp(-ell))[0]
    yy=yy.requires_grad_(True)
    J=torch.autograd.functional.jacobian(lambda a:model(a[None],xx)[0][0],yy)
    sign,jld=torch.linalg.slogdet(J);analytic=model(yy[None],xx)[1][0];delta=float((jld-analytic).abs().detach())
    result={'validation_positions':ix.tolist(),'inverse_maxabs':inverse,'inverse_logdet_cancel_maxabs':cancel,'Jacobian_position':0,'Jacobian_shape':[446,446],'Jacobian_fixed_head':True,'Jacobian_sign':float(sign),'Jacobian_logabsdet_error':delta,'analytic_logabsdet':float(analytic.detach()),'autodiff_logabsdet':float(jld.detach()),'arithmetic':'CPU float64','pass':inverse<=1e-8 and cancel<=1e-8 and delta<=1e-6}
    if not result['pass']:raise RuntimeError('residual mathematical checks '+repr(result))
    return result
