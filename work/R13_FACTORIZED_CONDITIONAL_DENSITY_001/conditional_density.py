from __future__ import annotations
import copy, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.linalg import solve_triangular
from scipy.stats import multivariate_normal
import density as den

class Coupling(nn.Module):
    def __init__(self, permutation, even):
        super().__init__()
        p = torch.tensor(permutation, dtype=torch.long)
        self.register_buffer('permutation', p)
        self.register_buffer('inverse_permutation', torch.argsort(p))
        self.register_buffer('fixed', torch.arange(0,17) if even else torch.arange(17,34))
        self.register_buffer('free', torch.arange(17,34) if even else torch.arange(0,17))
        self.net = nn.Sequential(nn.Linear(17,64), nn.SiLU(), nn.Linear(64,64), nn.SiLU(), nn.Linear(64,34))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def parameters_at(self, a):
        raw, shift = self.net(a).chunk(2, dim=1)
        return 3*torch.tanh(raw/3), shift

    def forward(self, x):
        xp = x.index_select(1, self.permutation)
        s,t = self.parameters_at(xp.index_select(1,self.fixed))
        out = xp.clone(); out[:,self.free] = xp.index_select(1,self.free)*torch.exp(s)+t
        return out,s.sum(dim=1)

    def inverse(self, y):
        s,t = self.parameters_at(y.index_select(1,self.fixed))
        xp = y.clone(); xp[:,self.free] = (y.index_select(1,self.free)-t)*torch.exp(-s)
        return xp.index_select(1,self.inverse_permutation),-s.sum(dim=1)

class Flow(nn.Module):
    def __init__(self, seed):
        super().__init__()
        rng = np.random.default_rng(seed+10)
        self.layers = nn.ModuleList([Coupling(rng.permutation(34), i%2==0) for i in range(8)])

    def forward(self,x):
        ld = torch.zeros(len(x),dtype=x.dtype,device=x.device)
        for layer in self.layers:
            x,d = layer(x); ld = ld+d
        return x,ld

    def inverse(self,z):
        ld = torch.zeros(len(z),dtype=z.dtype,device=z.device)
        for layer in reversed(self.layers):
            z,d = layer.inverse(z); ld = ld+d
        return z,ld

    def log_prob(self,x):
        z,ld = self(x)
        return -.5*(34*math.log(2*math.pi)+(z*z).sum(dim=1))+ld

class ConditionalTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(34,64), nn.SiLU(), nn.Linear(64,64), nn.SiLU(), nn.Linear(64,892))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        mean,raw = self.net(x).chunk(2,dim=1)
        return mean,3*torch.tanh(raw/3)

def coordinates(v,rep):
    h,r,valid = den.encode(v,rep)
    if not valid.all(): raise ValueError('frozen TRAIN/val has illegal factor coordinates')
    x = (h-rep['head_mean'])/rep['head_scale']
    m0 = rep['tail_intercept']+x@rep['tail_B']
    e = solve_triangular(rep['tail_cholesky'],(r-m0).T,lower=True,check_finite=True).T
    return x,e,h,r

def tail_log_prob(model,x,e,logdetL):
    mean,ell = model(x)
    z = (e-mean)*torch.exp(-ell)
    return -.5*(446*math.log(2*math.pi)+(z*z).sum(dim=1))-ell.sum(dim=1)-logdetL

def setup(seed,device):
    torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)

def clone_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def train(kind,train_x,train_e,val_x,val_e,rep,cfg,root,device,total_used,log,counter):
    seed = cfg['head_seed'] if kind=='head' else cfg['tail_seed']
    rng = setup(seed,device); start=time.monotonic()
    model = Flow(seed) if kind=='head' else ConditionalTail()
    initial = clone_state(model)
    torch.save({'model':initial,'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},root/(kind.upper()+'_INITIAL_STATE.pt'))
    model=model.to(device=device,dtype=torch.float32)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['train_lr'],weight_decay=cfg['train_weight_decay'])
    tx=torch.as_tensor(train_x,dtype=torch.float32,device=device); vx=torch.as_tensor(val_x,dtype=torch.float32,device=device)
    te=torch.as_tensor(train_e,dtype=torch.float32,device=device); ve=torch.as_tensor(val_e,dtype=torch.float32,device=device)
    logdetL=float(np.log(np.diag(rep['tail_cholesky'])).sum())
    trace=[]; best=None; best_step=None; best_state=None; updates=0; loss_last=None; reason='max_updates'
    def validate(step):
        nonlocal best,best_step,best_state
        with torch.no_grad():
            lp=model.log_prob(vx) if kind=='head' else tail_log_prob(model,vx,ve,logdetL)
            score=float(-lp.mean().item())
        if not np.isfinite(score): raise FloatingPointError(kind+' validation nonfinite')
        trace.append({'step':step,'all_validation_n':599,'NLL_nat_event':score,'train_loss_per_dimension_last':loss_last,'elapsed_s':time.monotonic()-start})
        if best is None or score<best:
            best=score;best_step=step;best_state=clone_state(model)
        log(f'{kind} step={step} valNLL={score:.8f}')
    validate(0)
    for step in range(1,cfg['train_updates']+1):
        if time.monotonic()-start>=min(cfg['training_network_limit_s'],cfg['training_total_limit_s']-total_used):
            reason='training_wall_budget';break
        ix=rng.integers(len(tx),size=cfg['train_batch']); bx=tx[ix]
        optimizer.zero_grad(set_to_none=True)
        lp=model.log_prob(bx) if kind=='head' else tail_log_prob(model,bx,te[ix],logdetL)
        loss=-lp.mean()/(34 if kind=='head' else 446)
        if not torch.isfinite(loss): raise FloatingPointError(kind+' train loss nonfinite')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train_gradient_clip'])
        if not torch.isfinite(norm): raise FloatingPointError(kind+' gradient nonfinite')
        optimizer.step();updates=step;counter[kind+'_updates']=step;loss_last=float(loss.detach().cpu())
        if device.type=='cuda' and torch.cuda.max_memory_allocated()>cfg['GPU_allocated_limit_bytes']: raise MemoryError('GPU allocated budget')
        if step%cfg['train_validation_every']==0: validate(step)
    # Only predetermined step0/100/... checkpoints are eligible; no off-grid early-stop validation.
    model.load_state_dict(best_state)
    result=model.to(device='cpu',dtype=torch.float64)
    torch.save(result.state_dict(),root/('FLOW.pt' if kind=='head' else 'TAIL.pt'))
    torch.save({'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},root/(kind.upper()+'_FINAL_RANDOM_STATE.pt'))
    elapsed=time.monotonic()-start
    record={'kind':kind,'seed':seed,'updates':updates,'selected_step':best_step,'selected_validation_NLL_nat_event_training_arithmetic':best,'validation_arithmetic':'float32 on '+str(device),'selection':'minimum full599 corresponding density NLL, exact ties keep earlier step','checkpoint_grid':'step0 and each100','trace':trace,'elapsed_s':elapsed,'stop_reason':reason,'optimizer':'Adam','lr':cfg['train_lr'],'weight_decay':cfg['train_weight_decay'],'gradient_norm_clip':cfg['train_gradient_clip'],'scheduler_or_restart':False,'random_index_stream':rng.bit_generator.state,'new_network_fit_attempts':1,'source_waveform_template_resampling':False}
    (root/(kind.upper()+'_TRAINING.json')).write_text(json.dumps(record,indent=2)+'\n')
    del optimizer,tx,te,vx,ve
    if device.type=='cuda': torch.cuda.empty_cache()
    return result,record

def mathematical_checks(flow,tail,init_tail,val_x,val_e,val_h,val_r,rep):
    ix=np.array([0,199,399,598]);x=torch.from_numpy(val_x[ix]);e=torch.from_numpy(val_e[ix]);L=rep['tail_cholesky'];logdetL=float(np.log(np.diag(L)).sum())
    with torch.no_grad():
        z,ld=flow(x);back,ild=flow.inverse(z)
        inverse=float((back-x).abs().max());cancel=float((ld+ild).abs().max())
        im,il=init_tail(x)
        init_formula=tail_log_prob(init_tail,x,e,logdetL).numpy()
        expected=den.gaussian_logpdf(val_r[ix],rep,val_x[ix])
        identity=float(max(im.abs().max(),il.abs().max()));identity_density=float(np.max(np.abs(init_formula-expected)))
        model_tail=tail_log_prob(tail,x,e,logdetL).numpy();m,ell=tail(x);m=m.numpy();ell=ell.numpy()
    first=torch.from_numpy(val_x[0].copy()).requires_grad_(True)
    J=torch.autograd.functional.jacobian(lambda a:flow(a[None])[0][0],first)
    sign,jacld=torch.linalg.slogdet(J)
    analytic=flow(first[None])[1][0]
    jac_error=float(torch.abs(jacld-analytic).detach())
    direct=[]
    for j,k in enumerate(ix):
        mean=rep['tail_intercept']+val_x[k]@rep['tail_B']+L@m[j]
        A=L*np.exp(ell[j])[None,:];C=A@A.T
        direct.append(multivariate_normal.logpdf(val_r[k],mean=mean,cov=C))
    delta=float(np.max(np.abs(model_tail-np.array(direct))))
    checks={'fixed_validation_positions':ix.tolist(),'flow_inverse_maxabs':inverse,'flow_forward_inverse_logdet_cancel_maxabs':cancel,'flow_Jacobian_position':0,'flow_full_Jacobian_shape':[34,34],'flow_Jacobian_sign':float(sign.detach()),'flow_analytic_vs_autodiff_logabsdet_error':jac_error,'tail_step0_mean_scale_zero_maxabs':identity,'tail_step0_gaussian_identity_maxabs_nat':identity_density,'tail_final_full_covariance_formula_maxabs_nat':delta,'arithmetic':'CPU float64','tail_L_frozen':True}
    if inverse>1e-8 or cancel>1e-8 or jac_error>1e-6 or identity!=0 or identity_density>1e-7 or delta>1e-6: raise RuntimeError('R13 mathematical checks failed '+str(checks))
    checks['pass']=True
    return checks

def validation_scores(v,rep,gmm,flow,tail):
    x,e,h,r=coordinates(v,rep);t=torch.from_numpy(x);et=torch.from_numpy(e)
    with torch.no_grad():
        head_flow=flow.log_prob(t).numpy();tail_cond=tail_log_prob(tail,t,et,float(np.log(np.diag(rep['tail_cholesky'])).sum())).numpy()
    head_gmm=den.gmm_logpdf(x,gmm);tail_global=den.gaussian_logpdf(r,rep,x)
    jac=rep['head_log_scale']+33*h[:,0]+rep['constant_logdet']
    out={};private={}
    for name,hp,tp in [('GMM_GLOBAL',head_gmm,tail_global),('FLOW_GLOBAL',head_flow,tail_global),('GMM_CONDITIONAL',head_gmm,tail_cond),('FLOW_CONDITIONAL',head_flow,tail_cond)]:
        score=hp+tp-jac
        if not np.isfinite(score).all(): raise FloatingPointError('complete validation density nonfinite '+name)
        private[name]=score
        out[name]={'all_validation_rows':599,'legal_rows':599,'illegal_rows':0,'complete_NLL':float(-score.mean()),'mean_head_std_logdensity':float(hp.mean()),'mean_tail_logdensity':float(tp.mean()),'mean_logJacobian':float(jac.mean()),'arithmetic':'CPU float64 saved selected weights','not_new_model_selection':True}
    return out,private

def sample_heads(rep,gmm,flow,inp,chunk=128):
    gmm_x=[];flow_x=[]
    for start in range(0,len(inp['draw_ids']),chunk):
        sl=slice(start,start+chunk);c=inp['component'][sl];z=inp['head_white'][sl];x=np.empty_like(z)
        for k in range(len(gmm['weights'])):
            use=c==k
            if use.any():x[use]=gmm['means'][k]+z[use]@gmm['cholesky'][k].T
        with torch.no_grad(): fx=flow.inverse(torch.from_numpy(z))[0].numpy()
        gmm_x.append(x);flow_x.append(fx)
    return np.concatenate(gmm_x),np.concatenate(flow_x)

def generate(name,rep,x,tail,inp,chunk=128):
    conditional=name.endswith('CONDITIONAL');pieces=[]
    for start in range(0,len(x),chunk):
        sl=slice(start,start+chunk);xx=x[sl];h=xx*rep['head_scale']+rep['head_mean'];white=inp['tail_white'][sl]
        if conditional:
            with torch.no_grad():m,ell=tail(torch.from_numpy(xx));innov=m.numpy()+np.exp(ell.numpy())*white
        else:innov=white
        r=rep['tail_intercept']+xx@rep['tail_B']+innov@rep['tail_cholesky'].T
        with np.errstate(over='ignore',invalid='ignore'):
            q=np.exp(h[:,0]);v=h[:,1,None]+q[:,None]*(rep['mean']+h[:,2:]@rep['basis'][:,:32].T)+r@rep['basis'][:,32:].T
        pieces.append(v)
    return np.concatenate(pieces)

def load_models(root,cfg):
    flow=Flow(cfg['head_seed']).double();flow.load_state_dict(torch.load(root/'FLOW.pt',map_location='cpu',weights_only=True));flow.eval()
    tail=ConditionalTail().double();tail.load_state_dict(torch.load(root/'TAIL.pt',map_location='cpu',weights_only=True));tail.eval()
    return flow,tail
