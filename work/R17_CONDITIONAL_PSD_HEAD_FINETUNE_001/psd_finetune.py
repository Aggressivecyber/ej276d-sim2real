from __future__ import annotations
import copy,hashlib,time,math,json,pathlib,signal
import numpy as np
import torch
import evaluation as ev
import conditional_density as cd

class FixedKernel:
    def __init__(self,tail,flow,rep,base,device='cpu',dtype=torch.float64):
        self.tail=copy.deepcopy(tail).to(device=device,dtype=dtype).eval().requires_grad_(False);self.flow=copy.deepcopy(flow).to(device=device,dtype=dtype).eval().requires_grad_(False)
        self.rep={k:torch.as_tensor(v,device=device,dtype=dtype) for k,v in rep.items() if isinstance(v,np.ndarray) and v.dtype.kind in 'fiu'}
        self.weights=torch.as_tensor(base['weights'],device=device,dtype=dtype);self.means=torch.as_tensor(base['means'],device=device,dtype=dtype);self.cholesky=torch.as_tensor(base['cholesky'],device=device,dtype=dtype);self.device=device;self.dtype=dtype
    def logbase(self,z):
        diff=z[None,:,:]-self.means[:,None,:];white=torch.linalg.solve_triangular(self.cholesky,diff.transpose(1,2),upper=False).transpose(1,2);lp=torch.log(self.weights)[:,None]-.5*(34*math.log(2*math.pi)+(white*white).sum(-1))-torch.log(torch.diagonal(self.cholesky,dim1=-2,dim2=-1)).sum(-1)[:,None];return torch.logsumexp(lp,dim=0)
    def loghead(self,head,x):
        z,ld=head(x);return self.logbase(z)+ld
    def generate(self,head,latent,white):
        x,_=head.inverse(latent);h=x*self.rep['head_scale']+self.rep['head_mean'];y,_=self.flow.inverse(white,x);mean,ell=self.tail(x);innov=mean+torch.exp(ell)*y;r=self.rep['tail_intercept']+x@self.rep['tail_B']+innov@self.rep['tail_cholesky'].T;return h[:,1,None]+torch.exp(h[:,0,None])*(self.rep['mean']+h[:,2:]@self.rep['basis'][:,:32].T)+r@self.rep['basis'][:,32:].T
    def generate_chunks(self,head,latent,white,chunk=128):
        out=[]
        for st in range(0,len(latent),chunk):out.append(self.generate(head,latent[st:st+chunk],white[st:st+chunk]))
        return torch.cat(out)

def features_torch(v):
    b=v[:,:100].mean(1);x=v-b[:,None];q=x[:,100:400].sum(1)*1000;fast=x[:,100:150].sum(1)*1000
    if not bool(torch.isfinite(v).all() and torch.isfinite(q).all() and (q>0).all()):raise FloatingPointError('nonfinite waveform or nonpositive signed Q in continuous surrogate; no event deletion')
    psd=1-fast/q
    if not bool(torch.isfinite(psd).all()):raise FloatingPointError('nonfinite PSD surrogate')
    return q,psd

def conditional(q,psd,edges,cfg):
    edges=torch.as_tensor(edges,device=q.device,dtype=q.dtype);centers=(edges[:-1]+edges[1:])/2;sigmas=edges[1:]-edges[:-1];weights=torch.exp(-.5*((torch.log(q)[:,None]-centers[None,:])/sigmas[None,:])**2);grid=torch.linspace(cfg['CDF_grid'][0],cfg['CDF_grid'][1],cfg['CDF_grid'][2],device=q.device,dtype=q.dtype);cdf=(weights.T@torch.sigmoid((grid[None,:]-psd[:,None])/cfg['CDF_temperature']))/(weights.sum(0)[:,None]+1e-8);occupancy=weights.mean(0);return cdf,occupancy,weights.sum(0)

def targets(v,edges,cfg):
    with torch.no_grad():
        q,p=features_torch(torch.from_numpy(v.astype('f8')));cdf,occupancy,support=conditional(q,p,edges,cfg)
    return {'CDF':cdf.numpy(),'occupancy':occupancy.numpy(),'window_weight_sum':support.numpy(),'N':len(v),'logQ_below':int((torch.log(q)<edges[0]).sum()),'logQ_above':int((torch.log(q)>edges[-1]).sum()),'PSD_below_grid':int((p<cfg['CDF_grid'][0]).sum()),'PSD_above_grid':int((p>cfg['CDF_grid'][1]).sum()),'nonfinite':0,'nonpositive_Q':0,'QC':len(v),'counts_before_any_window_weighting':True}

def sample_inputs(rng,n,base):
    u=rng.uniform(size=n);white=rng.normal(size=(n,34));tailwhite=rng.normal(size=(n,446));component=np.searchsorted(np.cumsum(base['weights']),u,side='right');latent=np.empty_like(white)
    for k in range(8):
        use=component==k;latent[use]=base['means'][k]+white[use]@base['cholesky'][k].T
    return {'uniform':u,'head_white':white,'tail_white':tailwhite,'component':component,'latent':latent}

def validation(head,kernel,x,noise,target,edges,cfg):
    model=copy.deepcopy(head).cpu().double().eval();x=torch.from_numpy(x);latent=torch.from_numpy(noise['latent']);white=torch.from_numpy(noise['tail_white'])
    with torch.no_grad():
        nll=float(-kernel.loghead(model,x).mean()/34);v=kernel.generate_chunks(model,latent,white);q,p=features_torch(v);cdf,occ,support=conditional(q,p,edges,cfg);cl=float((cdf-torch.from_numpy(target['CDF'])).abs().mean());ol=float((occ-torch.from_numpy(target['occupancy'])).abs().mean());j=nll+cfg['CDF_weight']*cl+cfg['occupancy_weight']*ol
    return {'head_NLL_div34':nll,'conditional_CDF_L1':cl,'window_occupancy_L1':ol,'J':j,'validation_real_N':len(x),'validation_generated_N':len(v),'logQ_below':int((torch.log(q)<edges[0]).sum()),'logQ_above':int((torch.log(q)>edges[-1]).sum()),'PSD_below_grid':int((p<cfg['CDF_grid'][0]).sum()),'PSD_above_grid':int((p>cfg['CDF_grid'][1]).sum()),'window_occupancy':occ.numpy(),'window_weight_sum':support.numpy(),'arithmetic':'CPUfloat64; fixed2000 validation innovations'}

def gradient_check(head,kernel,noise,target,edges,cfg):
    h=copy.deepcopy(head).double().train().requires_grad_(True);kernel.tail.zero_grad(set_to_none=True);kernel.flow.zero_grad(set_to_none=True);v=kernel.generate(h,torch.from_numpy(noise['latent'][:32]),torch.from_numpy(noise['tail_white'][:32]));q,p=features_torch(v);cdf,occ,_=conditional(q,p,edges,cfg);loss=cfg['CDF_weight']*(cdf-torch.from_numpy(target['CDF'])).abs().mean()+cfg['occupancy_weight']*(occ-torch.from_numpy(target['occupancy'])).abs().mean();loss.backward();grads=[x.grad for x in h.parameters() if x.grad is not None];finite=all(torch.isfinite(g).all()for g in grads);norm=float(torch.sqrt(sum((g*g).sum()for g in grads)));frozen_grad=any(p.grad is not None for p in list(kernel.tail.parameters())+list(kernel.flow.parameters()));return {'small_batch_N':32,'PSD_only_gradient_norm':norm,'gradient_finite':bool(finite),'gradient_nonzero':norm>0,'frozen_parameters_gradient_absent':not frozen_grad,'optimizer_only_head':True,'pass':bool(finite) and norm>0 and not frozen_grad}

def train_arm(name,initial,tail,flow,rep,base,tx,vx,noise,target_train,target_val,edges,cfg,root,device,counter,log):
    torch.manual_seed(cfg['train_seed'])
    if device.type=='cuda':torch.cuda.manual_seed_all(cfg['train_seed'])
    model=copy.deepcopy(initial).to(device=device,dtype=torch.float32).train().requires_grad_(True);kernel=FixedKernel(tail,flow,rep,base,device=device,dtype=torch.float32);cpu_kernel=FixedKernel(tail,flow,rep,base);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);xr=torch.as_tensor(tx,device=device,dtype=torch.float32);tgt={k:torch.as_tensor(target_train[k],device=device,dtype=torch.float32)for k in ['CDF','occupancy']};rng_i=np.random.default_rng(cfg['train_seed']);rng_g=np.random.default_rng(cfg['train_generation_seed']);folder=root/name;folder.mkdir(exist_ok=True);torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},folder/'INITIAL_HEAD.pt');torch.save({'Torch':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state_all() if device.type=='cuda'else [],'indices':rng_i.bit_generator.state,'generation':rng_g.bit_generator.state},folder/'INITIAL_RNG_STATE.pt');begin=time.monotonic();previous_timer=signal.getitimer(signal.ITIMER_REAL)[0];signal.setitimer(signal.ITIMER_REAL,min(previous_timer,cfg['train_each_wall_s']));trace=[];best=float('inf');best_step=0;digest_i=hashlib.sha256();digest_g=hashlib.sha256();updates=0;failure=None
    def check(step,losses=None):
        nonlocal best,best_step
        row=validation(model,cpu_kernel,vx,noise,target_val,edges,cfg);row.update(step=step,training_elapsed_s=time.monotonic()-begin,TRAIN_index_stream_sha256=digest_i.hexdigest(),TRAIN_generation_stream_sha256=digest_g.hexdigest(),last_training_batch=losses);trace.append(row)
        if row['J']<best:best=row['J'];best_step=step;torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},folder/'SELECTED_HEAD.pt')
        log(f'{name} step{step} J={row["J"]:.7f} NLL/34={row["head_NLL_div34"]:.7f} CDF={row["conditional_CDF_L1"]:.7f}');(folder/'TRAINING_TRACE.json').write_text(json.dumps(ev.safe(trace),indent=2,allow_nan=False)+'\n')
    try:
        check(0)
        for step in range(1,cfg['updates']+1):
            if time.monotonic()-begin>=cfg['train_each_wall_s']-10:break
            ix=rng_i.integers(len(tx),size=cfg['batch']);draw=sample_inputs(rng_g,cfg['batch'],base);digest_i.update(ix.astype('<i4').tobytes());digest_g.update(draw['latent'].tobytes());digest_g.update(draw['tail_white'].tobytes());optimizer.zero_grad(set_to_none=True);nll=-kernel.loghead(model,xr[ix]).mean()/34;cl=torch.zeros((),device=device);ol=torch.zeros((),device=device)
            if name=='NLL_PSD':
                v=kernel.generate(model,torch.as_tensor(draw['latent'],device=device,dtype=torch.float32),torch.as_tensor(draw['tail_white'],device=device,dtype=torch.float32));q,p=features_torch(v);cdf,occ,_=conditional(q,p,edges,cfg);cl=(cdf-tgt['CDF']).abs().mean();ol=(occ-tgt['occupancy']).abs().mean()
            loss=nll+cfg['CDF_weight']*cl+cfg['occupancy_weight']*ol
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite training loss; stopped arm')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            if not bool(torch.isfinite(grad)):raise FloatingPointError('nonfinite head gradient; stopped arm')
            optimizer.step();updates=step;counter['head_updates_'+name]=step
            if device.type=='cuda'and torch.cuda.max_memory_allocated()>cfg['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated limit')
            vals={'NLL_div34':float(nll.detach()),'CDF_L1':float(cl.detach()) if name=='NLL_PSD'else None,'occupancy_L1':float(ol.detach()) if name=='NLL_PSD'else None,'loss':float(loss.detach()),'gradient_norm_before_clip':float(grad),'random_generation_rows_consumed':cfg['batch'],'CONT_generation_stream_consumed_without_optimization':name=='NLL_CONT'}
            if step%cfg['validation_every']==0:check(step,vals)
        if trace[-1]['step']!=updates:check(updates,vals if updates else None)
    except (FloatingPointError,MemoryError,TimeoutError)as ex:failure=repr(ex)
    finally:
        signal.setitimer(signal.ITIMER_REAL,max(.001,previous_timer-(time.monotonic()-begin)))
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.monotonic()-begin;counter['training_cumulative_s']+=elapsed;torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},folder/'FINAL_HEAD.pt');torch.save(optimizer.state_dict(),folder/'FINAL_OPTIMIZER.pt');torch.save({'Torch':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state_all() if device.type=='cuda'else [],'indices':rng_i.bit_generator.state,'generation':rng_g.bit_generator.state},folder/'FINAL_RNG_STATE.pt')
    selected=copy.deepcopy(initial).cpu().double();selected.load_state_dict(torch.load(folder/'SELECTED_HEAD.pt',map_location='cpu',weights_only=True));selected.eval();rec={'status':'complete' if failure is None else 'stopped_branch_failure','error':failure,'updates':updates,'max_updates':cfg['updates'],'selected_step':best_step,'selected_J':best,'elapsed_s':elapsed,'limit_s':cfg['train_each_wall_s'],'initial_J':trace[0]['J'],'final_evaluated_J':trace[-1]['J'],'trace':trace,'TRAIN_indices_seed':cfg['train_seed'],'TRAIN_generation_seed':cfg['train_generation_seed'],'TRAIN_index_stream_sha256':digest_i.hexdigest(),'TRAIN_generation_stream_sha256':digest_g.hexdigest(),'same_val_J_used_for_both':True,'step0_included':True,'train_dtype':'float32','validation_dtype':'CPUfloat64','frozen_parameter_requires_grad_count':sum(p.requires_grad for p in list(kernel.tail.parameters())+list(kernel.flow.parameters()))};(folder/'TRAINING_RECEIPT.json').write_text(json.dumps(ev.safe(rec),indent=2,allow_nan=False)+'\n');return selected,rec
