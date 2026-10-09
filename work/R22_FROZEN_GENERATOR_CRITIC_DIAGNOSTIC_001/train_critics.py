from __future__ import annotations
import time,os,sys,copy,platform
START=time.monotonic()
import numpy as np,torch
from support import *
from legacy_model import Critic,input_channels
torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.manual_seed(20261009227);np.random.seed(np.random.SeedSequence(20261009227).generate_state(624))
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
device=torch.device('cuda:0'if torch.cuda.is_available()else'cpu')
if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.5,0)
pool=npz(ROOT/'FIXED_POOL_PRIVATE.npz')
data={k:torch.from_numpy(pool[k]).to(device)for k in ['real_train','real_val','gen_train_pre','gen_train_post','gen_val_pre','gen_val_post']}
with torch.no_grad():channels={k:input_channels(v)for k,v in data.items()}
mu=torch.from_numpy(pool['mu'].astype('f4')).to(device);scale=torch.from_numpy(pool['scale'].astype('f4')).to(device)
initial=torch.load(CFG['D_initial'],map_location=device,weights_only=False)
np.savez_compressed(ROOT/'INITIAL_TRAIN_RNG.npz',torch_cpu=torch.get_rng_state().numpy(),torch_cuda=torch.cuda.get_rng_state(0).cpu().numpy()if device.type=='cuda'else np.zeros(0,'u1'))
selected_grad=order(pool['source_val_ids'],'R22-gradient64')[:64]
torchweights={k:torch.from_numpy(pool[k].astype('f4')).to(device)for k in ['real_val_weight','gen_val_weight']}
alltrace={};scores={};grads={};trained=0;train_total=0.
def chunk_score(d,x):
    result=[]
    with torch.no_grad():
        for j in range(0,len(x),512):result.append(d.net(x[j:j+512]).flatten(1).mean(1).cpu().numpy())
    return np.concatenate(result)
def check_gpu():
    if device.type=='cuda'and torch.cuda.max_memory_allocated(0)>CFG['GPU_limit']:raise MemoryError('GPU allocated cap')
log({'PID':os.getpid(),'UTC':utc(),'phase':'critic_training','device':str(device),'torch':torch.__version__})
for arm in ['RAW','STANDARDIZED']:
    stage=ROOT/arm;stage.mkdir(exist_ok=False)
    x={k:(v if arm=='RAW'else(v-mu[None,:,:])/scale[None,:,:])for k,v in channels.items()}
    d=Critic().to(device);d.load_state_dict(initial);opt=torch.optim.Adam(d.parameters(),lr=CFG['lr'],betas=tuple(CFG['betas']))
    trace=[];best=float('inf');beststep=0;beststate=None;bestopt=None;last_loss=None;begin=time.monotonic()
    def validate(step):
        nonlocal_placeholder=None
        pr=chunk_score(d,x['real_val']);pg=chunk_score(d,x['gen_val_pre'])
        wr=pool['real_val_weight'];wg=pool['gen_val_weight']
        J=.5*(np.dot(wr,(pr.astype('f8')-1)**2)+np.dot(wg,pg.astype('f8')**2))
        row={'step':step,'elapsed_s':time.monotonic()-begin,'balanced_equalbin_MSE':float(J),'last_batch_D_loss':last_loss,
             'REAL_score_mean_unweighted':float(pr.mean()),'GEN_score_mean_unweighted':float(pg.mean()),
             'per_bin_MSE':[float(.5*(np.average((pr[pool['real_val_bin']==b].astype('f8')-1)**2,weights=wr[pool['real_val_bin']==b])+np.average(pg[pool['gen_val_bin']==b].astype('f8')**2,weights=wg[pool['gen_val_bin']==b])))for b in range(8)]}
        trace.append(row);log({'arm':arm,**row})
        return float(J)
    J=validate(0);best=J;beststate={k:v.detach().cpu().clone()for k,v in d.state_dict().items()};bestopt=copy.deepcopy(opt.state_dict())
    step=0
    for it in range(CFG['updates']):
        if time.monotonic()-begin>=CFG['train_s_per_arm']-2 or train_total+time.monotonic()-begin>=CFG['train_s_total']-2:break
        si=torch.from_numpy(pool['sampled_gen_index'][it].astype('i8')).to(device);ri=torch.from_numpy(pool['sampled_real_index'][it].astype('i8')).to(device)
        d.train();opt.zero_grad(set_to_none=True)
        real=d.net(x['real_train'][ri]).flatten(1).mean(1);gen=d.net(x['gen_train_pre'][si]).flatten(1).mean(1)
        loss=.5*((real-1).square().mean()+gen.square().mean())
        if not torch.isfinite(loss):raise RuntimeError('nonfinite D loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(d.parameters(),CFG['clip']);opt.step();last_loss=float(loss.detach());step=it+1;trained+=1;check_gpu()
        if step%250==0:
            d.eval();J=validate(step)
            if J<best:best=J;beststep=step;beststate={k:v.detach().cpu().clone()for k,v in d.state_dict().items()};bestopt=copy.deepcopy(opt.state_dict())
    if trace[-1]['step']!=step:
        d.eval();J=validate(step)
        if J<best:best=J;beststep=step;beststate={k:v.detach().cpu().clone()for k,v in d.state_dict().items()};bestopt=copy.deepcopy(opt.state_dict())
    arm_training_s=time.monotonic()-begin;train_total+=arm_training_s
    torch.save({'critic':d.state_dict(),'optimizer':opt.state_dict(),'step':step,'arm':arm},stage/'FINAL_CRITIC.pt')
    torch.save({'critic':beststate,'optimizer':bestopt,'step':beststep,'arm':arm,'mu':pool['mu'],'scale':pool['scale']},stage/'SELECTED_CRITIC.pt')
    for checkpoint in ['final','selected']:
        if checkpoint=='selected':d.load_state_dict(beststate)
        d.eval()
        for name,c in x.items():scores[arm+'_'+checkpoint+'_'+name]=chunk_score(d,c)
    for p in d.parameters():p.requires_grad_(False)
    v=data['gen_val_pre'][torch.from_numpy(selected_grad).to(device)].detach().clone().requires_grad_(True)
    c=input_channels(v)
    if arm=='STANDARDIZED':c=(c-mu[None,:,:])/scale[None,:,:]
    pred=d.net(c).flatten(1).mean(1);loss=(pred-1).square().mean();grad=torch.autograd.grad(loss,v)[0].detach().cpu().numpy()
    assert np.isfinite(grad).all()
    grads[arm]=grad
    alltrace[arm]={'updates':step,'selected_step':beststep,'selected_J':best,'terminal_J':trace[-1]['balanced_equalbin_MSE'],'training_elapsed_s':arm_training_s,'trace':trace,
      'gradient_loss':float(loss.detach()),'same_index_prefix':True,'completed_shared_stream_prefix':step}
    write('TRAINING_TRACE.json',alltrace)
    del x,d,opt,v,c;check_gpu()
np.savez_compressed(ROOT/'CRITIC_SCORES_PRIVATE.npz',**scores)
np.savez_compressed(ROOT/'INPUT_GRADIENTS_PRIVATE.npz',**grads,source_val_pool_index=selected_grad)
rec={'PID':os.getpid(),'UTC':utc(),'elapsed_s':time.monotonic()-START,'training_elapsed_s':train_total,'actual_D_updates':trained,'G_updates':0,'reader_updates':0,
     'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'device':str(device),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(0)if device.type=='cuda'else 0,
     'arms':{k:{n:v for n,v in r.items()if n!='trace'}for k,r in alltrace.items()},'environment_cross_version':'frozen G inferenceCPU2.10; critic trainingTorch2.12GPU; not bitwise old-online comparison'}
write('TRAINING_RECEIPT.json',rec);log(rec)
