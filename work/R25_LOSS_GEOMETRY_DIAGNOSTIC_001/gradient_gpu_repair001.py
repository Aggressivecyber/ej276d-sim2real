from __future__ import annotations
import os,time,copy
START=time.monotonic()
import numpy as np,torch
from pathlib import Path
from diagnostic_support import ROOT,CFG,write,log,load,digest,check_output
from model import Translator,StyleReader,Critic,input_channels,charge_psd,physical,pooled_shape,rank_losses,nce,sw
from fine_core import FineLoss,NAMES
torch.set_num_threads(4);torch.set_num_interop_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.5,0)
r23=Path(CFG['R23_root']);r24=Path(CFG['R24_root'])
with np.load(r23/'STANDARDIZED/TRAINING_STREAMS_PRIVATE.npz',allow_pickle=False) as z:
    actual={k:z[k][:1024].copy() for k in ['source','target','bin']};actual['NCE_positions']=z['NCE_positions'][:96].copy()
assert actual['NCE_positions'].shape==(96,12)
rs=__import__('json').loads((r23/'STANDARDIZED/TRAINING_RANDOM_STATES.json').read_text('utf8'))
noise=np.random.default_rng();noise.bit_generator.state=copy.deepcopy(rs['noise_initial'])
style=np.random.default_rng();style.bit_generator.state=copy.deepcopy(rs['style_initial'])
z=np.empty((1024,8),'f4');white=np.empty((1024,480),'f4');sty=np.empty((1024,8),'f4')
for b in range(32):
    sl=slice(b*32,(b+1)*32);z[sl]=noise.normal(size=(32,8)).astype('f4');white[sl]=noise.normal(size=(32,480)).astype('f4');sty[sl]=style.normal(size=(32,8)).astype('f4')
data=load(r23/'TRAIN_PRIVATE.npz',['source','target','source_bin','target_bin','edges'])
assert np.array_equal(data['source_bin'][actual['source']],actual['bin']) and np.array_equal(data['target_bin'][actual['target']],actual['bin'])
x=data['source'][actual['source']].copy();real=data['target'][actual['target']].copy();edges=data['edges'].copy();del data
payload={'source_waveforms_V':x,'REAL_waveforms_V':real,'z':z,'white':white,'style':sty,'source_index':actual['source'],'REAL_index':actual['target'],'bin':actual['bin'],'NCE_positions':actual['NCE_positions']}
np.savez_compressed(ROOT/'B_ACTUAL_INPUTS_PRIVATE.npz',**payload)
write('B_INPUT_REUSE.json',{'R23_actual_consumed_stream_path':str(r23/'STANDARDIZED/TRAINING_STREAMS_PRIVATE.npz'),'R23_recorded_initial_rng':rs['noise_initial'],'style_initial':rs['style_initial'],'prefix_noise_final':noise.bit_generator.state,'prefix_style_final':style.bit_generator.state,'reconstruction_order':'per original batch: z[32,8], white[32,480]; separate style[32,8] PCG64','actual_stream_indices_not_resampled':True,'noise_arrays_reconstructed_from_saved_exact_initial_state':True,'rows':1024,'NCE_rows':96,'arrays_sha256':{k:digest(a) for k,a in payload.items()}})
val=load(r23/'VALIDATION_INPUTS.npz',['physical_center','physical_scale','physical_projection','shape_projection'])
with np.load(r23/'FINE_CONSTANTS.npz',allow_pickle=False) as zc:constants={k:zc[k].copy() for k in zc.files}
scaler=load(r23/'CRITIC_SCALER.npz',['mu','scale']);fine=FineLoss(constants,device)
center,scale,proj,waveproj=[torch.from_numpy(val[k]).to(device) for k in ['physical_center','physical_scale','physical_projection','shape_projection']]
xx,rr,zz,ww,ss=[torch.from_numpy(a).to(device) for a in [x,real,z,white,sty]]
class Choice:
    def __init__(self,a):self.rows=a;self.at=0
    def choice(self,options,size,replace=False):
        a=self.rows[self.at];self.at+=1
        assert not replace and len(a)==size and len(np.unique(a))==size and np.isin(a,list(options)).all()
        return a.astype('i8')
class InputCritic(Critic):
    def __init__(self):
        super().__init__();self.register_buffer('input_mu',torch.as_tensor(scaler['mu'],dtype=torch.float32,device=device),persistent=False);self.register_buffer('input_scale',torch.as_tensor(scaler['scale'],dtype=torch.float32,device=device),persistent=False)
    def forward(self,x):return self.net((input_channels(x)-self.input_mu[None])/self.input_scale[None]).flatten(1).mean(1)
states=[('INITIAL',CFG['G_initial'],CFG['reader_initial'],CFG['D_initial']),('STD_D1_FINAL2950',str(r23/'STANDARDIZED/FINAL_TRANSLATOR.pt'),str(r23/'STANDARDIZED/FINAL_STYLE_READER.pt'),str(r23/'STANDARDIZED/FINAL_CRITIC.pt')),('STD_D5_FINAL2950',str(r24/'D5/FINAL_TRANSLATOR.pt'),str(r24/'D5/FINAL_STYLE_READER.pt'),str(r24/'D5/FINAL_CRITIC.pt'))]
weights=CFG['B']['weights'];terms=CFG['B']['losses'];counter={'source_forward_rows':0,'identity_forward_rows':0,'NCE_source_encoding_rows':0,'NCE_target_encoding_rows':0,'critic_forward_rows':0,'reader_forward_rows':0,'batches_complete':0,'optimizer_steps':0,'G_D_reader_updates':0}
def group(n):
    if n.startswith(('e0.','e1.','e2.')):return 'encoder'
    if n.startswith('project.'):return 'NCE_project'
    if n.startswith(('d0.','d1.','mean.')):return 'decoder_mean'
    if n.startswith('style_film.'):return 'style_FiLM'
    if n.startswith('noise.'):return 'electronic'
    raise RuntimeError('unassigned G parameter '+n)
def metrics(mat,ng,masks):
    g=mat[:,:ng];total=g.sum(0);norm=g.norm(dim=1);dot=g@total;tn=total.norm();den=norm*tn
    cosine=(g@g.T)/(norm[:,None]*norm[None,:]);cosine=torch.where((norm[:,None]*norm[None,:])>0,cosine,torch.full_like(cosine,float('nan')))
    aligned=torch.where(den>0,dot/den,torch.full_like(dot,float('nan')))
    grouped={}
    for key,mask in masks.items():
        a=mat[:,mask];s=a.sum(0);nn=a.norm(dim=1);dd=a@s;ssn=s.norm()
        grouped[key]={'term_norm':nn.cpu().tolist(),'total_norm':float(ssn),'term_dot_total':dd.cpu().tolist(),'term_cos_total':torch.where(nn*ssn>0,dd/(nn*ssn),torch.full_like(dd,float('nan'))).cpu().tolist()}
    whole=mat.sum(0);combined=float(whole.norm())
    return {'G_term_norm':norm.cpu().tolist(),'G_total_norm':float(tn),'G_cosine_matrix':cosine.cpu().tolist(),'G_term_dot_total':dot.cpu().tolist(),'G_term_cos_total':aligned.cpu().tolist(),'parameter_groups':grouped,'G_reader_total_norm':combined,'clip5_reference_scale':min(1.,5/combined) if combined>0 else 1.,'clip_was_applied':False}
allresults=[]
try:
    for label,gpath,rpath,dpath in states:
        model=Translator().to(device);reader=StyleReader().to(device);critic=InputCritic().to(device)
        for m,p in [(model,gpath),(reader,rpath),(critic,dpath)]:m.load_state_dict(torch.load(p,map_location=device,weights_only=True));m.train()
        critic.requires_grad_(False)
        params=[*model.parameters(),*reader.parameters()];names=[*model.named_parameters()]
        ng=sum(p.numel() for p in model.parameters());totaln=sum(p.numel() for p in params)
        ranges={};at=0
        for n,p in names:
            key=group(n);ranges.setdefault(key,[]).append((at,at+p.numel()));at+=p.numel()
        ranges['reader']=[(ng,totaln)];masks={k:torch.cat([torch.arange(a,b,device=device) for a,b in spans]) for k,spans in ranges.items()}
        original=[p.detach().clone() for p in [*model.parameters(),*reader.parameters(),*critic.parameters()]]
        rng=Choice(actual['NCE_positions']);rows=[];check=None
        for b in range(32):
            sl=slice(b*32,(b+1)*32);x=xx[sl];real=rr[sl];z=zz[sl];white=ww[sl];style=ss[sl]
            counter['source_forward_rows']+=32;y,signal,_=model(x,z,white,style)
            counter['critic_forward_rows']+=32;gan=(critic(y)-1).square().mean()
            counter['NCE_source_encoding_rows']+=32;counter['NCE_target_encoding_rows']+=32;content=nce(model,x,y,CFG['original_loss_config'],rng)
            counter['identity_forward_rows']+=32;identity=model(real,torch.zeros_like(z),torch.zeros_like(white),torch.zeros_like(style))[0];ident=(identity-real).abs().mean()/.1
            qrel,qr,pr,positive=rank_losses(x,y,edges)
            domain=sw((physical(y)-center)/scale,(physical(real)-center)/scale,proj)+sw(pooled_shape(y),pooled_shape(real),waveproj)
            finevalue,_=fine.training(y,real)
            counter['reader_forward_rows']+=32;styleloss=(reader(x,signal)-style).abs().mean()
            losses=[gan,content,ident,qrel,qr,pr,domain,positive,finevalue,styleloss]
            weighted=[w*l for w,l in zip(weights,losses)]
            if not all(torch.isfinite(l) for l in weighted):raise FloatingPointError('nonfinite component')
            grads=[]
            for loss in weighted:
                derivative=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                grads.append(torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p,g in zip(params,derivative)]))
            mat=torch.stack(grads);assert torch.isfinite(mat).all()
            if b==0:
                direct=torch.autograd.grad(sum(weighted),params,retain_graph=False,allow_unused=True)
                direct=torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p,g in zip(params,direct)])
                summation=mat.sum(0);error=summation-direct
                check={'float32_max_abs_error':float(error.abs().max()),'L2_error':float(error.norm()),'relative_L2_error':float(error.norm()/direct.norm().clamp_min(1e-30)),'direct_total_L2':float(direct.norm()),'sum_component_L2':float(summation.norm()),'finite':bool(torch.isfinite(direct).all()),'allclose_atol1e-5_rtol1e-4':bool(torch.allclose(summation,direct,atol=1e-5,rtol=1e-4))}
            row={'batch_1based':b+1,'unweighted_loss':[float(l.detach()) for l in losses],'weighted_loss':[float(l.detach()) for l in weighted],**metrics(mat,ng,masks)}
            rows.append(row);counter['batches_complete']+=1
            del y,signal,identity,losses,weighted,grads,mat,derivative
            if device.type=='cuda' and torch.cuda.max_memory_allocated()>CFG['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
            if (b+1)%8==0:log({'state':label,'batches_complete':b+1,'elapsed_s':time.monotonic()-START})
        unchanged=all(torch.equal(a,p) for a,p in zip(original,[*model.parameters(),*reader.parameters(),*critic.parameters()]))
        assert rng.at==96 and unchanged
        record={'state':label,'G_path':gpath,'reader_path':rpath,'D_path':dpath,'initial_D_untrained':label=='INITIAL','G_parameter_count':ng,'reader_parameter_count':totaln-ng,'parameters_unchanged':unchanged,'gradient_sum_check_first_batch':check,'batches':rows}
        allresults.append(record);write('B_GRADIENTS_PRIVATE.json',{'loss_names':terms,'weights':weights,'counter':counter,'states':allresults})
        del model,reader,critic,params,original
    rec={'PID':os.getpid(),'UTC':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'counter':counter,'torch':torch.__version__,'numpy':np.__version__,'device':str(device),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,'model_mode':'train, no BN/Dropout or mutable training state','D_requires_grad':False,'input_grad_through_D':True,'saved_models_modified':False,'optimizer_constructed':False,'direct_total_gradient_checks':3,'status':'complete'}
    write('B_NUMERIC_RECEIPT.json',rec);check_output();log(rec)
except BaseException as e:
    write('B_FAILURE.json',{'error':repr(e),'counter':counter,'elapsed_s':time.monotonic()-START,'completed_states':len(allresults)});raise
