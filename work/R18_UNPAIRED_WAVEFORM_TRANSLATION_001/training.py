from __future__ import annotations
import numpy as np,torch,time,copy,hashlib,json
import torch.nn.functional as F
from model import Translator,Critic,charge_psd,physical,pooled_shape,rank_losses,nce,sw

def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def train(source,target,source_val,target_val,edges,cfg,root,counter,log):
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']%(2**32));torch.cuda.manual_seed_all(cfg['seed'])
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=Translator().to(device);critic=Critic().to(device)
    torch.save({'torch_CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state_all() if device.type=='cuda'else [],'seed':cfg['seed']},root/'INITIAL_RANDOM_STATES.pt')
    torch.save(state(model),root/'INITIAL_TRANSLATOR.pt');torch.save(state(critic),root/'INITIAL_CRITIC.pt')
    og=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=cfg['betas']);od=torch.optim.Adam(critic.parameters(),lr=cfg['lr'],betas=cfg['betas'])
    xs=torch.from_numpy(source).to(device);xt=torch.from_numpy(target).to(device);vs=torch.from_numpy(source_val).to(device);vt=torch.from_numpy(target_val).to(device)
    with torch.no_grad():
        ft=physical(xt);center=ft.cpu().median(0).values.to(device);scale=ft.std(0).clamp_min(.05)
        qs=charge_psd(xs)[0].cpu().numpy();qt=charge_psd(xt)[0].cpu().numpy()
    sb=np.searchsorted(edges,np.log(np.maximum(qs*1000,1e-30)),side='right')-1;tb=np.searchsorted(edges,np.log(np.maximum(qt*1000,1e-30)),side='right')-1
    pools_s={i:np.flatnonzero(sb==i) for i in range(8)};pools_t={i:np.flatnonzero(tb==i) for i in range(8)};common=[i for i in range(8) if len(pools_s[i]) and len(pools_t[i])]
    if not common:raise RuntimeError('no common TRAIN charge-bin support')
    rngs=np.random.default_rng(cfg['train_source_seed']);rngt=np.random.default_rng(cfg['train_target_seed']);rngn=np.random.default_rng(cfg['train_noise_seed']);rngp=np.random.default_rng(cfg['seed']+4)
    rngv=np.random.default_rng(cfg['validation_seed']);vz=torch.from_numpy(rngv.normal(size=(len(vs),8)).astype('f4')).to(device);vw=torch.from_numpy(rngv.normal(size=(len(vs),480)).astype('f4')).to(device)
    projection=torch.from_numpy(rngv.normal(size=(6,32)).astype('f4')).to(device);projection=F.normalize(projection,dim=0)
    waveproj=torch.from_numpy(rngv.normal(size=(60,32)).astype('f4')).to(device);waveproj=F.normalize(waveproj,dim=0)
    np.savez_compressed(root/'VALIDATION_INPUTS.npz',source=source_val,z=vz.cpu().numpy(),white=vw.cpu().numpy(),physical_center=center.cpu().numpy(),physical_scale=scale.cpu().numpy(),physical_projection=projection.cpu().numpy(),shape_projection=waveproj.cpu().numpy())
    with torch.no_grad():
        vsq=charge_psd(vs)[0];vtq=charge_psd(vt)[0]
        vedges=torch.as_tensor(edges,device=device,dtype=vs.dtype)
        vsourcebin=torch.bucketize((vsq.clamp_min(.01)*1000).log(),vedges,right=True)-1
        vtargetbin=torch.bucketize((vtq.clamp_min(.01)*1000).log(),vedges,right=True)-1
        vcommon=[i for i in range(8) if (vsourcebin==i).any() and (vtargetbin==i).any()]
    if not vcommon:raise RuntimeError('no common frozen validation charge bin')
    @torch.no_grad()
    def validate(step):
        model.eval();parts=[]
        for j in range(0,len(vs),128):parts.append(model(vs[j:j+128],vz[j:j+128],vw[j:j+128])[0])
        y=torch.cat(parts);fy=(physical(y)-center)/scale;freal=(physical(vt)-center)/scale
        quant=torch.linspace(.005,.995,128,device=device);fv=[];wv=[];perbin=[]
        for i in vcommon:
            sm=vsourcebin==i;tm=vtargetbin==i
            one=(torch.quantile(fy[sm],quant,dim=0)-torch.quantile(freal[tm],quant,dim=0)).abs().mean()
            two=sw(pooled_shape(y[sm]),pooled_shape(vt[tm]),waveproj);fv.append(one);wv.append(two)
            perbin.append({'bin_id_1based':i+1,'source_n':int(sm.sum()),'REAL_n':int(tm.sum()),'physical_W1':float(one),'shape_SW':float(two)})
        feat=torch.stack(fv).mean();wave=torch.stack(wv).mean();qrel,qr,pr,positive=rank_losses(vs,y,edges)
        qin=charge_psd(vs)[0].clamp_min(.01);qout=charge_psd(y)[0].clamp_min(.01);qchange=(qout.log()-qin.log()).abs().mean()
        j=feat+wave+.5*qchange+2*pr
        row={'step':step,'J':float(j),'physical_feature_W1':float(feat),'shape_SW':float(wave),'source_logQ_abs_change':float(qchange),'PSD_rank_violation':float(pr),'Q_rank_violation':float(qr),'positive_charge_penalty':float(positive),'source_n':len(vs),'REAL_n':len(vt),'equal_weight_common_source_charge_bins':perbin,'unsupported_bins_1based':[i+1 for i in range(8)if i not in vcommon]}
        model.train();return row
    clock=time.monotonic();initial=validate(0);trace=[{'validation':initial,'elapsed_train_s':time.monotonic()-clock}];best=state(model);bestrow=initial;hdx=hashlib.sha256();hdt=hashlib.sha256();hdn=hashlib.sha256();updates=0;error=None
    counter['response_fit_attempts']+=1
    try:
        for step in range(1,cfg['updates']+1):
            if time.monotonic()-clock>=cfg['training_wall_s']-5:break
            bins=rngs.choice(common,size=cfg['batch']);si=np.array([rngs.choice(pools_s[int(b)]) for b in bins]);ti=np.array([rngt.choice(pools_t[int(b)]) for b in bins]);z=rngn.normal(size=(len(si),8)).astype('f4');w=rngn.normal(size=(len(si),480)).astype('f4')
            hdx.update(si.tobytes());hdt.update(ti.tobytes());hdn.update(z.tobytes());hdn.update(w.tobytes());x=xs[si];real=xt[ti];zz=torch.from_numpy(z).to(device);ww=torch.from_numpy(w).to(device)
            critic.requires_grad_(True);od.zero_grad(set_to_none=True)
            with torch.no_grad():fake=model(x,zz,ww)[0]
            ld=.5*((critic(real)-1).square().mean()+critic(fake).square().mean());ld.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg['gradient_clip']);od.step();counter['D_updates']+=1
            critic.requires_grad_(False);og.zero_grad(set_to_none=True);y=model(x,zz,ww)[0]
            gan=(critic(y)-1).square().mean();content=nce(model,x,y,cfg,rngp);identity=model(real,torch.zeros_like(zz),torch.zeros_like(ww))[0];ident=(identity-real).abs().mean()/.1
            qrel,qr,pr,positive=rank_losses(x,y,edges);domain=sw((physical(y)-center)/scale,(physical(real)-center)/scale,projection)+sw(pooled_shape(y),pooled_shape(real),waveproj)
            components={'adversarial':gan,'content_NCE':content,'identity':ident,'Q_relative':qrel,'Q_rank':qr,'PSD_rank':pr,'domain_SW':domain,'positive_charge':positive}
            loss=sum(cfg['loss_weights'][k]*v for k,v in components.items())
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss; recipe not changed')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);og.step();updates=step;counter['G_updates']=step
            if device.type=='cuda' and torch.cuda.max_memory_allocated()>cfg['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated limit')
            if step%cfg['validation_every']==0:
                row=validate(step);record={'step':step,'D_loss':float(ld),'G_loss':float(loss),'components':{k:float(v)for k,v in components.items()},'validation':row,'elapsed_train_s':time.monotonic()-clock};trace.append(record);log('train '+json.dumps(record))
                if row['J']<bestrow['J']:best=state(model);bestrow=row
                (root/'TRAINING_TRACE.json').write_text(json.dumps(trace,indent=2)+'\n')
    except BaseException as ex:
        error=repr(ex)
        if not isinstance(ex,TimeoutError):raise
    final=validate(updates)
    if final['J']<bestrow['J']:best=state(model);bestrow=final
    trace.append({'validation':final,'final':True,'elapsed_train_s':time.monotonic()-clock})
    torch.save(state(model),root/'FINAL_TRANSLATOR.pt');torch.save(state(critic),root/'FINAL_CRITIC.pt');torch.save({'G':og.state_dict(),'D':od.state_dict()},root/'OPTIMIZER_FINAL.pt');torch.save({'torch_CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state_all()if device.type=='cuda'else [],'source_rng':rngs.bit_generator.state,'target_rng':rngt.bit_generator.state,'noise_rng':rngn.bit_generator.state},root/'TRAINING_RANDOM_STATES.pt');torch.save(best,root/'TRANSLATOR.pt')
    counter['training_s']=time.monotonic()-clock
    record={'updates':updates,'D_updates':counter['D_updates'],'selected_step':bestrow['step'],'selected_validation':bestrow,'initial_validation':initial,'final_validation':final,'error':error,'training_s':counter['training_s'],'validation_trace':trace,'source_index_stream_sha256':hdx.hexdigest(),'target_index_stream_sha256':hdt.hexdigest(),'noise_stream_sha256':hdn.hexdigest(),'common_TRAIN_charge_bins_1based':[i+1 for i in common],'source_pool_counts':{str(i+1):len(v)for i,v in pools_s.items()},'target_pool_counts':{str(i+1):len(v)for i,v in pools_t.items()},'source_outside_bins':int(((sb<0)|(sb>7)).sum()),'target_outside_bins':int(((tb<0)|(tb>7)).sum()),'device':str(device),'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated()if device.type=='cuda'else 0,'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved()if device.type=='cuda'else 0,'source_target_events_unpaired':True,'DEV_selection':False}
    (root/'TRAINING_TRACE.json').write_text(json.dumps(trace,indent=2)+'\n');(root/'TRAINING_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
    result=Translator().cpu();result.load_state_dict(best);result.eval();del xs,xt,vs,vt,model,critic,og,od
    if device.type=='cuda':torch.cuda.empty_cache()
    return result,record
