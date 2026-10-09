from __future__ import annotations
import numpy as np,torch,time,json,hashlib
from model import Translator,Critic,charge_psd,physical,pooled_shape,rank_losses,nce,sw
from prior import weighted_quantile,RecordedChoice
from fine_core import FineLoss,NAMES

def state(model):return {k:v.detach().cpu().clone()for k,v in model.state_dict().items()}

def train(arm,source,target,source_val,target_val,h,hval,sb,tb,lambdas,edges,fine_constants,cfg,root,counter,log):
    root.mkdir(exist_ok=True);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']%(2**32));torch.cuda.manual_seed_all(cfg['seed'])
    device=torch.device('cuda:0'if torch.cuda.is_available()else'cpu');fine_loss=FineLoss(fine_constants,device)
    model=Translator().to(device);critic=Critic().to(device)
    old=root.parent/cfg['old_model_link'] if cfg.get('old_model_link')else __import__('pathlib').Path(cfg['remote_R18_root'])
    model.load_state_dict(torch.load(__import__('pathlib').Path(cfg['remote_R19_root'])/'TILT/TRANSLATOR.pt',map_location=device,weights_only=True));critic.load_state_dict(torch.load(old/'INITIAL_CRITIC.pt',map_location=device,weights_only=True))
    torch.save({'torch_CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state_all()if device.type=='cuda'else[],'seed':cfg['seed']},root/'INITIAL_RANDOM_STATES.pt')
    og=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=cfg['betas']);od=torch.optim.Adam(critic.parameters(),lr=cfg['lr'],betas=cfg['betas'])
    xs=torch.from_numpy(source).to(device);xt=torch.from_numpy(target).to(device);vs=torch.from_numpy(source_val).to(device);vt=torch.from_numpy(target_val).to(device)
    pools_s={i:np.flatnonzero(sb==i)for i in range(8)};pools_t={i:np.flatnonzero(tb==i)for i in range(8)};common=[i for i in range(8)if len(pools_s[i])and len(pools_t[i])]
    cdf={}
    for b in common:
        weights=np.exp(lambdas[b]*h[pools_s[b]]);cdf[b]=np.cumsum(weights/weights.sum());cdf[b][-1]=1.
    rngs=np.random.default_rng(cfg['train_source_seed']);rngt=np.random.default_rng(cfg['train_target_seed']);rngn=np.random.default_rng(cfg['train_noise_seed']);rngp=RecordedChoice(cfg['seed']+4)
    with np.load(old/'VALIDATION_INPUTS.npz',allow_pickle=False)as a:vv={k:a[k].copy()for k in a.files}
    if not np.array_equal(vv['source'],source_val):raise RuntimeError('frozen source validation identity/order disagreement')
    vals={k:torch.from_numpy(vv[k]).to(device)for k in ['z','white','physical_center','physical_scale','physical_projection','shape_projection']}
    vz=vals['z'];vw=vals['white'];center=vals['physical_center'];scale=vals['physical_scale'];projection=vals['physical_projection'];waveproj=vals['shape_projection']
    with torch.no_grad():
        qs=charge_psd(vs)[0];qt=charge_psd(vt)[0];ee=torch.as_tensor(edges,dtype=qs.dtype,device=device)
        vsb=torch.bucketize((qs.clamp_min(.01)*1000).log(),ee,right=True)-1;vtb=torch.bucketize((qt.clamp_min(.01)*1000).log(),ee,right=True)-1
        vcommon=[i for i in range(8)if(vsb==i).any()and(vtb==i).any()]
        vwgt=np.ones(len(source_val));vb=vsb.cpu().numpy()
        for b in range(8):vwgt[vb==b]=np.exp(lambdas[b]*hval[vb==b])
        weights=torch.as_tensor(vwgt,dtype=vs.dtype,device=device)
    validation_support=[{'bin_id_1based':b+1,'source_N':int((vb==b).sum()),'REAL_N':int((vtb==b).sum()),'Kish_ESS':float(vwgt[vb==b].sum()**2/np.dot(vwgt[vb==b],vwgt[vb==b]))if(vb==b).any()else None}for b in range(8)]
    @torch.no_grad()
    def validate(step):
        model.eval();y=torch.cat([model(vs[j:j+128],vz[j:j+128],vw[j:j+128])[0]for j in range(0,len(vs),128)])
        fy=(physical(y)-center)/scale;fr=(physical(vt)-center)/scale;levels=torch.linspace(.005,.995,128,device=device)
        rows=[];pf=[];pw=[];uf=[];uw=[]
        for b in vcommon:
            sm=vsb==b;tm=vtb==b
            fw=(weighted_quantile(fy[sm],weights[sm],levels)-torch.quantile(fr[tm],levels,dim=0)).abs().mean()
            wy=pooled_shape(y[sm])@waveproj;wr=pooled_shape(vt[tm])@waveproj
            ww=(weighted_quantile(wy,weights[sm],levels)-torch.quantile(wr,levels,dim=0)).abs().mean()
            fu=(torch.quantile(fy[sm],levels,dim=0)-torch.quantile(fr[tm],levels,dim=0)).abs().mean();wu=sw(pooled_shape(y[sm]),pooled_shape(vt[tm]),waveproj)
            pf.append(fw);pw.append(ww);uf.append(fu);uw.append(wu)
            rows.append({'bin_id_1based':b+1,'source_n':int(sm.sum()),'REAL_n':int(tm.sum()),'weighted_physical_W1':float(fw),'weighted_shape_SW':float(ww),'unweighted_physical_W1':float(fu),'unweighted_shape_SW':float(wu)})
        qrel,qr,pr,positive=rank_losses(vs,y,edges);qo,po,_=charge_psd(y);qi=charge_psd(vs)[0];qc=(qo.clamp_min(.01).log()-qi.clamp_min(.01).log()).abs().mean()
        f=torch.stack(pf).mean();w=torch.stack(pw).mean();u=torch.stack(uf).mean();v=torch.stack(uw).mean()
        fine_val,fine_groups,fine_bins=fine_loss.validation(y,vt,vsb,vtb,weights,vcommon)
        result={'step':step,'J':float(f+w+.5*qc+2*pr+fine_val),'J19_weighted':float(f+w+.5*qc+2*pr),'fine_val':float(fine_val),'fine_groups':fine_groups,'fine_per_bin':fine_bins,'original_unweighted_J':float(u+v+.5*qc+2*pr),'weighted_physical_W1':float(f),'weighted_shape_SW':float(w),'unweighted_physical_W1':float(u),'unweighted_shape_SW':float(v),'source_logQ_abs_change':float(qc),'PSD_rank_violation':float(pr),'Q_rank_violation':float(qr),'positive_charge_penalty':float(positive),'per_bin':rows,'validation_output_PSD_quantiles':np.quantile(po.cpu().numpy(),[0,.1,.5,.9,1]).tolist(),'validation_output_Q_quantiles_Vns':np.quantile(qo.cpu().numpy(),[0,.1,.5,.9,1]).tolist()}
        model.train();return result
    counter['response_fit_attempts']+=1;clock=time.monotonic();initial=validate(0);trace=[{'validation':initial,'elapsed_train_s':time.monotonic()-clock}];best=state(model);bestrow=initial;updates=0;d_updates=0;first_gradient_check=None
    streams={'source':[],'target':[],'bin':[],'source_uniform':[]};hdn=hashlib.sha256();initial_rng={'source':rngs.bit_generator.state,'target':rngt.bit_generator.state,'noise':rngn.bit_generator.state,'NCE':rngp.rng.bit_generator.state}
    for step in range(1,cfg['updates']+1):
        if time.monotonic()-clock>=cfg['training_wall_s']-5:break
        bins=rngs.choice(common,size=cfg['batch']);uu=rngs.random(cfg['batch'])
        si=np.asarray([pools_s[int(b)][min(len(pools_s[int(b)])-1,int(np.searchsorted(cdf[int(b)],u,side='right')))]for b,u in zip(bins,uu)],dtype='<u4')
        ti=np.asarray([rngt.choice(pools_t[int(b)])for b in bins],dtype='<u4');z=rngn.normal(size=(len(si),8)).astype('f4');white=rngn.normal(size=(len(si),480)).astype('f4')
        for k,a in [('source',si),('target',ti),('bin',bins.astype('u1')),('source_uniform',uu)]:streams[k].append(a.copy())
        hdn.update(z.tobytes());hdn.update(white.tobytes());x=xs[si.astype('i8')];real=xt[ti.astype('i8')];zz=torch.from_numpy(z).to(device);ww=torch.from_numpy(white).to(device)
        critic.requires_grad_(True);od.zero_grad(set_to_none=True)
        with torch.no_grad():fake=model(x,zz,ww)[0]
        ld=.5*((critic(real)-1).square().mean()+critic(fake).square().mean());ld.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg['gradient_clip']);od.step();counter['D_updates']+=1;d_updates+=1
        critic.requires_grad_(False);og.zero_grad(set_to_none=True);y=model(x,zz,ww)[0];gan=(critic(y)-1).square().mean();content=nce(model,x,y,cfg,rngp)
        identity=model(real,torch.zeros_like(zz),torch.zeros_like(ww))[0];ident=(identity-real).abs().mean()/.1;qrel,qr,pr,positive=rank_losses(x,y,edges)
        domain=sw((physical(y)-center)/scale,(physical(real)-center)/scale,projection)+sw(pooled_shape(y),pooled_shape(real),waveproj)
        components={'adversarial':gan,'content_NCE':content,'identity':ident,'Q_relative':qrel,'Q_rank':qr,'PSD_rank':pr,'domain_SW':domain,'positive_charge':positive}
        base_loss=sum(cfg['loss_weights'][k]*v for k,v in components.items());fine_value,fine_groups=fine_loss.training(y,real)
        if step==1:
            gy,gw=torch.autograd.grad(fine_value,(y,model.mean.weight),retain_graph=True)
            first_gradient_check={'fine_value':float(fine_value),'groups':{n:float(v)for n,v in zip(NAMES,fine_groups)},'output_gradient_all_finite':bool(torch.isfinite(gy).all()),'mean_weight_gradient_all_finite':bool(torch.isfinite(gw).all()),'output_gradient_L2':float(gy.norm()),'mean_weight_gradient_L2':float(gw.norm()),'nonzero':bool((gy!=0).any()and(gw!=0).any()),'gradient_query_not_optimizer_update':True}
            (root/'FIRST_FINE_GRADIENT_CHECK.json').write_text(json.dumps(first_gradient_check,indent=2)+'\n')
            if not(first_gradient_check['output_gradient_all_finite']and first_gradient_check['mean_weight_gradient_all_finite']and first_gradient_check['nonzero']):raise FloatingPointError('nonfinite or zero fine gradient')
        loss=base_loss+fine_value*cfg['fine_coefficient']if arm=='FINE'else base_loss
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite fixed-recipe loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);og.step();updates=step;counter['G_updates']+=1
        if device.type=='cuda'and torch.cuda.max_memory_allocated()>cfg['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
        if step%cfg['validation_every']==0:
            row=validate(step);record={'step':step,'D_loss':float(ld),'G_loss':float(loss),'base_G_loss':float(base_loss),'fine_train':float(fine_value),'fine_train_groups':{n:float(v)for n,v in zip(NAMES,fine_groups)},'fine_backpropagated':arm=='FINE','components':{k:float(v)for k,v in components.items()},'validation':row,'elapsed_train_s':time.monotonic()-clock};trace.append(record);log(arm+' train '+json.dumps(record))
            if row['J']<bestrow['J']:best=state(model);bestrow=row
            (root/'TRAINING_TRACE.json').write_text(json.dumps(trace,indent=2)+'\n')
    final=validate(updates)
    if final['J']<bestrow['J']:best=state(model);bestrow=final
    trace.append({'validation':final,'final':True,'elapsed_train_s':time.monotonic()-clock})
    torch.save(state(model),root/'FINAL_TRANSLATOR.pt');torch.save(state(critic),root/'FINAL_CRITIC.pt');torch.save(best,root/'TRANSLATOR.pt')
    torch.save({'G':og.state_dict(),'D':od.state_dict()},root/'OPTIMIZER_FINAL.pt')
    actual={k:np.concatenate(a)if a else np.array([])for k,a in streams.items()};actual['NCE_positions']=np.stack(rngp.records)if rngp.records else np.empty((0,12),dtype='<u2');np.savez_compressed(root/'TRAINING_STREAMS_PRIVATE.npz',**actual)
    hashes={k:hashlib.sha256(a.tobytes()).hexdigest()for k,a in actual.items()};hashes['noise_stream']=hdn.hexdigest()
    (root/'TRAINING_RANDOM_STATES.json').write_text(json.dumps({'initial':initial_rng,'final':{'source':rngs.bit_generator.state,'target':rngt.bit_generator.state,'noise':rngn.bit_generator.state,'NCE':rngp.rng.bit_generator.state},'noise_stream_sha256':hdn.hexdigest()},indent=2)+'\n')
    seen=actual['source'].astype('i8');counts=np.bincount(seen,minlength=len(source));elapsed=time.monotonic()-clock;counter['training_s']+=elapsed
    record={'arm':arm,'first_fine_gradient_check':first_gradient_check,'fine_coefficient_in_training':1. if arm=='FINE'else 0.,'updates':updates,'D_updates':d_updates,'selected_step':bestrow['step'],'initial_validation':initial,'selected_validation':bestrow,'final_validation':final,'training_s':elapsed,'validation_trace':trace,'validation_support':validation_support,'common_bins_1based':[b+1 for b in common],'stream_hashes':hashes,'actually_sampled_unique_source':int((counts>0).sum()),'sampled_source_outside_common':int(np.sum(~np.isin(sb[seen],common))),'sampled_source_affinity_mean':float(h[seen].mean())if len(seen)else None,'sampled_per_bin':[{'bin_id_1based':b+1,'draws':int((sb[seen]==b).sum()),'unique_source':int(np.sum((counts>0)&(sb==b))),'mean_h':float(h[seen[sb[seen]==b]].mean())if(sb[seen]==b).any()else None}for b in range(8)],'device':str(device),'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated()if device.type=='cuda'else 0,'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved()if device.type=='cuda'else 0,'selected_using_DEV':False,'source_target_unpaired':True}
    (root/'TRAINING_TRACE.json').write_text(json.dumps(trace,indent=2)+'\n');(root/'TRAINING_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
    result=Translator().cpu();result.load_state_dict(best);result.eval();del xs,xt,vs,vt,model,critic,og,od
    if device.type=='cuda':torch.cuda.empty_cache()
    return result,record,counts
