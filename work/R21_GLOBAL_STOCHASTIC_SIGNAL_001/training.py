from __future__ import annotations
import numpy as np,torch,time,json,hashlib,pathlib
from model import Translator,LegacyTranslator,StyleReader,Critic,charge_psd,physical,pooled_shape,rank_losses,nce,sw,adc
from prior import weighted_quantile
from fine_core import FineLoss,NAMES

def state(model):return{k:v.detach().cpu().clone()for k,v in model.state_dict().items()}

class ActualChoice:
    def __init__(self,rows):self.rows=rows;self.at=0;self.records=[]
    def choice(self,options,size,replace=False):
        if self.at>=len(self.rows):raise RuntimeError('exhausted frozen NCE stream')
        x=self.rows[self.at];self.at+=1
        assert len(x)==size and not replace and len(np.unique(x))==len(x)and np.isin(x,np.asarray(list(options))).all()
        self.records.append(x.copy());return x.astype(np.int64,copy=False)

def train(arm,source,target,source_val,target_val,h,hval,sb,tb,lambdas,edges,constants,cfg,root,counter,log):
    root.mkdir(exist_ok=True);device=torch.device('cuda:0'if torch.cuda.is_available()else'cpu')
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    # All inherited weights are overwritten from their frozen checkpoints. New modules use a dedicated RNG context.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['new_module_seed']);model=Translator();reader=StyleReader()
    oldG=torch.load(pathlib.Path(cfg['remote_R19_root'])/'TILT/TRANSLATOR.pt',map_location='cpu',weights_only=True)
    missing,unexpected=model.load_state_dict(oldG,strict=False)
    assert not unexpected and all(k.startswith('style_film.')for k in missing)
    model=model.to(device);reader=reader.to(device);critic=Critic().to(device)
    critic.load_state_dict(torch.load(pathlib.Path(cfg['remote_R18_root'])/'INITIAL_CRITIC.pt',map_location=device,weights_only=True))
    reference=pathlib.Path(cfg['remote_R20_root']);oldrecord=json.loads((reference/'FINE/TRAINING_RECEIPT.json').read_text())
    with np.load(reference/'FINE/TRAINING_STREAMS_PRIVATE.npz',allow_pickle=False)as a:actual={k:a[k].copy()for k in a.files}
    nold=oldrecord['updates'];batch=cfg['batch'];assert nold==4000 and len(actual['source'])==nold*batch
    rs=json.loads((reference/'FINE/TRAINING_RANDOM_STATES.json').read_text())
    rngnoise=np.random.default_rng(0);rngnoise.bit_generator.state=rs['initial']['noise']
    rngstyle=np.random.default_rng(cfg['style_train_seed']);rngnce=ActualChoice(actual['NCE_positions'])
    fine=FineLoss(constants,device)
    xs=torch.from_numpy(source).to(device);xt=torch.from_numpy(target).to(device);vs=torch.from_numpy(source_val).to(device);vt=torch.from_numpy(target_val).to(device)
    with np.load(pathlib.Path(cfg['remote_R18_root'])/'VALIDATION_INPUTS.npz',allow_pickle=False)as a:vv={k:a[k].copy()for k in a.files}
    assert np.array_equal(vv['source'],source_val)
    vals={k:torch.from_numpy(vv[k]).to(device)for k in ['z','white','physical_center','physical_scale','physical_projection','shape_projection']}
    vz=vals['z'];vw=vals['white'];center=vals['physical_center'];scale=vals['physical_scale'];proj=vals['physical_projection'];waveproj=vals['shape_projection']
    vstyle_np=np.random.default_rng(cfg['style_val_seed']).normal(size=(len(source_val),8)).astype('f4');vstyle=torch.from_numpy(vstyle_np).to(device)
    np.savez_compressed(root/'STYLE_VALIDATION_INPUTS.npz',style=vstyle_np)
    with torch.no_grad():
        qs=charge_psd(vs)[0];qt=charge_psd(vt)[0];ee=torch.as_tensor(edges,dtype=qs.dtype,device=device)
        vsb=torch.bucketize((qs.clamp_min(.01)*1000).log(),ee,right=True)-1;vtb=torch.bucketize((qt.clamp_min(.01)*1000).log(),ee,right=True)-1
        vcommon=[i for i in range(8)if(vsb==i).any()and(vtb==i).any()]
        vb=vsb.cpu().numpy();vwgt=np.ones(len(source_val))
        for b in range(8):vwgt[vb==b]=np.exp(lambdas[b]*hval[vb==b])
        weights=torch.as_tensor(vwgt,dtype=vs.dtype,device=device)
    support=[{'bin_id_1based':b+1,'source_N':int((vb==b).sum()),'REAL_N':int((vtb==b).sum()),'Kish_ESS':float(vwgt[vb==b].sum()**2/np.dot(vwgt[vb==b],vwgt[vb==b]))if(vb==b).any()else None}for b in range(8)]

    # The same unchanged local model/weights already passed 96 structural rows
    # before failed_002 reached the NCE indexing interface. Reuse that evidence.
    eqpath=pathlib.Path(__file__).resolve().parent/'failed_002/GLOBAL_STYLE/STRUCTURAL_EQUIVALENCE.json'
    equivalence=json.loads(eqpath.read_text('utf8'))
    assert equivalence['pass'] and equivalence['all_inherited_parameters_equal']
    assert all(torch.equal(model.state_dict()[k].cpu(),v)for k,v in oldG.items())
    assert bool((model.style_film[-1].weight==0).all()) and bool((model.style_film[-1].bias==0).all())
    equivalence={**equivalence,'reused_after_NCE_dtype_only_repair':True,'new_structural_forward_rows':0,'evidence_path':str(eqpath)}
    (root/'STRUCTURAL_EQUIVALENCE.json').write_text(json.dumps(equivalence,indent=2)+'\n')
    model.train()
    torch.save(state(model),root/'INITIAL_TRANSLATOR.pt');torch.save(state(reader),root/'INITIAL_STYLE_READER.pt')
    torch.save({'torch_CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state_all()if device.type=='cuda'else[],'new_module_seed':cfg['new_module_seed']},root/'INITIAL_RANDOM_STATES.pt')
    og=torch.optim.Adam([*model.parameters(),*reader.parameters()],lr=cfg['lr'],betas=cfg['betas']);od=torch.optim.Adam(critic.parameters(),lr=cfg['lr'],betas=cfg['betas'])

    @torch.no_grad()
    def validate(step):
        model.eval();reader.eval();yy=[];ss=[]
        for j in range(0,len(vs),128):
            y,s,_=model(vs[j:j+128],vz[j:j+128],vw[j:j+128],vstyle[j:j+128]);yy.append(y);ss.append(s)
        y=torch.cat(yy);signal=torch.cat(ss);style_error=(reader(vs,signal)-vstyle).abs().mean()
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
        _,qr,pr,positive=rank_losses(vs,y,edges);qo,po,_=charge_psd(y);qi=charge_psd(vs)[0];qc=(qo.clamp_min(.01).log()-qi.clamp_min(.01).log()).abs().mean()
        f=torch.stack(pf).mean();w=torch.stack(pw).mean();u=torch.stack(uf).mean();v=torch.stack(uw).mean();fine_val,fg,fb=fine.validation(y,vt,vsb,vtb,weights,vcommon)
        r={'step':step,'J':float(f+w+.5*qc+2*pr+fine_val),'J19_weighted':float(f+w+.5*qc+2*pr),'fine_val':float(fine_val),'fine_groups':fg,'fine_per_bin':fb,'original_unweighted_J':float(u+v+.5*qc+2*pr),'weighted_physical_W1':float(f),'weighted_shape_SW':float(w),'unweighted_physical_W1':float(u),'unweighted_shape_SW':float(v),'source_logQ_abs_change':float(qc),'PSD_rank_violation':float(pr),'Q_rank_violation':float(qr),'positive_charge_penalty':float(positive),'per_bin':rows,'style_reconstruction_L1':float(style_error),'validation_output_PSD_quantiles':np.quantile(po.cpu().numpy(),[0,.1,.5,.9,1]).tolist()}
        model.train();reader.train();return r

    counter['response_fit_attempts']+=1;clock=time.monotonic();initial=validate(0)
    old_initial=oldrecord['initial_validation'];delta=abs(initial['J']-old_initial['J'])
    # Hardware/library differences are recorded, not used as a new identity gate. Structural equivalence is checked in this local environment.
    trace=[{'validation':initial,'elapsed_train_s':time.monotonic()-clock}];best=state(model);best_reader=state(reader);bestrow=initial
    styles_saved=[];noise_hash=hashlib.sha256();updates=0;d_updates=0;first=None
    for step in range(1,min(cfg['updates'],nold)-1+1):
        if time.monotonic()-clock>=cfg['training_wall_s']-5-10.956509799987543:break
        sl=slice((step-1)*batch,step*batch);si=actual['source'][sl].astype('i8');ti=actual['target'][sl].astype('i8');bins=actual['bin'][sl]
        assert np.array_equal(sb[si],bins)and np.array_equal(tb[ti],bins)
        z=rngnoise.normal(size=(batch,8)).astype('f4');white=rngnoise.normal(size=(batch,480)).astype('f4');s=rngstyle.normal(size=(batch,8)).astype('f4')
        styles_saved.append(s);noise_hash.update(z.tobytes());noise_hash.update(white.tobytes())
        x=xs[si];real=xt[ti];zz=torch.from_numpy(z).to(device);ww=torch.from_numpy(white).to(device);sty=torch.from_numpy(s).to(device)
        critic.requires_grad_(True);od.zero_grad(set_to_none=True)
        with torch.no_grad():fake=model(x,zz,ww,sty)[0]
        ld=.5*((critic(real)-1).square().mean()+critic(fake).square().mean());ld.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg['gradient_clip']);od.step();counter['D_updates']+=1;d_updates+=1
        critic.requires_grad_(False);og.zero_grad(set_to_none=True);y,signal,_=model(x,zz,ww,sty);gan=(critic(y)-1).square().mean();content=nce(model,x,y,cfg,rngnce)
        identity=model(real,torch.zeros_like(zz),torch.zeros_like(ww),torch.zeros_like(sty))[0];ident=(identity-real).abs().mean()/.1;qrel,qr,pr,positive=rank_losses(x,y,edges)
        domain=sw((physical(y)-center)/scale,(physical(real)-center)/scale,proj)+sw(pooled_shape(y),pooled_shape(real),waveproj)
        components={'adversarial':gan,'content_NCE':content,'identity':ident,'Q_relative':qrel,'Q_rank':qr,'PSD_rank':pr,'domain_SW':domain,'positive_charge':positive}
        base_loss=sum(cfg['loss_weights'][k]*v for k,v in components.items());fine_value,fine_groups=fine.training(y,real)
        style_loss=(reader(x,signal)-sty).abs().mean();loss=base_loss+fine_value+cfg['style_reconstruction_weight']*style_loss
        if step==1:
            grad=torch.autograd.grad(style_loss,model.style_film[-1].weight,retain_graph=True)[0]
            first={'style_L1':float(style_loss),'modulation_gradient_finite':bool(torch.isfinite(grad).all()),'modulation_gradient_nonzero':bool((grad!=0).any()),'modulation_gradient_L2':float(grad.norm()),'query_not_extra_update':True}
            (root/'FIRST_STYLE_GRADIENT_CHECK.json').write_text(json.dumps(first,indent=2)+'\n')
            if not(first['modulation_gradient_finite']and first['modulation_gradient_nonzero']):raise FloatingPointError('style modulation gradient invalid')
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite fixed loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([*model.parameters(),*reader.parameters()],cfg['gradient_clip']);og.step()
        updates=step;counter['G_updates']+=1;counter['style_reader_updates']+=1
        if device.type=='cuda'and torch.cuda.max_memory_allocated()>cfg['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
        if step%cfg['validation_every']==0:
            val=validate(step);row={'step':step,'G_loss':float(loss),'D_loss':float(ld),'base_G_loss':float(base_loss),'fine_train':float(fine_value),'fine_train_groups':{n:float(v)for n,v in zip(NAMES,fine_groups)},'style_L1':float(style_loss),'components':{k:float(v)for k,v in components.items()},'validation':val,'elapsed_train_s':time.monotonic()-clock};trace.append(row);log(arm+' '+json.dumps(row))
            if val['J']<bestrow['J']:best=state(model);best_reader=state(reader);bestrow=val
            (root/'TRAINING_TRACE.json').write_text(json.dumps(trace,indent=2)+'\n')
    final=validate(updates)
    if final['J']<bestrow['J']:best=state(model);best_reader=state(reader);bestrow=final
    trace.append({'final':True,'validation':final,'elapsed_train_s':time.monotonic()-clock})
    torch.save(state(model),root/'FINAL_TRANSLATOR.pt');torch.save(state(reader),root/'FINAL_STYLE_READER.pt');torch.save(state(critic),root/'FINAL_CRITIC.pt');torch.save(best,root/'TRANSLATOR.pt');torch.save(best_reader,root/'STYLE_READER.pt');torch.save({'G_and_reader':og.state_dict(),'D':od.state_dict()},root/'OPTIMIZER_FINAL.pt')
    count=updates*batch;prefix={k:v[:count]for k,v in actual.items()if k!='NCE_positions'};prefix['NCE_positions']=np.asarray(rngnce.records,dtype='<u2');prefix['style']=np.concatenate(styles_saved)if styles_saved else np.empty((0,8),dtype='f4')
    np.savez_compressed(root/'TRAINING_STREAMS_PRIVATE.npz',**prefix)
    hashes={k:hashlib.sha256(a.tobytes()).hexdigest()for k,a in prefix.items()};hashes['noise_stream']=noise_hash.hexdigest()
    oldkeys=['source','target','bin','source_uniform','NCE_positions','noise_stream']
    full_match={k:hashes[k]==oldrecord['stream_hashes'][k]for k in oldkeys}if updates==nold else {k:None for k in oldkeys}
    expected_prefix={k:hashlib.sha256(actual[k][:count].tobytes()).hexdigest()for k in oldkeys if k not in ['noise_stream','NCE_positions']}
    expected_prefix['NCE_positions']=hashlib.sha256(actual['NCE_positions'][:3*updates].tobytes()).hexdigest()
    replay_noise=np.random.default_rng(0);replay_noise.bit_generator.state=rs['initial']['noise'];noise_expected=hashlib.sha256()
    for _ in range(updates):
        noise_expected.update(replay_noise.normal(size=(batch,8)).astype('f4').tobytes());noise_expected.update(replay_noise.normal(size=(batch,480)).astype('f4').tobytes())
    expected_prefix['noise_stream']=noise_expected.hexdigest()
    prefix_match={k:hashes[k]==expected_prefix[k]for k in oldkeys}
    if not all(prefix_match.values()):raise RuntimeError('completed old stream prefix mismatch')
    if updates==nold and not all(full_match.values()):raise RuntimeError('actual old stream mismatch')
    (root/'TRAINING_RANDOM_STATES.json').write_text(json.dumps({'old_noise_initial':rs['initial']['noise'],'old_noise_final':rngnoise.bit_generator.state,'style_seed':cfg['style_train_seed'],'style_final':rngstyle.bit_generator.state,'NCE_actual_rows_consumed':rngnce.at},indent=2)+'\n')
    seen=prefix['source'].astype('i8');counts=np.bincount(seen,minlength=len(source));elapsed=time.monotonic()-clock;counter['training_s']+=elapsed
    record={'arm':arm,'updates':updates,'D_updates':d_updates,'style_reader_updates':updates,'training_s':elapsed,'selected_step':bestrow['step'],'initial_validation':initial,'selected_validation':bestrow,'final_validation':final,'validation_trace':trace,'validation_support':support,'stream_hashes':hashes,'old_R20_FINE_full_stream_match':full_match,'old_stream_prefix_from_saved_actual_arrays':True,'old_R20_FINE_completed_prefix_match':prefix_match,'initial_J20_abs_difference':delta,'first_style_gradient_check':first,'structural_equivalence':equivalence,'new_module_seed':cfg['new_module_seed'],'actually_sampled_unique_source':int((counts>0).sum()),'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated()if device.type=='cuda'else 0,'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved()if device.type=='cuda'else 0,'device':str(device),'selected_using_DEV':False,'style_reader_not_used_for_inference':True,'comparison_is_modulation_plus_reconstruction_package':True}
    (root/'TRAINING_TRACE.json').write_text(json.dumps(trace,indent=2)+'\n');(root/'TRAINING_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
    result=Translator().cpu();result.load_state_dict(best);result.eval();del model,reader,critic,xs,xt,vs,vt,og,od
    if device.type=='cuda':torch.cuda.empty_cache()
    return result,record,counts
