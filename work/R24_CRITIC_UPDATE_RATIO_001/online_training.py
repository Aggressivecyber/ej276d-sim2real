from __future__ import annotations
import numpy as np,torch,time,json,hashlib,pathlib,copy
from model import Translator,StyleReader,Critic,input_channels,charge_psd,physical,pooled_shape,rank_losses,nce,sw
from fine_core import FineLoss,NAMES
from prior import weighted_quantile
def state(m):return{k:v.detach().cpu().clone()for k,v in m.state_dict().items()}
class Choice:
    def __init__(self,rows):self.rows=rows;self.at=0;self.records=[]
    def choice(self,options,size,replace=False):
        x=self.rows[self.at];self.at+=1
        assert len(x)==size and not replace and len(np.unique(x))==size and np.isin(x,list(options)).all()
        self.records.append(x.copy());return x.astype('i8')
class InputCritic(Critic):
    def __init__(self,arm,mu,scale,device):
        super().__init__();self.arm=arm;self.register_buffer('input_mu',torch.as_tensor(mu,dtype=torch.float32,device=device),persistent=False);self.register_buffer('input_scale',torch.as_tensor(scale,dtype=torch.float32,device=device),persistent=False)
    def forward(self,x):
        a=input_channels(x)
        if self.arm in ['STANDARDIZED','D5']:a=(a-self.input_mu[None,:,:])/self.input_scale[None,:,:]
        return self.net(a).flatten(1).mean(1)
def train(arm,data,val,constants,scaler,actual,extra,cfg,root,counter,log):
    root.mkdir(exist_ok=False);device=torch.device('cuda:0'if torch.cuda.is_available()else'cpu');torch.manual_seed(cfg['seed'])
    model=Translator().to(device);reader=StyleReader().to(device);critic=InputCritic(arm,scaler['mu'],scaler['scale'],device).to(device)
    model.load_state_dict(torch.load(cfg['G_initial'],map_location=device,weights_only=True));reader.load_state_dict(torch.load(cfg['reader_initial'],map_location=device,weights_only=True));critic.load_state_dict(torch.load(cfg['D_initial'],map_location=device,weights_only=True))
    og=torch.optim.Adam([*model.parameters(),*reader.parameters()],lr=cfg['lr'],betas=cfg['betas']);od=torch.optim.Adam(critic.parameters(),lr=cfg['lr'],betas=cfg['betas'])
    xs=torch.from_numpy(data['source']).to(device);xt=torch.from_numpy(data['target']).to(device);vs=torch.from_numpy(data['source_val']).to(device);vt=torch.from_numpy(data['target_val']).to(device)
    vals={k:torch.from_numpy(val[k]).to(device)for k in ['z','white','style','physical_center','physical_scale','physical_projection','shape_projection']};vz=vals['z'];vw=vals['white'];vstyle=vals['style'];center=vals['physical_center'];scale=vals['physical_scale'];proj=vals['physical_projection'];waveproj=vals['shape_projection']
    fine=FineLoss(constants,device);edges=data['edges'];lambdas=data['lambdas'];hval=data['hval'];sb=data['source_bin'];tb=data['target_bin']
    rngnoise=np.random.default_rng(cfg['old_noise_seed']);rngstyle=np.random.default_rng(cfg['style_train_seed']);rngnce=Choice(actual['NCE_positions']);initial_rng={'noise':copy.deepcopy(rngnoise.bit_generator.state),'style':copy.deepcopy(rngstyle.bit_generator.state)}
    extra_noise=np.random.default_rng(cfg['extra_D_noise_seed']);extra_style=np.random.default_rng(cfg['extra_D_style_seed']);extra_initial={'noise':copy.deepcopy(extra_noise.bit_generator.state),'style':copy.deepcopy(extra_style.bit_generator.state)}
    extra_noise_hash=hashlib.sha256();extra_style_hash=hashlib.sha256();extra_updates=0;original_D_updates=0
    with torch.no_grad():
        qv=charge_psd(vs)[0];qr=charge_psd(vt)[0];ee=torch.as_tensor(edges,dtype=qv.dtype,device=device);vsb=torch.bucketize((qv.clamp_min(.01)*1000).log(),ee,right=True)-1;vtb=torch.bucketize((qr.clamp_min(.01)*1000).log(),ee,right=True)-1;common=[i for i in range(8)if(vsb==i).any()and(vtb==i).any()]
    vb=vsb.cpu().numpy();rb=vtb.cpu().numpy();wv=np.ones(len(vs),'f8')
    for b in range(8):wv[vb==b]=np.exp(lambdas[b]*hval[vb==b])
    weights=torch.as_tensor(wv,dtype=vs.dtype,device=device);diagnostic={};validation_rows=0
    def normalize_weight(w,b):
        result=np.zeros(len(w),'f8')
        for j in common:use=b==j;result[use]=w[use]/w[use].sum()/len(common)
        return result
    wg=normalize_weight(wv,vb);wr=normalize_weight(np.ones(len(vt)),rb)
    @torch.no_grad()
    def validate(step):
        nonlocal validation_rows
        model.eval();reader.eval();critic.eval();yy=[];ss=[]
        for j in range(0,len(vs),128):
            y,s,_=model(vs[j:j+128],vz[j:j+128],vw[j:j+128],vstyle[j:j+128]);yy.append(y);ss.append(s)
        y=torch.cat(yy);signal=torch.cat(ss);validation_rows+=len(vs);style_error=(reader(vs,signal)-vstyle).abs().mean()
        fy=(physical(y)-center)/scale;fr=(physical(vt)-center)/scale;levels=torch.linspace(.005,.995,128,device=device);rows=[];pf=[];pw=[];uf=[];uw=[]
        for b in common:
            sm=vsb==b;tm=vtb==b;fw=(weighted_quantile(fy[sm],weights[sm],levels)-torch.quantile(fr[tm],levels,dim=0)).abs().mean();wy=pooled_shape(y[sm])@waveproj;wwr=pooled_shape(vt[tm])@waveproj;ww=(weighted_quantile(wy,weights[sm],levels)-torch.quantile(wwr,levels,dim=0)).abs().mean();fu=(torch.quantile(fy[sm],levels,dim=0)-torch.quantile(fr[tm],levels,dim=0)).abs().mean();wu=sw(pooled_shape(y[sm]),pooled_shape(vt[tm]),waveproj)
            pf.append(fw);pw.append(ww);uf.append(fu);uw.append(wu);rows.append({'bin_id_1based':b+1,'source_n':int(sm.sum()),'REAL_n':int(tm.sum()),'weighted_physical_W1':float(fw),'weighted_shape_SW':float(ww),'unweighted_physical_W1':float(fu),'unweighted_shape_SW':float(wu)})
        _,qr,pr,positive=rank_losses(vs,y,edges);qo,po,_=charge_psd(y);qi=charge_psd(vs)[0];qc=(qo.clamp_min(.01).log()-qi.clamp_min(.01).log()).abs().mean();f=torch.stack(pf).mean();w=torch.stack(pw).mean();u=torch.stack(uf).mean();v=torch.stack(uw).mean();fine_val,fg,fb=fine.validation(y,vt,vsb,vtb,weights,common)
        pg=critic(y).cpu().numpy();prr=critic(vt).cpu().numpy();diagnostic['step'+str(step)+'_gen']=pg;diagnostic['step'+str(step)+'_REAL']=prr
        weightedMSE=.5*(np.dot(wg,pg.astype('f8')**2)+np.dot(wr,(prr.astype('f8')-1)**2));BA=.5*(np.dot(wg,pg<.5)+np.dot(wr,prr>=.5))
        r={'step':step,'J':float(f+w+.5*qc+2*pr+fine_val),'J19_weighted':float(f+w+.5*qc+2*pr),'fine_val':float(fine_val),'fine_groups':fg,'fine_per_bin':fb,'original_unweighted_J':float(u+v+.5*qc+2*pr),'weighted_physical_W1':float(f),'weighted_shape_SW':float(w),'unweighted_physical_W1':float(u),'unweighted_shape_SW':float(v),'source_logQ_abs_change':float(qc),'PSD_rank_violation':float(pr),'Q_rank_violation':float(qr),'positive_charge_penalty':float(positive),'per_bin':rows,'style_reconstruction_L1':float(style_error),'validation_output_PSD_quantiles':np.quantile(po.cpu().numpy(),[0,.1,.5,.9,1]).tolist(),
          'online_D_weighted_MSE':float(weightedMSE),'online_D_weighted_BA':float(BA),'online_D_unweighted_MSE':float(.5*(np.mean(pg.astype('f8')**2)+np.mean((prr.astype('f8')-1)**2))),'online_D_unweighted_BA':float(.5*(np.mean(pg<.5)+np.mean(prr>=.5))),'online_D_GEN_score_mean':float(pg.mean()),'online_D_REAL_score_mean':float(prr.mean()),'D_not_in_J':True}
        model.train();reader.train();critic.train();return r
    counter['response_fit_attempts']+=1;clock=time.monotonic();initial=validate(0);trace=[{'validation':initial,'elapsed_train_s':time.monotonic()-clock}];best=state(model);bestreader=state(reader);bestD=state(critic);bestrow=initial;bestopt={'G_reader':copy.deepcopy(og.state_dict()),'D':copy.deepcopy(od.state_dict())};updates=0;dupdates=0;hashnoise=hashlib.sha256();hashstyle=hashlib.sha256()
    try:
        for step in range(1,cfg['updates']+1):
            if time.monotonic()-clock>=cfg['training_wall_s']-10 or counter['training_s']+time.monotonic()-clock>=900-10:break
            sl=slice((step-1)*cfg['batch'],step*cfg['batch']);si=actual['source'][sl].astype('i8');ti=actual['target'][sl].astype('i8');bn=actual['bin'][sl];assert np.array_equal(sb[si],bn)and np.array_equal(tb[ti],bn)
            z=rngnoise.normal(size=(cfg['batch'],8)).astype('f4');white=rngnoise.normal(size=(cfg['batch'],480)).astype('f4');sty=rngstyle.normal(size=(cfg['batch'],8)).astype('f4');hashnoise.update(z.tobytes());hashnoise.update(white.tobytes());hashstyle.update(sty.tobytes());x=xs[si];real=xt[ti];zz=torch.from_numpy(z).to(device);ww=torch.from_numpy(white).to(device);s=torch.from_numpy(sty).to(device)
            critic.requires_grad_(True);od.zero_grad(set_to_none=True)
            with torch.no_grad():fake=model(x,zz,ww,s)[0]
            ld=.5*((critic(real)-1).square().mean()+critic(fake).square().mean());ld.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg['gradient_clip']);od.step();dupdates+=1;counter['D_updates']+=1
            original_D_updates+=1;counter['original_D_updates']+=1;dlosses=[float(ld.detach())]
            # Only D changes. No BN/dropout or mutable training buffers exist in G.
            # Dedicated extra RNGs never touch the original G/NCE random streams.
            for j in range(cfg['extra_D_batches_per_G']):
                es=slice(extra_updates*cfg['batch'],(extra_updates+1)*cfg['batch']);ei=extra['source'][es].astype('i8');et=extra['target'][es].astype('i8');eb=extra['bin'][es]
                assert np.array_equal(sb[ei],eb)and np.array_equal(tb[et],eb)
                ez=extra_noise.normal(size=(cfg['batch'],8)).astype('f4');ew=extra_noise.normal(size=(cfg['batch'],480)).astype('f4');estyle=extra_style.normal(size=(cfg['batch'],8)).astype('f4')
                extra_noise_hash.update(ez.tobytes());extra_noise_hash.update(ew.tobytes());extra_style_hash.update(estyle.tobytes())
                with torch.no_grad():ef=model(xs[ei],torch.from_numpy(ez).to(device),torch.from_numpy(ew).to(device),torch.from_numpy(estyle).to(device))[0]
                od.zero_grad(set_to_none=True);eld=.5*((critic(xt[et])-1).square().mean()+critic(ef).square().mean());eld.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg['gradient_clip']);od.step()
                dlosses.append(float(eld.detach()));extra_updates+=1;dupdates+=1;counter['extra_D_updates']+=1;counter['D_updates']+=1;counter['extra_D_fake_rows']+=cfg['batch']
                if device.type=='cuda'and torch.cuda.max_memory_allocated()>cfg['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
            critic.requires_grad_(False);og.zero_grad(set_to_none=True);y,signal,_=model(x,zz,ww,s);gan=(critic(y)-1).square().mean();content=nce(model,x,y,cfg,rngnce);identity=model(real,torch.zeros_like(zz),torch.zeros_like(ww),torch.zeros_like(s))[0];ident=(identity-real).abs().mean()/.1;qrel,qr,pr,positive=rank_losses(x,y,edges);domain=sw((physical(y)-center)/scale,(physical(real)-center)/scale,proj)+sw(pooled_shape(y),pooled_shape(real),waveproj)
            components={'adversarial':gan,'content_NCE':content,'identity':ident,'Q_relative':qrel,'Q_rank':qr,'PSD_rank':pr,'domain_SW':domain,'positive_charge':positive};base=sum(cfg['loss_weights'][k]*v for k,v in components.items());fine_value,fine_groups=fine.training(y,real);style_loss=(reader(x,signal)-s).abs().mean();loss=base+fine_value+cfg['style_reconstruction_weight']*style_loss
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite frozen loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([*model.parameters(),*reader.parameters()],cfg['gradient_clip']);og.step();updates=step;counter['G_updates']+=1;counter['style_reader_updates']+=1
            if device.type=='cuda'and torch.cuda.max_memory_allocated()>cfg['GPU_allocated_limit_bytes']:raise MemoryError('GPU allocated cap')
            if step%cfg['validation_every']==0:
                valrow=validate(step);row={'step':step,'G_loss':float(loss),'D_loss':float(ld),'D_loss_all5_mean':float(np.mean(dlosses)),'extra_D_loss_mean':float(np.mean(dlosses[1:])),'base_G_loss':float(base),'fine_train':float(fine_value),'fine_train_groups':{n:float(v)for n,v in zip(NAMES,fine_groups)},'style_L1':float(style_loss),'components':{k:float(v)for k,v in components.items()},'validation':valrow,'elapsed_train_s':time.monotonic()-clock};trace.append(row);log(arm+' '+json.dumps(row))
                if valrow['J']<bestrow['J']:best=state(model);bestreader=state(reader);bestD=state(critic);bestrow=valrow;bestopt={'G_reader':copy.deepcopy(og.state_dict()),'D':copy.deepcopy(od.state_dict())}
                (root/'TRAINING_TRACE.json').write_text(json.dumps(trace,indent=2)+'\n','utf8');(root.parent/'TRAIN_PROGRESS.json').write_text(json.dumps({'arm':arm,'step':step,'D_updates_this_arm':dupdates,'counter':counter},indent=2)+'\n','utf8')
        final=trace[-1]['validation']if trace[-1]['validation']['step']==updates else validate(updates)
        if final['J']<bestrow['J']:best=state(model);bestreader=state(reader);bestD=state(critic);bestrow=final;bestopt={'G_reader':copy.deepcopy(og.state_dict()),'D':copy.deepcopy(od.state_dict())}
        trace.append({'final':True,'validation':final,'elapsed_train_s':time.monotonic()-clock})
        for n,obj in [('FINAL_TRANSLATOR.pt',state(model)),('FINAL_STYLE_READER.pt',state(reader)),('FINAL_CRITIC.pt',state(critic)),('TRANSLATOR.pt',best),('STYLE_READER.pt',bestreader),('CRITIC.pt',bestD),('OPTIMIZER_SELECTED.pt',bestopt),('OPTIMIZER_FINAL.pt',{'G_reader':og.state_dict(),'D':od.state_dict()})]:torch.save(obj,root/n)
        count=updates*cfg['batch'];prefix={k:v[:count]for k,v in actual.items()if k!='NCE_positions'};prefix['NCE_positions']=np.asarray(rngnce.records,dtype='<u2');np.savez_compressed(root/'TRAINING_STREAMS_PRIVATE.npz',**prefix)
        ex_prefix={k:v[:extra_updates*cfg['batch']]for k,v in extra.items()};np.savez_compressed(root/'EXTRA_D_TRAINING_STREAMS_PRIVATE.npz',**ex_prefix)
        ex_hashes={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in ex_prefix.items()};ex_hashes.update(noise_stream=extra_noise_hash.hexdigest(),style_stream=extra_style_hash.hexdigest())
        extra_record={'extra_D_updates':extra_updates,'original_D_updates':original_D_updates,'extra_D_generated_fake_rows':extra_updates*cfg['batch'],'draws_each_extra_domain':extra_updates*cfg['batch'],'unique_extra_source':len(np.unique(ex_prefix['source'])),'unique_extra_REAL':len(np.unique(ex_prefix['target'])),'unique_all_D_source':len(np.unique(np.r_[prefix['source'],ex_prefix['source']])),'unique_all_D_REAL':len(np.unique(np.r_[prefix['target'],ex_prefix['target']])),'stream_hashes':ex_hashes,'noise_initial':extra_initial['noise'],'noise_final':extra_noise.bit_generator.state,'style_initial':extra_initial['style'],'style_final':extra_style.bit_generator.state,'sampling_metadata':json.loads((root.parent/'EXTRA_D_PREPARED_RANDOM_STATES.json').read_text('utf8')),'additional_compute_and_exposure':True,'G_reader_state_updates_from_extra_D':0,'full_white_arrays_not_materialized':True}
        (root/'EXTRA_D_EXPOSURE_AND_RANDOM_STATES.json').write_text(json.dumps(extra_record,indent=2)+'\n','utf8')
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in prefix.items()};hashes.update(noise_stream=hashnoise.hexdigest(),style_stream=hashstyle.hexdigest());np.savez_compressed(root/'ONLINE_D_SCORES_PRIVATE.npz',**diagnostic,source_bin=vb,REAL_bin=rb,source_weight=wg,REAL_weight=wr)
        rs={'noise_initial':initial_rng['noise'],'noise_final':rngnoise.bit_generator.state,'style_initial':initial_rng['style'],'style_final':rngstyle.bit_generator.state,'NCE_rows':rngnce.at};(root/'TRAINING_RANDOM_STATES.json').write_text(json.dumps(rs,indent=2)+'\n','utf8')
        elapsed=time.monotonic()-clock;counter['training_s']+=elapsed;record={'arm':arm,'updates':updates,'D_updates':dupdates,'style_reader_updates':updates,'training_s':elapsed,'selected_step':bestrow['step'],'initial_validation':initial,'selected_validation':bestrow,'final_validation':final,'validation_trace':trace,'stream_hashes':hashes,'validation_generation_rows':validation_rows,'validation_support':[{'bin':b+1,'source_n':int((vb==b).sum()),'REAL_n':int((rb==b).sum())}for b in range(8)],'same_time_selected_G_reader_D':True,'selected_using_DEV_or_D':False,'actually_sampled_unique_source':len(np.unique(prefix['source'])),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()if device.type=='cuda'else 0,'device':str(device),'initial_G_reader':'R21 step2000','initial_D':'R18 initial no R22pretraining'}
        record.update(original_D_updates=original_D_updates,extra_D_updates=extra_updates,extra_D_exposure=extra_record,completed_G_prefix_n=updates,stop_reason='G_update_limit'if updates==cfg['updates']else'training_wall_limit',D5_completed_step_ratio_checked=dupdates==5*updates)
        assert dupdates==5*updates and dupdates<=cfg['D_updates_limit']
        (root/'TRAINING_TRACE.json').write_text(json.dumps(trace,indent=2)+'\n','utf8');(root/'TRAINING_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n','utf8');return record
    except BaseException as e:
        (root/'FAILED_TRAINING.json').write_text(json.dumps({'arm':arm,'error':repr(e),'G_updates':updates,'D_updates':dupdates,'original_D_updates':original_D_updates,'extra_D_updates':extra_updates,'NCE_rows_consumed':rngnce.at,'G_noise_final':rngnoise.bit_generator.state,'G_style_final':rngstyle.bit_generator.state,'extra_noise_final':extra_noise.bit_generator.state,'extra_style_final':extra_style.bit_generator.state,'counter':counter,'elapsed_s':time.monotonic()-clock},indent=2)+'\n','utf8');raise
