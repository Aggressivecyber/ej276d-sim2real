from __future__ import annotations
import pathlib,json,datetime,time,copy,hashlib
import numpy as np
import torch
import model_R2 as r2
import model as flexible
import energy_width as ew
import evaluation as ev

def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode()+b'\0');h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def fit(model,constructor,module,real,rt,rv,c,q,gt,gv,stage,arm,cfg,folder,write,log):
    folder.mkdir(exist_ok=False);phase=cfg['curriculum'][stage]
    tr=torch.tensor(real[rt]);val=torch.tensor(real[rv]);ct=torch.tensor(c[gt]);qt=torch.tensor(q[gt]);cv=torch.tensor(c[gv]);qv=torch.tensor(q[gv])
    if min(len(ct),len(tr),len(cv),len(val))<384:raise RuntimeError('frozen384 validation sample unsupported')
    rs,_,_=module.observables(tr);scale=(torch.quantile(rs,.75,dim=0)-torch.quantile(rs,.25,dim=0)).clamp_min(torch.tensor([.05,.5,.5,1.,.05,.02,.2,.005]))
    torch.save({'state_dict':copy.deepcopy(model.state_dict()),'constructor':constructor,'stage':stage,'arm':arm,'trainable_parameter_digest':parameter_digest(model)},folder/'initial_model.pt')
    target_fit=target_validation=None;edges=None
    if stage=='R3B':
        edges=np.linspace(*np.quantile(np.log(ev.features(real[rt])['features']['Qtotal_mVns']),[.05,.95]),9);centers=torch.tensor((edges[:-1]+edges[1:])/2,dtype=torch.float32);width=float(edges[1]-edges[0])
        target_fit=ew.conditional_targets(tr,centers,width);target_validation=ew.conditional_targets(val,centers,width)
        write(folder/'TARGET_STATISTICS.json',{'fit':target_fit,'validation':target_validation,'edges_logQ':edges,'only_this_arm_permitted_TRAIN':True,'no_OTHER_arm_or_DEV_EVAL_target':True})
    gen=torch.Generator().manual_seed(cfg['seed']);projection=torch.randn((480,32),generator=gen)
    projection=torch.nn.functional.avg_pool1d(projection.T[:,None,:],5,stride=1,padding=2).squeeze(1).T
    projection/=torch.linalg.vector_norm(projection,dim=0)[None,:];projection[:100]*=.1
    opt=torch.optim.Adam(model.parameters(),lr=phase['learning_rate']);bs=phase['batch_size'];nv=384
    vgen=torch.Generator().manual_seed(phase['validation_seed'])
    torch.save({'global_after_model_constructor':torch.get_rng_state(),'training_generator_after_projection':gen.get_state(),'validation_generator_before_draw':vgen.get_state(),'master_seed':cfg['seed']},folder/'INITIAL_RANDOM_STATES.pt')
    vi=torch.randperm(len(cv),generator=vgen)[:nv];ri=torch.randperm(len(val),generator=vgen)[:nv];vz=torch.randn((nv,8),generator=vgen);vw=torch.randn((nv,512),generator=vgen)
    def validation():
        model.eval()
        with torch.no_grad():
            fake,_=model(cv[vi],qv[vi],vz,vw);loss,parts=module.distribution_loss(fake,val[ri],scale,projection)
            if stage=='R3B':extra,cp=ew.conditional_loss(fake,target_validation);loss=loss+extra;parts.update(cp)
        model.train();return float(loss),{k:float(v) for k,v in parts.items()}
    best,parts=validation();state=copy.deepcopy(model.state_dict());beststep=0;trace=[{'update':0,'validation':best,'validation_parts':parts}]
    log(f'{arm}/{stage} initial validation={best:.5f};REALfit{len(tr)}/val{len(val)} G4fit{len(ct)}/val{len(cv)}')
    begin=time.monotonic();updates=0
    for step in range(1,phase['max_updates']+1):
        if time.monotonic()-begin>phase['training_wall_s']:log(f'{arm}/{stage} predeclared phasewall reached');break
        ii=torch.randint(len(ct),(bs,),generator=gen);jj=torch.randint(len(tr),(bs,),generator=gen);z=torch.randn((bs,8),generator=gen);white=torch.randn((bs,512),generator=gen)
        fake,_=model(ct[ii],qt[ii],z,white);loss,parts=module.distribution_loss(fake,tr[jj],scale,projection)
        fs,_,_=module.observables(fake);ch=ct[ii,1]-ct[ii,1].mean();ps=fs[:,0]-fs[:,0].mean();cov=(ch*ps).mean()
        total=loss+.2*torch.relu(.01-cov)+.0001*sum(p.square().mean() for p in list(model.mean.parameters())+list(model.random.parameters()))
        if stage!='R2':total=total+.002*model.last_time_coefficients.diff(n=2,dim=1).square().mean()
        conditional_parts={}
        if stage=='R3B':extra,conditional_parts=ew.conditional_loss(fake,target_fit);total=total+extra
        if not torch.isfinite(total):raise FloatingPointError('nonfinite curriculum loss')
        opt.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();updates=step
        if step%100==0:
            vl,vp=validation();r={'update':step,'training_loss':float(total.detach()),'validation':vl,'validation_parts':vp,'composition_cov':float(cov.detach()),'training_conditional_parts':{k:float(v.detach()) for k,v in conditional_parts.items()},'training_elapsed_s':time.monotonic()-begin};trace.append(r)
            if vl<best:best=vl;beststep=step;state=copy.deepcopy(model.state_dict())
            log(f'{arm}/{stage} update={step} validation={vl:.6f} best={beststep};elapsed={time.monotonic()-begin:.1f}s')
    if updates%100:
        vl,vp=validation();trace.append({'update':updates,'validation':vl,'validation_parts':vp,'wall_stop':True})
        if vl<best:best=vl;beststep=updates;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    pack={'state_dict':state,'constructor':constructor,'best_update':beststep,'arm':arm,'stage':stage,'phase_config':phase,'master_seed':cfg['seed']};torch.save(pack,folder/'model.pt')
    torch.save({'training_generator_final':gen.get_state(),'validation_generator_after_draw':vgen.get_state(),'global_torch_final':torch.get_rng_state()},folder/'FINAL_RANDOM_STATES.pt')
    receipt={'arm':arm,'phase':stage,'optimizer_updates':updates,'selected_update':beststep,'training_elapsed_s':time.monotonic()-begin,'fixed_TRAIN_validation':best,'initial_validation':trace[0]['validation'],
             'REAL_fit':len(tr),'REAL_validation':len(val),'G4_fit':len(ct),'G4_validation':len(cv),'all_statistics_own_permitted_fit':True,'EVAL_selection':0,'DEV_selection':0,
             'feature_scale':scale.numpy(),'conditional_target_edges_logQ':edges,'parameter_digest':parameter_digest(model),'noise_spectrum_digest':hashlib.sha256(model.noise_spectrum.numpy().tobytes()).hexdigest(),
             'learned_charge_gain':float(torch.exp(model.log_gain)),'learned_energy_exponent':float(1+.4*torch.tanh(model.energy_slope))}
    write(folder/'TRAINING_TRACE.json',trace);write(folder/'TRAINING_RECEIPT.json',receipt)
    return model,pack,receipt

def train_arms(real,real_charge,real_good,real_fit,real_val,c,q,g4_nominal,g4_split,cfg,root,write,log):
    lo,hi=cfg['gap_charge_mVns'];rgap=(real_charge>=lo)&(real_charge<hi);ggap=(g4_nominal>=lo)&(g4_nominal<hi)
    allowed={'FULL':{'rt':real_fit,'rv':real_val,'gt':g4_split,'gv':~g4_split},
             'GAP':{'rt':real_fit&~rgap,'rv':real_val&~rgap,'gt':g4_split&~ggap,'gv':(~g4_split)&~ggap}}
    # The shared trainable initializer uses only the GAP-allowed fit. No FULL q3 statistics can enter GAP.
    a=allowed['GAP'];lm,ls,spec,_=r2.noise_statistics(real[a['rt']]);gain=float(np.median(real[a['rt'],100:400].sum(axis=1)-300*real[a['rt'],:100].mean(axis=1))/np.median(q[a['gt']]))
    common_constructor={'log_gain':float(np.log(gain)),'noise_log_mean':float(lm),'noise_log_sd':float(ls),'noise_spectrum':spec.tolist()}
    torch.manual_seed(cfg['seed']);common=r2.SmoothResponse(**common_constructor)
    shared_parameters={k:v.detach().clone() for k,v in common.named_parameters()};shared_digest=parameter_digest(common)
    torch.save({'state_dict':common.state_dict(),'constructor':common_constructor,'master_seed':cfg['seed'],'parameters_only_sharing':'noise spectrum buffers independently estimated per arm','trainable_parameter_digest':shared_digest,'init_stats_from':'only GAP-allowed REAL/G4 TRAIN fit'},root/'COMMON_INITIAL_MODEL.pt')
    receipts={};final={};initial_digests={};noise_records={}
    for arm in ['FULL','GAP']:
        a=allowed[arm];lm0,ls0,spec0,power=r2.noise_statistics(real[a['rt']]);ctor={**common_constructor,'noise_spectrum':spec0.tolist()}
        torch.manual_seed(cfg['seed']);model=r2.SmoothResponse(**ctor)
        with torch.no_grad():
            for name,p in model.named_parameters():p.copy_(shared_parameters[name])
        initial_digests[arm]=parameter_digest(model);assert initial_digests[arm]==shared_digest
        noise_records[arm]={'REAL_fit_n':int(a['rt'].sum()),'estimated_log_RMS_mean':float(lm0),'estimated_log_RMS_SD':float(ls0),'estimated_noise_spectrum':spec0,'spectrum_SHA256':hashlib.sha256(spec0.tobytes()).hexdigest(),'gain_own_fit_diagnostic':float(np.median(real[a['rt'],100:400].sum(axis=1)-300*real[a['rt'],:100].mean(axis=1))/np.median(q[a['gt']])),
                            'shared_initial_trainable_parameters_from_GAP_fit':True,'no_FULL_buffer_copied_to_GAP':True}
        armroot=root/arm;armroot.mkdir(exist_ok=False);receipts[arm]=[]
        model,pack,r=fit(model,ctor,r2,real,a['rt'],a['rv'],c,q,a['gt'],a['gv'],'R2',arm,cfg,armroot/'R2',write,log);receipts[arm].append(r)
        torch.manual_seed(cfg['seed']);model=flexible.SmoothResponse(**pack['constructor']);missing,unexpected=model.load_state_dict(pack['state_dict'],strict=False)
        assert not unexpected and all(k.startswith(('temporal_','time_')) for k in missing)
        model,pack,r=fit(model,pack['constructor'],flexible,real,a['rt'],a['rv'],c,q,a['gt'],a['gv'],'R3',arm,cfg,armroot/'R3',write,log);receipts[arm].append(r)
        torch.manual_seed(cfg['seed']);model=flexible.SmoothResponse(**pack['constructor']);model.load_state_dict(pack['state_dict'],strict=True)
        model,pack,r=fit(model,pack['constructor'],flexible,real,a['rt'],a['rv'],c,q,a['gt'],a['gv'],'R3B',arm,cfg,armroot/'R3B',write,log);receipts[arm].append(r);final[arm]=model
    write(root/'INITIAL_PARAMETER_EQUALITY.json',{'shared_parameter_digest':shared_digest,'arm_initial_parameter_digests':initial_digests,'all_trainable_parameters_identical':len(set(initial_digests.values()))==1,'noise_buffers':noise_records,'noise_buffers_do_not_need_equal':True,'FULL_q3_stats_not_copied_to_GAP':True})
    return final,receipts,allowed,rgap,ggap
