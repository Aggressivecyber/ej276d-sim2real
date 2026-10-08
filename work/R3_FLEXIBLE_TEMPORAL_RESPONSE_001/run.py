from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='4'
import pathlib,json,time,datetime,hashlib,io,copy,traceback
ROOT=pathlib.Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime_cache/matplotlib')
os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'runtime_cache/joblib')
START=time.monotonic()
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
from model import SmoothResponse,observables,distribution_loss,noise_statistics
import evaluation as ev
CFG=json.loads((ROOT/'config.json').read_text())
# Before ANY response model constructor. This is distinct from the R2
# checkpoint's unknown historical training initialization.
torch.manual_seed(CFG['seed']);np.random.seed(CFG['seed'])
GLOBAL_TORCH_INITIAL=torch.get_rng_state().clone()
NUMPY_INITIAL=np.random.get_state()
(ROOT/'INITIAL_RNG.json').write_text(json.dumps({'Torch_main_seed':torch.initial_seed(),'numpy_seed':CFG['seed'],'numpy_state':[NUMPY_INITIAL[0],NUMPY_INITIAL[1].tolist(),int(NUMPY_INITIAL[2]),int(NUMPY_INITIAL[3]),float(NUMPY_INITIAL[4])],'set_before_any_model_constructor':True,'R2_initialization_historical_unknown':True},indent=2)+'\n',encoding='utf-8')
OUT=ROOT/'results';OUT.mkdir(exist_ok=False)
ev.OUT=OUT;ev.FIG=OUT/'figures';ev.FIG.mkdir()
ev.CFG=CFG['evaluation'];ev.NAMES=['SIM','GEN_W0','R2','R3','REAL']
ev.COLORS={'SIM':'#2878b5','GEN_W0':'#a33ea2','R2':'#bb9b48','R3':'#e65c24','REAL':'#181818'}
INPUTS=[]

def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)
def write(n,a):(OUT/n).write_text(json.dumps(ev.safe(a),ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def load_npz(p):
    b=pathlib.Path(p).read_bytes();INPUTS.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    return np.load(io.BytesIO(b),allow_pickle=False)
def prefix(p,dtype,n):
    size=np.dtype(dtype).itemsize*n
    with pathlib.Path(p).open('rb') as f:b=f.read(size)
    assert len(b)==size
    INPUTS.append({'path':str(p),'prefix_bytes':size,'prefix_sha256':hashlib.sha256(b).hexdigest()})
    return np.frombuffer(b,dtype=dtype).copy()

def clean_expectation(ee,had):
    u=np.maximum(np.arange(512)-128.,0)[None,:]
    n=len(ee);out=np.zeros((n,512),dtype=np.float64)
    for energy,taus,frac in [(ee,[13,35,270],[.7,.15,.15]),(.15*had,[13,59,460],[.5,.1,.4])]:
        for tau,weight in zip(taus,frac):out+=energy[:,None]*weight*(np.exp(-u/tau)-np.exp(-u/5))/(tau-5)
    energy=ee+.15*had;q=(energy+.03918)/.000132506/1000
    out*=q[:,None]/out.sum(axis=1,keepdims=True)
    peak=out.argmax(axis=1);threshold=out.max(axis=1)*.5;t=np.arange(512)[None,:]
    below=(t<peak[:,None])&(out<threshold[:,None]);tp=np.where(below,t,-1).max(axis=1)+1
    ix=(tp[:,None]-120+np.arange(480)[None,:]).clip(0,511)
    return np.take_along_axis(out,ix,axis=1).astype('f4'),q.astype('f4')

def condition(clean,ee,had):
    sf,_,_=observables(torch.from_numpy(clean.astype('f4')))
    energy=ee+.15*had;h=.15*had/np.maximum(energy,1e-15)
    # Four continuous sufficient descriptors of this clean scintillation
    # family: visible energy, composition, clean PSD, and clean tail fraction.
    c=np.column_stack([np.log(np.maximum(energy,1e-8)) - CFG['energy_log_center'],h,
                       sf[:,0].numpy(),sf[:,4].numpy()]).astype('f4')
    return c,energy,h

def inputs():
    with load_npz(CFG['evaluation']['input_npz']['SIM_METADATA']) as z:
        meta={k:z[k] for k in ['raw','real','sim_event_ids','real_event_ids','normalization_peak_V','source_label','sim_energy','edep_ee_MeV','edep_hadron_MeV','physical_r_h']}
    with load_npz(CFG['evaluation']['input_npz']['GEN_W0']) as z:
        assert np.array_equal(z['sim_event_ids'],meta['sim_event_ids']) and np.array_equal(z['before'],meta['raw'])
        historic=z['after'].copy()
    with load_npz(CFG['R2_generated_input']) as z:
        assert np.array_equal(z['sim_event_ids'],meta['sim_event_ids'])
        r2_wave=z['waveforms_V'];dc=z['G4_condition']
    cache=pathlib.Path(CFG['R2_cache']['path']);payload=cache.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==CFG['R2_cache']['sha256']
    INPUTS.append({'path':str(cache),'bytes':len(payload),'sha256':CFG['R2_cache']['sha256'],'role':'R2 bound G4 TRAIN cache'})
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        c=z['condition'];q=z['Qsource_Vns'];gt=z['internal_training'];train_ids=z['event_ids']
        assert len(c)==8192 and not set(train_ids.tolist())&set(meta['sim_event_ids'].tolist())
        assert np.array_equal(gt,np.array([ev.hashint('R2-G4-TRAIN-VALIDATION|'+str(s))%5!=0 for s in train_ids]))
    base=pathlib.Path(CFG['REAL_prefix_dir']);ident=pathlib.Path(CFG['REAL_identity_dir'])
    real=prefix(base/'window_V_f32.bin','<f4',5200*480).reshape(5200,480)
    qcreason=prefix(base/'QC_reason_u16.bin','<u2',5200)
    groups=prefix(ident/'selection_group.bin','<u2',5200);orig=prefix(ident/'selection_original_index.bin','<u4',5200)
    roles=prefix(ident/'selection_role.bin','u1',5200)
    assert np.all(roles[:2000]==0) and np.all(roles[2000:]==1) and np.array_equal(groups,orig//50)
    train=real[2000:];qc=qcreason[2000:]==0;tg=groups[2000:]
    prior_roles=json.loads(pathlib.Path(CFG['R2_roles']).read_text());valgroups=prior_roles['REAL_internal_validation_artificial_groups']
    rt=qc&~np.isin(tg,valgroups);rv=qc&np.isin(tg,valgroups)
    assert qc.sum()==3191 and rt.sum()==2592 and rv.sum()==599 and not set(tg[rt])&set(tg[rv])
    dq=((meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV']+.03918)/.000132506/1000).astype('f4')
    np.savez_compressed(ROOT/'generation_inputs.npz',G4_condition=dc,Qsource_Vns=dq,sim_event_ids=meta['sim_event_ids'],edep_ee_MeV=meta['edep_ee_MeV'],edep_hadron_MeV=meta['edep_hadron_MeV'])
    select_condition_pairs(meta)
    write('DATA_ROLES.json',{**prior_roles,'source':'R2 cached G4 TRAIN and identical REAL role/group selection','new_CSV_read':0,'SSH':0,'R2_checkpoint_is_external_input':CFG['R2_checkpoint'],'EVAL_read':0})
    ref=json.loads((base/'STATISTICS.json').read_text())
    return meta,historic,r2_wave,real[:2000],qcreason[:2000],groups[:2000],orig[:2000],ref,train,rt,rv,c,q,gt,dc,dq

PAIR_SELECTION=None
def select_condition_pairs(meta):
    global PAIR_SELECTION
    energy=meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV'];h=meta['physical_r_h'];ids=meta['sim_event_ids']
    had=np.flatnonzero(h>=.95);electron=np.flatnonzero(h<=.05)
    first=had[ev.order(ids[had],'R3-energy-pair')[0]]
    second=had[np.argmin(np.abs(np.log(energy[had]/(2*energy[first]))))]
    assert first!=second
    gap=np.abs(np.log(energy[electron,None]/energy[had][None,:]))
    flat=np.argmin(gap);ea,ha=np.unravel_index(flat,gap.shape)
    pairs=[{'kind':'energy_at_nearly_hadron_only_composition','indices':[int(first),int(second)]},
           {'kind':'composition_at_closest_existing_energy','indices':[int(electron[ea]),int(had[ha])]}]
    for p in pairs:
        ix=p['indices'];p.update(event_ids=ids[ix].tolist(),visible_energy_MeV=energy[ix].tolist(),r_h=h[ix].tolist(),energy_ratio=float(energy[ix[1]]/energy[ix[0]]))
    PAIR_SELECTION={'rule':CFG['fixed_condition_pairs'],'pairs':pairs,'selected_before_model_fit':True,'no_output_selection':True}
    write('FIXED_CONDITION_PAIRS.json',PAIR_SELECTION)


def condition_pair_examples(model,c,q,meta):
    gen=torch.Generator().manual_seed(CFG['fixed_condition_pairs']['shared_latent_seed']);results=[];saved={}
    for pair in PAIR_SELECTION['pairs']:
        ix=pair['indices'];z=torch.randn((1,8),generator=gen).repeat(2,1);white=torch.randn((1,512),generator=gen).repeat(2,1)
        with torch.no_grad():v,params=model(c[ix],q[ix],z,white)
        feat=ev.features(v.numpy())
        results.append({**pair,'PSD':feat['features']['PSD'],'Qtotal_mVns':feat['features']['Qtotal_mVns'],'peak_V':feat['features']['peak_V'],'rise_10_90_ns':feat['features']['rise_10_90_ns'],'same_z_and_same_white_noise':True})
        saved[pair['kind']+'__V']=v.numpy();saved[pair['kind']+'__z']=z.numpy();saved[pair['kind']+'__white']=white.numpy()
    np.savez_compressed(OUT/'fixed_latent_condition_pairs.npz',**saved)
    ev.plot_condition_pairs(results,saved)
    return {'selection':PAIR_SELECTION,'responses':results,'not_unique_physical_transfer_or_truth':True}

def sample(model,c,q,n,seed):
    gen=torch.Generator().manual_seed(seed)
    if n==2000 and seed==CFG['generation_seeds'][0]:
        torch.save({'generator_before_draw':gen.get_state(),'seed':seed,'n':n,'latent_dim':8,'white_width':512,'generation_chunk':256},ROOT/'GENERATION_RANDOM_STATE.pt')
    z=torch.randn((n,8),generator=gen);white=torch.randn((n,512),generator=gen)
    outputs=[];metadata={}
    with torch.no_grad():
        for i in range(0,n,256):
            w,m=model(c[i:i+256],q[i:i+256],z[i:i+256],white[i:i+256]);outputs.append(w.numpy())
            for k,v in m.items():metadata.setdefault(k,[]).append(v.numpy())
    return np.concatenate(outputs),{k:np.concatenate(v) for k,v in metadata.items()},z.numpy()

def train_model(real,rt,rv,c,q,gt):
    tr=torch.tensor(real[rt]);val=torch.tensor(real[rv]);ct=torch.tensor(c[gt]);qt=torch.tensor(q[gt]);cv=torch.tensor(c[~gt]);qv=torch.tensor(q[~gt])
    lm,ls,spec,power=noise_statistics(real[rt]);rs,_,_=observables(tr)
    gains=float(np.median(real[rt,100:400].sum(axis=1)-300*real[rt,:100].mean(axis=1))/np.median(q[gt]))
    assert gains>0
    p=pathlib.Path(CFG['R2_checkpoint']['path']);b=p.read_bytes()
    assert hashlib.sha256(b).hexdigest()==CFG['R2_checkpoint']['sha256'],'R2 weight binding'
    prior=torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
    constructor={k:(float(v) if k!='noise_spectrum' else list(v)) for k,v in prior['constructor'].items()}
    model=SmoothResponse(**constructor)
    missing,unexpected=model.load_state_dict(prior['state_dict'],strict=False)
    assert not unexpected and all(k.startswith(('temporal_','time_')) for k in missing)
    torch.save({'state_dict':model.state_dict(),'constructor':constructor,'config':CFG,'external_R2_checkpoint_sha256':CFG['R2_checkpoint']['sha256']},ROOT/'initial_model.pt')
    scale=(torch.quantile(rs,.75,dim=0)-torch.quantile(rs,.25,dim=0)).clamp_min(torch.tensor([.05,.5,.5,1.,.05,.02,.2,.005]))
    gen=torch.Generator().manual_seed(CFG['seed'])
    projection=torch.randn((480,32),generator=gen)
    # Smooth projections, same matrix for TRAIN and validation throughout.
    projection=torch.nn.functional.avg_pool1d(projection.T[:,None,:],5,stride=1,padding=2).squeeze(1).T
    projection/=torch.linalg.vector_norm(projection,dim=0)[None,:]
    projection[:100]*=.1
    opt=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
    bs=CFG['batch_size'];nv=384
    vgen=torch.Generator().manual_seed(CFG['validation_seed'])
    torch.save({'global_before_any_constructor':GLOBAL_TORCH_INITIAL,'global_after_model_constructor':torch.get_rng_state(),'training_generator_before_projection':torch.Generator().manual_seed(CFG['seed']).get_state(),'training_generator_after_projection':gen.get_state(),'validation_generator_before_draw':vgen.get_state()},ROOT/'INITIAL_RANDOM_STATES.pt')
    vi=torch.randperm(len(cv),generator=vgen)[:nv];ri=torch.randperm(len(val),generator=vgen)[:nv]
    vz=torch.randn((nv,8),generator=vgen);vw=torch.randn((nv,512),generator=vgen)
    def validation():
        model.eval()
        with torch.no_grad():
            fake,_=model(cv[vi],qv[vi],vz,vw)
            loss,parts=distribution_loss(fake,val[ri],scale,projection)
        model.train();return float(loss),{k:float(v) for k,v in parts.items()}
    best,parts=validation();state=copy.deepcopy(model.state_dict());beststep=0;trace=[{'update':0,'validation':best,'validation_parts':parts}]
    log(f'TRAIN initial validation={best:.5f};G4_fit={len(ct)}, REAL_fit={len(tr)}, REAL_val={len(val)}')
    trainstart=time.monotonic();updates=0
    for step in range(1,CFG['max_updates']+1):
        if time.monotonic()-trainstart>CFG['training_wall_s']:log('fixed training wall reached');break
        ii=torch.randint(len(ct),(bs,),generator=gen);jj=torch.randint(len(tr),(bs,),generator=gen)
        z=torch.randn((bs,8),generator=gen);white=torch.randn((bs,512),generator=gen)
        fake,m=model(ct[ii],qt[ii],z,white)
        loss,loss_parts=distribution_loss(fake,tr[jj],scale,projection)
        fs,_,_=observables(fake)
        ch=ct[ii,1]-ct[ii,1].mean();ps=fs[:,0]-fs[:,0].mean()
        composition_cov=(ch*ps).mean()
        # Weak nonnegative composition association, not a particle label loss.
        conditional=.2*torch.relu(.01-composition_cov)
        temporal_smooth=.002*model.last_time_coefficients.diff(n=2,dim=1).square().mean()
        regular=.0001*sum(p.square().mean() for p in list(model.mean.parameters())+list(model.random.parameters()))
        total=loss+conditional+regular+temporal_smooth
        if not torch.isfinite(total):raise FloatingPointError('nonfinite training loss')
        opt.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();updates=step
        if step%100==0:
            vl,vp=validation();record={'update':step,'training_loss':float(total.detach()),'validation':vl,'validation_parts':vp,'composition_cov':float(composition_cov.detach()),'training_elapsed_s':time.monotonic()-trainstart}
            trace.append(record)
            if vl<best:best=vl;beststep=step;state=copy.deepcopy(model.state_dict())
            log(f'update={step} train={record["training_loss"]:.5f} validation={vl:.5f} best={beststep}; elapsed={record["training_elapsed_s"]:.1f}s')
    # A final fixed validation is allowed when the wall stops between checkpoints.
    if updates%100:
        vl,vp=validation();trace.append({'update':updates,'validation':vl,'validation_parts':vp,'wall_stop':True})
        if vl<best:best=vl;beststep=updates;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    torch.save({'state_dict':state,'constructor':constructor,'best_update':beststep,'config':CFG},ROOT/'model.pt')
    write('TRAINING_TRACE.json',trace)
    torch.save({'training_generator_final':gen.get_state(),'validation_generator_after_draw':vgen.get_state(),'global_torch_final':torch.get_rng_state()},ROOT/'FINAL_RANDOM_STATES.pt')
    receipt={'optimizer_updates':updates,'selected_update':beststep,'best_fixed_TRAIN_validation':best,'initial_TRAIN_validation':trace[0]['validation'],'training_elapsed_s':time.monotonic()-trainstart,'learned_charge_gain':float(torch.exp(model.log_gain)),'learned_energy_exponent':float(1+.4*torch.tanh(model.energy_slope)),'learned_noise_log_RMS_V':float(model.noise_log_mean),'learned_noise_log_sd':float(model.noise_log_sd),'mean_stochastic_parameter_norms':{'mean':float(sum(p.square().sum() for p in model.mean.parameters()).sqrt()),'random':float(sum(p.square().sum() for p in model.random.parameters()).sqrt())},'random_scales':model.random_scale.detach().numpy(),'baseline_spectrum_resolution_MHz':10,'no_DEV_checkpoint_selection':True}
    write('TRAINING_RECEIPT.json',receipt);return model,receipt

def main():
    log(f'R3 worker PID={os.getpid()} torch={torch.__version__} CPU_threads={torch.get_num_threads()}')
    meta,historic,r2_wave,real,sourceqc,groups,orig,ref,realtrain,rt,rv,c,q,gt,dc,dq=inputs()
    log('R2 cached8192 G4 TRAIN and unchanged REAL group roles reused; EVAL0')
    model,training=train_model(realtrain,rt,rv,c,q,gt)
    dc_t=torch.tensor(dc);dq_t=torch.tensor(dq)
    waves=[];params=[];latent=[]
    for seed in CFG['generation_seeds']:
        w,p,z=sample(model,dc_t,dq_t,2000,seed);waves.append(w);params.append(p);latent.append(z)
    assert all(np.all(np.isfinite(w)) for w in waves)
    np.savez_compressed(ROOT/'generated.npz',waveforms_V=waves[0],independent_view_2_V=waves[1],independent_view_3_V=waves[2],sim_event_ids=meta['sim_event_ids'],edep_ee_MeV=meta['edep_ee_MeV'],edep_hadron_MeV=meta['edep_hadron_MeV'],visible_energy_proxy_MeV=meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV'],visible_hadron_fraction=meta['physical_r_h'],G4_condition=dc,latent_z=latent[0],**{'main_'+k:v for k,v in params[0].items()})
    # One bounded re-open of saved weights checks the concrete deliverable.
    loaded=torch.load(ROOT/'model.pt',map_location='cpu',weights_only=True)
    reloaded=SmoothResponse(**loaded['constructor']);reloaded.load_state_dict(loaded['state_dict']);reloaded.eval()
    # The white random sequence differs if requesting only16 instead of2000;
    # compare explicit saved latents/white rather than a misleading seed claim.
    gg=torch.Generator().manual_seed(CFG['generation_seeds'][0]);zz=torch.randn((2000,8),generator=gg);ww=torch.randn((2000,512),generator=gg)
    with torch.no_grad():check,_=reloaded(dc_t[:256],dq_t[:256],zz[:256],ww[:256])
    assert np.array_equal(check.numpy()[:16],waves[0][:16]),'saved model deterministic reload mismatch'
    domains={'SIM':ev.features(meta['raw'].astype(float)*meta['normalization_peak_V'][:,None]),'GEN_W0':ev.features(historic.astype(float)*meta['normalization_peak_V'][:,None]),'R2':ev.features(r2_wave),'R3':ev.features(waves[0]),'REAL':ev.features(real)}
    domains['REAL']['valid']&=sourceqc==0
    for name,d in domains.items():
        d['ids']=meta['real_event_ids'] if name=='REAL' else meta['sim_event_ids'];d['groups']=groups.astype(int) if name=='REAL' else np.arange(2000)
        d['region']=np.searchsorted(ref['CAL_charge_quantile_edges_mV_ns'],d['features']['Qtotal_mVns'],side='right').astype('u1')
    newstats,newdist=ev.characterize({'R3':domains['R3'],'REAL':domains['REAL']},ref)
    old=json.loads(pathlib.Path(CFG['R2_metrics']).read_text())
    stats={**{k:old['domains'][k] for k in ['SIM','GEN_W0','R2']},**newstats}
    comparisons={**{k:old['domain_vs_REAL'][k] for k in ['SIM','GEN_W0','R2']},**newdist}
    newc=ev.run_new_c2st(domains['R3'],domains['REAL'])
    c2st={k:old['C2ST'][k] for k in ['SIM_vs_REAL','GEN_W0_vs_REAL','R2_vs_REAL','REAL_vs_REAL']};c2st['R3_vs_REAL']=newc
    associations={}
    for k,v in [('visible_energy_proxy_MeV',meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV']),('visible_hadron_fraction',meta['physical_r_h'])]:
        associations[k]={f:float(ev.spearmanr(v,domains['R3']['features'][f]).statistic) for f in ['PSD','Qtotal_mVns','peak_V','rise_10_90_ns']}
    viewfeatures=[ev.features(w) for w in waves]
    random_diag={k:{'median_within_parent_SD':float(np.median(np.std(np.stack([d['features'][k] for d in viewfeatures]),axis=0,ddof=1))),'mean_within_parent_SD':float(np.mean(np.std(np.stack([d['features'][k] for d in viewfeatures]),axis=0,ddof=1)))} for k in ['PSD','Qtotal_mVns','peak_V','rise_10_90_ns','baseline_RMS_V']}
    conditional=condition_pair_examples(model,dc_t,dq_t,meta)
    metric={'status':'complete','training':training,'domains':stats,'domain_vs_REAL':comparisons,'C2ST':c2st,'REAL_group_reference':old['REAL_group_reference'],'parent_association_R3':associations,'three_view_within_parent_variation':random_diag,'fixed_latent_condition_examples':conditional,'generation':{'main_seed':CFG['generation_seeds'][0],'extra_views_not_independent_samples':True,'all_same_2000_R1_G4_parents':True,'physical_voltage':'learned TRAIN charge gain/energy exponent;not calibrated external sensor gain','padding_count':int(params[0]['padding'].sum()),'ADC_clip_count':int(params[0]['quantization_clip'].sum()),'saved_model_first16_bitwise_reload':True},'CAL_charge_edges_mVns':ref['CAL_charge_quantile_edges_mV_ns'],'CAL_reference_valid':[s['valid'] for s in ref['slots']],'q4_no_fallback':True,'historically_exposed_DEV':True,'current_model_fit_DEV_events':0,'EVAL_read':0,'new_G4':0,'CPU_threads':4,'worker_PID':os.getpid(),'R1_R2_metrics_reused_not_refit':True,'Torch_seed_set_before_model_constructor':True,'R2_fixed_checkpoint_input_SHA256':CFG['R2_checkpoint']['sha256'],'elapsed_before_plots_s':time.monotonic()-START}
    write('metrics.json',metric);write('INPUTS.json',INPUTS)
    ev.plots(domains,ref,stats,comparisons,c2st,meta)
    ev.plot_stochastic_examples(domains,meta,waves)
    metric['numeric_elapsed_s']=time.monotonic()-START;write('metrics.json',metric)
    (ROOT/'metrics.json').write_text((OUT/'metrics.json').read_text(),encoding='utf-8')
    log(f'R3 complete wall={metric["numeric_elapsed_s"]:.3f}s updates={training["optimizer_updates"]} selected={training["selected_update"]}')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write('FAILURE.json',{'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'error':repr(e),'traceback':traceback.format_exc()});raise
