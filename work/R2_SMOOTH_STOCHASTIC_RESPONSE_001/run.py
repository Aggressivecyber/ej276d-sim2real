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
OUT=ROOT/'results';OUT.mkdir(exist_ok=False)
ev.OUT=OUT;ev.FIG=OUT/'figures';ev.FIG.mkdir()
ev.CFG=CFG['evaluation'];ev.NAMES=['SIM','GEN_W0','R2','REAL']
ev.COLORS={'SIM':'#2878b5','GEN_W0':'#a33ea2','R2':'#e87425','REAL':'#181818'}
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
        for k in ['sim_event_ids','real_event_ids']:assert np.array_equal(z[k],meta[k])
        assert np.array_equal(z['before'],meta['raw']) and np.array_equal(z['real'],meta['real'])
        historic=z['after'].copy()
    base=pathlib.Path(CFG['REAL_prefix_dir']);ident=pathlib.Path(CFG['REAL_identity_dir'])
    real=prefix(base/'window_V_f32.bin','<f4',5200*480).reshape(5200,480)
    qcreason=prefix(base/'QC_reason_u16.bin','<u2',5200)
    groups=prefix(ident/'selection_group.bin','<u2',5200)
    orig=prefix(ident/'selection_original_index.bin','<u4',5200)
    roles=prefix(ident/'selection_role.bin','u1',5200)
    assert np.array_equal(groups,orig//50) and np.all(roles[:2000]==0) and np.all(roles[2000:]==1)
    normalized=prefix(base/'window_normalized_f32.bin','<f4',2000*480).reshape(2000,480)
    assert np.array_equal(normalized,meta['real'])
    train=real[2000:];tgroups=groups[2000:];qc=qcreason[2000:]==0
    assert qc.sum()==3191
    valid_train_groups=set(tgroups[qc].tolist())
    valgroups={g for g in valid_train_groups if ev.hashint('R2-REAL-TRAIN-VALIDATION|'+str(g))%5==0}
    rt=qc&~np.isin(tgroups,list(valgroups));rv=qc&np.isin(tgroups,list(valgroups))
    assert rt.sum()>=2000 and rv.sum()>=300 and not set(tgroups[rt])&set(tgroups[rv])
    csv=ROOT/'data/edep_nt_events.csv';cb=csv.read_bytes();assert hashlib.sha256(cb).hexdigest()==CFG['G4_CSV_SHA256']
    rows=np.loadtxt(io.BytesIO(cb),delimiter=',',comments='#',dtype=np.float64)
    assert rows.shape[1]==5 and np.all(np.isfinite(rows))
    eid=rows[:,0].astype(np.int64);ee=rows[:,2];had=rows[:,3]
    assert np.all(rows[:,0]==eid) and len(set(eid.tolist()))==len(eid) and np.all(ee>=0)&np.all(had>=0)
    source=CFG['G4_source'];ids=np.array(['sim:source:'+hashlib.sha256(json.dumps([source,str(int(e))],separators=(',',':')).encode()).hexdigest() for e in eid])
    split=np.array([int.from_bytes(hashlib.sha256(('event_hash_v1|42222|sim|'+str(s)).encode()).digest()[:8],'big')/2**64 for s in ids])
    devset=set(meta['sim_event_ids'].tolist());matches=np.isin(ids,meta['sim_event_ids'])
    assert matches.sum()==2000,'G4 source identity mismatch'
    byid={str(s):i for i,s in enumerate(ids)}
    devrow=np.array([byid[str(s)] for s in meta['sim_event_ids']])
    assert np.allclose(ee[devrow],meta['edep_ee_MeV'],rtol=0,atol=1e-12)
    assert np.allclose(had[devrow],meta['edep_hadron_MeV'],rtol=0,atol=1e-12)
    eligible=np.flatnonzero((split>=.2)&(ee+.15*had>0))
    eligible=eligible[sorted(range(len(eligible)),key=lambda j:hashlib.sha256(('order|42222|sim|'+str(ids[eligible[j]])).encode()).digest())][:8192]
    assert len(eligible)==8192 and not devset&set(ids[eligible].tolist())
    selected=rows[eligible];trainids=ids[eligible]
    clean,qs=clean_expectation(selected[:,2],selected[:,3]);cond,energy,h=condition(clean,selected[:,2],selected[:,3])
    gt=np.array([ev.hashint('R2-G4-TRAIN-VALIDATION|'+str(s))%5!=0 for s in trainids])
    dc,dq=clean_expectation(ee[devrow],had[devrow]);devcond,de,dh=condition(dc,ee[devrow],had[devrow])
    np.savez_compressed(ROOT/'data/G4_TRAIN_CACHE.npz',clean_expectation_V=clean,condition=cond,Qsource_Vns=qs,event_ids=trainids,source_event_id=selected[:,0].astype('i8'),pdg=selected[:,1].astype('i4'),edep_ee_MeV=selected[:,2],edep_hadron_MeV=selected[:,3],visible_energy_proxy_MeV=energy,visible_hadron_fraction=h,internal_training=gt)
    audit={'CSV_rows':len(rows),'original_source_PDG_counts':{str(int(k)):int((rows[:,1]==k).sum()) for k in np.unique(rows[:,1])},'selected_G4_TRAIN':8192,'selected_source_PDG_counts':{str(int(k)):int((selected[:,1]==k).sum()) for k in np.unique(selected[:,1])},'G4_internal_train_n':int(gt.sum()),'G4_internal_validation_n':int((~gt).sum()),'G4_DEV_identity_matches':int(matches.sum()),'G4_TRAIN_DEV_intersection':0,'G4_weighting':'none;original unweighted CSV source, deterministic hash order','REAL_source_TRAIN':3200,'REAL_QC_valid_TRAIN':int(qc.sum()),'REAL_internal_fit':int(rt.sum()),'REAL_internal_validation':int(rv.sum()),'REAL_internal_validation_artificial_groups':sorted(map(int,valgroups)),'REAL_group_train_validation_intersection':0,'EVAL_read':0,'CSV_zero_deposition_histories_not_stored':True,'source_provenance':'same SHA and unique source-ID rule;existing G4 assumptions reused,not new physical validation'}
    write('DATA_ROLES.json',audit)
    ref=json.loads((base/'STATISTICS.json').read_text())
    return meta,historic,real[:2000],qcreason[:2000],groups[:2000],orig[:2000],ref,train,rt,rv,cond,qs,gt,devcond,dq

def sample(model,c,q,n,seed):
    gen=torch.Generator().manual_seed(seed)
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
    model=SmoothResponse(np.log(gains),lm,ls,spec)
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
        regular=.0001*sum(p.square().mean() for p in list(model.mean.parameters())+list(model.random.parameters()))
        total=loss+conditional+regular
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
    torch.save({'state_dict':state,'constructor':{'log_gain':np.log(gains),'noise_log_mean':lm,'noise_log_sd':ls,'noise_spectrum':spec.tolist()},'best_update':beststep,'config':CFG},ROOT/'model.pt')
    write('TRAINING_TRACE.json',trace)
    receipt={'optimizer_updates':updates,'selected_update':beststep,'best_fixed_TRAIN_validation':best,'initial_TRAIN_validation':trace[0]['validation'],'training_elapsed_s':time.monotonic()-trainstart,'learned_charge_gain':float(torch.exp(model.log_gain)),'learned_energy_exponent':float(1+.4*torch.tanh(model.energy_slope)),'learned_noise_log_RMS_V':float(model.noise_log_mean),'learned_noise_log_sd':float(model.noise_log_sd),'mean_stochastic_parameter_norms':{'mean':float(sum(p.square().sum() for p in model.mean.parameters()).sqrt()),'random':float(sum(p.square().sum() for p in model.random.parameters()).sqrt())},'random_scales':model.random_scale.detach().numpy(),'baseline_spectrum_resolution_MHz':10,'no_DEV_checkpoint_selection':True}
    write('TRAINING_RECEIPT.json',receipt);return model,receipt

def main():
    log(f'R2 worker PID={os.getpid()} torch={torch.__version__} CPU_threads={torch.get_num_threads()}')
    meta,historic,real,sourceqc,groups,orig,ref,realtrain,rt,rv,c,q,gt,dc,dq=inputs()
    log('G4 parents bound;8192TRAIN/2000DEV disjoint;REAL3191QC TRAIN retained')
    model,training=train_model(realtrain,rt,rv,c,q,gt)
    dc_t=torch.tensor(dc);dq_t=torch.tensor(dq)
    waves=[];params=[];latent=[]
    for seed in CFG['generation_seeds']:
        w,p,z=sample(model,dc_t,dq_t,2000,seed);waves.append(w);params.append(p);latent.append(z)
    assert all(np.all(np.isfinite(w)) for w in waves)
    np.savez_compressed(ROOT/'generated.npz',waveforms_V=waves[0],independent_view_2_V=waves[1],independent_view_3_V=waves[2],sim_event_ids=meta['sim_event_ids'],edep_ee_MeV=meta['edep_ee_MeV'],edep_hadron_MeV=meta['edep_hadron_MeV'],visible_energy_proxy_MeV=meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV'],visible_hadron_fraction=meta['physical_r_h'],G4_condition=dc,latent_z=latent[0],**{'main_'+k:v for k,v in params[0].items()})
    # One bounded re-open of saved weights checks the concrete deliverable.
    loaded=torch.load(ROOT/'model.pt',map_location='cpu',weights_only=False)
    reloaded=SmoothResponse(**loaded['constructor']);reloaded.load_state_dict(loaded['state_dict']);reloaded.eval()
    # The white random sequence differs if requesting only16 instead of2000;
    # compare explicit saved latents/white rather than a misleading seed claim.
    gg=torch.Generator().manual_seed(CFG['generation_seeds'][0]);zz=torch.randn((2000,8),generator=gg);ww=torch.randn((2000,512),generator=gg)
    with torch.no_grad():check,_=reloaded(dc_t[:16],dq_t[:16],zz[:16],ww[:16])
    assert np.array_equal(check.numpy(),waves[0][:16]),'saved model deterministic reload mismatch'
    domains={'SIM':ev.features(meta['raw'].astype(float)*meta['normalization_peak_V'][:,None]),'GEN_W0':ev.features(historic.astype(float)*meta['normalization_peak_V'][:,None]),'R2':ev.features(waves[0]),'REAL':ev.features(real)}
    domains['REAL']['valid']&=sourceqc==0
    for name,d in domains.items():
        d['ids']=meta['real_event_ids'] if name=='REAL' else meta['sim_event_ids'];d['groups']=groups.astype(int) if name=='REAL' else np.arange(2000)
        d['region']=np.searchsorted(ref['CAL_charge_quantile_edges_mV_ns'],d['features']['Qtotal_mVns'],side='right').astype('u1')
    newstats,newdist=ev.characterize({'R2':domains['R2'],'REAL':domains['REAL']},ref)
    old=json.loads(pathlib.Path(CFG['R1_metrics']).read_text())
    stats={**{k:old['domains'][k] for k in ['SIM','GEN_W0']},**newstats}
    comparisons={**{k:old['domain_vs_REAL'][k] for k in ['SIM','GEN_W0']},**newdist}
    newc=ev.run_new_c2st(domains['R2'],domains['REAL'])
    c2st={k:old['C2ST'][k] for k in ['SIM_vs_REAL','GEN_W0_vs_REAL','REAL_vs_REAL']};c2st['R2_vs_REAL']=newc
    associations={}
    for k,v in [('visible_energy_proxy_MeV',meta['edep_ee_MeV']+.15*meta['edep_hadron_MeV']),('visible_hadron_fraction',meta['physical_r_h'])]:
        associations[k]={f:float(ev.spearmanr(v,domains['R2']['features'][f]).statistic) for f in ['PSD','Qtotal_mVns','peak_V','rise_10_90_ns']}
    viewfeatures=[ev.features(w) for w in waves]
    random_diag={k:{'median_within_parent_SD':float(np.median(np.std(np.stack([d['features'][k] for d in viewfeatures]),axis=0,ddof=1))),'mean_within_parent_SD':float(np.mean(np.std(np.stack([d['features'][k] for d in viewfeatures]),axis=0,ddof=1)))} for k in ['PSD','Qtotal_mVns','peak_V','rise_10_90_ns','baseline_RMS_V']}
    conditional=ev.fixed_latent_condition_examples(model,dc_t,dq_t,meta)
    metric={'status':'complete','training':training,'domains':stats,'domain_vs_REAL':comparisons,'C2ST':c2st,'REAL_group_reference':old['REAL_group_reference'],'parent_association_R2':associations,'three_view_within_parent_variation':random_diag,'fixed_latent_condition_examples':conditional,'generation':{'main_seed':CFG['generation_seeds'][0],'extra_views_not_independent_samples':True,'all_same_2000_R1_G4_parents':True,'physical_voltage':'learned TRAIN charge gain/energy exponent;not calibrated external sensor gain','padding_count':int(params[0]['padding'].sum()),'ADC_clip_count':int(params[0]['quantization_clip'].sum()),'saved_model_first16_bitwise_reload':True},'CAL_charge_edges_mVns':ref['CAL_charge_quantile_edges_mV_ns'],'CAL_reference_valid':[s['valid'] for s in ref['slots']],'q4_no_fallback':True,'historically_exposed_DEV':True,'current_model_fit_DEV_events':0,'EVAL_read':0,'new_G4':0,'CPU_threads':4,'worker_PID':os.getpid(),'R1_metrics_reused_not_refit':True,'elapsed_before_plots_s':time.monotonic()-START}
    write('metrics.json',metric);write('INPUTS.json',INPUTS)
    ev.plots(domains,ref,stats,comparisons,c2st,meta)
    ev.plot_stochastic_examples(domains,meta,waves)
    metric['numeric_elapsed_s']=time.monotonic()-START;write('metrics.json',metric)
    (ROOT/'metrics.json').write_text((OUT/'metrics.json').read_text(),encoding='utf-8')
    log(f'R2 complete wall={metric["numeric_elapsed_s"]:.3f}s updates={training["optimizer_updates"]} selected={training["selected_update"]}')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write('FAILURE.json',{'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'error':repr(e),'traceback':traceback.format_exc()});raise
