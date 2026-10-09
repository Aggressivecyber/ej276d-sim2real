from __future__ import annotations
import time,os,sys,json,platform
START=time.monotonic()
import torch,numpy as np
from support import *
from model import Translator,infer,input_channels,adc
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(20261009220)
old=json.loads(Path(CFG['old_config']).read_text('utf8'))
paths=old['local_input_paths']
def selected_binary(name,dtype,width=1):
    dt=np.dtype(dtype);p=Path(paths[name]);n=3200*width*dt.itemsize
    with p.open('rb')as f:f.seek(2000*width*dt.itemsize);v=f.read(n)
    if len(v)!=n:raise RuntimeError('short TRAIN-only read '+name)
    return np.frombuffer(v,dtype=dt).copy().reshape(3200,width)if width>1 else np.frombuffer(v,dtype=dt).copy()
v=selected_binary('REAL_V','<f4',480);qc=selected_binary('REAL_QC','<u2');idx=selected_binary('REAL_original_index','<u4');group=selected_binary('REAL_groups','<u2');role=selected_binary('REAL_roles','u1')
assert np.all(role==1)and np.array_equal(group,idx//50)
valgroups=json.loads(Path(paths['TRAIN_roles']).read_text('utf8'))['REAL_internal_validation_artificial_groups']
fit=(qc==0)&~np.isin(group,valgroups);val=(qc==0)&np.isin(group,valgroups)
assert fit.sum()==2592 and val.sum()==599
with np.load(CFG['REAL_addition'],allow_pickle=False)as z:
    good=z['QC_reason_u16']==0
    add=z['waveforms_V'][good].astype('f4');ai=z['original_index'][good].copy()
assert len(add)==18455
target=np.concatenate([v[fit],add]);realidx=np.concatenate([idx[fit],ai]);rv=v[val];rvi=idx[val]
assert len(target)==21047 and not np.isin(realidx//50,valgroups).any()
rb=bins(target);rvb=bins(rv)
# Freeze scaler on ALL REALfit, including support-outside, and no other role.
with torch.no_grad():ch=input_channels(torch.from_numpy(target)).numpy().astype('f8')
mu=ch.mean(0);sd=ch.std(0,ddof=0);floor=np.maximum(.05*np.median(sd,axis=1),1e-6);scale=np.maximum(sd,floor[:,None]);del ch
with np.load(CFG['G4_cache'],allow_pickle=False)as z:
    ids=z['event_ids'];mask=z['internal_training'].astype(bool)
    assert mask.sum()==6593
    vi=np.flatnonzero(~mask);vi=vi[order(ids[vi],'R18-source-validation')[:512]]
    s=z['clean_expectation_V'][mask].astype('f4');sv=z['clean_expectation_V'][vi].astype('f4')
    si=ids[mask].copy();svi=ids[vi].copy()
    ee=z['edep_ee_MeV'];had=z['edep_hadron_MeV'];h=np.divide(.15*had,ee+.15*had,out=np.zeros(len(ee),'f8'),where=(ee+.15*had)>0)
    sh=h[mask];svh=h[vi]
sb=bins(s);svb=bins(sv);sm=(sb>=0)&(sb<8);vm=(svb>=0)&(svb<8);rm=(rb>=0)&(rb<8);rvm=(rvb>=0)&(rvb<8)
assert sm.sum()==3753 and vm.sum()==295 and rm.sum()==18895
lambdas=np.array([a['lambda_value']for a in json.loads(Path(CFG['tilt_table']).read_text('utf8'))['bins']])
s=s[sm];sv=sv[vm];si=si[sm];svi=svi[vm];sb=sb[sm];svb=svb[vm];sh=sh[sm];svh=svh[vm];rb=rb[rm];rvb=rvb[rvm];target=target[rm];realidx=realidx[rm];rv=rv[rvm];rvi=rvi[rvm]
sw=weights(sb,sh,lambdas);svw=weights(svb,svh,lambdas);rw=weights(rb);rvw=weights(rvb)
def noise(n,style_seed,old_seed):
    r=np.random.default_rng(old_seed);z=r.standard_normal((n,8)).astype('f4');w=r.standard_normal((n,480)).astype('f4');style=np.random.default_rng(style_seed).standard_normal((n,8)).astype('f4');return z,w,style
z,w,st=noise(len(s),CFG['seeds']['train_style'],CFG['seeds']['train_old_noise']);vz,vw,vst=noise(len(sv),CFG['seeds']['val_style'],CFG['seeds']['val_old_noise'])
g=Translator();g.load_state_dict(torch.load(CFG['G_weights'],map_location='cpu',weights_only=False));g.eval()
for p in g.parameters():p.requires_grad_(False)
log({'PID':os.getpid(),'phase':'frozen_generation','TRAIN_source':len(s),'val_source':len(sv),'REAL_fit':len(target),'REAL_val':len(rv)})
y,signal,electronic=infer(g,s,z,w,st);yv,vs,ve=infer(g,sv,vz,vw,vst)
post,ap=adc(y);vpost,vap=adc(yv)
assert y.shape==(3753,480)and yv.shape==(295,480)and np.isfinite(y).all()and np.isfinite(yv).all()
# Every arm consumes exactly these saved draws; indices are relative to fixed pool.
rng=np.random.default_rng(CFG['seeds']['sampling']);sample_b=rng.integers(0,8,size=(4000,32),dtype=np.int16);sample_s=np.empty_like(sample_b,dtype='<i4');sample_r=np.empty_like(sample_s)
for b in range(8):
    m=sample_b==b;a=np.flatnonzero(sb==b);pro=sw[a]/sw[a].sum();rr=np.flatnonzero(rb==b)
    sample_s[m]=rng.choice(a,m.sum(),p=pro);sample_r[m]=rng.choice(rr,m.sum())
np.savez_compressed(ROOT/'FIXED_POOL_PRIVATE.npz',real_train=target,real_val=rv,gen_train_pre=y,gen_train_post=post,gen_val_pre=yv,gen_val_post=vpost,
 source_train=s,source_val=sv,source_train_ids=si,source_val_ids=svi,real_train_original_index=realidx,real_val_original_index=rvi,
 real_train_bin=rb,real_val_bin=rvb,gen_train_bin=sb,gen_val_bin=svb,gen_train_h=sh,gen_val_h=svh,
 real_train_weight=rw,real_val_weight=rvw,gen_train_weight=sw,gen_val_weight=svw,
 z_train=z,white_train=w,style_train=st,z_val=vz,white_val=vw,style_val=vst,mu=mu,scale=scale,std=sd,floor=floor,
 sampled_bin=sample_b,sampled_gen_index=sample_s,sampled_real_index=sample_r,lambdas=lambdas,
 gen_train_signal=signal,gen_train_electronic=electronic,gen_val_signal=vs,gen_val_electronic=ve)
rec={'PID':os.getpid(),'UTC':utc(),'elapsed_s':time.monotonic()-START,'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
 'G_updated':0,'D_updated':0,'generated_n':len(y)+len(yv),'source_train_loaded':6593,'source_train_common':len(y),'source_val_fixed':512,'source_val_common':len(yv),
 'REAL_fit_loaded':21047,'REAL_fit_common':len(target),'REAL_internal_val_loaded':599,'REAL_internal_val_common':len(rv),
 'support_counts':{k:np.bincount(b,minlength=8)for k,b in [('G4_TRAIN',sb),('G4_val',svb),('REAL_TRAIN',rb),('REAL_val',rvb)]},
 'scaler':{'population_n':21047,'floor':floor,'clipped_coordinate_n':np.sum(sd<floor[:,None],axis=1),'STD_minmax':[sd.min(1),sd.max(1)],'scale_minmax':[scale.min(1),scale.max(1)]},
 'ADC_train':ap,'ADC_val':vap,'new_generation_seeded_once':True,'EVAL':0,'DEV':0,'CAL':0,'new_G4':0,'raw_long_decode':0,'all_parent_ids_saved':True,
 'unpaired_domains':True,'REAL_groups_are_same_acquisition_artificial50':True,'sampling_algorithm':'pre-draw uniform bins then per-bin source TILT and REAL uniform, single preserved arrays reused by both arms'}
write('POOL_RECEIPT.json',rec);log(rec)
