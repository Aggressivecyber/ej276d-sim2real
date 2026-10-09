from __future__ import annotations
import os,time,json,pathlib
START=time.monotonic()
import numpy as np,torch
import data_support as ds,evaluation as ev
ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'));OLD=pathlib.Path(CFG['R21_root'])
torch.set_num_threads(4);torch.set_num_interop_threads(4)
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def save(n,v):ds.write(n,v)
xs,xt,vs,vt,source,ids,comp,pdg,real,rr,ref,edges,sel,roles=ds.inputs()
gc=ds.zload(pathlib.Path(CFG['remote_input_root'])/'G4_TRAIN_cache');mask=gc['internal_training'].astype(bool);av=np.flatnonzero(~mask);vi=av[ev.order(gc['event_ids'][av],'R18-source-validation')[:512]]
hh=.15*gc['edep_hadron_MeV'];h=np.divide(hh,gc['edep_ee_MeV']+hh,out=np.zeros_like(hh,dtype='f8'),where=(gc['edep_ee_MeV']+hh)>0)
sb=np.searchsorted(edges,np.log(ev.features(xs)['features']['Qtotal_mVns']),side='right')-1;tb=np.searchsorted(edges,np.log(ev.features(xt)['features']['Qtotal_mVns']),side='right')-1
assert ((sb>=0)&(sb<8)).sum()==3753 and ((tb>=0)&(tb<8)).sum()==18895
lambdas=np.array([r['lambda_value']for r in json.loads((ROOT/'FROZEN_TILT_TABLE.json').read_text())['bins']])
np.savez_compressed(ROOT/'TRAIN_PRIVATE.npz',source=xs,target=xt,source_val=vs,target_val=vt,h=h[mask],hval=h[vi],source_bin=sb,target_bin=tb,source_ids=gc['event_ids'][mask],source_val_ids=gc['event_ids'][vi],lambdas=lambdas,edges=edges)
vv=load(pathlib.Path(CFG['remote_R18_root'])/'VALIDATION_INPUTS.npz')
assert np.array_equal(vv['source'],vs)
vv['style']=load(OLD/'GLOBAL_STYLE/STYLE_VALIDATION_INPUTS.npz')['style']
np.savez_compressed(ROOT/'VALIDATION_INPUTS.npz',**vv)
np.savez_compressed(ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz',real_V=real['V'].astype('f4'),real_ids=real['ids'],real_groups=real['groups'],real_QC_valid=real['valid'],reference_V=rr['V'].astype('f4'),reference_ids=rr['ids'],reference_groups=rr['groups'])
save('CAL_REFERENCE_METADATA.json',ref);save('EVALUATOR_SELECTIONS_PRIVATE.json',sel)
actual=load(OLD/'ACTUAL_GENERATION_INPUTS.npz');assert np.array_equal(actual['source_ids'],ids);np.savez_compressed(ROOT/'ACTUAL_GENERATION_INPUTS.npz',**actual)
repeat=load(OLD/'results/joint8_REPEAT_PRIVATE.npz',keys=['source_parent_index','source_waveforms_V','z','white','style','parent_composition'])
assert len(repeat['source_parent_index'])==1024 and np.all(repeat['source_parent_index'].reshape(128,8)==repeat['source_parent_index'][::8,None])
assert np.array_equal(repeat['parent_composition'],comp[repeat['source_parent_index'][::8]])
np.savez_compressed(ROOT/'JOINT8_INPUTS_PRIVATE.npz',**repeat)
probe=load(OLD/'PROBE_INPUTS_CORRECTED.npz');assert len(probe['source_waveforms_V'])==832
np.savez_compressed(ROOT/'PROBE_INPUTS_CORRECTED.npz',**probe)
sc=load(pathlib.Path(CFG['R22_root'])/'FIXED_POOL_PRIVATE.npz',keys=['mu','scale','std','floor'])
np.savez_compressed(ROOT/'CRITIC_SCALER.npz',**sc)
save('CRITIC_SCALER_REUSE.json',{'fit_population':'R22 frozen all21047REAL TRAIN-fit','estimated_here':False,'coordinate_shape':list(sc['mu'].shape),'floor':sc['floor'],'centering_and_scale_combination':True})
rng=np.random.default_rng(CFG['source_REAL_bin_seed']);initial=rng.bit_generator.state
b=rng.integers(0,8,(4000,32),dtype=np.int16);us=rng.random((4000,32));ur=rng.random((4000,32));si=np.empty((4000,32),'i4');ti=np.empty_like(si)
for j in range(8):
    where=b==j;src=np.flatnonzero(sb==j);tar=np.flatnonzero(tb==j);weight=np.exp(lambdas[j]*h[mask][src]);cdf=np.cumsum(weight/weight.sum());cdf[-1]=1
    si[where]=src[np.searchsorted(cdf,us[where],side='right')];ti[where]=tar[(ur[where]*len(tar)).astype('i8')]
rn=np.random.default_rng(CFG['NCE_seed']);nce_initial=rn.bit_generator.state;rows=[]
for i in range(4000):
    for n in [480,240,120]:rows.append(rn.choice(range(max(1,int(80*n/480)),min(n,int(420*n/480))),CFG['patches_per_layer'],replace=False))
np.savez_compressed(ROOT/'SHARED_TRAIN_STREAMS_PRIVATE.npz',source=si.reshape(-1),target=ti.reshape(-1),bin=b.reshape(-1),source_uniform=us.reshape(-1),target_uniform=ur.reshape(-1),NCE_positions=np.asarray(rows,dtype='<u2'))
save('PREPARED_RANDOM_STATES.json',{'sampling_initial':initial,'sampling_final':rng.bit_generator.state,'NCE_initial':nce_initial,'NCE_final':rn.bit_generator.state,'old_noise_seed':CFG['old_noise_seed'],'style_seed':CFG['style_train_seed'],'complete_uint64_seeds_used':True})
save('PREPARATION_RECEIPT.json',{'PID':os.getpid(),'UTC':ev.datetime.datetime.now(ev.datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
 'source_fit_loaded':len(xs),'source_fit_common':3753,'REAL_fit_loaded':len(xt),'REAL_fit_common':18895,'source_val_used_fixed':len(vs),'REAL_val_used':len(vt),
 'main_inputs_reused':2000,'joint8_inputs_reused':1024,'probe_inputs_reused':832,'new_generation':0,'G_D_reader_updates':0,'EVAL':0,'CAL_waveform_decode':0,'critic_scaler_refitted':False,'old_source_versions_reused_no_new_provenance_audit':True})
print(json.dumps({'PID':os.getpid(),'phase':'preparation_complete','elapsed_s':time.monotonic()-START}),flush=True)
