from pathlib import Path
import os,json,time,datetime,copy
START=time.monotonic();ROOT=Path(__file__).resolve().parent
import numpy as np
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
with np.load(ROOT/'TRAIN_PRIVATE.npz',allow_pickle=False)as z:
    sb=z['source_bin'];tb=z['target_bin'];h=z['h'];lambdas=z['lambdas']
assert np.isin(sb,np.arange(8)).sum()==3753 and np.isin(tb,np.arange(8)).sum()==18895
rng=np.random.default_rng(CFG['extra_D_sampling_seed']);initial=copy.deepcopy(rng.bit_generator.state)
shape=(CFG['updates'],CFG['extra_D_batches_per_G'],CFG['batch'])
b=rng.integers(0,8,shape,dtype=np.int16);us=rng.random(shape);ur=rng.random(shape);si=np.empty(shape,'i4');ti=np.empty_like(si)
for j in range(8):
    where=b==j;src=np.flatnonzero(sb==j);tar=np.flatnonzero(tb==j)
    weight=np.exp(lambdas[j]*h[src]);cdf=np.cumsum(weight/weight.sum());cdf[-1]=1
    si[where]=src[np.searchsorted(cdf,us[where],side='right')];ti[where]=tar[(ur[where]*len(tar)).astype('i8')]
np.savez_compressed(ROOT/'EXTRA_D_PLANNED_STREAMS_PRIVATE.npz',source=si.reshape(-1),target=ti.reshape(-1),bin=b.reshape(-1),source_uniform=us.reshape(-1),target_uniform=ur.reshape(-1))
record={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'seed':CFG['extra_D_sampling_seed'],'bit_generator':type(rng.bit_generator).__name__,'seed_not_uint32_truncated':True,'sampling_initial':initial,'sampling_final':rng.bit_generator.state,'shape_steps_extra_batches_rows':list(shape),'source_fit_loaded':6593,'source_fit_common':3753,'REAL_fit_loaded':21047,'REAL_fit_common':18895,'source_val_fixed':512,'REAL_val_fixed':599,'additional_D_planned_draws_each_domain':si.size,'original_G_streams_copied_R23':True,'new_generation_fit_or_update':0,'source_strings_new_decode':0,'EVAL_read':0,'CAL_waveform_decode':0}
(ROOT/'EXTRA_D_PREPARED_RANDOM_STATES.json').write_text(json.dumps(record,indent=2)+'\n','utf8')
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n','utf8')
print(json.dumps(record),flush=True)
