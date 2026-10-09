from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,hashlib
import numpy as np,torch
import density as den
import conditional_density as cd
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text());start=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
rep=npz(ROOT/'representation.npz');rep['kind']=str(rep['kind'].item());gmm=npz(ROOT/'GMM_8.npz');inp=npz(ROOT/'GENERATION_INPUTS.npz');flow,tail=cd.load_models(ROOT,cfg)
gx,fx=cd.sample_heads(rep,gmm,flow,inp,cfg['generation_chunk']);m=json.loads((ROOT/'metrics.json').read_text());rows=[]
for name in cfg['models']:
    if m['models'][name]['status']!='complete':continue
    x=fx if name.startswith('FLOW') else gx;v=cd.generate(name,rep,x,tail,inp,cfg['generation_chunk']);post,adc=den.adc(v);old=npz(ROOT/name/'generated.npz')
    ma=float(np.max(np.abs(v-old['pre_ADC_V'])));pre=bool(ma<=cfg['pre_ADC_replay_atol_V']);after=bool(np.array_equal(post,old['waveforms_V']))
    rows.append({'name':name,'rows':len(v),'values':v.size,'pre_ADC_maxabs_V':ma,'pre_ADC_tolerance_pass':pre,'pre_ADC_bitwise':bool(np.array_equal(v,old['pre_ADC_V'])),'post_ADC_array_equal':after,'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest(),'response_refit':0})
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'status':'pass' if len(rows)==3 and all(r['pre_ADC_tolerance_pass'] and r['post_ADC_array_equal'] for r in rows) else 'failed_or_incomplete','models':rows,'fit_attempts':0,'new_random_draws':0,'full_evaluation_rerun':0,'migration_test':False,'threads':4,'EVAL_read':0,'arithmetic':'fresh-process CPU float64 saved selected weights and actual inputs'}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(0 if receipt['status']=='pass' else 1)
