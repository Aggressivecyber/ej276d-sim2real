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
rep=npz(ROOT/'representation.npz');rep['kind']=str(rep['kind'].item());gmm=npz(ROOT/'GMM_8.npz');inp=npz(ROOT/'GENERATION_INPUTS.npz');m=json.loads((ROOT/'metrics.json').read_text());rows=[]
for budget in ['UPDATES','EXPOSURES']:
    flow=cd.Flow(cfg['head_seed']).double();flow.load_state_dict(torch.load(ROOT/('FLOW_'+budget+'.pt'),map_location='cpu',weights_only=True));flow.eval()
    tail=cd.ConditionalTail().double();tail.load_state_dict(torch.load(ROOT/('TAIL_'+budget+'.pt'),map_location='cpu',weights_only=True));tail.eval()
    _,x=cd.sample_heads(rep,gmm,flow,inp,cfg['generation_chunk'])
    for tn in ['GLOBAL','CONDITIONAL']:
        name=budget+'_'+tn
        if m['models'][name]['status']!='complete':continue
        pre=cd.generate(name,rep,x,tail,inp,cfg['generation_chunk']);post,_=den.adc(pre);old=npz(ROOT/name/'generated.npz');ma=float(np.max(np.abs(pre-old['pre_ADC_V'])))
        rows.append({'name':name,'rows':len(pre),'values':pre.size,'pre_ADC_maxabs_V':ma,'pre_ADC_tolerance_pass':bool(ma<=1e-10),'pre_ADC_bitwise':bool(np.array_equal(pre,old['pre_ADC_V'])),'post_ADC_array_equal':bool(np.array_equal(post,old['waveforms_V'])),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest(),'response_refit':0})
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'status':'pass' if len(rows)==4 and all(r['pre_ADC_tolerance_pass'] and r['post_ADC_array_equal'] for r in rows) else 'failed_or_incomplete','models':rows,'fit_attempts':0,'new_random_draws':0,'full_evaluation_rerun':0,'migration_test':False,'threads':4,'EVAL_read':0,'arithmetic':'fresh-process CPUfloat64 actual saved selected weights and innovations'}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(0 if receipt['status']=='pass' else 1)
