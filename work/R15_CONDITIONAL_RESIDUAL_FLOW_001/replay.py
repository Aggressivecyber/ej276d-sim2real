from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,hashlib
import numpy as np,torch
import density as den
import conditional_density as cd
import residual_flow as rf
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text());start=time.monotonic();torch.set_num_threads(4);torch.set_num_interop_threads(4)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
old=pathlib.Path(cfg['remote_R14_root']);rep=npz(old/'representation.npz');rep['kind']=str(rep['kind'].item());inp=npz(ROOT/'ACTUAL_GENERATION_INPUTS.npz')
tail=cd.ConditionalTail().double();tail.load_state_dict(torch.load(old/'TAIL_EXPOSURES.pt',map_location='cpu',weights_only=True));tail.eval()
flow=rf.ResidualFlow(cfg['seed']).double();flow.load_state_dict(torch.load(ROOT/'RESIDUAL_FLOW.pt',map_location='cpu',weights_only=True));flow.eval()
pre=rf.generate(flow,tail,inp['x'],rep,inp,cfg['generation_chunk']);post,_=den.adc(pre);saved=npz(ROOT/'RESIDUAL_FLOW/generated.npz');ma=float(np.max(np.abs(pre-saved['pre_ADC_V'])))
result={'rows':len(pre),'values':pre.size,'draw_ids_equal':bool(np.array_equal(inp['draw_ids'],saved['draw_ids'])),'pre_ADC_maxabs_V':ma,'pre_ADC_tolerance_pass':bool(ma<=1e-10),'pre_ADC_bitwise':bool(np.array_equal(pre,saved['pre_ADC_V'])),'post_ADC_array_equal':bool(np.array_equal(post,saved['waveforms_V'])),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest()}
passed=result['rows']==2000 and result['draw_ids_equal'] and result['pre_ADC_tolerance_pass'] and result['post_ADC_array_equal']
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'status':'pass' if passed else 'failed','result':result,'fit_attempts':0,'new_random_draws':0,'full_evaluation_rerun':0,'migration_test':False,'threads':4,'EVAL_read':0,'arithmetic':'fresh-process CPUfloat64 saved actual head, innovations and selected weights'}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(0 if passed else 1)
