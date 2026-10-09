from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,hashlib
import numpy as np,torch
import density as den
import conditional_density as cd
import residual_flow as rf
import head_base as hb
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text());start=time.monotonic();torch.set_num_threads(4);torch.set_num_interop_threads(4)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
old=pathlib.Path(cfg['remote_R14_root']);old15=pathlib.Path(cfg['remote_R15_root']);rep=npz(old/'representation.npz');rep['kind']=str(rep['kind'].item());inp=npz(ROOT/'ACTUAL_GENERATION_INPUTS.npz')
head=cd.Flow(cfg['head_seed']).double();head.load_state_dict(torch.load(old/'FLOW_EXPOSURES.pt',map_location='cpu',weights_only=True));head.eval()
tail=cd.ConditionalTail().double();tail.load_state_dict(torch.load(old/'TAIL_EXPOSURES.pt',map_location='cpu',weights_only=True));tail.eval()
flow=rf.ResidualFlow(cfg['seed']).double();flow.load_state_dict(torch.load(old15/'RESIDUAL_FLOW.pt',map_location='cpu',weights_only=True));flow.eval()
results={};passed=True
for name in ['GMM8']:
    if not (ROOT/(name+'_BASE.npz')).exists():passed=False;results[name]={'status':'unavailable'};continue
    params=npz(ROOT/(name+'_BASE.npz'));x,latent,component=hb.sample_heads(name,params,head,inp,inp['GMM8_uniform'],cfg['generation_chunk']);pre=rf.generate(flow,tail,x,rep,inp,cfg['generation_chunk']);post,_=den.adc(pre);saved=npz(ROOT/name/'generated.npz');ma=float(np.max(np.abs(pre-saved['pre_ADC_V'])))
    result={'rows':len(pre),'values':pre.size,'draw_ids_equal':bool(np.array_equal(inp['draw_ids'],saved['draw_ids'])),'head_x_array_equal':bool(np.array_equal(x,inp[name+'__x'])),'latent_array_equal':bool(np.array_equal(latent,inp[name+'__latent'])),'component_array_equal':bool(np.array_equal(component,inp[name+'__component'])) if component is not None else None,'pre_ADC_maxabs_V':ma,'pre_ADC_tolerance_pass':bool(ma<=1e-10),'pre_ADC_bitwise':bool(np.array_equal(pre,saved['pre_ADC_V'])),'post_ADC_array_equal':bool(np.array_equal(post,saved['waveforms_V'])),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest()}
    passed=passed and result['rows']==2000 and result['draw_ids_equal'] and result['pre_ADC_tolerance_pass'] and result['post_ADC_array_equal'];results[name]=result
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'status':'pass' if passed else 'failed','results':results,'total_rows':sum(r.get('rows',0) for r in results.values()),'Gaussian2000_already_passed_in_original_replay':True,'combined_replayed_rows':2000+sum(r.get('rows',0) for r in results.values()),'fit_attempts':0,'new_random_draws':0,'full_evaluation_rerun':0,'migration_test':False,'threads':4,'EVAL_read':0,'arithmetic':'fresh-process CPUfloat64 saved base parameters, actual head_white/tail_white/uniform and fixed R14/R15 weights'}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(0 if passed else 1)
