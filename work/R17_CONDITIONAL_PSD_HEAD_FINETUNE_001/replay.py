from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,hashlib
import numpy as np,torch
import conditional_density as cd
import residual_flow as rf
import density as den
import psd_finetune as pf
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text());start=time.monotonic();torch.set_num_threads(4);torch.set_num_interop_threads(4)
def npz(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
old14=pathlib.Path(cfg['remote_R14_root']);old15=pathlib.Path(cfg['remote_R15_root']);old16=pathlib.Path(cfg['remote_R16_completion_root']);rep=npz(old14/'representation.npz');base=npz(old16/'GMM8_BASE.npz');inp=npz(ROOT/'ACTUAL_GENERATION_INPUTS.npz')
tail=cd.ConditionalTail().double();tail.load_state_dict(torch.load(old14/'TAIL_EXPOSURES.pt',map_location='cpu',weights_only=True));tail.eval();flow=rf.ResidualFlow(cfg['seed']).double();flow.load_state_dict(torch.load(old15/'RESIDUAL_FLOW.pt',map_location='cpu',weights_only=True));flow.eval();kernel=pf.FixedKernel(tail,flow,rep,base);results={};passed=True
for name in cfg['models']:
    head=cd.Flow(cfg['head_architecture_seed']).double();head.load_state_dict(torch.load(ROOT/name/'SELECTED_HEAD.pt',map_location='cpu',weights_only=True));head.eval()
    with torch.no_grad():pre=kernel.generate_chunks(head,torch.from_numpy(inp['GMM8__latent']),torch.from_numpy(inp['tail_white']),cfg['generation_chunk']).numpy()
    post,_=den.adc(pre);saved=npz(ROOT/name/'generated.npz');error=float(np.max(np.abs(pre-saved['pre_ADC_V'])));row={'rows':len(pre),'values':pre.size,'draw_ids_equal':bool(np.array_equal(inp['draw_ids'],saved['draw_ids'])),'pre_ADC_maxabs_V':error,'pre_ADC_tolerance_pass':error<=1e-10,'pre_ADC_bitwise':bool(np.array_equal(pre,saved['pre_ADC_V'])),'post_ADC_array_equal':bool(np.array_equal(post,saved['waveforms_V'])),'post_ADC_payload_sha256':hashlib.sha256(post.tobytes()).hexdigest()};passed=passed and row['rows']==2000 and row['draw_ids_equal'] and row['pre_ADC_tolerance_pass'] and row['post_ADC_array_equal'];results[name]=row
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'status':'pass'if passed else 'failed','results':results,'total_rows':sum(r['rows']for r in results.values()),'fit_attempts':0,'new_random_draws':0,'full_evaluation_rerun':0,'migration_test':False,'threads':4,'EVAL_read':0,'arithmetic':'fresh-process CPUfloat64 saved selected heads, fixed GMM8 and R15 kernel, actual latent/tailwhite'};(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True);raise SystemExit(0 if passed else 1)
