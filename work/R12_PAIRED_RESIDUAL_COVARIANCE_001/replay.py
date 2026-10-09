from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import pathlib,json,time,datetime,hashlib,traceback
START=time.monotonic();ROOT=pathlib.Path(__file__).resolve().parent
import numpy as np
import density as den
cfg=json.loads((ROOT/'config.json').read_text());metrics=json.loads((ROOT/'metrics.json').read_text());rows=[]
try:
    for name,row in metrics['models'].items():
        selected=row.get('selected_K_frozen')
        if selected is None:rows.append({'name':name,'replayed':False,'reason':'undefined selected model'});continue
        p=ROOT/name
        with np.load(p/'representation.npz',allow_pickle=False) as z:rep={k:z[k].copy() for k in z.files}
        rep['kind']=str(rep['kind'].item())
        with np.load(p/f'GMM_{selected}.npz',allow_pickle=False) as z:gmm={k:z[k].copy() for k in z.files}
        with np.load(p/'GENERATION_INPUTS.npz',allow_pickle=False) as z:inp={k:z[k].copy() for k in z.files}
        with np.load(p/'generated.npz',allow_pickle=False) as z:saved=z['pre_ADC_V'].copy();post=z['waveforms_V'].copy();ids=z['draw_ids'].copy()
        if not np.array_equal(ids,inp['draw_ids']):raise RuntimeError('saved draw order mismatch')
        actual=den.invert_samples(rep,gmm,inp,cfg['generation_chunk']);newpost,_=den.adc(actual);finite=np.isfinite(actual)&np.isfinite(saved);maxabs=float(np.max(np.abs(actual[finite]-saved[finite]))) if finite.any() else None
        prepass=bool(np.allclose(actual,saved,rtol=cfg['pre_ADC_replay_rtol'],atol=cfg['pre_ADC_replay_atol_V'],equal_nan=True));postpass=bool(np.array_equal(post,newpost,equal_nan=True))
        rows.append({'name':name,'rows':len(post),'values':post.size,'pre_ADC_maxabs_V':maxabs,'pre_ADC_tolerance_pass':prepass,'pre_ADC_bitwise':bool(np.array_equal(actual,saved,equal_nan=True)),'post_ADC_array_equal':postpass,'post_ADC_payload_sha256':hashlib.sha256(newpost.tobytes()).hexdigest(),'response_refit':0})
    result={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,'status':'pass' if all(r.get('pre_ADC_tolerance_pass',True) and r.get('post_ADC_array_equal',True) for r in rows) else 'mismatch','models':rows,'fit_attempts':0,'full_evaluation_rerun':0,'migration_test':False,'threads':4,'EVAL_read':0}
    (ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='pass':raise RuntimeError('Scientific model replay mismatch; saved outputs preserved')
except BaseException as e:
    (ROOT/'REPLAY_FAILURE.json').write_text(json.dumps({'PID':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START},indent=2)+'\n');raise
