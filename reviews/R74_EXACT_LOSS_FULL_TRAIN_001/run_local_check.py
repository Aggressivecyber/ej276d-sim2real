from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time,traceback
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'artificial_runtime')
W=R/'reviews/R73_EXACT_W1_THROUGHPUT_001/w1_candidate'
sys.path[:0]=[str(P),str(W),str(R/'work/R72_FULL_WAVEFORM_FULL_TRAIN_001')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[P/'combined_loss.py',P/'check_combined_loss.py',W/'exact_uniform_w1.py']
hashes={str(p):sha(p)for p in paths};start=time.perf_counter();exit_code=0
try:
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    from check_combined_loss import run
    result=run(20261011723,require_cuda=False)
    assert all(sha(p)==hashes[str(p)]for p in paths)
    receipt=dict(pass_all=result['pass_all'],result=result,source_SHA256=hashes)
except BaseException as exc:
    exit_code=1;receipt=dict(pass_all=False,error=repr(exc),traceback=traceback.format_exc(),source_SHA256=hashes)
receipt.update(UTC=datetime.now(timezone.utc).isoformat(),native_exit=exit_code,elapsed_s=time.perf_counter()-start,guard={**guard,**memory()},scientific_arrays_decoded=0)
name='ROOT_COMBINED_ARTIFICIAL_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json'
(P/name).write_text(json.dumps(receipt,indent=2)+'\n','utf8')
if exit_code==0:(P/'ROOT_COMBINED_ARTIFICIAL.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt),flush=True);sys.exit(exit_code)
