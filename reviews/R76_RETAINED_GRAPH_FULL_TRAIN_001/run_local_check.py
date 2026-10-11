from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time,traceback
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'artificial_runtime')
sys.path[:0]=[str(P),str(R/'work/R74_EXACT_LOSS_FULL_TRAIN_001')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[P/'check_retained_graph.py',R/'work/R74_EXACT_LOSS_FULL_TRAIN_001/waveform_response.py']
hashes={str(p):sha(p)for p in paths};start=time.perf_counter();exit_code=0
try:
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    from check_retained_graph import run
    result=run(20261011723,require_cuda=False)
    assert all(sha(p)==hashes[str(p)]for p in paths)
    receipt=dict(pass_all=result['pass_all'],result=result,source_SHA256=hashes)
except BaseException as exc:
    exit_code=1;receipt=dict(pass_all=False,error=repr(exc),traceback=traceback.format_exc(),source_SHA256=hashes)
receipt.update(UTC=datetime.now(timezone.utc).isoformat(),native_exit=exit_code,elapsed_s=time.perf_counter()-start,guard={**guard,**memory()},scientific_arrays_decoded=0)
name='ROOT_RETAINED_GRAPH_ARTIFICIAL_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json'
(P/name).write_text(json.dumps(receipt,indent=2)+'\n','utf8')
if exit_code==0:(P/'ROOT_RETAINED_GRAPH_ARTIFICIAL.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps(receipt),flush=True);sys.exit(exit_code)
