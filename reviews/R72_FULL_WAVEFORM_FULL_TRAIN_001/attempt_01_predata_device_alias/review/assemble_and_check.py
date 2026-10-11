"""Assemble reviewed modules and check the complete artificial CPU VJP."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil,sys,time,traceback
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'joint_artificial_runtime');start=time.monotonic()
assert not(P/'REGISTRATION.json').exists()
selection={'response_candidate':['waveform_response.py','waveform_numpy_reference.py','check_waveform_response.py'],
           'loss_candidate':['full_waveform_loss.py','check_full_waveform_loss.py']}
for folder,names in selection.items():
    for name in names:
        src=P/folder/name;assert src.is_file(),src
        shutil.copy2(src,S/name)
for path in S.glob('*.py'):ast.parse(path.read_text('utf8'));compile(path.read_text('utf8'),str(path),'exec')
sys.path.insert(0,str(S))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
attempt=1
while(P/('ROOT_JOINT_ARTIFICIAL_ATTEMPT_'+str(attempt)+'.json')).exists():attempt+=1
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),attempt=attempt,scientific_arrays_read=False,scientific_updates=0,
  source_SHA256={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest()for f in S.glob('*.py')})
try:
    from artificial_checks import joint_check
    receipt.update(result=joint_check('cpu',20261011723),native_exit=0,pass_all=True)
except BaseException as exc:
    receipt.update(error=repr(exc),traceback=traceback.format_exc(),native_exit=1,pass_all=False)
finally:
    receipt.update(elapsed_s=time.monotonic()-start,guard={**guard,**memory()})
    (P/('ROOT_JOINT_ARTIFICIAL_ATTEMPT_'+str(attempt)+'.json')).write_text(json.dumps(receipt,indent=2)+'\n','utf8')
    print(json.dumps({k:v for k,v in receipt.items()if k!='source_SHA256'},indent=2),flush=True)
if not receipt['pass_all']:raise SystemExit(1)
(P/'ROOT_JOINT_ARTIFICIAL.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
