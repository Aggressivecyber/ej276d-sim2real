"""Control-only recovery of an already transferred tar; no transfer retry or science."""
from pathlib import Path,PurePosixPath
import json,tarfile,os,datetime
ROOT=Path(__file__).resolve().parent
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
failure={'tool':'1e27c0','native_exit_code':1,'original_entry':'retrieve.py','cause':'Archive contains pre-uploaded INPUT_BINDINGS.json, rejected overwrite before extraction','normal_SCP_exit_code':0,'transfer_completed':True,'scientific_rerun':0,'transfer_retry':0,'original_source_kept':True,'UTC':utc()}
(ROOT/'FAILURE_RETRIEVAL_CONTROL_001.json').write_text(json.dumps(failure,indent=2)+'\n')
with tarfile.open(ROOT/'REMOTE_RETURN.tar') as t:
    members=t.getmembers();selected=[];skipped=[]
    for m in members:
        q=PurePosixPath(m.name)
        if not m.isfile() or '..' in q.parts or q.is_absolute():raise RuntimeError('unsafe return member')
        if m.name=='INPUT_BINDINGS.json':
            skipped.append(m.name);continue
        if (ROOT/m.name).exists():raise RuntimeError('Unexpected collision: '+m.name)
        selected.append(m)
    t.extractall(ROOT,members=selected,filter='data')
receipt=json.loads((ROOT/'RETURN_TRANSFER_RECEIPT.json').read_text())
receipt.update(control_recovery_PID=os.getpid(),stop_UTC=utc(),return_tar_bytes=(ROOT/'REMOTE_RETURN.tar').stat().st_size,returned_file_count=len(members),extracted_file_count=len(selected),skipped_preuploaded_controls=skipped,migration_content_validation=False,scientific_rerun=0,transfer_retry=0)
(ROOT/'RETURN_TRANSFER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
