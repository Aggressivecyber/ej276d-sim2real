"""One control-only byte seal; manifest and final receipt explicitly exclude self-reference."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,time
ROOT=Path(__file__).resolve().parent;CAP=512*1024**2
END=datetime.fromisoformat('2026-10-09T22:37:06.331729+00:00')
def now():return datetime.now(timezone.utc)
def encode(v):return(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
def digest(p):
    h=hashlib.sha256();n=0
    with p.open('rb')as f:
        while chunk:=f.read(1024**2):
            assert now()<END
            h.update(chunk);n+=len(chunk)
    return{'bytes':n,'sha256':h.hexdigest()}
start=time.monotonic();assert now()<END
excluded=['FINAL_MANIFEST.json','FINAL_RECEIPT.json']
assert all(not(ROOT/n).exists()for n in excluded)
resource=json.loads((ROOT/'RESOURCE_RECEIPT.json').read_text('utf8'))
qa=json.loads((ROOT/'FIGURE_QA.json').read_text('utf8'))
assert qa['pass']and qa['all_final_figures_actually_opened']and qa['figure_count']==11
assert resource['science_elapsed_s_including_plots_display_repair']<1800
files=sorted(p for p in ROOT.rglob('*')if p.is_file()and p.relative_to(ROOT).as_posix()not in excluded)
members={p.relative_to(ROOT).as_posix():digest(p)for p in files}
memberbytes=sum(v['bytes']for v in members.values())
manifest={'stage':'R38_SLICED_MONOTONE_TRANSPORT_001','seal_PID':os.getpid(),'UTC':now().isoformat(),
    'member_count':len(members),'member_bytes':memberbytes,'members':members,'excluded_self_references':excluded,
    'hash_is_pure_project_byte_seal':'no scientific decoding, model execution, fits or migration tests',
    'science_terminal':True,'scientific_decision':'close_this_fixed_sliced_monotone_recipe'}
mb=encode(manifest);assert memberbytes+len(mb)+4096<CAP
with(ROOT/'FINAL_MANIFEST.json').open('xb')as f:f.write(mb)
receipt={'stage':manifest['stage'],'status':'sealed','seal_PID':os.getpid(),'receipt_write_start_UTC':now().isoformat(),
    'native_exit_evidence':'external exec tool result after this process exits; no future return code inferred',
    'final_manifest':{'bytes':len(mb),'sha256':hashlib.sha256(mb).hexdigest(),'members':len(members)},
    'REPORT':members['REPORT.md'],'RESOURCE':members['RESOURCE_RECEIPT.json'],'NATIVE_CHAIN':members['NATIVE_CHAIN.json'],
    'science_elapsed_s':resource['science_elapsed_s_including_plots_display_repair'],'science_process_native_exits':resource['science_native_exits'],
    'counts':{'HGB':10,'density_slots':8,'EM':7,'direction_updates':5120,'scalar_maps':4096,'complete_CPU_replay_rows':2000},
    'gates':[True,False,False,False,True],'decision':manifest['scientific_decision'],
    'final_total_files':len(members)+2,'final_total_bytes':0,'output_cap_bytes':CAP,
    'byte_hash_cost_excluding_receipt_self_digest':memberbytes+len(mb),
    'seal_elapsed_before_final_write_s':time.monotonic()-start,'self_reference_exclusions':excluded,
    'no_further_stage_writes':True,'project_goal_closed':False}
for _ in range(8):
    rb=encode(receipt);total=memberbytes+len(mb)+len(rb)
    if receipt['final_total_bytes']==total:break
    receipt['final_total_bytes']=total
else:raise RuntimeError('final byte count did not stabilize')
assert now()<END and total<CAP
with(ROOT/'FINAL_RECEIPT.json').open('xb')as f:f.write(rb)
stat=(ROOT/'FINAL_RECEIPT.json').stat()
lastwrite=datetime.fromtimestamp(stat.st_mtime_ns/1e9,timezone.utc).isoformat()
print(json.dumps({'seal_PID':os.getpid(),'actual_last_write_UTC':lastwrite,'last_file':'FINAL_RECEIPT.json',
    'FINAL_MANIFEST_sha256':hashlib.sha256(mb).hexdigest(),'FINAL_RECEIPT_sha256':hashlib.sha256(rb).hexdigest(),
    'REPORT_sha256':members['REPORT.md']['sha256'],'RESOURCE_sha256':members['RESOURCE_RECEIPT.json']['sha256'],
    'member_count':len(members),'total_files':len(members)+2,'total_bytes':total,'seal_elapsed_s':time.monotonic()-start,
    'status':'sealed_no_further_stage_writes','native_exit_evidence':'actual outer tool exit'},ensure_ascii=False),flush=True)
