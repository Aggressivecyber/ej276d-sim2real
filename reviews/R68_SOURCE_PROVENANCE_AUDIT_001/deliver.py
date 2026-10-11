"""Publish the completed source-only addendum, preserving the R68 result."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'delivery_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(R).as_posix()
def call(args):
 if args[0]=='git':args=['git','-c','core.longpaths=true',*args[1:]]
 t=subprocess.run(args,cwd=R,text=True,capture_output=True,encoding='utf8',errors='replace',timeout=60)
 assert t.returncode==0,(args,t.returncode,t.stdout[-2000:],t.stderr[-2000:])
 return t.stdout.strip()
final=R/'R68_PROVENANCE_DELIVERY_RECEIPT.json';assert not final.exists()
assert not call(['git','diff','--cached','--name-only'])
for row in read(P/'INPUT_HASH_RECEIPT.json')['inputs']:assert sha(R/row['path'])==row['sha256']
assert sha(P/'REPORT.md')==read(P/'STATIC_AUDIT_RECEIPT.json')['report_SHA256']
assert read(R/'jobs.json')['active_scientific_jobs']==0
repo=json.loads(call(['gh','repo','view','Aggressivecyber/ej276d-sim2real','--json','visibility,url']));assert repo['visibility']=='PRIVATE'
base=call(['git','rev-parse','HEAD'])
selected=[rel(f)for f in P.iterdir()if f.is_file()and f.suffix in {'.py','.md','.json'}]
selected+=['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','STATUS.md','PROJECT_PROGRESS.json','PROGRAM.json']
write(P/'DELIVERY_SELECTION.json',dict(base_commit=base,paths=selected,scope='source-only addendum; no scientific recomputation'))
selected.append(rel(P/'DELIVERY_SELECTION.json'))
call(['git','add','-f','--',*selected])
assert set(call(['git','diff','--cached','--name-only']).splitlines()).issubset(selected)
assert not call(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
call(['git','commit','-m','Trace fixed source charge scaling and preserve calibration uncertainty'])
commit=call(['git','rev-parse','HEAD']);call(['git','push','origin','main'])
remote=call(['git','ls-remote','origin','refs/heads/main']).split()[0];assert remote==commit
backup=Path('D:/GAN_backups/research/sim2real_two_papers_20261008_0305')/('r68_source_provenance_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
payload=set(selected)|{r['path']for r in read(P/'INPUT_HASH_RECEIPT.json')['inputs']}
payload|={rel(f)for f in (P/'before_metadata').iterdir()if f.is_file()}
size=sum((R/n).stat().st_size for n in payload);assert shutil.disk_usage('D:/').free>size+10*1024**2
backup.mkdir(parents=True);manifest=[]
for name in sorted(payload):
 src=R/name;dst=backup/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 h=sha(dst);assert h==sha(src)
 manifest.append(dict(path=name,bytes=dst.stat().st_size,sha256=h))
bundle=backup/'r68_provenance_incremental.bundle';call(['git','bundle','create',str(bundle),base+'..HEAD'])
write(backup/'BACKUP_SHA256_MANIFEST.json',manifest)
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),purpose='R68_static_source_provenance_addendum',private_repository=repo,
 base_commit=base,local_commit=commit,remote_main_SHA=remote,push_exit=0,backup_root=str(backup),copied_files=len(manifest),copied_bytes=size,
 all_copied_file_SHA256_verified=True,bundle_SHA256=sha(bundle),bundle_bytes=bundle.stat().st_size,
 original_R68_delivery='R68_DELIVERY_RECEIPT.json',original_scientific_result_unchanged=True,scientific_jobs_added=0,
 active_scientific_jobs=0,original_A_F_complete=False,R65_automatic_resume=False,resource=dict(guard,**memory()))
write(final,receipt);shutil.copy2(final,backup/final.name);assert sha(final)==sha(backup/final.name)
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
