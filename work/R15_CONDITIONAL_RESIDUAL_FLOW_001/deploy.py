from __future__ import annotations
import pathlib,json,ast,hashlib,datetime,os,tarfile,subprocess
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text());sources=list(ROOT.glob('*.py'))
for f in sources:ast.parse(f.read_text())
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'AST_pass':True,'source_bindings':[{'file':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sources],'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),'numeric_started':False,'migration_tests':False,'inputs_reused_readonly_without_transfer':[cfg['remote_R14_root'],cfg['remote_input_root']]}
(ROOT/'READINESS.json').write_text(json.dumps(receipt,indent=2)+'\n')
with tarfile.open(ROOT/'UPLOAD.tar','w') as t:
    for f in sources+[ROOT/'config.json',ROOT/'PROTOCOL.md',ROOT/'READINESS.json',ROOT/'START_RECEIPT.json',ROOT/'INPUT_BINDINGS.json']:t.add(f,arcname=f.name,recursive=False)
host=cfg['execution']['location'];remote=cfg['remote_root'];opts=['-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8'];records=[]
def cmd(argv):
    p=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:out,err=p.communicate(timeout=120)
    except subprocess.TimeoutExpired:p.kill();out,err=p.communicate();raise
    r={'argv':argv,'PID':p.pid,'exit_code':p.returncode,'stdout':out.decode(errors='replace'),'stderr':err.decode(errors='replace'),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()};records.append(r);(ROOT/'TRANSFER_RECEIPT.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(r),flush=True)
    if p.returncode:raise RuntimeError('normal source-only deployment failure')
cmd(['ssh','-T',*opts,host,'test ! -e '+remote+' && mkdir -p '+remote])
cmd(['scp',*opts,str(ROOT/'UPLOAD.tar'),host+':'+remote+'/UPLOAD.tar'])
cmd(['ssh','-T',*opts,host,'cd '+remote+' && tar -xf UPLOAD.tar && test -f numeric_worker.py && test -f residual_flow.py && test -f '+cfg['remote_R14_root']+'/TRAIN_ADDITION.npz && test -f '+cfg['remote_R14_root']+'/FLOW_EXPOSURES.pt && test -f '+cfg['remote_R14_root']+'/TAIL_EXPOSURES.pt'])
print(json.dumps({'prepared_remote_root':remote,'PID':os.getpid(),'source_only_upload_bytes':(ROOT/'UPLOAD.tar').stat().st_size,'migration_SHA_checks':False}),flush=True)
