from __future__ import annotations
import pathlib,json,datetime,os,ast,hashlib
ROOT=pathlib.Path(__file__).resolve().parent
sources=[]
for p in sorted(ROOT.glob('*.py')):
    b=p.read_bytes();ast.parse(b.decode('utf8'));sources.append({'file':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
r={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'AST_pass':True,'sources':sources,
   'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
   'science_started':False,'migration_test':False,'SSH':0,'entry':'local_supervise.py','location':'local CPU4','prior_reference_readonly':True}
(ROOT/'READINESS.json').write_text(json.dumps(r,indent=2)+'\n','utf8');print(json.dumps({'PID':os.getpid(),'UTC':r['UTC'],'AST_sources':len(sources),'config_SHA':r['config_sha256'],'local_entry':'local_supervise.py'}),flush=True)
