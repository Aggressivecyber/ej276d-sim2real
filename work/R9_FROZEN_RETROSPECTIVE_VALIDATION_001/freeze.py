from __future__ import annotations
import pathlib,json,datetime,os,ast,hashlib
ROOT=pathlib.Path(__file__).resolve().parent
cfg=json.loads((ROOT/'config.json').read_text());sources=[]
for p in sorted(ROOT.glob('*.py')):
    b=p.read_bytes();ast.parse(b);sources.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
(ROOT/'READINESS.json').write_text(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),'frozen_SHA256':cfg['frozen_SHA256'],
 'execution_sources':sources,'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),'EVAL_scientific_values_this_stage_before_record':0,
 'frozen_bindings_checked_in':'FROZEN_BINDING_CHECK.json','response_updates':0,'SSH':0,'new_G4':0,'retrospective_previously_viewed_same_acquisition':True},indent=2)+'\n')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'AST_sources':len(sources),'EVAL_values':0,'readiness_sha256':hashlib.sha256((ROOT/'READINESS.json').read_bytes()).hexdigest()}))
