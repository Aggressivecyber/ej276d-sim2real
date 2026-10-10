from pathlib import Path
from datetime import datetime, timezone
import ast,json,os
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'R44_PSD_VELOCITY_FIT_DIAGNOSTIC_001'
for name in ['supervise.py','owned_worker.py']:
    code=(OLD/name).read_text('utf8');ast.parse(code);(ROOT/name).write_text(code,'utf8')
events=[
 {'tool':'478b2e','native_exit':1,'PID':'unknown','kind':'source/control query',
  'error':'rg given a literal Windows wildcard filename; ERROR123',
  'scientific_compute_s':0,'response_updates_or_forwards':0,'resolved':'use explicit files or rg -g'},
 {'tool':'71c89b','native_exit':1,'PID':'unknown','kind':'source/control query',
  'error':'R43/local_supervise.py does not exist; actual known helper is R44/supervise.py',
  'scientific_compute_s':0,'response_updates_or_forwards':0,'resolved':'read/copy actual R44 supervisor'},
]
(ROOT/'ENGINEERING_EVENTS.json').write_text(json.dumps(events,indent=2)+'\n','utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_only':True,
                  'actual_PID_RSS_handshake_reused':True,'probe_or_science_execution':0,
                  'local_auxiliary_CPU':4,'local_auxiliary_RSS_bytes':4294967296}))
