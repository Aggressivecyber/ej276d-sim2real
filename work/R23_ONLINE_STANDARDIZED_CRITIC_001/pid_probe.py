import os,time,json
from pathlib import Path
v=bytearray(64*1024*1024)
for i in range(0,len(v),4096):v[i]=1
time.sleep(.5)
(Path(__file__).parent/'PID_PROBE_RESULT.json').write_text(json.dumps({'PID':os.getpid(),'held_bytes':len(v),'trained_or_generated_science':0,'purpose':'actual worker RSS binding before scientific imports/training'}),encoding='utf8')
