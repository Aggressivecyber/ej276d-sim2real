from pathlib import Path
import ast,json,os
from datetime import datetime,timezone
R=Path(__file__).resolve().parent
for source,target in [('linux_supervise.py','linux_supervise_post.py'),('launch_server.py','launch_post.py')]:
    code=(R/source).read_text('utf8').replace('\r','')
    code=code.replace("'evaluate_saved.py','replay_all.py']","'evaluate_saved.py','replay_all.py','summarize_and_plot.py','render_train.py']")
    if target=='launch_post.py':code=code.replace('-B linux_supervise.py','-B linux_supervise_post.py')
    ast.parse(code);(R/target).write_bytes(code.encode())
(R/'POST_GUARD_BINDING.json').write_bytes((json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
 'changes':'add registered aggregate/plot entries; original strict resource admission/PID/RSS guard preserved',
 'original_guard_replaced':False,'no_scientific_array_read':True,'scientific_calls':0},indent=2)+'\n').encode())
print(json.dumps({'PID':os.getpid(),'source_prepared':True,'science':0}),flush=True)
