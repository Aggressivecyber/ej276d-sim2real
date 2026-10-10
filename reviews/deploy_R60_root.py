from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,subprocess
R=Path(__file__).resolve().parents[1];S=R/'work/R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001';C=json.loads((S/'config.json').read_text('utf8'))
remote=C['remote_root'];host='rylan@100.82.231.102'
assert remote.startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R60_ROOT_')
files=['config.json','INPUT_BINDING.json','PROTOCOL.md','REPRODUCTION.md','ARTIFICIAL_CHECK_LOCAL.json',
 'diagnose_train.py','diagnostic_math.py','full_population.py','render_diagnostic.py','artificial_check.py',
 'network.py','noise_core.py','source_flow_core.py','distribution_loss.py','linux_supervise.py','linux_owned_worker.py']
assert C['response_training_runs']==C['response_updates']==C['HGB_fits']==C['density_fits']==C['DEV_read']==C['EVAL_read']==C['CAL_decode']==0
assert json.loads((S/'ARTIFICIAL_CHECK_LOCAL.json').read_text('utf8'))['pass']
assert all(C[k]is None for k in ['science_budget','deadline_UTC','output_limit_bytes'])
binding=json.loads((S/'INPUT_BINDING.json').read_text('utf8'))
for name,v in binding['frozen_R59_helpers'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==v['sha256']
for name in files:
 assert(S/name).is_file()
 if name.endswith('.py'):ast.parse((S/name).read_text('utf8'),filename=name)
assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='step'for n in ast.walk(ast.parse((S/'diagnose_train.py').read_text('utf8'))))
review={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','AST_pass':True,'frozen_R59_helpers_identical':True,'artificial_checks_pass':True,
 'reviewed_definitions':['all6593/21047 original TRAIN parents and fixed single noise','original Heun128 output and original analytic training endpoint','17 objective gradient contributions with original denominator and row weights','float32 predecessor/successor rounding cells with ratio envelope','exact unequal empirical W1','fixed32 batches and all 17 parameter gradients','no optimizer step or validation decode','post-projection charge bounds and input hashes','undefined cosine retained and preset24/32 marker threshold','original A-F unchanged and native process/resource ownership'],
 'scientific_execution_started':False,'new_response_training':0,'ready_for_registered_diagnostic':True,
 'files':[{'name':n,'bytes':(S/n).stat().st_size,'sha256':hashlib.sha256((S/n).read_bytes()).hexdigest()}for n in files]}
(S/'ROOT_IMPLEMENTATION_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n','utf8');files.append('ROOT_IMPLEMENTATION_REVIEW.json')
events=[]
def invoke(argv,code=None):
 p=subprocess.run(argv,input=code,text=True,capture_output=True,timeout=60)
 events.append({'argv':argv,'native_exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 (S/'DEPLOYMENT_RECEIPT.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'control_PID':os.getpid(),'events':events},indent=2)+'\n','utf8')
 assert p.returncode==0,events[-1]
 return p.stdout
ssh=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3 -B -']
prefix='from pathlib import Path\nimport json,subprocess,time,os\np=Path('+repr(remote)+')\n'
invoke(ssh,prefix+'inputs='+repr(C['input_paths'])+'\nassert all(Path(v).is_file()for v in inputs.values())\nassert p.parent.is_dir()\np.mkdir(exist_ok=False)\nprint(str(p))\n')
invoke(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',*[str(S/n)for n in files],host+':'+remote+'/'])
code=prefix+'python='+repr(C['server_python'])+'\n'+'''
marker=p/'LAUNCH_ONCE.json'
with marker.open('x')as f:json.dump({'launcher_PID':os.getpid(),'launch_time_unix':time.time()},f)
out=(p/'supervisor.stdout.log').open('wb');err=(p/'supervisor.stderr.log').open('wb')
child=subprocess.Popen([python,'-B',str(p/'linux_supervise.py'),'diagnose_train.py'],cwd=p,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
record={'supervisor_PID':child.pid,'launch_time_unix':time.time(),'stage':str(p)}
marker.write_text(json.dumps(record,indent=2)+'\\n')
time.sleep(1.5)
record['actual_job']=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())if(p/'REMOTE_ACTIVE_JOB.json').exists()else None
record['immediate_exit']=child.poll()
print(json.dumps(record,indent=2))
'''
print(invoke(ssh,code))
