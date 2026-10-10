from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,subprocess
R=Path(__file__).resolve().parents[1];S=R/'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001';C=json.loads((S/'config.json').read_text('utf8'))
remote=C['remote_root'];host='rylan@100.82.231.102'
assert remote.startswith('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R62_ROOT_')
files=['config.json','INPUT_BINDING.json','PROTOCOL.md','REPRODUCTION.md','ARTIFICIAL_CHECK_LOCAL.json',
 'diagnose_train.py','residual_math.py','traced_response.py','residual_analysis.py','render_diagnostic.py','artificial_check.py','precision_order.py',
 'network.py','noise_core.py','source_flow_core.py','distribution_loss.py','linux_supervise.py','linux_owned_worker.py']
assert (S/'PROTOCOL.md').read_bytes()==(R/'tasks/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001.md').read_bytes()
assert C['response_training_runs']==C['response_updates']==C['HGB_fits']==C['density_fits']==C['DEV_read']==C['EVAL_read']==C['CAL_decode']==0
assert json.loads((S/'ARTIFICIAL_CHECK_LOCAL.json').read_text('utf8'))['pass']
assert all(C[k]is None for k in ['science_budget','deadline_UTC','output_limit_bytes'])
binding=json.loads((S/'INPUT_BINDING.json').read_text('utf8'))
for record in binding['original_acceptance']:assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256']
for name,v in binding['frozen_R61_helpers'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==v['sha256']
for name in files:
 assert(S/name).is_file()
 if name.endswith('.py'):ast.parse((S/name).read_text('utf8'),filename=name)
assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='step'for n in ast.walk(ast.parse((S/'diagnose_train.py').read_text('utf8'))))
review={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','AST_pass':True,'frozen_R61_helpers_identical':True,'artificial_checks_pass':True,
 'reviewed_definitions':['all6593/21047 original TRAIN parents and fixed single noise','bitwise traced original Heun128 output on artificial CPU and CUDA arrays','all480 waveform positions and nine fixed physical boundaries','five unchanged path representations and source-weighted full-window objective','exact unequal empirical W1 and fixed64 directions','fixed32 saved-endpoint batches without new neural calls','100x100 baseline covariance, ACF1-20 and correct Nyquist spectral weights','32 empirical REAL resamples matched to original SOURCE N in11 scopes','no optimizer, gradient, teacher, HGB, density fit or validation decode','source IDs, input/weight hashes and three R60 stream hashes preserved','original A-F unchanged and actual native process/resource ownership'],
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
