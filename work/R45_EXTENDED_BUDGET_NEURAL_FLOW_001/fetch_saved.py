from pathlib import Path
from datetime import datetime,timezone
import base64,json,os,subprocess,sys,tarfile,time
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'))
phase=sys.argv[1];assert phase in ['train','check','evaluation','replay']
start=time.monotonic();commands=[]
remote_code="""
from pathlib import Path
import sys,tarfile,json
R=Path.cwd();phase=sys.argv[1]
if phase=='train':
    names=['EXTENDED_FLOW_FINAL','checkpoints','probes','FIXED_TRAIN_PROBE_PRIVATE.npz','FIXED_TRAIN_PROBE_BINDING.json',
           'FIXED_PROBE_TRAJECTORY.json','INHERITANCE_RECEIPT.json','MODEL_STRUCTURE.json','TRAINING_NUMERIC_RECEIPT.json',
           'UNIT_CHARGE_CONSTANTS.npz','UNIT_CHARGE_CONSTANTS.json','TRAINING_INPUT_BINDING.json','ORIGINAL_STREAM_BINDING.json',
           'FORMULA_CHECK.json','TRAIN_PROGRESS.json','SERVER_ENVIRONMENT.json','TRAINING_FAILURE.json',
           'INITIAL_TORCH_STATE.pt','INITIAL_CUDA_STATE.pt']
elif phase=='check':names=['TRAIN_SOLVER_CHECK.json','TRAIN_SOLVER_CHECK_PRIVATE.npz']
elif phase=='evaluation':
    names=['EXTENDED_FLOW_FINAL/generated.npz','EXTENDED_FLOW_REAL_CONTROL','results','metrics.json',
           'GENERATION_RECEIPT.json','GENERATION_NUMERIC_RECEIPT.json','REAL_CONTROL_INPUT_BINDING.json','SOURCE_PROBES_SUMMARY.json',
           'JOINT8_SUMMARY.json','GUIDE_PERTURBATION_SUMMARY.json','GUIDE_PERTURBATION_COMPLETE_SUMMARY.json',
           'RAW_RR_REUSE_BINDING.json','SAME_TREATED_REAL_CONTROL_PAIRED.json','MEAN_SD_METRICS.json',
           'BOUNDARY_METRICS.json','DEV_COVARIANCE_DESCRIPTION.json','EVALUATION_NUMERIC_RECEIPT.json','C2ST_RUNNING.json']
else:names=['REPLAY_RECEIPT.json','REPLAY_PROGRESS.json']
entries={'train':['train_extended.py'],'check':['check_train_solver.py'],
         'evaluation':['generate_all.py','evaluate_saved.py'],'replay':['replay_all.py']}[phase]
names+=['REMOTE_RUN_HISTORY.json','REMOTE_ACTIVE_JOB.json','SERVER_START_RESOURCE.json']
for entry in entries:
    names += [entry+s for s in ['.NATIVE_RECEIPT.json','.SCIENCE_PID.json','.RSS_ACK.json','.WORKER_RECEIPT.json','.stdout.log','.stderr.log']]
path=R/('RETURN_'+phase.upper()+'.tar')
with tarfile.open(path,'w') as t:
    for name in names:
        p=R/name
        if p.exists():t.add(p,arcname=name)
    if phase=='train':t.add(R/'config.json',arcname='EXECUTED_TRAIN_CONFIG.json')
print(json.dumps({'phase':phase,'archive':str(path),'bytes':path.stat().st_size,'normal_saved_artifact_transfer':True}))
"""
def command(kind,args):
    p=subprocess.run(args,capture_output=True,text=True)
    (ROOT/('fetch_'+phase+'_'+kind+'.stdout.log')).write_text(p.stdout,'utf8')
    (ROOT/('fetch_'+phase+'_'+kind+'.stderr.log')).write_text(p.stderr,'utf8')
    commands.append({'kind':kind,'native_exit':p.returncode,'argv':args})
    if p.returncode:raise RuntimeError((kind,p.returncode,p.stderr[-1000:]))
    return p.stdout
try:
    encoded=base64.b64encode(remote_code.encode()).decode()
    expression="import base64,sys;sys.argv=['fetch',"+repr(phase)+"];exec(base64.b64decode('"+encoded+"'))"
    command('archive',['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102',
             'cd '+cfg['remote_root']+' && '+cfg['server_python']+' -B -c '+"'"+expression.replace("'","'\"'\"'")+"'"])
    archive=ROOT/('RETURN_'+phase.upper()+'.tar')
    command('download',['scp','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',
             'rylan@100.82.231.102:'+cfg['remote_root']+'/'+archive.name,str(archive)])
    with tarfile.open(archive,'r') as t:t.extractall(ROOT,filter='data')
finally:
    rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'phase':phase,'commands':commands,
         'elapsed_s':time.monotonic()-start,'migration_content_consistency_test':False,
         'scientific_calculation':0,'normal_saved_artifact_copy_only':True}
    (ROOT/('FETCH_'+phase.upper()+'_NATIVE_RECEIPT.json')).write_text(json.dumps(rec,indent=2)+'\n','utf8')
    print(json.dumps(rec))
