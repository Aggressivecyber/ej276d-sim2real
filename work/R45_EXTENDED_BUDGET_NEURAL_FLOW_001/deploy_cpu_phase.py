from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,tarfile,time
ROOT=Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text('utf8'))
phase=sys.argv[1];assert phase in ['check','evaluation']
assert json.loads((ROOT/'train_extended.py.LOCAL_NATIVE_RECEIPT.json').read_text('utf8'))['SSH_native_exit']==0
start=time.monotonic();records=[]
def command(kind,args):
    p=subprocess.run(args,capture_output=True,text=True)
    (ROOT/('deploy_'+phase+'_'+kind+'.stdout.log')).write_text(p.stdout,'utf8')
    (ROOT/('deploy_'+phase+'_'+kind+'.stderr.log')).write_text(p.stderr,'utf8')
    records.append({'kind':kind,'argv':args,'native_exit':p.returncode})
    if p.returncode:raise RuntimeError((kind,p.returncode,p.stderr[-1000:]))
try:
    if phase=='check':
        archive=ROOT/'UPLOAD_CPU_CHECK.tar'
        with tarfile.open(archive,'w') as t:
            for name in ['check_train_solver.py','R42_TRAIN_SOLVER_CHECK_PRIVATE.npz']:
                t.add(ROOT/name,arcname=name,recursive=False)
    else:
        assert json.loads((ROOT/'TRAIN_SOLVER_CHECK.json').read_text('utf8'))['pass']
        archive=ROOT/'UPLOAD_CPU_EVALUATION.tar'
        assert archive.is_file()
    command('upload',['scp','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8',
            str(archive),'rylan@100.82.231.102:'+cfg['remote_root']+'/'+archive.name])
    command('extract',['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102',
            'cd '+cfg['remote_root']+' && tar -xf '+archive.name+' --no-same-owner --no-same-permissions'])
finally:
    rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'phase':phase,'commands':records,
         'elapsed_s':time.monotonic()-start,'migration_content_consistency_test':False,
         'no_scientific_execution':True,'normal_transfer_and_extract_only':True}
    (ROOT/('DEPLOY_CPU_'+phase.upper()+'_NATIVE_RECEIPT.json')).write_text(json.dumps(rec,indent=2)+'\n','utf8')
    print(json.dumps(rec))
