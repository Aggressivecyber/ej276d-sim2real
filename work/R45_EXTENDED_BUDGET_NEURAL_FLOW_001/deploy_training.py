from pathlib import Path
from datetime import datetime,timezone
import subprocess,os,json
R=Path(__file__).resolve().parent;cfg=json.loads((R/'config.json').read_text())
ssh=['ssh','-T','-o','ForwardX11=no','-o','BatchMode=yes','-o','ConnectTimeout=8','rylan@100.82.231.102']
records=[]
def step(name,args,input=None):
    p=subprocess.run(args,input=input,text=True,capture_output=True)
    (R/(name+'.stdout.log')).write_text(p.stdout);(R/(name+'.stderr.log')).write_text(p.stderr)
    records.append({'name':name,'argv':args,'native_exit':p.returncode,'UTC':datetime.now(timezone.utc).isoformat()})
    (R/'DEPLOY_NATIVE_RECEIPT.json').write_text(json.dumps({'PID':os.getpid(),'steps':records,'migration_content_tests':False},indent=2)+'\n')
    if p.returncode:raise SystemExit(p.returncode)
    print(name,p.stdout.strip(),flush=True)
script="from pathlib import Path\nimport os,json\np=Path("+repr(cfg['remote_root'])+");p.mkdir(exist_ok=False)\nprint(json.dumps({'PID':os.getpid(),'new_root':str(p)}))\n"
step('deploy_create',ssh+[cfg['server_python']+' -B -'],script)
step('deploy_upload',['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8',str(R/'UPLOAD_TRAIN.tar'),'rylan@100.82.231.102:'+cfg['remote_root']+'/UPLOAD_TRAIN.tar'])
script="import os,tarfile,json\nos.chdir("+repr(cfg['remote_root'])+")\nwith tarfile.open('UPLOAD_TRAIN.tar','r:') as a:a.extractall('.',filter='data')\nprint(json.dumps({'PID':os.getpid(),'cwd':os.getcwd(),'entry_exists':os.path.isfile('train_extended.py')}))\n"
step('deploy_extract',ssh+[cfg['server_python']+' -B -'],script)
print(json.dumps({'PID':os.getpid(),'normal_transfers_native0':True,'test_or_scientific_computation':0}),flush=True)
