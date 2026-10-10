from pathlib import Path
import os,json,shutil,datetime,sys,importlib.metadata as md
BASE='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305'
REMOTE=BASE+'/R49_REMOTE_20261010_093411_UTC'
jobs=[]
for d in Path('/proc').iterdir():
    if not d.name.isdigit()or int(d.name)==os.getpid():continue
    try:
        argv=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf8','replace');cwd=str((d/'cwd').resolve())
        if 'python'in argv and(BASE in argv or cwd.startswith(BASE)):jobs.append({'PID':int(d.name),'argv':argv,'cwd':cwd})
    except(OSError,PermissionError):pass
mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),
 'python':sys.version,'python_path':sys.executable,'versions':{p:md.version(p)for p in ['numpy','torch','scipy','scikit-learn','matplotlib']},
 'CPU_count':os.cpu_count(),'MemAvailable_bytes':mem['MemAvailable'],'disk_free_bytes':shutil.disk_usage(BASE).free,
 'remote_root':REMOTE,'remote_leaf_exists':Path(REMOTE).exists(),'own_project_jobs':jobs,
 'scientific_execution_device':'cpu','GPU_admission_applied':False,'scientific_imports':False}))
