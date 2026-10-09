import os,json,time,resource
from pathlib import Path
from datetime import datetime,timezone
payload=bytearray(4*1024**2);payload[::4096]=bytes([1])*len(payload[::4096]);time.sleep(.2)
import torch,numpy
root=Path(__file__).resolve().parent
(root/'PID_PROBE_RESULT.json').write_text(json.dumps({'PID':os.getpid(),'parent_PID':os.getppid(),
    'UTC':datetime.now(timezone.utc).isoformat(),'cwd':str(root),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    'response_training':0,'response_generation':0,'scientific_dependency_imports':2,'touched_owned_bytes':len(payload),
    'torch':torch.__version__,'numpy':numpy.__version__,'CUDA_available':torch.cuda.is_available(),
    'GPU':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'GPU_allocated_bytes':torch.cuda.memory_allocated() if torch.cuda.is_available() else 0},indent=2)+'\n','utf8')
