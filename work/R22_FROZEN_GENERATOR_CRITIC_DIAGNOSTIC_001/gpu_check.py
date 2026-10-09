import os, json, datetime
import torch, numpy as np
torch.set_num_threads(4)
d={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':__import__('sys').executable,'torch':torch.__version__,'numpy':np.__version__,'CUDA_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda}
if d['CUDA_available']:
    d.update(device=torch.cuda.get_device_name(0),free_total_bytes=list(torch.cuda.mem_get_info(0)),allocated_bytes=torch.cuda.memory_allocated(0))
from pathlib import Path
(Path(__file__).parent/'GPU_ENVIRONMENT.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
print(json.dumps(d),flush=True)
