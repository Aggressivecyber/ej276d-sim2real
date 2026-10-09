import torch
DEVICE=torch.device('cuda:0')
LIMIT=4*1024**3
def configure(seed):
    if not torch.cuda.is_available():raise RuntimeError('R10 requires existing cuda:0 environment')
    torch.cuda.set_device(0)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.cuda.reset_peak_memory_stats()
def transfer(value):
    if isinstance(value,torch.Tensor):return value.to(DEVICE)
    if isinstance(value,dict):return {k:transfer(v) for k,v in value.items()}
    if isinstance(value,list):return [transfer(v) for v in value]
    if isinstance(value,tuple):return tuple(transfer(v) for v in value)
    return value
def guard():
    n=torch.cuda.max_memory_allocated()
    if n>LIMIT:raise RuntimeError('R10 GPU allocated limit exceeded: '+str(n))
    return n
def facts():
    return {'device':str(DEVICE),'name':torch.cuda.get_device_name(0),'Torch':torch.__version__,'CUDA':torch.version.cuda,'peak_allocated_bytes':guard(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'allocated_limit_bytes':LIMIT,'TF32':False,'cudnn_deterministic':True}
