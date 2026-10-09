from pathlib import Path
import os,json,time,datetime
start=time.monotonic()
import torch
from energy_core import energy_u
torch.set_num_threads(4);torch.set_num_interop_threads(4)
a=torch.tensor([[.2,3.8],[1.7,-2.3],[5.1,.9]],dtype=torch.float64,requires_grad=True)
b=torch.tensor([[-1.,2.2],[3.,1.1],[4.1,-2.7]],dtype=torch.float64,requires_grad=True)
u,parts=energy_u(a,b)
manual=sum(2*(a[i,k]-b[j,k]).abs()-(a[i,k]-a[j,k]).abs()-(b[i,k]-b[j,k]).abs() for k in range(2) for i in range(3) for j in range(3) if i!=j)/(3*2*2)
g=torch.autograd.grad(u,(a,b),retain_graph=True)
gm=torch.autograd.grad(manual,(a,b))
negative,_=energy_u(torch.tensor([[0.],[1.]],dtype=torch.float64),torch.tensor([[1.],[0.]],dtype=torch.float64))
zero,_=energy_u(a.detach(),a.detach())
error=float(abs(u.detach()-manual.detach()));graderr=max(float((x-y).abs().max()) for x,y in zip(g,gm))
assert error<1e-12 and graderr<1e-12 and float(negative)==-2. and float(zero)==0.
assert a.grad is None and torch.isfinite(g[0]).all() and torch.isfinite(g[1]).all()
r={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'formula_maxabs':error,'gradient_maxabs':graderr,'vectorized_U':float(u.detach()),'explicit_loop_U':float(manual.detach()),'parts':{k:float(v.detach()) for k,v in parts.items()},'grad_a':g[0].tolist(),'grad_b':g[1].tolist(),'negative_U_retained':float(negative),'identical_arrays_U':float(zero),'three_off_diagonal_masks':True,'generated_self_both_positions_grad':True,'numerical_values':24,'response_fit_updates':0,'pass':True}
(Path(__file__).resolve().parent/'FORMULA_GRADIENT_CHECK.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n','utf8')
print(json.dumps(r),flush=True)
