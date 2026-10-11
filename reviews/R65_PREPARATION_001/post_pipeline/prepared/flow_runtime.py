from __future__ import annotations
from pathlib import Path
import numpy as np,torch
from source_flow_core import FlowNetwork,solve
from noise_core import innovation

class Pipeline:
    def __init__(self,cfg,root,arm):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        self.network=FlowNetwork();self.network.load_state_dict(torch.load(cfg['endpoint_weight_paths'][arm],map_location='cpu',weights_only=True))
        self.network.to(cfg['inference_device']);self.network.eval();self.network.requires_grad_(False)
        with np.load(cfg['constants_path'],allow_pickle=False) as z:
            self.mu=z['mu_unit_q'].copy();self.scale=z['scale_unit_q'].copy();self.edges=z['edges_logQ'].copy()
        with np.load(Path(root)/'NOISE_CALIBRATION.npz',allow_pickle=False) as z:self.noise={k:z[k].copy()for k in z.files}
        assert arm in cfg['endpoints'];self.completion=True
        self.steps=cfg['solver_steps'];assert self.steps==128
        self.details=None;self.total_field_rows=0;self.total_network_rows=0;self.total_network_calls=0;self.last_receipt=None

def infer(model,source,z,white,style,chunk=64):
    pre,detail,rec=solve(model.network,source,white,model.mu,model.scale,model.edges,model.noise,model.completion,model.steps,chunk)
    model.details=detail;model.last_receipt=rec;model.total_field_rows+=rec['field_forward_rows'];model.total_network_rows+=rec['actual_network_rows'];model.total_network_calls+=rec['actual_network_calls']
    n,_=innovation(source,white,model.mu,model.edges,model.noise['p480'],model.noise['kappa2'],float(model.noise['tau2']),model.completion)
    return pre,np.asarray(source,dtype='f4'),n
