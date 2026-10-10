from __future__ import annotations
import numpy as np,torch
from source_flow_core import FlowNetwork,solve
class Pipeline:
    def __init__(self,cfg,root,arm):
        self.network=FlowNetwork();self.network.load_state_dict(torch.load(cfg['endpoint_weight_paths'][arm],map_location='cpu',weights_only=True));self.network.eval();self.network.requires_grad_(False)
        with np.load(cfg['constants_path'],allow_pickle=False)as z:
            self.mu=z['mu_unit_q'].copy();self.scale=z['scale_unit_q'].copy();self.edges=z['edges_logQ'].copy()
        self.steps=cfg['solver_steps'];assert self.steps==64;self.details=None;self.arm=arm;self.total_field_rows=0;self.last_receipt=None
def infer(model,source,z,white,style,chunk=64):
    pre,detail,rec=solve(model.network,source,white,model.mu,model.scale,model.edges,model.steps,chunk)
    model.details=detail;model.last_receipt=rec;model.total_field_rows+=rec['field_forward_rows']
    return pre,detail['base_signal_V'],detail['base_electronic_V']
