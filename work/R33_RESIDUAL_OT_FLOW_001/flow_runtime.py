from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from model import Translator,infer as base_infer
from flow_core import Velocity,correct

class Pipeline:
    def __init__(self,cfg,root,arm):
        self.base=Translator();self.base.load_state_dict(torch.load(cfg['base_G'],map_location='cpu',weights_only=True));self.base.eval()
        self.flow=Velocity();self.flow.load_state_dict(torch.load(cfg['endpoint_weight_paths'][arm],map_location='cpu',weights_only=True));self.flow.eval()
        self.base.requires_grad_(False);self.flow.requires_grad_(False)
        with np.load(Path(root)/'AFFINE_CONSTANTS.npz',allow_pickle=False)as z:
            self.mu=z['mu_V'].copy();self.scale=z['scale_V'].copy();self.edges=z['edges_logQ'].copy()
        self.steps=cfg['solver_steps'];self.details=None

def infer(model,source,z,white,style,chunk=128):
    base,signal,electronic=base_infer(model.base,source,z,white,style,chunk)
    pre,applied,b=correct(model.flow,base,source,model.mu,model.scale,model.edges,model.steps,chunk)
    model.details={'base_pre_ADC_V':base,'base_signal_V':signal,'base_electronic_V':electronic,
                   'flow_increment_V':pre-base,'flow_applied':applied,'source_Qbin_actual_preADC':b}
    return pre,signal,electronic
