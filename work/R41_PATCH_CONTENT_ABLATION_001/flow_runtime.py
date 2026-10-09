from __future__ import annotations
import numpy as np,torch
from model import Translator,infer as direct_infer
from flow_core import bins

class Pipeline:
    def __init__(self,cfg,root,arm):
        self.base=Translator();self.base.load_state_dict(torch.load(cfg['endpoint_weight_paths'][arm],map_location='cpu',weights_only=True))
        self.base.eval();self.base.requires_grad_(False)
        with np.load(cfg['affine_path'],allow_pickle=False)as z:self.edges=z['edges_logQ'].copy()
        self.details=None;self.arm=arm

def infer(model,source,z,white,style,chunk=128):
    pre,signal,electronic=direct_infer(model.base,source,z,white,style,chunk)
    b=bins(source,model.edges);support=(b>=0)&(b<8)
    model.details={'base_pre_ADC_V':pre.copy(),'base_signal_V':signal,'base_electronic_V':electronic,
        'base_sum_rounding_f64_V':pre.astype('f8')-signal.astype('f8')-electronic.astype('f8'),
        'flow_increment_V':np.zeros_like(pre),'flow_increment_f64_V':np.zeros(pre.shape,dtype='f8'),
        'flow_applied':support,'source_Qbin_actual_preADC':b,'G_applied_all_rows':np.ones(len(pre),bool)}
    return pre,signal,electronic
