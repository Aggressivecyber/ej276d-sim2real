from __future__ import annotations
import numpy as np
import torch
from model import Translator,infer as base_infer
from flow_core import bins
from sliced_core import transform

class Pipeline:
    def __init__(self,cfg,root,arm):
        self.base=Translator();self.base.load_state_dict(torch.load(cfg['base_G'],map_location='cpu',weights_only=True))
        self.base.eval();self.base.requires_grad_(False)
        with np.load(cfg['endpoint_weight_paths'][arm],allow_pickle=False)as z:
            self.mu=z['mu_V'].copy();self.scale=z['scale_V'].copy();self.edges=z['edges_logQ'].copy()
            self.A=z['A'].copy();self.xk=z['xk'].copy();self.yk=z['yk'].copy()
        assert self.A.shape==(8,32,480,16)and self.xk.shape==self.yk.shape==(8,32,257,16)
        assert all(np.isfinite(v).all()for v in [self.A,self.xk,self.yk,self.mu,self.scale])and np.all(self.scale>0)
        assert np.all(np.diff(self.xk,axis=2)>0)and np.all(np.diff(self.yk,axis=2)>0)
        self.base_chunk=128;self.details=None
    def edit(self,guide,b,chunk=128):
        pre,out64,used=transform(guide,b,self.A,self.xk,self.yk,self.mu,self.scale)
        g=np.asarray(guide,dtype='f4')
        self.details={'base_pre_ADC_V':g.copy(),'flow_increment_V':(pre-g).astype('f4'),
            'flow_increment_f64_V':pre.astype('f8')-g.astype('f8'),
            'calibration_delta_before_cast_f64_V':out64-g.astype('f8'),
            'calibrated_pre_ADC_f64_V':out64,'calibration_cast_rounding_f64_V':pre.astype('f8')-out64,
            'flow_applied':used,'source_Qbin_actual_preADC':np.asarray(b,dtype='i8')}
        return pre

def infer(model,source,z,white,style,chunk=128):
    base,signal,electronic=base_infer(model.base,source,z,white,style,model.base_chunk)
    pre=model.edit(base,bins(source,model.edges),chunk)
    model.details.update(base_signal_V=signal,base_electronic_V=electronic,
        base_sum_rounding_f64_V=base.astype('f8')-signal.astype('f8')-electronic.astype('f8'))
    return pre,signal,electronic
