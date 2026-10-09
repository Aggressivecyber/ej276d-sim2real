from __future__ import annotations
import json,datetime
from pathlib import Path
import numpy as np
import torch
from model import Translator,infer as base_infer
from flow_core import bins
from diffusion_core import Denoiser,reverse_chain

class Pipeline:
    def __init__(self,cfg,root,arm):
        self.base=Translator();self.base.load_state_dict(torch.load(cfg['base_G'],map_location='cpu',weights_only=True));self.base.eval();self.base.requires_grad_(False)
        self.denoiser=Denoiser();self.denoiser.load_state_dict(torch.load(cfg['endpoint_weight_paths'][arm],map_location='cpu',weights_only=True));self.denoiser.eval();self.denoiser.requires_grad_(False)
        with np.load(Path(root)/'DIFFUSION_CONSTANTS.npz',allow_pickle=False)as z:
            self.mu=z['mu_V'].copy();self.scale=z['scale_V'].copy();self.edges=z['edges_logQ'].copy()
            self.coeff={k:z[k].copy()for k in ['beta','alpha','abar','posterior_variance']}
            self.condition_stats={'mean':z['condition_mean'].copy(),'sample_sd':z['condition_sample_sd'].copy()}
        self.seeds=None;self.namespace=None;self.details=None;self.base_chunk=128
    def set_seeds(self,seeds,namespace):
        self.seeds=np.asarray(seeds,dtype='u8').copy();self.namespace=namespace
    def edit(self,guide,b,seeds=None,fixed_condition_raw=None,chunk=64):
        if seeds is None:seeds=self.seeds
        assert seeds is not None and len(seeds)==len(guide),'explicit record seed mapping required'
        pre,used,raw,c,hashes=reverse_chain(self.denoiser,guide,b,self.condition_stats,self.mu,self.scale,seeds,self.coeff,chunk,fixed_condition_raw)
        detail={'base_pre_ADC_V':np.asarray(guide,dtype='f4').copy(),'flow_increment_V':pre-guide,
            'flow_increment_f64_V':pre.astype('f8')-np.asarray(guide,dtype='f8'),
            'flow_applied':used,'source_Qbin_actual_preADC':np.asarray(b,dtype='i8'),
            'diffusion_record_seed':np.asarray(seeds,dtype='u8'),'condition_raw_logQ_PSD':raw,
            'condition_standardized':c,'actual_diffusion_noise_sha256':hashes}
        self.details=detail;return pre

def infer(model,source,z,white,style,chunk=64):
    base,signal,electronic=base_infer(model.base,source,z,white,style,model.base_chunk)
    b=bins(source,model.edges);pre=model.edit(base,b,chunk=chunk)
    model.details.update(base_signal_V=signal,base_electronic_V=electronic,
        base_sum_rounding_f64_V=base.astype('f8')-signal.astype('f8')-electronic.astype('f8'))
    return pre,signal,electronic
