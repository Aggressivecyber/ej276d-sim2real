from __future__ import annotations
import numpy as np,torch,json,pathlib
from model import sw
from prior import weighted_quantile
NAMES=['shape480','diff479','baseline100_V']

def numpy_vectors(x):
    x=np.asarray(x,dtype=np.float64);s=x-x[:,:100].mean(1,keepdims=True);p=np.maximum(s.max(1,keepdims=True),.0025);u=s/p
    return [u,np.diff(u,axis=1),s[:,:100]]

def fit_constants(target,cfg,root):
    # Only REAL TRAIN-fit21047; no validation/DEV/EVAL statistic enters this function.
    vectors=numpy_vectors(target);rng=np.random.default_rng(cfg['fine_projection_seed']);payload={};rows=[]
    for name,x in zip(NAMES,vectors):
        mean=x.mean(0,dtype=np.float64);std=x.std(0,ddof=0,dtype=np.float64);median=float(np.median(std));floor=max(.05*median,1e-6);scale=np.maximum(std,floor)
        raw=rng.normal(size=(x.shape[1],128));projection=(raw/np.linalg.norm(raw,axis=0,keepdims=True)).astype('f4')
        if not(np.isfinite(mean).all()and np.isfinite(scale).all()and(scale>0).all()):raise FloatingPointError('invalid frozen TRAIN fine-time scales')
        payload.update({name+'__mean':mean,name+'__population_std':std,name+'__scale':scale,name+'__projection':projection})
        rows.append({'group':name,'coordinates':x.shape[1],'TRAIN_N':len(x),'mean_range':list(map(float,[mean.min(),mean.max()])),'population_std_range':list(map(float,[std.min(),std.max()])),'median_coordinate_std':median,'floor':floor,'coordinates_touching_floor':int((std<=floor).sum()),'scale_range':list(map(float,[scale.min(),scale.max()])),'projection_columns':128,'projection_unit_norm_max_error':float(np.max(np.abs(np.linalg.norm(projection.astype('f8'),axis=0)-1))),'units':'V'if name=='baseline100_V'else'dimensionless'})
    np.savez_compressed(root/'FINE_CONSTANTS.npz',**payload)
    (root/'FINE_CONSTANTS_RECEIPT.json').write_text(json.dumps({'fit_role':'REAL_TRAIN_fit_only','N':len(target),'accumulation':'float64 population standard deviation ddof0','projection_seed':cfg['fine_projection_seed'],'projection_rng_does_not_change_training_rng':True,'first_parameter_update_not_yet_started':True,'groups':rows},indent=2)+'\n')
    return payload

class FineLoss:
    def __init__(self,constants,device,dtype=torch.float32):
        self.device=device;self.data={k:torch.as_tensor(v,device=device,dtype=dtype)for k,v in constants.items()}
    def projected(self,x):
        s=x-x[:,:100].mean(1,keepdim=True);u=s/s.amax(1,keepdim=True).clamp_min(.0025);values=[u,torch.diff(u,dim=1),s[:,:100]]
        return [(a-self.data[n+'__mean'])/self.data[n+'__scale']@self.data[n+'__projection']for n,a in zip(NAMES,values)]
    def training(self,y,real):
        a=self.projected(y);b=self.projected(real);q=torch.linspace(.005,.995,128,device=y.device,dtype=y.dtype)
        values=[(torch.quantile(x,q,dim=0)-torch.quantile(r,q,dim=0)).abs().mean()for x,r in zip(a,b)]
        return torch.stack(values).mean(),values
    def validation(self,y,real,source_bins,real_bins,weights,common):
        a=self.projected(y);b=self.projected(real);q=torch.linspace(.005,.995,128,device=y.device,dtype=y.dtype);rows=[];groups=[]
        for i in common:
            sm=source_bins==i;rm=real_bins==i
            vals=[(weighted_quantile(x[sm],weights[sm],q)-torch.quantile(r[rm],q,dim=0)).abs().mean()for x,r in zip(a,b)]
            groups.append(torch.stack(vals));rows.append({'bin_id_1based':i+1,'source_N':int(sm.sum()),'REAL_N':int(rm.sum()),**{n:float(v)for n,v in zip(NAMES,vals)},'equal_three_group_mean':float(torch.stack(vals).mean())})
        means=torch.stack(groups).mean(0)
        return means.mean(),{n:float(v)for n,v in zip(NAMES,means)},rows
