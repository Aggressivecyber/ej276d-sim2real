from __future__ import annotations
import numpy as np, torch, json, hashlib, pathlib, copy
from fine_core import FineLoss, FreshFineLoss, NAMES

SIGMAS=(.25,.5,1.,2.,4.)

def gram(a,b):
    # Explicit pair differences; no distance expansion or clamp.
    distance=sum((x[:,None,:]-y[None,:,:]).square().mean(-1) for x,y in zip(a,b))/3.
    return torch.stack([torch.exp(-distance/(2*s*s)) for s in SIGMAS]).mean(0)

def mmd_blocks(a,b):
    aa=gram(a,a);bb=gram(b,b);ab=gram(a,b)
    n,m=aa.shape[0],bb.shape[0]
    if min(n,m)<2:raise ValueError('MMD requires >=2 each domain')
    ga=(aa.sum()-aa.diagonal().sum())/(n*(n-1))
    gb=(bb.sum()-bb.diagonal().sum())/(m*(m-1))
    gc=ab.mean()
    return ga+gb-2*gc,[ga,gb,gc]

class JointKernelLoss(FineLoss):
    def __init__(self,constants,device,seed,root):
        super().__init__(constants,device)
        self.root=pathlib.Path(root);self.seed=seed;self.calls=0;self.c=None;self.rows=[]
    def blocks(self,x):
        # Original float32 waveform operations/scales, then cast standardized features.
        s=x-x[:,:100].mean(1,keepdim=True);u=s/s.amax(1,keepdim=True).clamp_min(.0025)
        values=[u,torch.diff(u,dim=1),s[:,:100]]
        return [((a-self.data[n+'__mean'])/self.data[n+'__scale']).to(torch.float64) for n,a in zip(NAMES,values)]
    def raw(self,y,r):return mmd_blocks(self.blocks(y),self.blocks(r))
    def calibrate(self,model,xs,xt,actual,batch=32):
        reference=FreshFineLoss({k:v.detach().cpu().numpy() for k,v in self.data.items()},xs.device,self.seed,self.root)
        # Copied PCG64; no training/NCE/noise/style/validation RNG consumer.
        rows=[];oldsum=0.;newsum=0.
        for j in range(64):
            sl=slice(j*batch,(j+1)*batch);ordinal=actual['original_row_ordinal'][sl]
            x=xs[actual['source'][sl].astype('i8')];r=xt[actual['target'][sl].astype('i8')]
            inputs=[torch.from_numpy(a).to(xs.device) for a in [actual['original_noise']['z'][ordinal],actual['original_noise']['white'][ordinal],actual['original_style'][ordinal]]]
            with torch.no_grad():y=model(x,*inputs)[0]
            y=y.detach().requires_grad_(True)
            old,_=reference.training(y,r);new,gg=self.raw(y,r)
            go=torch.autograd.grad(old,y,retain_graph=True)[0];gn=torch.autograd.grad(new,y)[0]
            no=float(go.to(torch.float64).square().sum());nn=float(gn.to(torch.float64).square().sum())
            if not np.isfinite([float(old),float(new),no,nn]).all():raise FloatingPointError('nonfinite initial calibration')
            oldsum+=no;newsum+=nn
            rows.append({'original_update_ordinal':j+1,'source_rows':actual['source'][sl].tolist(),'target_rows':actual['target'][sl].tolist(),
                'original_row_ordinal':ordinal.tolist(),'bin_ids_1based':(actual['bin'][sl]+1).tolist(),
                'old_fine':float(old.detach()),'raw_MMD_U':float(new.detach()),'old_grad_y_Frobenius_squared':no,'new_grad_y_Frobenius_squared':nn,
                'within_G_offdiag_mean':float(gg[0]),'within_REAL_offdiag_mean':float(gg[1]),'cross_all_mean':float(gg[2])})
        reference.persist()
        if not np.isfinite([oldsum,newsum]).all() or min(oldsum,newsum)<=1e-24:raise FloatingPointError('invalid calibration aggregate')
        self.c=float(np.sqrt(oldsum/newsum))
        if not np.isfinite(self.c) or self.c<=0:raise FloatingPointError('invalid fixed coefficient')
        rec={'batches':64,'source_extra_forward_rows':2048,'REAL_reused_rows':2048,'coefficient':self.c,
            'sum_old_grad_y_squared':oldsum,'sum_new_grad_y_squared':newsum,'rows':rows,
            'reference_direction_stream_calls':reference.calls,'reference_direction_matrices':192,
            'reference_PC64_seed':self.seed,'reference_directions_separate_from_all_original_streams':True,
            'network_or_optimizer_updates':0,'initial_output_gradient_scale_only':True}
        (self.root/'INITIAL_GRADIENT_SCALE.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n','utf8')
        self.calibration=rec;return rec
    def training(self,y,r):
        if self.c is None:raise RuntimeError('calibration missing')
        raw,g=self.raw(y,r);scaled=raw*self.c;self.calls+=1
        self.rows.append([float(raw.detach()),float(scaled.detach()),*[float(v.detach()) for v in g]])
        return scaled,g
    def persist(self):
        a=np.asarray(self.rows,dtype='f8').reshape(-1,5)
        np.savez_compressed(self.root/'MMD_GRAMS_BY_UPDATE_PRIVATE.npz',values=a)
        rec={'calls':self.calls,'Gram_groups':3*self.calls,'bandwidths':SIGMAS,'kernel_terms':15*self.calls,
            'within_G_offdiag_entries_per_step':992,'within_REAL_offdiag_entries_per_step':992,'cross_entries_per_step':1024,
            'columns':['raw_MMD_U','scaled_MMD','within_G_offdiag_mean','within_REAL_offdiag_mean','cross_all_mean'],
            'finite_negative_MMD_batches':int((a[:,0]<0).sum()),'coefficient':self.c,'no_projection_rng_in_updates':True}
        (self.root/'MMD_TRAINING_RECEIPT.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n','utf8')

def selfcheck():
    torch.manual_seed(39201)
    a=[torch.randn(4,d,dtype=torch.float64,requires_grad=True) for d in [3,2,2]]
    b=[torch.randn(5,d,dtype=torch.float64) for d in [3,2,2]]
    v,_=mmd_blocks(a,b)
    def direct(x,y):
        def k(i,j):
            d=sum((i0-j0).square().mean() for i0,j0 in zip(i,j))/3
            return sum(torch.exp(-d/(2*s*s)) for s in SIGMAS)/len(SIGMAS)
        rowsx=list(zip(*x));rowsy=list(zip(*y));n=len(rowsx);m=len(rowsy)
        return sum(k(rowsx[i],rowsx[j])for i in range(n)for j in range(n)if i!=j)/(n*(n-1))+sum(k(rowsy[i],rowsy[j])for i in range(m)for j in range(m)if i!=j)/(m*(m-1))-2*sum(k(i,j)for i in rowsx for j in rowsy)/(n*m)
    d=direct(a,b);ga=torch.autograd.grad(v,a,retain_graph=True);gb=torch.autograd.grad(d,a)
    same,_=mmd_blocks([x.detach()for x in a],[x.detach()for x in a]);exchange,_=mmd_blocks(b,a)
    perm,_=mmd_blocks([x[[2,0,3,1]]for x in a],[x[[4,1,3,0,2]]for x in b])
    errors={'value_loop_abs':abs(float(v-d)),'gradient_loop_maxabs':max(float((i-j).abs().max())for i,j in zip(ga,gb)),
        'exchange_abs':abs(float(v-exchange)),'row_permutation_abs':abs(float(v-perm)),'identical_sample_negative_U':float(same),
        'within_diagonal_removed_each':True,'cross_diagonal_not_removed':True,'two_generated_arguments_differentiate':True}
    assert max(errors[k]for k in ['value_loop_abs','gradient_loop_maxabs','exchange_abs','row_permutation_abs'])<1e-12 and float(same)<0
    return {'pass':True,'checks':errors,'toy_response_fit':0,'SIGMAS':SIGMAS}
