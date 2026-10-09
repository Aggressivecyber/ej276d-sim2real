from __future__ import annotations
import time,os,copy
START=time.monotonic()
import numpy as np,torch
from pathlib import Path
from diagnostic_support import ROOT,CFG,write,log,load,digest,check_output
from model import physical,pooled_shape
from fine_core import FineLoss,NAMES
torch.set_num_threads(4);torch.set_num_interop_threads(4)
old=Path(CFG['R23_root']);t=load(old/'TRAIN_PRIVATE.npz',['target','target_bin'])
assert t['target'].shape==(21047,480)
bin_all=t['target_bin'].astype('i8');use=(bin_all>=0)&(bin_all<8)
assert int(use.sum())==18895
v=load(old/'VALIDATION_INPUTS.npz',['physical_center','physical_scale','physical_projection','shape_projection'])
with np.load(old/'FINE_CONSTANTS.npz',allow_pickle=False) as z:constants={k:z[k].copy() for k in z.files}
fine=FineLoss(constants,'cpu')
with torch.no_grad():
    x=torch.from_numpy(t['target'])
    projected=[((physical(x)-torch.from_numpy(v['physical_center']))/torch.from_numpy(v['physical_scale']))@torch.from_numpy(v['physical_projection']),pooled_shape(x)@torch.from_numpy(v['shape_projection']),*fine.projected(x)]
    projected=[a.contiguous() for a in projected]
    assert all(a.dtype==torch.float32 and torch.isfinite(a).all() for a in projected)
    pools=[np.flatnonzero(bin_all==b) for b in range(8)]
    assert all(len(p)>0 for p in pools)
    means=[torch.stack([a[p].double().mean(0).float() for p in pools]) for a in projected]
del x,t
cs=torch.tensor(CFG['A']['c'],dtype=torch.float32)
q=torch.linspace(.005,.995,128,dtype=torch.float32)
coef=(2*torch.arange(32,dtype=torch.float64)-31)[None,:,None]
all_draws={};states={};values=[];regimes=['MIXED','HOMOGENEOUS'];counts=[]
for name in regimes:
    rng=np.random.default_rng(CFG['A']['seeds'][name]);initial=copy.deepcopy(rng.bit_generator.state)
    rowbins=rng.integers(0,8,size=(256,32)) if name=='MIXED' else np.repeat(np.repeat(np.arange(8),32)[:,None],32,axis=1)
    ix=np.empty((256,32),dtype='i8');iy=np.empty_like(ix)
    for r,bins in enumerate(rowbins):
        for b in range(8):
            mask=bins==b;n=int(mask.sum())
            ix[r,mask]=rng.choice(pools[b],n,replace=True);iy[r,mask]=rng.choice(pools[b],n,replace=True)
    all_draws[name+'__X']=ix;all_draws[name+'__Y']=iy;all_draws[name+'__bin']=rowbins.astype('u1')
    states[name]={'seed':CFG['A']['seeds'][name],'initial':initial,'final':rng.bit_generator.state,'X_indices_sha256':digest(ix),'Y_indices_sha256':digest(iy),'bins_sha256':digest(rowbins.astype('u1'))}
    # repeats x coefficient x (five groups plus original combined term) x two statistics
    result=np.empty((256,4,6,2),dtype='f8')
    with torch.no_grad():
        for r in range(256):
            for g,(a,mu) in enumerate(zip(projected,means)):
                xx=a[ix[r]];yy=a[iy[r]];mm=mu[rowbins[r]]
                xc=mm[None]+cs[:,None,None]*(xx-mm)[None]
                # c=1 is exactly the original projected empirical row, not a roundtrip.
                xc[2]=xx
                w=(torch.quantile(xc,q,dim=1)-torch.quantile(yy,q,dim=0)[:,None,:]).abs().mean(dim=(0,2))
                z=xc.double();y=yy.double()
                cross=(z[:,:,None,:]-y[None,None,:,:]).abs().sum(dim=(1,2))
                diagonal=(z-y[None]).abs().sum(dim=1)
                within_x=2*(z.sort(dim=1).values*coef).sum(dim=1)
                within_y=2*(y.sort(dim=0).values*coef[0]).sum(dim=0)
                u=(2*(cross-diagonal)-within_x-within_y[None]).mean(dim=1)/(32*31)
                result[r,:,g,0]=w.double().numpy();result[r,:,g,1]=u.numpy()
            result[r,:,5,:]=2*(result[r,:,0,:]+result[r,:,1,:])+result[r,:,2:5,:].mean(axis=1)
            if (r+1)%64==0:log({'phase':name,'complete_repetitions':r+1,'elapsed_s':time.monotonic()-START})
    values.append(result)
allvalues=np.stack(values);np.savez_compressed(ROOT/'A_DRAWS_PRIVATE.npz',**all_draws)
np.savez_compressed(ROOT/'A_VALUES_PRIVATE.npz',values=allvalues,c=np.array(CFG['A']['c']),**{n+'__mu':a.numpy() for n,a in zip(CFG['A']['groups'],means)})
write('A_RANDOM_STATES.json',states)
def estimate(a,stratified):
    if stratified:
        strata=a.reshape(8,32);mean=float(strata.mean(axis=1).mean());se=float(np.sqrt(np.sum(strata.var(axis=1,ddof=1)/32)/64))
    else:mean=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)))
    return {'mean':mean,'MC_SE':se,'CI95':[mean-1.96*se,mean+1.96*se]}
summary={}
names=CFG['A']['groups']+['original_distribution_combination']
for k,name in enumerate(regimes):
    rows={}
    for g,group in enumerate(names):
        rows[group]={}
        for s,stat in enumerate(['W1_quantile','energy_U']):
            cells=[]
            for c in range(4):
                a=allvalues[k,:,c,g,s];delta=a-allvalues[k,:,2,g,s]
                cell={'c':CFG['A']['c'][c],'absolute':estimate(a,k==1),'paired_minus_c1':estimate(delta,k==1)}
                if k==1:cell['per_bin']=[{'bin_1based':b+1,'absolute':estimate(a[b*32:(b+1)*32],False),'paired_minus_c1':estimate(delta[b*32:(b+1)*32],False)} for b in range(8)]
                cells.append(cell)
            rows[group][stat]=cells
    summary[name]=rows
receipt={'PID':os.getpid(),'elapsed_s':time.monotonic()-START,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','TRAIN_target_loaded':21047,'TRAIN_common':18895,'per_bin_counts':[len(p) for p in pools],'projection_shapes':[list(a.shape) for a in projected],'projection_dtype':'float32','c1_original_rows_exact':True,'batch_pairs':512,'controlled_coefficients':4,'frozen_projection_constants_reestimated':False,'gradient_or_parameter_updates':0,'CI_is_fixed_empirical_pool_Monte_Carlo_only':True,'cross_domain_diagonal_excluded':True,'negative_U_values_preserved':True,'physical_waveforms_generated':0,'new_fits':0,'DEV_EVAL_CAL_waveforms':0}
write('A_METRICS.json',{'receipt':receipt,'summary':summary});check_output();log(receipt)
