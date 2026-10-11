"""Float32 storage-interval order ablation; no fitted uncertainty or row removal."""
import numpy as np
import torch
from numpy_reference import q as q_np


def psd_np(v):
    x=np.asarray(v,dtype='f8');return 1-(x[:,100:150].sum(1)-.5*x[:,:100].sum(1))/q_np(x)


def rounding_psd_interval(source):
    x=np.asarray(source,dtype='f4');v=x.astype('f8')
    prev=np.nextafter(x,np.float32(-np.inf)).astype('f8');nxt=np.nextafter(x,np.float32(np.inf)).astype('f8')
    low=.5*(prev+v);high=.5*(v+nxt)
    qlo=low[:,100:400].sum(1)-3*high[:,:100].sum(1);qhi=high[:,100:400].sum(1)-3*low[:,:100].sum(1)
    flo=low[:,100:150].sum(1)-.5*high[:,:100].sum(1);fhi=high[:,100:150].sum(1)-.5*low[:,:100].sum(1)
    assert np.isfinite(qlo).all()and(qlo>0).all()
    corners=np.stack([flo/qlo,flo/qhi,fhi/qlo,fhi/qhi])
    lo=np.nextafter(1-corners.max(0),-np.inf);hi=np.nextafter(1-corners.min(0),np.inf)
    actual=psd_np(v)
    assert np.all(lo<=actual)and np.all(actual<=hi)
    return lo,hi


def precision_order_loss(output_psd, source_psd, low, high, row_weight, scale):
    ds=source_psd[:,None]-source_psd[None,:]
    dy=output_psd[:,None]-output_psd[None,:]
    weight=(row_weight[:,None]*row_weight[None,:])*(ds!=0)
    denominator=weight.sum().clamp_min(1e-30)
    resolved=(low[:,None]>high[None,:])|(high[:,None]<low[None,:])
    penalty=torch.relu(-torch.sign(ds)*dy)
    loss=(weight*resolved*penalty).sum()/denominator/scale
    diagnostic={
        'original_order_loss':((weight*penalty).sum()/denominator/scale).detach(),
        'omitted_order_loss':((weight*(~resolved)*penalty).sum()/denominator/scale).detach(),
        'original_nonzero_pair_mass':weight.sum().detach(),
        'resolved_pair_fraction':((weight*resolved).sum()/denominator).detach()}
    return loss,diagnostic


def artificial_precision_check(device='cpu'):
    device=torch.device(device)
    rng=np.random.Generator(np.random.PCG64(6101))
    ideal=rng.normal(0, .0001, (8, 480))
    ideal[:,100:400]+=.04*np.exp(-np.arange(300)/45)[None,:]
    low,high=rounding_psd_interval(ideal.astype('f4'))
    true_psd=psd_np(ideal)
    assert np.all(low<=true_psd)and np.all(true_psd<=high)
    t=lambda a:torch.tensor(a,dtype=torch.float64,device=device)
    p=t([.1,.100000001,.2,.3,.3]);w=t([.1,.15,.2,.25,.3])
    y=t([.35,.12,.25,.08,.09]).requires_grad_(True)
    lo=t([.09,.09,.19,.29,.29]);hi=t([.11,.11,.21,.31,.31])
    value,details=precision_order_loss(y,p,lo,hi,w,.01)
    pnp=p.cpu().numpy();ynp=y.detach().cpu().numpy();wnp=w.cpu().numpy()
    lonp=lo.cpu().numpy();hinp=hi.cpu().numpy()
    den=sum(wnp[i]*wnp[j]for i in range(5)for j in range(5)if pnp[i]!=pnp[j])
    numerator=sum(wnp[i]*wnp[j]*max(-np.sign(pnp[i]-pnp[j])*(ynp[i]-ynp[j]),0)
                  for i in range(5)for j in range(5)if pnp[i]!=pnp[j]and(lonp[i]>hinp[j]or hinp[i]<lonp[j]))
    assert abs(float(value.detach())-numerator/den/.01)<1e-12
    value.backward();gradient=y.grad.detach().cpu().numpy()
    finite=[]
    for k in range(5):
        shift=torch.zeros_like(y);shift[k]=1e-7
        a=precision_order_loss(y.detach()+shift,p,lo,hi,w,.01)[0]
        b=precision_order_loss(y.detach()-shift,p,lo,hi,w,.01)[0]
        finite.append(float((a-b)/(2e-7)))
    error=float(np.max(np.abs(np.asarray(finite)-gradient)))
    assert error<1e-7
    old_weight=(w[:,None]*w[None,:])*(p[:,None]!=p[None,:])
    old=(old_weight*torch.relu(-torch.sign(p[:,None]-p[None,:])*(y[:,None]-y[None,:]))).sum()/old_weight.sum().clamp_min(1e-30)/.01
    fully_resolved=precision_order_loss(y,p,p,p,w,.01)[0]
    assert torch.equal(fully_resolved,old)
    zero,zd=precision_order_loss(y,p,torch.zeros_like(p),torch.ones_like(p),w,.01)
    assert float(zero.detach())==0 and float(zd['resolved_pair_fraction'])==0
    touching=precision_order_loss(t([.5,.1]),t([.1,.2]),t([0.,.15]),t([.15,.3]),t([.5,.5]),.01)[0]
    assert float(touching)==0
    equal=precision_order_loss(t([.5,.1]),t([.2,.2]),t([.1,.1]),t([.3,.3]),t([.5,.5]),.01)[0]
    assert float(equal)==0
    assert abs(float(details['original_order_loss'])-float(value.detach())-float(details['omitted_order_loss']))<1e-12
    return {'artificial_only':True,'device':str(device),'latent_float64_PSD_inside_float32_envelope':True,
            'unchanged_original_nonzero_pair_denominator':True,'mixed_nonuniform_pair_sum_matches':True,
            'fully_resolved_matches_original_exactly':True,'all_overlap_zero':True,'touching_endpoints_overlap':True,
            'equal_PSD_and_duplicate_parent_handling':True,'gradient_finite_difference_max_error':error,
            'original_equals_resolved_plus_omitted':True,'new_scientific_data_read':0,'new_optimizer_updates':0}
