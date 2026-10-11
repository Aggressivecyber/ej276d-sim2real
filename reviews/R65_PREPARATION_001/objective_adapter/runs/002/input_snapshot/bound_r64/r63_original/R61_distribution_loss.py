"""Direct distribution objectives, with explicit original-parent global mass."""
import numpy as np
import torch
from precision_order import precision_order_loss
from source_flow_core import q_torch,canonical_REAL_torch,canonical_adc_torch

NAMES=['V480','U480','DU479','D2U478']
CATEGORIES=[-1,*range(8),8]

def feature_vectors(v):
    s=v-v[:,:100].mean(1,keepdim=True);u=s/s.amax(1,keepdim=True).clamp_min(.0025)
    return [v,u,torch.diff(u,dim=1),torch.diff(u,n=2,dim=1)]

def psd(v):
    q=q_torch(v);s=v[:,100:150].sum(1)-.5*v[:,:100].sum(1)
    return 1-s/q

def noise_features(v):
    b=v[:,:100]-v[:,:100].mean(1,keepdim=True);power=b.square().mean(1)
    return torch.stack((power.clamp_min(1e-20).sqrt(),(b[:,1:]*b[:,:-1]).mean(1)/power.clamp_min(1e-20)),1)

def weighted_w1(x,wx,y,wy):
    """Exact quantile-axis integral for two finite nonnegative weighted laws."""
    sx,ix=torch.sort(x,stable=True);sy,iy=torch.sort(y,stable=True)
    a=wx[ix]/wx.sum();b=wy[iy]/wy.sum();ca=torch.cumsum(a,0);cb=torch.cumsum(b,0)
    # All masses are fixed data, so the quantile partitions have no parameter gradient.
    ca=ca.detach();cb=cb.detach();ca[-1]=1.;cb[-1]=1.
    edges=torch.sort(torch.cat((ca.new_zeros(1),ca,cb)),stable=True).values
    length=edges[1:]-edges[:-1];middle=(edges[1:]+edges[:-1])*.5
    ia=torch.searchsorted(ca.contiguous(),middle.contiguous()).clamp_max(len(ca)-1)
    ib=torch.searchsorted(cb.contiguous(),middle.contiguous()).clamp_max(len(cb)-1)
    return (length*(sx[ia]-sy[ib]).abs()).sum()

@torch.no_grad()
def fit_statistics(real,source_psd,source_h,source_bins,REAL_bins,valley,cfg):
    """TRAIN-only coordinates, not a response fit or a cross-domain teacher."""
    v=canonical_REAL_torch(torch.as_tensor(real,dtype=torch.float64))
    if not (q_torch(v)>0).all():raise ValueError('nonpositive canonical REAL charge; no row deletion')
    answer={'source_counts':np.asarray([(source_bins==b).sum() for b in CATEGORIES],dtype='i8'),
        'REAL_counts':np.asarray([(REAL_bins==b).sum() for b in CATEGORIES],dtype='i8')}
    assert np.all(answer['source_counts']>0) and np.all(answer['REAL_counts']>0)
    for name,features in zip(NAMES,feature_vectors(v)):
        centers=[];scales=[]
        for b in CATEGORIES:
            selected=features[torch.from_numpy(REAL_bins==b)]
            centers.append(selected.mean(0).numpy());scales.append(selected.std(0,unbiased=False).clamp_min(cfg['feature_scale_floor'][name]).numpy())
        answer[name+'_center']=np.stack(centers);answer[name+'_scale']=np.stack(scales)
    nf=noise_features(v);answer['noise_mean']=nf.mean(0).numpy();answer['noise_scale']=np.maximum(nf.std(0,unbiased=False).numpy(),np.asarray(cfg['noise_scale_floor']))
    bounds=[];hfractions=[];rawfractions=[]
    for b in [2,3]:
        ix=(source_bins==b)&(source_h==1);de=valley['bins'][b]
        assert de['valid'] and ix.any()
        bounds.append([de['low'],de['high']]);hfractions.append(float(ix.sum()/np.sum(source_bins==b)))
        rawfractions.append(float(np.mean((source_psd[ix]>=de['low'])&(source_psd[ix]<=de['high']))))
    answer['valley_bounds']=np.asarray(bounds);answer['source_h_one_fraction']=np.asarray(hfractions);answer['source_raw_valley_fraction']=np.asarray(rawfractions)
    return answer

class DistributionLoss:
    def __init__(self,statistics,cfg,device):
        self.cfg=cfg;self.B=cfg['batch_per_region'];self.device=device
        self.stats={k:torch.as_tensor(v,device=device,dtype=torch.float64) for k,v in statistics.items()}
        self.source_mass=self.stats['source_counts']/self.stats['source_counts'].sum();self.REAL_mass=self.stats['REAL_counts']/self.stats['REAL_counts'].sum()
        self.sw=torch.repeat_interleave(self.source_mass/self.B,self.B);self.rw=torch.repeat_interleave(self.REAL_mass/self.B,self.B)
        self.direction_calls=0
    def draw_directions(self,rng):
        result=[]
        for dim in [480,480,479,478]:
            p=rng.standard_normal((dim,self.cfg['projection_directions']));p/=np.linalg.norm(p,axis=0,keepdims=True)
            result.append(p.astype('<f4'))
        self.direction_calls+=1
        return result
    def __call__(self,output,raw,reference,identity_input,identity_bins,source_psd,source_h,directions,source_interval_low,source_interval_high):
        n=10*self.B;candidate=canonical_adc_torch(output[:n],True);ref=canonical_REAL_torch(reference)
        ident=canonical_adc_torch(output[n:],True);ident_ref=canonical_REAL_torch(identity_input)
        pieces={};per_feature=[];feature_bin=[]
        for name,fy,fr,p in zip(NAMES,feature_vectors(candidate),feature_vectors(ref),directions):
            dim=fy.shape[1];center=self.stats[name+'_center'][:,None,:];scale=self.stats[name+'_scale'][:,None,:]
            a=((fy.reshape(10,self.B,dim)-center)/scale).to(torch.float32)
            b=((fr.reshape(10,self.B,dim)-center)/scale).to(torch.float32)
            direction=torch.as_tensor(p,device=self.device)
            marginal=(torch.sort(a,dim=1,stable=True).values-torch.sort(b,dim=1,stable=True).values).abs().mean((1,2))
            sliced=(torch.sort(a@direction,dim=1,stable=True).values-torch.sort(b@direction,dim=1,stable=True).values).abs().mean((1,2))
            loss=.5*(marginal+sliced);feature_bin.append(loss.detach());per_feature.append((loss*self.source_mass).sum())
        pieces['full_window']=torch.stack(per_feature).mean()
        yp=psd(candidate);rp=psd(ref)
        overall=weighted_w1(yp,self.sw,rp,self.rw)
        bybin=(torch.sort(yp.reshape(10,self.B),dim=1,stable=True).values-torch.sort(rp.reshape(10,self.B),dim=1,stable=True).values).abs().mean(1)
        conditional=bybin[1:8].mean();pieces['PSD']=.5*(overall+conditional)/self.cfg['PSD_loss_scale']
        pieces['source_order'],order_diagnostic=precision_order_loss(yp,source_psd,source_interval_low,source_interval_high,self.sw,self.cfg['PSD_loss_scale'])
        valleys=[];probabilities=[]
        for j,b in enumerate([2,3]):
            ix=slice((b+1)*self.B,(b+2)*self.B);low,high=self.stats['valley_bounds'][j]
            smooth=torch.sigmoid((yp[ix]-low)/self.cfg['valley_smoothing'])*torch.sigmoid((high-yp[ix])/self.cfg['valley_smoothing'])
            probability=(smooth*(source_h[ix]==1)).mean()/self.stats['source_h_one_fraction'][j]
            excess=torch.relu(probability-self.stats['source_raw_valley_fraction'][j]-self.cfg['valley_delta'])/self.cfg['valley_delta']
            valleys.append(excess.square());probabilities.append(probability.detach())
        pieces['source_valley']=torch.stack(valleys).mean()
        yn=noise_features(candidate);rn=noise_features(ref)
        pieces['noise']=torch.stack([weighted_w1(yn[:,j],self.sw,rn[:,j],self.rw)/self.stats['noise_scale'][j] for j in range(2)]).mean()
        scale=self.stats['V480_scale'][identity_bins+1]
        pieces['REAL_identity']=((ident-ident_ref)/scale).square().mean()
        correction=(output-raw).square().mean(1)
        pieces['box_correction']=.5*((correction[:n]*self.sw).sum()+correction[n:].mean())/(.5**2)
        total=sum(self.cfg['loss_weights'][k]*v for k,v in pieces.items())
        diagnostics={'overall_PSD_W1':overall.detach(),'Q1_Q7_mean_PSD_W1':conditional.detach(),
            'Q3_h_one_soft_valley':probabilities[0],'Q4_h_one_soft_valley':probabilities[1],
            'feature_components':torch.stack(per_feature).detach(),'feature_by_region':torch.stack(feature_bin),
            'PSD_by_region':bybin.detach(),**order_diagnostic}
        return total,pieces,diagnostics
