"""Frozen-objective decomposition and finite-precision diagnostics; no optimizer."""
import numpy as np
import torch
from distribution_loss import NAMES,CATEGORIES,feature_vectors,psd,weighted_w1
from source_flow_core import canonical_adc_torch,canonical_REAL_torch
from noise_core import q_np

ORIGINAL=['full_window','PSD','source_order','source_valley','noise','REAL_identity','box_correction']
COMPONENTS=ORIGINAL+['full_'+n for n in NAMES]+['source_order_roundoff_compatible','source_order_resolved',
 'full_window_inside','full_window_outside','PSD_overall','PSD_Q1_Q7']

def psd_np(v):
    x=np.asarray(v,dtype='f8');return 1-(x[:,100:150].sum(1)-.5*x[:,:100].sum(1))/q_np(x)

def noise_np(v):
    b=np.asarray(v,dtype='f8')[:,:100];b=b-b.mean(1,keepdims=True);p=(b*b).mean(1)
    return np.c_[np.sqrt(np.maximum(p,1e-20)),(b[:,1:]*b[:,:-1]).mean(1)/np.maximum(p,1e-20)]

def canonical_np(v,generated):
    if generated:
        f=np.asarray(v,dtype='f4');legacy=(-.5+np.round((np.clip(f,-.5,.5)+.5)*16383)/16383).astype('f4')
        v=legacy.astype('f8')
    return -.5+np.clip(np.round((np.asarray(v,dtype='f8')+.5)*16383),0,16383)/16383

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

def exact_w1_columns(a,b):
    a=np.asarray(a,dtype='f8');b=np.asarray(b,dtype='f8')
    if a.ndim==1:a=a[:,None]
    if b.ndim==1:b=b[:,None]
    assert a.shape[1]==b.shape[1]and len(a)>0 and len(b)>0
    sa=np.sort(a,axis=0,kind='stable');sb=np.sort(b,axis=0,kind='stable')
    ca=np.arange(1,len(a)+1,dtype='f8')/len(a);cb=np.arange(1,len(b)+1,dtype='f8')/len(b)
    edges=np.sort(np.r_[0.,ca,cb],kind='stable');mid=.5*(edges[1:]+edges[:-1]);mass=np.diff(edges)
    ia=np.minimum(np.searchsorted(ca,mid),len(a)-1);ib=np.minimum(np.searchsorted(cb,mid),len(b)-1)
    return (np.abs(sa[ia]-sb[ib])*mass[:,None]).sum(0)

def full_feature_summary(candidate,reference,source_bins,REAL_bins,stats,directions):
    mass=stats['source_counts']/stats['source_counts'].sum();rows=[];full=np.zeros(4)
    for j,region in enumerate(CATEGORIES):
        a=torch.from_numpy(candidate[source_bins==region]);b=torch.from_numpy(reference[REAL_bins==region])
        for k,(name,fa,fb,p)in enumerate(zip(NAMES,feature_vectors(a),feature_vectors(b),directions)):
            x=((fa.numpy()-stats[name+'_center'][j])/stats[name+'_scale'][j]).astype('f4')
            y=((fb.numpy()-stats[name+'_center'][j])/stats[name+'_scale'][j]).astype('f4')
            marginal=float(exact_w1_columns(x,y).mean());sliced=float(exact_w1_columns(x@p,y@p).mean());value=.5*(marginal+sliced)
            full[k]+=mass[j]*value
            rows.append({'region':region,'feature':name,'SOURCE_N':len(x),'REAL_N':len(y),'marginal_W1':marginal,'sliced_W1':sliced,'source_mass':float(mass[j]),'weighted_contribution_to_total':float(mass[j]*value/4)})
    return {'per_region_and_feature':rows,'per_feature':dict(zip(NAMES,full.tolist())),'full_window_loss':float(full.mean()),
      'exact_unequal_empirical_quantile_integral':True,'fixed_projection_count_each_feature':64,'original_parent_regions':True}

def order_population(source_psd,out_psd,low,high,h,scale=.01,chunk=256):
    n=len(source_psd);den=0;records={k:{'pairs':0,'penalty_sum':0.}for k in ['roundoff_compatible','resolved','both_h_one','not_both_h_one']}
    exact_equal=0;same_parent=0
    for start in range(0,n,chunk):
        ix=slice(start,min(start+chunk,n));ds=source_psd[ix,None]-source_psd[None,:];mask=ds!=0
        resolved=(low[ix,None]>high[None,:])|(high[ix,None]<low[None,:]);overlap=~resolved
        assert np.all(~resolved|mask)
        penalty=np.maximum(-np.sign(ds)*(out_psd[ix,None]-out_psd[None,:]),0)
        den+=int(mask.sum());exact_equal+=int((~mask).sum());same_parent+=min(chunk,n-start)
        pure=(h[ix,None]==1)&(h[None,:]==1)
        for k,m in [('roundoff_compatible',overlap),('resolved',resolved),('both_h_one',pure),('not_both_h_one',~pure)]:
            selected=m&mask;records[k]['pairs']+=int(selected.sum());records[k]['penalty_sum']+=float(penalty[selected].sum())
    for v in records.values():v.update(nonzero_pair_fraction=v['pairs']/den,contribution_to_original_order_loss=v['penalty_sum']/den/scale)
    return {'ordered_parent_pairs_total':n*n,'strict_nonzero_pairs':den,'exact_equal_pairs':exact_equal,'diagonal_same_parent_pairs':same_parent,
      'original_loss':sum(records[k]['penalty_sum']for k in ['roundoff_compatible','resolved'])/den/scale,'groups':records,
      'all_original_source_parents_equal':True,'roundoff_intervals_are_storage_only':True}

def decompose(obj,output,raw,reference,identity_input,identity_bins,source_psd,source_h,directions,low,high):
    total,pieces,diagnostics=obj(output,raw,reference,identity_input,identity_bins,source_psd,source_h,directions)
    n=10*obj.B;candidate=canonical_adc_torch(output[:n],True);ref=canonical_REAL_torch(reference)
    extra={};feature_by=[]
    for name,fy,fr,p in zip(NAMES,feature_vectors(candidate),feature_vectors(ref),directions):
        dim=fy.shape[1];center=obj.stats[name+'_center'][:,None,:];scale=obj.stats[name+'_scale'][:,None,:]
        a=((fy.reshape(10,obj.B,dim)-center)/scale).to(torch.float32);b=((fr.reshape(10,obj.B,dim)-center)/scale).to(torch.float32)
        direction=torch.as_tensor(p,device=obj.device)
        marginal=(torch.sort(a,dim=1,stable=True).values-torch.sort(b,dim=1,stable=True).values).abs().mean((1,2))
        sliced=(torch.sort(a@direction,dim=1,stable=True).values-torch.sort(b@direction,dim=1,stable=True).values).abs().mean((1,2))
        loss=.5*(marginal+sliced);feature_by.append(loss);extra['full_'+name]=(loss*obj.source_mass).sum()/4
    per_region=torch.stack(feature_by).mean(0)*obj.source_mass
    extra['full_window_inside']=per_region[1:9].sum();extra['full_window_outside']=per_region[[0,9]].sum()
    yp=psd(candidate);rp=psd(ref);ds=source_psd[:,None]-source_psd[None,:];dy=yp[:,None]-yp[None,:]
    weight=(obj.sw[:,None]*obj.sw[None,:])*(ds!=0);den=weight.sum().clamp_min(1e-30)
    resolved=(low[:,None]>high[None,:])|(high[:,None]<low[None,:]);penalty=torch.relu(-torch.sign(ds)*dy)
    assert not (resolved&(ds==0)).any()
    extra['source_order_roundoff_compatible']=(weight*(~resolved)*penalty).sum()/den/obj.cfg['PSD_loss_scale']
    extra['source_order_resolved']=(weight*resolved*penalty).sum()/den/obj.cfg['PSD_loss_scale']
    extra['PSD_overall']=.5*weighted_w1(yp,obj.sw,rp,obj.rw)/obj.cfg['PSD_loss_scale']
    bybin=(torch.sort(yp.reshape(10,obj.B),dim=1,stable=True).values-torch.sort(rp.reshape(10,obj.B),dim=1,stable=True).values).abs().mean(1)
    extra['PSD_Q1_Q7']=.5*bybin[1:8].mean()/obj.cfg['PSD_loss_scale']
    answer={k:(pieces[k]if k in pieces else extra[k])for k in COMPONENTS}
    fractions={'nonzero_pair_mass':float(den.detach()),'roundoff_compatible_fraction_of_nonzero_mass':float((weight*(~resolved)).sum().detach()/den),
      'resolved_fraction_of_nonzero_mass':float((weight*resolved).sum().detach()/den),'exact_equal_mass':float(((obj.sw[:,None]*obj.sw[None,:])*(ds==0)).sum().detach())}
    return total,answer,diagnostics,fractions

def recompositions():
    return {'full_window_views':('full_window',['full_'+n for n in NAMES]),
      'full_window_regions':('full_window',['full_window_inside','full_window_outside']),
      'PSD':('PSD',['PSD_overall','PSD_Q1_Q7']),
      'source_order':('source_order',['source_order_roundoff_compatible','source_order_resolved'])}

def gradient_geometry(vectors,values,cfg):
    v=np.asarray(vectors,dtype='f8');gram=v@v.T;norm=np.sqrt(np.maximum(np.diag(gram),0));total=v[:7].sum(0);tn=float(np.linalg.norm(total))
    denom=norm[:,None]*norm[None,:];cos=np.divide(gram,denom,out=np.full_like(gram,np.nan),where=denom>1e-30)
    dot=v@total;alignment=np.divide(dot,norm*tn,out=np.full_like(dot,np.nan),where=norm*tn>1e-30)
    checks={}
    for key,(whole,parts)in recompositions().items():
        i=COMPONENTS.index(whole);ix=[COMPONENTS.index(n)for n in parts]
        ve=float(abs(values[i]-sum(values[j]for j in ix)));ge=float(np.linalg.norm(v[ix].sum(0)-v[i])/max(norm[i],1e-12))
        checks[key]={'loss_value_abs_error':ve,'gradient_relative_error':ge,'pass':ve<=cfg['loss_value_atol']+cfg['loss_value_rtol']*abs(values[i])and ge<=cfg['gradient_recomposition_rtol']}
    def cosine(a,b):
        a=np.asarray(a);b=np.asarray(b);d=np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a,b)/d)if d>1e-30 else None
    def get(name):return v[COMPONENTS.index(name)]
    response=get('full_window')+get('PSD')+get('noise');uncertain=get('source_order_roundoff_compatible');resolved=get('source_order_resolved')
    inside=get('full_window_inside');outside=get('full_window_outside');valley=get('source_valley');p=get('PSD')
    frac=lambda a,b:float(np.linalg.norm(a)/(np.linalg.norm(a)+np.linalg.norm(b)))if np.linalg.norm(a)+np.linalg.norm(b)>1e-30 else None
    marker_inputs={'roundoff_response_cosine':cosine(uncertain,response),'roundoff_norm_fraction':frac(uncertain,resolved),
      'outside_inside_cosine':cosine(outside,inside),'outside_norm_fraction':frac(outside,inside),
      'valley_PSD_cosine':cosine(valley,p),'valley_to_PSD_norm_ratio':float(np.linalg.norm(valley)/np.linalg.norm(p))if np.linalg.norm(p)>1e-30 else None}
    return {'gram':gram,'norm':norm,'cosine':cos,'total_norm':tn,'dot_with_total':dot,'alignment_with_total':alignment,'checks':checks,'marker_inputs':marker_inputs}
