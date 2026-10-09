from __future__ import annotations
import numpy as np


def positive_qr(a):
    q,r=np.linalg.qr(np.asarray(a,dtype='f8'),mode='reduced')
    return q*np.where(np.diag(r)<0,-1.,1.)[None,:]


def scalar_maps(u,xk,yk):
    u=np.asarray(u,dtype='f8');out=np.empty_like(u)
    assert u.ndim==2 and xk.shape==yk.shape and xk.shape[1]==u.shape[1]
    assert np.isfinite(xk).all()and np.isfinite(yk).all()
    assert np.all(np.diff(xk,axis=0)>0)and np.all(np.diff(yk,axis=0)>0),'non-strict knots: recipe stops, no rescue'
    for j in range(u.shape[1]):
        out[:,j]=np.interp(u[:,j],xk[:,j],yk[:,j])
        low=u[:,j]<xk[0,j];high=u[:,j]>xk[-1,j]
        out[low,j]=u[low,j]+yk[0,j]-xk[0,j]
        out[high,j]=u[high,j]+yk[-1,j]-xk[-1,j]
    assert np.isfinite(out).all()
    return out


def layer(x,a,xk,yk,u=None):
    u=x@a if u is None else u
    return x+(scalar_maps(u,xk,yk)-u)@a.T


def learn_directions(x,y,a0,cfg,check_time,charge_step):
    import torch
    xt=torch.from_numpy(np.ascontiguousarray(x,dtype='f8'))
    yt=torch.from_numpy(np.ascontiguousarray(y,dtype='f8'))
    a=torch.from_numpy(a0.copy()).requires_grad_(True)
    def objective(mat):
        return (torch.sort(xt@mat,dim=0).values-torch.sort(yt@mat,dim=0).values).square().mean()
    trace=[]
    for step in range(cfg['direction_gradient_steps']):
        check_time();charge_step();j=objective(a)
        grad=torch.autograd.grad(j,a)[0]
        product=a.T@grad
        tangent=grad-a@((product+product.T)*.5)
        norm=torch.linalg.norm(tangent)
        assert bool(torch.isfinite(j))and bool(torch.isfinite(grad).all())and bool(torch.isfinite(norm))
        with torch.no_grad():
            zero=bool(norm<=cfg['direction_zero_norm'])
            if zero: anew=a.detach().clone()
            else:
                q,r=torch.linalg.qr(a+cfg['direction_stepsize']*tangent/norm,mode='reduced')
                anew=q*torch.where(torch.diag(r)<0,-1.,1.)[None,:]
            err=float((anew.T@anew-torch.eye(anew.shape[1],dtype=torch.float64)).abs().max())
            after=float(objective(anew))
        assert np.isfinite(after)and err<=cfg['orthogonality_tolerance']
        trace.append({'step':step+1,'J_before':float(j.detach()),'J_after':after,
            'euclidean_gradient_Frobenius':float(torch.linalg.norm(grad)),
            'tangent_gradient_Frobenius':float(norm),'zero_gradient_step':zero,
            'orthogonality_maxabs':err,'nonascending':after<=float(j.detach())})
        a=anew.detach().requires_grad_(True)
    return a.detach().numpy().copy(),trace


def apply_chain(x,aa,xks,yks,inverse=False):
    x=np.asarray(x,dtype='f8').copy()
    iterator=range(len(aa)-1,-1,-1)if inverse else range(len(aa))
    for i in iterator:
        x=layer(x,aa[i],yks[i],xks[i])if inverse else layer(x,aa[i],xks[i],yks[i])
        assert np.isfinite(x).all()
    return x


def transform(guide,b,aa,xks,yks,mu,scale):
    guide=np.asarray(guide,dtype='f4');b=np.asarray(b,dtype='i8')
    out64=guide.astype('f8');used=(b>=0)&(b<8)
    for k in range(8):
        rows=np.flatnonzero(b==k)
        if rows.size:
            x=(guide[rows].astype('f8')-mu[k])/scale[k]
            out64[rows]=mu[k]+scale[k]*apply_chain(x,aa[k],xks[k],yks[k])
    assert np.isfinite(out64).all()
    return out64.astype('f4'),out64,used


def selfcheck():
    # Tiny artificial arrays; separate RNG never consumes fitted-direction initialization.
    rng=np.random.Generator(np.random.PCG64(2468));a=positive_qr(rng.normal(size=(8,3)))
    xk=np.tile(np.array([-2.,-.5,1.,3.])[:,None],(1,3))
    yk=np.tile(np.array([-1.,0.,2.,4.])[:,None],(1,3))
    u=np.tile(np.linspace(-5,6,113)[:,None],(1,3))
    ident=float(np.max(np.abs(scalar_maps(u,xk,xk)-u)))
    z=scalar_maps(u,xk,yk);inverse=float(np.max(np.abs(scalar_maps(z,yk,xk)-u)))
    x=rng.normal(size=(12,8));identity_layer=float(np.max(np.abs(layer(x,a,xk,xk)-x)))
    returned=layer(layer(x,a,xk,yk),a,yk,xk);inverse_layer=float(np.max(np.abs(returned-x)))
    eps=1e-9;edges=np.stack([xk[0]-eps,xk[0],xk[0]+eps,xk[-1]-eps,xk[-1],xk[-1]+eps])
    zout=scalar_maps(edges,xk,yk)
    continuity=float(np.max(np.abs(zout[[0,2,3,5]]-zout[[1,1,4,4]])))
    slopes_one=bool(np.allclose(z[0],u[0]+yk[0]-xk[0],rtol=0,atol=1e-12)and
                    np.allclose(z[-1],u[-1]+yk[-1]-xk[-1],rtol=0,atol=1e-12))
    passed=ident<1e-12 and inverse<1e-12 and identity_layer<1e-12 and inverse_layer<1e-12 and continuity<1e-8 and slopes_one and bool(np.all(np.diff(z,axis=0)>0))
    assert passed
    return {'artificial_only':True,'identity_knots_maxabs':ident,'scalar_inverse_maxabs':inverse,
        'orthogonal_identity_maxabs':identity_layer,'orthogonal_inverse_maxabs':inverse_layer,
        'tail_boundary_continuity_error_at_1e_9':continuity,'tail_slopes_one':slopes_one,
        'strict_monotonicity':True,'pass':passed}
