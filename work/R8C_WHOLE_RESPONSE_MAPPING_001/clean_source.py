import numpy as np
def clean_expectation(ee,had):
    u=np.maximum(np.arange(512)-128.,0)[None,:]
    n=len(ee);out=np.zeros((n,512),dtype=np.float64)
    for energy,taus,frac in [(ee,[13,35,270],[.7,.15,.15]),(.15*had,[13,59,460],[.5,.1,.4])]:
        for tau,weight in zip(taus,frac):out+=energy[:,None]*weight*(np.exp(-u/tau)-np.exp(-u/5))/(tau-5)
    energy=ee+.15*had;q=(energy+.03918)/.000132506/1000
    out*=q[:,None]/out.sum(axis=1,keepdims=True)
    peak=out.argmax(axis=1);threshold=out.max(axis=1)*.5;t=np.arange(512)[None,:]
    below=(t<peak[:,None])&(out<threshold[:,None]);tp=np.where(below,t,-1).max(axis=1)+1
    ix=(tp[:,None]-120+np.arange(480)[None,:]).clip(0,511)
    return np.take_along_axis(out,ix,axis=1).astype('f4'),q.astype('f4')
