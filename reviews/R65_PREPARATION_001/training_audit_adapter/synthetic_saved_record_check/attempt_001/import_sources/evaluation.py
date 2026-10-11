import numpy as np

def summary(v):

 a=np.asarray(v);x=a[np.isfinite(a)]

 return {'n':len(a),'defined_n':len(x),'undefined_n':len(a)-len(x),'mean':float(x.mean()) if len(x) else None,'sample_sd':float(x.std(ddof=1)) if len(x)>1 else None,'quantile_levels':[0,.05,.25,.5,.75,.95,1],'quantiles':np.quantile(x,[0,.05,.25,.5,.75,.95,1]).tolist() if len(x) else None}
