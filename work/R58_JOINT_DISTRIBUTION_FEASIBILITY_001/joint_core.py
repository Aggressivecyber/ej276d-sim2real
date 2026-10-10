import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import rankdata,wasserstein_distance

CATEGORIES=[-1,*range(8),8]

def midrank(x):
    return (rankdata(np.asarray(x),method='average')-.5)/len(x)

def weighted_midrank(x,w):
    order=np.argsort(x,kind='stable');sx=np.asarray(x)[order];ww=np.asarray(w)[order]
    starts=np.r_[0,np.flatnonzero(sx[1:]!=sx[:-1])+1];mass=np.add.reduceat(ww,starts);end=np.cumsum(mass)
    out=np.empty(len(x));out[order]=np.repeat((end-.5*mass)/end[-1],np.diff(np.r_[starts,len(x)]));return out

def quantile_atoms(psd,K):
    """All REAL parents contribute exact integer mass K; each segment has N units."""
    psd=np.asarray(psd,dtype='f8');N=len(psd);order=np.lexsort((np.arange(N),psd));i=j=0;left_real=K;left_atom=N
    src=[];dst=[];units=[]
    while i<N and j<K:
        mass=min(left_real,left_atom);src.append(int(order[i]));dst.append(j);units.append(mass);left_real-=mass;left_atom-=mass
        if left_real==0:i+=1;left_real=K
        if left_atom==0:j+=1;left_atom=N
    src=np.asarray(src,dtype='i8');dst=np.asarray(dst,dtype='i8');units=np.asarray(units,dtype='i8')
    assert i==N and j==K and np.all(np.bincount(src,weights=units,minlength=N)==K) and np.all(np.bincount(dst,weights=units,minlength=K)==N)
    raw=np.bincount(dst,weights=units*psd[src],minlength=K)/N
    values,inv=np.unique(raw,return_inverse=True);weights=np.bincount(inv,minlength=len(values)).astype('f8')/K
    return {'values':values,'weights':weights,'unmerged_values':raw,'unmerged_to_merged':inv,'REAL_index':src,'quantile_index':dst,'integer_mass':units,'global_denominator':N*K}

def source_groups(source_psd,source_bins,h,rank_groups):
    ranks=midrank(source_psd);rank_bin=np.minimum(np.floor(rank_groups*ranks).astype('i8'),rank_groups-1)
    key=np.stack([source_bins,(np.asarray(h)==1).astype('i8'),rank_bin],axis=1)
    groups,index,counts=np.unique(key,axis=0,return_inverse=True,return_counts=True)
    means=np.bincount(index,weights=ranks,minlength=len(groups))/counts
    return {'keys':groups,'index':index,'counts':counts,'parent_ranks':ranks,'rank_means':means}

def cdf_intervals(atoms,reference):
    """Exact W1 interval geometry; candidate CDF is constant between its atoms."""
    rr=np.sort(np.asarray(reference,dtype='f8'));grid=np.unique(np.r_[atoms,rr]);left=grid[:-1];width=np.diff(grid)
    return {'left':left,'width':width,'atom_CDF_index':np.searchsorted(atoms,left,side='right')-1,'reference_CDF':np.searchsorted(rr,left,side='right')/len(rr)}

def interval_w1(atoms,weights,reference):
    g=cdf_intervals(atoms,reference);cum=np.r_[0,np.cumsum(weights)/np.sum(weights)]
    return float(np.dot(g['width'],np.abs(cum[g['atom_CDF_index']+1]-g['reference_CDF'])))

class Rows:
    def __init__(self):self.rows=[];self.cols=[];self.data=[];self.rhs=[]
    def add(self,cols,values,rhs):
        cols=np.asarray(cols,dtype='i8').reshape(-1);values=np.broadcast_to(values,cols.shape).astype('f8')
        self.rows.extend([len(self.rhs)]*len(cols));self.cols.extend(cols.tolist());self.data.extend(values.tolist());self.rhs.append(float(rhs))
    def finish(self,n):return coo_matrix((self.data,(self.rows,self.cols)),shape=(len(self.rhs),n)).tocsr(),np.asarray(self.rhs,dtype='f8')

def build(groups,atoms,atom_weights,source_psd,source_bins,h,real_psd,real_bins,valley,C):
    keys=groups['keys'];counts=groups['counts'];G=len(keys);K=len(atoms);N=len(source_psd);nx=G*K;nf=7*(K-1)
    assert K>=2
    eq=Rows();ub=Rows();xid=np.arange(nx).reshape(G,K)
    for g in range(G):eq.add(xid[g],1,counts[g])
    for k in range(K):eq.add(xid[:,k],1,N*atom_weights[k])
    interval_rows=[];constant_distances=[];bin_sizes=[]
    for b in range(7):
        gm=keys[:,0]==b;nb=int(counts[gm].sum());reference=real_psd[real_bins==b];assert nb>0 and len(reference)>0
        bin_sizes.append(nb);fids=nx+b*(K-1)+np.arange(K-1)
        for k in range(K-1):
            cols=np.r_[xid[gm,k],fids[k],fids[k-1] if k else np.array([],dtype='i8')]
            values=np.r_[np.full(gm.sum(),-1/nb),1.,-1. if k else np.array([],dtype='f8')];eq.add(cols,values,0.)
        intervals=cdf_intervals(atoms,reference);constant=0.
        for j,width,reference_cdf,left in zip(intervals['atom_CDF_index'],intervals['width'],intervals['reference_CDF'],intervals['left']):
            if j<0:constant+=width*reference_cdf
            elif j>=K-1:constant+=width*(1-reference_cdf)
            else:interval_rows.append((b,int(j),float(width),float(reference_cdf),float(left)))
        constant_distances.append(constant)
    nz=len(interval_rows);n=nx+nf+nz;wid=[]
    for i,(b,k,width,reference_cdf,left) in enumerate(interval_rows):
        fid=nx+b*(K-1)+k;zid=nx+nf+i
        ub.add([fid,zid],[1.,-1.],reference_cdf);ub.add([fid,zid],[-1.,-1.],-reference_cdf);wid.append(width)
    distance_row=len(ub.rhs);ub.add(nx+nf+np.arange(nz),wid,7*C['PSD_conditional_mean_max']-sum(constant_distances))
    valley_rules=[]
    for b in [2,3]:
        de=valley['bins'][b];assert de['valid'];parents=(source_bins==b)&(h==1);nh=int(parents.sum());assert nh>0
        original=float(np.mean((source_psd[parents]>=de['low'])&(source_psd[parents]<=de['high'])))
        ga=(keys[:,0]==b)&(keys[:,1]==1);ka=(atoms>=de['low'])&(atoms<=de['high']);limit=original+C['source_valley_delta_max']
        ub.add(xid[np.ix_(ga,ka)].reshape(-1),1,nh*limit)
        valley_rules.append({'raw_bin0':b,'source_parents':nh,'original_fraction':original,'allowed_fraction':limit,'low':de['low'],'high':de['high']})
    ay=weighted_midrank(atoms,atom_weights);cost=np.zeros(n);cost[:nx]=-((groups['rank_means']-.5)[:,None]*(ay-.5)[None,:]).reshape(-1)
    E,d=eq.finish(n);A,b=ub.finish(n);bounds=[(0.,None)]*nx+[(0.,1.)]*(nf+nz)
    description={'SOURCE_groups':G,'target_atoms':K,'x_variables':nx,'CDF_variables':nf,'absolute_distance_variables':nz,'variables':n,'equalities':len(d),'inequalities':len(b),'equality_nnz':E.nnz,'inequality_nnz':A.nnz,'constant_W1_by_Q1_Q7':constant_distances,'conditional_distance_constraint_row':distance_row,'valley_rules':valley_rules}
    return {'objective':cost,'E':E,'d':d,'A':A,'b':b,'bounds':bounds,'description':description,'interval_rows':np.asarray(interval_rows),'target_ranks':ay}

def adc_interval_bad(q,p,baseline_low,baseline_high):
    s=np.asarray(q)*(1-np.asarray(p));l=np.asarray(q)*np.asarray(p)
    low=np.maximum(np.maximum((-25-s)/50,(-125-l)/250),baseline_low)
    high=np.minimum(np.minimum((25-s)/50,(125-l)/250),baseline_high)
    return low>high,low-high

def artificial_check():
    original=np.asarray([-1.,-.1,.1,.4,1.]);a=quantile_atoms(original,3)
    assert sum(a['integer_mass'])==15 and abs(np.dot(a['values'],a['weights'])-original.mean())<1e-15
    xx=np.asarray([-.2,.4,.8]);ww=np.asarray([.2,.5,.3]);rr=np.asarray([.1,.2,.4,.9])
    exact=wasserstein_distance(xx,rr,u_weights=ww);assembled=interval_w1(xx,ww,rr);assert abs(exact-assembled)<1e-14
    bad,_=adc_interval_bad(np.asarray([10.,150.,150.,300.]),np.asarray([.2,5/6,.5,5/6]),np.asarray([0.,0.,0.,-.5]),np.asarray([0.,0.,0.,-.5]))
    assert bad.tolist()==[False,False,True,False]
    sx=np.asarray([0.,0.,1.]);gy=np.asarray([.1,.9]);law=np.asarray([[1.5,.5],[0.,1.]])
    rx=midrank(sx);ry=weighted_midrank(gy,law.sum(0)/3);mu=np.asarray([rx[:2].mean(),rx[2]])
    cov=float(np.sum(law*(mu-.5)[:,None]*(ry-.5)[None,:])/3)
    expanded=np.asarray([law[0]/2,law[0]/2,law[1]])/3
    independent=float(np.sum(expanded*(rx-.5)[:,None]*(ry-.5)[None,:]));assert abs(cov-independent)<1e-15
    # Construct a known feasible joint law in all seven constrained Q bins, without solving another LP.
    toy_psd=np.tile(np.asarray([.1,.3]),7);toy_bins=np.repeat(np.arange(7),2);toy_h=np.tile(np.asarray([0.,1.]),7)
    toy_groups=source_groups(toy_psd,toy_bins,toy_h,4);toy_atoms=np.asarray([.1,.3]);toy_valley={'bins':[{'valid':True,'low':.2,'high':.25} for _ in range(8)]}
    toy=build(toy_groups,toy_atoms,np.asarray([.5,.5]),toy_psd,toy_bins,toy_h,toy_psd,toy_bins,toy_valley,{'PSD_conditional_mean_max':.01,'source_valley_delta_max':.02})
    u=np.zeros(toy['description']['variables']);u[:28]=np.tile(np.eye(2).reshape(-1),7);u[28:35]=.5
    assert np.max(np.abs(toy['E']@u-toy['d']))<1e-14 and np.max(toy['A']@u-toy['b'])<1e-14
    assert abs(float(toy['objective']@u)+.875)<1e-14
    return {'artificial_only':True,'scientific_data_read':False,'extra_LP_solves':0,'integer_atom_conservation':True,'exact_interval_W1':assembled,'scipy_W1':float(exact),'ADC_known_boundary_cases':bad.tolist(),'group_covariance_equals_parent_expansion':True,'full_seven_bin_sparse_constraints_known_feasible_vector':True}
