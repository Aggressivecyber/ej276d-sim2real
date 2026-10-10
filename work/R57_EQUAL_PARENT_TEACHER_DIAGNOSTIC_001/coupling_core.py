import numpy as np

def integer_coupling(source_Q,source_PSD,REAL_Q,REAL_PSD,groups):
    """Equal-parent conditional marginals using exact positive integer masses."""
    ns,nr=len(source_Q),len(REAL_Q)
    assert ns>0 and nr>0 and groups>0 and ns*nr>=groups
    so=np.lexsort((np.arange(ns),source_Q));ro=np.lexsort((np.arange(nr),REAL_Q))
    sleft=np.arange(ns,dtype='i8')*nr;sright=sleft+nr
    rleft=np.arange(nr,dtype='i8')*ns;rright=rleft+ns
    sources=[];targets=[];masses=[];gids=[]
    for g in range(groups):
        lo=g*(ns*nr)//groups;hi=(g+1)*(ns*nr)//groups
        sm=np.maximum(0,np.minimum(sright,hi)-np.maximum(sleft,lo))
        rm=np.maximum(0,np.minimum(rright,hi)-np.maximum(rleft,lo))
        si=so[sm>0];ri=ro[rm>0];sm=sm[sm>0];rm=rm[rm>0]
        a=np.lexsort((si,source_PSD[si]));b=np.lexsort((ri,REAL_PSD[ri]))
        si=si[a];sm=sm[a].copy();ri=ri[b];rm=rm[b].copy()
        assert int(sm.sum())==int(rm.sum())==hi-lo
        i=j=0
        while i<len(si) and j<len(ri):
            n=min(int(sm[i]),int(rm[j]));assert n>0
            sources.append(int(si[i]));targets.append(int(ri[j]));masses.append(n);gids.append(g)
            sm[i]-=n;rm[j]-=n
            if sm[i]==0:i+=1
            if rm[j]==0:j+=1
        assert i==len(si) and j==len(ri)
    si=np.asarray(sources,dtype='i8');ri=np.asarray(targets,dtype='i8');mass=np.asarray(masses,dtype='i8')
    row=np.zeros(ns,dtype='i8');col=np.zeros(nr,dtype='i8');np.add.at(row,si,mass);np.add.at(col,ri,mass)
    assert np.all(row==nr) and np.all(col==ns) and int(mass.sum())==ns*nr
    return si,ri,mass,np.asarray(gids,dtype='i8'),{'SOURCE_N':ns,'REAL_N':nr,'mass_units':ns*nr,'edges':len(si),'SOURCE_each_units':nr,'REAL_each_units':ns,'integer_margins_exact':True}

def artificial_check():
    records=[]
    for ns,nr,g in [(2,3,2),(7,11,16),(5,5,16),(1,37,16)]:
        sq=np.linspace(1,2,ns);rq=np.linspace(1,3,nr)
        sp=np.arange(ns)[::-1]%3;rp=np.arange(nr)%2
        si,ri,m,groups,receipt=integer_coupling(sq,sp,rq,rp,g)
        assert np.all(m>0) and len(groups)==len(m)
        records.append(receipt)
    return {'artificial_only':True,'four_fixed_cases':records,'data_read':False}
