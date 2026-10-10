from pathlib import Path
from datetime import datetime, timezone
import os, json, time, hashlib, traceback, copy

ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/'config.json').read_text('utf8'))
START=time.monotonic()
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:C['CPU_threads']])
import numpy as np
import torch
import scipy
from scipy.stats import wasserstein_distance
from noise_core import q_np,bins_for,innovation,continuous_logq
from source_flow_core import FlowNetwork,TangentField,q_torch,solve
from teacher_core import short_q,endpoint
from canonical import canonical

COUNTER={'updates':0,'SOURCE_path_rows':0,'REAL_identity_path_rows':0,'SOURCE_ODE_rows':0,'field_forward_rows':0,'teacher_rows':0}
FKEYS=['PSD','logQ','baseline_RMS_V','baseline_ACF1','peak_V','Q_Vns']
PRIVATE={}

def utc():return datetime.now(timezone.utc).isoformat()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def progress(phase):
    r={'UTC':utc(),'PID':os.getpid(),'phase':phase,'counter':dict(COUNTER),'elapsed_s':time.monotonic()-START}
    write('PROGRESS.json',r);print(json.dumps(r),flush=True)
def load(n,keys=None):
    with np.load(C['input_paths'][n],allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def gpu_check():
    if max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>C['GPU_limit_bytes']:raise MemoryError('registered GPU allocated/reserved cap')
def stats(x):
    x=np.asarray(x,dtype='f8');ok=np.isfinite(x);v=x[ok]
    return {'N':int(x.size),'finite_N':int(ok.sum()),'undefined_N':int((~ok).sum()),'mean':float(v.mean()) if len(v) else None,'SD':float(v.std(ddof=1)) if len(v)>1 else None}
def desc(x,edges):
    v=np.asarray(x,dtype='f8');q=q_np(v);short=short_q(v);good=np.isfinite(v).all(1)&np.isfinite(q)&(q>0)
    p=np.full(len(v),np.nan);l=p.copy();p[good]=1-short[good]/q[good];l[good]=np.log(1000*q[good])
    e=v[:,:100]-v[:,:100].mean(1,keepdims=True);en=(e*e).sum(1)
    acf=np.divide((e[:,1:]*e[:,:-1]).sum(1),en,out=np.full(len(v),np.nan),where=en>0)
    peak=(v-v[:,:100].mean(1,keepdims=True)).max(1)
    raw=np.full(len(v),-99,dtype='i8');raw[good]=np.searchsorted(edges,l[good],side='right')-1;raw[l==edges[-1]]=7
    return {'PSD':p,'logQ':l,'baseline_RMS_V':np.sqrt(en/99),'baseline_ACF1':acf,'peak_V':peak,'Q_Vns':q,
        'rawbin':raw,'valid':good,'range_exceeded':np.any((v<-.5)|(v>.5),axis=1)}
def slice_desc(f,ix):return {k:v[ix] for k,v in f.items()}
def w1(a,b):
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():return None
    return float(wasserstein_distance(a,b))
def comparisons(a,b,abin=None):
    bins=a['rawbin'] if abin is None else np.asarray(abin)
    rows=[]
    for k in [None,*range(8),'outside']:
        am=np.ones(len(bins),bool) if k is None else ((bins<0)|(bins>7) if k=='outside' else bins==k)
        bm=np.ones(len(b['rawbin']),bool) if k is None else ((b['rawbin']<0)|(b['rawbin']>7) if k=='outside' else b['rawbin']==k)
        row={'bin':None if k is None else (k+1 if isinstance(k,int) else k),'N':int(am.sum()),'reference_N':int(bm.sum()),'invalid_N':int((am&~a['valid']).sum()),'reference_invalid_N':int((bm&~b['valid']).sum()),'features':{}}
        for f in FKEYS:row['features'][f]={'candidate':stats(a[f][am]),'reference':stats(b[f][bm]),'W1':w1(a[f][am],b[f][bm])}
        rows.append(row)
    return rows
def keep(prefix,f):
    for k,v in f.items():PRIVATE[prefix+'__'+k]=v
def wave_zero():return {'n':np.zeros(9,dtype='i8'),'sumV':np.zeros((9,480)),'sumV2':np.zeros((9,480)),'sumU':np.zeros((9,480)),'sumU2':np.zeros((9,480))}
def wave_add(w,x,b):
    x=np.asarray(x,dtype='f8');v=x-x[:,:100].mean(1,keepdims=True);u=v/np.maximum(v.max(1,keepdims=True),1e-12)
    for j in range(9):
        ix=np.ones(len(x),bool) if j==8 else b==j
        if not ix.any():continue
        w['n'][j]+=int(ix.sum())
        for name,a in [('V',x),('U',u)]:w['sum'+name][j]+=a[ix].sum(0);w['sum'+name+'2'][j]+=(a[ix]*a[ix]).sum(0)
def wave_finish(w):
    out={'N':w['n'].copy()}
    for name in ['V','U']:
        n=w['n'][:,None];m=np.divide(w['sum'+name],n,out=np.zeros_like(w['sum'+name]),where=n>0)
        var=np.divide(w['sum'+name+'2']-n*m*m,n-1,out=np.zeros_like(m),where=n>1)
        out[name+'_mean']=m;out[name+'_SD']=np.sqrt(np.maximum(var,0))
    return out
def wave_one(x,b):
    w=wave_zero();wave_add(w,x,b);return wave_finish(w)
def wave_comparison(a,b):
    return [{'bin':i+1 if i<8 else None,'candidate_N':int(a['N'][i]),'reference_N':int(b['N'][i]),
        **{k+'_difference_RMS':float(np.sqrt(np.mean((a[k][i]-b[k][i])**2))) if a['N'][i] and b['N'][i] else None for k in ['V_mean','V_SD','U_mean','U_SD']}} for i in range(9)]
def save_wave(prefix,w):
    for k,v in w.items():PRIVATE[prefix+'__wave_'+k]=v
def short_t(x):return x[...,100:150].sum(-1)-.5*x[...,:100].sum(-1)
DT=np.dtype([(k,'f8') for k in ['t','target_PSD_velocity','predicted_PSD_velocity','signed_PSD_residual','MSE480','total_energy','PSD_mode_energy','path_q_residual','field_q_residual']]+[(k,'i8') for k in ['bin0','time_bin0','source_index','matched_target_index']])
def aggregate(x):
    if not len(x):return {'N':0}
    total=float(x['total_energy'].sum());mode=float(x['PSD_mode_energy'].sum());B=float(x['signed_PSD_residual'].mean());D=float(np.abs(x['target_PSD_velocity']).mean())
    return {'N':len(x),'B_signed_mean':B,'D_target_abs_mean':D,'abs_B_over_D':abs(B)/D if D else None,
        'PSD_RMSE':float(np.sqrt(np.mean(x['signed_PSD_residual']**2))),'target_velocity_mean':float(x['target_PSD_velocity'].mean()),'predicted_velocity_mean':float(x['predicted_PSD_velocity'].mean()),
        'MSE480':float(x['MSE480'].mean()),'total_energy':total,'PSD_mode_energy':mode,'PSD_mode_energy_fraction':mode/total if total else None,
        'path_q_residual_max':float(x['path_q_residual'].max()),'field_q_residual_max':float(x['field_q_residual'].max())}
def path_groups(x):return {'overall':aggregate(x),'by_Q':[{'bin':b+1,**aggregate(x[x['bin0']==b])} for b in range(8)],'by_Q_time':[{'bin':b+1,'time_bin':t,**aggregate(x[(x['bin0']==b)&(x['time_bin0']==t)])} for b in range(8) for t in range(4)]}

@torch.no_grad()
def main():
    torch.set_num_threads(C['CPU_threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available();dev=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    binding=json.loads((ROOT/'INPUT_BINDING.json').read_text())['inputs']
    for k,p in C['input_paths'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==binding[k]['sha256'],k
    environment={'UTC':utc(),'PID':os.getpid(),'python':os.sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'device':str(dev),'GPU':torch.cuda.get_device_name(0),'CPU_threads':torch.get_num_threads(),'affinity':sorted(os.sched_getaffinity(0))}
    write('ENVIRONMENT.json',environment)
    assert all(environment[k]==v for k,v in C['expected_environment'].items()),'frozen model environment drift'
    tr=load('TRAIN_path',['source','target','source_ids','source_bin','target_bin','h','lambdas','edges']);source=tr['source'];real=tr['target']
    assert source.shape==(6593,480) and real.shape==(21047,480)
    co=load('constants_path');mu=co['mu_unit_q'];scale=co['scale_unit_q'];edges=co['edges_logQ'];assert np.array_equal(edges,tr['edges'])
    noise=load('noise_calibration_path');B=load('basis')['B']
    orig=load('original_stream',['source','matched_target','bin','original_row_ordinal']);si=orig['source'].astype('i8');oldmi=orig['matched_target'].astype('i8');sb,sraw,sq=bins_for(source,edges);rb,rraw,rq=bins_for(real,edges)
    assert len(si)==94400 and np.array_equal(sb[si],orig['bin'])
    mapfull=load('full_coupling_path',['source','matched_target_joint','matched_target_original','original_row_ordinal'])
    mappart=load('partial_coupling_path',['source','matched_target_joint','matched_target_original','original_row_ordinal'])
    maps={'full_cycle':mapfull,'final_partial_cycle':mappart}
    counts=16*np.bincount(si,minlength=len(source))+np.bincount(si[:89600],minlength=len(source));seen=counts>0
    rc=16*np.bincount(oldmi,minlength=len(real))+np.bincount(oldmi[:89600],minlength=len(real));rseen=rc>0
    training_receipt=json.loads(Path(C['input_paths']['original_training_receipt']).read_text())
    assert [int(seen.sum()),int(rseen.sum())]==training_receipt['actual_unique_SOURCE_REAL']
    assert int(counts.sum())==int(rc.sum())==1600000 and np.array_equal(sraw[si],orig['bin'])
    cat=np.where(tr['h']==0,'h_zero',np.where(tr['h']==1,'h_one','h_between'))
    groups=[('all',np.ones(len(source),bool)),('seen',seen),('unseen',~seen),('inside',(sraw>=0)&(sraw<8)),('outside',(sraw<0)|(sraw>7))]+[(str(k),cat==k) for k in np.unique(cat)]
    coverage={'SOURCE_total':len(source),'REAL_total':len(real),'SOURCE_seen':int(seen.sum()),'REAL_seen':int(rseen.sum()),'SOURCE_total_occurrences':int(counts.sum()),'REAL_total_occurrences':int(rc.sum()),
        'source_rawbin_vs_saved_mismatch':int((sraw!=tr['source_bin']).sum()),'REAL_rawbin_vs_saved_mismatch':int((rraw!=tr['target_bin']).sum()),'frozen_lambdas':tr['lambdas'].tolist(),
        'source_sampler':'R23 equal bin probability; within source bin exp(lambda*h); REAL uniform within bin; all later streams reused',
        'source_composition_proxy':'h=0, h=1, 0<h<1 only; not particle truth','groups':[],'by_raw_bin':[]}
    for name,ix in groups:coverage['groups'].append({'group':name,'parents':int(ix.sum()),'seen':int((ix&seen).sum()),'unseen':int((ix&~seen).sum()),'occurrences':int(counts[ix].sum())})
    for b in [-1,*range(8),8]:
        ix=sraw==b;ri=rraw==b
        coverage['by_raw_bin'].append({'raw_bin0':b,'SOURCE_parents':int(ix.sum()),'SOURCE_seen':int((ix&seen).sum()),'SOURCE_occurrences':int(counts[ix].sum()),'REAL_parents':int(ri.sum()),'REAL_seen':int((ri&rseen).sum()),'REAL_occurrences':int(rc[ri].sum())})
    write('COVERAGE.json',coverage)
    PRIVATE.update(SOURCE_ids=tr['source_ids'],SOURCE_exposure_count=counts,REAL_exposure_count=rc,SOURCE_h=tr['h'])
    sf=desc(source,edges);rf=desc(real,edges);rcan=desc(canonical(real),edges);keep('SOURCE',sf);keep('REAL',rf);keep('REAL_CANONICAL',rcan)
    wr=wave_one(real,rraw);ws=wave_one(source,sraw);save_wave('REAL',wr);save_wave('SOURCE',ws)
    teachers={};teacherfeatures={};teacherwaves={}
    for name,m in maps.items():
        n=94400 if name=='full_cycle' else 89600;ss=si[:n];mi=m['matched_target_joint'].astype('i8')
        assert np.array_equal(m['source'],ss) and np.array_equal(m['matched_target_original'],oldmi[:n]) and np.array_equal(m['original_row_ordinal'],orig['original_row_ordinal'][:n])
        assert np.array_equal(np.sort(mi),np.sort(oldmi[:n])) and np.array_equal(sb[ss],rb[mi])
        allf={};allc={};w=wave_zero();maxq=maxp=0.;baseline=True
        for lo in range(0,n,1024):
            sl=slice(lo,min(lo+1024,n));r=real[mi[sl]];t=endpoint(r,sq[ss[sl]],B);f=desc(t,edges);fc=desc(canonical(t),edges)
            for k,v in f.items():allf.setdefault(k,[]).append(v)
            for k,v in fc.items():allc.setdefault(k,[]).append(v)
            maxq=max(maxq,float(np.max(np.abs(f['Q_Vns']-sq[ss[sl]])/sq[ss[sl]])))
            maxp=max(maxp,float(np.max(np.abs(f['PSD']-rf['PSD'][mi[sl]]))))
            baseline &= np.array_equal(t[:,:100],r[:,:100]) and np.array_equal(t[:,400:],r[:,400:])
            wave_add(w,t,sraw[ss[sl]]);COUNTER['teacher_rows']+=len(t)
        tf={k:np.concatenate(v) for k,v in allf.items()};tc={k:np.concatenate(v) for k,v in allc.items()};teacherfeatures[name]=tf;teacherwaves[name]=wave_finish(w)
        keep(name+'_TEACHER',tf);keep(name+'_TEACHER_CANONICAL',tc);save_wave(name+'_TEACHER',teacherwaves[name])
        assert maxq<1e-10 and maxp<1e-10 and baseline
        teachers[name]={'occurrences':n,'relative_q_error_max':maxq,'PSD_change_max':maxp,'baseline_tail_exact':baseline,
            'matched_original_vs_REAL':comparisons(slice_desc(rf,oldmi[:n]),rf),'matched_joint_vs_REAL':comparisons(slice_desc(rf,mi),rf),
            'teacher_vs_REAL_own_bins':comparisons(tf,rf),'teacher_vs_REAL_source_bins':comparisons(tf,rf,sraw[ss]),
            'canonical_teacher_vs_canonical_REAL':comparisons(tc,rcan),'waveform_vs_REAL':wave_comparison(teacherwaves[name],wr)}
        progress('teacher_'+name)
    write('TEACHER_POPULATIONS.json',teachers)
    network=FlowNetwork().to(dev);network.load_state_dict(torch.load(C['input_paths']['weights'],map_location=dev,weights_only=True));network.eval();network.requires_grad_(False)
    mt=torch.as_tensor(mu,dtype=torch.float64,device=dev);st=torch.as_tensor(scale,dtype=torch.float64,device=dev);Bt=torch.as_tensor(B,dtype=torch.float64,device=dev)
    uS=torch.as_tensor(source.astype('f8')/sq[:,None],device=dev);rR=torch.as_tensor(real.astype('f8'),device=dev);uR=torch.as_tensor(real.astype('f8')/rq[:,None],device=dev)
    rqt=torch.as_tensor(rq,device=dev);rs=torch.as_tensor(short_q(real),device=dev)
    sh=continuous_logq(sq,sb,edges);rh=continuous_logq(rq,rb,edges);targetmi=mapfull['matched_target_joint'].astype('i8')
    streams={k:np.random.Generator(np.random.PCG64(seed)) for k,seed in C['PCG64_seeds'].items()}
    initial=json.loads(Path(C['input_paths']['random_initial']).read_text())
    assert all(g.bit_generator.state==initial[k] for k,g in streams.items())
    rowsS=np.empty(94400,dtype=DT);rowsI=np.empty(94400,dtype=DT);hashes={k:hashlib.sha256() for k in streams}
    qvec=np.zeros(480);qvec[:100]=-3.;qvec[100:400]=1.
    psvec=np.zeros(480);psvec[:100]=.5;psvec[100:150]=-1.
    aq=scale*qvec;d=scale*psvec;d=d-aq*(np.sum(aq*d,axis=1)/np.sum(aq*aq,axis=1))[:,None];dt=torch.as_tensor(d,device=dev);dn=torch.as_tensor(np.sum(d*d,axis=1),device=dev)
    for step in range(2950):
        sl=slice(step*32,(step+1)*32);ss=si[sl];ri=oldmi[sl];rt=targetmi[sl]
        ts=streams['source_t'].uniform(0,1,32).astype('f4');es=streams['source_epsilon'].standard_normal((32,480)).astype('f4')
        ti=streams['identity_t'].uniform(0,1,32).astype('f4');ei=streams['identity_epsilon'].standard_normal((32,480)).astype('f4')
        for k,v in [('source_t',ts),('source_epsilon',es),('identity_t',ti),('identity_epsilon',ei)]:hashes[k].update(v.tobytes())
        ns,_=innovation(source[ss],es,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        ni,_=innovation(real[ri],ei,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        bs=torch.as_tensor(sb[ss],device=dev);br=torch.as_tensor(rb[ri],device=dev);sqt=torch.as_tensor(sq[ss],device=dev)
        gs=(uS[ss]-mt[bs])/st[bs];gr=(uR[ri]-mt[br])/st[br]
        aS=gs+torch.as_tensor(ns/sq[ss,None],device=dev)/st[bs];aR=gr+torch.as_tensor(ni/rq[ri,None],device=dev)/st[br]
        alpha=sqt/rqt[rt]-1;coeff=torch.stack((alpha*rs[rt],alpha*(rqt[rt]-rs[rt])),1);target=rR[rt]+coeff@Bt.T;cS=(target/sqt[:,None]-mt[bs])/st[bs]
        tt=torch.as_tensor(np.concatenate((ts,ti)),dtype=torch.float64,device=dev);a=torch.cat((aS,aR));c=torch.cat((cS,gr));guide=torch.cat((gs,gr));bt=torch.cat((bs,br));ht=torch.as_tensor(np.concatenate((sh[ss],rh[ri])),device=dev)
        y=(1-tt[:,None])*a+tt[:,None]*c;v=c-a;pred=TangentField(network,mt,st,guide,ht)(y,tt,bt);res=pred-v
        tv=-short_t(st[bt]*v);pv=-short_t(st[bt]*pred);total=res.square().sum(1);mode=(res*dt[bt]).sum(1).square()/dn[bt]
        assert torch.isfinite(pred).all() and torch.all(mode<=total+1e-6)
        values={'t':tt,'target_PSD_velocity':tv,'predicted_PSD_velocity':pv,'signed_PSD_residual':pv-tv,'MSE480':total/480,'total_energy':total,'PSD_mode_energy':mode,'path_q_residual':q_torch(st[bt]*v).abs(),'field_q_residual':q_torch(st[bt]*pred).abs()}
        values={k:t.cpu().numpy() for k,t in values.items()}
        for branch,row,offset,indices,targetindices,bins,times in [('SOURCE',rowsS,0,ss,rt,sb[ss],ts),('REAL',rowsI,32,ri,ri,rb[ri],ti)]:
            for k,vv in values.items():row[k][sl]=vv[offset:offset+32]
            row['bin0'][sl]=bins;row['time_bin0'][sl]=np.minimum((times*4).astype('i8'),3);row['source_index'][sl]=indices;row['matched_target_index'][sl]=targetindices
        COUNTER['SOURCE_path_rows']+=32;COUNTER['REAL_identity_path_rows']+=32;COUNTER['field_forward_rows']+=64
        if (step+1)%500==0 or step==2949:gpu_check();progress('frozen_TRAIN_path')
    pathsummary={'SOURCE':path_groups(rowsS),'REAL_identity':path_groups(rowsI),'first_cycle_random_hashes':{k:v.hexdigest() for k,v in hashes.items()},'trained_path_not_independent_validation':True}
    np.savez_compressed(ROOT/'PATH_ROWS_PRIVATE.npz',SOURCE=rowsS,REAL_identity=rowsI)
    write('PATH_SUMMARY.json',pathsummary)
    rng=np.random.Generator(np.random.PCG64(C['endpoint_seed']));eps=rng.standard_normal((6593,480)).astype('f4');out=np.empty_like(source);detail={};maxq32=maxq64=0.
    for lo in range(0,len(source),64):
        sl=slice(lo,min(lo+64,len(source)));v,m,receipt=solve(network,source[sl],eps[sl],mu,scale,edges,noise,True,128,64)
        out[sl]=v
        for k,x in m.items():detail.setdefault(k,[]).append(x)
        maxq32=max(maxq32,receipt['max_relative_q_error_f32']);maxq64=max(maxq64,receipt['max_relative_q_error_f64'])
        COUNTER['SOURCE_ODE_rows']+=len(v);COUNTER['field_forward_rows']+=receipt['field_forward_rows']
        if lo%512==0 or lo+64>=len(source):gpu_check();progress('frozen_TRAIN_Heun128')
    detail={k:np.concatenate(v) for k,v in detail.items()}
    assert COUNTER['SOURCE_ODE_rows']==6593 and COUNTER['SOURCE_path_rows']==COUNTER['REAL_identity_path_rows']==94400
    assert COUNTER['field_forward_rows']==188800+6593*256 and np.all(detail['all_rows_new_flow'] if 'all_rows_new_flow' in detail else detail['neural_flow_applied_all_rows'])
    np.savez_compressed(ROOT/'TRAIN_ENDPOINT_PRIVATE.npz',preADC_V=out,source_ids=tr['source_ids'],**detail)
    of=desc(out,edges);oc=desc(canonical(out),edges);keep('R54_PRE',of);keep('R54_CANONICAL',oc)
    wo=wave_one(out,sraw);save_wave('R54_PRE',wo)
    outputgroups={}
    for name,ix in groups:
        sub=slice_desc(of,ix);post=slice_desc(oc,ix)
        outputgroups[name]={'parents':int(ix.sum()),'range_exceeded_preADC':int(of['range_exceeded'][ix].sum()),'invalid_preADC':int((~of['valid'][ix]).sum()),
            'preADC_vs_REAL_source_bins':comparisons(sub,rf,sraw[ix]),'preADC_vs_REAL_own_bins':comparisons(sub,rf),
            'canonical_vs_canonical_REAL_own_bins':comparisons(post,rcan)}
    exposure={};markers=[]
    for name,tf in teacherfeatures.items():
        n=94400 if name=='full_cycle' else 89600;ss=si[:n];pop=slice_desc(of,ss)
        exposure[name]={'not_independent_occurrences':True,'source_unique':len(np.unique(ss)),'against_teacher_source_bins':comparisons(pop,tf,sraw[ss]),'against_REAL_source_bins':comparisons(pop,rf,sraw[ss])}
    for b in range(1,7):
        teacherW=teachers['full_cycle']['teacher_vs_REAL_source_bins'][b+1]['features']['PSD']['W1'];mw=exposure['full_cycle']['against_teacher_source_bins'][b+1]['features']['PSD']['W1'];rw=exposure['full_cycle']['against_REAL_source_bins'][b+1]['features']['PSD']['W1']
        use=sb[si]==b;ps=pathsummary['SOURCE']['by_Q'][b];G=float(of['PSD'][si[use]].mean()-teacherfeatures['full_cycle']['PSD'][use].mean());ratio=ps['abs_B_over_D'];Bmean=ps['B_signed_mean']
        valid=all(x is not None for x in [teacherW,mw,rw,ratio]);hit=bool(valid and teacherW<=.01 and mw>.01 and rw>.01 and ratio>=.2 and Bmean*G>0)
        markers.append({'bin':b+1,'teacher_REAL_W1':teacherW,'R54_teacher_W1':mw,'R54_REAL_W1':rw,'B':Bmean,'D':ps['D_target_abs_mean'],'G':G,'abs_B_over_D':ratio,'same_sign':bool(Bmean*G>0),'learning_marker':hit})
    nlearn=sum(x['learning_marker'] for x in markers);nbias=sum(x['teacher_REAL_W1'] is not None and x['teacher_REAL_W1']>.01 for x in markers)
    decision='learning_marker_supports_new_PSD_tangent_mechanism' if nlearn>=4 else 'teacher_population_requires_attention' if nbias>=4 else 'fixed_direction_rule_inconclusive_review_full_evidence'
    result={'UTC':utc(),'status':'complete','decision':decision,'learning_markers':nlearn,'teacher_bias_markers':nbias,'markers':markers,'counter':COUNTER,'SOURCE_seen':int(seen.sum()),'SOURCE_unseen':int((~seen).sum()),
        'output_groups':outputgroups,'output_exposure_comparisons':exposure,'waveform_vs_REAL':wave_comparison(wo,wr),'q_relative_error_f64_max':maxq64,'q_relative_error_f32_max':maxq32,
        'original_A_F_unchanged':True,'model_goal_complete':False,'new_response_models':0,'DEV_read':0,'EVAL_read':0,'CAL_decode':0,'only_TRAIN_package_keys_decoded':['source','target','source_ids','source_bin','target_bin','h','lambdas','edges'],
        'endpoint_random_seed':C['endpoint_seed'],'endpoint_epsilon_sha256':hashlib.sha256(eps.tobytes()).hexdigest(),'model_weights_unchanged':hashlib.sha256(Path(C['input_paths']['weights']).read_bytes()).hexdigest()==binding['weights']['sha256']}
    write('RESULT.json',result);np.savez_compressed(ROOT/'FEATURES_PRIVATE.npz',**PRIVATE)
    from render_diagnostic import render
    render(ROOT,coverage,pathsummary,markers,PRIVATE)
    gpu_check();progress('complete')
    write('NUMERIC_RECEIPT.json',{'UTC':utc(),'status':'complete','PID':os.getpid(),'elapsed_s':time.monotonic()-START,'counter':COUNTER,'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'decision':decision,'model_goal_complete':False})

try:main()
except BaseException as exc:
    write('FAILURE.json',{'UTC':utc(),'PID':os.getpid(),'counter':COUNTER,'elapsed_s':time.monotonic()-START,'error':repr(exc),'traceback':traceback.format_exc()});raise
