from __future__ import annotations
import os,time,json,hashlib,pathlib,datetime,traceback
ROOT=pathlib.Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text('utf8'));START=time.monotonic()
import numpy as np,torch
from source_flow_core import FlowNetwork,TangentField,initial,q_np,q_torch
from reorder_core import reconstruct_and_reorder
THREADS=int(os.environ.get('R44_THREADS',CFG['CPU_threads']));torch.set_num_threads(THREADS);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
OUT=ROOT/'results';N=94400;attempted=committed=0;calls=0

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(n,v):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n','utf8')
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in(z.files if keys is None else keys)}
def f_np(x):return x[...,100:150].sum(-1)-.5*x[...,:100].sum(-1)
def f_torch(x):return x[...,100:150].sum(-1)-.5*x[...,:100].sum(-1)
def bins(q,edges):
    b=np.searchsorted(edges,np.log(1000*q),side='right')-1;b[np.log(1000*q)==edges[-1]]=7;return b
def summary(x):
    x=np.asarray(x,dtype='f8');v=x[np.isfinite(x)]
    return {'N':len(x),'finite_N':len(v),'undefined_N':len(x)-len(v),'mean':float(v.mean())if len(v)else None,'SD_ddof1':float(v.std(ddof=1))if len(v)>1 else None}
def checkpoint():
    put('PROGRESS.json',{'PID':os.getpid(),'UTC':utc(),'attempted_forward_rows':attempted,'committed_rows':committed,'calls':calls,'elapsed_s':time.monotonic()-START,'prefix_status':'complete'if committed==N else'incomplete','updates':0})
def timed():
    if time.monotonic()-START>CFG['science_budget']-45 or time.time()>datetime.datetime.fromisoformat(CFG['deadline_UTC']).timestamp()-90:raise TimeoutError('preserve committed prefix; no whole-scan rerun')

DT=np.dtype([(k,'<f8')for k in ['t','source_Q_Vns','target_Q_Vns','source_PSD','initial_PSD','target_PSD','target_PSD_velocity','predicted_PSD_velocity','signed_PSD_residual','MSE480','total_energy','PSD_mode_energy','complement_energy','energy_closure_error','PSD_identity_error','path_q_residual','field_q_residual']]+[(k,'<i8')for k in ['source_TRAIN_index','target_original_TRAIN_index','matched_target_TRAIN_index','original_noise_ordinal','bin0','time_bin0']])

def aggregate(rows,choose):
    z=rows[choose];count=len(z)
    if not count:return {'N':0,'signed_mean':None,'PSD_RMSE':None,'target_velocity_mean':None,'target_velocity_abs_mean':None,'PSD_mode_energy_fraction':None}
    total=float(np.sum(z['total_energy'],dtype='f8'));mode=float(np.sum(z['PSD_mode_energy'],dtype='f8'));comp=float(np.sum(z['complement_energy'],dtype='f8'))
    return {'N':count,'signed_mean':float(np.mean(z['signed_PSD_residual'])),'PSD_RMSE':float(np.sqrt(np.mean(z['signed_PSD_residual']**2))),
            'target_velocity_mean':float(np.mean(z['target_PSD_velocity'])),'target_velocity_abs_mean':float(np.mean(np.abs(z['target_PSD_velocity']))),
            'predicted_velocity_mean':float(np.mean(z['predicted_PSD_velocity'])),'MSE480_mean':float(np.mean(z['MSE480'])),'total_energy_sum':total,
            'PSD_mode_energy_sum':mode,'complement_energy_sum':comp,'PSD_mode_energy_fraction':mode/total if total>0 else None,
            'energy_sum_closure_absolute':abs(total-mode-comp)}

def population(source,real,stream,rows,edges):
    old=pathlib.Path(CFG['R43_root'])
    actual=load(old/'ACTUAL_GENERATION_INPUTS.npz',['source_waveforms_V','source_ids'])
    raw=load(old/'REAL_DEV_AND_REFERENCE_PRIVATE.npz',['real_V','real_ids','real_QC_valid'])
    generated=load(old/'SCALAR_TIME_FLOW_FINAL/generated.npz',['pre_ADC_V','waveforms_V','source_ids'])
    assert np.array_equal(actual['source_ids'],generated['source_ids'])
    features=load(old/'results/FEATURES_PRIVATE.npz',['REAL__PSD','SOURCE_INPUT__PSD','SCALAR_TIME_FLOW_FINAL__PSD'])
    sourceq=q_np(actual['source_waveforms_V']);actualb=bins(sourceq,edges)
    datasets={
      'consumed_REAL_targets':(real[stream['matched_target']],stream['bin']),
      'unique_consumed_REAL_targets':(real[np.unique(stream['matched_target'])],None),
      'full_REAL_TRAIN':(real,None),
      'RAW_REAL_DEV':(raw['real_V'],None),
      'R43_output_preADC_sourceQ':(generated['pre_ADC_V'],actualb),
      'R43_output_postADC_sourceQ':(generated['waveforms_V'],actualb),
      'SOURCE_input_preADC_sourceQ':(actual['source_waveforms_V'],actualb)}
    arrays={};result={}
    for name,(x,cond)in datasets.items():
        timed();q=q_np(x);ok=np.isfinite(x).all(1)&np.isfinite(q)&(q>0)
        p=np.full(len(x),np.nan);p[ok]=1-f_np(x[ok].astype('f8')/q[ok,None])
        b=np.full(len(x),-99,'i8');b[ok]=bins(q[ok],edges)if cond is None else np.asarray(cond)[ok]
        arrays[name+'__PSD']=p;arrays[name+'__bin0']=b
        rowsummary=[]
        for k in[None,*range(8),'outside']:
            use=ok if k is None else(ok&((b<0)|(b>7))if k=='outside'else ok&(b==k))
            values=p[use];rowsummary.append({'bin_id':None if k is None else('outside'if k=='outside'else k+1),**summary(values)})
            arrays[name+'__'+str(k)+'__CDF_sorted']=np.sort(values)
        result[name]={'loaded_N':len(x),'nonfinite_or_nonpositive_Q_N':int((~ok).sum()),'population_weight':'one per exposure'if name=='consumed_REAL_targets'else'one per original saved row',
          'Q_condition':'actual NN SOURCE preADC Q, not output-Q'if 'sourceQ'in name else'own original preADC Q','rows':rowsummary}
    result['RAW_REAL_DEV']['saved_QC_valid_N']=int(raw['real_QC_valid'].sum())
    result['RAW_REAL_DEV']['saved_QC_flags_not_used_to_reselect_this_background']=True
    pairs=[('RAW_REAL_DEV','REAL__PSD'),('SOURCE_input_preADC_sourceQ','SOURCE_INPUT__PSD'),('R43_output_postADC_sourceQ','SCALAR_TIME_FLOW_FINAL__PSD')]
    for name,key in pairs:
        a=arrays[name+'__PSD'];b=features[key];good=np.isfinite(a)&np.isfinite(b)
        result[name]['existing_FEATURES_maxabs_PSD_difference']=float(np.max(np.abs(a[good]-b[good])))if good.any()else None
    markers=[]
    for b in range(1,7):
        use=rows['bin0']==b;agg=aggregate(rows,use);candidate=arrays['R43_output_preADC_sourceQ__PSD'][actualb==b];target=arrays['consumed_REAL_targets__PSD'][stream['bin']==b]
        B=agg['signed_mean'];D=agg['target_velocity_abs_mean'];G=float(candidate.mean()-target.mean())if len(candidate)and len(target)else None
        ratio=abs(B)/D if B is not None and D is not None and D>0 else None
        hit=bool(ratio is not None and G is not None and ratio>=.20 and B*G>0)
        markers.append({'bin_id':b+1,'path_N':int(use.sum()),'endpoint_N':len(candidate),'consumed_target_N':len(target),'B_signed_velocity_residual':B,'D_mean_abs_target_velocity':D,'G_endpoint_minus_consumed_PSD_mean':G,'abs_B_over_D':ratio,'same_sign':bool(B*G>0)if B is not None and G is not None else None,'hit':hit})
    np.savez_compressed(OUT/'POPULATION_BACKGROUND_PRIVATE.npz',**arrays)
    put('POPULATION_BACKGROUND.json',result)
    return result,markers

def plot(rows,groups,pops,markers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white','savefig.facecolor':'white'})
    dest=OUT/'figures';dest.mkdir(exist_ok=True);figures={}
    def save(fig,n):
        fig.savefig(dest/n,dpi=180,bbox_inches='tight');plt.close(fig);data=(dest/n).read_bytes();figures[n]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    mats={k:np.array([[groups[b*4+j][k]if groups[b*4+j][k]is not None else np.nan for j in range(4)]for b in range(8)])for k in ['target_velocity_mean','predicted_velocity_mean','signed_mean','PSD_RMSE']}
    fig,axs=plt.subplots(2,2,figsize=(11.8,9));maxv=max(np.nanmax(np.abs(mats[k]))for k in ['target_velocity_mean','predicted_velocity_mean','signed_mean'])
    for ax,(k,a)in zip(axs.flat,mats.items()):
        im=ax.imshow(a,origin='lower',aspect='auto',cmap='viridis'if k=='PSD_RMSE'else'coolwarm',vmin=0 if k=='PSD_RMSE'else-maxv,vmax=np.nanmax(a)if k=='PSD_RMSE'else maxv)
        ax.set(xticks=range(4),xticklabels=['[0,.25)','[.25,.5)','[.5,.75)','[.75,1]'],yticks=range(8),yticklabels=range(1,9),xlabel='Actual saved interpolation t',ylabel='Original Q bin',title=k.replace('_',' '))
        for b in range(8):
            for j in range(4):ax.text(j,b,f'{a[b,j]:.3g}\nn={groups[b*4+j]["N"]}'if np.isfinite(a[b,j])else'undefined',ha='center',va='center',fontsize=7)
        fig.colorbar(im,ax=ax,label='dimensionless PSD velocity / unit t')
    fig.suptitle('Frozen final2950 on actual TRAIN paths; no integrated-trajectory residual claim');fig.tight_layout();save(fig,'PSD_velocity_Q_time.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.7));fractions=np.array([[groups[b*4+j]['PSD_mode_energy_fraction']if groups[b*4+j]['PSD_mode_energy_fraction']is not None else np.nan for j in range(4)]for b in range(8)])
    im=axs[0].imshow(100*fractions,origin='lower',aspect='auto',vmin=0,cmap='viridis');axs[0].set(xticks=range(4),xticklabels=['t0','t1','t2','t3'],yticks=range(8),yticklabels=range(1,9),xlabel='Four fixed time intervals',ylabel='Q bin',title='PSD tangent-direction share of original480 MSE')
    fig.colorbar(im,ax=axs[0],label='% ratio of energy sums')
    qb=[aggregate(rows,rows['bin0']==b)for b in range(8)];pv=np.array([q['PSD_mode_energy_sum']/q['N']/480 for q in qb]);cv=np.array([q['complement_energy_sum']/q['N']/480 for q in qb])
    axs[1].bar(range(1,9),cv,label='Orthogonal complement');axs[1].bar(range(1,9),pv,bottom=cv,label='PSD direction');axs[1].set(xlabel='Original Q bin',ylabel='Standardized480 velocity MSE',title='Geometric energy decomposition; small share ≠ unimportant');axs[1].legend();fig.tight_layout();save(fig,'PSD_MSE_geometry.png')
    z=load(OUT/'POPULATION_BACKGROUND_PRIVATE.npz');names=['consumed_REAL_targets','unique_consumed_REAL_targets','full_REAL_TRAIN','RAW_REAL_DEV','R43_output_preADC_sourceQ','SOURCE_input_preADC_sourceQ'];labels=['Consumed targets (exposures)','Unique targets','Full REAL TRAIN','RAW REAL DEV','R43 preADC (source-Q)','SOURCE input'];colors=['tab:orange','goldenrod','black','tab:green','tab:blue','gray']
    fig,axs=plt.subplots(2,4,figsize=(15,8),sharex=True,sharey=True)
    for b,ax in enumerate(axs.flat):
        for name,label,color in zip(names,labels,colors):
            x=z[name+'__'+str(b)+'__CDF_sorted'];ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=f'{label} n={len(x)}',color=color,alpha=.85)
        ax.set(title=f'Q bin{b+1}',xlabel='signed PSD',ylabel='Complete empirical CDF')
    handles,labels0=axs.flat[0].get_legend_handles_labels();fig.legend(handles,[a.split(' n=')[0]for a in labels0],loc='upper center',ncol=3,bbox_to_anchor=(.5,1.03));fig.suptitle('Populations separate from path residual: REAL own-Q; SOURCE/output actual source-Q',y=1.075);fig.tight_layout();save(fig,'population_PSD_CDF.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for name,label,color in zip(names,labels,colors):
        st=pops[name]['rows'][1:9];axs[0].errorbar(range(1,9),[a['mean']for a in st],yerr=[a['SD_ddof1']for a in st],label=label,color=color,marker='o',capsize=2,lw=1)
    axs[0].set(xlabel='Fixed Q bin',ylabel='Mean signed PSD ± sample SD',title='Different population measures; not a causal subtraction');axs[0].legend(fontsize=7)
    B=np.array([a['B_signed_velocity_residual']for a in markers]);G=np.array([a['G_endpoint_minus_consumed_PSD_mean']for a in markers]);ratio=np.array([a['abs_B_over_D']for a in markers]);ids=np.arange(2,8)
    axs[1].plot(ids,ratio,'o-',label='abs(B)/mean abs target velocity');axs[1].axhline(.2,ls='--',color='gray',label='Predefined .20 direction rule')
    for x,y,m in zip(ids,ratio,markers):axs[1].annotate('HIT'if m['hit']else'no', (x,y),xytext=(0,7),textcoords='offset points',ha='center')
    axs[1].set(xlabel='Original Q bin',ylabel='Dimensionless descriptive ratio',title=f'Direction markers {sum(a["hit"]for a in markers)}/6; sign agreement also required');axs[1].legend(fontsize=7);fig.tight_layout();save(fig,'population_means_direction_markers.png')
    put('PLOT_RECEIPT.json',{'PID':os.getpid(),'UTC':utc(),'figures':figures,'new_fits_generation_updates':0,'actual_visual_QA_pending':True})

def forward():
    global attempted,committed,calls
    try:
        env={'PID':os.getpid(),'UTC':utc(),'python':os.sys.version,'numpy':np.__version__,'torch':torch.__version__,'device':'cpu','CPU_threads':THREADS};put('ENVIRONMENT.json',env)
        train=load(CFG['TRAIN_path'],['source','target']);source=train['source'];real=train['target'];del train
        assert source.shape==(6593,480)and real.shape==(21047,480)
        const=load(CFG['constants_path']);mu=const['mu_unit_q'];scale=const['scale_unit_q'];edges=const['edges_logQ']
        device=torch.device(CFG['execution_device']);assert device.type=='cuda' and torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.cuda.set_per_process_memory_fraction(min(1.,CFG['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory));mt=torch.from_numpy(mu).to(device=device,dtype=torch.float64);st=torch.from_numpy(scale).to(device=device,dtype=torch.float64);env.update(device=str(device),GPU=torch.cuda.get_device_name(0));put('ENVIRONMENT.json',env)
        original=reconstruct_and_reorder(CFG,ROOT);stream=load(CFG['coupling_path']);receipt=json.loads(pathlib.Path(CFG['train_receipt_path']).read_text('utf8'))
        for key in ['source','target','bin','original_row_ordinal']:assert np.array_equal(original[key],stream[key])
        assert all(hashlib.sha256(stream[k].astype('i8'if k!='t'else'f4').tobytes()).hexdigest()==receipt['stream_hashes'][k]for k in ['source','target','bin','original_row_ordinal','t'])
        white=original['original_noise']['white'];z=original['original_noise']['z'];style=original['original_style'];hw=hashlib.sha256();hn=hashlib.sha256();hs=hashlib.sha256()
        for k in range(2950):
            sl=slice(k*32,(k+1)*32);order=stream['original_row_ordinal'][sl];w=white[order];hw.update(w.tobytes());hn.update(z[order].tobytes());hn.update(w.tobytes());hs.update(style[order].tobytes())
        binding={'white':hw.hexdigest()==receipt['stream_hashes']['white'],'z_white':hn.hexdigest()==receipt['stream_hashes']['z_white'],'style_unused':hs.hexdigest()==receipt['stream_hashes']['style']};assert all(binding.values());assert np.array_equal(np.take_along_axis(stream['target'].reshape(-1,32),stream['within_batch_target_position'].reshape(-1,32).astype('i8'),axis=1).reshape(-1),stream['matched_target'])
        put('ACTUAL_INPUT_STREAM_BINDING.json',{'actual_rows':N,'R43_stream_hashes':receipt['stream_hashes'],'white_random_reconstruction_and_unused_audit_match':binding,'original_rng_initial_final_verified':True,'saved_t_no_new_draw':True,'loaded_source_REAL':[len(source),len(real)],'unique_source_matched_REAL':[len(np.unique(stream['source'])),len(np.unique(stream['matched_target']))]})
        torch.manual_seed(20261010420);net=FlowNetwork();net.load_state_dict(torch.load(CFG['weight_path'],map_location='cpu',weights_only=True));net.to(device);net.eval();net.requires_grad_(False)
        parameter_before={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()for k,v in net.state_dict().items()};field=TangentField(net,mt,st)
        fast=np.zeros(480);fast[:100]=-.5;fast[100:150]=1;charge=np.zeros(480);charge[:100]=-3;charge[100:400]=1
        g=-scale*fast;h=scale*charge;gt=g-h*(np.sum(g*h,1)/np.sum(h*h,1))[:,None];norm=np.sum(gt*gt,1);assert np.isfinite(norm).all()and(norm>0).all()
        np.savez_compressed(OUT/'PSD_TANGENT_CONSTANTS.npz',g=g,h=h,gT=gt,gT_norm2=norm,scale=scale,mu=mu,edges_logQ=edges,c_fast=fast,c_charge=charge)
        rows=np.lib.format.open_memmap(OUT/'PATH_ROWS_PRIVATE.npy',mode='w+',dtype=DT,shape=(N,))
        maxidentity=maxpath=maxfield=maxmode=maxenergy=0.
        with torch.no_grad():
            for step in range(2950):
                timed();sl=slice(step*32,(step+1)*32);si=stream['source'][sl];ti=stream['matched_target'][sl];bn=stream['bin'][sl];ord0=stream['original_row_ordinal'][sl]
                b=torch.from_numpy(bn).to(device=device,dtype=torch.int64);a,qs,_=initial(torch.from_numpy(source[si]).to(device),torch.from_numpy(white[ord0]).to(device),mt[b],st[b]);target=torch.from_numpy(real[ti]).to(device=device,dtype=torch.float64);qt=q_torch(target);targetunit=target/qt[:,None];c=(targetunit-mt[b])/st[b]
                ts=torch.from_numpy(stream['t'][sl]).to(device=device,dtype=torch.float64);y=(1-ts[:,None])*a+ts[:,None]*c;v=c-a
                attempted+=32;calls+=1;checkpoint();pred=field(y,ts,b);e=(pred-v).cpu().numpy();sc=scale[bn]
                ltarget=-f_np(sc*v.cpu().numpy());lpred=-f_np(sc*pred.cpu().numpy());sourcePSD=1-f_np(source[si].astype('f8')/qs.cpu().numpy()[:,None]);initialPSD=1-f_torch(mt[b]+st[b]*a).cpu().numpy();targetPSD=1-f_np(targetunit.cpu().numpy());resid=lpred-ltarget
                dot=np.sum(e*gt[bn],1);emode=dot[:,None]/norm[bn,None]*gt[bn];comp=e-emode;total=np.sum(e*e,1);mode=np.sum(emode*emode,1);ce=np.sum(comp*comp,1)
                identity=ltarget-(targetPSD-initialPSD);pr=q_np(sc*v.cpu().numpy());fr=q_np(sc*pred.cpu().numpy());merr=dot-resid;closure=total-mode-ce
                maxidentity=max(maxidentity,float(np.max(np.abs(identity))));maxpath=max(maxpath,float(np.max(np.abs(pr))));maxfield=max(maxfield,float(np.max(np.abs(fr))));maxmode=max(maxmode,float(np.max(np.abs(merr))));maxenergy=max(maxenergy,float(np.max(np.abs(closure))))
                assert maxidentity<=1e-10 and maxpath<=1e-10 and maxfield<=1e-10 and maxmode<=1e-10
                vals={'t':ts.cpu().numpy(),'source_Q_Vns':qs.cpu().numpy(),'target_Q_Vns':qt.cpu().numpy(),'source_PSD':sourcePSD,'initial_PSD':initialPSD,'target_PSD':targetPSD,'target_PSD_velocity':ltarget,'predicted_PSD_velocity':lpred,'signed_PSD_residual':resid,'MSE480':total/480,'total_energy':total,'PSD_mode_energy':mode,'complement_energy':ce,'energy_closure_error':closure,'PSD_identity_error':identity,'path_q_residual':pr,'field_q_residual':fr}
                assert all(np.isfinite(a).all()for a in vals.values())
                for k,value in vals.items():rows[k][sl]=value
                for k,value in [('source_TRAIN_index',si),('target_original_TRAIN_index',stream['target'][sl]),('matched_target_TRAIN_index',ti),('original_noise_ordinal',ord0),('bin0',bn),('time_bin0',np.minimum((stream['t'][sl]*4).astype('i8'),3))]:rows[k][sl]=value
                rows.flush();committed+=32;checkpoint();assert torch.cuda.max_memory_allocated()<=CFG['GPU_limit_bytes'] and torch.cuda.max_memory_reserved()<=CFG['GPU_limit_bytes']
                if step%250==249:print(json.dumps({'committed_rows':committed,'elapsed_s':time.monotonic()-START,'PSD_identity_maxabs':maxidentity}),flush=True)
        assert field.rows==N and committed==attempted==N
        assert all(parameter_before[k]==hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()for k,v in net.state_dict().items())
        put('FORMULA_NUMERIC_CHECK.json',{'PSD_velocity_endpoint_identity_maxabs':maxidentity,'path_charge_tangent_maxabs':maxpath,'predicted_charge_tangent_maxabs':maxfield,'gT_dot_residual_vs_L_residual_maxabs':maxmode,'row_energy_closure_maxabs':maxenergy,'pass':True,'threshold_identity_charge':1e-10,'no_extra_network_forward_for_formula_check':True})
        allsummary=aggregate(rows,np.ones(N,bool));byq=[];groups=[]
        for b in range(8):
            byq.append({'bin_id':b+1,**aggregate(rows,rows['bin0']==b)})
            for t in range(4):groups.append({'bin_id':b+1,'time_bin0':t,**aggregate(rows,(rows['bin0']==b)&(rows['time_bin0']==t))})
        put('VELOCITY_SUMMARY.json',{'overall':allsummary,'by_Q':byq,'by_Q_time':groups,'MSE_geometry_units':'standardized480 coordinate velocity squared; PSD velocity dimensionless /unit t','weight':'actual consumed-row exposures, not independent acquisitions'})
        pops=markers=None;hits=None
        if not CFG.get('forward_only_server'):
            pops,markers=population(source,real,stream,rows,edges);hits=sum(a['hit']for a in markers)
        if markers is not None:put('DIRECTION_MARKERS.json',{'bins':markers,'hits':hits,'required_hits':4,'supported':hits>=4,'decision':'TRAIN_path_has_directionally_aligned_mean_PSD_residual'if hits>=4 else'close_mean_TRAIN_PSD_velocity_residual_explanation','causal_claim':False,'new_training_authorized':False})
        if markers is not None:plot(rows,groups,pops,markers)
        put('NUMERIC_RECEIPT.json',{'PID':os.getpid(),'UTC':utc(),'elapsed_s':time.monotonic()-START,'status':'complete','attempted_forward_rows':attempted,'committed_rows':committed,'forward_calls':field.calls,'network_forward_rows':field.rows,'chunk':32,'response_updates':0,'new_ODE_generation':0,'HGB_EM_bootstrap_fits':0,'weight_parameters_before_after_equal':True,'direction_hits':hits,'loaded_source_REAL':[len(source),len(real)],'unique_source_matched_REAL':[len(np.unique(stream['source'])),len(np.unique(stream['matched_target']))],'formula_pass':True,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        print(json.dumps({'complete':True,'PID':os.getpid(),'rows':committed,'hits':hits,'PSD_mode_energy_fraction':allsummary['PSD_mode_energy_fraction'],'elapsed_s':time.monotonic()-START}),flush=True)
    except BaseException as e:
        checkpoint();put('FAILURE.json',{'PID':os.getpid(),'UTC':utc(),'error':repr(e),'traceback':traceback.format_exc(),'attempted_forward_rows':attempted,'committed_rows':committed,'elapsed_s':time.monotonic()-START,'no_whole_forward_retry':True,'original_results_preserved':True});raise

if __name__=='__main__':forward()
