from __future__ import annotations
import pathlib,json,io,datetime,numpy as np
from scipy.stats import spearmanr
import evaluation as ev,assessment as aa
ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text());CACHE={}
def log(s):print(datetime.datetime.now(datetime.timezone.utc).isoformat(),s,flush=True)


def write(n,v):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(ev.safe(v),ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def read(k):
    if k not in CACHE:
        p=pathlib.Path(CFG['local_input_paths'][k]);n=CFG['input_files'][k]['bytes']
        with p.open('rb')as f:b=f.read(n+1)
        if len(b)!=n:raise RuntimeError('existing input size changed '+k)
        CACHE[k]=b
    return CACHE[k]


def zload(p,keys=None):
    p=pathlib.Path(CFG['local_input_paths'].get(pathlib.Path(p).name,str(p)))
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in (z.files if keys is None else keys)}


def clean_expectation(ee,had):
    ee=np.asarray(ee,dtype='f8');had=np.asarray(had,dtype='f8');u=np.maximum(np.arange(512)-128.,0)[None,:];out=np.zeros((len(ee),512))
    for energy,taus,frac in [(ee,[13,35,270],[.7,.15,.15]),(.15*had,[13,59,460],[.5,.1,.4])]:
        for tau,weight in zip(taus,frac):out+=energy[:,None]*weight*(np.exp(-u/tau)-np.exp(-u/5))/(tau-5)
    energy=ee+.15*had;q=(energy+.03918)/.000132506/1000;out*=q[:,None]/out.sum(1,keepdims=True)
    peak=out.argmax(1);threshold=out.max(1)*.5;t=np.arange(512)[None,:];below=(t<peak[:,None])&(out<threshold[:,None]);tp=np.where(below,t,-1).max(1)+1;ix=(tp[:,None]-120+np.arange(480)[None,:]).clip(0,511)
    return np.take_along_axis(out,ix,axis=1).astype('f4'),q.astype('f4')


def composition(ee,had):
    return np.where((ee>0)&(had==0),'pure_EE',np.where((had>0)&(ee==0),'pure_had',np.where((ee>0)&(had>0),'mixed','zero')))


def inputs():
    v=np.frombuffer(read('REAL_V'),dtype='<f4',count=5200*480).reshape(5200,480).copy();qc=np.frombuffer(read('REAL_QC'),dtype='<u2',count=5200);g=np.frombuffer(read('REAL_groups'),dtype='<u2',count=5200);idx=np.frombuffer(read('REAL_original_index'),dtype='<u4',count=5200);roles=np.frombuffer(read('REAL_roles'),dtype='u1',count=5200)
    assert np.all(roles[:2000]==0)and np.all(roles[2000:]==1)and np.array_equal(g,idx//50)
    vg=json.loads(read('TRAIN_roles'))['REAL_internal_validation_artificial_groups'];fit=(qc[2000:]==0)&~np.isin(g[2000:],vg);val=(qc[2000:]==0)&np.isin(g[2000:],vg);assert fit.sum()==2592 and val.sum()==599
    ref=json.loads(read('CAL_reference'));edges=np.array(json.loads(read('width_definition'))['edges_logQ']);sel=json.loads(read('R7_selections'))
    with np.load(io.BytesIO(read('SIM_identity')),allow_pickle=False)as z:rids=z['real_event_ids'].copy();sids=z['sim_event_ids'].copy();labels=z['source_label'].copy()
    idsfit=np.array(['RAW_ORIGINAL_INDEX|'+str(i)for i in idx[2000:][fit]]);pick=ev.order(idsfit,'R7-RR-TRAIN-2000')[:2000];assert np.array_equal(idx[2000:][fit][pick],sel['TRAIN_reference_original_indices'])
    real=aa.data(v[:2000],rids,g[:2000],'REAL',ref,qc[:2000]);rr=aa.data(v[2000:][fit][pick],idsfit[pick],g[2000:][fit][pick],'TRAIN_REFERENCE',ref)
    add=zload(pathlib.Path(CFG['remote_R14_root'])/'TRAIN_ADDITION.npz');good=add['QC_reason_u16']==0;assert good.sum()==18455
    target=np.concatenate([v[2000:][fit],add['waveforms_V'][good]]).astype('f4');fitidx=np.concatenate([idx[2000:][fit],add['original_index'][good]]);assert len(target)==21047 and not np.isin(fitidx//50,vg).any()
    assert np.array_equal(fitidx,np.fromfile(pathlib.Path(CFG['remote_R14_root'])/'expanded_fit_original_index.bin',dtype='<u4'))
    gc=zload(pathlib.Path(CFG['remote_input_root'])/'G4_TRAIN_cache');mask=gc['internal_training'].astype(bool);assert mask.sum()==6593 and (~mask).sum()==1599
    source=gc['clean_expectation_V'][mask].astype('f4');allval=np.flatnonzero(~mask);vi=allval[ev.order(gc['event_ids'][allval],'R18-source-validation')[:512]];sv=gc['clean_expectation_V'][vi].astype('f4')
    gi=zload(pathlib.Path(CFG['remote_input_root'])/'generation_inputs');dev,q=clean_expectation(gi['edep_ee_MeV'],gi['edep_hadron_MeV']);assert dev.shape==(2000,480)and np.array_equal(sids,gi['sim_event_ids'])and np.allclose(q,gi['Qsource_Vns'],rtol=2e-6,atol=1e-7)
    comp=composition(gi['edep_ee_MeV'],gi['edep_hadron_MeV']);labelstr=np.array([x.decode()if isinstance(x,bytes)else str(x)for x in labels]);pdg=np.where(np.isin(labelstr,['neutron','2112']),2112,22).astype('i4')
    rolesrec={'G4_TRAIN_fit':6593,'G4_internal_val':1599,'G4_val_used_fixed':512,'REAL_TRAIN_fit':21047,'REAL_internal_val':599,'DEV_source_parents':2000,'REAL_DEV':2000,'TRAIN_PDG_counts':{str(x):int((gc['pdg'][mask]==x).sum())for x in np.unique(gc['pdg'][mask])},'TRAIN_composition':{str(x):int((composition(gc['edep_ee_MeV'][mask],gc['edep_hadron_MeV'][mask])==x).sum())for x in np.unique(composition(gc['edep_ee_MeV'][mask],gc['edep_hadron_MeV'][mask]))},'DEV_composition':{str(x):int((comp==x).sum())for x in np.unique(comp)},'DEV_PDG_counts_inferred':{str(x):int((pdg==x).sum())for x in np.unique(pdg)},'DEV_PDG_provenance':'inferred from historical exact2112/non2112 labels and R2 bound CSV onlyPDG22/2112; not explicitly stored per DEV event','all_source_parent_ids_preserved':True,'source_waveforms':'existing analytic response to G4 totalEE/had deposit, no microscopic timing','unpaired_domains':True,'EVAL_read':0,'CAL_numeric_decode':0,'raw_long_waveform_decode':0,'new_G4':0,'artificial50record_groups_one_acquisition':True}
    write('DATA_ROLES.json',rolesrec);np.savez_compressed(ROOT/'DEV_SOURCE_METADATA.npz',source_ids=sids,edep_ee_MeV=gi['edep_ee_MeV'],edep_hadron_MeV=gi['edep_hadron_MeV'],composition=comp,pdg_inferred=pdg,Qsource_Vns=q,source_label=labels)
    return source,target,sv,v[2000:][val].astype('f4'),dev,sids,comp,pdg,real,rr,ref,edges,sel,rolesrec


def association(domains,comp,pdg,edges):
    source=domains['SOURCE_CLEAN'];out=domains['TRANSLATED'];sq=source['features']['Qtotal_mVns'];sp=source['features']['PSD'];oq=out['features']['Qtotal_mVns'];op=out['features']['PSD'];good=source['valid']&out['valid'];sourcebin=np.searchsorted(edges,np.log(np.maximum(sq,1e-30)),side='right')-1;outbin=np.searchsorted(edges,np.log(np.maximum(oq,1e-30)),side='right')-1
    def corr(a,b,m):
        ok=m&np.isfinite(a)&np.isfinite(b);return {'n':int(ok.sum()),'Spearman':float(spearmanr(a[ok],b[ok]).statistic)if ok.sum()>2 and len(np.unique(a[ok]))>1 and len(np.unique(b[ok]))>1 else None}
    matrix=np.zeros((10,10),dtype='i4')
    for a,b in zip(np.clip(sourcebin+1,0,9),np.where(out['valid'],np.clip(outbin+1,0,9),9)):matrix[a,b]+=1
    rows={}
    for category,mask in [('all',np.ones(2000,dtype=bool))]+[(str(n),comp==n)for n in np.unique(comp)]+[('PDG_inferred_'+str(n),pdg==n)for n in np.unique(pdg)]:
        bins=[]
        for i in range(8):
            use=mask&good&(sourcebin==i);bins.append({'source_bin_1based':i+1,'N':int(use.sum()),'source_PSD':ev.summary(sp[use]),'output_PSD':ev.summary(op[use]),'output_minus_source_PSD':ev.summary((op-sp)[use]),'PSD_ranking':corr(sp,op,use)})
        rows[category]={'source_n':int(mask.sum()),'joint_QC_n':int((mask&good).sum()),'source_Q_to_output_Q':corr(sq,oq,mask&good),'source_PSD_to_output_PSD':corr(sp,op,mask&good),'logQ_change':ev.summary(np.log(np.maximum(oq[mask&good],1e-30)/sq[mask&good])),'bins':bins}
    return {'composition_and_PDG':rows,'migration_source_8bin_rows_to_output_8bin_columns_with_low_high_invalid':matrix,'migration_labels':['below',*['bin'+str(i+1)for i in range(8)],'above_or_invalid'],'no_microscopic_particle_truth_claim':True}

