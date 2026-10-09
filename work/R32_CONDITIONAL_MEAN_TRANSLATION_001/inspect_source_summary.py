from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:os.environ[k]='4'
import pathlib,json,datetime,time
import numpy as np
import evaluation as ev
R=pathlib.Path(__file__).resolve().parent;start=time.monotonic();cfg=json.loads((R/'config.json').read_text('utf8'))
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
a=load(R/'ACTUAL_GENERATION_INPUTS.npz');m=load(R/'DEV_SOURCE_METADATA.npz')
edges=np.asarray(json.loads(pathlib.Path(cfg['local_input_paths']['width_definition']).read_text('utf8'))['edges_logQ'])
def bins(q):
    l=np.log(np.maximum(q,1e-30));b=np.searchsorted(edges,l,side='right')-1;b[l==edges[-1]]=7;return b
f=ev.features(a['source_waveforms_V']);bb=bins(f['features']['Qtotal_mVns']);bc=bins(m['Qsource_Vns']*1000)
comp=m['composition'];old={}
for n in ['R26_root','R28_root']:
    hist=json.loads((pathlib.Path(cfg[n])/'metrics.json').read_text('utf8'))
    old[n]=hist['source_composition']['primary_pure_had_source_bins3_4']
out={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,
    'fit_or_generation_or_response_update':0,'composition_counts':{str(k):int((comp==k).sum())for k in np.unique(comp)},
    'all_source_ID_order_equal':bool(np.array_equal(a['source_ids'],m['source_ids'])),
    'feature_charge_pure_had_counts_8bins':[int(((comp=='pure_had')&(bb==i)).sum())for i in range(8)],
    'metadata_Qsource_pure_had_counts_8bins':[int(((comp=='pure_had')&(bc==i)).sum())for i in range(8)],
    'feature_vs_metadata_Qsource_maxabs_mVns':float(np.max(np.abs(f['features']['Qtotal_mVns']-m['Qsource_Vns']*1000))),
    'historical_source_primary':old,'SOURCE_CLEAN_QC':int(f['valid'].sum()),
    'source_counts_are_before_QC':True}
(R/'SOURCE_POPULATION_INSPECTION.json').write_text(json.dumps(ev.safe(out),indent=2,allow_nan=False)+'\n','utf8')
print(json.dumps({k:v for k,v in out.items()if k!='historical_source_primary'}),flush=True)
