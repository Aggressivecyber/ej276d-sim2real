from pathlib import Path
import json,ast,hashlib,datetime,os
ROOT=Path(__file__).resolve().parent;P=ROOT.parents[1];OLD=P/'work/R33_RESIDUAL_OT_FLOW_001'
cfg=json.loads((ROOT/'config.json').read_text('utf8'))
add=P/'tasks/R34_AUXILIARY_PRECISION_RECOVERY_ADDENDUM_001.md'
payload=add.read_bytes();expected='982798afb33abce6ee4ddc47d68561bec23f2b7f149f9a94e2934954414e2cb0'
assert hashlib.sha256(payload).hexdigest()==expected
cfg.update(auxiliary_recovery_addendum={'path':str(add),'sha256':expected},auxiliary_recovery_inference_rows=3712,
           total_new_neural_inference_rows_limit=7712,auxiliary_saved_precision='exact original x1 recovered once by frozen chain and checked against all originally saved components')
(ROOT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n','utf8')
s=(OLD/'evaluate_endpoints_attempt002.py').read_text('utf8')
s=s.replace('import flow_diagnostics as fd','import flow_diagnostics as fd\nimport protected_diagnostics as pd\nfrom projection_core import protect,selfcheck\nR33=pathlib.Path(json.loads((pathlib.Path(__file__).resolve().parent/\'config.json\').read_text(\'utf8\'))[\'R33_root\'])')
s=s.replace("CONTROLS=['BASE']","CONTROLS=['BASE','INDEPENDENT_CFM','OT_CFM']")
s=s.replace("'response_updates':0,'completed_projection_group_blocks':0}","'response_updates':0,'completed_projection_group_blocks':0,'main_deterministic_transform_rows':0,'auxiliary_recovery_inference_rows':0}")
s=s.replace("results={'BASE':old['C2ST']['results']['CONTROL_FINAL']}","results={n:old['C2ST']['results'][n]for n in CONTROLS}")
start=s.index('    for label,candidate,reference in [(\'OT_minus_INDEPENDENT\'')
end=s.index('    np.savez_compressed(OUT/\'C2ST_PRIVATE.npz\'',start)
s=s[:start]+'''    for candidate in ARMS:
        for reference in ['BASE',CFG['endpoint_origins'][candidate]]:
            label=candidate+'_minus_'+reference;paired[label]={}
            for view in views:
                a=load(OUT/(candidate+'__'+view+'_PREDICTIONS_PRIVATE.npz'))
                bp=pathlib.Path(CFG['R32_root'])/'results'/('CONTROL_FINAL__'+view+'_PREDICTIONS_PRIVATE.npz') if reference=='BASE' else R33/'results'/(reference+'__'+view+'_PREDICTIONS_PRIVATE.npz')
                paired[label][view]=aa.paired_bootstrap(a,load(bp),CFG)
'''+s[end:]
s=s.replace("tables['BASE']=old['energy_conditional_width']['tables']['CONTROL_FINAL']", "tables.update({n:old['energy_conditional_width']['tables'][n]for n in CONTROLS})")
s=s.replace("[('all_final_endpoints',[*CONTROLS,*ARMS]),('OT_vs_independent',ARMS)]", "[('all_five_endpoints',[*CONTROLS,*ARMS]),*[(n+'_vs_BASE',['BASE',n])for n in ARMS],('protected_pair',ARMS)]")
s=s.replace("old=json.loads((pathlib.Path(CFG['R32_root'])/'metrics.json').read_text('utf8'))", "old=json.loads((R33/'metrics.json').read_text('utf8'))")
for f in ['ACTUAL_GENERATION_INPUTS.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz']:
    s=s.replace("load(ROOT/'"+f+"')","load(R33/'"+f+"')")
start=s.index('    generation={};repeats={};probe={};solver={}')
end=s.index('    stats,distance=ev.characterize',start)
s=s[:start]+'''    write('PROJECTION_FORMULA_CHECK.json',selfcheck())
    for orig in ['INDEPENDENT_CFM','OT_CFM']:
        saved=load(R33/orig/'generated.npz');assert np.array_equal(saved['source_ids'],ids)
        domains[orig]=aa.data(saved['waveforms_V'],ids,np.arange(2000),'G4_SOURCE',ref)
    generation={};repeats={};probe={};precision={};recoveries={}
    for name in ARMS:
        orig=CFG['endpoint_origins'][name];log('deterministic main P '+name);(ROOT/name).mkdir(exist_ok=True)
        saved=load(R33/orig/'generated.npz');assert np.array_equal(saved['source_ids'],ids)
        assert np.array_equal(saved['base_pre_ADC_V'],base_saved['pre_ADC_V'])
        payload,ad,provenance=protect(saved['base_pre_ADC_V'],saved['pre_ADC_V']);COUNTER['main_deterministic_transform_rows']+=len(ids)
        np.savez_compressed(ROOT/name/'generated.npz',**payload,source_ids=ids,original_pre_ADC_f32_V=saved['pre_ADC_V'],
            base_pre_ADC_V=saved['base_pre_ADC_V'],base_signal_V=saved['base_signal_V'],base_electronic_V=saved['base_electronic_V'],
            flow_applied=saved['flow_applied'],source_Qbin_actual_preADC=saved['source_Qbin_actual_preADC'])
        report,charge_payload=pd.precision_report(payload,saved['base_pre_ADC_V'],base_saved['waveforms_V'],saved['flow_applied'])
        precision[name]=report;assert report['actual_constraints']['constraint_pass']
        np.savez_compressed(OUT/(name+'_PROJECTION_PRECISION_PRIVATE.npz'),**charge_payload)
        post=payload['waveforms_V'];domains[name]=aa.data(post,ids,np.arange(2000),'G4_SOURCE',ref)
        generation[name]={'rows':2000,'all_original_source_ids_preserved':True,'ADC':ad,'projection_provenance':provenance,
            'flow_applied_N':int(saved['flow_applied'].sum()),'unsupported_source_Q_base_fallback_N':int((~saved['flow_applied']).sum()),
            'source_composition':{str(c):int((comp==c).sum())for c in np.unique(comp)},'main_neural_inference_rows':0,
            'f64_projection_sha256':hashlib.sha256(payload['pre_ADC_f64_V'].tobytes()).hexdigest(),
            'ADC_input_f32_sha256':hashlib.sha256(payload['ADC_input_f32_V'].tobytes()).hexdigest(),
            'post_ADC_sha256':hashlib.sha256(post.tobytes()).hexdigest(),'original_flow':orig,
            'fixed_base_weight_file':CFG['base_G'],'fixed_flow_weight_file':CFG['endpoint_weight_paths'][orig],
            'flow_steps':32,'final_ADC_once':True,'physical_final_signal_electronic_split':'unknown'}
        write('GENERATION_PROGRESS.json',{'generation':generation,'counters':COUNTER});progress()
        pipeline=Pipeline(CFG,R33,orig)
        probe[name]=pd.probes(pipeline,post,actual['style'],name,ROOT,OUT,CFG,EDGES,COUNTER)
        write('SOURCE_PROBES_SUMMARY.json',probe);progress()
        repeats[name]=pd.joint(pipeline,source,comp,name,ROOT,OUT,CFG,EDGES,VALLEY,COUNTER)
        write('JOINT8_SUMMARY.json',repeats);progress()
        recoveries[name]={'probe':probe[name]['recovery'],'joint':repeats[name]['groups']['joint8']['recovery']}
        write('AUXILIARY_PRECISION_RECOVERY_RECEIPT.json',{'PID':os.getpid(),'endpoints':recoveries,'rows':COUNTER['auxiliary_recovery_inference_rows'],'exact_recovery_pass':all(x['exact_recovery_pass']for r in recoveries.values()for x in r.values()),'addendum':CFG['auxiliary_recovery_addendum']})
        del pipeline,saved,payload
    write('ACTUAL_PROJECTION_CONSTRAINTS.json',precision)
    assert COUNTER['main_deterministic_transform_rows']==4000 and COUNTER['auxiliary_recovery_inference_rows']==3712
'''+s[end:]
s=s.replace("for n in ['REAL','BASE',*ARMS]", "for n in ['REAL',*CONTROLS,*ARMS]")
s=s.replace("describe_means(tdomains,['BASE',*ARMS]", "describe_means(tdomains,[*CONTROLS,*ARMS]")
s=s.replace("for n in ['BASE',*ARMS]", "for n in [*CONTROLS,*ARMS]")
s=s.replace("training={n:json.loads((ROOT/n/'TRAINING_RECEIPT.json').read_text('utf8'))for n in CFG['arms']}", "training={'new_response_updates':0,'new_flow_updates':0,'inherited_fixed_models':CFG['endpoint_weight_paths']}\n    boundary=pd.boundary_metrics(domains,OUT);write('BOUNDARY_METRICS.json',boundary)")
s=s.replace("'solver_check':solver,", "'precision_constraints':precision,'boundary_differences':boundary,'auxiliary_precision_recovery':recoveries,")
s=s.replace("json.loads((ROOT/'PAIRED_FLOW_RECEIPT.json').read_text('utf8'))", "json.loads((R33/'PAIRED_FLOW_RECEIPT.json').read_text('utf8'))")
s=s.replace("'two_current_server_flow_fits_as_registered':True", "'new_response_or_flow_fits':0,'same_fixed_projection_on_both_frozen_flows':True")
s=s.replace("'solver_check_rows':64", "'solver_check_rows':0,'deterministic_transform_rows':7712")
s=s.replace("log('R33 evaluation complete ", "log('R34 evaluation complete ")
(ROOT/'evaluate_protected.py').write_text(s,'utf8')
replay='''from __future__ import annotations
import os,time,pathlib,json,datetime
import numpy as np,torch
from flow_runtime import Pipeline,infer
from projection_core import protect
ROOT=pathlib.Path(__file__).resolve().parent;CFG=json.loads((ROOT/'config.json').read_text('utf8'));OLD=pathlib.Path(CFG['R33_root']);START=time.monotonic()
torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.use_deterministic_algorithms(True)
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
actual=load(OLD/'ACTUAL_GENERATION_INPUTS.npz');rows={};count=0
for name in CFG['endpoints']:
    orig=CFG['endpoint_origins'][name];pipeline=Pipeline(CFG,OLD,orig)
    pre,signal,electronic=infer(pipeline,actual['source_waveforms_V'],actual['z'],actual['white'],actual['style'],CFG['generation_chunk'])
    payload,ad,provenance=protect(pipeline.details['base_pre_ADC_V'],pre,origin='replayed');saved=load(ROOT/name/'generated.npz')
    check={k:bool(np.array_equal(v,saved[k]))for k,v in payload.items()}
    for key in ['base_pre_ADC_V','base_signal_V','base_electronic_V','flow_applied','source_Qbin_actual_preADC']:
        check[key]=bool(np.array_equal(pipeline.details[key],saved[key]))
    check['source_ids']=bool(np.array_equal(actual['source_ids'],saved['source_ids']))
    check['original_pre_ADC_f32_V']=bool(np.array_equal(pre,saved['original_pre_ADC_f32_V']))
    rows[name]={'rows':len(pre),'checks':check,'bitwise_pass':all(check.values()),'ADC':ad};count+=len(pre);del pipeline,saved,payload
passed=all(r['bitwise_pass']for r in rows.values())
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,
    'rows':count,'pass':passed,'endpoints':rows,'independent_process':True,'full_source_to_base_to_flow_to_P_replayed':True,
    'fit_or_update':0,'REAL_TRAIN_validation_read':False,'device':'cpu','torch':torch.__version__,'NumPy':np.__version__}
(ROOT/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\\n','utf8');print(json.dumps(receipt),flush=True);assert passed
'''
(ROOT/'replay_protected.py').write_text(replay,'utf8')
sources=[]
for f in sorted(ROOT.glob('*.py')):
    b=f.read_bytes();ast.parse(b);sources.append({'file':f.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
manifest=json.loads((OLD/'FINAL_MANIFEST.json').read_text('utf8'))
(ROOT/'DEPENDENCY_INDEX.json').write_text(json.dumps({'old_stage':str(OLD),'old_manifest_sha256':'8ccd397071d9bb85fd5c8e8902e478668a5bbf831e349da0323ac363e8b9cec4','frozen_models':cfg['endpoint_weight_paths'],'base_G':cfg['base_G'],'affine':str(OLD/'AFFINE_CONSTANTS.npz'),'source_main_inputs':str(OLD/'ACTUAL_GENERATION_INPUTS.npz'),'auxiliary_inputs':[str(OLD/x)for x in ['JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz']],'source_binding_inheritance':'original manifest/source versions retained; no migration rehash test'},indent=2)+'\n','utf8')
(ROOT/'SOURCE_FREEZE.json').write_text(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':sources,'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),'AST_pass':True,'numerical_started':False},indent=2)+'\n','utf8')
protocol='''# R34 frozen projection protocol

Both R33 final flow endpoints are retained. No fitting or hyperparameter choice enters P.
Promote actual original x1 and x0 separately to f64, subtract, and apply rank378 Euclidean P: zero0:100, remove the separate mean100:150 and150:400, retain400:480. Add to f64 x0, then retain the original f32 ADC conversion and one14bit ADC. Store all three precision levels and actual Q/PSD changes.

Main4000 rows use exact saved original preADC. Auxiliary2048+1664 original preADC was not separately stored; the registered precision recovery addendum permits a single frozen-chain recovery. Match every saved base S/E, f32 increment, mask/Qbin and original postADC before P. Never approximate a recovered original x1 from a saved f32 subtraction. The independent main4000 replay plus auxiliary3712 recovery is the7712 new neural-inference maximum; deterministic projected rows are separately7712.

No updates, new affine/scaler fits, changed support/ODE/coupling, new REAL/CAL/EVAL/G4/SSH or installations. Full population and original600/250 classifier rows/groups retained, 500 fixed-classifier paired cluster resamples,20 HGB and16 density slots/EM caps, all fit and slot outputs persisted. Existing BASE/Independent/OT/RR classifier predictions and density fits reused. Conditional CDF support N>=40; density failure for BASE-valid bins2–7 is explicit, with per-protected common comparisons and all-five common support separate. Native raw boundary differences are fixed only at99/100,149/150,399/400. Complete flow bounds and all four task gates remain unchanged.

Joint algebra contains base S/E, f32 base sum-rounding residual, kept and removed flow. Cross terms retained; final physical signal/electronic split unknown. Repeats do not add independent source or acquisition events. All outputs and temporary caches E/D. One CPU4 actual-PID/RSS guarded science process at a time; cumulative600s, RSS4GiB, unique new outputs384MiB, absolute executor stop19:56:09.348003UTC.
'''
(ROOT/'PROTOCOL.md').write_text(protocol,'utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'AST_sources':len(sources),'science_started':False,'entry':'evaluate_protected.py'}))
