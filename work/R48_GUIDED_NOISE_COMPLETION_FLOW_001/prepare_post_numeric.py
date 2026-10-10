from pathlib import Path
from datetime import datetime,timezone
import ast,os,json,hashlib,subprocess
R=Path(__file__).resolve().parent;P=R.parent;OLD=P/'R46_FIXED_INTEGRATION_BUDGET_001'
C=json.loads((R/'config.json').read_text('utf8'))
# This control script does not open any scientific input or fit anything.
base=json.loads((OLD/'config_cpu.json').read_text('utf8'));cpu={**C}
for k in ['local_input_paths','local_training_input_paths','R7_root','R26_root','R27_root','R32_root','R33_root','R35_root','R36_root','R37_root','R38_root','R39_root','old_BASE_REAL_generated','old_BASE_REAL_metrics']:
    cpu[k]=base[k]
cpu['threads']=12;cpu['frozen_eval_input_root']=base['remote_root'];cpu['evaluation_code_reference_root']=base['remote_root']
cpu['historical_training']={'R48_control_updates':50000,'R48_completion_updates':50000,'new_REAL_unique_from_calibration':True}
cpu['new_response_training_runs']=2
cpu['counter_scope_note']='generation/evaluation counters record no additional training; two fresh training runs have separate actual receipts'
(R/'config_cpu.json').write_bytes((json.dumps(cpu,indent=2,allow_nan=False)+'\n').encode())
bindings=[]
for n in ['evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py','common_math.py',
          'model.py','legacy_model.py','endpoint_common.py','temporal_common.py','mean_description.py','flow_core.py']:
    d=(OLD/n).read_text('utf8').replace('\r','');ast.parse(d)
    (R/n).write_bytes(d.encode());bindings.append({'name':n,'from':str(OLD/n),'LF_source_SHA':hashlib.sha256(d.encode()).hexdigest()})

s=(OLD/'generate_all.py').read_text('utf8').replace('\r','')
s=s.replace("['SHORT_2950_REAL_CONTROL','LONG_50000_REAL_CONTROL']","['GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_COMPLETION_REAL_CONTROL']")
s=s.replace("ROOT/'ACTUAL_GENERATION_INPUTS.npz'", "pathlib.Path(CFG['frozen_eval_input_root'])/'ACTUAL_GENERATION_INPUTS.npz'")
s=s.replace("ROOT/'DEV_SOURCE_METADATA.npz'", "pathlib.Path(CFG['frozen_eval_input_root'])/'DEV_SOURCE_METADATA.npz'")
s=s.replace("ROOT/'REAL_DEV_AND_REFERENCE_PRIVATE.npz'", "pathlib.Path(CFG['frozen_eval_input_root'])/'REAL_DEV_AND_REFERENCE_PRIVATE.npz'")
s=s.replace("ROOT/'VALLEY_DEFINITION.json'", "pathlib.Path(CFG['frozen_eval_input_root'])/'VALLEY_DEFINITION.json'")
a=s.index("'functional_component_semantics':");b=s.index("'unused_random_inputs'",a)
s=s[:a]+"'functional_component_semantics':{'source_input':'fixed complete guide','innovation':'projected colored seed, not physical electronics','flow_plus_final_cast':'total learned correction not independent noise'},\n  "+s[b:]
ast.parse(s);(R/'generate_all.py').write_bytes(s.encode())
s=(OLD/'replay_all.py').read_text('utf8').replace('\r','')
for n in ['ACTUAL_GENERATION_INPUTS.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz']:
    s=s.replace("ROOT/'"+n+"'", "pathlib.Path(CFG['frozen_eval_input_root'])/'"+n+"'")
ast.parse(s);(R/'replay_all.py').write_bytes(s.encode())

s=(OLD/'evaluate_saved.py').read_text('utf8').replace('\r','')
s=s.replace("REAL_ARMS=['BASE_REAL_CONTROL','SHORT_2950_REAL_CONTROL','LONG_50000_REAL_CONTROL']", "REAL_ARMS=['BASE_REAL_CONTROL','GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_COMPLETION_REAL_CONTROL']")
s=s.replace("CONTROLS=['BASE','INDEPENDENT_CFM','OT_CFM','PSD_ORDERED_CFM','CONDITIONAL_DIFFUSION_EDIT','FULL_COVARIANCE_CALIBRATION','SLICED_MONOTONE_TRANSPORT']", "CONTROLS=['BASE','INDEPENDENT_CFM']")
for n in ['ACTUAL_GENERATION_INPUTS.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','DEV_SOURCE_METADATA.npz','VALLEY_DEFINITION.json','CAL_REFERENCE_METADATA.json','DATA_ROLES.json']:
    s=s.replace("ROOT/'"+n+"'", "pathlib.Path(CFG['frozen_eval_input_root'])/'"+n+"'")
# Only required old BASE and boundary comparator outputs are read; all old fits reused.
a=s.index("    for n,path in [('BASE'");b=s.index("        a=load(path,keys=['waveforms_V','source_ids'])",a)
s=s[:a]+"    for n,path in [('BASE',pathlib.Path(CFG['R32_root'])/'CONTROL_FINAL/generated.npz'),\n        ('INDEPENDENT_CFM',pathlib.Path(CFG['R33_root'])/'INDEPENDENT_CFM/generated.npz')]+[(n,ROOT/n/'generated.npz')for n in SOURCE_ARMS]:\n"+s[b:]
a=s.index("    tdomains=");b=s.index("    source_domains=",a)
s=s[:a]+"    md={};mp={}\n"+s[b:]
s=s.replace("np.savez_compressed(OUT/'PLOT_WAVEFORMS_PRIVATE.npz',**waves)", "write('PLOT_WAVEFORM_REFERENCES.json',{'new':{n:str(ROOT/n/'generated.npz') for n in ARMS},'old_input_root':CFG['frozen_eval_input_root'],'old_outputs_referenced_not_copied':True})")
a=s.index("    from covariance_description import describe_covariance");b=s.index("    metrics=",a)
s=s[:a]+"    cv={'not_requested':True,'new_full_covariance_arrays_saved':0};coverage={};projection={}\n"+s[b:]
s=s.replace("'training':{'new_updates':0,'optimizer_loaded':False,'frozen_historical_training':CFG['historical_training'],'LONG_only_primary':True,'selection':'fixed two weights; no DEV checkpoint selection'}", "'training':{a:json.loads((ROOT/a/'TRAINING_RECEIPT.json').read_text('utf8'))for a in SOURCE_ARMS}")
a=s.index('    from budget_comparison import describe as budget_compare');b=s.index("    write('metrics.json',metrics)",a)
s=s[:a]+"    compare={}\n    for v in cs['results'][SOURCE_ARMS[0]]:\n        pp=[load(OUT/(n+'__'+v+'_PREDICTIONS_PRIVATE.npz'),keys=['probability','label','cluster']) for n in SOURCE_ARMS]\n        compare[v]=aa.paired_bootstrap(pp[1],pp[0],CFG)\n    metrics['paired_primary_minus_control_C2ST']=compare\n"+s[b:]
s=s.replace("log('R46 saved evaluation complete '","log('R48 saved evaluation complete '")
ast.parse(s);(R/'evaluate_saved.py').write_bytes(s.encode())
(R/'POST_NUMERIC_SOURCE_BINDINGS.json').write_bytes((json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'reused_evaluators':bindings,
    'control_only_source_preparation':True,'scientific_arrays_read':0,'no_DEV_work_until_both_numerical_pass':True,'historical_arrays_copied':0},indent=2)+'\n').encode())
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'post_numeric_code_prepared':True,'science':0}),flush=True)
