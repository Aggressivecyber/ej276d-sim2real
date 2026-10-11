"""Prepare frozen-endpoint evaluation while the sole R63 training job runs."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001';O=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
C=json.loads((S/'config.json').read_text('utf8'));ARM=C['arm'];CTRL='COVERAGE_REAL_CONTROL'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bound=json.loads((S/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))['files']
protected=[S/r['name']for r in bound]+[S/'ROOT_IMPLEMENTATION_REVIEW.json',R/C['protocol'],R/'PROJECT_ACCEPTANCE.json',R/'PROJECT_ACCEPTANCE.md']
before={str(p):sha(p)for p in protected}
for row in bound:assert sha(S/row['name'])==row['sha256'],row['name']
def convert(text):
 text=text.replace('PRECISION_RESOLVED_RESPONSE',ARM).replace('PRECISION_RESOLVED_REAL_CONTROL',CTRL)
 return text.replace('R61','R63').replace('R59','R61').replace('r59','r61').replace('DIRECT_DISTRIBUTION_RESPONSE','PRECISION_RESOLVED_RESPONSE').replace('DIRECT_DISTRIBUTION_REAL_CONTROL','PRECISION_RESOLVED_REAL_CONTROL')
P=json.loads((O/'config_post.json').read_text('utf8'));P.update(C)
for key in ['expected_R59_initial_state_SHA','expected_R59_stream_hashes','fixed_endpoint_single_factor_ablation','constants_reused_from_frozen_R59']:P.pop(key,None)
P.update(endpoints=[ARM,CTRL],REAL_arms=['BASE_REAL_CONTROL',CTRL],endpoint_weight_paths={n:C['remote_root']+'/'+ARM+'/FINAL_FLOW.pt'for n in [ARM,CTRL]},historical_training_root=C['remote_root'],
 parameter_initialization='same fresh seed and initial tensor hash as frozen R61',objective=C['sole_scientific_change'],scientific_scope='R63 two-term distribution coverage extension versus frozen R61; original A-F unchanged',
 execution={'training':'one fresh R61-matched network, exactly50000 updates, unchanged original seven losses plus coordinate RMS and temporal joint losses','inference':'same SOURCE/REAL operator, Heun128 cached displacement and explicit box-charge projection'})
assert not(S/'config_post.json').exists();(S/'config_post.json').write_bytes((json.dumps(P,indent=2)+'\n').encode())
files=['canonical.py','flow_runtime.py','numerical_check.py','generate_all.py','evaluate_saved.py','replay_all.py','evaluation.py','diagnostic_core.py','assessment.py','common_math.py','energy_width.py','data_support.py','model.py','legacy_model.py','flow_core.py','endpoint_common.py','flow_diagnostics.py','infer_saved.py','summarize_and_plot.py','render_R61_figures.py','post_audit.py']
for name in files:
 dest=S/name.replace('render_R61','render_R63');assert not dest.exists();dest.write_bytes(convert((O/name).read_text('utf8')).encode())
shutil.copy2(O/'TRAIN_SOLVER_CHECK.json',S/'HISTORICAL_R61_TRAIN_SOLVER_CHECK.json')
assert json.loads((S/'HISTORICAL_R61_TRAIN_SOLVER_CHECK.json').read_text())['both_pass']
def replace(name,old,new):
 p=S/name;text=p.read_text('utf8');assert text.count(old)==1,(name,text.count(old),old);p.write_bytes(text.replace(old,new).encode())
replace('post_audit.py','from precision_audit import precision_single_factor_audit','from coverage_audit import coverage_extension_audit')
replace('post_audit.py','precision=precision_single_factor_audit(root,cfg,receipt,losses,columns)','precision=coverage_extension_audit(root,cfg,receipt,losses,columns)')
replace('post_audit.py','losses.shape==(50000,20)','losses.shape==(50000,22)')
replace('post_audit.py',"'six_consumed_stream_hashes'","'seven_consumed_stream_hashes'")
replace('post_audit.py',"'seven_losses_sum'","'nine_losses_sum'")
replace('post_audit.py',"'comparison_with_R61_is_single_factor_ablation'","'comparison_with_R61_is_frozen_objective_extension'")
replace('post_audit.py',"'single_factor_precision_order'","'coverage_extension'")
replace('post_audit.py',"'stream_reconstruction_executed':False","'all_original_six_streams_reconstructed_again':False,'seventh_stream_reconstructed':True")
replace('post_audit.py',"    return {'pass':all(checks.values()),'checks':checks,","    checks={k:bool(v)for k,v in checks.items()}\n    return {'pass':all(checks.values()),'checks':checks,")
replace('evaluate_saved.py',"'single_factor_ablation':json.loads((ROOT/'TRAINING_FROZEN_AUDIT.json').read_text())['pass']","'single_factor_ablation':False,'frozen_objective_extension_integrity':json.loads((ROOT/'TRAINING_FROZEN_AUDIT.json').read_text())['pass']")
replace('evaluate_saved.py','R63 sole final50000 precision-order ablation; same R61 initialization, six consumed streams, parents and frozen constants','R63 sole final50000 coverage extension; original seven losses, same R61 initialization, original six consumed streams and parents; new TRAIN temporal constants and independent seventh direction stream')
replace('summarize_and_plot.py','R63 precision-resolved source-order numerator ablation; original full480 network, parent weights, other six losses and same-REAL operator','R63 full-coordinate RMS and temporal joint distribution coverage; original full480 network, parent weights, seven losses and same-REAL operator')
p=S/'summarize_and_plot.py';p.write_bytes(p.read_text('utf8').replace('close_fixed_R63_precision_order_ablation','close_fixed_R63_coverage_extension').encode())
with (S/'render_R63_figures.py').open('a',encoding='utf8',newline='\n')as f:
 f.write('''
    coverage=load(R/arm/'COVERAGE_BY_UPDATE_PRIVATE.npz');cv=coverage['values'];cn=coverage['columns'].tolist()
    fig,ax=plt.subplots(1,2,figsize=(12.5,4.7))
    for key in ['coordinate_RMS','temporal_joint','temporal_marginal','temporal_sliced']:
        ax[0].plot(steps,cv[:,cn.index(key)].reshape(-1,1000).mean(1),label=key)
    for key in ['V480_coordinate_RMS','U480_coordinate_RMS','DU479_coordinate_RMS','D2U478_coordinate_RMS']:
        ax[1].plot(steps,cv[:,cn.index(key)].reshape(-1,1000).mean(1),label=key.removesuffix('_coordinate_RMS'))
    for a in ax:a.set(xlabel='Update (sole final 50,000)',ylabel='Mean loss per 1,000 updates',ylim=(0,None));a.legend(fontsize=8)
    ax[0].set_title('Preregistered coverage losses');ax[1].set_title('All-coordinate RMS residual by full-window view')
    fig.suptitle('Training surrogate traces; original A-F acceptance remains separate')
    save(fig,'coverage_loss_decomposition.png')
''')
Q=dict(C);Q.update(allowed_scientific_entries=['summarize_and_plot.py'],GPU_scientific_entries=[],reporting_only=True,new_fit_or_inference_authorized=False)
(S/'config_reporting.json').write_bytes((json.dumps(Q,indent=2)+'\n').encode())
(S/'linux_supervise_reporting.py').write_bytes((S/'linux_supervise.py').read_text('utf8').replace("ROOT/'config.json'","ROOT/'config_reporting.json'").encode())
for p in S.glob('*.py'):ast.parse(p.read_text('utf8'),filename=str(p))
assert before=={str(p):sha(p)for p in protected}
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','training_files_and_protocol_acceptance_untouched':before,'new_generation_rows':5984,'replay_rows':4000,'new_HGB_fits':10,'density_slots':16,
 'frozen_R61_control_new_fits':0,'objective_extension_integrity_required_before_numerical_inference_and_DEV':True,'same_raw_SOURCE_definitions':True,'all_asts_parsed':True,'new_scientific_execution':0}
(S/'POSTPROCESS_PREPARATION.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
print(json.dumps({'post_files_prepared':len(files)+5,'training_code_protocol_acceptance_untouched':True,'AST':'pass'}))
