from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, shutil

R = Path(__file__).resolve().parents[1]
S = R / 'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
O = R / 'work/R54_CONTINUOUS_CHARGE_FLOW_001'
C = json.loads((S / 'config.json').read_text('utf8'))
ARM = C['arm']; CTRL = 'DIRECT_DISTRIBUTION_REAL_CONTROL'
protected = [S / n for n in ['config.json', 'train_final.py', 'network.py', 'source_flow_core.py',
    'noise_core.py', 'distribution_loss.py', 'artificial_check.py', 'linux_supervise.py', 'linux_owned_worker.py']]
protected += [R / C['protocol'], R / 'PROJECT_ACCEPTANCE.json', R / 'PROJECT_ACCEPTANCE.md']
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
P = json.loads((O / 'config_post.json').read_text('utf8'))
P.update(C)
for key in ['consumed_R26_stream', 'consumed_R26_receipt', 'actual_training_random_states',
    'actual_training_receipt', 'original_training_receipt_path', 'training_input_paths',
    'local_training_input_paths', 'R28_root', 'affine_path', 'batch',
    'historical_R53_root', 'historical_R53_controls_new_GPU_candidate_disclosed',
    'historical_R53_training_receipt', 'source_local_stream', 'full_coupling_path', 'partial_coupling_path',
    'original_initial_state_path', 'original_training_receipt', 'historical_control_weights',
    'teacher_relative_Q_error_limit', 'teacher_PSD_error_limit', 'teacher_baseline_tail_exact',
    'full_cycle_repeats', 'full_cycle_rows', 'partial_cycle_rows', 'teacher_definition',
    'continuous_condition_embedding_index', 'unchanged_zero_embedding_indices', 'sole_change']:
    P.pop(key, None)
P.update(endpoints=[ARM, CTRL], REAL_arms=['BASE_REAL_CONTROL', CTRL],
    endpoint_weight_paths={n: C['remote_root'] + '/' + ARM + '/FINAL_FLOW.pt' for n in [ARM, CTRL]},
    historical_training_root=C['remote_root'], HGB_cap=10, HGB_plan=10,
    density_EM_max=16, width_slots=16, main_generation_rows=4000, replay_rows=4000,
    parameter_initialization='fresh seed 20261010590, zero output head',
    architecture='full480 U-Net with continuous logQ and 8-bin one-hot conditioning; constant input-conditioned displacement',
    objective='seven preregistered direct unpaired distribution and preservation losses, no teacher',
    selection='sole final update50000; original TRAIN128/256 before DEV',
    execution={'training':'one direct-distribution network, exactly50000 updates',
        'inference':'same SOURCE/REAL operator, Heun128 cached displacement and explicit box-charge projection'},
    scientific_scope='R59 new fixed recipe; frozen R54 and R49 comparisons, not a single-factor ablation',
    historical_controls_frozen_no_new_fit=True, new_response_training_runs=1)
evaluation_keys = ['C2ST','C2ST_views','train_each_domain','holdout_each_domain','bootstrap_repeats',
    'bootstrap_seed','local_input_paths','R26_root','R32_root','R33_root','R39_root',
    'old_BASE_REAL_generated','old_BASE_REAL_metrics','canonical_R51_root','frozen_eval_input_root',
    'endpoints','REAL_arms','endpoint_weight_paths','historical_training_root','HGB_max_fits','HGB_cap',
    'HGB_plan','density_EM_max','width_slots','main_generation_rows','replay_rows',
    'parameter_initialization','architecture','objective','selection','execution','scientific_scope',
    'historical_controls_frozen_no_new_fit','new_response_training_runs',
    'planned_new_classifier_fits','planned_new_density_slots','canonical_ADC_features',
    'raw_SOURCE_input_quantized_for_features','historical_CPU_controls_new_GPU_candidate_disclosed',
    'postprocessing_registered_before_results','preADC_charge_relative_error_limit']
P = {k:v for k,v in P.items() if k in C or k in evaluation_keys}
assert not (S / 'config_post.json').exists()
(S / 'config_post.json').write_text(json.dumps(P, indent=2) + '\n', 'utf8')
files = ['canonical.py', 'flow_runtime.py', 'numerical_check.py', 'generate_all.py',
    'evaluate_saved.py', 'replay_all.py', 'evaluation.py', 'diagnostic_core.py', 'assessment.py',
    'common_math.py', 'energy_width.py', 'data_support.py', 'model.py', 'legacy_model.py',
    'flow_core.py', 'endpoint_common.py', 'flow_diagnostics.py', 'infer_saved.py',
    'summarize_and_plot.py', 'render_R54_figures.py']
for name in files:
    dest = S / name.replace('render_R54', 'render_R59')
    assert not dest.exists(), name
    text = (O / name).read_text('utf8')
    text = text.replace('CONTINUOUS_Q_RESPONSE', ARM).replace('CONTINUOUS_Q_REAL_CONTROL', CTRL)
    text = text.replace('R54', 'R59').replace('R53', 'R54').replace('r53', 'r54')
    text = text.replace('JOINT_PULSE_RESPONSE', 'CONTINUOUS_Q_RESPONSE').replace('JOINT_PULSE_REAL_CONTROL', 'CONTINUOUS_Q_REAL_CONTROL')
    dest.write_text(text, 'utf8')
for name in ['HISTORICAL_TRAIN_SOLVER_CHECK.json', 'UNIT_CHARGE_CONSTANTS.json']:
    if (S / name).exists():
        assert (S / name).read_bytes() == (O / name).read_bytes(), name
    else:
        shutil.copy2(O / name, S / name)

def replace(name, old, new):
    p = S / name; value = p.read_text('utf8'); assert old in value, (name, old)
    p.write_text(value.replace(old, new), 'utf8')

replace('flow_runtime.py', 'self.details=None;self.total_field_rows=0;self.last_receipt=None',
    'self.details=None;self.total_field_rows=0;self.total_network_rows=0;self.total_network_calls=0;self.last_receipt=None')
replace('flow_runtime.py', "model.details=detail;model.last_receipt=rec;model.total_field_rows+=rec['field_forward_rows']",
    "model.details=detail;model.last_receipt=rec;model.total_field_rows+=rec['field_forward_rows'];model.total_network_rows+=rec['actual_network_rows'];model.total_network_calls+=rec['actual_network_calls']")
replace('flow_runtime.py', '    def __init__(self,cfg,root,arm):',
    '    def __init__(self,cfg,root,arm):\n        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True')
replace('numerical_check.py', 'torch.set_num_threads(12);torch.set_num_interop_threads(12);torch.use_deterministic_algorithms(True)',
    'torch.set_num_threads(12);torch.set_num_interop_threads(12);torch.use_deterministic_algorithms(True)\ntorch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True')
replace('generate_all.py', 'import flow_diagnostics as fd', 'import flow_diagnostics as fd\nfrom post_audit import constraint_description')
replace('generate_all.py', "initial[arm]=initial_description(detail);write('INITIAL_NOISE_DESCRIPTIONS.json',initial)",
    "initial[arm]=initial_description(detail);initial[arm]['constraint_projection']=constraint_description(detail);write('INITIAL_NOISE_DESCRIPTIONS.json',initial)")
replace('generate_all.py', "initial[control]=initial_description(cd);write('INITIAL_NOISE_DESCRIPTIONS.json',initial)",
    "initial[control]=initial_description(cd);initial[control]['constraint_projection']=constraint_description(cd);write('INITIAL_NOISE_DESCRIPTIONS.json',initial)\n  write('CONSTRAINT_PROJECTION_SUMMARY.json',{k:v['constraint_projection']for k,v in initial.items()})")
replace('generate_all.py', "generations={};probes={};joint={};gps={};field_rows={};initial={};identity={}",
    "generations={};probes={};joint={};gps={};field_rows={};network_rows={};network_calls={};initial={};identity={}")
replace('generate_all.py', "field_rows[arm]=pipeline.total_field_rows;progress(arm+' complete SOURCE set')",
    "field_rows[arm]=pipeline.total_field_rows;network_rows[arm]=pipeline.total_network_rows;network_calls[arm]=pipeline.total_network_calls;progress(arm+' complete SOURCE set')")
replace('generate_all.py', "field_rows[control]=ctrl.total_field_rows",
    "field_rows[control]=ctrl.total_field_rows;network_rows[control]=ctrl.total_network_rows;network_calls[control]=ctrl.total_network_calls")
replace('generate_all.py', "'field_forward_rows_by_arm':field_rows,",
    "'field_forward_rows_by_arm':field_rows,'actual_network_rows_by_arm':network_rows,'actual_network_calls_by_arm':network_calls,'time_state_constant_displacement_cached':True,'explicit_internal_box_charge_projection':True,")
replace('generate_all.py', "'flow_plus_final_cast':'total learned correction not independent noise'",
    "'flow_and_constraint_plus_final_cast':'total learned and explicit constraint correction, not independent physical noise'")
replace('evaluate_saved.py', "'training_scope':'R59 sole final50000; historical R48 stream matched'",
    "'training_scope':'R59 sole final50000 direct unpaired distribution learning; all original TRAIN parents eligible; new sampling streams'")
replace('evaluate_saved.py', "'single_common_initialization_and_random_stream':'checked in TRAINING_RECEIPT.json'",
    "'single_factor_ablation':False,'independent_fresh_training_recipe':True")
replace('evaluate_saved.py', "'projection_distances':projection,",
    "'projection_distances':projection,'explicit_constraint_projection':json.loads((ROOT/'CONSTRAINT_PROJECTION_SUMMARY.json').read_text('utf8')),")

replace('summarize_and_plot.py', 'import evaluation as ev', 'import evaluation as ev\nfrom post_audit import training_audit')
replace('summarize_and_plot.py', "feat=load(R/'results/FEATURES_PRIVATE.npz')",
    "feat=load(R/'results/FEATURES_PRIVATE.npz')\ntrain_audit=training_audit(R,C,training[L]);write('TRAINING_FROZEN_AUDIT.json',train_audit)")
path = S / 'summarize_and_plot.py'; value = path.read_text('utf8')
start = value.index(" g6['frozen_final_training_stream_pass']=")
end = value.index(" g6['pass']=", start)
value = value[:start] + " g6['frozen_final_training_stream_pass']=train_audit['pass']\n g6['training_receipt_audit']=train_audit\n g6['direct_distribution_no_teacher']=True\n" + value[end:]
value = value.replace('R59 single continuous-logQ condition ablation with unchanged R54 teacher and full-guide REAL identity',
    'R59 direct unpaired full480 distribution network, original-parent global weights, explicit box-charge constraint and same-REAL identity')
value = value.replace('close_fixed_R59_continuous_charge_recipe', 'close_fixed_R59_direct_distribution_recipe')
value = value.replace("'R59_training':tc,", "'R59_training':training,'training_audit':train_audit,'explicit_constraint_projection':M['explicit_constraint_projection'],")
value = value.replace("'no_new_invalid_nonfinite_ADC_clip':", "'internal_constraint_projection':M['explicit_constraint_projection'][arm],'no_new_invalid_nonfinite_ADC_clip':")
path.write_text(value, 'utf8')

path = S / 'render_R59_figures.py'; value = path.read_text('utf8')
start = value.index("    loss=load(R/arm/'LOSS_BY_UPDATE_PRIVATE.npz')['values']")
end = value.index("    views=C['C2ST_views']", start)
value = value[:start] + '''    lossdata=load(R/arm/'LOSS_BY_UPDATE_PRIVATE.npz');loss=lossdata['values'];cols=lossdata['columns'].tolist()
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    for label in ['full_window','PSD','source_order','source_valley','noise','REAL_identity','box_correction']:
        block=loss[:,cols.index(label)].reshape(-1,1000).mean(1)
        ax[0].plot(np.arange(1,51)*1000,np.maximum(block,1e-12),label=label)
    ax[0].set(yscale='log',xlabel='Update (sole final 50,000)',ylabel='Mean loss per 1,000 updates',title='Direct unpaired distribution training');ax[0].legend(fontsize=7,ncol=2)
    tr=M['training'][arm];audit=json.loads((R/'TRAINING_FROZEN_AUDIT.json').read_text())
    txt='Teacher waveforms: 0\\nAll source parents: '+str(tr['actual_unique_SOURCE_REAL_reference_REAL_identity'][0])+' / 6593\\n'
    txt+='All REAL reference parents: '+str(tr['actual_unique_SOURCE_REAL_reference_REAL_identity'][1])+' / 21047\\n'
    txt+='Training updates: '+str(tr['updates'])+'\\nTraining receipt audit: '+('PASS' if audit['pass']else'FAIL')+'\\n'
    txt+='A / B / C: '+' / '.join('PASS' if acceptance[arm][k]['pass']else'FAIL' for k in ['A','B','C'])+'\\n\\n'
    txt+='Losses are training surrogates.\\nResponse acceptance remains unchanged.'
    ax[1].axis('off');ax[1].text(.02,.94,txt,va='top',linespacing=1.7)
    save(fig,'training_and_checks.png')

''' + value[end:]
value += '''
    fig,ax=plt.subplots(1,3,figsize=(12.5,4.1))
    for n in [arm,ctrl]:
        d=load(R/n/'generated.npz');r=d['constraint_projection_RMS_V']*1000
        nonzero=r[r>0]
        if len(nonzero):ax[0].hist(nonzero,bins=40,histtype='step',label=names[n],color=colors[n])
        else:ax[0].plot([],[],label=names[n]+' (zero correction)',color=colors[n])
        ax[1].hist(d['constraint_projection_rail_sample_count'],bins=np.arange(-.5,481,10),histtype='step',label=names[n],color=colors[n])
    a=M['explicit_constraint_projection'][arm];b=M['explicit_constraint_projection'][ctrl]
    ax[2].axis('off');ax[2].text(.02,.92,'Range projection events\\nSOURCE: '+str(a['range_active_events'])+' / 2000\\nREAL: '+str(b['range_active_events'])+' / 2000\\n\\nInternal projection is part of the model.\\nZero ADC clipping does not imply\\nzero internal waveform correction.',va='top',linespacing=1.6)
    ax[0].set(xlabel='Nonzero constraint correction RMS (mV)',ylabel='Events',title='Explicit constraint contribution');ax[0].legend(fontsize=8)
    ax[1].set(xlabel='Samples on voltage rails per event',ylabel='Events',title='Bounded endpoint saturation');ax[1].legend(fontsize=8)
    save(fig,'explicit_constraint_projection.png')
'''
path.write_text(value, 'utf8')
Q = dict(C); Q.update(allowed_scientific_entries=['summarize_and_plot.py'], GPU_scientific_entries=[],
    reporting_only=True, new_fit_or_inference_authorized=False)
(S / 'config_reporting.json').write_text(json.dumps(Q, indent=2) + '\n', 'utf8')
(S / 'linux_supervise_reporting.py').write_text((S / 'linux_supervise.py').read_text('utf8').replace("ROOT/'config.json'", "ROOT/'config_reporting.json'"), 'utf8')
for p in S.glob('*.py'): ast.parse(p.read_text('utf8'), filename=str(p))
after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
assert before == after
(S / 'POSTPROCESS_PREPARATION.json').write_text(json.dumps({'UTC': datetime.now(timezone.utc).isoformat(),
    'owner':'/root','training_files_and_protocol_acceptance_untouched':before,
    'new_generation_rows':5984,'replay_rows':4000,'new_HGB_fits':10,'density_slots':16,
    'historical_R54_comparison_is_not_single_factor_ablation':True,'same_raw_SOURCE_definitions':True,
    'all_asts_parsed':True,'new_scientific_execution':0},indent=2)+'\n','utf8')
print(json.dumps({'post_files_prepared':len(files)+5,'training_code_protocol_acceptance_untouched':True,'AST':'pass'}))
