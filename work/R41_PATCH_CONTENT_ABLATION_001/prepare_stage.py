from pathlib import Path
from datetime import datetime, timezone
import json, os, shutil, ast, hashlib
ROOT=Path(__file__).resolve().parent; P=ROOT.parents[1]
BASE=ROOT.parent/'R32_CONDITIONAL_MEAN_TRANSLATION_001'; OLD=ROOT.parent/'R40_GLOBAL_TIME_MIXER_TRANSLATION_001'; R39=ROOT.parent/'R39_FULL_JOINT_KERNEL_TRANSLATION_001'
assert not (ROOT/'PREPARATION_RECEIPT.json').exists()
reg=json.loads((P/'reviews/R41_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
cfg=json.loads((BASE/'config.json').read_text('utf8')); previous=json.loads((OLD/'config.json').read_text('utf8'))
for k,v in previous.items():
    if k not in cfg:cfg[k]=v
for k in ['time_mixer_seed','time_mixer_width','time_mixer_parameters','module_zero_description_rows','new_initial_equivalence_model_rows','kernel_sigmas','initial_gradient_calibration_batches','REAL_control_scope_addendum']:cfg.pop(k,None)
ARM='NO_PATCH_NCE_FINAL'; REAL='NO_PATCH_NCE_REAL_CONTROL'
cfg.update(stage=ROOT.name,registered_UTC=reg['registered_UTC'],deadline_UTC=reg['executor_seal_deadline_UTC'],absolute_deadline_UTC=reg['stage_deadline_UTC'],
    remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R41_REMOTE_20261009_224636_UTC',
    arms=[ARM],endpoints=[ARM,REAL],updates=2950,training_wall_s=900,training_total_limit_s=900,science_budget=1800,
    mean_coefficient=0.,NCE_evaluated=False,NCE_forward_calls=0,NCE_positions_audit_only=True,
    scientific_package='original R32 CNN; omit actual PatchNCE computation and objective only; all other losses and fresh directions unchanged',
    selection='sole final2950 DEV endpoint; original J20 best checkpoint saved as record only',
    endpoint_weight_paths={ARM:str(ROOT/ARM/'FINAL_TRANSLATOR.pt'),REAL:str(ROOT/ARM/'FINAL_TRANSLATOR.pt')},
    HGB_max_fits=20,HGB_plan=20,HGB_cap=20,density_EM_max=16,width_slots=16,main_generation_rows=4000,repeated_rows_cap=1024,source_probe_rows=832,replay_rows=4000,
    R39_root=str(R39),old_BASE_REAL_generated=str(R39/'BASE_REAL_CONTROL/generated.npz'),old_BASE_REAL_metrics=str(R39/'metrics.json'),
    first_actual_command={'tool':'d9d41c','PID':13652,'native_exit':0,'UTC':'2026-10-09T22:47:05.7590854Z'},
    execution={'training':'R32 Linux CUDA environment original full CNN no PatchNCE','evaluation_replay':'local CPU4 E/D','CPU_python':cfg['CPU_python'],'actual_scientific_PID_nonce_RSS_before_imports':True},
    REAL_control_gate='candidate-minus-reused-same-treated-BASE_REAL three-view paired95%upper<=.10 and pointdelta<=.01; RAW absolute descriptive',
    old_reference_training_allowed=False,endpoint_selection_uses_DEV=False,
    primary_intervention_comparison='NO_PATCH_NCE_FINAL2950 versus saved R32 CONTROL_FINAL2950',critic_intervention='explicit STANDARDIZED',
    sole_training_intervention='actual content_NCE omitted; project heads retained unused; no new module/loss/coefficient changes')
cfg['original_content_NCE_weight']=cfg['loss_weights']['content_NCE'];cfg['loss_weights']['content_NCE']=0.
cfg['BASE_initial_J20']=json.loads((BASE/'SERVER_FRESH_CONTROL/TRAINING_RECEIPT.json').read_text('utf8'))['initial_validation']['J'];cfg['initial_J20_tolerance']=1e-6
cfg['remote_fresh_reference']='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R32_REMOTE_20261009_170201_UTC/SERVER_FRESH_CONTROL/FRESH_PROJECTION_STREAM.json'
for n in ['model.py','legacy_model.py','prior.py','fine_core.py','reorder_core.py']:shutil.copy2(BASE/n,ROOT/n)
for n in ['evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py','endpoint_common.py','temporal_common.py','common_math.py','mean_description.py','random_diagnostics.py','flow_diagnostics.py','flow_core.py','owned_worker.py','supervise.py','linux_owned_worker.py','linux_supervise.py','launch_remote.py','resource_check.py','covariance_description.py','flow_runtime.py','generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py','deploy_training.py','retrieve.py']:shutil.copy2(OLD/n,ROOT/n)
payload=['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','AFFINE_CONSTANTS.npz']
for n in payload:shutil.copy2(OLD/n,ROOT/n)

def change(s,a,b):
    assert a in s,('missing expected source',a[:100]);return s.replace(a,b)
s=(ROOT.parent/'R28_FRESH_PROJECTION_TRANSLATION_001/online_training.py').read_text('utf8')
s=change(s,',rank_losses,nce,sw',',rank_losses,sw')
s=change(s,"'FRESH_FINE']","'FRESH_FINE','NO_PATCH_NCE_FINAL']")
s=s.replace("actual.pop('original_noise')","actual['original_noise']").replace("actual.pop('original_style')","actual['original_style']")
s=change(s,"    og=torch.optim.Adam", "    original_states={'G':torch.load(cfg['G_initial'],map_location='cpu',weights_only=True),'reader':torch.load(cfg['reader_initial'],map_location='cpu',weights_only=True),'D':torch.load(cfg['D_initial'],map_location='cpu',weights_only=True)}\n    initial_state_equal={n:set(m.state_dict())==set(original_states[n])and all(torch.equal(v.detach().cpu(),original_states[n][k])for k,v in m.state_dict().items())for n,m in [('G',model),('reader',reader),('D',critic)]}\n    assert all(initial_state_equal.values())\n    project_initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()if k.startswith('project.')}\n    assert len(project_initial)==6 and all(p.requires_grad for p in model.parameters())\n    torch.save(project_initial,root/'INITIAL_PROJECT_HEADS.pt')\n    og=torch.optim.Adam")
s=change(s,"permutation=actual['original_row_ordinal'];grad_rows=[]", "permutation=actual['original_row_ordinal'];grad_rows=[];component_rows=[];project_grad_checks=0\n    def consume_unused_positions():\n        for length in [480,240,120]:\n            start=max(1,int(80*length/480));stop=min(length,int(420*length/480))\n            rngnce.choice(range(start,stop),cfg['patches_per_layer'],replace=False)")
s=change(s,"counter['response_fit_attempts']+=1;clock=time.monotonic();initial=validate(0);trace=", "counter['response_fit_attempts']+=1;clock=time.monotonic();initial=validate(0);initial_check={'state_key_and_value_equal':initial_state_equal,'initial_J20':initial['J'],'reference_J20':cfg['BASE_initial_J20'],'absolute_difference':abs(initial['J']-cfg['BASE_initial_J20']),'tolerance':cfg['initial_J20_tolerance'],'checks_before_first_D_G_reader_update':True,'G_updates_before_check':counter['G_updates'],'D_updates_before_check':counter['D_updates'],'reader_updates_before_check':counter['style_reader_updates']}\n    (root/'INITIAL_EQUIVALENCE.json').write_text(json.dumps(initial_check,indent=2)+'\\n','utf8')\n    assert initial_check['absolute_difference']<=cfg['initial_J20_tolerance']\n    trace=")
s=s.replace("counter['training_s']+time.monotonic()-clock>=900-10","counter['training_s']+time.monotonic()-clock>=cfg['training_total_limit_s']-10")
s=change(s,'content=nce(model,x,y,cfg,rngnce);','consume_unused_positions();')
s=change(s,"'content_NCE':content,",'')
s=change(s,'loss.backward();g_norm', "component_rows.append([*[float(v.detach())for v in components.values()],float(fine_value.detach()),float(style_loss.detach()),float(loss.detach())]);loss.backward();assert all(p.grad is None for p in model.project.parameters());project_grad_checks+=1;g_norm")
s=change(s,"valrow=validate(step);row=", "fine.persist();valrow=validate(step);row=")
s=change(s,"'components':{k:float(v)for k,v in components.items()},", "'components':{k:float(v)for k,v in components.items()},'NCE_evaluated':False,'NCE_loss_value':None,'NCE_forward_calls':0,'NCE_objective_weight':0.,")
s=change(s,'fine.persist();assert fine.calls==updates', "fine.persist();assert fine.calls==updates\n        assert all(p.grad is None for p in model.project.parameters())\n        project_final={k:v.detach().cpu().clone()for k,v in model.state_dict().items()if k.startswith('project.')}\n        torch.save(project_final,root/'FINAL_PROJECT_HEADS.pt')\n        project_equal={k:torch.equal(project_initial[k],v)for k,v in project_final.items()};assert all(project_equal.values())\n        participation={'NCE_evaluated':False,'NCE_loss_value':None,'NCE_objective_weight':0.,'NCE_dedicated_forward_calls':0,'NCE_retained_position_lists':rngnce.at,'unused_NCE_positions_only':True,'project_grad_None_update_checks':project_grad_checks,'project_initial_final_bitwise_equal':project_equal,'project_parameter_count':sum(p.numel()for p in model.project.parameters()),'all_original_G_parameters_requires_grad':all(p.requires_grad for p in model.parameters()),'total_G_parameters':sum(p.numel()for p in model.parameters()),'participating_G_parameters':sum(p.numel()for n,p in model.named_parameters()if not n.startswith('project.')),'participating_G_last_gradient_nonNone_parameters':sum(p.numel()for n,p in model.named_parameters()if not n.startswith('project.')and p.grad is not None),'all_other_G_and_reader_parameters_finite':all(torch.isfinite(p).all().item()for p in [*model.parameters(),*reader.parameters()]),'all_other_G_parameter_names_with_None_final_gradient':[n for n,p in model.named_parameters()if not n.startswith('project.')and p.grad is None],'no_optimizer_freeze_of_original_encoder':True,'dedicated_encoding_calls_avoided':updates*2}\n        assert participation['all_other_G_and_reader_parameters_finite']and not participation['all_other_G_parameter_names_with_None_final_gradient']\n        (root/'NCE_PARTICIPATION_RECEIPT.json').write_text(json.dumps(participation,indent=2)+'\\n','utf8')\n        np.savez_compressed(root/'BASIC_LOSS_COMPONENTS_BY_UPDATE_PRIVATE.npz',values=np.asarray(component_rows,dtype='f8'),columns=np.asarray([*components,'fresh_fine','style_L1','total_G']))")
s=change(s,"'fresh_projection_calls':fine.calls", "'initial_equivalence':initial_check,'NCE_participation':participation,'fresh_projection_calls':fine.calls")
assert 'nce(model'not in s and 'content=nce'not in s
(ROOT/'online_training.py').write_text(s,'utf8')

remote=dict(cfg);oldremote=json.loads((BASE/'remote_config.json').read_text('utf8'))
for k in ['training_input_paths','consumed_R26_stream','actual_training_random_states','actual_training_receipt','consumed_R26_receipt','G_initial','reader_initial','D_initial']:remote[k]=oldremote[k]
for n,c in [('config.json',cfg),('remote_config.json',remote)]:(ROOT/n).write_text(json.dumps(c,indent=2,allow_nan=False)+'\n','utf8')
for n in ['generate_all.py','evaluate_saved.py','replay_all.py','deploy_training.py','retrieve.py']:
    s=(ROOT/n).read_text('utf8').replace('GLOBAL_TIME_MIXER_FINAL',ARM).replace('GLOBAL_TIME_MIXER_REAL_CONTROL',REAL)
    if n=='generate_all.py':
        start=s.index('    # Final-module-zero');end=s.index("    old=json.loads",start);s=s[:start]+s[end:]
    if n=='evaluate_saved.py':
        s=s.replace("'final_module_zero_description':json.loads((ROOT/'FINAL_MODULE_ZERO_DESCRIPTION.json').read_text('utf8')),",'')
    if n=='deploy_training.py':s=s.replace("'initialization_check.py',",'')
    (ROOT/n).write_text(s,'utf8')
(ROOT/'START_RECEIPT.json').write_text(json.dumps(cfg['first_actual_command'],indent=2)+'\n','utf8')
(ROOT/'PROTOCOL.md').write_text('''# R41: PatchNCE removal, sole final2950

Authoritative immutable tasks/R41_PATCH_CONTENT_ABLATION_001.md and its registration. Original R32 full neural configuration/architecture and R28 fresh-fine train loop, STANDARDIZED critic, newAdam .0002(.5,.999), clip5/D1. Only actual PatchNCE computation and objective omitted (coefficient0). No measured NCE loss: evaluated=false/value=null/dedicated forward calls0. All other loss definitions, weights, order and RNG remain. Three project heads retained in state and optimizer, requires_grad unchanged, gradientsNone every backward and initial/final weights bitwise equal; all other parameters train normally and finite. Removing dedicated source/output encodes changes compute/clip/Adam context, not an equal-FLOP or unique mechanism claim. No BN/dropout in original model.

Before any update, strict original G/reader/D state keys/values match and actual J20 matches original same-server value within1e-6. Original94400 source/REAL/bin/ordinal/z-white/style rows; NCE 8850 saved lists consumed within original480/240/120 ranges, no new draws and never used in loss. Actual unique exposure3752/18739, loaded6593/21047. Fresh PCG64(20261009281) shape480/diff479/baseline100 matrices128 each, 8850 groups preserved; validation uses original fixed directions without consuming fresh RNG. OriginalJ20 internal selection record only; DEV sole fixed2950 endpoint, no prefix rescue.

Full source480->G singleADC, all2000 parents including417 supportoutside use newG. SOURCE2000 and same-treated REAL2000; joint1024, originalprobe832 and SOURCE150:400 zero-integral1mV perturbation128. No moduleoff endpoint. Reuse sealedR39 BASE_REAL waves/10fits/density/replay; RAW9refs no refit/2D pairedprediction unknown. New20HGB/16densityslots only. Fixed600/250 rows/groups,500classifierconditionalbootstrap. IndependentCPU4000 fullsource/REAL chain chunk128 with exact input/noise/component/prepost comparisons. Functional signal/electronic are not independent physical noise.

Actual preADC sourceQ purehad bins3/4 N213/181 distinct from historypostADC207/182. Same acquisition/artificial groups/no independent particle truth. Source sorting and cross-domain valley are proxies, not particle labels. Six task gates conjunctive, undefined fails; closes this single coefficient0 recipe on failure, no coefficient/seed/weight/architecture search. Same-treatedREAL gate uses candidate minus sealedBASE_REAL, not RAW identity. Full493/u point<=.934/.928 and paired95upper<0; PSD/width/center/noise relative10%, no new BASEvalid2–7 densityfailure; source ranks/.02valley,9boundaries+LSB,128inputperturbation, threeREAL point+.01/upper+.10, initial/flow/NCE/replay/resource checks unchanged.

Science1800s, unique train900s inclinitialJ/validation/finalsave; CPU4/actualRSS/GPUallocated4GiB/output512MiB. Server fresh own leaf, existing environment/noinstall; local writes/cache/tmp E/D. NoEVAL/CAL/newREAL/G4/agent/Git/migration equality testing. Executor stops2026-10-10T00:06:36.130049Z; root00:16:36.130049Z. CUT is cited in task as provenance of adapted 1D local correspondence, not claimed full implementation.
''','utf8')
bindings={}
for p in ROOT.glob('*.py'):
    b=p.read_bytes();ast.parse(b.decode('utf8'));bindings[p.name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_AST':bindings,'dependency_payload_bytes':sum((ROOT/n).stat().st_size for n in payload),'science_started':False,'old_stage_writes':0,'migration_content_tests':0,'complete_neural_config_origin':str(BASE/'config.json')}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps({'PID':os.getpid(),'UTC':rec['UTC'],'AST_count':len(bindings),'dependency_bytes':rec['dependency_payload_bytes']}),flush=True)
