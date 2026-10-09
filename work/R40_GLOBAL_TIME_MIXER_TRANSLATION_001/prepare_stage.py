from pathlib import Path
from datetime import datetime, timezone
import json, os, shutil, ast, hashlib
ROOT=Path(__file__).resolve().parent; P=ROOT.parents[1]
OLD=ROOT.parent/'R39_FULL_JOINT_KERNEL_TRANSLATION_001'; BASE=ROOT.parent/'R32_CONDITIONAL_MEAN_TRANSLATION_001'
assert not (ROOT/'PREPARATION_RECEIPT.json').exists()
reg=json.loads((P/'reviews/R40_REGISTRATION_001/RECEIPT.json').read_text('utf8'))
old=json.loads((OLD/'config.json').read_text('utf8')); cfg=json.loads((BASE/'config.json').read_text('utf8'))
for k,v in old.items():
    if k not in cfg: cfg[k]=v
for k in ['kernel_sigmas','initial_gradient_calibration_batches','REAL_control_scope_addendum']:
    cfg.pop(k,None)
arm='GLOBAL_TIME_MIXER_FINAL'; control='GLOBAL_TIME_MIXER_REAL_CONTROL'
cfg.update(stage=ROOT.name,registered_UTC=reg['registered_UTC'],deadline_UTC=reg['executor_seal_deadline_UTC'],absolute_deadline_UTC=reg['stage_deadline_UTC'],
    remote_root='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R40_REMOTE_20261009_221922_UTC',
    arms=[arm],endpoints=[arm,control],updates=2950,training_wall_s=900,training_total_limit_s=900,science_budget=1800,
    mean_coefficient=0.,time_mixer_seed=20261010040,time_mixer_width=128,time_mixer_parameters=123488,
    scientific_package='one zero-output initialized shared-channel full-time MLP before original mean and electronic heads; original fresh-fine objectives restored',
    selection='sole final2950 DEV endpoint; original J20 best checkpoint saved as record only',
    endpoint_weight_paths={arm:str(ROOT/arm/'FINAL_TRANSLATOR.pt'),control:str(ROOT/arm/'FINAL_TRANSLATOR.pt')},
    HGB_max_fits=20,HGB_plan=20,HGB_cap=20,density_EM_max=16,width_slots=16,main_generation_rows=4000,repeated_rows_cap=1024,source_probe_rows=832,replay_rows=4000,
    new_initial_equivalence_model_rows=256,module_zero_description_rows=128,R39_root=str(OLD),
    old_BASE_REAL_generated=str(OLD/'BASE_REAL_CONTROL/generated.npz'),old_BASE_REAL_metrics=str(OLD/'metrics.json'),
    first_actual_command={'tool':'42132d','PID':40996,'native_exit':0,'UTC':'2026-10-09T22:19:56.5577730Z'},
    execution={'training':'same R32 Linux CUDA environment one new neural fit','evaluation_replay':'local CPU4 E/D','CPU_python':cfg['CPU_python'],'actual_scientific_PID_nonce_RSS_before_imports':True},
    REAL_control_gate='candidate-minus-reused-same-treated-BASE_REAL three-view paired95%upper<=.10 and pointdelta<=.01; RAW absolute descriptive',
    old_reference_training_allowed=False,endpoint_selection_uses_DEV=False)
cfg['primary_intervention_comparison']='GLOBAL_TIME_MIXER_FINAL2950 versus saved R32 CONTROL_FINAL2950'
cfg['critic_intervention']='new arm explicitly STANDARDIZED'
cfg['sole_training_intervention']='single shared-channel zero-output time MLP; no mean or MMD'
for n in ['legacy_model.py','prior.py','fine_core.py','reorder_core.py']:
    shutil.copy2(BASE/n,ROOT/n)
for n in ['evaluation.py','diagnostic_core.py','assessment.py','energy_width.py','data_support.py','endpoint_common.py','temporal_common.py','common_math.py','mean_description.py','random_diagnostics.py','flow_diagnostics.py','flow_core.py','owned_worker.py','supervise.py','linux_owned_worker.py','linux_supervise.py','launch_remote.py','resource_check.py','covariance_description.py','generate_all.py','evaluate_saved.py','replay_all.py','flow_runtime.py','prepare_evaluation.py']:
    shutil.copy2(OLD/n,ROOT/n)
payload=['ACTUAL_GENERATION_INPUTS.npz','JOINT8_INPUTS_PRIVATE.npz','PROBE_INPUTS_CORRECTED.npz','DEV_SOURCE_METADATA.npz','REAL_DEV_AND_REFERENCE_PRIVATE.npz','CAL_REFERENCE_METADATA.json','DATA_ROLES.json','VALLEY_DEFINITION.json','AFFINE_CONSTANTS.npz']
for n in payload: shutil.copy2(OLD/n,ROOT/n)
s=(BASE/'model.py').read_text('utf8').replace('class Translator(LegacyTranslator):','class OriginalTranslator(LegacyTranslator):')
s=s.replace('        signal=x+.1*self.mean(h)[:,0]','        h=self.apply_time_mixer(h)\n        signal=x+.1*self.mean(h)[:,0]')
needle='    def forward(self,x,z,white,style=None):'
s=s.replace(needle,'    def apply_time_mixer(self,h):return h\n'+needle,1)
at=s.index('\ndef style_view')
addition='''
class Translator(OriginalTranslator):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(20261010040)
            self.time_mixer=nn.Sequential(nn.Linear(480,128,bias=True),nn.GELU(approximate='none'),nn.Linear(128,480,bias=True))
            nn.init.zeros_(self.time_mixer[-1].weight);nn.init.zeros_(self.time_mixer[-1].bias)
        assert sum(p.numel() for p in self.time_mixer.parameters())==123488
        self.disable_time_mixer=False
        self.last_time_mixer_output=None
    def apply_time_mixer(self,h):
        delta=self.time_mixer(h)
        self.last_time_mixer_output=delta.detach()
        return h if self.disable_time_mixer else h+delta
'''
s=s[:at]+addition+s[at:];(ROOT/'model.py').write_text(s,'utf8')
s=(ROOT.parent/'R28_FRESH_PROJECTION_TRANSLATION_001/online_training.py').read_text('utf8')
s=s.replace("'FRESH_FINE']","'FRESH_FINE','GLOBAL_TIME_MIXER_FINAL']")
s=s.replace("actual.pop('original_noise')","actual['original_noise']").replace("actual.pop('original_style')","actual['original_style']")
s=s.replace("model.load_state_dict(torch.load(cfg['G_initial'],map_location=device,weights_only=True));", "initial_state=torch.load(cfg['G_initial'],map_location=device,weights_only=True);missing,unexpected=model.load_state_dict(initial_state,strict=False);assert set(missing)=={'time_mixer.0.weight','time_mixer.0.bias','time_mixer.2.weight','time_mixer.2.bias'} and not unexpected;assert all(torch.equal(model.state_dict()[k],v)for k,v in initial_state.items());")
s=s.replace("counter['response_fit_attempts']+=1;clock=time.monotonic();initial=validate(0)", "counter['response_fit_attempts']+=1;clock=time.monotonic();from initialization_check import check_initial;initial_equivalence=check_initial(model,reader,critic,xs,xt,actual,cfg,root);initial=validate(0);initial_equivalence['initial_J20']=initial['J'];(root/'INITIAL_EQUIVALENCE.json').write_text(json.dumps(initial_equivalence,indent=2)+'\\n','utf8')")
s=s.replace("counter['training_s']+time.monotonic()-clock>=900-10", "counter['training_s']+time.monotonic()-clock>=cfg['training_total_limit_s']-10")
s=s.replace('grad_rows=[]','grad_rows=[];mixer_grad_rows=[];component_rows=[]')
s=s.replace('loss.backward();g_norm', "component_rows.append([*[float(v.detach())for v in components.values()],float(fine_value.detach()),float(style_loss.detach()),float(loss.detach())]);loss.backward();mixer_grad_rows.append([float(torch.linalg.vector_norm(p.grad))if p.grad is not None else 0. for p in model.time_mixer.parameters()]);g_norm")
s=s.replace("if step%cfg['validation_every']==0:\n                valrow", "if step==1:(root/'FIRST_UPDATE_MIXER_GRADIENTS.json').write_text(json.dumps({'parameter_order':list(dict(model.time_mixer.named_parameters())),'preclip_norms':mixer_grad_rows[-1]},indent=2)+'\\n','utf8')\n            if step%cfg['validation_every']==0:\n                fine.persist();valrow")
s=s.replace('fine.persist();assert fine.calls==updates', "fine.persist();np.savez_compressed(root/'MIXER_GRADIENT_NORMS_PRIVATE.npz',values=np.asarray(mixer_grad_rows,dtype='f8'));np.savez_compressed(root/'BASIC_LOSS_COMPONENTS_BY_UPDATE_PRIVATE.npz',values=np.asarray(component_rows,dtype='f8'));assert fine.calls==updates")
s=s.replace("'fresh_projection_calls':fine.calls", "'initial_equivalence':initial_equivalence,'time_mixer_parameters':123488,'time_mixer_final':{n:{'norm':float(torch.linalg.vector_norm(p.detach())),'finite':bool(torch.isfinite(p).all()),'last_preclip_gradient_norm':mixer_grad_rows[-1][j]if mixer_grad_rows else None}for j,(n,p)in enumerate(model.time_mixer.named_parameters())},'fresh_projection_calls':fine.calls")
(ROOT/'online_training.py').write_text(s,'utf8')
remote_cfg=json.loads(json.dumps(cfg));oldremote=json.loads((BASE/'remote_config.json').read_text('utf8'))
for k in ['training_input_paths','consumed_R26_stream','actual_training_random_states','actual_training_receipt','consumed_R26_receipt','G_initial','reader_initial','D_initial']:remote_cfg[k]=oldremote[k]
for n,c in [('config.json',cfg),('remote_config.json',remote_cfg)]: (ROOT/n).write_text(json.dumps(c,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
(ROOT/'START_RECEIPT.json').write_text(json.dumps(cfg['first_actual_command'],indent=2)+'\n','utf8')
source=[]
for p in ROOT.glob('*.py'):
    b=p.read_bytes();ast.parse(b.decode('utf8'));source.append({'name':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_AST':source,'dependency_payload_bytes':sum((ROOT/n).stat().st_size for n in payload),'science_started':False,'old_stage_writes':0,'migration_content_tests':0,'complete_neural_config_origin':str(BASE/'config.json')}
(ROOT/'PREPARATION_RECEIPT.json').write_text(json.dumps(rec,indent=2)+'\n','utf8');print(json.dumps({'PID':os.getpid(),'UTC':rec['UTC'],'AST_count':len(source),'dependency_bytes':rec['dependency_payload_bytes']}),flush=True)
