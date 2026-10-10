"""Prepare the preregistered R61 endpoint checks; never run or alter training."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, shutil

R=Path(__file__).resolve().parents[1]
S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
O=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
C=json.loads((S/'config.json').read_text('utf8'))
ARM=C['arm'];CTRL='PRECISION_RESOLVED_REAL_CONTROL'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bound=json.loads((S/'ROOT_IMPLEMENTATION_REVIEW.json').read_text('utf8'))['files']
protected=[S/row['name'] for row in bound]+[S/'ROOT_IMPLEMENTATION_REVIEW.json',R/C['protocol'],R/'PROJECT_ACCEPTANCE.json',R/'PROJECT_ACCEPTANCE.md']
before={str(p):sha(p) for p in protected}
for row in bound:assert sha(S/row['name'])==row['sha256'],row['name']
P=json.loads((O/'config_post.json').read_text('utf8'));P.update(C)
P.update(endpoints=[ARM,CTRL],REAL_arms=['BASE_REAL_CONTROL',CTRL],
    endpoint_weight_paths={n:C['remote_root']+'/'+ARM+'/FINAL_FLOW.pt' for n in [ARM,CTRL]},
    historical_training_root=C['remote_root'],
    parameter_initialization='same fresh seed 20261010590 and initial tensor hash as frozen R59',
    objective='R59 seven losses with precision-resolved source-order numerator; original denominator and other six losses unchanged',
    scientific_scope='R61 one-factor precision-order ablation versus frozen R59; original A-F unchanged',
    execution={'training':'one fresh R59-matched network, exactly50000 updates; sole loss modification is precision-order numerator',
               'inference':'same SOURCE/REAL operator, Heun128 cached displacement and explicit box-charge projection'})
assert not (S/'config_post.json').exists()
(S/'config_post.json').write_text(json.dumps(P,indent=2)+'\n','utf8')
files=['canonical.py','flow_runtime.py','numerical_check.py','generate_all.py','evaluate_saved.py',
    'replay_all.py','evaluation.py','diagnostic_core.py','assessment.py','common_math.py','energy_width.py',
    'data_support.py','model.py','legacy_model.py','flow_core.py','endpoint_common.py','flow_diagnostics.py',
    'infer_saved.py','summarize_and_plot.py','render_R59_figures.py','post_audit.py']
for name in files:
    dest=S/name.replace('render_R59','render_R61');assert not dest.exists(),dest
    text=(O/name).read_text('utf8')
    text=text.replace('DIRECT_DISTRIBUTION_RESPONSE',ARM).replace('DIRECT_DISTRIBUTION_REAL_CONTROL',CTRL)
    text=text.replace('R59','R61').replace('R54','R59').replace('r54','r59')
    text=text.replace('CONTINUOUS_Q_RESPONSE','DIRECT_DISTRIBUTION_RESPONSE').replace('CONTINUOUS_Q_REAL_CONTROL','DIRECT_DISTRIBUTION_REAL_CONTROL')
    dest.write_text(text,'utf8')
shutil.copy2(O/'TRAIN_SOLVER_CHECK.json',S/'HISTORICAL_R59_TRAIN_SOLVER_CHECK.json')
assert json.loads((S/'HISTORICAL_R59_TRAIN_SOLVER_CHECK.json').read_text())['both_pass']

def replace(name,old,new):
    p=S/name;text=p.read_text('utf8');assert text.count(old)==1,(name,old,text.count(old))
    p.write_text(text.replace(old,new),'utf8')

replace('numerical_check.py','from post_audit import constraint_description','from post_audit import constraint_description,training_audit')
replace('numerical_check.py',"R/'HISTORICAL_TRAIN_SOLVER_CHECK.json'","R/'HISTORICAL_R59_TRAIN_SOLVER_CHECK.json'")
replace('numerical_check.py',"    net=FlowNetwork();", "    audit=training_audit(R,C,training);write('TRAINING_FROZEN_AUDIT.json',audit)\n    assert audit['pass'],audit['checks']\n    net=FlowNetwork();")
replace('post_audit.py','import evaluation as ev','import evaluation as ev\nfrom precision_audit import precision_single_factor_audit')
replace('post_audit.py',"    fields=['SOURCE_global_range_projection_fraction'", "    precision=precision_single_factor_audit(root,cfg,receipt,losses,columns)\n    checks.update(precision['checks'])\n    fields=['SOURCE_global_range_projection_fraction'")
replace('post_audit.py',"'comparison_with_R59_is_single_factor_ablation':False", "'comparison_with_R59_is_single_factor_ablation':precision['pass'],\n        'single_factor_precision_order':precision")
replace('evaluate_saved.py',"'single_factor_ablation':False,'independent_fresh_training_recipe':True", "'single_factor_ablation':json.loads((ROOT/'TRAINING_FROZEN_AUDIT.json').read_text())['pass'],'single_training_seed':True,'training_seed_variability_not_estimated':True")
replace('evaluate_saved.py',"R61 sole final50000 direct unpaired distribution learning; all original TRAIN parents eligible; new sampling streams", "R61 sole final50000 precision-order ablation; same R59 initialization, six consumed streams, parents and frozen constants")
replace('summarize_and_plot.py',"'R61 direct unpaired full480 distribution network, original-parent global weights, explicit box-charge constraint and same-REAL identity'", "'R61 precision-resolved source-order numerator ablation; original full480 network, parent weights, other six losses and same-REAL operator'")
path=S/'summarize_and_plot.py';text=path.read_text('utf8').replace('close_fixed_R61_direct_distribution_recipe','close_fixed_R61_precision_order_ablation');path.write_text(text,'utf8')

# Use count displays when a projection quantity is exactly zero, avoiding empty
# histograms with negative automatic axes. This only affects saved-figure layout.
path=S/'render_R61_figures.py';text=path.read_text('utf8')
start=text.index('    fig,ax=plt.subplots(1,3,figsize=(12.5,4.1))')
text=text[:start]+'''    fig,ax=plt.subplots(1,3,figsize=(14.5,4.8))
    saved={n:load(R/n/'generated.npz') for n in [arm,ctrl]}
    all_zero=all(np.all(d['constraint_projection_RMS_V']==0) for d in saved.values())
    all_rails_zero=all(np.all(d['constraint_projection_rail_sample_count']==0) for d in saved.values())
    short=['SOURCE','Same-treated REAL']
    if all_zero:
        ax[0].bar(short,[2000,2000],color=[colors[arm],colors[ctrl]],width=.6)
        for i in range(2):ax[0].text(i,2030,'2,000 / 2,000',ha='center',fontsize=9)
        ax[0].set(ylim=(0,2300),ylabel='Events with zero correction',title='Exact zero correction in main outputs')
    else:
        for n,d in saved.items():
            r=d['constraint_projection_RMS_V']*1000;nonzero=r[r>0]
            if len(nonzero):ax[0].hist(nonzero,bins=40,histtype='step',label=names[n],color=colors[n])
            else:ax[0].plot([],[],label=names[n]+' (all zero)',color=colors[n])
        ax[0].set(xlim=(0,None),ylim=(0,None),xlabel='Nonzero RMS correction (mV)',ylabel='Events',title='Explicit constraint contribution');ax[0].legend(fontsize=8)
    if all_rails_zero:
        ax[1].bar(short,[2000,2000],color=[colors[arm],colors[ctrl]],width=.6)
        for i in range(2):ax[1].text(i,2030,'2,000 / 2,000',ha='center',fontsize=9)
        ax[1].set(ylim=(0,2300),ylabel='Events with no rail samples',title='No endpoint samples on voltage rails')
    else:
        for n,d in saved.items():ax[1].hist(d['constraint_projection_rail_sample_count'],bins=np.arange(-.5,481,10),histtype='step',label=names[n],color=colors[n])
        ax[1].set(xlim=(0,480),ylim=(0,None),xlabel='Rail samples per event',ylabel='Events',title='Bounded endpoint saturation');ax[1].legend(fontsize=8)
    a=M['explicit_constraint_projection'][arm];b=M['explicit_constraint_projection'][ctrl]
    frac=M['training_audit']['constraint_loss_trace_summary']['SOURCE_global_range_projection_fraction']['mean'] if 'training_audit' in M else audit['constraint_loss_trace_summary']['SOURCE_global_range_projection_fraction']['mean']
    info='Range-active main events\\nSOURCE: '+str(a['range_active_events'])+' / 2000\\nREAL: '+str(b['range_active_events'])+' / 2000\\n\\n'
    info+='Projection remains part of the model.\\nZero ADC clipping does not establish\\nzero internal correction.\\n\\n'
    info+=f'TRAIN update-mean range activation: {100*frac:.2f}%.\\nWeighted minibatch trace; not exhaustive\\nfrozen-TRAIN inference.'
    ax[2].axis('off');ax[2].text(.02,.97,info,va='top',linespacing=1.45,fontsize=9)
    save(fig,'explicit_constraint_projection.png')

    precision=load(R/arm/'PRECISION_ORDER_BY_UPDATE_PRIVATE.npz')
    values=precision['values'];pn=precision['columns'].tolist();steps=np.arange(1,51)*1000
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for key,label in [('original_order_loss','Original order penalty'),('omitted_order_loss','Omitted overlap penalty')]:
        ax[0].plot(steps,values[:,pn.index(key)].reshape(-1,1000).mean(1),label=label)
    ax[0].plot(steps,loss[:,cols.index('source_order')].reshape(-1,1000).mean(1),label='Retained resolved penalty')
    ax[0].set(xlabel='Update (sole final 50,000)',ylabel='Mean loss per 1,000 updates',title='Original = retained + omitted');ax[0].legend(fontsize=8)
    ax[1].plot(steps,values[:,pn.index('resolved_pair_fraction')].reshape(-1,1000).mean(1),color='#136bc2')
    ax[1].set(ylim=(0,1),xlabel='Update',ylabel='Resolved fraction of original pair mass',title='Original denominator retained')
    fig.suptitle('Precision-order trace: training surrogate; no acceptance exemptions')
    save(fig,'precision_order_decomposition.png')
'''
path.write_text(text,'utf8')
Q=dict(C);Q.update(allowed_scientific_entries=['summarize_and_plot.py'],GPU_scientific_entries=[],reporting_only=True,new_fit_or_inference_authorized=False)
(S/'config_reporting.json').write_text(json.dumps(Q,indent=2)+'\n','utf8')
(S/'linux_supervise_reporting.py').write_text((S/'linux_supervise.py').read_text('utf8').replace("ROOT/'config.json'","ROOT/'config_reporting.json'"),'utf8')
for p in S.glob('*.py'):ast.parse(p.read_text('utf8'),filename=str(p))
assert before=={str(p):sha(p)for p in protected}
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','training_files_and_protocol_acceptance_untouched':before,
    'new_generation_rows':5984,'replay_rows':4000,'new_HGB_fits':10,'density_slots':16,
    'frozen_R59_control_new_fits':0,'single_factor_integrity_required_before_numerical_inference_and_DEV':True,
    'same_raw_SOURCE_definitions':True,'all_asts_parsed':True,'new_scientific_execution':0}
(S/'POSTPROCESS_PREPARATION.json').write_text(json.dumps(receipt,indent=2)+'\n','utf8')
print(json.dumps({'post_files_prepared':len(files)+5,'training_code_protocol_acceptance_untouched':True,'AST':'pass'}))
