"""Assemble already-prepared code and native records; no scientific array reads."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import sys

P=Path(__file__).resolve().parent
R=P.parents[1]
T=R/'work/R77_FIXED25000_FULL_DATA_TRAIN_001'
TP=R/'reviews/R77_FIXED25000_FULL_DATA_TRAIN_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'assembly_runtime')

def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf8')as f:
        f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

now=datetime.now(timezone.utc).isoformat()
assert not(P/'REGISTRATION.json').exists()
native=read(T/'train_full.py.NATIVE_RECEIPT.json')
result=read(T/'RESULT.json')
assert native['actual_worker_native_exit']==0 and native['termination_reason']is None
assert result['training']['completed_updates']==25000
assert result['postcheck']['numerical_and_TRAIN_replay_pass']
assert sha(T/'FINAL_SHAPE.pt')==result['training']['final_weight_file_SHA256']
write(P/'USER_EVALUATION_INSTRUCTION.json',dict(recorded_UTC=now,
    latest_direct_user_instruction='去进行评价',
    prior_direct_user_instruction='不需要复算 是不是正在跑模型 还需要多久',
    formal_evaluation_authorized=True,extra_numeric_recomputation_cancelled=True,
    original_formal_4000_heldout_replay_retained=True,
    extra_ROOT_TRAIN_recalculation_cancelled=True,
    duplicate_summary_waveform_Q_ADC_CDF_migration_recalculation_cancelled=True,
    application_goal_last_observed_status='paused',
    bounded_current_evaluation_directly_authorized=True,
    additional_model_training_authorized_by_this_receipt=False))

sources={}
def add(p,name=None):
    name=name or p.name
    assert name not in sources and p.is_file(),(name,p)
    if p.suffix=='.py':ast.parse(p.read_text('utf8'))
    sources[name]=p

adapter=R/'reviews/R74_EXACT_LOSS_FULL_TRAIN_001/evaluation_candidate'
for name in ['evaluation_io.py','full_response_adapter.py','generate_full_response.py','replay_full_response.py']:
    add(adapter/name)
evaluation=P/'evaluator_candidate'
for name,row in read(evaluation/'COMPLETION_RECEIPT.json')['production_modules'].items():
    assert sha(evaluation/name)==row['sha256'],name
    if name!='evaluate_saved.py':add(evaluation/name)
variant=P/'summary_receipt_only'
assert(variant/'COMPLETION_RECEIPT.json').is_file()
add(variant/'evaluate_saved_receipt_variant.py','evaluate_saved.py')
add(variant/'summarize_and_plot.py')
for name in ['acceptance_gates.py','render_full_response.py']:
    add(P/'root_summary_candidate'/name)
for name in ['linux_owned_worker.py','linux_supervise.py']:
    add(T/name)
for name in ['pipeline_receipt.py','run_pipeline.sh']:
    add(P/'runtime_candidate'/name)
for p in [P/'prepare_register.py',P/'control.py',P/'collect.py']:
    ast.parse(p.read_text('utf8'))
rows=[dict(name=name,source=p.relative_to(R).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
      for name,p in sorted(sources.items())]
write(P/'STATIC_ASSEMBLY_RECEIPT.json',dict(UTC=now,
    root_static_integration_review_pass=True,production_sources=rows,
    review_scope='Prior static preparation plus minimal receipt-only summary adaptation; AST syntax and source binding only',
    no_new_test_suite=True,no_scientific_array_read=True,no_model_forward=True,
    no_numerical_recomputation=True,actual_scientific_pass_not_claimed=True,
    variant_completion_SHA256=sha(variant/'COMPLETION_RECEIPT.json'),
    guard={**guard,**memory()}))
write(TP/'NATIVE_COMPLETION_SUMMARY.json',dict(UTC=now,
    native_exit=native['actual_worker_native_exit'],worker_PID=native['actual_worker_PID'],
    completed_updates=25000,unique_endpoint=25000,
    training_elapsed_s=result['training']['training_elapsed_s'],
    final_weight_file_SHA256=result['training']['final_weight_file_SHA256'],
    postcheck=result['postcheck'],collection_receipt_SHA256=sha(TP/'COLLECTION_RECEIPT.json'),
    numerical_values_from_existing_native_receipts=True,root_extra_recomputation=False,
    heldout_acceptance_completed=False,model_goal_complete=False))
print(json.dumps(dict(assembled_files=len(rows),science_started=False,
    additional_numeric_recomputation=False,training_updates=25000)),flush=True)
