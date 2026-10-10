"""Prepare one explicitly reviewed numerical retry and continuation controller."""
from pathlib import Path
import ast,json
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001';A=S/'repairs/AUDIT_JSON_001'
assert json.loads((A/'DEPLOY_RECEIPT.json').read_text('utf8'))['verified']['all_31_file_hashes_verified']
old=(R/'reviews/launch_R61_entry_root.py').read_text('utf8')
old_guard="assert active['entry']==previous and active.get('actual_worker_native_exit')==0 and active['termination_reason']is None,active"
new_guard="""assert entry=='numerical_check.py' and active['entry']==entry and active.get('actual_worker_native_exit')==1 and active['termination_reason']is None,active
assert 'bool_' in active['worker_self_receipt']['error'] and active['GPU_process_memory_peak_bytes']==0
review=json.loads((p/'repairs/AUDIT_JSON_001/REPAIR_REVIEW.json').read_text())
assert review['pass'] and review['all_28_checks_pass'] and review['all_audit_values_equal_before_after']
assert hashlib.sha256((p/'post_audit.py').read_bytes()).hexdigest()==review['new_post_audit_SHA256']
assert hashlib.sha256((p/arm/'FINAL_FLOW.pt').read_bytes()).hexdigest()==review['frozen_weight_SHA256']
assert (p/'repairs/AUDIT_JSON_001/LAUNCH_numerical_check.py.json').is_file()
assert not (p/'TRAIN_SOLVER_CHECK.json').exists() and not (p/'GENERATION_RECEIPT.json').exists()"""
assert old.count(old_guard)==1
retry=old.replace(old_guard,new_guard)
watch=(R/'reviews/watch_R61_pipeline_root.py').read_text('utf8')
before="events=[];current='train_final.py';last=None"
after="events=[{'kind':'root_reviewed_serialization_repair','record':'repairs/AUDIT_JSON_001/REPAIR_REVIEW.json','failed_attempt_retained':True}];current='numerical_check.py';last=None"
assert watch.count(before)==1
watch=watch.replace(before,after).replace("S/'LOCAL_PIPELINE_CONTROL.started.json'","S/'LOCAL_PIPELINE_CONTROL_AUDIT_JSON_REPAIR.started.json'")
for name,source in [('retry_R61_numerical_after_json_repair_root.py',retry),('watch_R61_repaired_pipeline_root.py',watch)]:
    ast.parse(source);p=R/'reviews'/name
    with p.open('x',encoding='utf8',newline='\n')as f:f.write(source)
print(json.dumps({'prepared':True,'one_reviewed_retry':'numerical_check.py','training_rerun':False}))
