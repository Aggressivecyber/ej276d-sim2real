"""Prepare normal private-Git and D-drive delivery; execute only after final QA."""
from pathlib import Path
import ast,json
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
text=(R/'reviews/deliver_R59_root.py').read_text('utf8')
text=text.replace('R59_DIRECT_DISTRIBUTION_RESPONSE_001',S.name).replace('R59','R61').replace('R58','R60')
text=text.replace('r59_direct_distribution_final_','r61_precision_order_final_').replace('r59_science_incremental.bundle','r61_science_incremental.bundle')
old="reg=[rel(p) for d in ['R61_REGISTRATION_001','R61_POST_PREEXECUTION_001','R61_TRANSFER_RECOVERY_001','R61_FIGURE_REPAIR_001'] for p in (R/'reviews'/d).iterdir() if p.is_file()]"
new="reg=[rel(p) for d in ['R61_REGISTRATION_001','R61_ARTIFICIAL_RECEIPT_REPAIR_001','R61_POST_AUDIT_ARTIFICIAL_001'] for p in (R/'reviews'/d).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md','.png'] and 'cache' not in p.parts]\nrootfiles+=['reviews/R61_POSTPROCESS_READY_001.json']"
assert text.count(old)==1;text=text.replace(old,new)
needle="meta=['AGENTS.md'"
guard="""final=read('work/'+STAGE+'/FINAL_RECEIPT.json')
assert final['status']=='scientific_endpoint_complete_delivery_pending'
assert read('work/'+STAGE+'/FIGURE_QA.json')['all_figures_opened']
assert read('work/'+STAGE+'/FROZEN_ARTIFACT_REVIEW.json')['pass']
assert read('work/'+STAGE+'/TRAINING_FROZEN_AUDIT.json')['pass']
assert not read('work/'+STAGE+'/LIVE_OBSERVATION.json')['actual_process']['exists']
"""
assert text.count(needle)==1;text=text.replace(needle,guard+needle)
text=text.replace('Record direct unpaired full-window response training and unchanged joint acceptance','Record precision-resolved source-order ablation and unchanged joint acceptance')
old='R61 direct-distribution incremental backup.'
new='R61 precision-resolved source-order incremental backup.'
assert text.count(old)==1;text=text.replace(old,new)
text=text.replace('R61 portable inference weights and constants, final random state and parent exposure counts, private outputs, descriptions and predictions are included.',
    'R61 portable inference weights and constants, final random state, R59-matched parent exposure counts, precision-order traces and intervals, private outputs, descriptions and predictions are included.')
text=text.replace('no teacher coupling is used.',
    'Frozen R59 training records and distribution statistics are required for the registered single-factor audit; they are retained through the prior R60/R59 backup chain. No teacher coupling is used.')
dest=R/'reviews/deliver_R61_root.py';assert not dest.exists();ast.parse(text,filename=str(dest));dest.write_text(text,'utf8')
print(json.dumps({'delivery_script_prepared':str(dest.relative_to(R)),'delivery_executed':False,'publication_executed':False,'D_copy_executed':False}))
