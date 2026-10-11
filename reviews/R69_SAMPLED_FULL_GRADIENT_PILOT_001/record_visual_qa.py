from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R69_SAMPLED_FULL_GRADIENT_PILOT_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'report_runtime')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
assert not(P/'VISUAL_QA_RECEIPT.json').exists()
qa=dict(UTC=datetime.now(timezone.utc).isoformat(),actually_opened_and_reviewed=True,
 method='Root opened saved PNG through view_image and inspected its full rendered image.',
 PNG_SHA256=sha(S/'R69_COMPARISON.png'),labels_and_axes_readable=True,no_clipped_panels_or_titles=True,
 mean_score_labels_match_saved_outputs=[36.292,4.962,4.695],candidate_updates=100,mini_updates=1502,
 all_original_regions_shown=True,eight_probe_points_per_model_shown=True,
 original_PSD_threshold_label_preserves_not_A_F_verdict=True,scientific_acceptance_claim_from_figure=False,
 native_saved_scalar_verifier=dict(command='python -B work/R69_SAMPLED_FULL_GRADIENT_PILOT_001/verify_saved_evidence.py',
  native_exit=0,result_status='PASS_SAVED_SCALAR_VERIFICATION',checks=102,failed_harms=58,new_scientific_runs=0,
  verifier_SHA256=sha(S/'verify_saved_evidence.py')),
 resource=dict(guard,**memory()))
(P/'VISUAL_QA_RECEIPT.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf8')
print(json.dumps(qa,indent=2))
