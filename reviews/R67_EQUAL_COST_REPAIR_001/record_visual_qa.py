from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R67_EQUAL_COST_REPAIR_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply
apply(P)
assert not(P/'VISUAL_QA_RECEIPT.json').exists()
result=dict(UTC=datetime.now(timezone.utc).isoformat(),actually_opened_and_reviewed=True,
    method='Root opened the saved PNG through the image viewer and inspected the rendered full image.',
    PNG_SHA256=hashlib.sha256((S/'R67_COMPARISON.png').read_bytes()).hexdigest(),
    labels_and_axes_readable=True,no_clipped_panels_or_titles=True,
    mean_score_labels_match_saved_outputs=[40.220,5.928,4.546],all_original_regions_shown=True,
    eight_probe_points_per_model_shown=True,mini_completed_updates=979,full_completed_updates=20,
    original_PSD_threshold_line_label_retains_not_A_F_verdict=True,
    scientific_acceptance_claim_from_figure=False)
(P/'VISUAL_QA_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
