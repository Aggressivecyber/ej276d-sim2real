"""Record root's actual image inspection after the annotation layout repair."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R68_TRAIN_CHARGE_COMPATIBILITY_001'
png=S/'R68_CHARGE_ENVELOPE.png';h=hashlib.sha256(png.read_bytes()).hexdigest()
assert h=='a2c65c4d3c2c21d97a1971a9ccdfc5267cda1656b18ef01a70c81f26df83af23'
qa=dict(UTC=datetime.now(timezone.utc).isoformat(),actually_opened_and_reviewed=True,viewer='view_image actual rendered pixels',
        PNG_SHA256=h,two_panels_checked=['full original TRAIN range','low-charge detail with exact witness'],
        annotation_moved_after_first_review=True,final_text_data_overlap=False,final_text_clipping=False,
        legend_and_axis_units_readable=True,all_population_counts_and_scope_caption_present=True,
        exact_witness_JSON_matches_annotation=True,display_only_zoom_not_population_filter=True,
        Codex_panel_open_result='queued_thread_not_visible',user_panel_immediate_open_claim=False)
(P/'VISUAL_QA_RECEIPT.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(qa,ensure_ascii=False))
