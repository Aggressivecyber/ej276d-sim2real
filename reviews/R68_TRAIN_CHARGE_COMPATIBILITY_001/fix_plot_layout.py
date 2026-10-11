"""Apply the visual-QA annotation adjustment without rerunning any analysis."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction as F
import json,sys,hashlib,shutil
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R68_TRAIN_CHARGE_COMPATIBILITY_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'plot_layout_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
old=P/'before_plot_QA_adjustment';old.mkdir()
for name in ['R68_CHARGE_ENVELOPE.png','R68_CHARGE_ENVELOPE.svg']:shutil.copy2(S/name,old/name)
shutil.copy2(P/'REPORT_RENDER_RECEIPT.json',old/'REPORT_RENDER_RECEIPT.json')
x=read(S/'RESULT.json');b=x['robust_interval_result'];w=b['witness'];ns=w['SOURCE_N'];nr=w['REAL_N']
exact=F(b['KS_lower_bound']['numerator'],b['KS_lower_bound']['denominator'])
source=(P/'review_and_report.py').read_text('utf8')
plot=source.split('# Plotting reads only the output curve already computed by the completed worker.\n',1)[1].split("for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:",1)[0]
assert "xytext=(2.05,.085)" in plot and "RESULT.json" not in plot and 'certificate(' not in plot
exec(compile(plot,str(P/'review_and_report.py')+':plot_only','exec'),globals())
write(P/'REPORT_RENDER_RECEIPT.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PNG_SHA256=sha(S/'R68_CHARGE_ENVELOPE.png'),
      SVG_SHA256=sha(S/'R68_CHARGE_ENVELOPE.svg'),source='saved_CDF_ENVELOPE_PRIVATE.npz',new_waveform_or_model_evaluations=0,
      visual_adjustment='Moved gap annotation below the REAL curve after first actual image inspection.',resource=dict(guard,**memory())))
print(json.dumps(dict(PNG_SHA256=sha(S/'R68_CHARGE_ENVELOPE.png'),resource=dict(guard,**memory())),ensure_ascii=False))
