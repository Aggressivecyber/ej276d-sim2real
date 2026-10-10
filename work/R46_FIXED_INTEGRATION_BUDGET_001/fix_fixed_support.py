"""Pre-evaluation fixed-support definition correction; no numerical work."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os,shutil
R=Path(__file__).resolve().parent;old=R/'engineering_evaluation_definition_001';old.mkdir(exist_ok=False)
names=['budget_comparison.py','evaluate_saved.py','summarize_and_plot.py','FULL_PIPELINE_FREEZE.json']
for n in names:shutil.copyfile(R/n,old/n)
p=R/'budget_comparison.py';s=p.read_text('utf8')
s=s.replace("pn=metrics['PSD_main'];bins=[i for i in range(8)if all(pn['tables'][n]['bins'][i]['N']>=40 for n in ['REAL',short,long])]",
 "pn=metrics['PSD_main'];bins=list(range(7));qmissing={n:[i+1 for i in bins if pn['tables'][n]['bins'][i]['N']<40]for n in ['REAL',short,long]}")
s=s.replace("def mean(name):return float(np.mean([pn['vs_REAL'][name]['bins'][i]['W1']for i in bins]))if bins else None",
 "def mean(name):return float(np.mean([pn['vs_REAL'][name]['bins'][i]['W1']for i in bins]))if not qmissing['REAL']and not qmissing[name]else None")
s=s.replace("common_bins_1based=[i+1 for i in bins])", "common_bins_1based=[i+1 for i in bins],fixed_bins_not_candidate_intersection=True,unsupported_required_bins_by_domain=qmissing)")
s=s.replace("good=[i for i in range(8)if all(wn[n]['slices'][i]['valid']and len(wn[n]['slices'][i]['components'])==2 and wn[n]['slices'][i]['components'][k][field]is not None for n in ['REAL',short,long])]",
 "good=list(range(1,7));missing={n:[i+1 for i in good if not(wn[n]['slices'][i]['valid']and len(wn[n]['slices'][i]['components'])==2 and wn[n]['slices'][i]['components'][k][field]is not None)]for n in ['REAL',short,long]}")
s=s.replace("for i in good]))if good else None", "for i in good]))if not missing['REAL']and not missing[name]else None")
s=s.replace("common_bins_1based=[i+1 for i in good])", "common_bins_1based=[i+1 for i in good],fixed_bins_not_candidate_intersection=True,failed_required_bins_by_domain=missing)")
assert 'range(8)if all'not in s;ast.parse(s);p.write_bytes(s.encode())
p=R/'evaluate_saved.py';s=p.read_text('utf8')
oldline="bins=[i for i in range(8)if all(tables[n]['slices'][i]['valid']and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field]is not None for n in['REAL',*names])]"
assert oldline in s
s=s.replace(oldline,"bins=list(range(1,7));missing={n:[i+1 for i in bins if not(tables[n]['slices'][i]['valid']and len(tables[n]['slices'][i]['components'])==2 and tables[n]['slices'][i]['components'][k][field]is not None)]for n in ['REAL',*names]}")
s=s.replace("'common_bin_ids_1based':[i+1 for i in bins],", "'common_bin_ids_1based':[i+1 for i in bins],'fixed_required_bins_not_candidate_intersection':True,'failed_required_bins_by_domain':missing,")
s=s.replace("for i in bins]))if bins else None for n in names", "for i in bins]))if not missing['REAL']and not missing[n]else None for n in names")
ast.parse(s);p.write_bytes(s.encode())
p=R/'summarize_and_plot.py';s=p.read_text('utf8')
s=s.replace("common=[i for i in range(8)if all(psd['tables'][n]['bins'][i]['N']>=40 for n in ['REAL','BASE',arm])]",
 "common=list(range(7));missingQ={n:[i+1 for i in common if psd['tables'][n]['bins'][i]['N']<40]for n in ['REAL','BASE',arm]}")
s=s.replace("for i in common]))if common else None for n in ['BASE',arm]", "for i in common]))if not missingQ['REAL']and not missingQ[n]else None for n in ['BASE',arm]")
s=s.replace("'common_bins_1based':[i+1 for i in common],'N_each_domain_ge':40", "'common_bins_1based':[i+1 for i in common],'N_each_domain_ge':40,'fixed_required_bins':True,'unsupported_required_bins_by_domain':missingQ")
s=s.replace("required=[i for i in range(1,7)if width['tables']['BASE']['slices'][i]['valid']]", "required=list(range(1,7));assert all(width['tables']['BASE']['slices'][i]['valid']for i in required)")
ast.parse(s);p.write_bytes(s.encode())
rec={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'phase':'before any HGB or density execution',
 'science_attempts_or_generation_added':0,'HGB_added':0,'EM_added':0,
 'rule':'fixed PSDQ1-7 and peak/raw width originalQ2-7, failed required bins retained as undefined; no candidate-valid intersection',
 'old_sources_and_freeze_preserved':str(old),'files':{n:{'bytes':(R/n).stat().st_size,'sha256':hashlib.sha256((R/n).read_bytes()).hexdigest()}for n in names[:3]}}
(R/'FIXED_SUPPORT_CORRECTION_RECEIPT.json').write_bytes((json.dumps(rec,indent=2)+'\n').encode())
print(json.dumps(rec))
