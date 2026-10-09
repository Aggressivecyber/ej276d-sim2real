from pathlib import Path
import ast
p=Path(__file__).resolve().parent;s=(p/'_WORKER_PREAMBLE.txt').read_text()+(p/'resume_body.txt').read_text();ast.parse(s);(p/'numeric_resume.py').write_text(s)
r=(p.parent/'replay.py').read_text().replace("for name in cfg['models']:","for name in ['GMM8']:")
r=r.replace("'total_rows':sum(r.get('rows',0) for r in results.values())","'total_rows':sum(r.get('rows',0) for r in results.values()),'Gaussian2000_already_passed_in_original_replay':True,'combined_replayed_rows':2000+sum(r.get('rows',0) for r in results.values())")
(p/'replay_gmm.py').write_text(r)
print('resume AST ready; no scientific execution')
