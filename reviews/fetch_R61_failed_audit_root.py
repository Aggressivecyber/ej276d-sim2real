"""Normal artifact transfer after the confirmed pre-forward JSON failure."""
from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parents[1]
source=(R/'reviews/fetch_R61_root.py').read_text('utf8')
old="a.get('actual_worker_native_exit')==0"
assert source.count(old)==1
source=source.replace(old,"a.get('actual_worker_native_exit')==1 and a['entry']=='numerical_check.py' and 'bool_' in a['worker_self_receipt']['error']")
source=source.replace("('FETCH_'+required+'.json')","('FETCH_AFTER_AUDIT_JSON_FAILURE.json')")
sys.argv=[sys.argv[0],'train_final.py']
exec(compile(source,str(__file__),'exec'))
