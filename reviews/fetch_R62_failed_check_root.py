from pathlib import Path
R=Path(__file__).resolve().parents[1]
src=(R/'reviews/fetch_R62_root.py').read_text('utf8')
src=src.replace("a.get('actual_worker_native_exit')==0","a.get('actual_worker_native_exit')==1")
src=src.replace("n['actual_worker_native_exit']==0 and n['worker_self_receipt']['native_exit']==0","n['actual_worker_native_exit']==1 and n['worker_self_receipt']['native_exit']==1")
src=src.replace("'FETCH_RECEIPT.json'","'FETCH_AFTER_ARTIFICIAL_FAILURE.json'")
exec(compile(src,str(__file__),'exec'))
