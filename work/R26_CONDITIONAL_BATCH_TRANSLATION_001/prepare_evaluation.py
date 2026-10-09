from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R24_CRITIC_UPDATE_RATIO_001'
src=(OLD/'evaluate_selected.py').read_text('utf8');tree=ast.parse(src)
keep=['psd_metrics','source_composition','probes','joint_repeats','online_auc']
out='\n'.join(ast.get_source_segment(src,x)for x in tree.body if isinstance(x,ast.FunctionDef)and x.name in keep)
(ROOT/'endpoint_common.py').write_text(out+'\n','utf8')
replay=(OLD/'replay_selected.py').read_text('utf8')
(ROOT/'REPLAY_SOURCE_TEMPLATE.txt').write_text(replay,'utf8')
print('extracted five existing evaluation functions; no scientific execution')
