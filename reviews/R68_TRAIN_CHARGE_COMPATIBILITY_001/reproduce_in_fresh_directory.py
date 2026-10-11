"""Optional explicit local replay in a new E/D directory; preserve sealed outputs."""
from pathlib import Path
import argparse,importlib.util,shutil,time,sys,subprocess,json
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=ap.parse_args()
out=Path(args.output).resolve()
assert out.drive.upper() in {'E:','D:'} and not out.exists(),'Use a NEW directory on E or D.'
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R68_TRAIN_CHARGE_COMPATIBILITY_001'
if not args.worker:
 sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
 from local_resource_guard import apply
 apply(P/'reproduction_controller')
 start=time.monotonic();proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(out),'--worker'])
 reason=None
 try:proc.wait(timeout=60)
 except subprocess.TimeoutExpired:reason='hard_60_second_replay_cap';proc.kill();proc.wait(timeout=10)
 if out.exists():(out/'REPLAY_NATIVE_RECEIPT.json').write_text(json.dumps(dict(PID=proc.pid,native_exit=proc.returncode,elapsed_s=time.monotonic()-start,termination_reason=reason),indent=2)+'\n',encoding='utf8')
 sys.exit(0 if proc.returncode==0 else 1)
spec=importlib.util.spec_from_file_location('frozen_R68_certificate',S/'charge_certificate.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
out.mkdir(parents=True)
shutil.copy2(S/'config.json',out/'config.json')
mod.S=out;mod.GUARD=mod.apply(out);mod.START=time.monotonic()
mod.main()
