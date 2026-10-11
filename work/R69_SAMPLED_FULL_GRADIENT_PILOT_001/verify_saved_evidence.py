"""Verify saved R69 scalar gates only; no model, fitting, or new array access."""
from pathlib import Path
from fractions import Fraction as F
import json,math,sys
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply
apply(R/'reviews/R69_SAMPLED_FULL_GRADIENT_PILOT_001/verify_runtime')
def read(p):return json.loads(p.read_text('utf8'))
v=read(S/'RESULT.json');a=read(S/'GRADIENT_ADMISSION.json')
M=a['original_block_count'];k=a['sampled_block_count']
variance=F(M*(M-k),k*(M-1))*(F.from_float(a['sum_block_squared_norm'])-F.from_float(a['D_squared_norm'])/M)
assert math.isclose(float(variance),a['conditional_MSE_trace'],rel_tol=2e-12)
assert a['pass_gate']==(float(variance)<=a['D_squared_norm']and float(variance)<=a['D_plus_C_squared_norm'])
count=0;fail=0
if v['status']in['POSITIVE_LOCAL_ONLY','NEGATIVE_LOCAL_SUPPORT']:
 ev=[read(S/n/'EVALUATION.json')for n in ['baseline','candidate','mini']]
 scores=[[r['score']for r in e['probes']]for e in ev]
 for e in ev:
  assert len(e['probes'])==8 and e['SOURCE_N']==6593 and e['REAL_N']==21047
  for r in e['probes']:assert math.isclose(r['score'],sum(r['pieces'].values()),rel_tol=2e-12)
 b,f,m=[sum(s)/8 for s in scores];wins=sum(x<y for x,y in zip(scores[1],scores[2]))
 primary=f<=.9*m and f<=.9*b and wins>=6
 for group in ['pieces','nonnegative','correlations','valleys','REAL_pieces','REAL_nonnegative']:
  rg=group.startswith('REAL_');g=group[5:]if rg else group
  keys=ev[0]['REAL_control'][g]if rg else ev[0]['probes'][0][g]
  for key in keys:
   seq=[[e['REAL_control'][g][key]]if rg else[r[g][key]for r in e['probes']]for e in ev]
   if any(x is None for s in seq for x in s):ok=False
   else:
    bv,fv,mv=[sum(s)/len(s)for s in seq]
    ok=fv>=max(bv,mv)-.02 if g=='correlations'else fv<=min(bv,mv)+.01 if g=='valleys'else fv<=min(bv,mv)+.05*max(bv,mv)+1e-10
   old=v['harm_checks'][count];assert(old['group'],old['key'],old['pass_gate'])==(group,key,ok)
   count+=1;fail+=int(not ok)
 assert count==102 and primary==v['primary']['pass_gate']
 assert bool(primary and not fail)==(v['status']=='POSITIVE_LOCAL_ONLY')
elif v['status']=='NEGATIVE_GRADIENT_ADMISSION':assert not a['pass_gate']and v['new_response_updates']==v['new_response_probes']==0
assert not v['original_A_F_complete']and not v['automatic_long_resume']
print(json.dumps(dict(status='PASS_SAVED_SCALAR_VERIFICATION',result_status=v['status'],gradient_gate=a['pass_gate'],harm_checks=count,failed_harms=fail,new_scientific_runs=0,original_A_F_complete=False)))
