"""Keep final requirement readouts tied to the current evaluated method."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
R=Path(__file__).resolve().parents[1]

finalizer=R/'reviews/finalize_R61_root.py';old=finalizer.read_text('utf8')
anchor="    write(name,x)\nfor name,key,ident in [('jobs.json','jobs','stage')"
assert old.count(anchor)==1
block="""    if name=='PROJECT_PROGRESS.json':
        evidence=['work/'+ID+'/ACCEPTANCE_A_C.json','work/'+ID+'/GATE_METRICS.json','R61_REPORT.md']
        updates={
            'target_distribution_learning':('R61_A_pass' if A['A']['pass'] else 'R61_A_failed_original_thresholds_retained',evidence),
            'G4_conditioned_full_response':('R61_A_C_pass' if gate['primary_A_C_pass'] else 'R61_joint_A_C_not_passed',evidence),
            'charge_conditional_width_noise_waveform':('R61_B_pass' if A['B']['pass'] else 'R61_B_failed_original_thresholds_retained',evidence),
            'task_utility':('not_evaluated_for_frozen_R61',['PROJECT_ACCEPTANCE.json','reviews/FROZEN_A_F_VALIDATION_PREPARATION_001.md']),
            'charge_gap_generalization':('not_evaluated_for_frozen_R61',['PROJECT_ACCEPTANCE.json','reviews/FROZEN_A_F_VALIDATION_PREPARATION_001.md']),
            'reproducible_final_model':('R61_inference_and_replay_verified_full_A_F_unproven',['work/'+ID+'/FROZEN_ARTIFACT_REVIEW.json','work/'+ID+'/REPLAY_RECEIPT.json']),
            'final_research_delivery':('R61_scientific_endpoint_complete_private_Git_D_pending',['work/'+ID+'/FINAL_RECEIPT.json'])}
        for row in x.get('requirements',[]):
            if row['id']not in updates:continue
            row.setdefault('prior_status_records',[]).append({'status':row['status'],'latest_evidence':row.get('latest_evidence',[])})
            row['status'],row['latest_evidence']=updates[row['id']]
            row['candidate_stage']=ID
        x['full_A_F_proven']=False
"""
new=old.replace(anchor,block+anchor);ast.parse(new,filename=str(finalizer));finalizer.write_text(new,'utf8')

delivery=R/'reviews/deliver_R61_root.py';old_delivery=delivery.read_text('utf8')
anchor=" write(fn,x)\nfor fn,key,identity in [('jobs.json','jobs','stage')"
assert old_delivery.count(anchor)==1
block=""" if fn=='PROJECT_PROGRESS.json':
  for row in x.get('requirements',[]):
   if row['id']=='final_research_delivery':
    row['status']='R61_stage_delivery_complete_full_model_goal_unmet'
    row['latest_evidence']=['R61_REPORT.md','R61_PUBLICATION_RECEIPT.json','R61_BACKUP_RECEIPT.json']
    row['candidate_stage']=STAGE
"""
new_delivery=old_delivery.replace(anchor,block+anchor);ast.parse(new_delivery,filename=str(delivery));delivery.write_text(new_delivery,'utf8')
record={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root',
    'prepared_before_results':True,'actual_requirements_not_updated_yet':True,
    'scientific_code_config_protocol_and_acceptance_modified':False,'new_scientific_execution':0,
    'reason':'After actual R61 result verification, replace stale R49 requirement readouts while retaining their prior status and evidence; D/E remain untested and the full model goal remains unproven.',
    'files':[{'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in [finalizer,delivery]]}
out=R/'reviews/R61_REQUIREMENT_READOUT_PREPARATION_001.json';assert not out.exists()
out.write_text(json.dumps(record,indent=2)+'\n','utf8')
print(json.dumps(record,indent=2))
