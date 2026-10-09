from pathlib import Path
from datetime import datetime,timezone
import json,ast,hashlib,os
ROOT=Path(__file__).resolve().parent
src=(ROOT/'evaluate_endpoints.py').read_text('utf8')
# The failed entry is immutable. New entry uses only already saved outputs/fits.
old="COUNTER={'HGB_attempts':0,'density_description_slots':0,'density_EM_attempts':0,'main_generation_rows':0,\n         'repeat_noise_rows':0,'source_probe_rows':0,'EVAL_read':0,'CAL_waveform_decode':0,\n         'response_updates':0,'completed_projection_group_blocks':0}"
assert old in src
src=src.replace(old,"COUNTER=json.loads((ROOT/'EVALUATION_PROGRESS.json').read_text('utf8'))['counters'].copy()")
start=src.index('    for name in ARMS:\n        results[name]={};d=domains[name]')
end=src.index('    paired={}\n',start)
src=src[:start]+'''    for name in ARMS:
        results[name]={}
        for view in views:
            slot=name+'__'+view
            row=json.loads((OUT/('C_'+slot+'.json')).read_text('utf8'))
            p=load(OUT/(slot+'_PREDICTIONS_PRIVATE.npz'))
            results[name][view]=row
            private.update({slot+'__'+k:p[k] for k in ['probability','label','cluster']})
''' +src[end:]
start=src.index('    for name in ARMS:\n        if name in alias and alias[name] in tables:')
end=src.index('    errors={}\n',start)
src=src[:start]+'''    for name in ARMS:
        tables[name]=json.loads((OUT/('WIDTH_'+name+'_TABLE.json')).read_text('utf8'))
        assert len(tables[name]['slices'])==8
''' +src[end:]
start=src.index('    generation={};repeats={};probe={};solver={}\n')
end=src.index('    stats,distance=ev.characterize(',start)
src=src[:start]+'''    generation=json.loads((ROOT/'GENERATION_PROGRESS.json').read_text('utf8'))['generation']
    repeats=json.loads((ROOT/'JOINT8_SUMMARY.json').read_text('utf8'))
    probe=json.loads((ROOT/'SOURCE_PROBES_SUMMARY.json').read_text('utf8'))
    solver=json.loads((ROOT/'SOLVER_CHECK.json').read_text('utf8'))
    for name in ARMS:
        saved=load(ROOT/name/'generated.npz')
        assert np.array_equal(saved['source_ids'],ids)
        domains[name]=aa.data(saved['waveforms_V'],ids,np.arange(2000),'G4_SOURCE',ref)
''' +src[end:]
src=src.replace("tdomains={n:domains[n]for n in ['REAL','BASE',*ARMS]}","tdomains={n:domains[n]for n in ['REAL',*CONTROLS,*ARMS]}")
assert "tdomains={n:domains[n]for n in ['REAL',*CONTROLS,*ARMS]}" in src
src=src.replace("'one_new_server_flow_fit_as_registered':True,", "'one_new_server_flow_fit_as_registered':True,'engineering_resume_from_saved_outputs_fits':True,'resume_new_inference_HGB_EM':0,")
src=src.replace("'elapsed_s':time.monotonic()-START,'counters':COUNTER,'solver_check_rows':32", "'elapsed_s':time.monotonic()-START,'counters':COUNTER,'solver_check_rows':32,'resumed_new_main_aux_generation_rows':0,'resumed_new_HGB_EM':0")
ast.parse(src)
(ROOT/'resume_evaluation.py').write_text(src,'utf8')
record={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
    'original_entry':'evaluate_endpoints.py','original_worker':37968,'original_native_exit':1,
    'original_native_tool':'cc645f','original_elapsed_s':31.46035320003284,
    'failure':'temporal.fine_distances expected INDEPENDENT_CFM but summary tdomains had only BASE and new arm',
    'repair':'include all existing control domains in the pure saved-output descriptions',
    'immutable_original_source_sha256':hashlib.sha256((ROOT/'evaluate_endpoints.py').read_bytes()).hexdigest(),
    'resume_entry':'resume_evaluation.py','resume_source_sha256':hashlib.sha256(src.encode()).hexdigest(),
    'already_consumed':json.loads((ROOT/'EVALUATION_PROGRESS.json').read_text('utf8'))['counters'],
    'resume_allowed_blocks':['saved waveform statistics','saved density table aggregation','existing prediction paired bootstrap','projection/mean descriptions','source/boundary summaries'],
    'new_response_training_generation_HGB_EM_solver_checks':0,'original_receipts_overwritten':False}
(ROOT/'ENGINEERING_RESUME_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n','utf8')
print(json.dumps(record),flush=True)
