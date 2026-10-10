"""Adapt the proven serialized controller; all scientific entries are frozen first."""
from pathlib import Path
import ast,json
R=Path(__file__).resolve().parents[1]
S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
oldstage='R59_DIRECT_DISTRIBUTION_RESPONSE_001';stage=S.name
for name in ['launch_R59_entry_root.py','watch_R59_pipeline_root.py','fetch_R59_root.py']:
    p=R/'reviews'/name;dest=R/'reviews'/name.replace('R59','R61');assert not dest.exists(),dest
    text=p.read_text('utf8').replace(oldstage,stage).replace('R59','R61')
    if name.startswith('launch_'):
        old="assert train['teacher_waveforms_constructed']==0 and all(train[k]==0 for k in ['DEV_read','EVAL_read','CAL_decode'])"
        new=old+"\ncfg=json.loads((p/'config.json').read_text())\nassert train['initial_state_SHA']==cfg['expected_R59_initial_state_SHA']\nassert train['stream_hashes']==cfg['expected_R59_stream_hashes']\nassert all(train['single_factor_integrity'][k] is True for k in ['same_R59_initial_state','all_six_consumed_stream_hashes_equal_R59','all_parent_exposures_equal_R59','final_random_states_equal_R59','frozen_R59_distribution_statistics_reused'])"
        assert text.count(old)==1;text=text.replace(old,new)
        old="if entry!='numerical_check.py':assert json.loads((p/'TRAIN_SOLVER_CHECK.json').read_text())['both_pass']"
        new="if entry!='numerical_check.py':\n assert json.loads((p/'TRAIN_SOLVER_CHECK.json').read_text())['both_pass']\n assert json.loads((p/'TRAINING_FROZEN_AUDIT.json').read_text())['pass']"
        assert text.count(old)==1;text=text.replace(old,new)
    if name.startswith('fetch_'):
        # The previous transfer has already demonstrated that aggregate fetches
        # can exceed one minute; allow transport to finish, without rerunning science.
        text=text.replace('capture_output=True,timeout=60','capture_output=True,timeout=180')
    ast.parse(text,filename=str(dest));dest.write_text(text,'utf8')
name='deploy_R59_post_root.py';text=(R/'reviews'/name).read_text('utf8').replace(oldstage,stage)
text=text.replace('render_R59_figures.py','render_R61_figures.py').replace('historical_R54_root','historical_R59_root')
text=text.replace("'HISTORICAL_TRAIN_SOLVER_CHECK.json'","'HISTORICAL_R59_TRAIN_SOLVER_CHECK.json'")
text=text.replace("'UNIT_CHARGE_CONSTANTS.json','post_audit.py']","'UNIT_CHARGE_CONSTANTS.json','post_audit.py','precision_audit.py','POST_AUDIT_ARTIFICIAL_CHECK.json']")
old="assert p.is_dir()\nconflicts="
new="assert p.is_dir()\nactive=json.loads((p/'REMOTE_ACTIVE_JOB.json').read_text())\nassert active['entry']=='train_final.py' and active.get('actual_worker_native_exit') is None,active\nworker=Path('/proc',str(active['actual_worker_PID']))\nassert worker.is_dir() and str((worker/'cwd').resolve())==str(p)\nassert b'train_final.py' in (worker/'cmdline').read_bytes()\noriginal=json.loads((p/'ROOT_IMPLEMENTATION_REVIEW.json').read_text())\nfor row in original['files']:assert hashlib.sha256((p/row['name']).read_bytes()).hexdigest()==row['sha256'],row['name']\nconflicts="
assert text.count(old)==1;text=text.replace(old,new)
dest=R/'reviews/deploy_R61_post_root.py';assert not dest.exists();ast.parse(text,filename=str(dest));dest.write_text(text,'utf8')
print(json.dumps({'controller_scripts_created':4,'science_resubmission_on_observation_timeout':False,'training_fields_modified':False}))
