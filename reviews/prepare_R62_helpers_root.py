from pathlib import Path
import ast,json
R=Path(__file__).resolve().parents[1];S=R/'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001'
for name in ['deploy','poll','fetch']:
    src=(R/'reviews'/(name+'_R60_root.py')).read_text('utf8')
    src=src.replace('R60_FROZEN_TRAIN_GRADIENT_DIAGNOSTIC_001','R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001').replace('R60','R62').replace('frozen_R59_helpers','frozen_R61_helpers')
    if name=='deploy':
        old="'diagnose_train.py','diagnostic_math.py','full_population.py','render_diagnostic.py','artificial_check.py',"
        new="'diagnose_train.py','residual_math.py','traced_response.py','residual_analysis.py','render_diagnostic.py','artificial_check.py','precision_order.py',"
        assert old in src;src=src.replace(old,new).replace('frozen_R59_helpers_identical','frozen_R61_helpers_identical')
        start=src.index(" 'reviewed_definitions':[");end=src.index(" 'scientific_execution_started':False",start)
        src=src[:start]+" 'reviewed_definitions':['all6593/21047 original TRAIN parents and fixed single noise','bitwise traced original Heun128 output on artificial CPU and CUDA arrays','all480 waveform positions and nine fixed physical boundaries','five unchanged path representations and source-weighted full-window objective','exact unequal empirical W1 and fixed64 directions','fixed32 saved-endpoint batches without new neural calls','100x100 baseline covariance, ACF1-20 and correct Nyquist spectral weights','32 empirical REAL resamples matched to original SOURCE N in11 scopes','no optimizer, gradient, teacher, HGB, density fit or validation decode','source IDs, input/weight hashes and three R60 stream hashes preserved','original A-F unchanged and actual native process/resource ownership'],\n"+src[end:]
        anchor="assert C['response_training_runs']=="
        i=src.index(anchor)
        src=src[:i]+"assert (S/'PROTOCOL.md').read_bytes()==(R/'tasks/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001.md').read_bytes()\n"+src[i:]
        anchor="for name,v in binding['frozen_R61_helpers'].items():"
        i=src.index(anchor)
        src=src[:i]+"for record in binding['original_acceptance']:assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256']\n"+src[i:]
    if name=='poll':
        src=src.replace("'FULL_TRAIN_SUMMARY.json','GRADIENT_SUMMARY.json',","'PATH_SUMMARY.json','BASELINE_JOINT_SUMMARY.json','LOSS_COVERAGE_SUMMARY.json','PSD_SUMMARY.json','PROJECTION_SUMMARY.json',")
    ast.parse(src);target=R/'reviews'/(name+'_R62_root.py');assert not target.exists();target.write_bytes(src.encode('utf8'))
print(json.dumps({'generated':['deploy_R62_root.py','poll_R62_root.py','fetch_R62_root.py'],'launch_executed':False}))
