"""Server pretraining proofs, all synthetic; caller supplies the owned guard."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import time


def run_all(root, cfg, implementation_review_SHA):
    import torch
    from portable_fixture import run_check as legacy_objective
    from network_replay_artificial import check as legacy_network
    from portable_objective_checks import run_check as joint_objective
    from network_update_check import run_check as joint_network

    root = Path(root)
    target = root/'artificial_results'
    target.mkdir(exist_ok=False)
    started = time.monotonic()
    receipts = []

    def write(path, value):
        path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())

    def run(name, fn, passed):
        progress = {'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),
            'current_check':name,'completed_checks':[x['name'] for x in receipts],
            'scientific_arrays_decoded':0,'scientific_training_updates':0}
        write(root/'ARTIFICIAL_SUITE_PROGRESS.json',progress)
        print(json.dumps({'artificial_check_started':name}),flush=True)
        try:
            value = fn()
            assert passed(value), (name,'failure in artificial receipt')
            assert torch.cuda.max_memory_allocated()<=cfg['GPU_limit_bytes']
            assert torch.cuda.max_memory_reserved()<=cfg['GPU_limit_bytes']
        except BaseException as exc:
            failure = getattr(exc,'receipt',{'error':repr(exc),'passed':False})
            write(target/(name+'.FAILED.json'),failure)
            raise
        path = target/(name+'.json')
        write(path,value)
        receipts.append({'name':name,'path':str(path.relative_to(root)).replace('\\','/'),
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps({'artificial_check_passed':name,'elapsed_suite_s':time.monotonic()-started}),flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    for device,label in (('cpu','CPU'),('cuda:0','CUDA')):
        run('R64_OBJECTIVE_'+label,
            lambda device=device:legacy_objective(device,adapter_root=root/'distribution_adapter'),
            lambda value:value['compact']['passed'] and value['compact']['checks']==69)
        run('R65_JOINT_OBJECTIVE_'+label,
            lambda device=device:joint_objective(device,objective_adapter_root=root/'objective_adapter'),
            lambda value,label=label:value['status']=='ARTIFICIAL_OBJECTIVE_ADAPTER_PASSED_NOT_SCIENTIFIC_REGISTRATION'
                and value['check_count']==(86 if label=='CPU' else 85) and all(x['passed'] for x in value['tests'].values()))
        run('R64_NETWORK_REPLAY_'+label,
            lambda device=device,label=label:legacy_network(device,4 if label=='CPU' else 64),
            lambda value:value['pass'] and value['state_unchanged'] and value['scientific_updates']==0)
        run('R65_JOINT_NETWORK_ADAM_'+label,
            lambda device=device:joint_network(device,objective_adapter_root=root/'objective_adapter'),
            lambda value:value['passed'] and value['caller_state_preserved']
                and value['compact']['synthetic_optimizer_steps']==2
                and value['scientific_training_updates']==0 and all(x['passed'] for x in value['checks']))
    result = {'pass':True,'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
        'implementation_review_SHA':implementation_review_SHA,'results':receipts,
        'elapsed_s':time.monotonic()-started,'scientific_arrays_decoded':0,
        'scientific_weights_loaded':0,'scientific_training_updates':0,
        'synthetic_optimizer_updates_separate_from_training':4,
        'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(),
        'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),
        'all_8_cpu_cuda_artificial_receipts_passed':True}
    write(root/'ARTIFICIAL_SUITE_PROGRESS.json',dict(result,status='complete_before_TRAIN_decode'))
    return result
