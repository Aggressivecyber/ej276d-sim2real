"""One registered full finite-parent D + original C response training run.

No scientific arrays are decoded until the artificial CPU/CUDA suite passes.
Recovery is the same logical run and requires an explicit, evidence-bound request.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
import os
import sys
import time
import traceback

START = time.monotonic()
R = Path(__file__).resolve().parent
C = json.loads((R/'config.json').read_text('utf8'))
ARM = C['arm']
D = R/ARM
CACHE = R/'cache'
CACHE.mkdir(exist_ok=True)
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:C['CPU_threads']])
sys.path.insert(0, str(R/'distribution_adapter'))
sys.path.insert(0, str(R/'objective_adapter'))
sys.path.insert(0, str(R/'network_update_check'))

import numpy as np
import scipy
import torch
from network import ResponseNetwork
from noise_core import q_np, bins_for, continuous_logq, innovation
from source_flow_core import endpoint, q_torch, canonical_adc_torch
from distribution_loss import DistributionLoss, CATEGORIES, NAMES
from distribution_gradient import FrozenDistributionObjective, ParentLayout
from objective_adapter import joint_total_D, original_C_microbatch, D_PIECE_ORDER, C_NAMES
from network_update_check import (parameter_vjp_cpu_f64,
    scalar_parameter_gradient_cpu_f64, install_cpu_f64_clip_step)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write(name, value):
    path = R/name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
        stream.flush()
    os.replace(tmp, path)


def read(name):
    return json.loads((R/name).read_text('utf8'))


def state_hash(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def learning_rate(step):
    if step <= C['warmup_updates']:
        return C['lr_initial']+(C['lr']-C['lr_initial'])*(step-1)/(C['warmup_updates']-1)
    fraction = (step-C['warmup_updates'])/(C['updates']-C['warmup_updates'])
    return C['lr_final']+.5*(C['lr']-C['lr_final'])*(1+math.cos(math.pi*fraction))


def raw_psd(value):
    x = np.asarray(value, dtype='f8')
    return 1-(x[:,100:150].sum(1)-.5*x[:,:100].sum(1))/q_np(x)


def chain_update(old, value):
    return hashlib.sha256(old+np.ascontiguousarray(value).tobytes()).digest()


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key:to_cpu(v) for key,v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return copy.deepcopy(value)


def charge_range(value, target):
    if not torch.isfinite(value).all():
        raise FloatingPointError('nonfinite projected endpoint')
    eq64 = ((q_torch(value)-target).abs()/target).max()
    eq32 = ((q_torch(value.float().double())-target).abs()/target).max()
    if max(float(eq64), float(eq32)) > 1e-5:
        raise FloatingPointError('registered training charge constraint violated')
    if ((value < -.5)|(value > .5)).any():
        raise FloatingPointError('registered training voltage range violated')
    return float(eq64), float(eq32)


PHYSICAL_KEYS = (
    'neural_forward_rows', 'neural_forward_calls', 'full_D_objective_calls',
    'full_D_endpoint_reverse_calls', 'D_parameter_VJP_calls', 'C_objective_calls',
    'C_parameter_reverse_calls', 'parameter_cast_calls', 'gradient_clip_calls',
    'optimizer_step_calls', 'D_source_noise_rows', 'C_SOURCE_noise_rows',
    'C_identity_noise_rows', 'full_window_direction_draw_calls',
    'temporal_direction_draw_calls', 'endpoint_replay_checked_chunks',
    'full_D_SOURCE_law_rows','full_D_REAL_law_rows','C_SOURCE_objective_rows',
    'C_REAL_reference_objective_rows','C_REAL_identity_objective_rows',
)
physical = {key:0 for key in PHYSICAL_KEYS}
counter = {key:0 for key in C['expected_completed_update_counts']}
counter.update(response_fit_attempts=0, teacher_waveforms_constructed=0,
    new_HGB_fits=0, new_density_fits=0, DEV_read=0, EVAL_read=0, CAL_decode=0)
updates = 0
current_update = 0
last_checkpoint_update = 0
phase = 'startup_before_artificial_checks'
inflight = None
attempt_id = os.environ['OWNED_SCIENCE_NONCE']
logical_run_id = C['stage']+'@'+C['remote_root']
resume_history = []
previous_attempt_compute = []
attempt_start_update = 0
full_SOURCE_epsilon_calls = 0
gradient_execution_counts = {key:0 for key in (
    'combined_gradient_cast_events','combined_clip_calls','optimizer_step_calls',
    'separate_D_or_C_clip_calls','parameter_updates_between_D_passes',
    'discarded_micro_D_backward_calls','endpoint_mismatch_chunks',
    'canonical_code_mismatch_chunks','Box_free_mask_mismatch_chunks')}


def physical_schema_counter():
    return {
        'new_response_updates':physical['optimizer_step_calls'],
        'SOURCE_training_rows':physical['C_SOURCE_objective_rows'],
        'REAL_reference_rows':physical['C_REAL_reference_objective_rows'],
        'REAL_identity_rows':physical['C_REAL_identity_objective_rows'],
        'full_D_SOURCE_law_rows':physical['full_D_SOURCE_law_rows'],
        'full_D_REAL_law_rows':physical['full_D_REAL_law_rows'],
        'scientific_neural_forward_rows':physical['neural_forward_rows'],
        'scientific_neural_forward_calls':physical['neural_forward_calls'],
        'D_parameter_VJP_calls':physical['D_parameter_VJP_calls'],
        'C_parameter_reverse_calls':physical['C_parameter_reverse_calls'],
        'full_D_endpoint_reverse_calls':physical['full_D_endpoint_reverse_calls'],
        'full_D_objective_calls':physical['full_D_objective_calls'],
        'replay_chunks_checked':physical['endpoint_replay_checked_chunks'],
    }


def phase_write(next_phase):
    global phase
    phase = next_phase
    write('PHYSICAL_PROGRESS.json', {
        'UTC':utc(), 'PID':os.getpid(), 'logical_run_id':logical_run_id,
        'native_attempt_id':attempt_id, 'phase':phase,
        'current_update':current_update, 'last_completed_update':updates,
        'last_checkpoint_update':last_checkpoint_update,
        'known_completed_physical_operations_this_attempt':dict(physical),
        'physical_schema_counter_lower_bound':physical_schema_counter(),
        'attempt_start_committed_update':attempt_start_update,
        'effective_completed_update_counts':dict(counter),
        'inflight_operation':inflight,
        'uncertainty_if_process_dies': 'Counters are an exact lower bound at this timestamp. Operations after this snapshot until the next snapshot may be unknown; never silently equate them to completed-update counts.',
    })


try:
    assert C['updates'] == C['unique_terminal_updates'] == 50000
    assert (C['SOURCE_N'], C['REAL_N'], C['batch_per_region'], C['identity_batch']) == (6593,21047,32,64)
    assert C['neural_replay_chunk'] == C['W1_feature_chunk'] == 64
    assert C['box_projection_iterations'] == 64 and C['solver_steps'] == 128 and C['solver_check_steps'] == 256
    assert C['region_categories'] == CATEGORIES and C['feature_names'] == NAMES
    assert tuple(C['full_D_scalar_order']) == D_PIECE_ORDER and tuple(C['C_scalar_order']) == C_NAMES
    assert all(value == 1 for value in C['loss_weights'].values())
    assert not C['new_statistics_fit'] and not C['original_D_C_random_coupling_preserved']
    torch.set_num_threads(C['CPU_threads'])
    torch.set_num_interop_threads(C['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.mha.set_fastpath_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError('registered CUDA unavailable')
    dev = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    reviewed = read('ROOT_IMPLEMENTATION_REVIEW.json')
    assert reviewed['ready_for_launch'] and reviewed['original_A_F_changed'] is False
    for record in reviewed['files']:
        assert sha(R/record['name']) == record['sha256'], record['name']
    binding = read('INPUT_BINDING.json')['inputs']
    for key, path in C['input_paths'].items():
        assert sha(path) == binding[key]['sha256'], key
    environment = {
        'python':sys.version, 'numpy':np.__version__, 'torch':torch.__version__,
        'scipy':scipy.__version__, 'device':str(dev), 'GPU':torch.cuda.get_device_name(0),
        'CPU_threads':torch.get_num_threads(), 'CPU_affinity':sorted(os.sched_getaffinity(0)),
        'model_training_mode':True, 'MHA_fastpath':False,
        'TF32':False, 'deterministic_algorithms':True,
    }
    assert all(environment[k] == C['expected_R33_environment'][k]
               for k in ('python','numpy','torch','scipy','device','GPU')), environment
    frozen_binding = {
        'protocol_SHA':sha(R/'PROTOCOL.md'), 'config_SHA':sha(R/'config.json'),
        'input_binding_SHA':sha(R/'INPUT_BINDING.json'),
        'implementation_review_SHA':sha(R/'ROOT_IMPLEMENTATION_REVIEW.json'),
        'environment':environment,
    }
    request_path = R/'RECOVERY_REQUEST.json'
    recovery_requested = request_path.exists()
    recovery = None
    if recovery_requested:
        request = read('RECOVERY_REQUEST.json')
        assert request['logical_run_id'] == logical_run_id
        assert request['prior_actual_PID_gone'] and request['prior_native_attempt_preserved']
        assert request['frozen_binding'] == frozen_binding
        assert sha(CACHE/'RECOVERY_SAME_RUN.pt') == request['checkpoint_SHA256']
        recovery = torch.load(CACHE/'RECOVERY_SAME_RUN.pt', map_location='cpu', weights_only=False)
        assert recovery['checkpoint_transaction_version'] == 1
        assert recovery['logical_run_id'] == logical_run_id and recovery['frozen_binding'] == frozen_binding
        assert recovery['committed_update'] == request['committed_update'] < C['updates']
        assert request['previous_native_attempt_id'] == recovery['native_attempt_id']
        previous_attempt_compute = request['preserved_physical_attempt_compute']
        assert previous_attempt_compute and all(x['attempt_id'] != attempt_id for x in previous_attempt_compute)
        resume_history = request['recovery_history']
        assert resume_history
        suite = read('ARTIFICIAL_SUITE_RECEIPT.json')
        assert suite['pass'] and suite['implementation_review_SHA'] == frozen_binding['implementation_review_SHA']
    else:
        assert not D.exists() and not (CACHE/'RECOVERY_SAME_RUN.pt').exists(), 'Existing run requires explicit same-run recovery'
        from artificial_suite import run_all
        suite = run_all(R, C, frozen_binding['implementation_review_SHA'])
        assert suite['pass']
        write('ARTIFICIAL_SUITE_RECEIPT.json', suite)
    torch.backends.mha.set_fastpath_enabled(False)
    phase_write('artificial_passed_before_TRAIN_decode')
    historical = json.loads(Path(C['input_paths']['R63_training_receipt']).read_text('utf8'))
    assert historical['initial_state_SHA'] == C['expected_R63_initial_state_SHA']
    assert historical['stream_hashes'] == C['expected_R63_stream_hashes']
    if not recovery_requested:
        D.mkdir()
        write(ARM+'/ENVIRONMENT.json', dict(environment, PID=os.getpid(), UTC=utc(), cwd=str(R), arm=ARM))
    with np.load(C['TRAIN_path'], allow_pickle=False) as z:
        source = np.asarray(z['source'], dtype='f4').copy()
        real = np.asarray(z['target'], dtype='f4').copy()
        source_ids, source_h = z['source_ids'].copy(), z['h'].copy()
        saved_source_bins, saved_REAL_bins = z['source_bin'].copy(), z['target_bin'].copy()
    assert source.shape == (6593,480) and real.shape == (21047,480)
    assert len(source_ids) == len(source_h) == len(source)
    with np.load(C['constants_path'], allow_pickle=False) as z:
        mu, scale, edges = z['mu_unit_q'].copy(), z['scale_unit_q'].copy(), z['edges_logQ'].copy()
    with np.load(C['noise_calibration_path'], allow_pickle=False) as z:
        noise = {key:z[key].copy() for key in z.files}
    valley = json.loads(Path(C['valley_definition']).read_text('utf8'))
    assert valley['source'] == 'R3B results/WIDTH_PRETRAIN.json REAL_TRAIN_fit2592' and not valley['new_fit']
    sb, sraw, sq = bins_for(source, edges)
    rb, rraw, rq = bins_for(real, edges)
    assert np.array_equal(sraw,saved_source_bins) and np.array_equal(rraw,saved_REAL_bins)
    assert np.all(scale > 0) and np.max(np.abs(q_np(mu)-1)) < 1e-10
    sh, rh, sp = continuous_logq(sq,sb,edges), continuous_logq(rq,rb,edges), raw_psd(source)
    assert np.isfinite(sp).all() and np.all(np.abs(sq)<=300) and np.all(np.abs(rq)<=300)
    groupsS = [np.flatnonzero(sraw==region) for region in CATEGORIES]
    groupsR = [np.flatnonzero(rraw==region) for region in CATEGORIES]
    assert all(len(group)>0 for group in groupsS+groupsR)
    with np.load(C['input_paths']['distribution_statistics'], allow_pickle=False) as z:
        statistics = {key:z[key].copy() for key in z.files}
    with np.load(C['input_paths']['temporal_statistics'], allow_pickle=False) as z:
        temporal = {key:z[key].copy() for key in z.files}
    assert np.array_equal(statistics['source_counts'],[len(g) for g in groupsS])
    assert np.array_equal(statistics['REAL_counts'],[len(g) for g in groupsR])
    assert np.array_equal(temporal['temporal_REAL_counts'],statistics['REAL_counts'])
    statistics.update(temporal)
    assert sha(R/'TRAIN_DISTRIBUTION_CONSTANTS.npz') == binding['distribution_statistics']['sha256']
    assert sha(R/'TRAIN_TEMPORAL_CONSTANTS.npz') == binding['temporal_statistics']['sha256']
    with np.load(R/'SOURCE_PSD_PRECISION_PRIVATE.npz',allow_pickle=False) as z:
        assert np.array_equal(z['source_ids'],source_ids) and np.array_equal(z['source_original_psd'],sp)
        interval_low,interval_high = z['PSD_interval_low'].copy(),z['PSD_interval_high'].copy()
    assert read('SOURCE_ORDER_PRECISION_DEFINITION.json')['interval_rule'] == C['source_order_rule']
    assert np.all(interval_low<=sp) and np.all(sp<=interval_high)
    if not recovery_requested:
        write('FROZEN_VALLEY_DEFINITION.json',valley)
        write('TRAIN_POPULATION_AND_COORDINATES.json', {
            'UTC':utc(), 'SOURCE_parents':len(source), 'REAL_parents':len(real),
            'SOURCE_ids_SHA256':hashlib.sha256(np.ascontiguousarray(source_ids).tobytes()).hexdigest(),
            'source_counts':statistics['source_counts'].tolist(), 'REAL_counts':statistics['REAL_counts'].tolist(),
            'region_categories':CATEGORIES,
            'SOURCE_global_region_mass':(statistics['source_counts']/len(source)).tolist(),
            'REAL_global_region_mass':(statistics['REAL_counts']/len(real)).tolist(),
            'full_D_parent_sampling':'Every original SOURCE and REAL parent once per update; domains independently uniform.',
            'C_parent_sampling':'Original32 independent uniform-with-replacement per region; original importance weights; identity64 uniform REAL.',
            'source_Q_range':[float(sq.min()),float(sq.max())], 'REAL_Q_range':[float(rq.min()),float(rq.max())],
            'source_original_range_excess_parents':int(np.any(np.abs(source)>.5,axis=1).sum()),
            'source_val_target_val_decoded':False, 'new_statistics_fit':False,
            'frozen_R61_distribution_statistics_reused':True, 'frozen_R63_temporal_statistics_reused':True,
            'all_source_parents_eligible':True, 'h_or_PDG_network_input':False,
        })
    xs = torch.as_tensor(source,dtype=torch.float64,device=dev)
    xr = torch.as_tensor(real,dtype=torch.float64,device=dev)
    mt, st = (torch.as_tensor(value,dtype=torch.float64,device=dev) for value in (mu,scale))
    sbt, sqt, sht = (torch.as_tensor(value,device=dev) for value in (sb,sq,sh))
    torch.manual_seed(C['model_seed'])
    torch.cuda.manual_seed_all(C['model_seed'])
    model = ResponseNetwork().to(dev).train()
    initial_hash = state_hash(model.state_dict())
    assert initial_hash == C['expected_R63_initial_state_SHA'], ('initialization_changed',initial_hash)
    named = tuple(model.named_parameters())
    params = tuple(value for _,value in named)
    parameter_count = sum(p.numel() for p in params)
    named_parameter_order = [name for name,_ in named]
    frozen_buffers = {name:value.detach().clone() for name,value in model.named_buffers()}
    if not recovery_requested:
        write(ARM+'/INITIALIZATION.json', {
            'UTC':utc(),'seed':C['model_seed'],'state_tensor_SHA':initial_hash,
            'parameter_count':parameter_count,'loaded_previous_trained_response':False,
            'zero_output_head':True,'network_teacher':False,'static_conditional_displacement':True,
            'named_parameter_order':named_parameter_order,
        })
    objective = DistributionLoss(statistics,C,dev)
    layout = ParentLayout.full_reference(sraw,rraw)
    optimizer = torch.optim.Adam(params,lr=C['lr_initial'],betas=tuple(C['betas']),eps=1e-8,weight_decay=0)
    streams = {key:np.random.Generator(np.random.PCG64(seed)) for key,seed in C['PCG64_seeds'].items()}
    initial_states = {key:copy.deepcopy(value.bit_generator.state) for key,value in streams.items()}
    hashes = {key:bytes(32) for key in streams}
    countsS, countsR, countsI = np.zeros(6593,dtype='i8'), np.zeros(21047,dtype='i8'), np.zeros(21047,dtype='i8')
    countsDS, countsDR = np.zeros(6593,dtype='i8'), np.zeros(21047,dtype='i8')
    columns, order_columns, replay_columns = C['loss_columns'], C['precision_columns'], C['replay_columns']
    dense, order_dense, replay_dense, discarded_dense, trace = [], [], [], [], []
    discarded_columns = list(D_PIECE_ORDER)+['original_nine_total','discarded_D_total']
    global_max_q64 = global_max_q32 = 0.
    global_max_D_q64 = global_max_D_q32 = global_max_C_q64 = global_max_C_q32 = 0.
    if recovery_requested:
        model.load_state_dict(recovery['model'],strict=True)
        optimizer.load_state_dict(recovery['optimizer'])
        assert all(torch.isfinite(p).all() for p in params)
        assert recovery['initial_state_SHA'] == initial_hash and recovery['named_parameter_order'] == named_parameter_order
        updates = int(recovery['committed_update'])
        assert recovery['step'] == updates and recovery['log_commit_lengths'] == {name:updates for name in ('loss','precision','replay','discarded_D')}
        assert recovery['loss_columns'] == columns and recovery['precision_columns'] == order_columns and recovery['replay_columns'] == replay_columns
        counter = recovery['counter']
        for key in streams:
            streams[key].bit_generator.state = recovery['PCG64_states'][key]
        hashes = recovery['stream_chains']
        countsS, countsR, countsI = (recovery[key] for key in ('SOURCE_exposure_counts','REAL_reference_exposure_counts','REAL_identity_exposure_counts'))
        countsDS, countsDR = recovery['full_D_SOURCE_exposure_counts'], recovery['full_D_REAL_exposure_counts']
        dense, order_dense, replay_dense, discarded_dense = (list(recovery[key]) for key in ('dense_losses','precision_values','replay_values','discarded_D_values'))
        assert all(len(x) == updates for x in (dense,order_dense,replay_dense,discarded_dense))
        trace = recovery['trace']
        objective.direction_calls = recovery['direction_calls']
        objective.temporal_direction_calls = recovery['temporal_direction_calls']
        full_SOURCE_epsilon_calls = recovery['full_SOURCE_epsilon_calls']
        gradient_execution_counts = recovery['gradient_execution_counts']
        assert objective.direction_calls == objective.temporal_direction_calls == updates
        assert full_SOURCE_epsilon_calls == updates
        global_max_q64,global_max_q32,global_max_D_q64,global_max_D_q32,global_max_C_q64,global_max_C_q32 = recovery['max_q_errors']
        torch.set_rng_state(recovery['torch_CPU_RNG'])
        torch.cuda.set_rng_state(recovery['torch_CUDA_RNG'],0)
        assert read(ARM+'/INITIAL_RANDOM_STATES.json') == initial_states
        assert np.all(countsDS == updates) and np.all(countsDR == updates)
        attempt_start_update = last_checkpoint_update = updates
    else:
        write(ARM+'/INITIAL_RANDOM_STATES.json',initial_states)
        counter['response_fit_attempts'] = 1
        write('TRAIN_STARTED.json', {
            'UTC':utc(),'PID':os.getpid(),'counter':counter,'sole_terminal_update':50000,
            'logical_run_id':logical_run_id,'native_attempt_id':attempt_id,
            'teacher_waveforms_constructed':0,'selection':'sole final50000',
            'progress_log_every_updates':C['progress_interval_updates'],
            'recovery_cache_every_updates':C['recovery_interval_updates'],
            'frozen_binding':frozen_binding,
        })
    schedule = [(lo,min(lo+64,len(source))) for lo in range(0,len(source),64)]
    assert len(schedule) == 104 and schedule[-1] == (6592,6593)
    phase_write('scientific_training_ready')

    def checkpoint_save():
        global last_checkpoint_update
        assert updates == current_update and len(dense) == len(order_dense) == len(replay_dense) == len(discarded_dense) == updates
        checkpoint = {
            'checkpoint_transaction_version':1, 'logical_run_id':logical_run_id,
            'native_attempt_id':attempt_id, 'committed_update':updates, 'step':updates,
            'frozen_binding':frozen_binding, 'model':to_cpu(model.state_dict()),
            'optimizer':to_cpu(optimizer.state_dict()), 'counter':dict(counter),
            'PCG64_states':{key:copy.deepcopy(v.bit_generator.state) for key,v in streams.items()},
            'stream_chains':dict(hashes), 'SOURCE_exposure_counts':countsS.copy(),
            'REAL_reference_exposure_counts':countsR.copy(),'REAL_identity_exposure_counts':countsI.copy(),
            'full_D_SOURCE_exposure_counts':countsDS.copy(),'full_D_REAL_exposure_counts':countsDR.copy(),
            'direction_calls':objective.direction_calls,'temporal_direction_calls':objective.temporal_direction_calls,
            'full_SOURCE_epsilon_calls':full_SOURCE_epsilon_calls,
            'gradient_execution_counts':dict(gradient_execution_counts),
            'loss_columns':columns,'dense_losses':np.asarray(dense,dtype='f8'),
            'precision_columns':order_columns,'precision_values':np.asarray(order_dense,dtype='f8'),
            'replay_columns':replay_columns,'replay_values':np.asarray(replay_dense,dtype='i8'),
            'discarded_D_columns':discarded_columns,'discarded_D_values':np.asarray(discarded_dense,dtype='f8'),
            'log_commit_lengths':{name:updates for name in ('loss','precision','replay','discarded_D')},
            'initial_state_SHA':initial_hash,'named_parameter_order':named_parameter_order,
            'torch_CPU_RNG':torch.get_rng_state(),'torch_CUDA_RNG':torch.cuda.get_rng_state(0),
            'max_q_errors':[global_max_q64,global_max_q32,global_max_D_q64,global_max_D_q32,global_max_C_q64,global_max_C_q32],
            'trace':copy.deepcopy(trace),'physical_this_attempt':dict(physical),
            'attempt_start_update':attempt_start_update,'physical_previous_attempts':previous_attempt_compute,
        }
        tmp, target = CACHE/'RECOVERY_SAME_RUN.tmp.pt', CACHE/'RECOVERY_SAME_RUN.pt'
        with tmp.open('wb') as stream:
            torch.save(checkpoint,stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp,target)
        last_checkpoint_update = updates
        write('RECOVERY_CHECKPOINT.json', {
            'UTC':utc(),'committed_update':updates,'native_attempt_id':attempt_id,
            'logical_run_id':logical_run_id,'transaction_version':1,
            'checkpoint_path':'cache/RECOVERY_SAME_RUN.pt','bytes':target.stat().st_size,
            'all_fields_after_completed_optimizer_and_log_commit':True,
        })

    for step in range(updates+1,C['updates']+1):
        current_update = step
        inflight = 'draw_original_C_and_fresh_full_D_noise_and_directions'
        phase_write('draws')
        # Original seven streams retain the exact R63 per-step API and order.
        si = np.concatenate([g[streams['source_rows'].integers(0,len(g),size=C['batch_per_region'])] for g in groupsS]).astype('i8')
        ri = np.concatenate([g[streams['REAL_reference_rows'].integers(0,len(g),size=C['batch_per_region'])] for g in groupsR]).astype('i8')
        ii = streams['REAL_identity_rows'].integers(0,len(real),size=C['identity_batch']).astype('i8')
        assert np.array_equal(sraw[si],np.repeat(CATEGORIES,32)) and np.array_equal(rraw[ri],np.repeat(CATEGORIES,32))
        epsilonS = streams['source_epsilon'].standard_normal((len(si),480)).astype('<f4')
        epsilonI = streams['identity_epsilon'].standard_normal((len(ii),480)).astype('<f4')
        physical['C_SOURCE_noise_rows'] += len(si)
        physical['C_identity_noise_rows'] += len(ii)
        nS,_ = innovation(source[si],epsilonS,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        nI,_ = innovation(real[ii],epsilonI,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        directions = objective.draw_directions(streams['projection_directions'])
        temporal_directions = objective.draw_temporal_directions(streams['baseline_projection_directions'])
        physical['full_window_direction_draw_calls'] += 1
        physical['temporal_direction_draw_calls'] += 1
        epsilonD = streams['full_SOURCE_epsilon'].standard_normal((6593,480)).astype('<f4')
        full_SOURCE_epsilon_calls += 1
        physical['D_source_noise_rows'] += 6593
        nD,_ = innovation(source,epsilonD,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
        nd = torch.as_tensor(nD,device=dev)
        rate = learning_rate(step)
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        versions = tuple(p._version for p in params)
        before_physical = dict(physical)
        projected, raw_collected = [], []
        phase_write('full_D_firstpass')
        with torch.no_grad():
            for lo,hi in schedule:
                inflight = {'firstpass_chunk':[lo,hi]}
                output,raw = endpoint(model,xs[lo:hi],nd[lo:hi],sqt[lo:hi],sbt[lo:hi],sht[lo:hi],mt,st)
                physical['neural_forward_calls'] += 1
                physical['neural_forward_rows'] += hi-lo
                projected.append(output)
                raw_collected.append(raw)
            cache = torch.cat(projected)
            full_raw = torch.cat(raw_collected)
            dq64,dq32 = charge_range(cache,sqt)
            D_projection_fraction = float(((full_raw<-.5)|(full_raw>.5)).any(1).double().mean())
            D_projection_RMS = float((cache-full_raw).square().mean().sqrt())
        del projected,raw_collected,full_raw,output,raw
        inflight = 'one_joint_D_objective_and_one_endpoint_reverse'
        phase_write('full_D_objective_and_GY')
        # The constructor clones directions: build this object with current draws.
        full_objective = FrozenDistributionObjective(statistics,C,directions,temporal_directions,
            sraw,rraw,feature_chunk=64,device=dev,uniform_fastpath=True)
        distribution = joint_total_D(full_objective,cache,xr,layout)
        physical['full_D_objective_calls'] += 1
        physical['full_D_endpoint_reverse_calls'] += 1
        physical['full_D_SOURCE_law_rows'] += 6593
        physical['full_D_REAL_law_rows'] += 21047
        gy = distribution['GY']
        assert gy.dtype == torch.float64 and gy.shape == cache.shape and torch.isfinite(gy).all()
        assert torch.isfinite(distribution['total']) and all(torch.isfinite(v) for v in distribution['pieces'].values())
        D_gradient = torch.zeros(parameter_count,dtype=torch.float64,device='cpu')
        matched_y = matched_codes = matched_free = 0
        phase_write('full_D_parameter_replay')
        for lo,hi in schedule:
            inflight = {'replay_forward_and_VJP_chunk':[lo,hi]}
            output,raw = endpoint(model,xs[lo:hi],nd[lo:hi],sqt[lo:hi],sbt[lo:hi],sht[lo:hi],mt,st)
            physical['neural_forward_calls'] += 1
            physical['neural_forward_rows'] += hi-lo
            expected = cache[lo:hi]
            same_y = torch.equal(output.detach(),expected)
            same_codes = torch.equal(canonical_adc_torch(output.detach(),False),canonical_adc_torch(expected,False))
            same_free = torch.equal((output.detach()>-.5)&(output.detach()<.5),(expected>-.5)&(expected<.5))
            if not (same_y and same_codes and same_free):
                write('REPLAY_MISMATCH.json', {'step':step,'chunk':[lo,hi],
                    'endpoint_bitwise':same_y,'canonical_codes':same_codes,'Box_free_mask':same_free})
                raise FloatingPointError('Original-schedule endpoint replay mismatch')
            matched_y += 1
            matched_codes += 1
            matched_free += 1
            physical['endpoint_replay_checked_chunks'] += 1
            D_gradient += parameter_vjp_cpu_f64(output,params,gy[lo:hi])
            physical['D_parameter_VJP_calls'] += 1
            del output,raw
        assert tuple(p._version for p in params) == versions
        assert all(torch.equal(value,frozen_buffers[name]) for name,value in model.named_buffers())
        assert torch.isfinite(D_gradient).all()
        D_gradient_norm = float(torch.linalg.vector_norm(D_gradient))
        del full_objective,gy,cache,nd,nD
        inflight = 'original384_row_C_forward_original_nine_objective_four_C_reverse'
        phase_write('original_C_and_combined_update')
        bx = torch.as_tensor(np.r_[sb[si],rb[ii]],device=dev,dtype=torch.int64)
        q = torch.as_tensor(np.r_[sq[si],rq[ii]],device=dev)
        hc = torch.as_tensor(np.r_[sh[si],rh[ii]],device=dev)
        x = torch.cat((xs[si],xr[ii]))
        n = torch.as_tensor(np.concatenate((nS,nI)),device=dev)
        output,raw = endpoint(model,x,n,q,bx,hc,mt,st)
        physical['neural_forward_calls'] += 1
        physical['neural_forward_rows'] += len(x)
        constraints = original_C_microbatch(objective,output,raw,xr[ri],xr[ii],
            torch.as_tensor(rraw[ii],device=dev),torch.as_tensor(sp[si],device=dev),
            torch.as_tensor(source_h[si],device=dev),directions,
            torch.as_tensor(interval_low[si],device=dev),torch.as_tensor(interval_high[si],device=dev),temporal_directions)
        physical['C_objective_calls'] += 1
        physical['C_SOURCE_objective_rows'] += len(si)
        physical['C_REAL_reference_objective_rows'] += len(ri)
        physical['C_REAL_identity_objective_rows'] += len(ii)
        assert torch.isfinite(constraints['total'])
        cq64,cq32 = charge_range(output.detach(),q)
        C_gradient = scalar_parameter_gradient_cpu_f64(constraints['total'],params)
        physical['C_parameter_reverse_calls'] += 1
        assert torch.isfinite(C_gradient).all()
        C_gradient_norm = float(torch.linalg.vector_norm(C_gradient))
        assert tuple(p._version for p in params) == versions
        assert all(torch.equal(value,frozen_buffers[name]) for name,value in model.named_buffers())
        combined = D_gradient+C_gradient
        assert torch.isfinite(combined).all()
        inflight = 'single_final_cast_clip_Adam'
        update_meta = install_cpu_f64_clip_step(params,combined,optimizer,clip=C['gradient_clip'],capture=False)
        physical['parameter_cast_calls'] += 1
        physical['gradient_clip_calls'] += 1
        physical['optimizer_step_calls'] += 1
        assert update_meta['cast_calls'] == update_meta['clip_calls'] == update_meta['optimizer_step_calls'] == 1
        assert all(torch.isfinite(p).all() for p in params)
        norm = float(update_meta['pre_clip_norm'])
        pieces = dict(distribution['pieces'],**{key:value.detach() for key,value in constraints['pieces'].items()})
        total = float(distribution['total'])+float(constraints['total'].detach())
        diagnostic = constraints['diagnostics']
        with torch.no_grad():
            active = ((raw<-.5)|(raw>.5)).any(1)
            correction = (output-raw).square().mean(1)
            C_S_projection_fraction = float((active[:320].double()*objective.sw).sum())
            C_R_projection_fraction = float(active[320:].double().mean())
            C_S_projection_RMS = float((correction[:320]*objective.sw).sum().sqrt())
            C_R_projection_RMS = float(correction[320:].mean().sqrt())
        row_values = dict(pieces)
        row_values.update(total_loss=total,learning_rate=rate,gradient_norm=norm,
            max_relative_q_f64=max(dq64,cq64),max_relative_q_f32=max(dq32,cq32),
            SOURCE_global_range_projection_fraction=C_S_projection_fraction,
            REAL_range_projection_fraction=C_R_projection_fraction,
            SOURCE_global_projection_RMS_V=C_S_projection_RMS,REAL_projection_RMS_V=C_R_projection_RMS,
            overall_PSD_W1=float(distribution['diagnostics']['overall_PSD_W1']),
            Q1_Q7_mean_PSD_W1=float(distribution['diagnostics']['Q1_Q7_mean_PSD_W1']),
            Q3_h_one_soft_valley=float(diagnostic['Q3_h_one_soft_valley']),
            Q4_h_one_soft_valley=float(diagnostic['Q4_h_one_soft_valley']),
            D_total=float(distribution['total']),C_total=float(constraints['total'].detach()),
            full_D_SOURCE_range_projection_fraction=D_projection_fraction,
            full_D_SOURCE_projection_RMS_V=D_projection_RMS,
            full_D_max_relative_q_f64=dq64,full_D_max_relative_q_f32=dq32,
            C_max_relative_q_f64=cq64,C_max_relative_q_f32=cq32,
            D_parameter_gradient_L2=D_gradient_norm,C_parameter_gradient_L2=C_gradient_norm)
        row = np.array([float(row_values[key]) for key in columns],dtype='f8')
        order_row = np.array([float(diagnostic[key]) for key in order_columns],dtype='f8')
        discarded_values = {key:float(v) for key,v in constraints['discarded_D_values'].items()}
        discarded_row = np.array([discarded_values[key] for key in D_PIECE_ORDER]+
            [float(constraints['original_nine_total_value']),sum(discarded_values.values())],dtype='f8')
        assert np.isfinite(row).all() and np.isfinite(order_row).all() and np.isfinite(discarded_row).all()
        assert np.all(order_row>=0)
        assert abs(order_row[0]-order_row[1]-float(pieces['source_order'])) <= 1e-10+1e-10*abs(order_row[0])
        assert abs(sum(float(pieces[key]) for key in C['full_D_scalar_order'])-float(distribution['total'])) <= 1e-10+1e-10*abs(total)
        assert abs(sum(float(pieces[key]) for key in C['C_scalar_order'])-float(constraints['total'].detach())) <= 1e-10+1e-10*abs(total)
        replay_row = np.array([6593,6593,len(schedule),matched_y,matched_codes,matched_free,1,1,len(schedule),1,1,1],dtype='i8')
        assert np.array_equal(replay_row,C['expected_replay_row'])
        assert physical['neural_forward_rows']-before_physical['neural_forward_rows'] == 13570
        assert physical['neural_forward_calls']-before_physical['neural_forward_calls'] == 209
        assert physical['D_parameter_VJP_calls']-before_physical['D_parameter_VJP_calls'] == 104
        # A completed logical update commits counters, streams and all logs together.
        np.add.at(countsS,si,1)
        np.add.at(countsR,ri,1)
        np.add.at(countsI,ii,1)
        countsDS += 1
        countsDR += 1
        for key,values in (('source_rows',si.astype('<u2')),('REAL_reference_rows',ri.astype('<u2')),
            ('REAL_identity_rows',ii.astype('<u2')),('source_epsilon',epsilonS),('identity_epsilon',epsilonI),
            ('full_SOURCE_epsilon',epsilonD)):
            hashes[key] = chain_update(hashes[key],values)
        for direction in directions:
            hashes['projection_directions'] = chain_update(hashes['projection_directions'],direction)
        hashes['baseline_projection_directions'] = chain_update(hashes['baseline_projection_directions'],temporal_directions)
        for key,expected in C['expected_completed_update_counts'].items():
            assert expected % C['updates'] == 0
            counter[key] += expected//C['updates']
        gradient_execution_counts['combined_gradient_cast_events'] += update_meta['cast_calls']
        gradient_execution_counts['combined_clip_calls'] += update_meta['clip_calls']
        gradient_execution_counts['optimizer_step_calls'] += update_meta['optimizer_step_calls']
        dense.append(row)
        order_dense.append(order_row)
        replay_dense.append(replay_row)
        discarded_dense.append(discarded_row)
        updates = step
        global_max_D_q64,global_max_D_q32 = max(global_max_D_q64,dq64),max(global_max_D_q32,dq32)
        global_max_C_q64,global_max_C_q32 = max(global_max_C_q64,cq64),max(global_max_C_q32,cq32)
        global_max_q64,global_max_q32 = max(global_max_q64,dq64,cq64),max(global_max_q32,dq32,cq32)
        inflight = None
        if torch.cuda.max_memory_allocated() > C['GPU_limit_bytes'] or torch.cuda.max_memory_reserved() > C['GPU_limit_bytes']:
            raise MemoryError('Registered GPU allocated/reserved safety limit')
        if step%C['progress_interval_updates'] == 0 or step in (1,C['updates']):
            recent = min(len(dense),C['progress_interval_updates'])
            record = {
                'UTC':utc(),'PID':os.getpid(),'arm':ARM,'global_step':step,
                'native_attempt_id':attempt_id,'elapsed_s_this_attempt':time.monotonic()-START,
                'last_update':dict(zip(columns,row.tolist())),
                'last_up_to100_mean':dict(zip(columns,np.asarray(dense[-recent:]).mean(0).tolist())),
                'precision_order_last_up_to100_mean':dict(zip(order_columns,np.asarray(order_dense[-recent:]).mean(0).tolist())),
                'max_output_relative_q_error_f64':global_max_q64,'max_output_relative_q_error_f32':global_max_q32,
                'unique_SOURCE_REAL_reference_REAL_identity':[int(np.count_nonzero(a)) for a in (countsS,countsR,countsI)],
                'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(),
                'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),
                'effective_counter':dict(counter),'physical_this_attempt':dict(physical),
            }
            trace.append(record)
            write(ARM+'/TRAINING_TRACE.json',trace)
            write('TRAIN_PROGRESS.json',record)
            print(json.dumps(record),flush=True)
        if step%C['recovery_interval_updates'] == 0:
            checkpoint_save()
        phase_write('completed_update_committed')
        del distribution,constraints,output,raw,combined,D_gradient,C_gradient,pieces,row_values,x,n

    assert updates == 50000 and objective.direction_calls == objective.temporal_direction_calls == 50000
    assert full_SOURCE_epsilon_calls == 50000
    assert all(torch.isfinite(p).all() for p in params)
    for key,value in C['expected_completed_update_counts'].items():
        assert counter[key] == value,key
    assert countsS.sum() == counter['SOURCE_training_rows'] and countsR.sum() == counter['REAL_reference_rows']
    assert countsI.sum() == counter['REAL_identity_rows']
    assert np.all(countsDS == updates) and np.all(countsDR == updates)
    assert np.all(countsS>0) and np.all(countsR>0) and np.all(countsI>0)
    assert {key:hashes[key].hex() for key in C['original_streams']} == C['expected_R63_stream_hashes']
    final_states = {key:copy.deepcopy(value.bit_generator.state) for key,value in streams.items()}
    assert {key:final_states[key] for key in C['original_streams']} == json.loads(Path(C['input_paths']['R63_final_random_states']).read_text('utf8'))
    with np.load(C['input_paths']['R63_parent_exposures'],allow_pickle=False) as z:
        for key,value in (('source_ids',source_ids),('source_counts',countsS),('REAL_reference_counts',countsR),
                ('REAL_identity_counts',countsI),('source_bin0',sraw),('REAL_bin0',rraw)):
            assert np.array_equal(z[key],value),key
    for key,path in C['input_paths'].items():
        assert sha(path) == binding[key]['sha256'],key
    for record in reviewed['files']:
        assert sha(R/record['name']) == record['sha256'],record['name']
    np.savez_compressed(D/'LOSS_BY_UPDATE_PRIVATE.npz',columns=np.asarray(columns),values=np.asarray(dense,dtype='f8'))
    np.savez_compressed(D/'PRECISION_ORDER_BY_UPDATE_PRIVATE.npz',columns=np.asarray(order_columns),values=np.asarray(order_dense,dtype='f8'))
    np.savez_compressed(D/'REPLAY_BY_UPDATE_PRIVATE.npz',columns=np.asarray(replay_columns),values=np.asarray(replay_dense,dtype='i8'))
    np.savez_compressed(D/'DISCARDED_MICRO_D_BY_UPDATE_PRIVATE.npz',columns=np.asarray(discarded_columns),values=np.asarray(discarded_dense,dtype='f8'))
    np.savez_compressed(D/'PARENT_EXPOSURES_PRIVATE.npz', source_ids=source_ids,source_counts=countsS,
        REAL_reference_counts=countsR,REAL_identity_counts=countsI,source_bin0=sraw,REAL_bin0=rraw,
        source_h=source_h,source_original_psd=sp,source_original_q=sq,
        full_D_source_counts=countsDS,full_D_REAL_counts=countsDR)
    assert not (D/'FINAL_FLOW.pt').exists()
    torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()},D/'FINAL_FLOW.pt')
    torch.save({'CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state(0)},D/'FINAL_TORCH_RNG.pt')
    write(ARM+'/FINAL_RANDOM_STATES.json',final_states)
    physical_record = {
        'attempt_id':attempt_id,'PID':os.getpid(),'resumed_from_update':attempt_start_update,
        'last_completed_update':updates,'completed_operations':dict(physical),
        'counter':physical_schema_counter(),'counters_complete':True,'unknown_intervals':[],
        'exact_completed_operation_counts':True,'unknown_after_crash_fields':[],
        'elapsed_s':time.monotonic()-START,'prior_native_attempts_separately_preserved':True,
    }
    receipt = {
        'PID':os.getpid(),'UTC':utc(),'arm':ARM,'updates':updates,'elapsed_s':time.monotonic()-START,
        'counter':counter,'selection':'sole final50000','logical_run_id':logical_run_id,
        'native_attempt_id':attempt_id,'frozen_binding':frozen_binding,
        'source_parents':6593,'REAL_parents':21047,
        'actual_unique_SOURCE_REAL_reference_REAL_identity':[int(np.count_nonzero(a)) for a in (countsS,countsR,countsI)],
        'SOURCE_parent_exposure_min_max':[int(countsS.min()),int(countsS.max())],
        'REAL_reference_exposure_min_max':[int(countsR.min()),int(countsR.max())],
        'REAL_identity_exposure_min_max':[int(countsI.min()),int(countsI.max())],
        'full_D_SOURCE_parent_exposure_min_max':[int(countsDS.min()),int(countsDS.max())],
        'full_D_REAL_parent_exposure_min_max':[int(countsDR.min()),int(countsDR.max())],
        'source_global_measure':'original equal parent law in full D; original regional importance weights in C',
        'realized_C_exposure_counts_are_not_uniform_claim':True,
        'model_seed':C['model_seed'],'initial_state_SHA':initial_hash,'final_state_SHA':state_hash(model.state_dict()),
        'stream_hashes':{key:value.hex() for key,value in hashes.items()},'PCG64_seeds':C['PCG64_seeds'],
        'stream_hash_algorithm':'SHA256(previous32byte_digest + actual consumed little-endian array bytes); initial32 zero bytes; four full-window directions sequentially in original order',
        'stream_reconstruction':dict(historical['stream_reconstruction'],full_SOURCE_white='standard_normal((6593,480)) defaultfloat64 then astype(<f4); original SOURCE parent order once per update'),
        'direction_calls':objective.direction_calls,'projection_direction_calls':objective.direction_calls,
        'temporal_direction_calls':objective.temporal_direction_calls,'full_SOURCE_epsilon_calls':full_SOURCE_epsilon_calls,
        'gradient_execution_counts':gradient_execution_counts,
        'new_response_training_runs':1,'teacher_waveforms_constructed':0,'R58_law_or_atoms_training_target':False,
        'h_PDG_network_inputs':False,'network_parameter_count':parameter_count,'named_parameter_order':named_parameter_order,
        'max_output_relative_q_error_f64':global_max_q64,'max_output_relative_q_error_f32':global_max_q32,
        'full_D_max_output_relative_q_error_f64':global_max_D_q64,'full_D_max_output_relative_q_error_f32':global_max_D_q32,
        'C_max_output_relative_q_error_f64':global_max_C_q64,'C_max_output_relative_q_error_f32':global_max_C_q32,
        'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),
        'CPU_threads':C['CPU_threads'],'source_val_target_val_decoded':False,'DEV_read':0,'EVAL_read':0,'CAL_decode':0,
        'compute_budget_enforced':False,'model_goal_complete':False,
        'source_order_rule':C['source_order_rule'],'precision_order_columns':order_columns,'precision_order_trace_rows':len(order_dense),
        'loss_columns':columns,'loss_trace_rows':len(dense),'replay_columns':replay_columns,'replay_trace_rows':len(replay_dense),
        'discarded_D_columns':discarded_columns,'discarded_D_trace_rows':len(discarded_dense),
        'full_empirical_D_integrity': {
            'same_R63_initial_state':True,'seven_original_stream_hashes_equal_R63':True,
            'seven_original_final_states_equal_R63':True,'C_parent_exposures_equal_R63':True,
            'frozen_distribution_and_temporal_statistics_reused':True,'new_statistics_fit':False,
            'new_normalization_fit':False,'new_noise_calibration_fit':False,'new_valley_fit':False,
            'joint_D_C_noise_coupling_preserved':False,'discarded_micro_D_not_in_backward':True,
            'cast_clip_Adam_once_per_update':True,'full_D_fresh_direction_binding_every_update':True,
            'D_endpoint_GY_once_per_update':True,'all_original64_chunks_replayed':True,
            'all_endpoint_codes_free_masks_exact':True,'D_then_C_CPU_float64_parameter_accumulation':True,
            'source_and_REAL_masses_independently_preserved':True,'no_REAL_feature_cache':True,
            'C_original_nine_call_four_piece_selection':True,'sole_change':C['sole_scientific_change'],
        },
        'parameter_gradient_rule':C['parameter_gradient_accumulation'],
        'full_D_noise_rule':C['new_stream'],'all_eight_initial_states_file':'INITIAL_RANDOM_STATES.json',
        'all_eight_final_states_file':'FINAL_RANDOM_STATES.json','eighth_stream_independently_regenerated_for_audit':False,
        'physical_attempt_compute':previous_attempt_compute+[physical_record],
        'recovery_history':resume_history,'replayed_compute':'Recorded per physical attempt; completed logical updates remain exactly50000. Unknown crash intervals remain unknown.',
        'unique_final_weight_step':50000,'unique_final_weight_SHA256':sha(D/'FINAL_FLOW.pt'),
        'artificial_suite_SHA256':sha(R/'ARTIFICIAL_SUITE_RECEIPT.json'),
        'artifact_sha256':{name:sha(D/name) for name in ('FINAL_FLOW.pt','INITIALIZATION.json',
            'INITIAL_RANDOM_STATES.json','FINAL_RANDOM_STATES.json','LOSS_BY_UPDATE_PRIVATE.npz',
            'PRECISION_ORDER_BY_UPDATE_PRIVATE.npz','REPLAY_BY_UPDATE_PRIVATE.npz',
            'DISCARDED_MICRO_D_BY_UPDATE_PRIVATE.npz','PARENT_EXPOSURES_PRIVATE.npz')},
    }
    write(ARM+'/TRAINING_RECEIPT.json',receipt)
    phase_write('scientific_training_endpoint_complete')
    print(json.dumps({'training_terminal':{'updates':updates,'elapsed_s':receipt['elapsed_s'],'model_goal_complete':False}}),flush=True)
except BaseException as exc:
    write('TRAINING_FAILURE.json', {
        'UTC':utc(),'PID':os.getpid(),'logical_run_id':logical_run_id,'native_attempt_id':attempt_id,
        'updates':updates,'current_update':current_update,'last_checkpoint_update':last_checkpoint_update,
        'phase':phase,'inflight_operation':inflight,'counter':counter,
        'known_completed_physical_operations_this_attempt':physical,
        'physical_schema_counter_lower_bound':physical_schema_counter(),
        'inflight_operation_completion_not_claimed':True,
        'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START,
    })
    raise
