"""Independent root recomputation of saved R71 endpoint evidence; no inference."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'root_review_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
native=read(S/'train_full.py.NATIVE_RECEIPT.json');assert native['actual_worker_native_exit']==0 and native['termination_reason']is None
assert read(R/'jobs.json')['active_scientific_jobs']==0
manifest=read(S/'OUTPUT_MANIFEST.json')
for row in manifest:assert sha(S/row['name'])==row['sha256'],row['name']
for row in read(P/'REGISTRATION.json')['local_files']:assert sha(R/row['path'])==row['sha256'],row['path']
write(P/'ROOT_REVIEW_REGISTRATION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),before_saved_arrays_read=True,
    scope='Recompute saved registered TRAIN endpoint metrics/identities/charge/ADC/migration; no model fitting or inference',
    source_code_SHA256=sha(Path(__file__)),native_worker_complete=True,local_CPU=1,local_RSS_bytes=1073741824))
import numpy as np
from scipy.stats import wasserstein_distance,ks_2samp
cfg=read(S/'config.json');receipt=read(S/'TRAINING_RECEIPT.json');cost=read(S/'COST_ADMISSION.json');saved=read(S/'TRAIN_METRICS.json')['metrics']
assert receipt['completed_updates']==50000 and receipt['engineering_updates_discarded']==50
assert receipt['SOURCE_main_exposures']==6593*50000 and receipt['REAL_main_exposures']==21047*50000
assert cost['pass_gate']and cost['optimizer_empty']and cost['RNG_restored']and cost['engineering_parameters_discarded']
assert cost['restored_state_SHA256']==cost['initial_state_SHA256']==receipt['initial_state_SHA256']
assert sha(S/'FINAL_HEAD.pt')==receipt['final_weight_file_SHA256']
assert native['job_tree_RSS_peak_bytes']<=cfg['RSS_limit_bytes']and native['GPU_process_memory_peak_bytes']<=cfg['GPU_limit_bytes']
assert native['handshake_before_imports']and native['actual_worker_PID']==native['worker_self_receipt']['PID']
assert read(S/'ARTIFICIAL_CHECKS.json')['pass_all']
def q(v):return v[:,100:400].sum(1)-3*v[:,:100].sum(1)
def psd(v):return 1-(v[:,100:150].sum(1)-.5*v[:,:100].sum(1))/q(v)
def canonical_input(v):return -.5+np.clip(np.rint((v.astype('f8')+.5)*16383),0,16383)/16383
def canonical_output(v):
    v=v.astype('f4');legacy=(-np.float32(.5)+np.rint((np.clip(v,-.5,.5)+np.float32(.5))*np.float32(16383))/np.float32(16383)).astype('f4')
    return canonical_input(legacy)
train=R/'work/R23_ONLINE_STANDARDIZED_CRITIC_001/TRAIN_PRIVATE.npz'
assert sha(train)==read(S/'INPUT_BINDING.json')['inputs']['TRAIN_path']['sha256']
with np.load(train,allow_pickle=False)as z:
    target=canonical_input(z['target']);source_ids=z['source_ids'].copy();source_q0=q(z['source'].astype('f8'));real_q0=q(z['target'].astype('f8'))
with np.load(S/'PARENT_MANIFEST_PRIVATE.npz',allow_pickle=False)as z:
    assert np.array_equal(source_ids,z['source_ids'])and np.array_equal(z['REAL_original_TRAIN_row'],np.arange(21047))
    assert np.allclose(z['source_Q0'],source_q0,rtol=0,atol=1e-12)and np.allclose(z['REAL_Q0'],real_q0,rtol=0,atol=1e-12)
    assert np.array_equal(z['SOURCE_original_weight'],np.full(6593,1/6593))and np.array_equal(z['REAL_original_weight'],np.full(21047,1/21047))
with np.load(S/'HEAD_STATISTICS.npz',allow_pickle=False)as z:edges=z['edges'].copy()
with np.load(S/'LOSS_BY_UPDATE_PRIVATE.npz',allow_pickle=False)as z:
    loss=z['values'];assert loss.shape==(50000,6)and np.isfinite(loss).all()
    assert np.max(np.abs(loss[:,0]-loss[:,1:4].sum(1)))<1e-12
    first_loss=loss[0].tolist();last_loss=loss[-1].tolist()
def bin_of(charge):
    logq=np.log(1000*charge);b=np.searchsorted(edges,logq,side='right')-1;b[logq==edges[-1]]=7;return b
checks={}
targetq=q(target)
target_codes=np.rint((target+.5)*16383).astype('i8')
target_integer_charge=q(target_codes)
target_max_point_mass=float(np.unique(target_integer_charge,return_counts=True)[1].max()/len(target_integer_charge))
roundoff_audit={}
for label in ['IDENTITY_CONTROL_SOURCE','IDENTITY_CONTROL_REAL','LEARNED_SOURCE','LEARNED_REAL']:
    N=6593 if label.endswith('SOURCE')else 21047
    with np.load(S/(label+'_PRIVATE.npz'),allow_pickle=False)as z:
        pre=z['pre_ADC_V'].copy();qt=z['Qstar'].copy();qi=z['Q0'].copy()
        assert pre.shape==(N,480)and np.isfinite(pre).all()and np.max(np.abs(pre))<=.5
        assert np.isfinite(qt).all()and ((qt>0)&(qt<300)).all()
        assert np.allclose(qi,source_q0 if label.endswith('SOURCE')else real_q0,rtol=0,atol=1e-12)
        if label.endswith('SOURCE'):assert np.array_equal(z['source_parent_ID'],source_ids)
        else:assert np.array_equal(z['REAL_original_TRAIN_row'],np.arange(N))
        pq=q(pre.astype('f8'));adc=canonical_output(pre);aq=q(adc)
        q_metadata_maxabs=max(float(np.max(np.abs(aq-z['QADC']))),float(np.max(np.abs(pq-z['QpreADC']))))
        assert q_metadata_maxabs<=1e-10
        relative=np.max(np.abs(pq-qt)/qt);assert relative<=1e-5
        for key,value in [('original_bin_raw',qi),('target_bin_raw',qt),('preADC_bin_raw',pq),('ADC_bin_raw',aq)]:assert np.array_equal(z[key],bin_of(value)),(label,key)
        migration=read(S/(label+'_MIGRATION.json'));assert sum(row['count']for row in migration['rows'])==N
        assert abs(sum(row['weight']for row in migration['rows'])-1)<1e-12
        metrics=dict(Q_W1_mVns=float(wasserstein_distance(aq*1000,targetq*1000)),Q_KS=float(ks_2samp(aq,targetq).statistic),
            overall_PSD_W1=float(wasserstein_distance(psd(adc),psd(target))))
        integer_charge=q(np.rint((adc+.5)*16383).astype('i8'))
        lattice_residual=float(np.max(np.abs(aq*16383-integer_charge)))
        assert lattice_residual<.001
        lattice_KS=float(ks_2samp(integer_charge,target_integer_charge).statistic)
        max_point_mass=float(np.unique(integer_charge,return_counts=True)[1].max()/N)
        # Canonical charges occupy one integer lattice. Float64 summation may
        # only reorder events WITHIN an exact-charge atom. The combined largest
        # atom masses bound the KS change from arbitrary such tie splitting.
        tie_splitting_bound=max_point_mass+target_max_point_mass
        for key,value in metrics.items():
            if key=='Q_KS':
                assert abs(value-saved[label][key])<=tie_splitting_bound+1e-12
                assert abs(lattice_KS-saved[label][key])<=tie_splitting_bound+1e-12
            else:assert abs(value-saved[label][key])<1e-9,(label,key,value,saved[label][key])
        roundoff_audit[label]=dict(primary_saved_KS=saved[label]['Q_KS'],CPU_recomputed_float_KS=metrics['Q_KS'],
            exact_ADC_integer_charge_KS=lattice_KS,KS_difference=metrics['Q_KS']-saved[label]['Q_KS'],
            largest_SOURCE_or_REAL_candidate_charge_atom_mass=max_point_mass,largest_REAL_reference_charge_atom_mass=target_max_point_mass,
            certified_tie_splitting_KS_bound=tie_splitting_bound,max_integer_lattice_residual=lattice_residual)
        checks[label]=dict(N=N,parents_and_original_Q_unchanged=True,canonical_ADC_from_preADC_recomputed=True,Q_metadata_CPU_GPU_maxabs=q_metadata_maxabs,migration_population_preserved=True,
            relative_Qstar_error_max=float(relative),metrics_recomputed=metrics)
numerical=read(S/'NUMERICAL_REFERENCE.json');replay=read(S/'TRAIN_SERIALIZATION_REPLAY.json')
assert all(v['pass_gate']for v in numerical['results'].values())
assert all(v['preADC_bitwise']and v['canonical_ADC_bitwise']and v['Qstar_bitwise']for v in replay['results'].values())
answer=dict(UTC=datetime.now(timezone.utc).isoformat(),root_saved_evidence_recomputation_pass=True,complete_main_updates=50000,
    native_exit=0,training_elapsed_seconds=receipt['training_elapsed_s'],process_elapsed_seconds=native['elapsed_s'],
    native_peak_RSS_bytes=native['job_tree_RSS_peak_bytes'],native_GPU_process_peak_bytes=native['GPU_process_memory_peak_bytes'],
    first_loss=first_loss,last_loss=last_loss,results=checks,numerical_reference=numerical['results'],TRAIN_component_replay=replay['results'],
    input_bindings_and_initial_restore_verified=True,canonical_Q_KS_float_tie_audit=roundoff_audit,
    primary_saved_metrics_unchanged=True,whole_model_A_F_complete=False,final_heldout_replay_complete=False,
    historical_acceptance_v4_unchanged=True,current_acceptance_version=5,local_guard={**guard,**memory()})
write(P/'ROOT_RESULT_REVIEW.json',answer)
write(P/'CANONICAL_Q_KS_ROUNDOFF_AUDIT.json',dict(initial_exact_scalar_assertion_failed=True,
    first_failure={'domain':'IDENTITY_CONTROL_REAL','recomputed_KS':0.00014253812894950146,'saved_KS':0.00019005083859929828},
    cause='Float64 canonical voltage sums split exact ADC-charge ties; stable integer-code charge audit added, primary results unchanged',
    original_script='root_review_initial.py',registered_scientific_code_unchanged=True,metrics=roundoff_audit))
print(json.dumps(answer),flush=True)
