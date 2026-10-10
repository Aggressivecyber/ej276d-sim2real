from pathlib import Path
from datetime import datetime,timezone
import os,json,time,hashlib,traceback,warnings
R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text('utf8'));START=time.monotonic()
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:C['CPU_threads']])
import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
from noise_core import q_np
from joint_core import quantile_atoms,source_groups,build,adc_interval_bad,artificial_check,CATEGORIES

COUNTER={'LP_attempts':0,'new_response_updates':0,'teacher_waveforms_constructed':0,'HGB_fits':0,'density_fits':0,'ODE_generation':0}
def utc():return datetime.now(timezone.utc).isoformat()
def write(name,value):(R/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def progress(phase):
    value={'UTC':utc(),'PID':os.getpid(),'phase':phase,'elapsed_s':time.monotonic()-START,'counter':dict(COUNTER)};write('PROGRESS.json',value);print(json.dumps(value),flush=True)

def input_adc_bounds(q,bins,overrange,baseline):
    bmin=float(baseline.min());bmax=float(baseline.max());limit=150-300*bmin;rows=[]
    for b in [None,*CATEGORIES]:
        ix=np.ones(len(q),bool) if b is None else bins==b
        rows.append({'raw_bin0':b,'source_parents':int(ix.sum()),'original_input_overrange_parents':int(np.sum(ix&overrange)),'free_PSD_observed_baseline_impossible_parents':int(np.sum(ix&(q>limit))),'absolute_ADC_charge_impossible_parents':int(np.sum(ix&(q>300)))})
    return {'observed_REAL_baseline_mean_min_V':bmin,'observed_REAL_baseline_mean_max_V':bmax,'observed_baseline_free_PSD_max_Q':limit,'absolute_ADC_max_Q':300.,'source_Q_min':float(q.min()),'source_Q_max':float(q.max()),'rows':rows,'joint_probability_law_required':False,'not_original_DEV_acceptance':True}

def adc_summary(source_Q,source_bins,source_input_overrange,law,group,counts,atoms,baseline):
    bmin=float(baseline.min());bmax=float(baseline.max());badprob=np.zeros(len(source_Q));minimum_margin=np.zeros(len(source_Q))
    for g in range(len(counts)):
        parents=np.flatnonzero(group==g);bad,gap=adc_interval_bad(source_Q[parents,None],atoms[None,:],bmin,bmax)
        probabilities=law[g]/counts[g];badprob[parents]=bad@probabilities;positive=probabilities>0
        minimum_margin[parents]=gap[:,positive].min(1) if positive.any() else np.nan
    free_observed=source_Q>150-300*bmin;free_absolute=source_Q>300
    rows=[]
    for b in [None,*CATEGORIES]:
        parents=np.ones(len(source_Q),bool) if b is None else source_bins==b;n=int(parents.sum());ix=parents&source_input_overrange;normal=parents&~source_input_overrange
        rows.append({'raw_bin0':b,'source_parents':n,'original_input_overrange_parents':int(ix.sum()),'parents_with_positive_impossible_probability':int(np.sum(parents&(badprob>0))),
            'impossible_global_mass':float(badprob[parents].sum()/len(source_Q)),'impossible_conditional_probability':float(badprob[parents].mean()) if n else None,
            'originally_in_range_impossible_global_mass':float(badprob[normal].sum()/len(source_Q)),'originally_overrange_impossible_global_mass':float(badprob[ix].sum()/len(source_Q)),
            'free_PSD_observed_baseline_impossible_parents':int(np.sum(parents&free_observed)),'absolute_ADC_charge_impossible_parents':int(np.sum(parents&free_absolute))})
    return {'observed_REAL_baseline_mean_min_V':bmin,'observed_REAL_baseline_mean_max_V':bmax,'observed_baseline_free_PSD_max_Q':150-300*bmin,'absolute_ADC_max_Q':300.,'source_Q_min':float(source_Q.min()),'source_Q_max':float(source_Q.max()),'rows':rows,
        'is_full_waveform_feasibility_proof':False,'is_DEV_failure_estimate':False,'inequality_definition':'short=q*(1-p)+50*b in [-25,25]; long=q*p+250*b in [-125,125]; b inside observed REAL TRAIN baseline-mean range'},badprob,minimum_margin

def main():
    write('ARTIFICIAL_CHECK.json',artificial_check())
    binding=json.loads((R/'INPUT_BINDING.json').read_text('utf8'))['inputs']
    for key,path in C['input_paths'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==binding[key]['sha256'],key
    needed=['source_ids','source_h']+['SOURCE__'+k for k in ['PSD','Q_Vns','rawbin','valid','range_exceeded']]+['REAL__'+k for k in ['PSD','Q_Vns','rawbin','valid']]
    with np.load(C['input_paths']['R57_features'],allow_pickle=False) as z:data={k:z[k].copy() for k in needed}
    with np.load(C['input_paths']['TRAIN_path'],allow_pickle=False) as z:real=z['target'].copy()
    with np.load(C['input_paths']['constants_path'],allow_pickle=False) as z:edges=z['edges_logQ'].copy()
    valley=json.loads(Path(C['input_paths']['valley_definition']).read_text('utf8'));write('FROZEN_VALLEY_DEFINITION.json',valley)
    assert valley['source']=='R3B results/WIDTH_PRETRAIN.json REAL_TRAIN_fit2592' and not valley['new_fit']
    source_psd=data['SOURCE__PSD'];source_Q=data['SOURCE__Q_Vns'];source_bins=data['SOURCE__rawbin'];h=data['source_h'];real_psd=data['REAL__PSD'];real_bins=data['REAL__rawbin'];N=len(source_psd)
    assert N==C['SOURCE_N']==6593 and real.shape==(C['REAL_N'],480) and len(real_psd)==21047
    assert data['SOURCE__valid'].all() and data['REAL__valid'].all() and np.isfinite(source_psd).all() and np.isfinite(real_psd).all()
    assert np.array_equal(q_np(real),data['REAL__Q_Vns'])
    baseline=real[:,:100].astype('f8').mean(1);del real
    input_adc=input_adc_bounds(source_Q,source_bins,data['SOURCE__range_exceeded'],baseline);write('ADC_INPUT_BOUNDS.json',input_adc)
    atoms=quantile_atoms(real_psd,C['target_quantile_atoms']);groups=source_groups(source_psd,source_bins,h,C['source_rank_groups'])
    np.savez_compressed(R/'TARGET_ATOMS_PRIVATE.npz',**atoms)
    compressed_W1=float(wasserstein_distance(atoms['values'],real_psd,u_weights=atoms['weights']))
    description={'UTC':utc(),'SOURCE_parents':N,'REAL_parents':len(real_psd),'SOURCE_groups':len(groups['counts']),'target_atoms_requested':C['target_quantile_atoms'],'target_atoms_after_equal_value_merge':len(atoms['values']),
        'target_atoms_vs_full_REAL_W1':compressed_W1,'all_REAL_parent_integer_contributions_preserved':True,'SOURCE_original_weights_unchanged':True,
        'source_groups':[{'group':g,'raw_bin0':int(key[0]),'h_one':bool(key[1]),'SOURCE_rank_segment':int(key[2]),'parents':int(groups['counts'][g]),'mean_SOURCE_PSD_midrank':float(groups['rank_means'][g])} for g,key in enumerate(groups['keys'])]}
    write('POPULATION.json',description)
    model=build(groups,atoms['values'],atoms['weights'],source_psd,source_bins,h,real_psd,real_bins,valley,C);write('LP_CONSTRUCTION.json',{'UTC':utc(),**model['description']});progress('LP_constructed')
    COUNTER['LP_attempts']+=1;assert COUNTER['LP_attempts']==C['LP_attempts']==1;progress('LP_solving');started=time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');solution=linprog(model['objective'],A_ub=model['A'],b_ub=model['b'],A_eq=model['E'],b_eq=model['d'],bounds=model['bounds'],method=C['LP_method'],options=C['LP_options'])
    solver={'UTC':utc(),'success':bool(solution.success),'status':int(solution.status),'message':str(solution.message),'nit':int(solution.nit),'elapsed_s':time.monotonic()-started,'method':C['LP_method'],'options':C['LP_options'],'warnings':[str(w.message) for w in caught],
        'numerical_objective':float(solution.fun) if solution.fun is not None else None,'not_a_certified_global_feasibility_bound':True}
    write('LP_SOLVER_RECEIPT.json',solver);progress('LP_solved')
    payload={'objective':model['objective'],'x_variables':model['description']['x_variables'],'equality_rhs':model['d'],'inequality_rhs':model['b'],'solution':solution.x if solution.x is not None else np.asarray([],dtype='f8'),'interval_rows':model['interval_rows']}
    for name,matrix in [('E',model['E']),('A',model['A'])]:payload.update({name+'_data':matrix.data,name+'_indices':matrix.indices,name+'_indptr':matrix.indptr,name+'_shape':np.asarray(matrix.shape)})
    if solution.success:
        payload['equality_dual']=solution.eqlin.marginals;payload['inequality_dual']=solution.ineqlin.marginals;payload['lower_dual']=solution.lower.marginals;payload['upper_dual']=solution.upper.marginals
    np.savez_compressed(R/'LP_PRIVATE.npz',**payload)
    lawdata={'source_ids':data['source_ids'],'source_PSD':source_psd,'source_Q':source_Q,'source_bin0':source_bins,'source_h':h,'source_input_overrange':data['SOURCE__range_exceeded'],'source_group':groups['index'],'group_keys':groups['keys'],'group_parent_count':groups['counts'],
        'source_midrank':groups['parent_ranks'],'group_mean_midrank':groups['rank_means'],'target_PSD':atoms['values'],'target_weights':atoms['weights'],'REAL_PSD':real_psd,'REAL_bin0':real_bins,'REAL_baseline_mean':baseline,'edges_logQ':edges}
    verification=None;adc=None
    if solution.success:
        lawdata['x_group_atom']=solution.x[:model['description']['x_variables']].reshape(len(groups['counts']),len(atoms['values']))
        np.savez_compressed(R/'JOINT_LAW_PRIVATE.npz',**lawdata)
        from verify_joint import verify
        verification=verify(R);write('INDEPENDENT_VERIFICATION.json',verification)
        if verification['validation']['strict_nonnegative_probability_law']:
            adc,bad,margin=adc_summary(source_Q,source_bins,data['SOURCE__range_exceeded'],lawdata['x_group_atom'],groups['index'],groups['counts'],atoms['values'],baseline)
            np.savez_compressed(R/'ADC_BOUND_PRIVATE.npz',source_ids=data['source_ids'],source_bin0=source_bins,source_Q=source_Q,source_input_overrange=data['SOURCE__range_exceeded'],impossible_probability=bad,best_atom_baseline_interval_gap=margin)
            write('ADC_BOUND.json',adc)
    else:
        np.savez_compressed(R/'POPULATION_PRIVATE.npz',**lawdata)
    statistically_feasible=bool(verification is not None and verification['validation']['numeric_feasible'] and verification['PSD'] is not None and verification['PSD']['overall_within_numerical_tolerance'] and verification['PSD']['conditional_within_numerical_tolerance'] and all(row['within_numerical_tolerance'] for row in verification['valleys']))
    decision='joint_statistical_constraints_feasible_in_fixed_family' if statistically_feasible else 'fixed_discretization_infeasible_no_general_impossibility_claim' if solution.status==2 else 'solver_or_verification_numerically_inconclusive'
    result={'UTC':utc(),'status':'complete','decision':decision,'statistically_feasible_within_registered_tolerance':statistically_feasible,'counter':dict(COUNTER),'solver':solver,'construction':model['description'],'target_quantization_W1':compressed_W1,'verification':verification,'ADC_integral_bounds':adc,'input_ADC_bounds':input_adc,
        'original_DEV_BASE_PSD_rho_context':C['original_DEV_BASE_PSD_rho_for_context'],'original_DEV_rho_floor_context':C['original_DEV_BASE_PSD_rho_for_context']-C['original_DEV_rho_drop_limit'],'TRAIN_rho_is_not_DEV_C_acceptance':True,
        'all_source_parents':N,'all_REAL_parents':len(real_psd),'teacher_waveforms_constructed':0,'new_response_training_started':False,'original_A_F_unchanged':True,'evaluation_source_population_changed':False,'model_goal_complete':False,'DEV_read':0,'EVAL_read':0,'CAL_decode':0}
    write('RESULT.json',result)
    from render_joint import render
    render(R,result,description,lawdata)
    write('NUMERIC_RECEIPT.json',{'UTC':utc(),'status':'complete','PID':os.getpid(),'elapsed_s':time.monotonic()-START,'counter':dict(COUNTER),'decision':decision,'model_goal_complete':False,'environment':{'python':os.sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'CPU_affinity':sorted(os.sched_getaffinity(0))}});progress('complete')

try:main()
except BaseException as exc:
    write('FAILURE.json',{'UTC':utc(),'PID':os.getpid(),'counter':dict(COUNTER),'elapsed_s':time.monotonic()-START,'error':repr(exc),'traceback':traceback.format_exc()});raise
