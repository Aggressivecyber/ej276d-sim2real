from pathlib import Path
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata,wasserstein_distance

def verify(root):
    root=Path(root)
    with np.load(root/'LP_PRIVATE.npz',allow_pickle=False) as z:
        x=z['solution'];nx=int(z['x_variables']);E=csr_matrix((z['E_data'],z['E_indices'],z['E_indptr']),shape=tuple(z['E_shape']));A=csr_matrix((z['A_data'],z['A_indices'],z['A_indptr']),shape=tuple(z['A_shape']));rhsE=z['equality_rhs'];rhsA=z['inequality_rhs'];objective=z['objective']
    with np.load(root/'JOINT_LAW_PRIVATE.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    C=json.loads((root/'config.json').read_text('utf8'));valley=json.loads((root/'FROZEN_VALLEY_DEFINITION.json').read_text('utf8'))
    law=data['x_group_atom'];atom=data['target_PSD'];weights=data['target_weights'];source=data['source_PSD'];bins=data['source_bin0'];h=data['source_h'];group=data['source_group'];keys=data['group_keys'];ng=data['group_parent_count'];N=len(source)
    strict_nonnegative=bool(np.all(law>=0));row=law.sum(1);col=law.sum(0);lifted=row[group]/ng[group]
    eqerr=float(np.max(np.abs(E@x-rhsE)));uberr=float(max(0.,np.max(A@x-rhsA)));negerr=float(max(0.,-x.min()));uppererr=float(max(0.,x[nx:].max()-1))
    validation={'equality_residual_max':eqerr,'inequality_violation_max':uberr,'lower_bound_violation_max':negerr,'CDF_slack_upper_bound_violation_max':uppererr,'group_row_error_max':float(np.max(np.abs(row-ng))),'target_column_error_max':float(np.max(np.abs(col-N*weights))),'per_parent_probability_sum_error_max':float(np.max(np.abs(lifted-1))),'SOURCE_global_weight_error_max':float(np.max(np.abs(lifted/N-1/N))),'strict_nonnegative_probability_law':strict_nonnegative,'objective_from_saved_solution':float(objective@x)}
    validation['numeric_feasible']=bool(strict_nonnegative and all(validation[k]<=C['verification_tolerance'] for k in ['equality_residual_max','inequality_violation_max','lower_bound_violation_max','CDF_slack_upper_bound_violation_max','group_row_error_max','target_column_error_max','per_parent_probability_sum_error_max']))
    if not strict_nonnegative:return {'validation':validation,'PSD':None,'valleys':None,'weighted_SOURCE_PSD_rho':None}
    rows=[];rv=data['REAL_PSD'];rb=data['REAL_bin0']
    for b in [None,-1,*range(8),8]:
        gm=np.ones(len(keys),bool) if b is None else keys[:,0]==b;rm=np.ones(len(rv),bool) if b is None else rb==b;m=law[gm].sum(0);mass=float(m.sum())
        rows.append({'raw_bin0':b,'source_parents':int(ng[gm].sum()),'REAL_parents':int(rm.sum()),'joint_count_mass':mass,'W1':float(wasserstein_distance(atom,rv[rm],u_weights=m)) if mass>0 and rm.any() else None})
    bybin={r['raw_bin0']:r for r in rows};mean=float(np.mean([bybin[b]['W1'] for b in range(7)]));vrows=[]
    for b in [2,3]:
        de=valley['bins'][b];parents=(bins==b)&(h==1);gm=(keys[:,0]==b)&(keys[:,1]==1);am=(atom>=de['low'])&(atom<=de['high']);original=float(np.mean((source[parents]>=de['low'])&(source[parents]<=de['high'])));fraction=float(law[np.ix_(gm,am)].sum()/parents.sum())
        vrows.append({'bin':b+1,'source_parents':int(parents.sum()),'original_fraction':original,'candidate_fraction':fraction,'delta':fraction-original,'allowed_delta':C['source_valley_delta_max'],'within_numerical_tolerance':bool(fraction-original<=C['source_valley_delta_max']+C['verification_tolerance'])})
    # Calculate marginal midranks again from the saved original SOURCE and actual solved target marginal.
    rx=(rankdata(source,method='average')-.5)/N;actualw=col/col.sum();end=np.cumsum(actualw);ry=end-.5*actualw
    means=np.bincount(group,weights=rx,minlength=len(ng))/ng;mx=float(np.sum(rx*lifted)/N);my=float(np.dot(actualw,ry));varx=float(np.sum(lifted*(rx-mx)**2)/N);vary=float(np.dot(actualw,(ry-my)**2))
    covariance=float(np.sum(law*(means-mx)[:,None]*(ry-my)[None,:])/N);rho=covariance/np.sqrt(varx*vary)
    return {'validation':validation,'PSD':{'overall_W1':rows[0]['W1'],'common_Q1_Q7_equal_mean_W1':mean,'by_bin':rows,'overall_within_numerical_tolerance':bool(rows[0]['W1']<=C['PSD_overall_max']+C['verification_tolerance']),'conditional_within_numerical_tolerance':bool(mean<=C['PSD_conditional_mean_max']+C['verification_tolerance'])},'valleys':vrows,'weighted_SOURCE_PSD_rho':float(rho),'weighted_SOURCE_PSD_covariance':covariance,'source_rank_variance':varx,'output_rank_variance':vary,'original_DEV_rank_reference_is_not_TRAIN_acceptance':True}

if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent),indent=2,allow_nan=False))
