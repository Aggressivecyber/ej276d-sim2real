from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='4'
import numpy as np

R=Path(__file__).resolve().parents[1];S=R/'work/R58_JOINT_DISTRIBUTION_FEASIBILITY_001';now=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
v=read(S/'RESULT.json');a=v['verification']['validation'];ps=v['verification']['PSD'];adc=v['ADC_integral_bounds'];native=read(S/'numeric_worker.py.NATIVE_RECEIPT.json')
assert v['decision']=='joint_statistical_constraints_feasible_in_fixed_family' and a['numeric_feasible'] and a['strict_nonnegative_probability_law']
assert v['solver']['success'] and v['counter']=={'LP_attempts':1,'new_response_updates':0,'teacher_waveforms_constructed':0,'HGB_fits':0,'density_fits':0,'ODE_generation':0}
assert native['actual_worker_native_exit']==native['worker_self_receipt']['native_exit']==0 and native['termination_reason'] is None
assert not read(S/'LIVE_OBSERVATION.json')['actual_process']['exists']
assert v['DEV_read']==v['EVAL_read']==v['CAL_decode']==0 and not v['model_goal_complete']
with np.load(S/'LP_PRIVATE.npz',allow_pickle=False) as z:
    solution=z['solution'];nx=int(z['x_variables']);objective=z['objective']
with np.load(S/'JOINT_LAW_PRIVATE.npz',allow_pickle=False) as z:
    law=z['x_group_atom'];counts=z['group_parent_count'];groups=z['source_group'];ids=z['source_ids'];atoms=z['target_PSD'];weights=z['target_weights']
    assert np.array_equal(law.reshape(-1),solution[:nx]) and len(ids)==6593 and len(z['REAL_PSD'])==21047
    assert np.array_equal(np.bincount(groups,minlength=len(counts)),counts) and np.all(law>=0)
    row_error=float(np.max(np.abs(law.sum(1)-counts)));col_error=float(np.max(np.abs(law.sum(0)-6593*weights)))
    assert max(row_error,col_error)<=1e-7 and abs(float(objective@solution)-v['solver']['numerical_objective'])<1e-9
with np.load(S/'TARGET_ATOMS_PRIVATE.npz',allow_pickle=False) as z:
    assert np.array_equal(atoms,z['values']) and np.array_equal(weights,z['weights'])
    assert np.all(np.bincount(z['REAL_index'],weights=z['integer_mass'],minlength=21047)==512)
    assert np.all(np.bincount(z['quantile_index'],weights=z['integer_mass'],minlength=512)==21047)
assert ps['overall_within_numerical_tolerance'] and ps['conditional_within_numerical_tolerance']
assert all(x['within_numerical_tolerance'] for x in v['verification']['valleys'])
assert adc['rows'][0]['originally_in_range_impossible_global_mass']==0 and adc['rows'][0]['free_PSD_observed_baseline_impossible_parents']==0
names=['population_and_solver','joint_PSD_CDF','source_association_and_valleys','ADC_integral_bounds']
observations=[
    'All ten original Q regions and both original parent populations are visible. The solver success and finite-family limitation are readable.',
    'All ten Q regions plus the overall distribution retain full observed signed-PSD tails. Large exterior-region differences and Q1-Q3 differences remain visible; no failed region is hidden.',
    'The complete 574 by 512 conditional law is shown. Both valley constraints sit at 0.02, and the TRAIN rank 0.924 is clearly distinguished from the DEV context reference.',
    'Original SOURCE overrange-parent fractions and impossible joint-law mass use explicitly different denominators. The latter is confined to the upper exterior region; the necessary-condition limitation is visible.'
]
qa={'UTC':now,'owner':'/root','all_figures_opened':True,'layout_pass':True,
    'method':'Actually opened all four fetched final PNGs with the image viewer. No plot or scientific calculation was rerun.',
    'observations':observations,'figures':[{'path':n+'.png','bytes':(S/(n+'.png')).stat().st_size,'sha256':hashlib.sha256((S/(n+'.png')).read_bytes()).hexdigest(),'opened':True,'layout_pass':True} for n in names]}
write(S/'FIGURE_QA.json',qa)
next_action='封存R58有限统计族结果，以此作为联合PSD与源关联并非已被排除的证据。下一轮预登记直接无配对的全窗神经分布训练：保留全部原TRAIN源父事件等权，联合考虑总体/条件PSD、真实全窗条件分布、源秩/谷概率、精确电荷、ADC范围和同处理REAL控制；不把R58有限原子或分数组合充当逐事件真实目标波形。先仅用合法TRAIN固定网络、损失和一个科学终点，再执行冻结候选的原数值检查、完整A–C及独立重放；通过后继续原D/E。R58的2.92% TRAIN不可实现质量不授权修改原DEV人口、阈值或删除支持外源。'
conclusion='R58一次固定LP及保存解核验通过：原6593源等权边缘与全部21047 REAL贡献保留，总体PSD W1为0.0006015000，共同Q1–7均值为0.0100000000000698，后者处于预登记数值容差内的边界。Q3/Q4 h==1谷增量均约2个百分点，TRAIN加权源PSD秩关联为0.924042；这只支持本有限统计族的可行性，不是原A–F通过。联合律有2.9210%质量在已观测REAL基线范围下不满足ADC积分必要条件，均来自原输入已越界的高端支持外源；自由PSD总电荷硬界未排除任何源事件。0教师波形、0神经训练，完整模型目标未完成。'
d={'UTC':now,'owner':'/root','scientific_decision':v['decision'],'model_goal_complete':False,'application_goal_status':'active',
    'conclusion_zh':conclusion,'source_parents':6593,'REAL_parents':21047,'PSD_summary':ps,
    'valleys':v['verification']['valleys'],'weighted_SOURCE_PSD_rho':v['verification']['weighted_SOURCE_PSD_rho'],
    'independent_solver_verification':a,'saved_law_equals_saved_LP_solution':True,'exact_REAL_integer_quantile_mass_checked':True,
    'root_saved_marginal_check':{'group_row_error_max':row_error,'target_column_error_max':col_error},
    'ADC_summary':adc['rows'][0],'original_A_F_changed':False,'evaluation_source_population_changed':False,'teacher_waveforms_constructed':0,'new_response_training':0,
    'next_action':next_action,'interpretation_limits':[
        'The result is a floating-point feasible law in the single registered finite family, not a rational certificate or an upper bound over all mappings.',
        'The conditional mean 0.0100000000000698 and both valley fractions are at their numerical constraint boundaries; individual Q1-Q3 W1 values exceed 0.01.',
        'The source parent marginal is unchanged, but statistical fractional assignments are not true particle labels or physical event pairs.',
        'The TRAIN rank correlation and valley references are not the original DEV C-gate comparison.',
        'ADC impossibility is conditional on the observed REAL baseline range; no waveform, noise, width, source perturbation, boundary or DEV acceptance is established.',
        'All ten regions and failed exterior behavior remain in the saved law and report; no region is removed.',
        'The same frozen neural method must still pass the complete original A-F and actual delivery.'
    ]}
write(R/'reviews/R58_ROOT_DIRECTION_001.json',d)
print(json.dumps({'layout_pass':True,'saved_law_matches_LP':True,'scientific_decision':d['scientific_decision'],'next_action':next_action,'model_goal_complete':False},ensure_ascii=False,indent=2))
