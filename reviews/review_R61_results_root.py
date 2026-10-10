"""Record root review of the actual frozen results and all nine opened figures."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,struct
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
read=lambda p:json.loads(p.read_text('utf8'))
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
M=read(S/'METRICS_COMPACT.json');G=read(S/'GATE_METRICS.json');arm='PRECISION_RESOLVED_RESPONSE';ctrl='PRECISION_RESOLVED_REAL_CONTROL'
A=M['acceptance_A_C'][arm];g=G['models'][arm]
assert G['decision']=='close_fixed_R61_precision_order_ablation' and not G['primary_A_C_pass']
assert all(not A[k]['pass'] for k in ['A','B','C'])
assert [g[k]['pass']for k in g if k.startswith('gate')]==[False,False,True,False,True,True]
assert read(S/'FROZEN_ARTIFACT_REVIEW.json')['pass'] and read(S/'REPLAY_RECEIPT.json')['pass']
figures=['training_and_checks.png','canonical_five_view_response.png','PSD_all_eight_bins.png','PSD_centers_widths_and_W1.png','noise_and_source_association.png','complete_waveform_population.png','fixed_time_boundaries.png','explicit_constraint_projection.png','precision_order_decomposition.png']
observations=[
 'Seven training surrogate traces and actual FAIL/FAIL/FAIL labels are visible; TRAIN coverage and sole endpoint are explicit.',
 'All five SOURCE and same-treated REAL confidence intervals are visible alongside historical comparisons and the unchanged acceptance band.',
 'All eight fixed Q bins remain visible; Q8 SOURCE density failure is labelled; Q3/Q4 PSD concentrations remain visible.',
 'Q1-7 mean requirement is labelled; Q8 W1 and failed density gaps are retained; both component widths are visible.',
 'Baseline RMS and ACF1 plus complete original source charge and PSD associations are visible with particle-truth caveat.',
 'Complete 480-sample equal-parent mean and SD are displayed; the visible global amplitude difference is not treated as a causal explanation.',
 'All nine boundary differences and fixed limits are visible; three failures around index150 are retained.',
 'Zero correction and zero rail samples in both 2000-row main outputs are shown as counts; nonzero TRAIN trace scope remains explicit.',
 'Original, retained and omitted order losses are shown with unchanged denominator and per-update TRAIN scope.'
]
items=[]
for n,note in zip(figures,observations):
    p=S/'figures'/n;b=p.read_bytes();assert b[:8]==b'\x89PNG\r\n\x1a\n';width,height=struct.unpack('>II',b[16:24])
    items.append({'file':'figures/'+n,'sha256':hashlib.sha256(b).hexdigest(),'pixels':[width,height],'opened_with_view_image':True,'layout_pass':True,'observation':note})
write(S/'FIGURE_QA.json',{'UTC':datetime.now(timezone.utc).isoformat(),'reviewer':'/root','all_figures_opened':True,'layout_pass':True,'figure_count':9,'figures':items,'rendered_images_repaired_after_execution':False,'scientific_acceptance_claim':False})
next_action='封存R61唯一末点，保留单因素消融带来部分PSD和电压指标改善、但原A–C仍失败的结论。下一轮先预登记仅用合法TRAIN的冻结R61完整波形残差诊断：一次覆盖全部原父事件的前向，分开记录输入、神经位移、约束投影及规范ADC，在固定电荷区间核对全窗电压/归一化形状的一阶二阶差分、基线联合噪声与原PSD谷区，检查现有分布损失的覆盖及随机小批量误差；同处理REAL与全部支持外源同时保留。依据TRAIN证据选择下一项全响应机制；不续训R61、不选择中间检查点或改变原A–F及最终SOURCE人口，也不将R60未触发的标记改判。'
direction={'UTC':datetime.now(timezone.utc).isoformat(),'stage':'R61_PRECISION_RESOLVED_SOURCE_ORDER_001','scientific_decision':G['decision'],'model_goal_complete':False,'application_goal_status':'active','root_only':True,'A_C':{k:A[k]['pass']for k in ['A','B','C']},'fixed_stage_gates':{k:v['pass']for k,v in g.items()if k.startswith('gate')},'PSD':A['B'],'SOURCE_C2ST':A['A']['views'][arm],'same_treated_REAL_C2ST':A['A']['views'][ctrl],'paired_R61_minus_R59':M['paired_R61_minus_frozen_R59'][arm],'failed_boundaries':[x for x in g['gate4_boundary_source']['boundaries']if not x['pass']],'single_factor_audit_pass':read(S/'TRAINING_FROZEN_AUDIT.json')['single_factor_precision_order']['pass'],'serialization_only_repair':'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001/repairs/AUDIT_JSON_001/REPAIR_REVIEW.json','new_TRAIN_diagnostic_registered':False,'D_E_started':False,'next_action':next_action,'limits':['Single training seed; paired bootstrap intervals condition on the two fixed fitted classifiers.','Lower C2ST and PSD errors in some views do not establish joint success or a unique cause.','R60 three preregistered conflict flags remain untriggered; no retrospective reinterpretation.','Existing single-acquisition development evidence does not establish independent-acquisition or physical event-pair validity.'],'evidence':['work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001/ACCEPTANCE_A_C.json','work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001/GATE_METRICS.json','work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001/FROZEN_ARTIFACT_REVIEW.json','work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001/FIGURE_QA.json']}
write(R/'reviews/R61_ROOT_DIRECTION_001.json',direction)
print(json.dumps({'nine_figures_reviewed':True,'decision':G['decision'],'model_goal_complete':False,'next_action':next_action},ensure_ascii=False,indent=2))
