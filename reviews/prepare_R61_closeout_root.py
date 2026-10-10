"""Prepare artifact validation and report generation without reading new results."""
from pathlib import Path
import ast,json
R=Path(__file__).resolve().parents[1];S=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'

def rep(text,old,new):
    assert text.count(old)==1,(old,text.count(old));return text.replace(old,new)

name='review_R59_frozen_artifacts_root.py'
text=(R/'reviews'/name).read_text('utf8').replace('R59_DIRECT_DISTRIBUTION_RESPONSE_001',S.name)
text=rep(text,'from network import ResponseNetwork','from network import ResponseNetwork\nfrom post_audit import training_audit')
needle="P=read('config_post.json')\nassert len(set(P['endpoint_weight_paths'].values()))==1"
new=needle+"\nH=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'\nlocal_cfg=dict(P);local_cfg['historical_R59_root']=str(H)\nlocal_cfg['input_paths']=dict(P['input_paths'],distribution_statistics=str(H/'TRAIN_DISTRIBUTION_CONSTANTS.npz'))\nlocal_audit=training_audit(S,local_cfg,train)\nassert local_audit['pass'],local_audit['checks']\nassert local_audit['checks']==read('TRAINING_FROZEN_AUDIT.json')['checks']\nassert train['initial_state_SHA']==C['expected_R59_initial_state_SHA']\nassert train['stream_hashes']==C['expected_R59_stream_hashes']"
text=rep(text,needle,new)
text=rep(text,"'local_CPU_waveform_inference_executed':False,'model_goal_complete':False", "'local_CPU_waveform_inference_executed':False,'model_goal_complete':False,\n    'independent_local_saved_TRAIN_record_audit':local_audit,'remote_and_local_training_audit_checks_equal':True")
dest=R/'reviews/review_R61_frozen_artifacts_root.py';assert not dest.exists();ast.parse(text);dest.write_text(text,'utf8')

text=(R/'reviews/finalize_R59_root.py').read_text('utf8')
text=text.replace('R59_DIRECT_DISTRIBUTION_RESPONSE_001',S.name)
text=text.replace('DIRECT_DISTRIBUTION_RESPONSE','PRECISION_RESOLVED_RESPONSE').replace('DIRECT_DISTRIBUTION_REAL_CONTROL','PRECISION_RESOLVED_REAL_CONTROL')
text=text.replace('R59','R61').replace('R54','R59')
text=rep(text,"'# R61：直接无配对全窗分布训练'","'# R61：存储精度排序单因素消融'")
text=rep(text,"'本轮不构造教师波形，使用全部 6593 个源父事件和 21047 个实测父事件。每步分区抽样并显式恢复原区域质量的全局权重，训练一个全 480 点条件 U-Net；只使用预登记的第 50000 次更新。源 h 只用于谷约束，不输入网络，也不为实测添加粒子标签。'",
    "'本轮只取消 float32 PSD 保守区间重叠父对的顺序罚，保留原非零父对分母。全部 6593 源和 21047 REAL 父事件、初始化、六条实际随机流、曝光次数、冻结常数、其他六项损失及唯一 50000 更新终点与 R59 匹配。该区间是存储精度包络，不是粒子标签或实测不确定性。'")
text=rep(text,"'配对区间使用相同留出身份、组和 500 次 bootstrap，条件于这两个已拟合分类器。R61 同时改变训练目标、抽样及初始化等因素，不能把差异归因于单一结构或推广为重复训练、独立采集收益。'",
    "'配对区间使用相同留出身份、组和 500 次 bootstrap，条件于两个冻结分类器。一致性审计支持本固定运行的单因素比较；一个训练种子不能估计重复训练波动，历史单次采集也不能证明独立采集收益。'")
needle="num=local('TRAIN_SOLVER_CHECK.json')['models'][ARM]"
extra="""precision=audit['single_factor_precision_order']
lines+=['| 单因素核验 | 结果 |','|---|---|']
for k,v in precision['checks'].items():lines.append('| '+k+' | '+outcome(v)+' |')
lines+=['','原排序罚 = 保留可分辨罚 + 省略重叠罚，全部50000步分解最大绝对残差 '+value(precision['original_minus_resolved_minus_omitted_maxabs'])+'。下表为逐更新小批量记录均值，不是冻结模型全TRAIN评价。','',
    '| 排序训练记录 | 逐更新均值 |','|---|---:|']
for k,v in precision['trace_summary'].items():lines.append('| '+k+' | '+value(v['mean'])+' |')
lines+=['','R60 的三个预设冲突标记均未触发的结论保持；R61 是另行预登记的存储精度假设，不把本轮结果倒填为 R60 已定位原因。','']
"""
text=rep(text,needle,extra+needle)
text=rep(text,"'fixed_time_boundaries.png','explicit_constraint_projection.png']", "'fixed_time_boundaries.png','explicit_constraint_projection.png','precision_order_decomposition.png']")
text=rep(text,"'UNIT_CHARGE_CONSTANTS.npz','NOISE_CALIBRATION.npz',ARM+'/FINAL_FLOW.pt',ARM+'/TRAINING_RECEIPT.json']", "'UNIT_CHARGE_CONSTANTS.npz','NOISE_CALIBRATION.npz','SOURCE_ORDER_PRECISION_DEFINITION.json','SOURCE_PSD_PRECISION_PRIVATE.npz','precision_audit.py',ARM+'/PRECISION_ORDER_BY_UPDATE_PRIVATE.npz',ARM+'/FINAL_FLOW.pt',ARM+'/TRAINING_RECEIPT.json']")
text=text.replace('R61 直接分布训练科学终点完成','R61 存储精度排序消融科学终点完成').replace('R61 已完成唯一 50000 步直接无配对训练','R61 已完成唯一 50000 步存储精度排序消融训练')
dest=R/'reviews/finalize_R61_root.py';assert not dest.exists();ast.parse(text);dest.write_text(text,'utf8')
print(json.dumps({'prepared_scripts':2,'scientific_results_read':0,'new_scientific_execution':0,'closeout_executed':False}))
