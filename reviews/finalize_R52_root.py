from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path(__file__).resolve().parents[1];STAGE='R52_JOINT_CHARGE_PSD_COUPLING_001';S=R/'work'/STAGE;NOW=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads((R/p).read_text('utf8'))
def write(p,x):(R/p).write_bytes((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def bind(p):
 p=R/p;return {'path':str(p.relative_to(R)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
n=read('work/'+STAGE+'/numeric_worker.py.NATIVE_RECEIPT.json');d=read('work/'+STAGE+'/JOINT_DECISION.json')
assert n['actual_worker_native_exit']==0 and n['worker_self_receipt']['native_exit']==0 and n['termination_reason'] is None and n['memory_probe_failures']==0
assert not d['all_blocks_pass'] and all(b['noise_failed_bins']==[1,8] and b['PSD_direction_hits_bins2_7']==6 for b in d['blocks'].values())
write('work/'+STAGE+'/FIGURE_QA.json',{'UTC':NOW,'reviewer':'/root','actual_image_opened':True,'figure':bind('work/'+STAGE+'/figures/joint_target_checks.png'),'observation':'Four panels opened; both full and partial blocks, all eight bins, noise thresholds, legends and SD ratios visible without clipping. Q1 and Q8 failure remains visible.','acceptance_from_layout':False})
action='Close fixed 16-quantile coupling as a sufficient target repair. Next register a new endpoint construction that retains the measured baseline and adds a fixed smooth pulse-only correction preserving source charge and target PSD, with analytical TRAIN checks before one newly registered response training. No quantile-count scan.'
decision={**d,'scientific_decision':'fixed_joint_coupling_noise_failed_Q1_Q8_closed','old_recipe_reopened_as_success':False,'next_action':action,'application_goal_status':'active'}
write('reviews/R52_ROOT_DIRECTION_001.json',decision)
lines=['# R52：联合配对改善成立，但端点噪声要求仍失败','',
 '唯一的16电荷分位内PSD排序配方已完成。完整94400行与末89600行两块都保持SOURCE顺序及逐区REAL出现次数；Q2–Q7父事件目标PSD标准差全部满足至少降低50%的要求。但两块的Q1、Q8仍超过预登记目标噪声容差，本配方因此关闭，新增响应训练0。没有修改组数、删去失败区或降低容差。','',
 '| 块 | Q区 | 原缩放目标RMS W1 / mV | 联合配对后 / mV | 固定上限 / mV | 父事件SD比值 | 噪声通过 |','|---|---:|---:|---:|---:|---:|---|']
for block in ['full_cycle','final_partial_cycle']:
 m=read('work/'+STAGE+'/'+block+'_METRICS.json')
 for a in m['by_bin']:lines.append(f"| {block} | {a['bin']} | {a['original']['scaled_teacher_RMS_W1_V']*1000:.6f} | {a['joint']['scaled_teacher_RMS_W1_V']*1000:.6f} | {a['noise_W1_limit_V']*1000:.6f} | {a['median_SD_ratio_joint_over_original']:.4f} | {'是' if a['noise_pass'] else '否'} |")
lines+=['',
 'Q1、Q8的完整周期中位电荷比例分别约0.9879与0.9826，局部电荷配对仍不能使整条波形的缩放完全消失。这个结果只关闭本次固定构造，不能证明所有统计耦合都无法满足噪声要求。旧R51全局PSD排序与本轮联合配对均作为负结果保留，不靠扫描分位组数继续。','',
 '下一步另行定义保持实测基线的加性脉冲修正：以固定平滑基函数承载所需电荷差，并保持目标signed PSD。这会改变训练端点，须在TRAIN中核对电荷/PSD恒等式、基线/尾段保留和波形扰动，再决定新响应训练。原A–F、原源码输出、完整源人口和Q边界保持。','',
 f"服务器原生退出0，worker {n['actual_worker_PID']} / supervisor {n['supervisor_PID']}，科学{n['elapsed_s']:.6f}秒。worker峰值{n['actual_worker_RSS_peak_bytes']} B，轮询作业树峰值{n['job_tree_RSS_peak_bytes']} B；观测失败0。0响应训练、HGB、密度拟合、DEV/EVAL读取及CAL解码。长期模型目标active且未完成。",'',
 '[协议](tasks/'+STAGE+'.md) · [固定判定](work/'+STAGE+'/JOINT_DECISION.json) · [完整周期指标](work/'+STAGE+'/full_cycle_METRICS.json) · [末周期指标](work/'+STAGE+'/final_partial_cycle_METRICS.json) · [原生回执](work/'+STAGE+'/numeric_worker.py.NATIVE_RECEIPT.json) · [下一步](reviews/R52_ROOT_DIRECTION_001.json)','',
 '![两个暴露块的固定检查](work/'+STAGE+'/figures/joint_target_checks.png)','']
(R/'R52_REPORT.md').write_bytes(('\n'.join(lines)).encode())
(S/'REPRODUCTION.md').write_text('# R52复现\n\n在独立目录放置本轮脚本、config、Q_DEFINITION与PROTOCOL。恢复config所指的原TRAIN和R43实际关联依赖，更新路径后运行 `python -B linux_supervise_post.py numeric_worker.py`。已运行目录不可清除.used标记重跑。固定16组，两块各独立构造；输出两套私有关联/父事件描述、固定判定、图及原生回执。无新响应训练或DEV读取。复现须独立记账，不能覆盖本轮失败证据。\n','utf8')
resource={'UTC':NOW,'stage':STAGE,'science_s':n['elapsed_s'],'actual_native_exit':0,'actual_worker_PID':n['actual_worker_PID'],'supervisor_PID':n['supervisor_PID'],'worker_peak_RSS_bytes':n['actual_worker_RSS_peak_bytes'],'sampled_tree_peak_RSS_bytes':n['job_tree_RSS_peak_bytes'],'memory_probe_failures':0,'response_training_runs':0,'HGB_fits':0,'density_fits':0,'DEV_read':0,'CAL_decode':0,'EVAL_read':0,'compute_budget_enforced':False}
write('work/'+STAGE+'/RESOURCE_RECEIPT.json',resource)
write('work/'+STAGE+'/FINAL_RECEIPT.json',{'UTC':NOW,'stage':STAGE,'status':'closed_negative_science_complete','science':resource,'model_goal_complete':False,'scientific_decision':decision['scientific_decision'],'required_files':[bind('work/'+STAGE+'/'+p) for p in ['JOINT_DECISION.json','full_cycle_METRICS.json','final_partial_cycle_METRICS.json','NUMERIC_RECEIPT.json','FIGURE_QA.json','RESOURCE_RECEIPT.json','REPRODUCTION.md']],'publication_backup':'pending separate receipts'})
b=read('RESOURCE_BUDGET.json');bl=b['improvement_block'];assert not any(v['stage']==STAGE for v in bl['actual_stage_receipts'])
bl['actual_stage_receipts'].append({'stage':STAGE,'receipt':'work/'+STAGE+'/RESOURCE_RECEIPT.json','science_s':n['elapsed_s'],'response_training_runs_started':0,'new_response_updates':0,'status':'closed_negative_diagnostic','stage_science_complete':True});bl['actual_science_seconds_recorded']+=n['elapsed_s'];next(v for v in bl['registered_stages'] if v['stage']==STAGE)['status']='closed_negative_diagnostic';write('RESOURCE_BUDGET.json',b)
job={'stage':STAGE,'status':'closed_negative_delivery_pending','active_scientific_jobs':0,'actual_worker_PID':n['actual_worker_PID'],'supervisor_PID':n['supervisor_PID'],'actual_native_exit':0}
for fn in ['PROGRAM.json','PROJECT_PROGRESS.json']:
 x=read(fn);x.update(current_job=job,current_stage_progress='closed_negative_delivery_pending',application_goal_status='active',active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0,next_action=action,latest_completed_stage=STAGE,model_goal_complete=False,latest_completed_report='R52_REPORT.md',latest_scientific_decision=decision['scientific_decision'],latest_key_result_review='reviews/R52_ROOT_DIRECTION_001.json')
 if fn=='PROGRAM.json':x.update(status='R52_closed_negative_delivery_pending_goal_active',updated_at_utc=NOW)
 else:x.update(project_status='R52_closed_negative_goal_active',current_activity='root_delivery_then_new_endpoint_protocol',current_direction=action,updated_UTC=NOW)
 write(fn,x)
for fn,key,ident in [('jobs.json','jobs','stage'),('experiments.json','experiments','id')]:
 x=read(fn);a=next(v for v in x[key] if v[ident]==STAGE);a.update(status='closed_negative_delivery_pending',scientific_decision=decision['scientific_decision'],science_s=n['elapsed_s'],native_exit=0,final_receipt='work/'+STAGE+'/FINAL_RECEIPT.json',model_goal_complete=False);write(fn,x)
for fn in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:
 p=R/fn;_,tail=p.read_text('utf8').split('## 以下为历史记录',1)
 top='# 当前执行：R52固定配方失败关闭，长期目标active\n\n两块联合配对在Q2–Q7改善目标PSD与噪声，但Q1/Q8噪声仍失败，0新响应训练。本轮不扫描组数，不删区、不降原A–F。下一协议将检查保留基线的平滑脉冲电荷修正。主助手独立执行，无子智能体/计算预算，机器资源、单科学作业和本机E/D约束保持。当前完成私有Git和D盘交付。详见R52_REPORT.md。\n\n'
 p.write_bytes((top+'## 以下为历史记录'+tail).encode())
print(json.dumps({'science_s':n['elapsed_s'],'cumulative_science_s':bl['actual_science_seconds_recorded'],'model_goal_complete':False,'decision':decision['scientific_decision']},indent=2))
