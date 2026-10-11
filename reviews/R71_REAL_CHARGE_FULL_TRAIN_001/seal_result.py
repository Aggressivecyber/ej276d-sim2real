"""Seal actual R71 component outcome, visual review and active-goal continuation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work'/P.name
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'seal_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
review=read(P/'ROOT_RESULT_REVIEW.json');adc=read(P/'adc_representation_audit/ROOT_ADC_AUDIT_RECEIPT.json')
assert review['root_saved_evidence_recomputation_pass']and adc['native']['actual_worker_native_exit']==0
assert all(v['ADC_integer_codes_bitwise']for v in adc['result']['results'].values())
assert read(R/'jobs.json')['active_scientific_jobs']==0
figs=['CHARGE_COMPONENT_DIAGNOSTICS.png','FULL_TRAIN_LOSSES.png']
write(P/'VISUAL_QA.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_images_opened=True,
    files=[dict(path='work/'+P.name+'/'+n,sha256=sha(S/n))for n in figs],
    findings=['CDF charge matching improvement visible with all original laws','SOURCE PSD remains narrower and differently shaped than REAL',
              'Same-treated REAL curves shown separately','Full50000 loss trace visible; no checkpoint selection','Labels and legends readable, no clipped content'],
    model_goal_complete=False))
report='''# R71：实测电荷目标下的全量公共响应头

本轮完整训练和数值检查已完成；**完整模型 A–F 尚未完成**。用户要求“数小时内能完成就直接全量”，实际每步6593 SOURCE+21047 REAL、唯一50000步终点已执行。没有选检查点、删父事件或继续旧R65。

主训练耗时 **351.434秒（5.86分钟）**，包括工程测时、保存、全人口生成和检查的主作业总耗时 **363.575秒（6.06分钟）**，原生exit0。工程50步计入单独暴露，之后权重、空Adam与CPU/CUDA RNG恢复同初态。实测步时中位6.785毫秒，原保守总耗时估计22.48分钟，满足预登记4小时执行条件。

| 全部 TRAIN 人口的实际 canonical ADC 指标 | 恒等头公共控制 | 训练后 SOURCE | 训练后同处理 REAL |
|---|---:|---:|---:|
| 电荷 KS，对原 REAL | 0.256430 | **0.035679** | 0.003231 |
| 电荷 W1，mV ns | 3028.994 | **227.903** | 13.783 |
| 总体 signed-PSD W1 | 0.113850 | **0.042892** | 0.000225 |
| 输出自身 Q1–7 条件 PSD W1 均值 | 0.114901 | **0.066180** | 0.001190 |
| baseline RMS W1，V | 9.12924e-5 | 9.12997e-5 | 9.53507e-7 |
| baseline ACF1 W1 | 0.060386 | 0.060126 | 0.000183 |

电荷 W1 比同噪声恒等头控制减少约92.5%，但PSD总体/条件两项仍高于最终目标0.010；这不是held-out A/B验收。当前固定头只拟合Q，完整波形残差保持零，图中仍能看到源PSD峰形过窄。没有新设电荷通过线，也没有运行或用DEV/EVAL选择模型。

SOURCE原Q→输出Q Spearman为0.995745，原PSD→输出PSD为0.605092；全部6593父事件、原组成、顺序与均匀权重保留。原TRAIN purehad bins3/4分母为482/435，输出谷比例均0。它们是TRAIN描述，不能与held-out213/181分母混用。输出支持外质量从恒等控制43.061%变为14.470%，父事件没有删除。范围投影触发率从12.392%变为8.024%；同处理REAL为0。原支持标记与所有四阶段条件迁移均另存。

公共路径为480维完整归一化波形→16维SiLU电荷头，预测有界Q*；连续单位电荷模板用于lifting和噪声投影，之后是原Box及ADC。SOURCE与REAL无域标记或身份旁路。所有学习/物理计算float64。该阶段是直接无配对电荷分布训练；不把它称为完成的完整波形模型。

独立NumPy参考的128条preADC波形与主结果逐位一致，独立Q*最大相对差约1.92e-15。所有全人口输出的最终float32相对Q*积分误差最高2.94e-8，低于1e-5。重新加载唯一权重并恢复噪声流后，TRAIN SOURCE2000+REAL2000的Q*/preADC/canonical输出逐位重放通过；最终held-out独立4000门仍未执行。

原`NUMERICAL_REFERENCE.json`中`ADC_code_equal=false`实际比较的是float64电压数组。补充独立量化审计没有模型前向或训练：128条/61440点的NumPy/CUDA **整数ADC码完全相同**，电压表示最大差1.11e-16V。REAL恒等控制电荷KS复算相差1/21047，位于相同ADC整数电荷并列值的CDF浮点拆分范围；原报告值不追改，整数码审计与严格上界另存。该现象不改变本轮学习后SOURCE/REAL的KS结论。

主作业峰值进程树RSS2.79GiB、GPU进程占用约1.00GiB，CPU0..11；均在CPU12/RSS48GiB/GPU10GiB内。本地回收和独立复算由1CPU/1GiB守护。新增本地源码、参数、数组、日志及缓存均位于E/D。图已实际打开复核。

下一方向：保留这个冻结电荷前置，注册一个从零训练的轻量完整480点波形残差，按实际canonical输出Q更新条件分布损失，联合保留PSD、源顺序/谷、REAL保持、噪声和边界约束。新完整方法同样先测完整更新开销，若保守完整终点在数小时内就直接全量。此处只记下一方向，尚未注册或启动下一科学作业。长期目标保持active；同一冻结方法的v5 A–F、D/E和最终交付仍需真实证据。

证据：`work/R71_REAL_CHARGE_FULL_TRAIN_001/`中的唯一FINAL_HEAD.pt、训练/数值/迁移/重放回执和图；`reviews/R71_REAL_CHARGE_FULL_TRAIN_001/ROOT_RESULT_REVIEW.json`、`CANONICAL_Q_KS_ROUNDOFF_AUDIT.json`及`adc_representation_audit/ROOT_ADC_AUDIT_RECEIPT.json`。实际私有Git与D盘交付以`R71_DELIVERY_RECEIPT.json`为准。
'''
(R/'R71_REPORT.md').write_text(report,'utf8')
snapshot=P/'before_result_closure';snapshot.mkdir()
for name in ['AGENTS.md','STATUS.md','PLAN.md','PROJECT_PROGRESS.json','PROGRAM.json','experiments.json','jobs.json']:shutil.copy2(R/name,snapshot/name)
prefix='''# 当前执行：R71 全量电荷响应已完成，v5目标继续active

R71实际完成50000次完整6593 SOURCE+21047 REAL更新，主训练5.86分钟，主进程6.06分钟、exit0。电荷KS从同路径恒等控制0.256430降到0.035679，W1从3028.994降到227.903mV ns；总体PSD W1仍0.042892、输出Q1–7均值0.066180。仅TRAIN电荷组件证据，完整模型A–F未完成。数值参考、TRAIN4000重放和实际ADC整数码审计通过，原float64表示差异单独保留。

没有活动科学作业。下一方向为冻结此Q前置、从零训练轻量完整波形残差，保持全部父事件、来源约束、同处理REAL和v5；先测完整步时，数小时内则直接全量。下一科学任务尚未登记。R65不恢复。原始v4及历史失败不改；机器共享资源/单作业/E-D存储要求保持。主助手统筹，辅助实现静态审核使用既有gpt-6.1-sol/max。

结果：R71_REPORT.md；根复算：reviews/R71_REAL_CHARGE_FULL_TRAIN_001/ROOT_RESULT_REVIEW.json；实际交付以R71_DELIVERY_RECEIPT.json为准。

## 以下为历史记录

'''
for name in ['AGENTS.md','STATUS.md','PLAN.md']:(R/name).write_text(prefix+(R/name).read_text('utf8'),'utf8')
now=datetime.now(timezone.utc).isoformat()
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
    d=read(R/name);d.update(updated_UTC=now,status='R71_complete_TRAIN_charge_component_full_model_incomplete',current_stage=P.name,
        current_stage_progress='completed50000_and_postchecks_root_verified',current_activity='R71_delivery_and_next_full_waveform_preparation',
        active_scientific_jobs=0,current_job=None,application_goal_status='active',model_goal_complete=False,numeric_execution_released=False,
        latest_completed_stage=P.name,latest_report='R71_REPORT.md',current_acceptance_version=5,
        latest_acceptance_observation_role='TRAIN_charge_component_only_not_A_F',numerical_protocol_status='R71_component_pass_final_whole_waveform_not_run',
        next_prepared_task=None,next_action='Deliver sealed R71; preregister light full-waveform residual with frozen public Q head and measured all-parent cost',
        current_direction='Extend verified public charge component to jointly trained full waveform residual',
        delivery_status='R71_delivery_pending_actual_private_Git_and_D_receipt')
    write(R/name,d)
for name,key in [('jobs.json','jobs'),('experiments.json','experiments')]:
    d=read(R/name);j=next(v for v in d[key]if v.get('stage',v.get('id'))==P.name)
    j.update(status='COMPONENT_ENDPOINT_AND_POSTCHECKS_COMPLETE',completed_UTC=now,main_updates=50000,engineering_updates=50,native_exit=0,
        report='R71_REPORT.md',root_review='reviews/'+P.name+'/ROOT_RESULT_REVIEW.json',model_goal_complete=False,
        supplementary_quantizer_audit='reviews/'+P.name+'/adc_representation_audit/ROOT_ADC_AUDIT_RECEIPT.json')
    d.update(updated_UTC=now,active_scientific_jobs=0,active_remote_scientific_jobs=0,active_local_scientific_jobs=0);write(R/name,d)
write(P/'ROOT_CLOSURE_RECEIPT.json',dict(UTC=now,complete_main_updates=50000,root_review_pass=True,images_actually_opened=True,
    goal_status='active',current_acceptance_version=5,whole_model_goal_complete=False,active_scientific_jobs=0,
    new_scalar_or_benefit_gate_added=False,initial_failed_exact_KS_recompute_preserved=True,
    next_stage_registered=False,guard={**guard,**memory()}))
print(json.dumps(dict(sealed=P.name,goal_status='active',whole_model_complete=False,report='R71_REPORT.md')))
