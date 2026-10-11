"""Seal a source-only provenance trace; no data-array or model decoding."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,shutil
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def read(p):return json.loads(p.read_text('utf8'))
inputs=['work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/run.py','work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/config.json',
 'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/REPORT.md','work/P1H_FIXED_INPUT_COMPOSITION_001/REPORT.md',
 'work/R23_ONLINE_STANDARDIZED_CRITIC_001/prepare_data.py','work/R23_ONLINE_STANDARDIZED_CRITIC_001/data_support.py',
 'work/R33_RESIDUAL_OT_FLOW_001/config.json','work/R33_RESIDUAL_OT_FLOW_001/TRAINING_Q_BINDING.json',
 'work/R33_RESIDUAL_OT_FLOW_001/DATA_ROLES.json','work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/config.json',
 'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/source_flow_core.py','work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/evaluation.py',
 'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001/diagnose_train.py',
 'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001/DATA_ACCESS_RECEIPT.json',
 'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001/endpoint_geometry.py','CURRENT_ROLE_INDEX.json',
 'reviews/R45_GOAL_COMPATIBILITY_NOTE_001.md','PROJECT_ACCEPTANCE.md','PROJECT_ACCEPTANCE.json']
bindings=[]
for name in inputs:
 f=R/name;bindings.append(dict(path=name,bytes=f.stat().st_size,sha256=sha(f)))
now=datetime.now(timezone.utc).isoformat()
report='''# R68 来源追查补充：输入电荷并非已独立标定的物理不变量

本补充只读取现有代码、元数据和封存报告；未重新计算任何事件或拟合模型。它补充已交付R68电荷包络证书，不修改该结果或原A–F。

当前可确认：源和实测来自不同事件人口；源的幅度/形状带有固定解析模型与0.15强子可见能量代理。已有报告明确把独立淬灭、增益与能量标定留为unknown。没有证据把R68电荷谱差距全部归因为训练不足，也没有证据据此宣称原始仿真或实测存在一个可直接修正的单位错误。

## 已核对的来源链

1. R2 REPORT.md:7–15与run.py:75–97绑定服务器旧G4沉积CSV，SHA256 da68b5a55777c60f23c178483ef91707fd69f3ab1d6f1e7fa11044a091e1b0ff。封存报告记录176121条已存沉积历史，经原身份/hash规则选8192个TRAIN父，内部6593/1599；未写入CSV的零沉积历史没有被恢复。本次不重新读取CSV，也不重选任何父事件。
2. R2 run.py:33–43构造512点清洁波形：电子分量衰减13/35/270、权重0.7/0.15/0.15；强子分量0.15×沉积、衰减13/59/460、权重0.5/0.1/0.4；共同上升5，名义起点128。令E=ee+0.15had，名义全脉冲电荷q_nominal=(E+0.03918)/0.000132506/1000；先以512点总和归一，再按半峰上升沿裁成480点float32。它是现有解析响应假设，不是微观沉积时间记录或新物理标定。
3. R23 data_support.py:40–53读取既有G4_TRAIN_CACHE中的clean_expectation_V，按internal_training选原6593条。REAL为原拟合2592条加R14新增18455条共同QC有效记录，共21047；固定599内部验证不混入。prepare_data.py:12–17将这两域、原父ID、h和原分区写入TRAIN_PRIVATE.npz。
4. R33 config.json的training_input_paths与remote_input_sources把该R23文件送至R33远端inputs目录。R63 config.json仍明确指向这一远端TRAIN和本地R23来源；R64 diagnose_train.py:187–213检查6593×480/21047×480、原SOURCE ID摘要、原分区与计数，再沿同一q定义保存原电荷。R64 DATA_ACCESS_RECEIPT给出原父ID数组摘要8a9cd601e1846fca27f15b68c116059930340ba9700fe9bc49a2f06af65c4f22。
5. R64 endpoint_geometry.py:375–382保存SOURCE_original_q与REAL_canonical_q。R68只解码这两个标量数组；没有从当前报告推导或替换任何源电荷。

## 两种电荷量必须区分

源码的q_nominal用于裁剪前512点总和归一；当前数值门则用最终480点输入的q_window=sum(v[100:400])-3sum(v[0:100])。当前评价电荷为同一减基线积分再乘1000。两个量采用不同求和区间；源码没有在裁窗后重新强制q_window=q_nominal。本次只据公式指出它们不是同一个定义，没有计算其逐事件差值，也没有证明这一选择是bug。

P1H REPORT.md:17明确：REAL声明能量来自converter对原integral_mV_ns套CAL_A=0.000132506、CAL_B=-0.03918；标定链仍unknown。对本项目数据通过E=ee+0.15had的算术检查，只说明实现一致，不能独立验证淬灭或MeVee。R2 REPORT.md:15也明确，其曾学得的gain与能量指数依赖输入谱和无配对分布，不能当作独立器件参数。

因此不能把一个无配对电荷分位数映射称为已经验证的物理校准，也不能把名义q写回原q字段后宣称现行数值门通过。原A–F继承的是对实际输入q_window的1e-5检查；任何改变这项目标的方案都属于验收变更，尚未执行或获准。

## 人口与评价的限制

R33 DATA_ROLES记录TRAIN纯EE/纯had/mixed为1905/4475/213，原DEV为417/1518/65。来源选择已不同；PDG与沉积组成不能混为同一个标签。CURRENT_ROLE_INDEX记录REAL来自同一已暴露采集，50条一组是人工分组，不是物理批次。真实入射谱、触发效率、未记录零沉积、淬灭与独立能量标定，不由当前无配对波形相似性共同识别。

本补充未重新评价原DEV。已封存R45说明给出的2000源/实测q KS=0.2685，与R68完整TRAIN保守包络下界0.248400366是不同数据人口、不同统计陈述。二者都不允许把原固定HGB留出门改判为普遍不可能；更不能据此删417支持外父、重权或降低验收线。

## 对下一步的约束

保留原验收、原父人口和R65成本hold。当前没有已登记的新训练；本轮源追查不提供恢复full-D或重复R67的收益证据。下一方案必须同时处理可改变的波形/PSD/噪声残差与不可由现有电荷保持类消除的人口差异，明确其评价解释，并先用封顶、同起点/同成本的联合指标比较证明收益。仅改批量、延长步数、缩小学习率，或悄悄放弃电荷门，都不构成由本证据支持的解决方案。

若要验证物理电荷标度或改变无条件总体匹配目标，需要明确的标定/源谱依据或用户对具体新目标的授权。这里没有把该可能的后续条件冒称为本轮已证明的普遍阻塞；原A–F模型目标仍active且未完成。

复核入口：INPUT_HASH_RECEIPT.json绑定以上实际读取来源；STATIC_AUDIT_RECEIPT.json说明工作范围。R68主证书、图示与实际Git/D交付保持在R68_REPORT.md、R68_DELIVERY_RECEIPT.json。本补充的实际提交/备份另见R68_PROVENANCE_DELIVERY_RECEIPT.json。
'''
(P/'REPORT.md').write_text(report,encoding='utf8')
write(P/'INPUT_HASH_RECEIPT.json',dict(UTC=now,inputs=bindings,raw_scientific_file_decodes=0,scope='source and saved report/metadata only'))
write(P/'STATIC_AUDIT_RECEIPT.json',dict(UTC=now,status='STATIC_SOURCE_TRACE_COMPLETE',new_training=0,model_calls=0,
  source_csv_reads=0,waveform_array_decodes=0,DEV_array_reads=0,EVAL_reads=0,CAL_array_reads=0,
  input_population_or_weights_changed=False,acceptance_changed=False,calibration_or_quench_independently_verified=False,
  source_code_window_definition_difference_not_declared_bug=True,original_AF_impossibility_proven=False,
  R65_automatic_resume=False,report_SHA256=sha(P/'REPORT.md'),resource=dict(guard,**memory())))
before=P/'before_metadata';before.mkdir()
for name in ['PROJECT_PROGRESS.json','PROGRAM.json','AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','STATUS.md']:shutil.copy2(R/name,before/name)
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
 d=read(R/name);d.update(updated_UTC=now,latest_static_provenance_review='reviews/R68_SOURCE_PROVENANCE_AUDIT_001/REPORT.md',
   current_activity='R68_finite_charge_and_source_trace_closed_no_new_training',
   next_action='Synthesize a compatibility-aware lower-cost scientific decision from preserved evidence; do not resume R65, change source charge/population, or claim fixed-HGB impossibility.')
 write(R/name,d)
old='Next inspect existing input population/scale provenance before further compute.'
new='Existing input provenance has now been traced: nominal analytic source scaling, independent calibration/quench unknown, and distinct full-pulse/window charge definitions. No new training is registered; original charge/population/A-F remain unchanged. See reviews/R68_SOURCE_PROVENANCE_AUDIT_001/REPORT.md.'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','STATUS.md']:
 text=(R/name).read_text('utf8');assert old in text
 (R/name).write_text(text.replace(old,new,1),encoding='utf8')
print(json.dumps(dict(status='STATIC_SOURCE_TRACE_COMPLETE',input_files=len(bindings),report_SHA256=sha(P/'REPORT.md'),resource=dict(guard,**memory())),ensure_ascii=False))
