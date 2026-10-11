"""Review frozen compact evidence and render already-computed CDF curves."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction as F
import hashlib,json,math,sys
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R68_TRAIN_CHARGE_COMPATIBILITY_001'
A=R/'reviews/R68_DIRECTION_PREPARATION_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P/'report_runtime')
def read(p):return json.loads(p.read_text('utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now(timezone.utc).isoformat()
x=read(S/'RESULT.json');native=read(P/'NATIVE_RECEIPT.json');reg=read(P/'REGISTRATION.json')
assert native['actual_worker_native_exit']==0 and native['elapsed_s']<60 and native['termination_reason'] is None
assert x['artificial_cases_passed']==5 and x['independent_witness_count_verified']
assert x['data_access']['decoded_fields']==['SOURCE_original_q','REAL_canonical_q']
for row in reg['files']:assert sha(R/row['path'])==row['sha256'],row['path']
assert sha(R/reg['input_archive'])==reg['input_SHA256']
for rel,info in read(A/'component_audit/SHA256_MANIFEST.json')['inputs'].items():assert sha(R/rel)==info['SHA256']
for row in read(A/'charge_compatibility/INPUT_HASH_RECEIPT.json')['inputs']:assert sha(R/row['path'])==row['sha256']
b=x['robust_interval_result'];w=b['witness'];ns=w['SOURCE_N'];nr=w['REAL_N']
exact=F(w['SOURCE_lower_count'],ns)-F(w['REAL_count'],nr)
assert w['direction']=='SOURCE_CDF_above_REAL'
assert exact==F(b['KS_lower_bound']['numerator'],b['KS_lower_bound']['denominator'])
assert exact>F(1,5) and (1+exact)/2==F(b['best_threshold_BA_lower_bound']['numerator'],b['best_threshold_BA_lower_bound']['denominator'])
assert ns==6593 and nr==21047 and w['integer_gap_numerator']==34468748
# Binding to the executable float64 1e-5 threshold: the added margin dominates this discrepancy.
binary_epsilon_gap=abs(F.from_float(1e-5)-F(1,100000))
assert binary_epsilon_gap*301 < F(1,10**17)
# Bounded preADC gives |q|<=300; a nonempty relative-charge class has input q<301.
# Worst-case float64 reductions of bounded coordinates are far below 1e-7 V samples.
u=F(1,2**53);gamma299=299*u/(1-299*u);gamma99=99*u/(1-99*u)
reduction_bound=150*gamma299+150*gamma99+1000*u
assert reduction_bound+binary_epsilon_gap*301 < F(1,10**7)
M=16383
first_adc_error=F(1,2*M)+F(1,2**24)+F(1,2**11*M)
recode_error_codes=M*(F(1,2**24)+F(1,2**25)+F(1,2**50))
assert first_adc_error+F(1,2**50)<F(1,M) and recode_error_codes<F(1,2)
# Root independently adds the original saved probe scalars; no new probe is evaluated.
parts=['full_window','coordinate_RMS','PSD','noise','temporal_joint','source_order','source_valley','REAL_identity','box_correction']
means={};scores={}
for arm in ['baseline','full','mini']:
 ev=read(R/('work/R67_EQUAL_COST_REPAIR_001/'+arm+'/EVALUATION.json'))
 means[arm]={k:math.fsum(p['pieces'][k]for p in ev['probes'])/8 for k in parts}
 scores[arm]=math.fsum(p['score']for p in ev['probes'])/8
 assert abs(sum(means[arm].values())-scores[arm])<1e-12
 saved=read(A/'component_audit/DECOMPOSITION.json')['summaries'][arm]
 for k in parts:assert abs(means[arm][k]-saved[k]['mean'])<1e-12
diff={k:means['full'][k]-means['mini'][k]for k in parts}
for k in ['noise','temporal_joint']:assert diff[k]>0
assert sum(diff[k]for k in parts[:5])<0
rows=[]
for k in parts:
 rows.append(f'| {k} | {means["full"][k]:.9f} | {means["mini"][k]:.9f} | {diff[k]:+.9f} |')
review=dict(UTC=now,status='ROOT_ACCEPTED_FINITE_TRAIN_CERTIFICATE_WITH_LIMITS',native=native,
    frozen_source_protocol_input_and_original_acceptance_hashes_match=True,all_agent_input_hashes_match=True,
    exact_count_identity=f'{w["SOURCE_lower_count"]}*{nr}-{w["REAL_count"]}*{ns}={w["integer_gap_numerator"]}',
    exact_KS_lower_bound=str(exact),binary_threshold_gap_times301=float(binary_epsilon_gap*301),
    conservative_float64_reduction_bound=float(reduction_bound),declared_arithmetic_slack=1e-7,
    first_ADC_error_V_upper=float(first_adc_error),canonical_recode_error_codes_upper=float(recode_error_codes),
    preADC_range_and_no_saturation_are_required=True,current_R63_ADC_only=True,
    pure_integer_CDF_witness_verified=True,right_continuous_CDF_ties_processed_together=True,
    scope='fixed TRAIN empirical laws under all-parent charge preservation; not original DEV/HGB acceptance',
    original_HGB_AF_impossibility_proven=False,original_A_F_complete=False,
    root_R67_components_recomputed_from_saved_JSON=True,
    R67_mean_full_minus_mini=diff,R67_full_minus_mini_D5=sum(diff[k]for k in parts[:5]),
    R67_full_minus_mini_C4=sum(diff[k]for k in parts[5:]),
    prior_commentary_correction='D5 sum is lower; noise and temporal_joint individually are higher. No all-five improvement claim.',
    R65_automatic_resume=False,active_scientific_jobs=0,resource=dict(guard,**memory()))
write(P/'ROOT_RESULT_REVIEW.json',review)
report=f'''# R68：先核清人口约束，再决定训练成本

R67 等成本短试验没有支持恢复昂贵的全量训练。对保存结果的分解进一步显示，五项分布损失**合计**略低，四项保持约束的损失更高；不能把局部分布改善当作联合收益。R68 的一次 TRAIN 电荷标量分析又发现：即使允许原电荷误差及保守 ADC 舍入，电荷 CDF 的差距仍至少为 **{float(exact):.9f}**。这是当前人口和约束留下的残差下限，增加训练不能消除它。

本轮原 A–F 未完成，也没有证明其固定 HGB 留出门在所有方法下必然失败。R65 保持退休与成本 hold；没有启动新模型、恢复训练或重跑 R67。

## R67：收益不足的分项证据

两个端点由同一 update500 模型、Adam 和随机流状态出发，固定相同学习率，full20 用98.519539秒、mini979用98.614961秒。共同评分为 full {scores['full']:.9f}、mini {scores['mini']:.9f}；full 高30.394%。这只是已保存的短程 TRAIN 比较，mini 也不是合格最终方法。

| 保存评分项 | full | mini | full−mini |
|---|---:|---:|---:|
{chr(10).join(rows)}

五D合计差 {review['R67_full_minus_mini_D5']:+.9f}，四C合计差 {review['R67_full_minus_mini_C4']:+.9f}。source_valley 占净总差85.72%；事后扣去该项，full仍8/8较差，均值差0.197379286，固定端点经验配对区间[0.162481838,0.231828255]。扣项只用于解释，原评分和63/102退化门失败保持。完整表、精确枚举经验重采样方法与限制见 reviews/R68_DIRECTION_PREPARATION_001/component_audit/COMPONENT_AUDIT.md。

C 的更新/曝光机会之比为979/20=48.95，值得用于设计下一次有依据的低成本机制检查；但 Adam 轨迹、曝光和 D 计算规则也共同改变，不能声称已识别独立因果作用。没有逐更新分项轨迹来补造这个解释。两项 D（noise、temporal_joint）实际更高。

## 原验收的电荷含义

PROJECT_ACCEPTANCE.md 的 C 继承原 R43 六门。R43、R49、R63 的源码都把 preADC 电荷残差定义为 abs(q(output)-q(original_input))/q(original_input)，并要求 f32 主输出最大值≤1e-5。它与 N/2N 电压积分误差是不同检查。源秩合格不授权把电荷目标替换为 REAL 分位数，或换分母、积分窗、源父权重。静态证据及文件行号见 reviews/R68_DIRECTION_PREPARATION_001/charge_compatibility/REPORT.md。

## 本次精确有限样本结果

仅解码 R64 已保存的 SOURCE_original_q（6593条）与 REAL_canonical_q（21047条），保留完整等权人口。电荷为 q=sum(v[100:400])-3sum(v[0:100])，单位V samples。使用当前无裁剪的R63 ADC链；每一源父允许的电荷区间半径为 q/100000+600/16383+1e-7。600是电荷系数L1范数；ADC项允许每坐标一完整LSB，算术项另外包住浮点求和与原二进制1e-5常数微差。这是用于证明下界的宽包络，不是改变原容差。

| 描述 | 值 |
|---|---:|
| 原保存两域电荷经验KS | {x['raw_finite_charge_comparison']['KS_lower_bound']['decimal']:.9f} |
| 允许上述变化后的KS下界 | {float(exact):.9f} |
| 完整TRAIN经验律的对应电荷阈值BA下界 | {float((1+exact)/2):.9f} |
| 见证阈值，V samples | {w['threshold']['decimal']:.12f} |
| 阈值以下SOURCE不可移出的父数 | {w['SOURCE_lower_count']} / {ns} |
| 同阈值下REAL父数 | {w['REAL_count']} / {nr} |
| 原生进程耗时 | {native['elapsed_s']:.6f}秒 |
| 科学算术耗时（含导入/输出） | {x['analysis_elapsed_s']:.6f}秒 |
| 工作进程峰值RSS | {x['resource']['peak_RSS_bytes']}字节 |

在该阈值处，即使每个SOURCE都选择最有利的允许电荷，至少{w['SOURCE_lower_count']}/{ns}仍落在阈值以下，而REAL只有{w['REAL_count']}/{nr}。两者之差恰为 {exact}。源条件输出即使是随机分布，只要逐父始终处在该区间，仍服从同一CDF包络。端点和CDF计数使用精确Fraction/整数交叉乘法，保存见证另用直接逐项计数核对；5个人工边界例通过。

## 结论边界与下一步

这排除了所声明类内**本完整TRAIN有限经验联合Q–PSD律的严格相等**。阈值BA由同一完整TRAIN标量人口选定并评价；它没有留出置信区间，不是原2000 DEV父事件、固定HGB或未知采集总体的结论。未读取DEV/EVAL/CAL，未拟合HGB、LP、密度或响应网络。不能据此宣称原A–F普遍不可能，也不能把原验收改为条件成功。

下一项有用工作是从现有输入谱、标度和事件构造记录追查这一电荷人口差异的来源，区分输入分布差异与转换实现错误，并核对它与原2000父评价人口的已封存证据关系。只分析既有合法证据；不以更多步数、较低D或改变源权重作为解决人口残差的依据。若以后提出新训练，必须指出能改变的剩余误差及先验证收益的封顶比较；本报告不授权任何昂贵恢复或新试验。

## 执行、图示与复核

协议 tasks/R68_TRAIN_CHARGE_COMPATIBILITY_001.md 在标量解码前绑定。工作进程 native exit0，CPU affinity1、硬RSS上限1GiB，原生60秒上限内完成；没有重试。原PROJECT_ACCEPTANCE两个文件的SHA256保持。jobs.json 中过时的R65 running条目已按新的服务器原生观测修复，R66/R67终态补齐；旧账本已保存。服务器观测没有本项目科学Python进程，GPU进程列表为空。

结果：work/R68_TRAIN_CHARGE_COMPATIBILITY_001/RESULT.json；数据用途：同目录DATA_ACCESS_RECEIPT.json；根复核：reviews/R68_TRAIN_CHARGE_COMPATIBILITY_001/ROOT_RESULT_REVIEW.json；图：work/R68_TRAIN_CHARGE_COMPATIBILITY_001/R68_CHARGE_ENVELOPE.png。图示由保存CDF曲线绘制，精确见证以JSON整数为准；实际打开检查另记VISUAL_QA_RECEIPT.json。交付完成后查看R68_DELIVERY_RECEIPT.json取得真实私有Git远端SHA与D盘哈希回执；报告本身不预先宣称交付成功。
'''
(R/'R68_REPORT.md').write_text(report,encoding='utf8')
# Plotting reads only the output curve already computed by the completed worker.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with np.load(S/'CDF_ENVELOPE_PRIVATE.npz',allow_pickle=False) as z:curve=z['columns']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
for ax in axes:
 ax.fill_between(curve[:,0],curve[:,1],curve[:,2],step='post',color='#007f86',alpha=.23,label='Allowed SOURCE CDF envelope')
 ax.step(curve[:,0],curve[:,1],where='post',color='#007f86',lw=1)
 ax.step(curve[:,0],curve[:,2],where='post',color='#007f86',lw=1)
 ax.step(curve[:,0],curve[:,3],where='post',color='#d66124',lw=1.5,label='Frozen REAL CDF')
 ax.set(xlabel='Charge (V samples)',ylabel='Cumulative probability',ylim=(-.01,1.01))
 ax.grid(alpha=.16)
axes[0].set(xlim=(min(0,float(curve[:,0].min())),float(curve[:,0].max())*1.02),title='All original TRAIN parents')
axes[0].legend(loc='lower right',frameon=False,fontsize=9)
axes[1].set(xlim=(0,4),ylim=(0,.62),title='Low-charge detail and exact witness')
t=w['threshold']['decimal'];y1=float(F(w['REAL_count'],nr));y2=float(F(w['SOURCE_lower_count'],ns))
axes[1].plot([t,t],[y1,y2],color='#263746',lw=2)
axes[1].scatter([t,t],[y1,y2],s=26,color='#263746',zorder=5)
axes[1].annotate(f'Unavoidable CDF gap\n{float(exact):.4f}',xy=(t,(y1+y2)/2),xytext=(2.05,.085),
                 arrowprops={'arrowstyle':'->','color':'#263746'},fontsize=11,color='#263746')
fig.suptitle('Charge preservation leaves a population mismatch',fontsize=16,fontweight='bold')
fig.supxlabel('SOURCE n=6,593  |  REAL n=21,047  |  finite TRAIN laws; no held-out HGB claim',fontsize=10)
fig.savefig(S/'R68_CHARGE_ENVELOPE.png',dpi=160)
fig.savefig(S/'R68_CHARGE_ENVELOPE.svg')
plt.close(fig)
# Normalize XML line-end whitespace before Git without changing drawing/text content.
svg=S/'R68_CHARGE_ENVELOPE.svg'
svg.write_text('\n'.join(line.rstrip()for line in svg.read_text('utf8').splitlines())+'\n',encoding='utf8')
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
 d=read(R/name);d.update(updated_UTC=now,status='R68_finite_TRAIN_charge_certificate_closed',project_status='R68_finite_TRAIN_charge_certificate_closed',
   current_stage_progress='completed_root_reviewed',current_activity='R68_visual_QA_and_delivery',
   latest_completed_stage=S.name,latest_diagnostic_stage=S.name,latest_completed_report='R68_REPORT.md',
   latest_completed_review='reviews/'+S.name+'/ROOT_RESULT_REVIEW.json',active_scientific_jobs=0,
   latest_diagnostic_conclusion='Exact finite TRAIN charge KS lower bound0.248400366 under declared preservation/ADC class; not fixed-HGB or A-F impossibility.',
   latest_scientific_decision='Preserve R67 negative benefit and R65 hold; trace input population/scale evidence before any new training.',
   next_action='Read-only provenance and population compatibility analysis; no automatic R65 resume, new pilot, population or acceptance changes.')
 write(R/name,d)
jobs=read(R/'jobs.json');j=next(v for v in jobs['jobs']if v['stage']==S.name)
j.update(status='closed_finite_TRAIN_charge_certificate',active_scientific_jobs=0,result='work/'+S.name+'/RESULT.json',
         root_review='reviews/'+S.name+'/ROOT_RESULT_REVIEW.json',model_goal_complete=False)
write(R/'jobs.json',jobs)
header='# Current: R68 finite TRAIN charge bound completed; original A-F unmet\n\nThe exact preserved-charge/ADC envelope leaves aTRAIN charge KS gap of at least0.248400366, corresponding to a descriptive threshold BA lower bound0.624200183 on these same finite laws. This is not a proof against the original held-out fixed-HGB gate. R67 full-D remains negative in the joint equal-cost test; itsD5 sum is slightly lower butC4 higher, and noise/temporal D terms are individually worse. R65 remains retired on cost hold. No active scientific worker, no new training, no source-weight or A-F change. Next inspect existing input population/scale provenance before further compute. See R68_REPORT.md and the root review.\n\n## Earlier records\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md','STATUS.md']:
 (R/name).write_text(header+(R/name).read_text('utf8'),encoding='utf8')
write(P/'REPORT_RENDER_RECEIPT.json',dict(UTC=datetime.now(timezone.utc).isoformat(),PNG_SHA256=sha(S/'R68_CHARGE_ENVELOPE.png'),
      SVG_SHA256=sha(svg),source='saved_CDF_ENVELOPE_PRIVATE.npz',new_waveform_or_model_evaluations=0,resource=dict(guard,**memory())))
print(json.dumps(dict(status=review['status'],robust_KS=float(exact),PNG=str(S/'R68_CHARGE_ENVELOPE.png'),resource=dict(guard,**memory())),ensure_ascii=False),flush=True)
