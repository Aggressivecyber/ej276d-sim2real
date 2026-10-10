from pathlib import Path
from datetime import datetime,timezone
import json,os
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text('utf8'))
def fmt(x):return 'undefined'if x is None else f'{x:.6g}'
def ci(x):return '['+', '.join(fmt(v)for v in x)+']'
C=read('config.json');M=read('METRICS_COMPACT.json');G=read('GATE_METRICS.json');A=read('ACCEPTANCE_A_C.json')
N=read('TRAIN_SOLVER_CHECK.json');RP=read('REPLAY_RECEIPT.json');H=read('RUN_HISTORY.json')
source=C['arms'];real=['GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_COMPLETION_REAL_CONTROL']
views=['full_historical_shape','normalized_u480','full_voltage','Q_PSD_2D','baseline_local']
lines=['# R49：冻结两组权重的 Heun128 完整响应评价','',
       '本轮没有新增训练或噪声校准。R48 两组 50000 更新末点采用共同 Heun128，256 步仅作原128条 TRAIN检查的参考。原阈值、源人口与全部权重保持。本次是用户授权的本机恢复；本机数组、评价器和重放单独执行，远端旧指标不绑定为本机结果。',
       '',f"两臂数值门均通过，8000主输出独立重放 {'通过'if RP['pass']else'失败'}。指定主候选的 A/B/C 为 "+'/'.join(str(A[source[1]][k]['pass'])for k in ['A','B','C'])+
       f"；决定：{G['decision']}。项目模型目标尚未完成，D/E未执行。对照不能替代主候选成为赢家。",'']
lines+=['## 数值与复现','', '| 冻结臂 | TRAIN RMS中位/LSB | p95/LSB | 最大相对q误差 |', '|---|---:|---:|---:|']
for arm in source:
 r=N['models'][arm];lines.append(f"| {arm} | {fmt(r['RMS_median_LSB'])} | {fmt(r['RMS_p95_LSB'])} | {fmt(r['preADC_relative_q_max'])} |")
lines+=['','原限值为0.25/1 LSB和1e-5；没有调整检查人口或尝试第三种步数。所有主生成和重放使用本机同一CPU后端、4线程、chunk64，float64坐标/积分、float32网络与末次cast、一次ADC。256步不是精确真解。','']
lines+=['## 完整响应与直接两臂比较','', '| 人口/冻结臂 | 视图 | BA | 95%分组区间 | AUC |', '|---|---|---:|---|---:|']
scores=M['C2ST']['results']
for arm in [*source,*real]:
 for view in views:
  r=scores[arm][view];lines.append(f"| {arm} | {view} | {fmt(r['balanced_accuracy'])} | {ci(r['BA_95CI'])} | {fmt(r['AUC'])} |")
lines+=['','A要求同一个主候选的SOURCE及同处理REAL五个视图95%区间均包含在[0.40,0.60]内；不是仅凭点值接近0.5。以下差值为主候选减固定噪声对照，复用已拟合预测和固定500次配对区组bootstrap；没有追加分类器。',
        'baseline_local采用100点基线逐事件去均值、除RMS后的z及ACF/归一化谱，检验标准化基线局部形态与相对噪声特征，没有附加log(std)。该视图通过不能独立证明绝对基线或噪声幅度匹配；完整电压视图与B中的绝对baseline RMS距离另行保留。','',
        '| 人口 | 视图 | ΔBA | 95%配对区间 |','|---|---|---:|---|']
for population,key in [('SOURCE','paired_primary_minus_control_C2ST'),('REAL','paired_completion_minus_fixed_same_REAL_C2ST')]:
 for view in views:
  r=M[key][view];lines.append(f"| {population} | {view} | {fmt(r['BA_delta_first_minus_second'])} | {ci(r['BA_delta_95CI'])} |")
paired_intervals=[M[key][view]['BA_delta_95CI']for key in ['paired_primary_minus_control_C2ST','paired_completion_minus_fixed_same_REAL_C2ST']for view in views]
if all(interval[0]<=0<=interval[1]for interval in paired_intervals):
 lines+=['','SOURCE与同处理REAL原五个视图的全部两臂配对区间都包含零。本轮固定分类器结果尚不能确认噪声扣除规则相对固定噪声带来改善；这些区间仍允许两臂差异，不能据此认定等效。']
lines+=['','同处理REAL同时报告相对RAW的绝对改变和相对旧BASE_REAL的预登记差；不预设随机响应对已含噪REAL恒等。RAW二维参照缺失继续unknown，不补拟合。分组区间条件于当前已拟合评价器和采集，不包含重新训练或独立采集不确定性。','']
lines+=['## PSD、噪声、源保真和门槛','', '| 臂 | 总体PSD W1 | 固定Q1–7等权W1 | 低/高峰位MAE(Q2–7) | 原六门 |', '|---|---:|---:|---|---|']
width_details=[]
for arm in source:
 b=A[arm]['B'];gr=G['models'][arm]
 gates=[gr[k]['pass']for k in ['gate1_complete_response','gate2_PSD_width_centers_noise','gate3_source_preservation','gate4_boundary_source','gate5_same_treated_REAL_control','gate6_numerical_replay']]
 lines.append(f"| {arm} | {fmt(b['overall_PSD_W1'])} | {fmt(b['common_Q1_7_PSD_W1'])} | "+'/'.join(fmt(b['peak_MAE_required_Q2_7'][str(k)])for k in [0,1])+' | '+str(gates)+' |')
 width_details.append(f"{arm} raw-width相对误差（固定Q2–7）："+'；'.join(f"component{k}: {fmt(b['raw_width_relative_errors'][str(k)]['mean_absolute_relative_rawSD_error'])}, failed bins {b['raw_width_relative_errors'][str(k)]['failed_required_bins']}"for k in [0,1]))
for detail in width_details:lines+=['',detail,'']
lines+=['固定必需区失败或缺失不删后平均；undefined不是0，也不能从密度拟合失败断言真实粒子峰消失。低/高拟合成分不是粒子真值。','']
for arm in source:
 g=G['models'][arm];g2=g['gate2_PSD_width_centers_noise'];g3=g['gate3_source_preservation'];g4=g['gate4_boundary_source'];g5=g['gate5_same_treated_REAL_control']
 lines +=[f'### {arm}','']
 for key in ['baseline_RMS_V_overall_W1','ACF1_overall_W1']:
  r=g2['measures'][key];lines.append(f"- {key}: {fmt(r['candidate'])}，BASE {fmt(r['BASE'])}，相对变化 {fmt(r['relative_change'])}；pass={r['pass']}。")
 for r in g3['whole_population_source_ranks']:
  lines.append(f"- {r['measure']}: {fmt(r['candidate'])}，BASE {fmt(r['BASE'])}；pass={r['pass']}。")
 for r in g3['actual_source_pure_had_valleys']:
  lines.append(f"- 实际未量化source-{r['actual_source_bin']} pure-had N={r['N']}，谷比例 {fmt(r['candidate'])}，BASE {fmt(r['BASE'])}，差 {fmt(r['delta'])}；pass={r['pass']}。")
 lines.append('- 失败时间边界：'+str([r['location']for r in g4['boundaries']if not r['pass']])+'。')
 lines.append('- 同处理REAL失败门视图：'+str([v for v,r in g5['three_views'].items()if not r['pass']])+'。')
 lines+=['']
lines+=['原2000 SOURCE父ID、组成、顺序和权重保留，417支持外行也走同一新流；只有条件索引夹界，没有波形/电荷裁剪或BASE回退。实际输入source-Q与历史ADC参考source-Q分列。pure-had谷比例是冻结源分组代理，不是粒子误分类率。源置换敏感性不证明物理逐事件映射正确。功能分量含末次float32cast且保留交叉项，不能称独立物理电子噪声。','',
        '## 执行、失败与交付','', '| 入口 | 实际worker PID | native exit | 秒 | RSS峰值/字节 |','|---|---:|---:|---:|---:|']
for r in H:lines.append(f"| {r['entry']} | {r['actual_worker_PID']} | {r['actual_worker_native_exit']} | {r['elapsed_s']:.6f} | {r['actual_worker_RSS_peak_bytes']} |")
lines+=['',f"本机科学原生作业累计 {sum(r['elapsed_s']for r in H):.6f} 秒；新训练0/新校准0。R48训练各50000更新仅作历史输入，不再次计入本轮耗时。",'科学/墙钟/deadline/阶段体积配额为null，本机安全RSS4GiB、一次一个本机科学作业和CPU4保持；CPU没有GPU准入门。每作业按历史峰值和系统余量检查可用内存，RSS缺测原因不明确时标unknown。',
       '远端旧尝试已完成2107.3375221949973科学秒；其原重放worker218727最后确认下界120.15029214299284秒，10:22:45 UTC后传输中断、终态unknown。用户明确授权维护期间换本机；停止重连且未取消或重复提交该远端作业，不宣称已确认全局只有一个物理作业。本地重新生成和评价的费用与历史分开。',
       '准备环境探测因psutil缺失和编码名utf8-sig出现两次控制失败；执行期间一次小JSON读取命令因PowerShell管道语法失败，另一次读取尚未出现的管线进度文件返回native1。五份控制事件均保留：另一次文案补丁因上下文匹配失败而零写入，随后正确应用，以既有Windows ctypes监督、正确编码和读取语法处理，没有科学推理、训练或拟合。若存在科学原生失败，其日志和原件列在NATIVE_CHAIN及RESOURCE_RECEIPT。',
       '本次DEV已在历史多轮暴露，只属于同采集开发比较，不称新盲测或独立物理校准。完整私有数组、预测、身份、资源/native回执在本目录；发布版METRICS_COMPACT只保留聚合。历史封存阶段保持不变；R49旧远端尝试仍未完成，当前本机阶段在终端回执后封存。Git/D由根代理完成，正常复制不宣称迁移一致性验证。',
       '早期SOURCE_BINDINGS与LOCAL_SOURCE_FREEZE保留准备时指纹。之后三份文件有工程或报告修订：evaluate_saved.py修正纯日志标签并保存前像；render_figures.py移除重复TRAIN绘制调用；make_report.py补充本机来源、指标解释及配对区间说明。它们不改变科学公式、参数、索引或已执行结果。最终文件字节以FINAL_MANIFEST为准，不声称这三项仍等于早期指纹。',
       '边界图初版的图例遮住标题，原PNG和脚本保留在display_initial；独立render_boundary_layout.py仅重绘这一张已存统计图，实际PID/RSS守护和费用见BOUNDARY_LAYOUT_CORRECTION_001与原生回执，不重复生成或拟合。修复后的最终PNG已再次实际打开。',
       '', '图件见results/figures，全部最终图实际打开检查的逐图SHA见FIGURE_QA.json。FINAL_MANIFEST显式排除两个自引用终端文件；最终native退出由实际工具回执绑定。','']
(R/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'))
reproduction='''# R49 本机恢复复现\n\n本轮复用R48两个50000更新末点，新增训练/校准为0。实际本机数值、生成和独立重放见TRAIN_SOLVER_CHECK、GENERATION_RECEIPT、REPLAY_RECEIPT与原生链；科学通过情况见GATE_METRICS及ACCEPTANCE_A_C。远端旧尝试的重放终态仍unknown，未恢复或冒充为本地结果。\n\n实际本机根：'''+str(R)+'''\nPython：'''+C['CPU_python']+'''\n\n在新的独立E/D目录恢复本轮源码及config，把既有R48权重/核、R43坐标常数、R46检查包和冻结评价输入以及旧reference目录映射到实际只读位置。所需逐项路径在config_cpu.json和LOCAL_INPUT_INVENTORY.json；历史stage/模型不重复复制，原白噪声/ID/组成必须完整保留。Windows Python3.13/Torch2.10 CPU/NumPy2.2与原服务器版本不同，不宣称跨平台逐位一致。\n\n先用python -B supervise_local.py check_models.py；只有两臂原128条TRAIN均通过才能启动python -B run_local_pipeline.py，依次运行generate_all.py、evaluate_saved.py、replay_all.py、summarize_and_plot.py、render_train.py。每入口实际PID/nonce在科学导入前与监督器握手。已有.used/active标记须先辨认真实终态，不在未知状态下重提；在新目录复现不改封存目录。\n\n候选Heun128，参考Heun256只用于TRAIN；CPU4、chunk64、同一本机确定性后端。独立重放对8000主行pre/postADC及身份/辅助字段逐值比较。参数零更新；没有加载Adam或重新生成训练随机流。样本/40分类器/32密度槽是固定设计，失败必需区不可删除。\n\n早期SOURCE_BINDINGS与LOCAL_SOURCE_FREEZE为准备时指纹。evaluate_saved.py纯日志标签修正、render_figures.py移除重复TRAIN绘制、make_report.py本机来源和指标解释补充属于已记录的工程/报告修订；科学定义及已执行结果不变，最终字节以FINAL_MANIFEST为准。\n\n时间/体积预算字段保持null；机器4GiB安全内存、一次一个本机科学作业和真实磁盘容量依然检查，CPU不受GPU忙的准入条件限制。源码、日志、缓存、临时与结果仅E/D。科学replay是实际算法检查，不是迁移一致性测试。报告、图件实际QA和最终清单必须在最终封存前完成；封存后停止写入。\n'''
actual_outcome=f"本次两臂128/256数值检查均通过；8000条独立主输出重放{'通过'if RP['pass']else'失败'}。指定主候选A/B/C为"+'/'.join(str(A[source[1]][k]['pass'])for k in ['A','B','C'])+'，项目联合目标尚未完成。'
reproduction=reproduction.replace('本轮复用R48两个50000更新末点，新增训练/校准为0。',actual_outcome+'\n\n本轮复用R48两个50000更新末点，新增训练/校准为0。',1)
reproduction+='\n本次边界图还执行了python -B supervise_boundary_layout.py render_boundary_layout.py作单张布局修复；原图/脚本在display_initial，实际费用进入RUN_HISTORY。它只读已保存九个统计量，不重新生成、拟合或评价。\n'
(R/'REPRODUCTION.md').write_bytes(reproduction.encode())
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'report_written':True,'new_science':0}))
