from __future__ import annotations
import pathlib,json,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text())
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
m=read('metrics.json');cfg=read('config.json');native=read('REMOTE_NATIVE_RECEIPT.json');replay=read('REPLAY_RECEIPT.json');roles=read('DATA_ROLES.json')
compact=json.loads(json.dumps(m))
for row in compact['representations'].values():row.get('selfcheck',{}).pop('fixed_validation_positions',None)
def summarize(x):
    if isinstance(x,dict):
        if set(['defined_n','undefined_n','quantiles']).issubset(x):return {k:x[k] for k in ['n','defined_n','undefined_n','mean','sample_sd','quantiles']}
        return {k:summarize(v) for k,v in x.items() if k not in ['histogram_counts','histogram_edges_PSD','initial']}
    if isinstance(x,list):return [summarize(v) for v in x]
    return x
compact=summarize(compact)
compact['publication_scope']={'aggregate_only':True,'all_draw_IDs_source_IDs_roles_C2ST_predictions_NN_indices_models_actual_noise_arrays_kept_local':True}
write('METRICS_COMPACT.json',compact)
resource={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),'registered_UTC':cfg['registered_UTC'],'first_native_PID':cfg['first_native_PID'],'first_native_UTC':cfg['first_native_UTC'],'executor_stop_deadline_UTC':cfg['deadline_UTC'],'numeric_cumulative_s':native['numeric_cumulative_s'],'numeric_limit_s':1800,'fit_phase_s':m['fit_phase_s'],'fit_soft_limit_s':1200,'main':native['history'][0],'replay':native['history'][1],'RSS_peak_bytes':max(r['RSS_peak_bytes'] for r in native['history']),'RSS_probe_failures':sum(r['RSS_probe_failures'] for r in native['history']),'RSS_limit_bytes':4294967296,'CPU_threads':4,'GPU_used':False,'counters':m['counters'],'other_fixed_estimation_calls':{'full_PCA':2,'tail_ridge':2,'tail_LedoitWolf':2,'TRAIN_head_scaling':2,'validation_selection':2},'scientific_failures':0,'new_generation_not_G4':4000,'fresh_model_replay':4000,'model_fit_replay':0,'new_G4':0,'response_training':0,'EVAL_read':0,'CAL_waveform_numerical_decode':0,'source_migration_hash_tests':0,'migration_normal_transfer_exit':0,'output_cap_bytes':536870912,'source_READINESS':read('READINESS.json'),'failures_or_warnings':[{'type':'environment_warning','where':'numeric_worker.py.stderr.log','warning':'Matplotlib Axes3D unavailable; no 3D plots requested or generated','science_retried':False}],'python_dependency_IO_not_project_data_accounting':True}
write('RESOURCE_RECEIPT.json',resource)
if (ROOT/'LAYOUT_RECEIPT.json').exists():
    layout=read('LAYOUT_RECEIPT.json');resource['layout_only']=layout;resource['layout_command_wall_upper_s']=3.1122385;resource['numeric_cumulative_measured_including_layout_s']=native['numeric_cumulative_s']+layout['elapsed_s'];resource['numeric_budget_charged_with_layout_command_upper_s']=native['numeric_cumulative_s']+3.1122385;resource['layout_metric_refits']=0;write('RESOURCE_RECEIPT.json',resource)
names=cfg['models'];fmt=lambda x:'undefined' if x is None else f'{x:.6g}'
lines=['# R11：可逆目标密度模型仍未达到完整响应分布相似','',
'本轮连续执行已完成，两个固定模型、6 个单次 head GMM、18 个 HGB 评价和独立保存后重放全部正常退出。可逆表示没有损失完整波形信息；独立生成的电荷/PSD及局部前沿明显比旧目标只固定源模型更合理，但完整波形仍高度可分，不能作为项目完成或 G4→真实响应已解决的证据。','',
'## 固定数据与表示','',
'只使用同一次真实采集：REAL TRAIN 的 3200 条中 3191 条 QC 有效，固定人工组划分 TRAIN-fit2592 / 内部 val599。PCA、尺度、ridge、LedoitWolf 和 GMM 都只在 fit 数据估计，K 只按完整599条 val 连续 NLL 选择。DEV2000只报告。CAL 波形解码0，EVAL读取0；已冻结 CAL 电荷边界和高置信参考、8个描述区间原样继承。人工50条组和新随机draw不是新的独立采集。','',
'RAW 使用完整480维原V PCA，head32 / tail448。FACTOR 使用 baseline b、positive signed charge q 和478维受约束 shape坐标；head34=[logq,b,shape32]，tail446 为余下shape系数乘q、保留V单位。完整反变换 V=b·1+q(ms+Uhead·z)+Utail·r。rank32重建只作投影诊断，真正生成包含完整 Gaussian tail。','',
'每臂用一个共享的条件线性 Gaussian tail、固定ridge1e-3、LedoitWolf加1e-12V²；head GMM仅K=1/4/8、一次初始化、最多300迭代。FACTOR完整连续log-density含head尺度和33logq+log|det(1,ms,Uhead,Utail)|，RAW正交变换绝对行列式1。两者均为连续密度代理，不能直接当ADC格点概率质量。负NLL来自V坐标的连续密度单位，不等于错误或等效证明。','',
'新生成每臂2000条独立密度draw，不携带G4父事件，不使用真实事件模板/拒绝重抽。pre-ADC float64，旧14bit[-.5,.5]V round/clamp后保存float32。没有额外CFD裁窗，因为输入/表示已在对齐后的480点上建模；这与旧R8C的完整512点再裁窗处理有区别。','',
'## 重建与连续密度选型','',
'| 表示 | 所选K | 完整val NLL nat/event | rank32 val RMS mV | rank32 PSD重建MAE | 全维val最大误差V |','|---|---:|---:|---:|---:|---:|']
for n in names:
    r=m['representations'][n];rs=r['reconstruction']['TRAIN_val'];chosen=next(a for a in r['fits'] if a['components']==r['selected_components'])
    lines.append('|'+n+'|'+str(r['selected_components'])+'|'+fmt(chosen['complete_validation_NLL'])+'|'+fmt(rs[2]['RMS_mean_V']*1000)+'|'+fmt(rs[2]['PSD_abs_error_mean'])+'|'+fmt(rs[-1]['maxabs_V'])+'|')
lines+=['','6个候选均收敛；所有599验证行都得到有限完整密度。RAW K4比K8仅好约0.012nat/event，严格按已冻规则保留K4，没有因为DEV改变选择。FACTOR K8最优，但同V坐标完整NLL仍弱于RAW；FACTOR的PSD重建/生成边际更好不改变该事实。','',
'完整逆变换、Gaussian tail公式、head mixture公式及FACTOR Jacobian自检均通过。固定4个val位置只用于数学自检；完整val重建误差已单独统计。FACTOR Jacobian logdet公式差0，tail密度与独立SciPy Gaussian公式差≤1.37e-12nat。有限差分检查也保存，最大导数差约9.4e-9，不作为新增选型。','',
'## 独立生成：整体仍失败','',
'| 指标 | RAW | FACTOR | REAL–REAL采样参照 |','|---|---:|---:|---:|']
cs=m['C2ST']['results']
for view in cfg['C2ST_views']:lines.append('| C2ST '+view+' BA |'+fmt(cs[names[0]][view]['balanced_accuracy'])+'|'+fmt(cs[names[1]][view]['balanced_accuracy'])+'|'+fmt(cs['REAL_REAL'][view]['balanced_accuracy'])+'|')
for field in ['PSD','Qtotal_mVns','rise_10_90_ns','baseline_RMS_V']:
    lines.append('| '+field+' W1 |'+fmt(m['domain_vs_REAL'][names[0]]['overall'][field]['W1'])+'|'+fmt(m['domain_vs_REAL'][names[1]]['overall'][field]['W1'])+'|本轮未新计算该标量参照|')
lines+=['','C2ST严格使用原R7位置，先取两新臂有效draw位置交集，再与REAL/REAL-TRAIN-reference共同定量截取：600训练、249留出每域。RAW有1条原留出位置无效，因此未用桶外替换事件；实际全人口RAW1994/2000有效、FACTOR2000/2000，原人口统计均保留。500次区组bootstrap仅反映固定分类器条件下的抽样波动。REAL–REAL是同N的TRAIN-reference与DEV比较，不能当物理等效界限。','',
'FACTOR的PSD W1约0.00926、Q W1约113.47mVns，边际覆盖比旧R10固定源明显合理；RAW约0.01962/195.40。完整shape BA仍约0.946，完整V约0.986（RAW约0.948/0.970）。仅匹配PSD、均值曲线、rise时间边际无法证明完整响应相似。旧R3B/R10是历史设计/环境参照，不是本轮严格配对训练对照。','',
'剩余差异有清楚位置：baseline RMS均值REAL0.646mV、RAW0.940mV、FACTOR1.013mV；RMS KS约0.940/0.946。baseline_local视图按本区尺度归一化，本来就不包含该绝对RMS幅度，所以其较低BA不否定这个差异。tail局部BA约0.926/0.936。峰后曲率随电荷的中位数变化在实测约2.5→6.2mV，新模型约4.6→4.9mV，固定Gaussian残差没有恢复实测的条件涨落跨度。prompt100:150包括峰/早期下降，不能把该窗口曲率差全部称纯上升沿问题。','',
'## 条件峰宽与未知项','',
'8区新描述槽共16个，实际EM也16次。RAW仅bin7、8满足冻结的双成分资格；FACTOR0/8满足。未收敛、分离不足、有效成分数不足均保留，不能将不合格参数作为两峰真值。两新臂与REAL共同合格区间为空，因此共同峰位/原宽/去趋势宽度MAE全部undefined；没有拿不同支持区间的平均值作优劣判定。历史REAL/R3B/R10表原样复用、未重新拟合。','',
'这意味着本轮没有复现实测两峰随能区收窄的完整合格证据；不合格描述拟合本身也不能证明物理两类不存在。两成分与高置信伪标签都不是独立粒子真值。旧CAL q4参考失败仍全unknown/FoM undefined；q1高置信neutron-like仍0，没有补参考或fallback。','',
'## 最近邻与复现','',
'每种新生成与REAL DEV各2000条query，完整2592条TRAIN-fit作参考，在V和own-shape两个视图均2000可定义。无重抽/缩减；无完全相同最近邻记录。V最近邻RMS均值REAL4.100mV、RAW4.396mV、FACTOR4.389mV；shape均值约0.02643/0.03108/0.02963。少量固定hash例图可看出类似形状，但不是配对真值；无精确复制不能证明无记忆。逐query距离/邻居索引、选择和实际随机输入留本地。','',
'新进程从保存参数及component/head_white/tail_white重放全部4000条，无重新fit、无完整评价重跑。pre-ADC与主保存bitwise相同、最大差0，post-ADC array_equal通过；共1920000个点。该证据是科学模型重放，不是迁移测试。用户要求的普通传输执行正常exit0，没有源/目的内容一致性检测。','',
'## 预算与可复现交付','',
f'服务器主worker134962 native0、{native["history"][0]["elapsed_s"]:.6f}s；新进程replay135146 native0、{native["history"][1]["elapsed_s"]:.6f}s。合计{native["numeric_cumulative_s"]:.6f}s/1800s，fit阶段{m["fit_phase_s"]:.6f}s/1200s，观测RSS峰{resource["RSS_peak_bytes"]}B/4GiB，4CPU线程。无工程失败/训练重试；Matplotlib Axes3D导入警告保留，所有图为2D并已实际目检。无EVAL、新G4、响应模型训练、安装和Git操作。',
'', '实际Python3.12.3/NumPy1.26.4/sklearn1.8.0/SciPy1.11.4。各模型representation.npz、所选GMM参数、GENERATION_INPUTS及pre/post生成均保存。metrics.json完整本地；METRICS_COMPACT仅聚合、没有逐事件身份/预测/邻居/噪声载荷。REPRODUCTION.md说明原路径和新目录恢复入口。READINESS.json绑定实际执行源码，后续控制报告代码仅进入最终清单。','',
'## 下一轮方向建议','',
'全维逆变换误差仅浮点级，现有证据没有指向丢弃完整坐标；rank32单独投影仍有约2.2mV误差，但本模型已用完整tail，不能把所有失败简单归为维数不足。目标分布在该坐标系下可以表达，却没有被这套GMM+共享线性Gaussian充分学到。优先检验tail创新的条件尺度/时窗结构：当前全波形物理单位的共享LedoitWolf收缩可能向基线分配信号残差方差，而同一协方差不能表达q/shape相关的峰后涨落范围。这是基于结果的待检验假设，不是因果结论；仅缩小baseline噪声也未必修复tail联合分布。下一轮应在TRAIN/val定义结构化条件残差密度，并保留本轮两个固定参考，完整评价而不最大化FoM。只有目标密度充分接近后再讨论如何绑定G4条件；本轮不是G4输入映射且不能据此评价G4物理正确性。','',
'API依据：[PCA官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)、[GaussianMixture官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)、[LedoitWolf官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)。网页版本可能不同，实际执行环境和保存源码为准。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if (ROOT/'LAYOUT_RECEIPT.json').exists():
    with (ROOT/'REPORT.md').open('a',encoding='utf-8') as f:f.write('\n补充图 charge_psd_main_body.png 在全部五域使用共同PSD[0,0.6]和共同电荷范围，原全范围图保留。图外点REAL/R3B/R10均0、RAW41（低于0的30、高于0.6的11）、FACTOR8（全部低于0）。该图只使用已保存特征重绘，没有重新计算指标或拟合。服务器绘图PID137126，native0，自计0.960703s；按整个传输/命令3.112239s保守记账，累计数值/绘图预算约20.346526s。\n')
(ROOT/'REPRODUCTION.md').write_text('''# R11 reproduction

Executed server: `rylan@100.82.231.102`, host `rylan-GTi15`.
Directory: `/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R11_REMOTE_20261009_020915`.
Python: `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`.
Main: `python -B remote_supervise.py`, invokes CPU4 `numeric_worker.py` then a separate `replay.py`.

Read-only inputs: sibling `R10_REMOTE_20261009_012338/inputs`, keyed by config input_files. Historical R10 `metrics.json` and `generated.npz` are read-only from that sibling root. Local originals are listed in config.json; REAL metadata/waveforms from P1K, fixed roles from R2, R7 position selections, R3B density definition/generated waves. REAL V file contains6800 rows but only5200 rows are numerically viewed; CAL1600 is not decoded. No EVAL files required.

For fresh reconstruction of saved science, restore source/config and the two model folders in a **new** E/D or Linux study directory. Keep readonly inputs separate and update only path locations in a new configuration, preserving science settings. `python -B replay.py` requires saved metrics.json, each representation.npz, selected GMM_K.npz, GENERATION_INPUTS.npz and generated.npz. It reconstructs all4000 draws and compares pre/post outputs without fitting or redrawing. The executed REPLAY_RECEIPT.json records PID135146, pass, pre/post bitwise and maxabs0. Restore all files before running; old scientific originals must not be overwritten.

For a from-scratch reproduction, create a new directory and run the saved numeric_worker/remote_supervise with readonly input path mapping. Six GMM seeds, initialization, solver, coordinate maps, ADC and chunk sizes are explicit. Current server Python3.12.3, NumPy1.26.4, sklearn1.8.0, SciPy1.11.4, CPU4. No environment install/upgrade was used. Floating point behavior across different environments is not claimed bitwise identical. The old R3B/R10 models and CPU/GPU histories are historical comparisons, not newly paired fits.

All saved random inputs are actual component/head_white/tail_white arrays. New draws have no G4 parent event correspondence. Normal source upload/result transfer exited0; no migration equality test or new original-input hash audit was executed. Full local metrics, model parameters, noise and identity selections stay E/D; Git publication uses aggregate summary only.
''',encoding='utf-8')
qa={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Actual view_image inspection after normal return,12 original rendered PNGs plus shared main-body supplemental view','figures':[{'file':p.name,'visually_checked':True,'status':'readable','note':('shared full range retains RAW extreme PSD outliers; shared0-0.6 main-body supplementary figure additionally inspected' if p.name=='charge_psd.png' else 'labels/axes/legend and failure gaps visible')} for p in sorted((ROOT/'results/figures').glob('*.png'))],'metrics_refit_for_QA':0,'front_window_note':'95:125 saved index window may include peak; not eventwise pure-prepeak claim','unavailable_widths_shown_not_filled':True}
write('FIGURE_QA.json',qa)
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report_bytes':(ROOT/'REPORT.md').stat().st_size,'compact_bytes':(ROOT/'METRICS_COMPACT.json').stat().st_size,'figures_checked':len(qa['figures'])},ensure_ascii=False))
