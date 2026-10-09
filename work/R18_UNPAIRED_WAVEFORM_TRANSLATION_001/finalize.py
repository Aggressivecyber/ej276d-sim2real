from __future__ import annotations
import pathlib,json,hashlib,os,datetime
ROOT=pathlib.Path(__file__).resolve().parent
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8'))
def write(n,o):(ROOT/n).write_text(json.dumps(o,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
m=read('metrics.json');cfg=read('config.json');native=read('REMOTE_NATIVE_RECEIPT.json');replay=read('REPLAY_RECEIPT.json');t=m['training']
figs=['charge_psd_source_composition','conditional_psd_CDF','conditional_psd_by_composition','energy_peak_width','source_controls','repeated_noise','conditional_waveforms','training_and_C2ST']
write('FIGURE_QA.json',{'UTC':utc(),'method':'all eight rendered PNGs actually opened with view_image and visually examined; corrected source_controls opened again','figures':[{'path':'results/figures/'+n+'.png','bytes':(ROOT/'results/figures'/(n+'.png')).stat().st_size,'result':'readable axes/legends/support gaps; no obscured panels or clipped titles'}for n in figs],'specific_notes':{'charge_psd':'shared full observed Q coordinates, source composition colors, not particle truth','conditional_by_composition':'24 small panels, source-charge conditioning explicitly stated','widths':'failures retain gaps; fitted components not particle labels','source_controls':'corrected fixed-source noise RMS uses broadcast main output of that source; source roll uses actualroll source and unchanged reference','waveforms':'median proximity accompanied by explicit visible periodic artifacts; does not demonstrate joint distribution match'},'fits_or_generation_for_QA':0})
failures=[]
for a in ['attempt_01','attempt_02']:
    r=read(a+'/REMOTE_NATIVE_RECEIPT.json');f=list((ROOT/a).glob('FAILURE_*.json'));details=json.loads(f[0].read_text())if f else{}
    failures.append({'attempt':a,'native':r,'failure':details})
science_main=sum(x['native']['numeric_cumulative_s']for x in failures)+native['numeric_cumulative_s']
correction=read('SOURCE_CONTROL_CORRECTION_RECEIPT.json');correction['native_tool']='3eb10a';correction['native_exit']=0;correction['native_tool_wall_upper_s']=7.82785;correction['local_RSS_peak_bytes']='unknown; not sampled for short already-terminated postprocessing process'
science=science_main+correction['native_tool_wall_upper_s']
resource={'UTC':utc(),'first_native_PID':14124,'first_native_UTC':'2026-10-09T07:12:21.779219+00:00','registered_UTC':'2026-10-09T07:11:26.541622+00:00','executor_deadline_UTC':cfg['deadline_UTC'],'stage_absolute_deadline_UTC':cfg['absolute_deadline_UTC'],'main_science_tool':'6038de -> d277d5, native0','main_native':native,'local_native':read('LOCAL_NATIVE_RECEIPT.json'),'ordinary_transfer':read('RETURN_TRANSFER_RECEIPT.json'),'failures':failures,'preparation_failure':read('PREPARATION_FAILURES.json'),'science_cumulative_s':science,'science_limit_s':1800,'main_training_s':t['training_s'],'training_plus_all_failed_process_wall_upper_bound_s':t['training_s']+sum(x['native']['numeric_cumulative_s']for x in failures),'training_limit_s':1200,'G_updates_cumulative':3999,'D_updates_cumulative_including_failed_step':4000,'one_recipe_one_scientific_arm':True,'training_invocations_with_nonzero_updates':2,'failed_first_D_step_not_saved':'unknown; known failedG0/D1, original receipts preserved','main_GPU_allocated_peak_bytes':t['GPU_allocated_peak_bytes'],'main_GPU_reserved_peak_bytes':t['GPU_reserved_peak_bytes'],'failed_process_GPU_allocated_peak_bytes':'unknown; no completed worker environment receipt','RSS_peak_observed_bytes':max(r['RSS_peak_bytes']for x in failures+[{'native':native}]for r in x['native']['history']),'RSS_probe_failures':sum(r['RSS_probe_failures']for x in failures+[{'native':native}]for r in x['native']['history']),'threads':4,'HGB_fits':m['counters']['HGB_attempts'],'width_EM_attempts':m['counters']['density_EM_attempts'],'width_description_slots':m['counters']['density_description_slots'],'main_new_rows':2000,'probe_new_rows':832,'repeated_noise_new_rows':1024,'fresh_replay_rows':2000,'EVAL_read':0,'CAL_waveform_numeric_decode':0,'new_G4':0,'raw_long_waveform_decode':0,'migration_equality_tests':0,'new_dependency_install':0,'new_agents':0,'Git_operations_by_executor':0,'source_root':'R10_REMOTE inputs and R14 existing expanded REAL','local_data_write_drive':'E','versions':m['versions'],'numeric_replay':replay,'GPU_limit_bytes':4294967296,'RSS_limit_bytes':4294967296,'output_limit_bytes':268435456,'scientific_negative_result_not_engineering_failure':True}
resource['science_training_evaluation_replay_and_failures_s']=science_main;resource['saved_output_source_control_correction']=correction;resource['science_cumulative_s_includes_conservative_correction_tool_wall']=True
write('RESOURCE_RECEIPT.json',resource)
compact={k:m[k]for k in ['status','data_roles','domains','domain_vs_REAL','local','PSD_main','energy_conditional_width','source_association','source_probes','C2ST','generation','counters','versions','EVAL_read','no_particle_truth_from_real_density']}
compact['training']={k:v for k,v in t.items()if k!='validation_trace'}
compact['same_parent_noise']={k:v for k,v in m['same_parent_noise'].items()if k!='rows'}
compact['resource_summary']={k:resource[k]for k in ['science_cumulative_s','main_training_s','G_updates_cumulative','D_updates_cumulative_including_failed_step','main_GPU_allocated_peak_bytes','RSS_peak_observed_bytes','HGB_fits','width_EM_attempts','EVAL_read']}
compact['replay']=replay;compact['no_per_event_ids_indices_prediction_or_parent_lists_in_compact']=True
write('METRICS_COMPACT.json',compact)
report='''# R18：完整G4派生波形到实测风格的无配对转换

本轮恢复了项目主线：每条输出由对应完整480点G4派生clean波形与随机输入产生，并保留源父ID。实现与重放通过，PSD峰位距离改善，但完整实测响应分布仍失败。它不是R11–R17的实测目标密度独立抽样，也不能称为已验证的物理响应映射。

输入为既有G4 TRAIN8192（6593 fit/1599内部val，固定使用512 val）、R14 REAL21047 fit/599 val，以及原2000 DEV父事件/2000 REAL DEV。DEV沉积组成是pureEE417、purehad1518、mixed65；既有TRAIN显式PDG为22:1647、2112:4946。DEV推定PDG22:317、2112:1683，推定依据是旧2112/non2112标签规则与已绑定R2 CSV只有22/2112的证据，区别于逐事件显式存储。incident neutron含纯电子沉积并不矛盾，不能把purehad与所有neutron等同。没有重新读CSV/原始长波形，没有EVAL/CAL波形数值或新G4。

源波形来自总EE/hadron沉积的已用解析响应，完整输入增加了结构化波形描述，没有恢复缺失的微观沉积时间。对比对象SOURCE_CLEAN含ADC但没有历史SIM噪声；它与旧noisy SIM不是同一输入分布。REAL没有独立粒子真值标签，拟合两成分仅为PSD密度描述。

首次更新前冻结1D残差U-Net16/32/64、完整波形双尺度输入、8维latent与480白噪声的源条件随机分支。LSGAN、三层内容NCE、REAL identity、源相对电荷与排序、同一固定窄电荷区内PSD排序、联合物理/池化形状SW共同训练。源与REAL按共同Qbin独立抽样，没有随机sim-real配对MSE，没有PDG/ID输入。PSD排序只约束差值>0.005的同Qbin源对，允许共同/能量相关偏移。方法受[CUT](https://arxiv.org/abs/2007.15651)启发，不是原论文复现；无配对域匹配本身不能识别唯一的响应对应。

训练实际3999 G/3999 D更新，114.767033秒；另一次工程失败完成D1/G0，累计D4000。内部val最低J选step3500：step0=1.646692，选中=0.291510，末点=0.323432，不能概括为始终单调稳定。J的域距离按共同SOURCE/REAL电荷区等权，不用总混合先验或DEV选点。所有8区内部val有支持，但最高区只有4个source val父事件，选点在该区的稳定性受限。初始/所选/末点权重、优化器和随机状态均保存。

2000/2000输出与源clean均QC有效，ADC裁剪0。整体PSD W1由0.103300降为0.036083；八固定观察电荷区等权W1由0.106024降为0.041020，8区均有CDF支持。这个改善不等于双带成功：bin3/4的谷区占比相对REAL分别多0.682230/0.467252；最高区仍W1=0.045556、KS=0.682370。完整谷区/CDF、支持与总体占比在metrics保留。

两密度成分合格区：REAL1–8，SOURCE_CLEAN1–4，TRANSLATED1–7，最高转换区失败未替换。三者共同1–4区的低成分峰位MAE由0.127868降0.015522，高成分由0.094480降0.035376；低成分原宽度MAE却由0.003615增至0.014681，高成分宽度MAE仍0.043739。原宽度与去趋势宽度均保留。转换后的峰明显过窄，不能拿距离下降替代宽度/占比检验。

固定600/250隔离池C2ST：SOURCE_CLEAN完整shape/V均1.000，转换均0.996，REAL–REAL为0.512/0.546。转换Q/PSD二维BA0.834，相对源0.966的配对差-0.132，95%区组CI[-0.167050,-0.095706]；REAL–REAL0.496。波形prompt/postpeak/tail/baseline局部BA分别0.988/0.996/0.992/0.986。训练分数和留出分数及实际2列输入在本地保存，没有以近0.5证明相等。REAL–REAL是同采集人工分组的TRAIN–DEV采样参照，CI条件于已拟合判别器，不是响应训练不确定性或物理等效阈值。

源电荷到输出电荷Spearman0.999681；全局源PSD到输出PSD0.725077，纯EE/纯had/mixed分别0.691268/0.646258/0.331862。电荷基本保留并不代表正确响应；观察电荷分布W1仍1772.003 mV ns（源1851.116）。分Q的排序/迁移保留，包括bin2总体PSD相关仅0.009527：原解析同类内部PSD差很小，不在>0.005排序约束之内，也不能仅靠一个相关系数推断粒子桶反转。组成散点仍可辨别电子/强子源带，但它们移到错误位置并过窄。

128同父×8噪声（纯EE42/纯had44/mixed42）没有任何父事件跨冻结谷区两侧；各组平均同父PSD SD为0.001858/0.002194/0.002003。这支持本轮未靠同一纯源随机制造远离源关联的双峰，但只有这组父事件/噪声的有限证据，不证明所有输入的类别保真或真值正确。

832源探针中，同输入同噪声128条与主输出逐位一致；固定噪声滚动交换源无固定点，输出RMS平均改变15.222 mV。更换幅度和同电荷解析快/慢形状会改变输出。固定一个源而改变128组噪声，PSD SD0.000939、Q SD9.970 mV ns，随机宽度很窄。原fixed-source RMS错误地比较128个不同主父输出，将源替换混入噪声统计；原图/原统计保留于PROBE_REPAIR_001。本地只从已存载荷修复参考：广播该固定源的首个主输出，正确RMS均值1.253068 mV、首行0。新SOURCE_PROBES_CORRECTED.npz分别记录实际源父、噪声来源和对照父；source-roll真实源父roll(pick)，噪声/参考仍pick。没有新生成/拟合，主2000及重放值未改。快慢插值不是真正Geant4反事实事件；非零敏感性不能证明映射物理正确。

基线RMS边际W1改善至0.019437 mV，平均0.636436 mV（REAL0.646423）；但ACF1均值-0.238093对REAL-0.011884，局部基线仍易区分。上升时间W1从1.912781降1.544696 ns，转换平均7.861007 ns对REAL6.330870。已实际打开全部8图，条件中位波形可见重复纹理/波动，平均峰接近不足以证明联合形状或涨落匹配。小白噪声/5 mV随机分支、池化域损失和当前判别结构可能限制真实条件宽度与细时间纹理；这是待检验的模型解释，不把误差归因于G4物理错误。

两次CUDA确定性工程错误在G更新前：median索引接口、linear1d反向。分别保留原退出、3.805838/4.407198秒及源/初态；修复为同定义CPU median与邻切片0.75/0.25半像素线性上采样，没有切换配方/种子或利用DEV调参。失败D1的更新权重未保存，标unknown，不补造。

最终主worker197234 native0，重放197894 native0；保存模型+实际源/噪声在新进程重放2000条，预/后ADC逐值maxabs0，父ID一致，无TRAIN/target依赖。主科学含两失败/重放138.796297秒，加已保存输出修复原生工具墙钟上界7.827850秒，合计146.624147秒。远端峰RSS2,888,953,856 B，主GPU allocated265,882,112 B，4线程；失败GPU峰、短本地修复进程RSS缺采样为unknown，不反推。21 HGB、16新EM、2000主生成+832探针+1024重复+2000重放。迁移仅正常传输exit0，没有一致性测试或重复输入哈希。

本轮结论是：完整源条件输入与保存/独立推理路线成立，PSD位置改善并保留了部分源关联；条件峰宽、谷占比、全波形联合结构及噪声相关仍失败。下一科学问题应继续在源条件转换内隔离随机响应自由度和细时间域匹配的不足，而非回到目标域无条件自生成。没有追加训练或自行启动下一阶段。模型/源/噪声/逐事件完整结果留本地，METRICS_COMPACT只含聚合内容。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
repro=f'''# Reproduction

Actual server: rylan@100.82.231.102, `{cfg['remote_root']}`; Python `{cfg['python']}`. Entry `remote_supervise.py` launches `numeric_worker.py` and then one `replay.py`. Existing environment versions are in metrics.json; four CPU threads, deterministic Torch, TF32 off, CUDA training/CPUfloat32 final inference. Training uses R10 inputs/G4_TRAIN_cache and REAL prefixes plus R14 TRAIN_ADDITION.npz. R3B charge definitions, old REAL width tables from R17, and fixed R7 classifier pools are read-only. No EVAL input.

Local complete artifacts: `{ROOT.as_posix()}`. Selected weights `TRANSLATOR.pt`; initial/final G/D weights, optimizer and initial/final random states separate. Actual inference inputs `ACTUAL_GENERATION_INPUTS.npz` contain full source480, z8, white480, source parent IDs. `generated.npz` contains source IDs, pre/postADC, deterministic mean and random branch. These inputs and selected model alone generate a new response; `replay.py` additionally reads generated.npz only as its equality reference.

Replay actually executed in a fresh server process PID197894: pre/postADC bitwise true, maxabs0 on2000 rows/960000 points. Do not rerun in a sealed directory. To infer in a new independent directory, copy model.py, selected model and actual generation inputs, set E/D local temp/cache or a new server leaf, load Translator state with weights_only=True, and call infer with chunk128, then adc. No TRAIN/REAL target is needed. For a new from-scratch reproduction, use saved config/seeds and the same existing input roles/environment; a different hardware/library version is not claimed bitwise identical.

Executed source-control code had a reference-index bookkeeping error. After a new science run, apply correct_source_controls.py once in its new directory to derive corrected probe metadata/statistics from saved outputs only; it does not generate or fit. In this delivered run it already executed, original statistics/plot are preserved in PROBE_REPAIR_001 and the authoritative source/noise/reference metadata are in results/SOURCE_PROBES_CORRECTED.npz. The main2000 outputs and all primary PSD/C2ST values were unchanged.

Attempts01/02 retain deterministic CUDA failures and their costs. Attempt02's failed D1 weight was not saved (unknown), G0; successful retry is from the fixed initial seed, not a resumed partial critic checkpoint. Main cap3999 plus failedD1 respected the4000 budget. No migration equality tests were performed. This is unpaired detector-style translation with observed source association checks, not a uniquely calibrated per-event physical mapping.
'''
(ROOT/'REPRODUCTION.md').write_text(repro,encoding='utf-8')
write('DELIVERY_BUILD_RECEIPT.json',{'PID':os.getpid(),'UTC':utc(),'pure_control_summary_only':True,'new_fits_or_generation':0,'eight_actual_view_image_QA':True,'science_terminal':True,'deadline_UTC':cfg['deadline_UTC']})
print(json.dumps({'PID':os.getpid(),'UTC':utc(),'REPORT_ready':True,'science_cumulative_s':science,'remaining_work':'single byte manifest/seal only'}),flush=True)
