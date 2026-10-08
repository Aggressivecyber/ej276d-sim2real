from __future__ import annotations
import pathlib,json,hashlib,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8'))
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def binding(p):
    h=hashlib.sha256();n=0
    with p.open('rb') as f:
        while b:=f.read(2*1024*1024):h.update(b);n+=len(b)
    return {'path':str(p.relative_to(ROOT)).replace('\\','/'),'bytes':n,'sha256':h.hexdigest()}
start=read('START_RECEIPT.json');cfg=read('config.json');m=read('metrics.json');h=read('RUN_HISTORY.json');num=read('NUMERIC_RECEIPT.json');eng=read('ENGINEERING_CHECKS.json')
assert m['status']=='complete' and h[-1]['worker_native_exit']==0
assert m['counters']['HGB_attempts']==52 and m['counters']['fixed_inference_rows']==4000 and m['counters']['response_updates']==m['counters']['EVAL_read']==m['counters']['SSH']==0
ci=m['C2ST'];views=[x['view'] for x in cfg['matrix'] if x['operator']=='native' and x['contrast']=='GEN_vs_REAL']
contrasts=['GEN_vs_REAL','REAL_TRAIN_vs_DEV','positive_vs_REAL']
def result(op,v,c):
    r=ci[op+'|'+v+'|'+c];z=r['heldout_CI']['balanced_accuracy_95CI'];return f"{r['balanced_accuracy']:.3f} [{z[0]:.3f},{z[1]:.3f}]"
rows=['|原始视图|GEN–REAL BA及簇95%区间|REAL TRAIN–DEV|尾部+噪声阳性|','|---|---:|---:|---:|']
for v in views:rows.append('|'+v+'|'+'|'.join(result('native',v,c) for c in contrasts)+'|')
oprows=['|共同操作|GEN–REAL 完整shape|REAL TRAIN–DEV|同操作阳性|','|---|---:|---:|---:|']
for op in ['native','fractional_align','coarse_ADC4','smooth_sigma1','align_then_smooth']:oprows.append('|'+op+'|'+'|'.join(result(op,'full_historical_shape',c) for c in contrasts)+'|')
ivrows=['|固定模型干预|完整电压BA|完整历史shape BA|CFD窗起点改变记录|填充/削顶|','|---|---:|---:|---:|---|']
for op,d in m['interventions'].items():ivrows.append(f"|{op}|{result(op,'full_voltage','GEN_vs_REAL')}|{result(op,'full_historical_shape','GEN_vs_REAL')}|{d['window_left_difference_rows']}|{d['padding_rows']}/{d['clipped_rows']}|")
curve=['|自身Qt区域|同N每域|prompt中位曲率GEN/REAL (mV)|postpeak中位曲率GEN/REAL (mV)|postpeak W1 GEN/同N RR (mV)|','|---|---:|---:|---:|---:|']
for r,x in m['same_N_conditional_reference'].items():
    a=x['comparison']['GEN'];rr=x['comparison']['TRAIN_REFERENCE'];p=a['prompt_curvature_RMS_V'];s=a['postpeak_curvature_RMS_V']
    curve.append(f"|{r}|{x['N_per_domain']}|{1000*p['candidate']['quantiles'][3]:.3f}/{1000*p['REAL']['quantiles'][3]:.3f}|{1000*s['candidate']['quantiles'][3]:.3f}/{1000*s['REAL']['quantiles'][3]:.3f}|{1000*s['distance']['W1']:.4f}/{1000*rr['postpeak_curvature_RMS_V']['distance']['W1']:.4f}|")
widthfail={op:{n:{'qualified_bins_1based':[x['bin']+1 for x in d['slices'] if x['valid']],'failures_1based':{str(x['bin']+1):x['failure_reasons'] for x in d['slices'] if not x['valid']}} for n,d in dd.items()} for op,dd in m['PSD_density_widths'].items() if dd}
operator_values=[]
for op in ['native','fractional_align','coarse_ADC4','smooth_sigma1','align_then_smooth']:
    for n,d in m['after_operator_statistics'][op].items():
        f=d['features'];operator_values.append({'operator':op,'domain':n,'QC_n':d['QC_after_operator_n'],'median_PSD':f['PSD']['quantiles'][3],'median_rise_ns':f['rise_10_90_ns']['quantiles'][3],'median_Qtotal_mVns':f['Qtotal_mVns']['quantiles'][3],'median_peak_V':f['peak_V']['quantiles'][3]})
recommendation={
 'priority_recipe':'same R3B conditional mean backbone + unsmoothed local joint / first-and-second-difference distribution constraints + low-dimensional signal-envelope gated short-correlated zero-mean process',
 'rationale':'prompt conditional curvature is excessive except q1, whereas postpeak curvature lacks charge-dependent spread and local-baseline residual is separate; do not indiscriminately increase global white noise or smooth every pulse',
 'process_equation_proposal':'delta_v(t)=A(c)*e(t;c,z)*eta_rho(t); eta_rho is normalized zero-mean short filtered independent noise; e is zero on pretrigger baseline and has an explicit smooth postpeak envelope; A uses allowed G4 condition, not a sampled REAL template',
 'coupled_constraints':'complete QC TRAIN distributions of local voltage/shape patches and derivatives, balanced prompt/postpeak/tail blocks, conditional charge occupancy/CDF retained; no particle-truth or FoM-maximization loss',
 'not_photon_statistics_or_unique_hardware_cause':True,
 'mechanism_ablation':'two arms both receive the identical new local joint constraint, same R3B checkpoint/trainable initialization, allowed TRAIN statistics, seed/update/selection budget; signal-envelope process on versus identically disabled; share parent/z/baseline white and save additional independent process RNG',
 'specific_risks':['Adding noise to q3/q4 prompt can worsen existing excess curvature; separate the postpeak envelope from the prompt shape constraint.','The new gated process does not itself repair pretrigger RMS spread/ACF; baseline-local C2ST remains a separate outcome.','Positive/negative control sensitivity and all retained distribution metrics must accompany candidate comparisons.','Only a bounded new task can implement this proposal; no R8 fit in R7.']}
compact={k:m[k] for k in ['status','question','C2ST','matrix','selection','processing_chain','common_operator_receipts','after_operator_statistics','same_N_conditional_reference','PSD_density_widths','CAL_reference_valid','q4_no_fallback','counters','REAL_REAL_definition','negative_control_not_equivalence_tolerance','DEV_historical_exposure','no_response_fit_or_EVAL','elapsed_before_plots_s','worker_PID']}
compact['source_statistics']=m['source_statistics'];compact['interventions']={n:{k:v for k,v in d.items() if k!='parameter_tensor_SHA256s'} for n,d in m['interventions'].items()};compact['recommended_next_test']=recommendation
compact['publication_payload']='summary statistics/config only; no event IDs, selections or individual heldout predictions; full local result/predictions preserved'
write('METRICS_COMPACT.json',compact);write('NEXT_RECIPE_RECOMMENDATION.json',recommendation)
phase=m['source_statistics']['GEN']['regions']['overall']['features'];real=m['source_statistics']['REAL']['regions']['overall']['features']
report=f'''R7 的固定开发诊断发现，剩余差异属于多个时间区域的联合响应：prompt局部标准化后的HGB平衡准确率为0.992，postpeak为0.928，tail为0.938；去掉prompt仍0.956。共同分数对齐、4倍ADC再量化或1ns平滑、以及固定权重的关ADC/信号块平均干预均未使完整shape接近同N实测阴性参照，仍为0.986–0.998。现有结果不支持把完整残差主要归咎于这些已试处理链差异，也不证明唯一物理原因。

所有52预定HGB、4000固定权重干预输出与80次固定窄区间密度EM都完成，响应更新0；EVAL、SSH、新Geant4、环境安装、新代理和Git操作均0。成功作业和所有科学输出保留，首次工程失败未隐藏。9图已经实际视觉检查。

研究对象和用途

原R3B checkpoint/保存主2000 G4父条件/latent/white及生成波形复用，所有数值前config及PROTOCOL已经落盘。REAL DEV2000、REAL TRAIN3200按既有QC/人工组划分得到fit2592/val599；只在fit2592按固定身份散列取2000作阴性池。每个变体使用同一父/组划分和选序，每域分类训练600/留出250，GEN与REAL可用记录分别1418/582与1385/615；阴性池1236/764，没有支持后改样本。真实训练池与DEV的原始50记录人工组不交叠。classification training-only中位填补/标准化，不使用留出目标统计；响应模型此轮没有训练。

REAL–REAL为同次采集的TRAIN–DEV比较，明确不同于旧R4的EVAL内半组参照。它表征当前角色内抽样/时序群差异，不能当跨采集真实独立run、物理等效容差或新的外部盲测。阳性采用已规定尾部+噪声扰动，对同一REAL DEV事件建立配对两视图，簇bootstrap保持其同父关系。500次bootstrap条件于固定分类器，不含响应训练不确定性，也没有作多重因果检验。

区域信息边界与完整矩阵

区域voltage直接取本段原始V；local只减本段均值/除本段RMS。baseline的本区RMS、ACF1..8、非零标准谱不含脉冲峰/积分信息。no_prompt仅有0:100与150:480，未带入PSD/rise或全局peak。所有10个区域视图在人工输入上“任意修改全部区域外值，特征逐位不变”通过。原始QC选择来自统一整事件规则，故这是QC条件下的区域证据；不能把共同QC选择也说成只用区域信息。完整历史shape明示包含全局峰归一化与全局标量，只作为完整视图。其BA与区域视图不能当严格可加的重要性。

{chr(10).join(rows)}

36个原始视图拟合、12个共同操作/对照拟合、4个两干预完整视图拟合，共52/60。各视图均含完整QC事件池，unknown不删除。阴性16个视图/操作的簇95%区间均包括0.5（某些图中下限因三位舍入显示0.500）；阳性在所有视图仍检出差异，baseline局部只0.644表明去幅度会降低对该扰动的能力。没有以negative未检出推出完全等效。

共同变换与信息变化

{chr(10).join(oprows)}

原native完整shape GEN–REAL BA0.996；分数对齐0.998、ADC4 0.996、sigma1 0.986、先对齐再平滑0.986，各95%区间与native大范围重叠。本次结果更稳妥的表述是“这些固定操作下仍有强可检差异”，不是为较小下降声称修复、显著改善或因果解释。对应阳性0.982/0.994/0.978/0.964没有丢失全部检验能力，阴性0.490/0.508/0.474/0.488。

操作统一施加两域/阴性/阳性；50%最后主峰上升交点移到120，线性插值与边缘值填充一致，支持变化计入common_operator_receipts。全部原2000 QC仍在各操作后有效，没有借重QC删掉困难事件。平滑令REAL median rise从5.9024升到6.2071ns、GEN从5.8127升到6.0312ns；对齐+平滑为6.2855/6.0592ns。REAL/GEN median PSD分别从0.25865/0.24880移到0.26377/0.25231，峰幅也下降；损失/再分配了物理信息，不能当响应模型改进。所有操作的电荷、峰位/幅度、rise/PSD摘要和密度宽度在metrics。

采用原TRAIN定义8窄logQ区间、固定初始化、最多500次更新及旧支持/floor/分离门槛，只有描述密度EM，不是CAL拟合或新粒子标签。REAL native及各操作均8区合格，GEN前7区合格，第8区各操作不收敛；sigma1的高成分还不足20，失败保留。曲线的峰宽接近/变化很小不改变完整C2ST失败。既有CAL q4参考仍失败，全unknown/FoMundefined，没有fallback；q1 HC neutron缺口未被宣称修复。

固定权重干预

{chr(10).join(ivrows)}

两干预使用实际保存模型、同主latent和white，没有重放已完成的主2000推理或另选随机响应。26个可学习参数张量逐位等于绑定checkpoint。signal_block_mean20仅改变派生time buffer，把解析核/时间RBF在k+j/20ns评估并均值，再归一化并加原1ns层噪声、量化/CFD；不是对完整0.05ns真实采集链的复演，未建模其高频噪声/带宽或实际ADC。它的CFD窗起点变化814/2000，因而大maxabs差既含相位/窗口移动，不是逐事件物理转换误差。关ADC有2条窗起点改变，最大波形差同样受此影响。

处理链直接证据

REAL保存转换是10000点原波形先20点均值为500→float64 ADC→float32前100基线/全峰→主峰相连整数CFD→480窗；GEN解析信号直接1ns点采样+512点平稳谱噪声→Torch f32 ADC→前102基线/全峰→同整数分支→480窗。已有工程脉冲的NumPy/Torch分支与边缘支持一致，f64/f32 ADC在远离半码案例误差<1e-7V；这不是对所有实测波形/硬件的完整独立校准。

保存REAL/GEN都0削顶/0填充，因此不能用这两项说明本批主差距。cross50相位中位GEN {phase['cross50_phase']['quantiles'][3]:.6f}、REAL {real['cross50_phase']['quantiles'][3]:.6f}；GEN峰index中位127、REAL126。ADC栅格RMS残差中位GEN {phase['ADC_lattice_RMS_steps']['quantiles'][3]:.8g}steps、REAL {real['ADC_lattice_RMS_steps']['quantiles'][3]:.8g}steps，二者都远小于一ADC码，符合不同浮点算术的细微残差；不能把它等同真实硬件分辨率差或分类原因。共同粗量化与关ADC仍高可分，是限制该简单解释的受控证据。

局部联合统计与电荷依赖

曲率定义为该段原始V的二阶差分RMS；relative版本除同段总体RMS，ACF1亦只使用同段去均值/全段分母。严格符号极值计数忽略ADC平台，不能称光子数或脉冲个数。条件图为自身signed Qt的CAL桶，不是入射能量或粒子真值；matched-N摘要对每桶GEN/REAL DEV/REAL TRAIN都按固定散列取相同记录数，字段undefined数另列。

{chr(10).join(curve)}

overall GEN prompt二阶差分RMS中位8.458mV vs REAL6.342mV，postpeak则1.509mV vs3.974mV；同N RR的postpeak W1仅0.0373mV。更关键的是GEN postpeak中位随q1–q4几乎恒定1.512/1.488/1.515/1.527mV，REAL为2.539/3.339/4.577/6.219mV；GEN postpeak相邻相关也偏强。prompt q1相反偏低，q2–q4偏高；不能以全局降曲率/全局加白噪解决。

baseline RMS中位GEN0.6315mV/REAL0.6280mV近，但SD0.0494mV/0.1052mV，GEN分布尾窄；baseline ACF1中位0.0310/−0.0312，其同N W1 0.06445大于RR0.00586。baseline局部BA0.652说明除幅度后仍有差别。非DC谱图显示部分频率及ACF偏差，TRAIN fit/val与DEV摘要支持它不是单个DEV角色的孤立异常；没有进行电子噪声独立标定，也未把平稳谱接近等同涨落全部匹配。

一个优先修正配方与一个机制消融

建议在R3B主干/条件CDF基础上，加入不预先avg-pool的局部整段联合及一、二阶差分分布约束，平衡prompt/postpeak/tail并保持电荷条件目标；同时用一个低维、信号包络门控的短相关零均值随机过程提供峰后局部随机自由度。拟议形式delta_v(t)=A(c)e(t;c,z)eta_rho(t)，pretrigger包络为0，峰后包络可独立于prompt控制，eta为固定定义短滤波独立随机流；condition来自G4父事件，不采样REAL模板。它只是模型中的随机响应假设，不能命名为已验证的光子统计、PMT涨落定律或唯一硬件来源。

局部联合约束可能先纠正过陡前沿，但现有RBF在峰后宽12/40ns，GEN峰后曲率近乎不随电荷变化，与基线噪声贡献占主导的解释一致；本轮没有单独识别信号和噪声的贡献，单纯调旧平滑时间系数是否足够尚未证明。故更有区分度的配对消融是：两臂同R3B可学习初始化、同TRAIN合法统计/随机种子/更新数/内部验证，都加同一新局部联合约束；仅主臂开启该信号过程，消融臂关闭它。共同父条件、latent、基线white保持一致，并保存新增独立随机流。这样检验新增随机自由度的价值，不把多训练或联合约束本身混成过程效应。

风险必须保留：新增噪声可能恶化q3/q4已有过大的prompt曲率，包络/局部约束需把前沿与峰后作用分开；baseline前不启用的过程不能自动修复baseline RMS/ACF残差。次轮全部波形/QPSD/涨落/C2ST/控制仍要保留，不能只看postpeak一项。R7未执行这些修改；根代理另登记单配方及配对机制消融。DEV已参与历史架构开发，下一验证必须如实声明旧EVAL为已查看同采集基准，不重新包装成新外部盲测。

运行、自检与交付

首实际工具7a0487/PID3932/native0，{start['start_UTC']}，阶段硬截止{start['deadline_UTC']}。准备ea62a3/PID35080、AST/绑定PID37188原生0。首次监督37788/worker46932工具a8db80→6bc0d2/native1，10.8453012s，在人工工程检查发现借用模块np=None未注入，实际HGB/推理/实测数组解码都0；RSS null、43失败探针原样保留。原source_numeric_1、FAILURE及ENGINEERING_FAILURE_001均保留，不改写为合规成功。

在同预算内只修复所借接口模块依赖注入，以及Windows RSS结构遗漏的QuotaNonPagedPoolUsage；没有改科学矩阵/模型/阈值/选序。修复后tool e24ddf→e8d053/native0，supervisor61504/worker13648，16:19:22.922796→16:20:36.660101，73.7371089s、RSS1118969856B、0探针失败。含失败累计{sum(x['wall_s'] for x in h):.9f}s/1800，CPU4、成功RSS1.12GB/4GiB，失败RSS未知不能追溯补造。准备时两次错误源路径检索保留在config；纯字节绑定/最终清单哈希经过容器压缩body，不等于所有成员被数组解码。

已实际查看9图：regional_c2st、common_transform_c2st、fixed_intervention_c2st、processing_local_shape、conditional_fronts、conditional_curvature、baseline_noise、common_observable_changes、common_width_changes。区域/对照/坐标/样本数可读，谱排除近零DC，宽度失败保留缺口；conditional_fronts较密但无截断。详细统计metrics、精简METRICS_COMPACT、配置/协议/实现、两干预同父波形、分类留出预测及输入绑定/本清单交付。逐事件选序/预测/波形只留本地，不进入Git；Git由ROOT办理。封存后不在本阶段追加数值；随后在根代理已登记的新R8目录执行下一任务。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
figures=[q.name for q in sorted((ROOT/'results/figures').glob('*.png'))]
assert len(figures)==9
write('VISUAL_QA.json',{'actual_viewed':figures,'performed_with':'view_image tool','axes_legends_counts_readable':True,'no_layout_clipping':True,'width_failures_preserved_as_gaps':True,'nonzero_baseline_spectrum':True,'plots_refit':0})
write('RESOURCE_RECEIPT.json',{'numeric_history':h,'numeric_cumulative_wall_s':sum(x['wall_s'] for x in h),'numeric_limit_s':1800,'successful_RSS_peak_bytes':h[-1]['peak_RSS_observed'],'first_failed_RSS_unknown':h[0]['peak_RSS_observed'] is None,'RSS_limit_bytes':4294967296,'threads':4,'counters':m['counters'],'stage_deadline_UTC':start['deadline_UTC'],'all_data_cache_E_D':True,'system_dependency_load_IO_not_instrumented':True})
now=datetime.datetime.now(datetime.timezone.utc);assert now<datetime.datetime.fromisoformat(start['deadline_UTC'])
members=[binding(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']]
write('FINAL_MANIFEST.json',{'files':members,'members':len(members),'bytes':sum(x['bytes'] for x in members),'self_exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'pure_byte_hash_no_array_decode_at_seal':True})
r={'status':'sealed_complete','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_UTC':start['deadline_UTC'],'report':binding(ROOT/'REPORT.md'),'metrics':binding(ROOT/'metrics.json'),'metrics_compact':binding(ROOT/'METRICS_COMPACT.json'),'manifest':binding(ROOT/'FINAL_MANIFEST.json'),'numeric_cumulative_wall_s':sum(x['wall_s'] for x in h),'HGB_fits':52,'fixed_inference_rows':4000,'response_updates':0,'EVAL_read':0,'SSH':0,'native_exit_supplied_by_external_tool':True,'no_more_stage_writes':True}
write('FINAL_RECEIPT.json',r)
print(json.dumps({**r,'last_file_write_UTC':datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat(),'total_files':len(members)+2,'total_bytes':sum(x['bytes'] for x in members)+(ROOT/'FINAL_MANIFEST.json').stat().st_size+(ROOT/'FINAL_RECEIPT.json').stat().st_size},ensure_ascii=False),flush=True)
