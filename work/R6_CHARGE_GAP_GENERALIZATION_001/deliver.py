from __future__ import annotations
import pathlib,json,hashlib,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8'))
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def binding(p):
    b=p.read_bytes();return {'path':str(p.relative_to(ROOT)).replace('\\','/'),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def f(x,n=6):return 'undefined' if x is None else f'{x:.{n}f}'
start=read('START_RECEIPT.json');m=read('metrics.json');h=read('RUN_HISTORY.json');rep=read('REPLAY_CHECK.json');e=m['evaluation'];roles=m['data_roles'];eq=read('INITIAL_PARAMETER_EQUALITY.json')
assert m['status']=='complete' and rep['status']=='pass' and h[-1]['worker_native_exit']==0
trainrows=['|臂|阶段|实际更新|所选更新|内部验证值|训练秒|','|---|---|---:|---:|---:|---:|']
for arm,phases in m['training'].items():
    for p in phases:trainrows.append(f"|{arm}|{p['phase']}|{p['optimizer_updates']}|{p['selected_update']}|{p['fixed_TRAIN_validation']:.6f}|{p['training_elapsed_s']:.3f}|")
tab=['|候选|q3 QC|PSD W1|PSD KS|rise W1(ns)|Qt W1(mV ns)|peak W1(V)|RMS W1(V)|FoM|HC低/高/unknown|','|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
for name,d in e['domains'].items():
    b=d['regions']['q3'];x=e['domain_vs_EVAL_REAL'].get(name,{}).get('q3',{});fm=b['FoM']
    def dist(k,key):return f(x.get(k,{}).get(key)) if name!='REAL' else 'reference'
    tab.append(f"|{name}|{b['QC_n']}|{dist('PSD','W1')}|{dist('PSD','KS')}|{dist('rise_10_90_ns','W1')}|{dist('Qtotal_mVns','W1')}|{dist('peak_V','W1')}|{dist('baseline_RMS_V','W1')}|{f(fm['FoM'])}|{b['HC_gamma_like_n']}/{b['HC_neutron_like_n']}/{b['unknown_n']}|")
outside=['|候选|分区|QC|PSD W1|rise W1(ns)|FoM参考有效|','|---|---|---:|---:|---:|---|']
for name in ['R2','R3','R3B','FULL','GAP']:
    for r in ['q1','q2','q4']:
        b=e['domains'][name]['regions'][r];d=e['domain_vs_EVAL_REAL'][name][r];outside.append(f"|{name}|{r}|{b['QC_n']}|{f(d['PSD']['W1'])}|{f(d['rise_10_90_ns']['W1'])}|{b['reference_valid']}|")
c2=e['C2ST_q3'];c2text='固定 q3 C2ST 未定义。原因：'+c2.get('reason','')+'。各域支持='+json.dumps(c2['supports'],ensure_ascii=False)
if c2['defined']:
    ct=['|候选|shape BA|shape AUC|shape+amp BA|shape+amp AUC|','|---|---:|---:|---:|---:|']
    for name,v in c2['results'].items():
        a=v['shape_only'];b=v['shape_plus_amplitude_proxy'];ct.append(f"|{name}|{a['balanced_accuracy']:.3f}|{a['AUC']:.3f}|{b['balanced_accuracy']:.3f}|{b['AUC']:.3f}|")
    c2text='\n'.join(ct)+'\n\n所有候选共同使用每域 200 训练/100 留出，保持原人工组/父事件 SHA 划分；训练内中位数填充与固定 HGB 参数。heldout CI 条件于已训练分类器，块置换只表示相关特征依赖，不是因果归因。此分类器仅评价，没有优化响应。'
widthrows=['|域|合格窄区间(1-based)|失败/缺口|','|---|---|---|']
for name,w in e['narrow_widths'].items():
    good=[r['bin']+1 for r in w['slices'] if r['valid']];bad={str(r['bin']+1):r['failure_reasons'] for r in w['slices'] if not r['valid']};widthrows.append(f'|{name}|{good}|{json.dumps(bad,ensure_ascii=False)}|')
gap=roles['GAP'];rr=e['REAL_REAL_q3'];q=e['full_vs_gap_q3']
summary=f"q3 PSD W1 FULL={f(q['PSD']['FULL_W1'])}, GAP={f(q['PSD']['GAP_W1'])}；rise W1 FULL={f(q['rise_10_90_ns']['FULL_W1'])} ns, GAP={f(q['rise_10_90_ns']['GAP_W1'])} ns。"
nominal=e['fixed_nominal_Q3_parent_supplement']
compact={k:m[k] for k in ['status','training','generation','CAL_reference_valid','q4_CAL_failure_preserved','retrospective_single_acquisition','architectures_have_full_DEV_historical_exposure','current_GAP_no_q3_fit_or_validation_or_statistics','not_incident_energy_or_cross_condition_generalization','no_paired_source_REAL_energy_calibration','source_nominal_charge_is_saved480window100_400','worker_PID','master_seed','SSH','new_G4','numeric_elapsed_s']}
compact['data_roles_counts']={a:{k:v for k,v in roles[a].items() if not isinstance(v,list)} for a in ['FULL','GAP']};compact['source_coverage']=roles['source_coverage']
compact['evaluation']={k:e[k] for k in ['domains','domain_vs_EVAL_REAL','charge_migration','fixed_nominal_Q3_parent_supplement','C2ST_q3','full_vs_gap_q3']}
compact['evaluation']['narrow_widths']={name:{'N':w['N'],'below_logQ_range':w['below_logQ_range'],'above_logQ_range':w['above_logQ_range'],'slices':[{k:v for k,v in r.items() if k not in ['histogram_counts']} for r in w['slices']]} for name,w in e['narrow_widths'].items()}
compact['evaluation']['REAL_REAL_q3']={k:v for k,v in rr.items() if k!='references_by_N'}
compact['evaluation']['REAL_REAL_q3']['references_by_N']={k:{a:b for a,b in v.items() if a!='records'} for k,v in rr['references_by_N'].items()}
compact['shared_parameter_digest']=eq['shared_parameter_digest'];compact['arm_initial_parameter_digests']=eq['arm_initial_parameter_digests'];compact['all_trainable_parameters_identical']=True
compact['replay']=rep;compact['publication_payload']='summary only, no record identity lists/predictions/individual RR selections; complete local metrics retained'
write('METRICS_COMPACT.json',compact)
report=f'''R6 完成了固定连续电荷代理区间排除的配对回顾实验。{summary}这些是按输出自身电荷q3筛选、对照同一 EVAL q3 的结果。必要的固定源名义q3父记录全输出对照则给出 PSD W1 FULL={f(nominal['FULL']['vs_same_EVAL_q3']['PSD']['W1'])}、GAP={f(nominal['GAP']['vs_same_EVAL_q3']['PSD']['W1'])}，排序相反；因此不能概括GAP在所有指标和所有条件定义下都恶化。前沿与联合响应仍明显不符：q3 C2ST FULL/GAP 两视图 BA=1.0，prompt块置换下降.49，仅是相关诊断。没有据测试结果增改模型、课程、区间或随机种子。全部候选与失败缺口保留，不能从单个分布或 FoM 数值宣称全面响应等效。

任务在 R4 打开 EVAL 前冻结，SHA {start['task_SHA256']}，FROZEN_METHOD SHA b4e1f86807dba40f6004bda48a40d0644fe5039535a3c3eb2e402fc213065186。固定区间为 [3703.7782948464155,6711.804904974998) mV ns，是保存窗 signed 总电荷/可见沉积代理；不是入射中子能区。全部记录来自用户确认的一次真实采集，人工 original_index//50 组不等于独立 run。

当前数据隔离与初始参数

FULL 使用 REAL 2592 fit/599 内部验证、G4 6593 fit/1599 内部验证。GAP 使用 REAL {gap['REAL_fit_n']} fit/{gap['REAL_val_n']} 验证、G4 {gap['G4_fit_n']} fit/{gap['G4_val_n']} 验证；分别移除 REAL {gap['removed_REAL_fit']}/{gap['removed_REAL_validation']} 与 G4 {gap['removed_G4_fit']}/{gap['removed_G4_validation']} 个 fit/验证记录。DATA_ROLES_AND_GAP_EXCLUSIONS 保存确切索引/ID，允许与排除集合交叠 0，REAL 拟合/验证人工组交叠 0。

G4 排除轴由已有无噪声、原 CFD 对齐后的 480 点 clean cache，按自身保存窗基线/100:400 signed 总电荷计算，既不使用学习增益，也不拿原 512 点积分 Qsource 替代。保留区间两侧父事件；输入条件覆盖和 EE/hadron 可见组成统计见完整清单。原 G4 沉积历史只保留非零沉积/触发条件，当前没有新 Geant4 仿真，也没有新物理校准。

主 Torch/NumPy 种子 202610086 在任何模型构造前设置并保存。共享新初始化的所有可学习参数 SHA 为 {eq['shared_parameter_digest']}，FULL/GAP 完全相同。初始化标量只由 GAP 允许 fit 估计并供两臂共享，避免把被留区间统计量传入 GAP；没有加载旧 R2/R3/R3B 已训练权重。每臂固定噪声谱 buffer 独立由本臂允许 REAL fit 估计，参数相等不要求数据统计 buffer 相等。初始标量源与每臂估计值均记录。特征尺度、投影/固定验证流、条件 CDF/窗口占比目标同样保持本臂用途分离；GAP 的噪声、尺度或 CDF 目标未使用排除记录，也未复制 FULL 目标。

配对固定课程

{chr(10).join(trainrows)}

R2 的 batch128/lr.002/max1600，R3 batch128/lr.002/max2000，R3B batch256/lr.001/max2000及原损失、Adam、梯度裁剪5、每100检查/严格最小内部验证选点全部复用。各阶段使用主种子训练随机流与原验证种子2026100821；两臂每阶段对应流一致。R3只继承本臂选中R2，新增时间头零输出初始化；R3B只继承本臂R3。R3B八个软logQ窗口中心/宽度和完整 fit/验证 CDF 分别从本臂允许 REAL TRAIN 推导，评价窄区间则固定为旧 R3B 开发定义。没有借 FULL 目标填 GAP，也没有根据 EVAL 提前停止、挑阶段或追加拟合。

共同主生成与电荷迁移

两个最终模型和 RNG/同父输入先保存，再用同 G4 DEV2000 与共同随机流2026100862生成各一个主实现；不挑输出。之后 MODELS_AND_GENERATION_FROZEN_BEFORE_EVAL 保存实际 SHA、模型课程、主波形，才打开本地 R4 EVAL 缓存。输出均按共同自身前100点基线、100:150快速/100:400总积分计算 signed PSD。全部 QC 有效输出与 unknown 保留。

迁移矩阵的行是源名义 q1..q4，列是生成后自身 Qt q1..q4：

FULL={json.dumps(e['charge_migration']['FULL'],ensure_ascii=False)}

GAP={json.dumps(e['charge_migration']['GAP'],ensure_ascii=False)}

主比较按输出自身电荷 q3 对照相同 EVAL q3。名义源 q3 的396父事件全部输出另作为必要补充，未按输出落回 q3 筛选；FULL237/GAP222落回q3，而FULL156/GAP173迁入q4。固定源全输出PSD W1为FULL0.02166037/GAP0.01707192，和按输出q3筛选的FULL0.00503589/GAP0.00825670排序相反；电荷条件和迁移改变了比较对象，不能据某一条件宣称全面更优。各父事件迁移数、完整特征和距离在 fixed_nominal_Q3_parent_supplement。源与真实事件没有一一配对能量标定，因此区间对应是操作定义，电荷迁移必须随主比较一起解释。

主 q3 结果

{chr(10).join(tab)}

完整全部有限值距离、位置/分位/散布、上升沿、峰幅、电荷、RMS/谱/ACF与后验计数见 metrics。表内 FoM 是冻结 CAL 切点两侧总体均值/样本标准差换算 Gaussian FWHM，至少20每侧，否则undefined；它不是粒子真值分离能力或重新混合拟合。HC低/高是 CAL 0.99 密度后验成分，仍不是 n/gamma 真值。

{c2text}

同源真实抽样参考

EVAL q3 有 {rr['REAL_q3_N']} 个 QC 记录、{rr['REAL_q3_groups']} 个人工组。固定64次整组半分，逐域请求与其输出 q3 数相同的每侧样本；若任一次半组支持不足，该域同N参考保留undefined而不补样本。请求数为 {json.dumps(rr['requested_candidate_N'])}，每次半组最小可用数 {rr['minimum_group_half_available']}，附加共同可行数 {rr['common_feasible_N']}。实际每次 N/组数及 W1/KS/FoM 位于 Q3_REAL_REAL_REFERENCE，候选分布的不同N明确列出，没有把重复真实抽样作为跨采集等效界或训练不确定性区间。

窄电荷区间峰位与宽度

{chr(10).join(widthrows)}

使用原开发阶段冻结的8个 logQ 边界，PSD密度EM初始化/.5权重/25–75%均值、500上限/1e-8参数容差/1e-8方差floor、N*w>=20、均值间隔/pooledsigma>=1、无重启；原宽度和责任权重局部 PSD~logQ 去趋势宽度都保存。失败/不收敛/floor/支持不足都维持 undefined；两成分不是粒子标签，不强造1/sqrt(E)收窄规律，也不最大化FoM。训练中心与评价边界用途不同且各自绑定。

本轮 REAL 用全部7583 QC EVAL，而 R4 主图只用固定2000；因此本轮 REAL bin8有700条、密度诊断合格，与R4主2000的bin8有195条且不收敛的支持条件不同。不能把这差异当新校准成功。CAL q4的既存参考依然失败，当前EVAL密度描述不会改变其标签或FoM参考。GAP窄bin6不收敛与FULL bin8低成分支持不足都如实保留。

外区间控制

{chr(10).join(outside)}

q4 CAL 参考保持失败：全unknown、FoMundefined，没有全局 fallback 或补拟合。FULL/GAP的当前课程和同初始化差异才是配对实验；未知初始化的历史 R3 只作冻结参照。指标可比较，却不能隐藏条件下样本量迁移、较弱尾部或某宽度定义失败。

证据边界与运行

架构及配方已用完整旧DEV开发，包含q3；源解析响应幅度、闪烁常量和历史实测噪声先验也已有暴露。当前实验仅证明“这次拟合/验证/统计排除q3”后的表现，不能称整个研究从未见过该区间、物理未见能区或跨仪器/工况推广。冻结CAL区间/参考是既有先验，未作为GAP响应目标。

首实际工具7c9290/PID50264/native0，{start['start_UTC']}；监督与工作实际身份、起止、退出/RSS完整见RUN_HISTORY。累计数值 {sum(r['wall_s'] for r in h):.6f}s/2400，峰RSS {max(r['peak_RSS_observed'] or 0 for r in h)}B/4GiB，CPU4，所有数据/缓存/源码/日志在E，SSH与新G4均0。保存后独立进程重新读取两臂最终模型，使用固定保存的主2000条件/z/white按同256批推理；FULL/GAP各960000值 bitwise={rep['arms']['FULL']['bitwise']}/{rep['arms']['GAP']['bitwise']}、maxabs={rep['arms']['FULL']['maxabs']}/{rep['arms']['GAP']['maxabs']}；没有重训或全套重评。原生工具ID在执行交付消息列出，失败如有仍保留日志及费用。

八张关键图已实际查看：charge_migration、charge_psd、q3_response_features、conditional_waveforms、fluctuations、energy_peak_width、energy_psd_slices、outside_gap_controls。所有候选、坐标和样本数可读，无布局裁切；失败宽度以曲线缺口保留。PSD显示范围为[-.1,.6]，窄区间密度分母包含全区间事件并记录网格外数量；RMS显示范围0..08，完整距离与分位没有删尾。fluctuations频谱图包含扣基线后数值近零的DC，压缩了非零频谱的视觉差异，不能据曲线重叠称噪声谱完全一致；ACF曲线/完整指标保留不符。推理原生回执的response_training字段沿用阶段标签，实际replay的fits=0/evaluations=0以REPLAY_CHECK为准，不能解释为重放又训练。

源码准备时工具d7f8a4的rg曾指向误记的旧诊断文件名并返回路径错误（同一命令最后GetContent成功，native0不代表该子读取成功），在数值启动前按已冻结DIAGNOSTIC_DEFINITION路径修正；数值源与任务未变。训练日志的requires_grad转标量警告只发生在保存诊断摘要，没有非有限梯度或训练失败。交付两臂三阶段初始/最终权重、RNG、主输入/波形、排除清单、共同metrics与关键图及本清单。METRICS_COMPACT去除身份清单/逐次RR记录/预测供Git，完整metrics与模型/数据留本地。此轮结束不自行开R7或扩大科学搜索。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
write('VISUAL_QA.json',{'actual_viewed_eight':['charge_migration','charge_psd','q3_response_features','conditional_waveforms','fluctuations','energy_peak_width','energy_psd_slices','outside_gap_controls'],'readable_axes_legends_counts':True,'failed_width_gaps_preserved':True,'performed_via':'view_image tool','scientific_refit':0})
write('RESOURCE_RECEIPT.json',{'numeric_history':h,'numeric_cumulative_wall_s':sum(r['wall_s'] for r in h),'numeric_limit_s':2400,'RSS_observed_peak_bytes':max(r['peak_RSS_observed'] or 0 for r in h),'RSS_limit_bytes':4294967296,'threads':4,'SSH':0,'new_G4':0,'current_stage_responses_trained':'two fixed paired1600/2000/2000 curricula only','replay_retraining':0,'stage_deadline_UTC':start['deadline_UTC'],'E_or_D_all_cache_and_data':True,'system_dependency_loading_IO_not_instrumented':True})
members=[binding(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']]
write('FINAL_MANIFEST.json',{'files':members,'members':len(members),'bytes':sum(p['bytes'] for p in members),'self_exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'all_hashes_byte_only_at_seal':True})
now=datetime.datetime.now(datetime.timezone.utc);assert now<datetime.datetime.fromisoformat(start['deadline_UTC'])
r={'status':'sealed_complete','seal_PID':os.getpid(),'seal_UTC':now.isoformat(),'deadline_UTC':start['deadline_UTC'],'report':binding(ROOT/'REPORT.md'),'metrics':binding(ROOT/'metrics.json'),'manifest':binding(ROOT/'FINAL_MANIFEST.json'),'numeric_cumulative_wall_s':sum(x['wall_s'] for x in h),'external_command_supplies_actual_native_exit':True,'no_more_stage_writes':True}
write('FINAL_RECEIPT.json',r)
print(json.dumps({**r,'last_file_write_UTC':datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat(),'total_files':len(members)+2,'total_bytes':sum(x['bytes'] for x in members)+(ROOT/'FINAL_MANIFEST.json').stat().st_size+(ROOT/'FINAL_RECEIPT.json').stat().st_size}),flush=True)
