from __future__ import annotations
import pathlib,json,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text('utf-8'))
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf-8')
def clean(x):
    if isinstance(x,dict):return{k:clean(v)for k,v in x.items()if k not in ['validation_trace','source_parent_index','fixed_source_parent_index','histogram_counts','trace','initial']}
    if isinstance(x,list):return[clean(a)for a in x]
    return x
def f(x):return 'undefined'if x is None else str(x)if isinstance(x,bool)else f'{x:.6g}'if isinstance(x,(int,float))else str(x)
m=read('metrics.json');native=read('REMOTE_NATIVE_RECEIPT.json');replay=read('REPLAY_RECEIPT.json');pair=read('PAIRING_RECEIPT.json')
c={k:v for k,v in m.items()if k not in ['training','same_parent_noise']};c['training']={a:{k:v for k,v in t.items()if k!='validation_trace'}for a,t in m['training'].items()};c['same_parent_noise']={a:{k:v for k,v in d.items()if k!='rows'}for a,d in m['same_parent_noise'].items()}
write('METRICS_COMPACT.json',clean(c))
resource={'stage':'R20','seal_deadline_UTC':'2026-10-09T10:07:53.627396+00:00','source_only_first_native':read('START_RECEIPT.json'),'science_cumulative_s':native['numeric_cumulative_s'],'scientific_history':native['history'],'counters':m['counters'],'GPU_allocated_peak_bytes':max(a['GPU_allocated_peak_bytes']for a in m['training'].values()),'GPU_reserved_peak_bytes':max(a['GPU_reserved_peak_bytes']for a in m['training'].values()),'RSS_peak_bytes':max(a['RSS_peak_bytes']for a in native['history']),'CPU_threads':4,'limits':{'numeric_s':1800,'training_each_s':600,'training_total_s':1200,'G_D_each_arm':4000,'RSS_bytes':4294967296,'GPU_allocated_bytes':4294967296,'HGB_plan':14,'HGB_max':24,'width_EM_max':16,'output_bytes':268435456},'no_new_source_hash_or_migration_test':True,'new_G4':0,'EVAL_scientific_decode':0,'CAL_waveform_decode':0,'raw_long_waveform_decode':0,'fresh_replay':replay,'normal_transfer':read('RETURN_TRANSFER_RECEIPT.json'),'failed_science_history':[a for a in native['history']if a['exit_code']!=0]}
write('RESOURCE_RECEIPT.json',resource)
supp=read('SAVED_RESULT_SUMMARY_NATIVE_RECEIPT.json')
supp_failed=read('summary_attempt_001/SAVED_RESULT_SUMMARY_NATIVE_RECEIPT.json')
repeat_wave=read('REPEAT_WAVEFORM_SUMMARY.json')
c['same_parent_waveform_variation']=repeat_wave
write('METRICS_COMPACT.json',clean(c))
resource['remote_main_and_replay_s']=native['numeric_cumulative_s']
resource['existing_result_summary_history']=[supp_failed,supp]
resource['engineering_startup_failure']={'tool_chunk_id':'8bcef5','native_exit':1,'PID':'unknown','elapsed_s_upper_including_shell':.4821916,'exception':'ModuleNotFoundError: psutil; no scientific worker started; replaced local monitor with Windows GetProcessMemoryInfo, no install.'}
resource['control_inventory_failure']={'tool_chunk_id':'8c55a7','native_exit':1,'PID':'unknown','elapsed_s':.6614258,'exception':'requested nonexistent plots path; actual outputs in results/figures; no scientific work.'}
resource['science_cumulative_s']=native['numeric_cumulative_s']+supp_failed['elapsed_s']+supp['elapsed_s']+.4821916
resource['science_after_remote_terminal']='Existing-result repeat waveform summary and trace plotting only, 0 inference/fit/generation; failed and successful helper time counted.'
resource['failed_science_history']+=[supp_failed,resource['engineering_startup_failure']]
write('RESOURCE_RECEIPT.json',resource)
lines=['# R20：完整源条件转换中的细时间分布项','',
'**本轮整体失败：新增细时间项降低了内部验证目标，但没有得到完整实测响应相似性，并使预指定纯had组的中电荷PSD中心退化。** FINE所选fine_val=.357049，比CONT的.377267低；完整shape C2ST BA仍为.996（CONT .994），基线ACF1 W1变为.338123（CONT .303329、R19 TILT .279615）。不能将新目标下降当作成功。',
'',
'预指定pure-had source-bin3/4的FINE谷区比例回到68.1159%/23.0769%，CONT均为0；FINE平均PSD .240649/.247486，低于旧R19 TILT .273332/.272148。共同支持bins1–4的高密度成分中心误差 .0281572 大于R19 TILT .00829695，两成分仍明显过窄。零同父跨谷仅描述每条源的8个重复噪声，不能证明这些父事件的中心仍被保留。',
'',
'图中两臂的峰后重复纹理均较R19減弱，这包括共同继续训练的变化；不能把全部视觉变化独归FINE。两臂在500/750步选中，4000步末的J20均反弹，完整轨迹保留。总体输出PSD W1小幅降低与源条件中心退化并存，结论以完整总体、条件和源保真同时报告。',
'',
'同一 R19/TILT 所选完整480点转换器、同一 R18 初始判别器、新相同Adam继续训练；两臂源采样均固定TILT，网络与随机容量未变。CONT保留旧损失，FINE额外+1×三组细时间SW；两臂共同J20选点。全部2000源父及原总体比例/实际噪声保留。这是已历史暴露的同采集TRAIN/DEV诊断，非新盲测。',
'','## TRAIN冻结常数','',
'只用REAL TRAIN-fit21047做float64坐标均值/总体标准差(ddof0)，scale=max(std,.05×组坐标std中位数,1e-6)。每组128个单位列高斯投影、独立seed20261009201；不改变采样流，不重估R19 lambda。shape480为扣前100点基线后按正峰下限.0025V归一，diff479为一阶差分，baseline100_V是扣基线的伏特值。常数在任何参数更新前保存；实际范围/触底数量：','',
'|组|坐标|均值范围|std范围|scale范围|floor|触底数|','|---|---:|---|---|---|---:|---:|']
for r in m['fine_constants']['groups']:lines.append('|'+ '|'.join(f(r[k])for k in ['group','coordinates','mean_range','population_std_range','scale_range','floor','coordinates_touching_floor'])+'|')
lines+=['','细时间损失按每组标准化投影分布的128个type7线性分位W1平均，再三组等权；不是无配对逐行MSE。验证沿用R19加权中点CDF的源分位，与等权REAL type7比较，各区/各组等权；保留估计器差异。最高source-val区只有4条。',
'','## 等预算训练与选点','',
'|臂|G/D|训练秒|选step|初始J20|所选J20|末J20|所选J19|末J19|所选原未加权J|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for a,t in m['training'].items():
    lines.append('|'+ '|'.join([a,str(t['updates'])+'/'+str(t['D_updates']),f(t['training_s']),str(t['selected_step']),f(t['initial_validation']['J']),f(t['selected_validation']['J']),f(t['final_validation']['J']),f(t['selected_validation']['J19_weighted']),f(t['final_validation']['J19_weighted']),f(t['selected_validation']['original_unweighted_J'])])+'|')
lines+=['','step0、每250与末点参加共同J20最低选点、同分早者。训练/验证三组完整轨迹及所选/末模型全部保留。','',
'|臂/点|fine_val|shape480|diff479|baseline100V|','|---|---:|---:|---:|---:|']
for a,t in m['training'].items():
    for n in ['initial_validation','selected_validation','final_validation']:
        v=t[n];lines.append('|'+a+'/'+n+'|'+f(v['fine_val'])+'|'+ '|'.join(f(v['fine_groups'][k])for k in ['shape480','diff479','baseline100_V'])+'|')
lines+=['','首实际批次梯度检查（autograd.grad查询不向CONT优化器反传新项）：','',
'|臂|新项初值|output梯度L2|mean_weight梯度L2|全有限/非零|','|---|---:|---:|---:|---|']
for a,t in m['training'].items():
    z=t['first_fine_gradient_check'];lines.append('|'+a+'|'+f(z['fine_value'])+'|'+f(z['output_gradient_L2'])+'|'+f(z['mean_weight_gradient_L2'])+'|'+str(z['output_gradient_all_finite']and z['mean_weight_gradient_all_finite']and z['nonzero'])+'|')
lines+=['','两臂全部source/REAL/bin/U/NCE/noise实际流一致：'+str(all(pair['common_stream_hashes_equal'].values()))+'；与旧R19 TILT相同流：'+str(all(pair['same_old_R19_TILT_streams'].values()))+'。完整私有索引/RNG保留；6593加载源中共同Q支持3753/区外2840，主评价不删区外。',
'','## 完整响应与PSD副作用','',
'|视图|R19 TILT BA|CONT BA|FINE BA|FINE−CONT BA /95%CI|','|---|---:|---:|---:|---|']
for v in m['C2ST']['results']['CONT']:
    d=m['C2ST']['FINE_minus_CONT_paired_CI'][v];lines.append('|'+v+'|'+f(m['C2ST']['reference_results_reused']['R19_TILT'][v]['balanced_accuracy'])+'|'+f(m['C2ST']['results']['CONT'][v]['balanced_accuracy'])+'|'+f(m['C2ST']['results']['FINE'][v]['balanced_accuracy'])+'|'+f(d['BA_delta_first_minus_second'])+' / '+str(d['BA_delta_95CI'])+'|')
lines+=['','固定7视图/600+250组隔离/14 HGB，无DEV调参；原SOURCE/RR/R19拟合复用。区组CI条件于已拟合分类器，不含响应训练不确定性。新损失变小、单特征改善或BA接近.5均不能独立证明整段等效。',
'','|域|QC|整体PSD W1|八区均值PSD W1|基线ACF1 W1|基线RMS W1 mV|rise W1 ns|','|---|---:|---:|---:|---:|---:|---:|']
for a in ['SOURCE_CLEAN','R19_TILT','CONT','FINE']:
    d=m['domain_vs_REAL'][a]['overall'];lines.append('|'+a+'|'+str(m['PSD_main']['tables'][a]['QC_n'])+'|'+f(d['PSD']['W1'])+'|'+f(m['PSD_main']['vs_REAL'][a]['mean8_W1'])+'|'+f(d['ACF1']['W1'])+'|'+f(d['baseline_RMS_V']['W1']*1000)+'|'+f(d['rise_10_90_ns']['W1'])+'|')
lines+=['','预指定purehad source-bin3/4中心/谷区结果，保留全部八区与两种条件定义：','',
'|域|bin3 N/mean/谷比例|bin4 N/mean/谷比例|','|---|---|---|']
for a,rows in m['source_composition']['primary_pure_had_source_bins3_4'].items():lines.append('|'+a+'|'+ '|'.join(str(rows[k]['N'])+'/'+f(rows[k]['output_PSD']['mean'])+'/'+f(rows[k]['valley_fraction'])for k in ['bin3','bin4'])+'|')
lines+=['','峰位/原宽度/去趋势宽度使用同一共同支持，全部单域支持失败仍留undefined：','',
'|成分/字段|共同bin IDs|R19 TILT MAE|CONT MAE|FINE MAE|','|---|---|---:|---:|---:|']
for r in m['energy_conditional_width']['common_support_errors']:lines.append('|'+str(r['component'])+'/'+r['field']+'|'+str(r['common_bin_ids_1based'])+'|'+ '|'.join(f(r['MAE'][a])for a in ['R19_TILT','CONT','FINE'])+'|')
for a in ['CONT','FINE']:
    failed=[{'bin_id_1based':i+1,'reasons':v['failure_reasons']}for i,v in enumerate(m['energy_conditional_width']['tables'][a]['slices'])if not v['valid']];lines+=['',a+'密度支持缺口：'+json.dumps(failed,ensure_ascii=False)]
lines+=['','## 同父噪声、源保真及资源','',
'|臂|EE/had/mixed同父PSD SD均值|跨谷父数|固定源换噪声RMS mV|源Q rho|源PSD rho|','|---|---|---:|---:|---:|---:|']
for a,d in m['same_parent_noise'].items():
    z=m['source_association'][a]['composition_and_PDG']['all'];lines.append('|'+a+'|'+ '/'.join(f(d['by_composition'][k]['within_parent_PSD_SD']['mean'])for k in ['pure_EE','pure_had','mixed'])+'|'+str(sum(v['parents_straddling_valley']for v in d['by_composition'].values()))+'|'+f(m['source_probes'][a]['rows']['fixed_source_changed_noise']['waveform_RMS_vs_correct_reference_V']['mean']*1000)+'|'+f(z['source_Q_to_output_Q']['Spearman'])+'|'+f(z['source_PSD_to_output_PSD']['Spearman'])+'|')
lines+=['','同父128×8不增加独立G4父数量，实测跨事件带宽不是同事件重复真值。源probe保留实际源/噪声来源/参考父三种索引；固定源比较该唯一源的主输出；源roll是敏感性证据而非正确物理映射。pureEE/had/mixed与PDG分开，REAL密度不是粒子标签。源解析波形仅有总EE/had沉积，不能恢复未输入的微观时序。',
'',f"远端supervisor PID {native['supervisor_PID']}；累计科学 {f(native['numeric_cumulative_s'])} s，峰RSS {resource['RSS_peak_bytes']} B/GPU allocated {resource['GPU_allocated_peak_bytes']} B，CPU4。实际 G/D {m['counters']['G_updates']}/{m['counters']['D_updates']}、HGB {m['counters']['HGB_attempts']}、宽度槽 {m['counters']['density_description_slots']}/实际EM {m['counters']['density_EM_attempts']}；独立4000重放 pass={replay['pass']}。",
'','旧阶段未修改，无EVAL/CAL波形/原始长波形/新G4，无来源重新审计或迁移一致性测试。PRIVATE结果/模型/随机序列留E/D；公开METRICS_COMPACT只聚合，详细复现见REPRODUCTION.md。科学负结果不引发系数扫描、第三臂、改网络或扩噪声，本轮结束后ROOT另登记下一问题。']
lines+=['','## 已保存同父波形变化','',
'只读取本轮保存的128×8重复输出：对每父每点用分母7的无偏样本方差，再对固定窗口平均并开方，单位mV。没有再生成或拟合。分组摘要与分位数见REPEAT_WAVEFORM_SUMMARY.json；不把它当作实测同事件重复真值。','',
'|臂|全480|基线0:100|prompt100:150|峰后150:220|尾220:400|','|---|---:|---:|---:|---:|---:|']
for arm,row in repeat_wave['arms'].items():lines.append('|'+arm+'|'+ '|'.join(f(row['whole_population'][k]['mean'])for k in ['full480','baseline100','prompt100_150','postpeak150_220','tail220_400'])+'|')
lines+=['','pure-had输入PSD本身很窄；同Q排序损失还排除了源PSD差≤.005的近等对。源PSD相关系数下降不单独等于粒子标签翻转，也不取消预指定组中心/谷区退化这一更直接的事实。此处pure-had是沉积组成，不能与入射PDG或实测两成分标签混称。',
'',f"累计预算采用保守记账：远端主作业/独立重放{f(native['numeric_cumulative_s'])}s，加既存结果汇总失败与成功{f(supp_failed['elapsed_s']+supp['elapsed_s'])}s，以及缺少psutil的启动失败上界.482192s，合计{f(resource['science_cumulative_s'])}s。首次绘图helper误将validation_trace包装行当作validation本体而KeyError，原native1/日志/源码保留于summary_attempt_001；修复只读取相同保存结果，未触及训练/指标主作业。目录探针plots路径错误原native1也单列保留。",
'','这次新增项并未同时解决完整响应和源条件PSD。下一问题应区分总体分布约束如何与源条件中心/扩散发生冲突，不能按本轮DEV再选系数或重新分源类。本阶段到此封存，等待ROOT另行登记。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf8');print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_array_compute':False,'assembled_existing_results':True}))
