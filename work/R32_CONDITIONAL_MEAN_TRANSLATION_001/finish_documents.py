from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
R=Path(__file__).resolve().parent;cfg=json.loads((R/'config.json').read_text('utf8'));m=json.loads((R/'metrics.json').read_text('utf8'));h=json.loads((R/'RUN_HISTORY.json').read_text('utf8'));rep=json.loads((R/'REPLAY_RECEIPT.json').read_text('utf8'));plot=json.loads((R/'PLOT_RECEIPT.json').read_text('utf8'));gate=m['stage_gate'];sem=m['source_semantics']
def write(n,v):(R/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def trcompact(t):
    keys=['arm','updates','D_updates','style_reader_updates','training_s','selected_step','actually_sampled_unique_source','actually_sampled_unique_REAL','actually_consumed_rows','validation_generation_rows','global_gradient_norms','GPU_peak_allocated_bytes','mean_negative_batches','mean_singleton_rows','mean_coefficient','training_stop_reason','selected_using_DEV_or_D','critic_STANDARDIZED_input']
    def val(v):return{k:v[k]for k in ['step','J','J19_weighted','fine_val','original_unweighted_J','weighted_physical_W1','weighted_shape_SW','style_reconstruction_L1','online_D_weighted_BA','D_not_in_J']if k in v}
    return{**{k:t[k]for k in keys if k in t},'initial_validation':val(t['initial_validation']),'selected_validation':val(t['selected_validation']),'final_validation':val(t['final_validation']),'validation_trajectory':[val(x['validation'])for x in t['validation_trace']]}
compact={'stage':R.name,'status':'closed_negative_fixed_coefficient_mean_recipe','training':{n:trcompact(t)for n,t in m['training'].items()},'data_roles':m['data_roles'],'counters':m['counters'],'domains':m['domains'],'domain_vs_REAL':m['domain_vs_REAL'],'C2ST':{k:m['C2ST'][k]for k in ['results','paired_MEAN_minus_CONTROL','REAL_REAL_reference','train_each_domain','holdout_each_domain','fixed_classifier_CI_not_training_uncertainty']},'PSD_main':m['PSD_main'],'energy_conditional_width':m['energy_conditional_width'],'conditional_mean_SD':m['conditional_mean_SD'],'projection_coverage':m['projection_coverage'],'projection_common_support':m['projection_distances']['common_support_summary'],'source_composition_reference_Q_primary':m['source_composition']['primary_pure_had_source_bins3_4'],'source_association':m['source_association'],'actual_NN_input_source_Q_final_pure_had':sem['actual_NN_input_source_Q_endpoints'],'source_semantics_summary':{k:sem[k]for k in ['parent_ID_order_checks','composition_counts','reference_reconstruction_check','ADC','source_ADC_clipped_parent_count','source_waveform_difference','charge_difference_post_ADC_minus_actual_input_mVns','PSD_difference_post_ADC_minus_actual_input','source_Q_bin_changed_parent_count','pure_had_actual_preADC_counts_8bins','pure_had_reference_postADC_counts_8bins']},'random_decomposition':{n:{'parent_count':v['groups']['joint8']['parent_count'],'draws_each':v['groups']['joint8']['draws_each'],'by_composition':v['groups']['joint8']['by_composition'],'repeats_not_independent_G4_events':True}for n,v in m['random_decomposition'].items()},'source_probe_summary':{n:v['rows']for n,v in m['source_probes'].items()},'gate':gate,'generation_QC_ADC':{n:{'rows':v['rows'],'ADC':v['ADC'],'all_original_source_ids_preserved':v['all_original_source_ids_preserved']}for n,v in m['generation'].items()},'same2950_pairing':{'both2950':m['row_exposure']['both_2950'],'original_flow_all_checks':all(m['row_exposure']['same_stream_hashes'].values()),'complete_source_parent_sequence_preserved':True},'replay':{'pass':rep['pass'],'rows':rep['rows'],'bitwise_pre_post_signal_electronic':True,'ID_order_equal':True},'versions':m['versions'],'EVAL_read':0,'no_new_G4':True,'private_payloads_omitted':True}
write('METRICS_COMPACT.json',compact)
native_tools={'pid_probe.py':'24bb8b','train_server.py':'1604dc -> 61d017','control_failure_001':'4e00bd','evaluate_endpoints.py':'bf9921 -> 5619cb','inspect_source_summary.py':'d7a5ad','assemble_saved_results.py':'43323b -> b9cf3a','source_semantics_and_gate.py':'cf2a74','replay_endpoints.py':'5eb035','render_figures.py':'6d281e'}
controlerrors=[{'tool':'b3517d','native_exit':1,'actual_PID':None,'PID_status':'unknown_not_printed','elapsed_tool_s':.3691032,'reason':'control read requested nonexistent numeric_1 receipt paths; actual receipt is entry-prefixed at stage root','science_arrays_or_fits':0},{'tool':'d4bd5e','native_exit':1,'actual_PID':None,'PID_status':'unknown_not_printed','elapsed_tool_s':.3082034,'reason':'pure JSON query used domains.SOURCE_CLEAN.overall instead of regions.overall; no stage output was modified','science_arrays_or_fits':0}]
write('CONTROL_READ_ERRORS.json',{'errors':controlerrors,'missing_source_lookup_warnings':{'tool':'3a9d3b','native_exit':0,'missing':['R18/data_support.py','R21/prepare_pool.py','R21/evaluate_model.py'],'actual_R18_numeric_worker_source_read':True},'first_summary_failure_preserved':True})
native=[{**r,'native_tool':native_tools[r['entry']]}for r in h];write('NATIVE_CHAIN.json',{'science_jobs':native,'control_read_errors':controlerrors,'original_receipts_unmodified':True,'all_after_failure_jobs_are_summaries_semantic_checks_replay_plotting_not_refits':True})
train=m['training'];rss=max(r.get('actual_worker_RSS_peak_bytes')or 0 for r in h);size=sum(p.stat().st_size for p in R.rglob('*')if p.is_file());upload=json.loads((R/'TRANSFER_RECEIPT.json').read_text('utf8'));download=json.loads((R/'RETURN_TRANSFER_RECEIPT.json').read_text('utf8'))
res={'registered_UTC':cfg['registered_UTC'],'executor_seal_deadline_UTC':cfg['deadline_UTC'],'first_actual_command':cfg['first_actual_command'],'science_elapsed_including_failures_s':sum(r['elapsed_s']for r in h),'science_limit_s':2400,'training_body_sum_s':sum(t['training_s']for t in train.values()),'training_limit_total_s':1800,'individual_training_s':{n:t['training_s']for n,t in train.items()},'scientific_jobs':native,'actual_worker_RSS_peak_bytes':rss,'RSS_limit_bytes':4294967296,'GPU_allocated_peak_bytes':max(t['GPU_peak_allocated_bytes']for t in train.values()),'GPU_allocated_limit_bytes':4294967296,'server_training_memory_query_failures':2,'failure_time_and_cause':'unknown: loop recorded total but no per-failure timestamps; cannot establish exit-boundary attribution','server_training_self_peak_RSS_bytes':2710675456,'observed_and_self_peak_equal':True,'guard_threshold_triggered':False,'CPU_numeric_threads':4,'response_fit_attempts':2,'G_D_reader_updates_each':sum(t['updates']for t in train.values()),'actually_consumed_source_and_REAL_rows_each_arm':94400,'sampled_unique_source_each_arm':3752,'sampled_unique_REAL_each_arm':18739,'internal_validation_generated_source_rows':sum(t['validation_generation_rows']for t in train.values()),'evaluation_counters':m['counters'],'replay_rows':rep['rows'],'represented_replay_rows':rep['represented_endpoint_rows'],'original_failed_evaluation_science_cost_s':73.82232009997824,'transfer_control_failure_conservatively_charged_s':1.1297573,'other_pure_control_query_errors_not_scientific_s':sum(e['elapsed_tool_s']for e in controlerrors),'stage_bytes_at_documentation':size,'stage_files_at_documentation':sum(1 for p in R.rglob('*')if p.is_file()),'scientific_output_limit_bytes':536870912,'transport_packages_counted_in_stage_bytes':True,'reused_readonly_upload_dependency_bytes':32665453,'upload_tar_bytes':32778240,'returned_tar_bytes':28057600,'transfer_native_exit_success':True,'migration_content_test':False,'necessary_own_job_SSH_only':True,'server_science_native_terminal':0,'no_new_REAL_EVAL_CAL_G4_raw_long_decoding':True,'input_ADC_semantic_comparison_not_transfer_test':True,'transfers_receipt_paths':['TRANSFER_RECEIPT.json','RETURN_TRANSFER_RECEIPT.json'],'final_size_is_in_FINAL_MANIFEST_and_FINAL_RECEIPT':True}
write('RESOURCE_RECEIPT.json',res)
qa={'UTC':datetime.now(timezone.utc).isoformat(),'method':'actual view_image inspection of all final PNGs','executor_view_tools':['four image batches after 6d281e render'],'all_16_inspected':True,'figures':{n:{**d,'actual_visual_inspection':'passed','notes':'axes, legends, full480 range or stated scalar distribution, support gaps and source-condition meaning readable'}for n,d in plot['figures'].items()},'semantic_checks':['C2ST short labels prevent overlap','PSD common axes preserve counts; zero outside range each domain','mean/SD show all480 points; bin8 low-support dotted','source composition conditions distinguish actual preADC Q from frozen postADC Q','density failures retained as gaps'],'no_layout_repaint_needed':True,'no_additional_fit_generation':True}
write('FIGURE_QA.json',qa)
report=f'''# R32 交付：条件均值有局部收益，完整响应与PSD联合门槛失败

固定 `lambda_mean=1` 配方关闭，不追加系数、种子或步数。两个同服务器2950步末点比较中，共同观察电荷区1–7的完整480点平均波形MSE由0.0001687688降至0.0000876409（−48.07%），但完整493特征与原始u480分类器没有可信的可分性下降；总体PSD距离、高侧PSD宽度误差同时恶化。平均曲线改善不代表总体响应匹配。此结论是已历史暴露的同采集DEV诊断，不能称新独立盲测。

| 2950步末点指标 | 同平台Control | Conditional mean | 结论 |
|---|---:|---:|---|
| shape493 BA | .978 | .976 | Δ−.002，95%CI[−.020018,.016422] |
| raw u480 BA | .978 | .978 | Δ0，95%CI[−.017929,.017956] |
| 共同区平均波形MSE | .0001687688 | .0000876409 | −48.07%；区1–7等权 |
| 总体PSD W1 | .02306546 | .02712657 | +17.61% |
| 共同区PSD W1 | .03476580 | .04228886 | +21.64%；区1–7 |
| 低PSD原宽度MAE | .01039272 | .00905381 | −12.88%；共同密度区2–7 |
| 高PSD原宽度MAE | .01305117 | .01691473 | +29.60%；共同密度区2–7 |
| baseline RMS W1 (V) | .0000193584 | .0000184454 | −4.72% |
| ACF1 W1 | .06423497 | .04776065 | −25.65% |

原J20选择与末点分开。Control所选750步/J=.5610488653，末点J=.5795796514；Mean所选1750步/J=.5375945568，末点J=.5773797035。所选端点shape BA分别.998/.990，raw u480 .996/.988；两端点均仍高度可分。验证不使用DEV，新增均值项不加入J20，在线D不选模型。原验证轨迹与所有同期状态保留。两臂训练各114.0338/110.3142秒，均完成2950次G/D/reader更新，共5900；94400行原多重集及NCE位置/源/目标/旧随机量/style/fresh流一致。同一波形内时间patch是NCE负例，未引入跨事件负例。两臂实际各暴露3752个G4父、18739个REAL TRAIN行；加载数6593/21047不能替代实际暴露数。

唯一新增项使用TRAIN-fit 21047行中的8区均值，区计数1670/2286/2533/2604/2689/2624/2526/1963，2152个范围外TRAIN行不用于区均值；599内部验证排除。目标行身份为既有TRAIN容器的row ordinal，当前容器未显式存原始记录索引，既有来源链保留。全480点峰归一化及shape480 TRAIN尺度不变。有限批U负值24次原样保留，混合批singleton贡献0并记1行。小张量显式循环/代数/梯度误差≤1.11e−16；这不证明当前随训练历史更新的样本具有理想iid无偏性。

四端点均QC2000/2000、无新输出ADC裁剪，417 pureEE/1518 pure-had/65 mixed父记录一条不删，主比例不重加权。密度共同支持仅区2–7：Control末点及Mean所选的最低区成分未分离；Control所选最高区少数组成分Nw=3<20，其他三个最高区总N<40，保持undefined。不能把两成分当粒子真值，不能删unknown或以宽度缺口自动通过。全部原始/去趋势宽度、CDF、迁移、局部时间结构、源关联与随机分量均保留。

源Q语义已明确。实际NN输入是未ADC的G4派生解析clean480；历史SOURCE_CLEAN评价参考是对它执行14bit ADC后保存的波形。R26与R28两份参考均逐值等于ADC(actual_input)，maxabs0且父ID次序相同，源ADC无裁剪。纯量化最大波形差30.587μV、逐行RMS均值21.556μV，100:400扣基线电荷均值差(postADC−input)=−8.57409mVns，PSD均值差−.002628；23父事件换8区。元数据Qsource_Vns来自解析full512目标，不等于裁窗后固定积分窗电荷，后者差异来自明确不同的积分对象。没有发现训练输入被更换或父记录改变。

| pure-had两末点 | 条件 | N(bin3/bin4) | Control中心/SD | Mean中心/SD | 谷比例Control→Mean |
|---|---|---|---|---|---|
| bin3 | 实际NN输入preADC Q | 213 | .281402/.026703 | .289707/.018991 | 5.6338%→.4695% |
| bin4 | 实际NN输入preADC Q | 181 | .280560/.019818 | .293087/.014082 | .5525%→0 |

历史postADC参考Q分组207/182也保留用于原门槛，谷比例5.7971%→.4831%和.5495%→0。实际输入Q与参考Q不能混称；这里的经验SD没有新拟合，也不是粒子类别保真真值。完整clean来自总EE/had沉积与既定解析响应，没有新增微观时序。源交换非零敏感性与同父随机扩散不证明正确物理对应，signal也不等于条件期望。所有实测来自一次采集，人工50行分组不是独立采集。

首次评价用实际preADC输入构造SOURCE_CLEAN，触发207/182断言。原失败(40116/native1)、73.8223秒费用、源码/日志与原错误source-Q描述保留。修复仅恢复既有postADC参考、重新组装已保存输出/40个分类器/32个描述槽(实际29次EM)，没有第二次拟合、生成或训练。观察output-Q均值/SD描述逐字段确认不变；实际输入source-Q描述另外补列。另有一次打包控制占位符SyntaxError(原PID未打印，unknown)、1.1298秒计费，修复后正常返回，无迁移一致性测试。纯控制读取路径/JSON字段错误另列CONTROL_READ_ERRORS，不改写原生退出码。

Linux真实worker23246/super23245，native0，227.9446秒、RSS峰2710675456B，同自报峰一致；2次内存查询失败的精确时间/原因unknown，不能写成0或推断为退出边界。GPU allocated峰276043264B。科学累计{sum(r['elapsed_s']for r in h):.6f}/2400秒，包含首轮失败、语义统计、恢复、绘图与回放；单作业、CPU4、RSS/GPU均未触发4GiB守护。独立本机CPU重放8000条，pre/postADC、signal/electronic及父ID次序全部bitwise相同；这是保存模型推理复现，不是跨平台训练逐位复现。16张最终PNG已实际打开检查。训练/评价环境版本差异保留，历史R28仅跨平台参照，主要比较为本轮同服务器两臂。

后续问题应围绕完整条件分布：均值项能压低系统平均残差，却未补齐PSD高侧宽度及细时间联合结构。值得下一轮检验的是保留源电荷/组成关联的条件波动结构与PSD双带分布之间的冲突，而非继续根据均值曲线调大同一系数。本轮不自行追加实验。

关键文件：METRICS_COMPACT.json（发布聚合）、metrics.json（完整本地）、GATE_METRICS.json、SOURCE_SEMANTICS.json、RESOURCE_RECEIPT.json、NATIVE_CHAIN.json、REPLAY_RECEIPT.json、FIGURE_QA.json；模型/逐事件/原训练流仅本地E/D及授权服务器目录。最终字节清单与停止时间见FINAL_MANIFEST/FINAL_RECEIPT。
'''
(R/'REPORT.md').write_text(report,'utf8')
repro=f'''# R32复现入口

实际训练位置为 `{cfg['remote_root']}`，Python `{cfg['server_python']}`（Python3.12.3，NumPy1.26.4，Torch2.12.0+cu130，RTX5070）。先在全新同任务目录准备依赖，再以 `linux_supervise.py train_server.py` 执行。已有目录的一次性token已消费，不在原目录重复启动。DEPLOY/TRANSFER/RETURN回执记录正常传输退出，未做迁移一致性测试；不安装共享环境。

本机CPU入口 `{cfg['CPU_python']}`，`supervise.py` 使用真实PID/nonce导入前握手及RSS守护。环境变量TEMP/TMP指E项目缓存，-B/UTF8，4线程。原 `evaluate_endpoints.py` 已消费且在最终汇总断言处失败；已生成/拟合的数值不重做。`assemble_saved_results.py`仅从保存输出/逐fit/slot组装，`source_semantics_and_gate.py`仅做授权保存数组ADC语义核对及门槛汇总，`render_figures.py`读取保存图数据。

独立主推理 `replay_endpoints.py` 实际已执行（worker37712，native0），8000条pre/postADC/signal/electronic逐值及源ID次序一致，证据在REPLAY_RECEIPT。可在全新恢复目录重放，模型输入为两个臂所选/末点TRANSLATOR.pt、实际ACTUAL_GENERATION_INPUTS的source/z/white/style、原model/legacy_model；推理不需要TRAIN/REAL或reader。不能把新推理写进已封存目录。

PRETRAIN_FREEZE、remote_receipts保存服务器实际执行配置/源码。config的R28_root是在首CPU评价前补的本地历史结果路径，未修改已执行远端配方；原remote_config及freeze保留。继承的旧numeric_wall_limit描述不控制本轮2400科学预算，实际执行字段/RUN_HISTORY/监督器和task为准。源码/配置/权重及科学输入在FINAL_MANIFEST引用。所需复用依赖来自R23 TRAIN/VALIDATION/FINE/critic scaler、R26行流、R21初G/reader、R18初D；11依赖正常上传，原目录只读。CPU Torch2.10.0+cpu/NumPy2.2.6与服务器环境不同，因此历史R28不作新损失的同平台因果对照。

科学公式、角色和固定停止规则见本轮PROTOCOL与任务。不是新盲测，不恢复被输入遗漏的微观沉积时间；不根据DEV重选端点。固定lambda=1门槛失败已关闭，复现不意味着授权扩训练。
'''
(R/'REPRODUCTION.md').write_text(repro,'utf8')
write('DOCUMENTATION_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'science_recomputed':False,'all_final_figures_actual_viewed':16,'science_output_count_and_time_from_saved_native':True,'compact_has_no_event_ID_index_prediction_or_per_parent_rows':True,'report_sha256':sha(R/'REPORT.md'),'compact_sha256':sha(R/'METRICS_COMPACT.json')})
print(json.dumps({'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'documentation_complete':True,'report_bytes':(R/'REPORT.md').stat().st_size,'compact_bytes':(R/'METRICS_COMPACT.json').stat().st_size,'science_s':res['science_elapsed_including_failures_s'],'gate_pass':gate['fixed_recipe_retained']}))
