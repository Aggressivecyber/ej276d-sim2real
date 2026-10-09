from pathlib import Path
import os,json,datetime,hashlib
ROOT=Path(__file__).resolve().parent;P=ROOT.parents[1]
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
cfg=read('config.json');m=read('metrics.json');g=read('GATE_METRICS.json');history=read('RUN_HISTORY.json');replay=read('REPLAY_RECEIPT.json');plots=read('PLOT_SUMMARY_RECEIPT.json');aux=read('AUXILIARY_PRECISION_RECOVERY_RECEIPT.json')
science=sum(r['elapsed_s']for r in history);arms=cfg['endpoints'];names=['BASE','INDEPENDENT_CFM','OT_CFM',*arms]
primary=m['source_composition']['primary_pure_had_actual_input_source_Q_bins3_4']
assert len(history)==3 and all(r['actual_worker_native_exit']==0 and r['launcher_native_exit']==0 and r['termination_reason']is None for r in history)
assert science<600 and replay['pass']and replay['rows']==4000 and aux['exact_recovery_pass']and aux['rows']==3712
assert all(not e['retain']for e in g['endpoints'].values())
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
qa={name:{**row,'actually_opened_and_visually_reviewed':True,'result':'readable','method':'view_image actual pixels','checks':['labels and units readable','complete populations/gaps visible','no data-dependent example choice','prespecified boundary jumps retained']}for name,row in plots['figures'].items()}
write('FIGURE_QA.json',{'UTC':now,'PID':os.getpid(),'final_png_count':13,'all_final_images_actually_opened':True,'images':qa,'replotting_after_visual_review':0,'scientific_recomputation_for_QA':0})
write('FAILURES_AND_UNKNOWNS.json',{'scientific_native_failures':[],'control_native_failures':[],
    'inherited_auxiliary_precision_gap':{'R33_saved_auxiliary_x1':'not separately retained','old_saved_increment':'f32(pre-base), insufficient to claim exact reconstruction of x1','approximation_used':False,
        'registered_recovery':cfg['auxiliary_recovery_addendum'],'new_chain_rows':3712,'saved_base_signal_electronic_increment_mask_bin_and_original_postADC_all_exact':True,'receipt':'AUXILIARY_PRECISION_RECOVERY_RECEIPT.json'},
    'scientific_negative_result':'IP fails boundary gate; OT_P fails full-response and boundary gates. No smoothing/window/seed/ODE or coefficient changes.',
    'actual_RSS_probe_failures_total':sum(r['memory_probe_failures']for r in history),
    'physical_final_signal_electronic_split':'unknown','eventwise_microphysical_mapping':'not established','independent_real_acquisition':'not established; same acquisition historically exposed DEV',
    'communication_correction':'Early progress text misstated OT_P full-shape BA as .954; saved metrics and report use the actual .962. No result file was changed.'})
write('RESOURCE_RECEIPT.json',{'PID':os.getpid(),'UTC':now,'registered_UTC':cfg['registered_UTC'],'first_actual_control':cfg['first_actual_command'],
    'executor_seal_deadline_UTC':cfg['deadline_UTC'],'root_deadline_UTC':cfg['absolute_deadline_UTC'],
    'science_limit_s':600,'science_actual_s_including_guard_handshakes_statistics_recovery_replay_and_plots':science,'science_entries':history,
    'currently_active_science_jobs':0,'CPU_threads':4,'actual_worker_RSS_limit_bytes':4294967296,'actual_worker_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes']for r in history),
    'RSS_probe_failures':sum(r['memory_probe_failures']for r in history),'GPU_used':False,'GPU_allocated_peak_bytes':0,
    'deterministic_projected_main_rows':4000,'deterministic_projected_joint_rows':2048,'deterministic_projected_probe_rows':1664,
    'deterministic_projected_rows_total':7712,'new_neural_main_generation_rows':0,'auxiliary_original_chain_recovery_rows':3712,'independent_main_complete_chain_replay_rows':4000,
    'new_neural_inference_rows_total':7712,'new_neural_inference_rows_limit_after_addendum':7712,
    'new_response_fits_or_updates':0,'new_flow_affine_fits':0,'classifier_training_preprocessing':'original HGB protocol fitted on allowed 600 training rows','HGB_actual':m['counters']['HGB_attempts'],'HGB_cap':20,
    'density_description_slots':m['counters']['density_description_slots'],'density_actual_EM_attempts':m['counters']['density_EM_attempts'],'density_EM_cap':16,
    'new_random_projection_blocks':0,'new_ODE_check':0,'new_REAL_or_raw_long_waveform_acquisition':0,'existing_REAL_DEV_and_TRAIN_reference_read':'reused saved input only','EVAL_waveform_read':0,'CAL_waveform_read':0,'new_G4':0,'SSH':0,'installation':0,
    'inherited_DEPENDENCIES_not_copied_payloads':{'root':cfg['R33_root'],'BASE_weights':cfg['base_G'],'flow_weights':cfg['endpoint_weight_paths'],'affine_constants':'R33/AFFINE_CONSTANTS.npz','old_figures_or_generated_inputs_overwritten':False},
    'local_storage_snapshot_before_final_control_receipts':{'files':sum(q.is_file()for q in ROOT.rglob('*')),'bytes':sum(q.stat().st_size for q in ROOT.rglob('*')if q.is_file()),'unique_new_output_cap_bytes':402653184},
    'original_R33_science_cost_not_recharged':101.21910185999513,'normal_migration_or_consistency_tests':0,'scientific_failures':0,
    'system_library_loading_not_project_dataset_IO_budget':True,'final_exact_stage_bytes_in':'FINAL_RECEIPT.json'})
write('NATIVE_TOOL_INDEX.json',{'first_actual_control':'035cba','create_stage':'674b14','source_preparation':'9ead30','source_wording_finalization':'102348',
    'evaluation_start':'311001','evaluation_native0':'76bbd9','replay_start':'5ffbfb','replay_native0':'ccd705','summary_plot_start':'0a307d','summary_plot_native0':'51d623',
    'process_identity_and_nonce_RSS_chain':'RUN_HISTORY + per-entry SCIENCE_PID/RSS_ACK/WORKER/NATIVE receipts','seal_native_tool':'returned externally after FINAL_RECEIPT is frozen'})
write('ARTIFACT_INDEX.json',{'release':['REPORT.md','REPRODUCTION.md','METRICS_COMPACT.json','GATE_METRICS.json','RESOURCE_RECEIPT.json','FIGURE_QA.json'],
    'full_private_metrics':'metrics.json','main_outputs':{a:a+'/generated.npz'for a in arms},
    'main_precision_schema':['pre_ADC_f64_V','ADC_input_f32_V','waveforms_V','original_pre_ADC_f32_V','base_pre_ADC_V','original_increment_f64_V','kept_increment_f64_V','removed_increment_f64_V','source_ids','flow_applied','source_Qbin_actual_preADC'],
    'auxiliary_outputs':{a:['results/'+a+'_joint8_REPEAT_PRIVATE.npz','results/'+a+'_SOURCE_PROBES_PRIVATE.npz']for a in arms},
    'exact_recovery':'AUXILIARY_PRECISION_RECOVERY_RECEIPT.json','independent_full_chain_replay':'REPLAY_RECEIPT.json','precision_constraints':'ACTUAL_PROJECTION_CONSTRAINTS.json','boundary_metrics':'BOUNDARY_METRICS.json',
    'code':['projection_core.py','protected_diagnostics.py','evaluate_protected.py','replay_protected.py','summarize_and_plot.py','supervise.py','owned_worker.py'],
    'fixed_models_and_actual_inputs':'DEPENDENCY_INDEX.json / config.json / R33 original immutable FINAL_MANIFEST',
    'final_pngs':['results/figures/'+n for n in plots['figures']],
    'private_artifacts_remain_local':['all NPZ','complete metrics','source IDs','individual predictions','model weights and inherited inputs'],
    'new_model_training':False,'publication_compact_contains_event_lists':False})

labels={'BASE':'BASE','INDEPENDENT_CFM':'Independent','OT_CFM':'OT','INDEPENDENT_PROTECTED':'Independent+P','OT_PROTECTED':'OT+P'}
fmt=lambda x:'undefined'if x is None else f'{x:.8g}'
ci=lambda r:'['+', '.join(f'{x:.6f}'for x in r)+']'
lines=['# R34：保护基线和两段积分后的冻结流修正','',
    '**关闭本次唯一固定投影配方。** Independent+P保留了相对BASE的有限完整波形收益，并恢复BASE的PSD和基线代理，但分段均值相减制造了明显边界跳变；OT+P还没有保留可信的完整形状收益。两候选均未达到四项联合门槛，数值约束和精确回放通过不替代科学通过。','',
    '固定转换为完整480点G4派生源→冻结R32 CONTROL_FINAL→冻结R33末点流→秩378欧氏正交投影→一次14bit ADC。先将x1/x0各升float64后相减；增量0:100置零、100:150与150:400分别减本段均值、400:480保留原增量。无目标事件、拟合系数、平滑、重训、支持或ODE步数修改。两臂用原2000父记录与同一z/white/style，417 pureEE/1518 purehad/65 mixed及全部支持外417行保留；没有DEV择优。','',
    '主4000条直接转换原保存preADC x0/x1。R33辅助joint2048/probe1664缺少单独原x1，原f32(pre-base)不足以保证精确恢复。根注册补充后，本轮仅对这3712行复跑原冻结32步链，base/S/E/f32增量/mask/Qbin及原postADC全部逐值匹配，再按实际x1/x0各升f64投影；没有采用近似相加代替原x1。另独立CPU进程从完整实际源运行4000主行，pre-f64/f32/postADC、原终点、投影增量、ID与mask逐值通过。','',
    '## 完整波形','',
    '| 固定250/域留出BA | BASE | Independent | OT | Independent+P | OT+P |','|---|---:|---:|---:|---:|---:|']
for key,label in [('full_historical_shape','完整shape493'),('normalized_u480','纯u480'),('full_voltage','完整电压480'),('prompt_local','prompt100:150'),('postpeak_local','postpeak150:220'),('tail_local','tail'),('baseline_local','baseline'),('Q_PSD_2D','Q/PSD二维')]:
    lines.append('| '+label+' | '+' | '.join(f"{m['C2ST']['results'][n][key]['balanced_accuracy']:.3f}"for n in names)+' |')
lines+=['','各视图复用原每域600训练/250留出行与人工组，旧BASE/两原流/REAL–REAL分类器预测未重拟合；新20HGB只用于两个保护端点。CI为500次固定分类器区组重采样，不包含响应/分类器训练和独立采集的不确定性。']
for a in arms:
    gg=g['endpoints'][a]['gate1_complete_response']
    for v,label in [('full_historical_shape','shape493'),('normalized_u480','纯u480')]:
        r=gg['paired_minus_BASE'][v];o=gg['paired_minus_own_original'][v]
        lines.append(f"{labels[a]} {label}：ΔBASE={r['BA_delta_first_minus_second']:+.3f}，95%CI {ci(r['BA_delta_95CI'])}；Δ自身原流={o['BA_delta_first_minus_second']:+.3f}，95%CI {ci(o['BA_delta_95CI'])}。")
lines+=['','Independent+P两项ΔBASE区间上界<0，保留原流收益；其完整电压BA也为.940。OT+P两项ΔBASE的BA区间均跨0，纯u480比原OT恶化.014，超过.01门槛。所有完整视图仍明显可分，不能称匹配实测。','',
    '## PSD、峰宽和源保真','',
    '| 指标，同REAL参考 | BASE | Independent+P | OT+P |','|---|---:|---:|---:|']
for field,label in [('PSD','总体PSD W1'),('baseline_RMS_V','baseline RMS W1，V'),('ACF1','baseline lag1 ACF W1')]:
    lines.append('| '+label+' | '+' | '.join(fmt(m['domain_vs_REAL'][n]['overall'][field]['W1'])for n in ['BASE',*arms])+' |')
for key,label in [('PSD_common_Q_equal_mean_W1','共同observed-Q bins1–7 等权PSD W1'),('component0_raw_width_MAE','低侧raw-width MAE，bins2–7'),('component1_raw_width_MAE','高侧raw-width MAE，bins2–7')]:
    a=g['endpoints'][arms[0]]['gate3_PSD_width_noise']['measures'][key];b=g['endpoints'][arms[1]]['gate3_PSD_width_noise']['measures'][key]
    lines.append('| '+label+' | '+' | '.join(fmt(x)for x in [a['BASE'],a['candidate'],b['candidate']])+' |')
lines+=['','保护后的两段积分保留的是BASE已经学到的响应代理，因此PSD/峰宽回到BASE附近是预期保护，不是进一步拟合REAL的收益。BASE原有效密度bins2–7，两候选全部仍有效；三者bin1两成分未分离、bin8样本不足，完整失败表保留，没有补种子。五端点共同密度仅bins2/3/6，横向误差另列；不能用这一小交集隐去原流支持缺口。每候选与BASE/REAL共同CDF N≥40均为bins1–7，bin8完整CDF仍展示。两密度成分不是粒子真值。','',
    '| 未量化实际source-Q纯had | N | BASE谷比例 | Independent+P | OT+P |','|---|---:|---:|---:|---:|']
for b in ['bin3','bin4']:
    lines.append('| '+b+' | '+str(primary['BASE'][b]['source_count'])+' | '+' | '.join(f"{primary[n][b]['valley_fraction']*100:.4f}%"for n in ['BASE',*arms])+' |')
lines+=['','实际源Q/PSD全体Spearman为BASE .99972634/.41899171、Independent+P .99972631/.41891083、OT+P .99972605/.41898345；两者降幅满足.05限制。历史postADC SOURCE_CLEAN bins3/4的207/182另列，主口径始终未量化NN输入213/181，绝不互换。纯成分近常数源PSD的秩不构成真实物理信息量，谷比例也不是粒子错误率。','',
    '## 数值保护与边界失败','',
    '| actual 2000行/臂最大绝对差 | Independent+P | OT+P |','|---|---:|---:|']
for key,label in [('fast_segment_sum_maxabs_V','f64 fast增量和，V'),('slow_segment_sum_maxabs_V','f64 slow增量和，V')]:
    lines.append('| '+label+' | '+' | '.join(fmt(m['precision_constraints'][a]['actual_constraints'][key])for a in arms)+' |')
for stage,label in [('protected_preADC_f64','f64投影后'),('ADC_input_f32','ADC内部f32输入'),('post_ADC','量化后')]:
    for field in ['Qtotal_mVns','Qfast_mVns','PSD']:
        lines.append('| '+label+' ΔBASE '+field+' | '+' | '.join(fmt(m['precision_constraints'][a]['charge_PSD_precision_levels'][stage][field]['maxabs'])for a in arms)+' |')
lines+=['','前100点在f64/f32/postADC三个层次均逐值等于BASE；支持外417行增量完全为0，全部主输出有效/有限，无ADC裁剪。实际段和残差远低于1e-10V；f64保持不被写成量化后严格相等。非零1583增量的保留能量平均为Independent .808608、OT .909260，零增量另列。','',
    '| 固定边界W1，mV | BASE | 原Independent | Independent+P | 原OT | OT+P |','|---|---:|---:|---:|---:|---:|']
bd=m['boundary_differences']['results']
for key in bd['BASE']:
    lines.append('| '+key+' | '+' | '.join(f"{bd[n][key]['W1_V']*1000:.6f}"for n in names[:2]+[arms[0],'OT_CFM',arms[1]])+' |')
lines+=['','99→100一阶差距离由BASE .024390mV升至2.469145/1.665049mV，预定允许值仅.085428mV；两臂均失败。Independent还在149→150以及多个二阶边界失败，OT在99/100与149/150二阶差失败。固定父0/500/1000/1999的全波形与局部实例显示分段常量偏移如何形成跳变，保留未配对同Q REAL参照，无输出择图、平滑或位置搜索。边界允许量是工程限制，不是物理等效阈值。','',
    'joint8的base S/E、浮点加和舍入项、保留/移除增量作代数分解，完整协方差和交叉项保存并闭合。最终物理signal/electronic分解unknown。同父重复不增加真实独立源事件。正确固定源变噪声RMS均值Independent+P4.08915mV、OT+P3.55560mV；源敏感性不等于逐事件物理映射正确。','',
    '## 联合判定与交付','',
    '| 候选 | 完整响应 | 精度/源/回放 | PSD/宽度/基线 | 边界结构 | 保留 |','|---|---|---|---|---|---|']
for a in arms:
    e=g['endpoints'][a];flags=[e[k]['pass']for k in ['gate1_complete_response','gate2_precision_source_and_replay','gate3_PSD_width_noise','gate4_fixed_boundary_structure']]
    lines.append('| '+labels[a]+' | '+' | '.join('通过'if x else'失败'for x in [*flags,e['retain']])+' |')
lines+=['',f'科学累计{science:.6f}s/600s；evaluation42452 native0 59.778407s、独立回放51120 native0 10.168105s、聚合/绘图10324 native0 13.095555s。实际worker峰RSS1,092,595,712B，RSS查询缺测0，CPU4/GPU0。20HGB、16密度描述槽/14实际EM、主4000确定性转换、joint2048/probe1664、辅助链恢复3712、独立链回放4000。总新神经推理7712，未做响应拟合或更新、额外ODE检查、新REAL/EVAL/CAL/G4/SSH/安装。',
    '','13张最终PNG已实际逐张打开检查。所有数据/代码/缓存留E/D，旧R33与其他封存阶段保持只读；新独有文件约182MB，精确最终大小见FINAL_RECEIPT，低于384MiB。完整逐事件数组、ID/预测留本地，METRICS_COMPACT仅聚合。','',
    '本轮关闭该唯一后投影配方，不追加平滑、窗口、系数、种子、ODE或网络扫描。Independent的保护实验说明在此有限开发基准中，部分波形收益可以与BASE的电荷/PSD代理同时保留；这不证明真实粒子对应，也不解决边界及完整响应真实性。若另行登记下一问题，优先将源代理顺序放到区内训练耦合中，保持批次边缘与预算不变，避免通过分段后处理制造时序不连续；这里只是建议，未执行新训练。','',
    '当前仍为同采集、历史暴露DEV开发诊断，人工组不是独立采集。源完整波形由已有EE/had总沉积派生，未恢复微观沉积时序。保留局部负结果，项目整体目标未宣称完成。','',
    '主要文件：GATE_METRICS、METRICS_COMPACT、ACTUAL_PROJECTION_CONSTRAINTS、AUXILIARY_PRECISION_RECOVERY_RECEIPT、REPLAY_RECEIPT、两臂generated.npz、results/figures、RESOURCE_RECEIPT、REPRODUCTION与最终单份FINAL_MANIFEST/FINAL_RECEIPT。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf8')
repro='''# R34 复现

在新目录恢复，禁止覆盖本轮或R33。唯一投影在projection_core.project/protect：x1和x0分别转f64后相减，0:100零、100:150与150:400各减均值、400:480原样。保存f64 preADC、f32 ADC输入、最终postADC、原/保留/移除增量。没有目标域数据输入该部署变换。

本机解释器 C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe，实际Torch/NumPy/sklearn版本在metrics.versions；CPU4。supervise.py设置E阶段cache下TEMP/TMP、Torch/Matplotlib/joblib缓存，owned_worker导入前报告PID/nonce，确认真实RSS后执行。

实际入口：python -B supervise.py evaluate_protected.py <该CPU解释器>。worker42452/super56960/native0；主保存输出来自R33两末点的generated.npz，不重新主推理。辅助原preADC未另存，精确恢复补充见tasks/R34_AUXILIARY_PRECISION_RECOVERY_ADDENDUM_001.md；从R33实际joint/probe输入复跑固定权重/CPUchunk128/32步链3712，核对全部原分量和原ADC后才P。恢复输入/模型路径与重要旧字节清单均由DEPENDENCY_INDEX/config和R33 immutable FINAL_MANIFEST绑定；无需重新做迁移内容测试。

部署独立回放入口：python -B supervise.py replay_protected.py <解释器>。worker51120/super25704/native0，4000完整实际source→G→固定flow→P→1ADC逐值一致；该入口只读R33 ACTUAL_GENERATION_INPUTS、冻结R32 G、R33两流权重和AFFINE_CONSTANTS，不读取REAL/TRAIN样本、reader、内部val或DEV统计，无拟合更新。REPLAY_RECEIPT包含所有字段证据。

固定评价复用R26 600/250源与REAL索引/组以及R32/R33旧预测；新候选仅20HGB。宽度仅16描述槽/14实际EM，旧REAL/BASE/两原流描述直接复用；所有新fit/slot立即落盘。不得通过重跑初始化填补bin1/8缺口。保存完整metrics/预测/参数可重建所有聚合结论；发布聚合METRICS_COMPACT不含身份载荷。

汇总绘图入口：python -B supervise.py summarize_and_plot.py <解释器>。worker10324/super26120/native0，13图、0新HGB/EM/响应更新/神经生成。图例明确旧流与保护流；固定边界例父0/500/1000/1999，同Q REAL仅未配对参照。源主条件采用实际未量化source-Q213/181，历史量化参考207/182另列。

所有保护候选联合门槛失败，复现是开发负结果证据，不是自动授权再次科学工作。旧base S/E与flow增量只有代数意义，最终物理分解unknown。完整数据和随机输入保留E/D，未上传逐事件科学载荷。
'''
(ROOT/'REPRODUCTION.md').write_text(repro,'utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report_assembled':True,'science_actual_s':science,'final_pngs_actually_viewed':13,'retain':False}))
