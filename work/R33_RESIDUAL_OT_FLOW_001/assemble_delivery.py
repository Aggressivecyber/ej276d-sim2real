from __future__ import annotations
import pathlib, json, datetime, os

ROOT = pathlib.Path(__file__).resolve().parent
P = ROOT.parent.parent

def read(name):
    return json.loads((ROOT / name).read_text('utf8'))

def write(name, obj):
    (ROOT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', 'utf8')

m=read('metrics.json'); g=read('GATE_METRICS.json'); h=read('RUN_HISTORY.json'); plots=read('PLOT_SUMMARY_RECEIPT.json')
train=read('TRAINING_NUMERIC_RECEIPT.json'); replay=read('REPLAY_RECEIPT.json')
up=read('TRANSFER_RECEIPT.json'); down=read('RETURN_TRANSFER_RECEIPT.json')
science=sum(r['elapsed_s'] for r in h)
assert science<2400 and m['counters']['HGB_attempts']==20 and m['counters']['density_EM_attempts']==14
assert m['counters']['main_generation_rows']==4000 and replay['pass'] and replay['rows']==4000
assert train['counter']['flow_updates']==5900 and train['counter']['fixed_G_forward_training_rows']==188800
assert plots['figure_count']==11 and not g['retain']
assert all(r['actual_worker_PID'] is not None and r['handshake_before_imports'] for r in h)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
write('FIGURE_QA.json', {'UTC':now, 'reviewer':'executor actual visual inspection with view_image',
    'viewed_final_files':list(plots['figures']), 'figure_bindings':plots['figures'], 'all_11_actually_viewed':True,
    'layout_pass':True, 'scientific_display_checks':['shared Q/PSD axes retain complete populations','8-bin density failures are visible gaps',
        'observed-Q and actual-source-Q labels are distinct','raw480 waveform voltage is not smoothed or independently normalized',
        'fixed-classifier CI and fixed direction ranges have different labels','algebraic covariance plot retains signed cross terms'],
    'root_review':'Root reported actual viewing of six key figures; no new layout changes requested',
    'new_scientific_fit_or_generation':0})
write('FAILURES_AND_UNKNOWNS.json', {'scientific_failure001': {'entry':'evaluate_endpoints.py','native_exit':1,'actual_worker_PID':47072,
    'elapsed_s':h[1]['elapsed_s'],'native_tool_chunk':'2b805f','stage':'module import before data/fit/generation',
    'error':"ModuleNotFoundError: random_diagnostics", 'counters_at_failure':{'HGB':0,'EM':0,'new_main_generation':0},
    'repair':'Copied the necessary existing read-only R32 helper and used attempt002 with unchanged evaluator mathematics',
    'original_source_logs_receipts_preserved':True},
    'control_schema_query':{'tool_chunk':'0816d4','error':"KeyError: bin_id_1based (the saved density slice uses zero-based bin)",
        'role':'read-only schema inspection; no scientific job, fit, array inference or result write',
        'shell_native_exit':0,'embedded_python_exit_not_separately_preserved':'unknown','repair':'used the existing bin field'},
    'remote_RSS_query': {'failures':h[0]['memory_probe_failures'], 'events':h[0]['memory_probe_failure_events'],
        'cause':'unknown; timestamp was near worker exit, but causal location is not proven',
        'RSS_peak_valid_measured_bytes':h[0]['actual_worker_RSS_peak_bytes'],
        'worker_self_peak_RSS_bytes':h[0]['worker_self_receipt']['self_peak_RSS_bytes']},
    'scientific_negative_result':'fixed OT-CFM gate2/3/4 fail; not treated as an implementation error',
    'density_gaps':g['gate3_PSD_width_noise_guardrails']['density_support_gaps'],
    'physical_final_signal_electronic_split':'unknown','microphysical_source_timing':'not present in existing totalEE/had-derived source',
    'independent_acquisition_validation':'not established; this is historically exposed single-acquisition DEV'})
artifact_files=[f for f in ROOT.rglob('*') if f.is_file()]
local_bytes=sum(f.stat().st_size for f in artifact_files)
remote_return_files=sum(b for _,b in down['remote_outputs']['files'])
# Normal deployment creates one uploaded archive and its extracted payload. This is
# a storage bound using recorded transfers, not a rehash/consistency test of migration.
remote_known_bound=up['upload_tar_bytes']*2+remote_return_files+down['return_tar_bytes']
resource={'UTC':now,'PID':os.getpid(),'registration_UTC':'2026-10-09T17:52:39.460916+00:00',
    'first_actual_control':{'UTC':'2026-10-09T17:53:14.399903+00:00','PID':51776,'tool_chunk':'63fd7e','native_exit':0},
    'executor_seal_deadline_UTC':'2026-10-09T19:42:39.460916+00:00','science_limit_s':2400,'science_actual_s_including_failure_and_plots':science,
    'science_entries':h,'science_jobs_currently_active':0,'CPU_threads':4,'RSS_limit_bytes':4294967296,
    'actual_worker_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes'] for r in h),
    'GPU_allocated_peak_bytes':train['GPU_peak_allocated_bytes'],'GPU_allocated_limit_bytes':4294967296,
    'actual_remote_device':train['device'],'remote_versions':read('remote_receipts/SERVER_ENVIRONMENT.json'),
    'local_evaluation_versions':m['versions'],'fixed_G_training_forward_rows':188800,'fixed_G_D_reader_updates':0,
    'new_flow_fit_attempts':2,'new_flow_updates_per_arm':2950,'new_flow_updates_total':5900,
    'new_HGB':20,'new_density_description_slots':16,'new_density_EM':14,
    'new_main_rows':4000,'joint8_rows':2048,'source_probe_rows':1664,'solver_check64_rows':64,
    'independent_replay_rows':4000,'EVAL_rows':0,'CAL_waveform_decode':0,'new_REAL':0,'raw_long_decode':0,'new_G4':0,'installs':0,
    'storage_snapshot_before_final_control_documents':{'local_files':len(artifact_files),'local_bytes':local_bytes,
        'remote_returned_regular_file_bytes':remote_return_files,'remote_known_artifact_storage_bound_bytes':remote_known_bound,
        'combined_local_plus_remote_known_bound_bytes':local_bytes+remote_known_bound,'cap_bytes':536870912,
        'caveat':'snapshot and bound; transient process/system caches are not measured as scientific artifacts; final local bytes in FINAL_RECEIPT'},
    'upload_archive_bytes':up['upload_tar_bytes'],'return_archive_bytes':down['return_tar_bytes'],
    'normal_transfer_native_exits':{'create':up['create']['native_exit'],'upload':up['transfer']['native_exit'],
        'extract':up['extract']['native_exit'],'return_pack':down['remote_pack_native_exit'],'return_scp':down['scp_native_exit']},
    'migration_content_tests':0,'failures_retained':['evaluate_endpoints.py native1','one remote RSS query miss','read-only schema query'],
    'query_missing_points_not_reported_as_zero':True,'scientific_generation_replay_complete':True,
    'control_assembly_hashing_not_model_or_scientific_fit':True}
assert local_bytes+remote_known_bound+1048576<536870912
write('RESOURCE_RECEIPT.json', resource)
write('NATIVE_TOOL_INDEX.json', {'first_control':'63fd7e','resource_preflight':'b380be','preparation':'162061','deployment':'486836',
    'remote_training_start':'14a1cc','remote_training_native0':'c7793e','normal_return':'414036',
    'evaluation_failure_native1':'2b805f','evaluation_success_native0':'e4524b','replay_native0':'d76b1c','summary_plot_native0':'4de570',
    'note':'Actual process identity, elapsed time, RSS, nonce and native exit are also in RUN_HISTORY and per-entry receipts; final seal tool is returned after this file is frozen'})
write('ARTIFACT_INDEX.json', {'aggregate_release':['REPORT.md','REPRODUCTION.md','METRICS_COMPACT.json','GATE_METRICS.json','RESOURCE_RECEIPT.json','FIGURE_QA.json'],
    'complete_private_results':['metrics.json','results/FEATURES_PRIVATE.npz','results/C2ST_PRIVATE.npz','results/TEMPORAL_DIAGNOSTICS_PRIVATE.npz',
        'results/PSD_SECTIONS_PRIVATE.npz','results/MEAN_SD_DESCRIPTIONS_PRIVATE.npz','results/RAW_VOLTAGE_MOMENTS_PRIVATE.npz'],
    'trained_models':{a:a+'/FINAL_FLOW.pt'for a in ['INDEPENDENT_CFM','OT_CFM']},
    'frozen_base':str(P/'work/R32_CONDITIONAL_MEAN_TRANSLATION_001/SERVER_FRESH_CONTROL/FINAL_TRANSLATOR.pt'),
    'main_outputs':{a:{'file':a+'/generated.npz','rows':2000,'points_per_row':480,
        'schema':['waveforms_V','pre_ADC_V','base_pre_ADC_V','base_signal_V','base_electronic_V','flow_increment_V','source_ids','flow_applied','source_Qbin_actual_preADC']}for a in ['INDEPENDENT_CFM','OT_CFM']},
    'actual_source_and_rng':'ACTUAL_GENERATION_INPUTS.npz; both models use exactly the same saved parent/source/z-white/style inputs',
    'affine':'AFFINE_CONSTANTS.npz + AFFINE_METADATA.json','training_actual_pair':'PAIRED_FLOW_RECEIPT.json + each arm ACTUAL_CFM_STREAM_PRIVATE.npz',
    'replay':'REPLAY_RECEIPT.json','solver':'SOLVER_CHECK.json','figures':list(plots['figures']),
    'publication_privacy':'Keep IDs, individual predictions, streams, all NPZ/PT/TAR and complete metrics local; compact metrics contain aggregate values only'})

R=lambda x:f'{x:.8f}'
C=lambda r:'['+','.join(f'{x:.5f}'for x in r)+']'
source=g['gate4_source_guardrails']['pure_had_valley'];ranks=g['gate4_source_guardrails']['whole_population_rank'];measures=g['gate3_PSD_width_noise_guardrails']['measures']
lines=[
'# R33：固定源驱动响应后的无配对全波形流',
'',
'本轮关闭固定 OT-CFM 配方。两种流相对冻结基线降低了完整波形的分类可分性，但仍明显区别于实测；OT 没有达到优于独立耦合的预定要求，同时损害了条件 PSD、低侧峰宽、基线涨落和纯 had 源条件结构。求解精度和独立重放均通过，因此这些负结果不能通过重试训练或改变评价规则消除。',
'',
'固定链路为完整 G4 派生 480 点源波形 → R32 CONTROL_FINAL 冻结网络 → 480 维条件流 → 一次 14bit ADC。新流不从独立目标密度采样；每条输出保留实际源父记录。两臂只有同一窄 Q 区内的训练耦合不同：独立目标置换或最小均方标准化电压代价的 OT 一对一匹配。统计耦合没有被解释成物理事件配对。',
'',
'两臂均用相同初态、相同 2950×32 训练行、同 t、source/REAL/bin/z/white/style 与冻结 G 输出；各更新 2950 次，只发布实际末点，没有用 DEV 选 checkpoint。REAL TRAIN-fit载入21047，其中8区内18895行用于分区原始电压均值/尺度，2152行在8区外；原599内部验证不参与本轮拟合。SD 用 ddof0、floor=1e-6V，8区无坐标触 floor。主 DEV 为原2000父记录与原2000 REAL；1583父输入在源 Q 支持内使用流，417行支持外保留冻结基线并标记。全体人口不删、不重加权。',
'',
'## 完整分布与配对比较',
'',
'| 固定留出视图 BA（越低越接近不可分） | BASE | Independent | OT |',
'|---|---:|---:|---:|']
for key,label in [('full_historical_shape','完整shape493'),('normalized_u480','纯u480'),('full_voltage','完整电压480'),('prompt_local','prompt100:150'),('postpeak_local','postpeak150:220'),('tail_local','tail'),('baseline_local','baseline'),('Q_PSD_2D','Q/PSD二维')]:
    lines.append('| '+label+' | '+' | '.join(f"{m['C2ST']['results'][n][key]['balanced_accuracy']:.3f}"for n in ['BASE','INDEPENDENT_CFM','OT_CFM'])+' |')
lines+=['','每域固定600训练/250留出行和原人工组；区间为500次固定分类器的区组重采样，不包含响应训练或分类器训练的不确定性。旧 BASE 和 REAL–REAL 预测直接复用，20个新HGB只评价新两臂。']
for key,label in [('full_historical_shape','完整shape493'),('normalized_u480','纯u480')]:
    b=g['gate1_full_response_vs_BASE']['paired_OT_minus_BASE'][key];i=g['gate2_OT_over_INDEPENDENT']['paired_OT_minus_INDEPENDENT'][key]
    lines.append(f"{label}：OT−BASE ΔBA={b['BA_delta_first_minus_second']:.3f}，95%CI {C(b['BA_delta_95CI'])}；OT−Independent ΔBA={i['BA_delta_first_minus_second']:.3f}，95%CI {C(i['BA_delta_95CI'])}。")
lines+=['','两项 OT−BASE 区间上界均小于0；但 OT 完整shape点值高于 Independent，两项 OT−Independent 区间均含0。因此不能支持 OT 耦合的完整响应增益。局部峰后/尾部收益仍保留，不能替代总体成功。',
'','## PSD、峰宽与噪声',
'','| 保存指标（同一 REAL 参考） | BASE | Independent | OT |','|---|---:|---:|---:|']
for field,label in [('PSD_overall_W1','全人口PSD W1'),('PSD_common_Q_equal_mean_W1','共同observed-Q bins1–7 等权PSD W1'),('component0_raw_width_MAE','低侧raw-width MAE，共同bins2/3/6'),('component1_raw_width_MAE','高侧raw-width MAE，共同bins2/3/6'),('baseline_RMS_V_overall_W1','baseline RMS W1，V'),('ACF1_overall_W1','baseline lag1 ACF W1')]:
    values=measures[field]['values'];lines.append('| '+label+' | '+' | '.join(R(values[n])for n in ['BASE','INDEPENDENT_CFM','OT_CFM'])+' |')
lines+=['','OT 全人口 PSD W1 较 BASE 下降5.72%，共同电荷区 W1 却增加29.59%；低侧 raw-width MAE 增加32.23%，baseline RMS/ACF W1增加17.23%/32.35%。Independent 的区内 PSD CDF/高侧宽度有收益，但其全人口PSD W1更差，源PSD秩关联明显下降。人口边际、条件分布和源关联不能互相替代。',
'','CDF共同门槛明确要求所有域每区N≥40，故bin8不纳入共同距离，原bin8完整CDF仍展示。两成分密度共同支持只有bins2/3/6：BASE bin1未分离、bin8样本不足；Independent bins4/5未在500更新内收敛、bin7成分支持不足、bin8不足；OT bin1未收敛、bin8不足。所有缺口保留，未补拟合或改初始化；两成分不称粒子真值。',
'','完整480的条件均值仍有收益：同observed-Q bins1–7原始电压均值MSE等权为BASE 3.89580e-6、Independent 8.54863e-7、OT 3.62813e-6 V²；全人口对应1.15289e-4/1.12764e-4/1.34597e-4 V²。归一化u的同区均值MSE为1.68769e-4/5.15573e-5/8.27391e-5。条件均值靠近并未消除峰后时间结构、SD、PSD和完整C2ST差距。',
'','## 同一实际源输入下的保真约束',
'','主条件始终来自 ACTUAL_GENERATION_INPUTS 的未量化NN输入；BASE也用同一源Q/PSD重算描述。历史 postADC SOURCE_CLEAN 的207/182只是另列参考，绝不用于主delta。',
'','| actual-input pure had 源Q区 | N | BASE谷比例 | Independent谷比例 | OT谷比例 | OT−BASE百分点 |','|---|---:|---:|---:|---:|---:|']
for r in source:
    v=r['values'];lines.append(f"| bin{r['source_bin_1based']} | {r['source_N']} | {100*v['BASE']:.3f}% | {100*v['INDEPENDENT_CFM']:.3f}% | {100*v['OT_CFM']:.3f}% | {100*r['OT_minus_BASE_fraction']:+.3f} |")
lines+=['','| 全2000父记录Spearman（同未量化输入） | BASE | Independent | OT |','|---|---:|---:|---:|']
for r in ranks:lines.append('| '+r['measure']+' | '+' | '.join(f"{r['values'][n]:.6f}"for n in ['BASE','INDEPENDENT_CFM','OT_CFM'])+' |')
lines+=['','OT 的两个秩降幅均在0.05限内，但两个纯had谷比例增量超过2百分点。Independent 的源PSD秩从0.419降至0.138。原417 pureEE/1518 purehad/65 mixed父记录和ID顺序保留，所有新主输出QC有效、无新增非有限或ADC裁剪；这不证明逐事件内容正确。谷区是冻结统计参照，不直接等于粒子内容错误。PDG推定来自已绑定CSV仅22/2112及旧标签规则，区别于每条显式保存PDG。',
'','正确固定源变噪声的波形RMS均值为Independent 5.1503mV、OT 3.6569mV；换源保持噪声为14.7574/14.3633mV。源敏感性不是正确物理对应。joint128×8的base_signal/base_electronic/flow_increment是代数分解，保留交叉项；负协方差清楚显示增量与原信号耦合。最终物理信号/电子噪声分解unknown，重复噪声不增加真实独立父事件。',
'','## 预定结论与资源',
'','门槛1（对BASE两完整视图改善）通过；门槛2（OT优于Independent）失败；门槛3（PSD/宽度/噪声不显著退化）失败；门槛4（源结构约束）失败；门槛5（数值精度/独立重放）通过。**关闭本固定OT-CFM配方**，不按DEV改网络/epsilon/匹配代价/种子/ODE步数。',
'','预定前32父记录的32vs64步Heun检查：Independent最大差2.36407e-5V、OT1.00061e-5V，均低于1/16383V。发布仍32步。独立CPU进程从实际源和随机输入重放两臂全部4000行，pre/postADC、base各分量、增量、mask/Qbin/ID全部逐位一致。',
'',f'实际科学累计{science:.6f}秒（2400秒限额），包括一次6.356629秒缺少辅助模块的导入失败；原失败源码/日志/退出保留，未到任何拟合。两臂服务器GPU各2950更新，训练主体合计20.204971秒；CPU固定评价55.974668秒、独立回放9.366128秒、聚合绘图6.052146秒。实际worker峰RSS2,027,515,904B，GPUallocated148,003,840B。远端1次RSS查询缺测有时间戳、原因unknown，不能写0。',
'','本輪11图已实际逐张查看。新HGB20、密度描述16槽/实际EM14、主4000、joint2048、源探针1664、64步检查64行、独立重放4000；新G4/EVAL/CAL波形/长原始采集/安装均0。服务器训练与本机CPU评价环境分别记录，不声称跨平台训练逐位相同。正常传输退出码全部0，没有迁移内容一致性测试。完整输入、模型、逐事件结果和预测留本地；发布用METRICS_COMPACT。',
'','## 证据边界与下一唯一问题',
'','当前REAL是单批采集、历史已暴露DEV，人工50记录组不等于独立采集批次；G4源是总EE/had沉积的解析完整波形，未携带微观沉积时序。CFM在有限目标、网络、配方与训练预算下失败，不等于无配对源驱动转换不可能，也不证明G4本身错误。',
'','本轮支持优先核验一个有限问题：完整修正能否在保留冻结BASE的baseline与既定两段积分后，继续保留波形收益而避免PSD/噪声与源条件结构退化。它应作为另行登记的冻结输出干预，不在本轮补算，不从目前跨域谷区直接推断粒子类别错误。',
'','方法采用现有general-source CFM与minibatch OT的有限版本：[Tong等论文](https://arxiv.org/html/2302.00482v3)，[作者实现](https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/conditional_flow_matching.py)。其统计耦合理论不构成本探测器响应、真实事件配对或新颖性的证据。',
'','主要产物：GATE_METRICS.json、METRICS_COMPACT.json、results/figures、两臂FINAL_FLOW.pt/generated.npz、REPLAY_RECEIPT.json、RESOURCE_RECEIPT.json、REPRODUCTION.md；最终FINAL_MANIFEST提供本阶段单份字节清单。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf8')

repro='''# R33 复现入口

这是固定配方负结果的复现资料，不默认授权再次训练；复现应放新目录，保持本阶段只读。

实际训练位置：rylan@100.82.231.102，
`/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R33_REMOTE_20261009_175239_UTC`。
既有解释器 `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`，CUDA:0、CPU4。
入口 `linux_supervise.py train_server.py <解释器>` 经 `launch_remote.py` 发起，实际worker31584/super31583/native0；实际环境在remote_receipts/SERVER_ENVIRONMENT.json。

正常上传/返回由deploy.py、retrieve.py完成。现有输入为R23 TRAIN_PRIVATE、R26完整同区94400行及原R23 z/white/style流、R32 CONTROL_FINAL权重；精确路径配置在remote_config/config及PRETRAIN_FREEZE。
仅复用了已有TRAIN/DEV，未取EVAL/CAL波形或长采集。部署使用正常传输退出和加载状态，没有迁移对等哈希测试。

保存的INITIAL_FLOW和实际t/置换/匹配目标、训练全stream、初末RNG均保留；初态完全相同、2950实际末点、只速度MSE。
表示/仿射常数在AFFINE_CONSTANTS与AFFINE_METADATA，源Q的8区定义不改。匹配按每批原Q子组，不跨Q；NCE旧位置只记录而不用于本轮CFM。

本机实际解释器 `C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`，Torch2.10.0+cpu/NumPy2.2.6。
所有TEMP/TMP/TORCH_HOME/CUDA_CACHE_PATH/MPLCONFIGDIR/JOBLIB临时路径由supervise放到本阶段cache，线程4，导入前PID握手/RSS守护。

本轮实际成功评价入口：
`python -B supervise.py evaluate_endpoints_attempt002.py <本机解释器>`。
attempt001因缺少random_diagnostics在导入时退出1，保留原件，attempt002仅补必要旧辅助源码，不变数学。
实际worker33800/super35696/native0，20HGB、14EM、16描述槽；每fit/slot逐项持久化。

保存权重独立推理入口：
`python -B supervise.py replay_endpoints.py <本机解释器>`。
实际worker21608/super43484/native0。读取ACTUAL_GENERATION_INPUTS、冻结BASE权重、两份FINAL_FLOW、AFFINE_CONSTANTS；不读取REAL/TRAIN/内部验证，不拟合或更新。
完整源→G→固定32步Heun→一次14bit ADC的4000行全部逐位通过，结果REPLAY_RECEIPT。
32vs64只预定前32父/臂检查，未据结果选步数；不以64替代发布32。

纯保存结果汇总/绘图入口 `summarize_and_plot.py`；实际worker61876/super44116/native0，11图，0fit/0生成/0新随机投影。完整数值metrics留本地；METRICS_COMPACT是无身份/预测聚合版。

文件传输或新平台训练不承诺逐位一致；本轮bitwise只指已执行的本机保存权重完整链回放。
最终代数分量base_signal/base_electronic/flow_increment可重构总输出，不能当作新的物理信号/独立电子噪声分解。
'''
(ROOT/'REPRODUCTION.md').write_text(repro,'utf8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'assembled':True,
                  'science_actual_s':science,'gate_retained':False,'figures_actually_viewed':11},ensure_ascii=False),flush=True)
