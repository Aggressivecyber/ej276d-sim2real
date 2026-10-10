"""Control-only R43 report/resource closure and one terminal byte manifest."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,math,hashlib
ROOT=Path(__file__).resolve().parent;START=time.monotonic()
def utc():return datetime.now(timezone.utc).isoformat()
def load(n):return json.loads((ROOT/n).read_text('utf8'))
def enc(v):return (json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
cfg=load('config.json');deadline=datetime.fromisoformat(cfg['deadline_UTC']);A='SCALAR_TIME_FLOW_FINAL'
def put(n,v):
    p=ROOT/n;assert not p.exists(),n
    if datetime.now(timezone.utc)>=deadline:raise TimeoutError('R43 executor seal deadline')
    p.write_bytes(enc(v))
def prose(n,s):
    p=ROOT/n;assert not p.exists(),n
    if datetime.now(timezone.utc)>=deadline:raise TimeoutError('R43 executor seal deadline')
    p.write_text(s.strip()+'\n','utf8')
def digest(p):
    h=hashlib.sha256();n=0
    with p.open('rb')as f:
        while b:=f.read(1024**2):h.update(b);n+=len(b)
    return {'bytes':n,'sha256':h.hexdigest()}
def main():
    h=load('RUN_HISTORY.json');t=load(A+'/TRAINING_RECEIPT.json');s=load('TRAIN_SOLVER_CHECK.json')
    init=load('INITIALIZATION_RECEIPT.json');part=load(A+'/SCALAR_TIME_PARTICIPATION.json')
    gate=load('GATE_METRICS.json');qa=load('FIGURE_QA.json');plot=load('PLOT_SUMMARY_RECEIPT.json')
    gen=load('GENERATION_NUMERIC_RECEIPT.json');ev=load('EVALUATION_NUMERIC_RECEIPT.json');replay=load('REPLAY_RECEIPT.json')
    tools=load('EXECUTION_TOOLS.json');science=math.fsum(r['elapsed_s']for r in h)
    assert science<=2700 and h[0]['elapsed_s']<=900 and s['pass']
    assert t['updates']==2950 and t['actual_rows']==94400 and all(t['original_7_audit_streams_match'].values())
    assert part['zero_gradient_update_checks']==part['unchanged_value_update_checks']==2950 and part['initial_final_zero_columns_bitwise_equal']
    assert ev['counters']['HGB_attempts']==20 and ev['counters']['density_description_slots']==16 and ev['counters']['density_EM_attempts']<=16
    assert gen['counters']['main_generation_rows']==replay['rows']==4000 and replay['pass']
    assert qa['all_final_figures_opened'] and set(qa['figures'])==set(plot['figures'])
    for n,v in qa['figures'].items():assert v['sha256']==plot['figures'][n]['sha256'] and v['visual_pass']
    gates=[gate[k]for k in ['gate1_complete_response','gate2_PSD_width_centers_noise','gate3_source_preservation','gate4_boundary_source','gate5_same_treated_REAL_control','gate6_numerical_replay']]
    native_failures=[r for r in h if r.get('actual_worker_native_exit')!=0 or r.get('termination_reason')is not None]
    counters={'velocity_models_fitted':1,'velocity_updates':2950,'training_rows_per_domain':94400,
        'training_network_forward_rows':t['network_training_forward_rows'],'artificial_initial_zero_forward_rows':4,
        'TRAIN_solver_rows':128,'TRAIN_solver32_field_forward_rows':s['solver32']['field_forward_rows'],'TRAIN_solver64_reference_field_forward_rows':s['solver64']['field_forward_rows'],
        **gen['counters'],'density_description_slots':ev['counters']['density_description_slots'],'density_EM_attempts':ev['counters']['density_EM_attempts'],
        'HGB_attempts':ev['counters']['HGB_attempts'],'independent_main_replay_rows':replay['rows'],'old_reference_fits_generation':0,'TRAIN_constant_refits':0,'coupling_resorts_resamples':0}
    missing=[{'entry':r['entry'],'count':r['memory_probe_failures'],'events':r['memory_probe_failure_events']}for r in h if r['memory_probe_failures']]
    resource={'registered_UTC':cfg['registered_UTC'],'first_actual_command':cfg['first_actual_command'],'executor_deadline_UTC':cfg['deadline_UTC'],
        'science_total_s':science,'science_limit_s':2700,'unique_training_supervised_s':h[0]['elapsed_s'],'training_limit_s':900,
        'by_job':h,'actual_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes']for r in h),'RSS_limit_bytes':4294967296,
        'GPU_peak_allocated_bytes':t['GPU_peak_allocated_bytes'],'GPU_limit_bytes':4294967296,'CPU_threads':4,'single_science_job_at_a_time':True,
        'fit_generation_counters':counters,'RSS_query_gaps':missing,'RSS_query_gap_causes':'unknown; timestamps near exit do not establish cause',
        'native_science_failures':len(native_failures),'control_failures':tools.get('control_failures',[]),
        'all_local_code_data_cache_temp_E_D':True,'remote_leaf':cfg['remote_root'],'migration_content_tests':0,
        'system_dependency_disk_IO':'not measured; not asserted zero','output_limit_bytes':536870912,'final_exact_output_bytes':'FINAL_RECEIPT.json',
        'old_stages_modified':False,'no_extra_candidates_solver_encodings_or_seeds':True}
    put('RESOURCE_RECEIPT.json',resource)
    put('NATIVE_CHAIN.json',{'scientific_processes':h,'tools':tools,'science_total_s':science,'self_receipt_not_external_exit_evidence':True})
    put('FAILURE_ACCOUNTING.json',{'scientific_native_failures':native_failures,'control_failures':tools.get('control_failures',[]),
        'scientific_gate_results':[g['pass']for g in gates],'RSS_query_gaps':missing,'RSS_query_gap_cause':'unknown',
        'all_original_native_logs_retained':True,'science_failed_attempt_time_not_refunded':True,'no_new_scope_to_rescue_results':True})
    put('SEMANTIC_FIELD_MAP.json',{'inside_TRAIN_support':'original frozen Q support mask, not whether NN ran',
        'neural_flow_applied_all_rows':'actual flow execution flag, true for all2000 SOURCE including417 outside',
        'legacy_flow_applied_field':'removed from output arrays; descriptive compatibility labels refer only to support mask',
        'base_signal_V':'actual source waveform; functional naming only','base_electronic_V':'projected initial Gaussian; not detector electronics',
        'flow_increment_f64_V':'learned full480 increment plus final casting accounting; not physical independent signal',
        'projection_gradient_status':'static differentiable construction; no standalone autograd test',
        'R42_initialization':'saved original initial weights compared directly to freshly seeded R43, never loaded as initialization',
        'initialization_reuse_vs_coupling':'old TRAIN constants and actual statistical target permutation reused; old final weights not reused'})
    first=gate['gate1_complete_response'];second=gate['gate2_PSD_width_centers_noise'];third=gate['gate3_source_preservation'];fifth=gate['gate5_same_treated_REAL_control']
    rows=['| 门 | 实际结果 |','|---|---|']+[f'| {i+1} | {"pass" if g["pass"] else "fail"} |'for i,g in enumerate(gates)]
    values=['| 对 BASE 的预定指标 | 本轮 | BASE | 相对变化 | 通过 |','|---|---:|---:|---:|---|']
    for k,r in second['measures'].items():
        f=lambda x:'undefined'if x is None else f'{x:.8g}'
        values.append(f'| {k} | {f(r["candidate"])} | {f(r["BASE"])} | {f(r["relative_change"])} | {r["pass"]} |')
    controls=['| 同处理 REAL 视图 | BASE | R43 | 配对差 | 95% CI |','|---|---:|---:|---:|---|']
    for k,r in fifth['three_views'].items():controls.append(f'| {k} | {r["BASE_G_REAL_BA"]:.6f} | {r["new_G_REAL_BA"]:.6f} | {r["BA_delta_first_minus_second"]:.6f} | {r["BA_delta_95CI"]} |')
    prose('REPORT.md',f'''
# R43：标量时间源初态神经流

固定配方决定：**{gate['decision']}**。唯一2950末点；六门依次为 **{[g['pass']for g in gates]}**。TRAIN数值通过与整体科学响应验收分别报告，不把native0、MSE下降或精确回放当成分布匹配。

{chr(10).join(rows)}

只将R42的17个时间槽改为 `[t,zero16]`，再接原8维Q条件，仍为1,371,841参数的原U-Net/attention；构造顺序与seed20261010420保持，新Adam .0002/.9,.999/clip5。实际R42保存的初始state与本轮新初态全部张量逐值相等；没有加载R42末点或旧G/D/reader。2048恒零输入列在2950次更新中实际零梯度、位值保持，t/bin有梯度的步数各{part['scalar_t_nonzero_gradient_steps']}/{part['bin_inputs_nonzero_gradient_steps']}，其他参数路径{part['other_parameters_nonzero_gradient_steps']}步。1,369,793只是可能获得数据梯度的参数数上界，不声称每个剩余参数每步非零。

R42固定REAL TRAIN每Q区mu/scale/edges、已保存PSD同序耦合和target/white/t全部复用，零重拟合/重采样。94400行/域、七项原审计流和2950次源/目标边缘保留；载入6593/21047，实际唯一消费3752/18739。PSD同序是统计耦合而非物理配对，人工组/重复事件不增加独立采集。完整SOURCE/q加1mV投影white初态，float64公共坐标与电荷切空间、float32网络，唯一速度MSE；没有GAN/NCE/identity/SW/J20/style训练项。频率buffer保留兼容但不参与forward。

相同128条TRAIN与white的32/64步RMS差中位 **{s['RMS_median_LSB']:.6f} LSB**、p95 **{s['RMS_p95_LSB']:.6f} LSB**，限值.25/1；R42原结果{ s['R42_reference_RMS_median_LSB']:.6f}/{s['R42_reference_RMS_p95_LSB']:.6f}。32/64实际f32 preADC最大相对q误差 {s['solver32']['max_relative_q_error_f32']:.8g}/{s['solver64']['max_relative_q_error_f32']:.8g}。此单次表示干预与误差下降相符，不能证明高频网格混叠是唯一原因；64步始终仅数值参考，不是候选。没有最终重归一化、步数救援或第三编码。

主SOURCE2000父记录、顺序、组成及权重全部保留；包括417支持外都经过新流，仅条件区索引取边缘，不clip实际q/电压或回退BASE。主shape493/u480 BA **{first['candidate_BA']}**，对应预定阈值.934/.928，配对差及CI在GATE/METRICS_COMPACT保留。实际source-Q纯had bins3/4与历史ADC参考213/181、207/182明确分列，谷区不是粒子真值。源Q/PSD秩代理与源扰动只能检验模型关联，不能证明逐事件物理对应。

{chr(10).join(values)}

每域原8区密度支持缺口保留，新失败 **{second['new_required_density_failures']}**；条件MAE仅共同合格区比较，不能用交集隐去新增失败或把两个密度成分当粒子真值。完整电荷分布保留，没有用硬保留q遮掩源/实测组成差异。源谷/相关门明细：{third['actual_source_pure_had_valleys']}；{third['whole_population_source_ranks']}。

{chr(10).join(controls)}

新REAL控制使用相同固定2000 TRAIN_REFERENCE与原主white；封存R39 BASE_REAL与RAW旧预测/密度零新拟合，RAW Q/PSD2D配对缺失继续unknown。REAL随机处理不预设恒等，门槛只用同处理新减BASE配对差，RAW绝对差作为描述。这是见过的同采集人工分组开发比较，不能称独立盲测。

生成4000主、1024同父重复、832源探针和128完整源扰动；新20HGB、16密度槽/实际{ev['counters']['density_EM_attempts']}EM。独立CPU完整链4000条及功能分量/pre/postADC逐值回放 **{replay['pass']}**，同chunk64、固定32Heun与一次14bitADC；回放不等于科学成功。SOURCE/初始投影Gaussian/学习增量/舍入项是功能与代数分量，不称物理独立电子/信号噪声。所有{plot['figure_count']}张最终图已实际打开，未平滑、未独立重标定或从输出挑漂亮事件。

累计科学 **{science:.6f}s /2700s**；唯一服务器训练worker90500/super90499 native0，{h[0]['elapsed_s']:.6f}s。实际RSS峰{resource['actual_RSS_peak_bytes']:,}B，GPU allocated{resource['GPU_peak_allocated_bytes']:,}B，均在4GiB。所有原始退出、费用和带时间戳RSS查询缺测保留，缺测原因unknown。首评价因配置缺少R26固定分区路径 native1，13.077363秒计入累计科学，HGB/EM均0；补原绑定路径后由evaluate_recovery_001续接同一20/16计划，没有重复拟合、训练或生成。另两次纯控制路径/源码摘录错误记录在FAILURE_ACCOUNTING，科学0；没有未知作业重提。所有本机数据/缓存E/D，正常传输无迁移一致性测试，系统依赖磁盘IO未测不写0。EVAL/CAL科学值、新REAL/长采集/G4/安装/代理/Git操作均0。

科学门失败关闭此固定配方，未扫描种子、solver、噪声、网络、耦合或阈值。R42数值失败永久保留；项目目标继续，下一具体问题由根登记。
''')
    prose('REPRODUCTION.md',f'''
# R43 复现入口与范围

训练：`train_server.py`，独立服务器目录 `{cfg['remote_root']}`，Python `{cfg['server_python']}`；Linux3.12.3/NumPy1.26.4/Torch2.12.0+cu130、RTX5070。原R42常数与saved耦合只读复用。实际初态、optimizer、2950末点、CFM配位/源目标/t流及随机状态在 `SCALAR_TIME_FLOW_FINAL`。seed20261010420新建模型，原R42 INITIAL仅比较，不作为加载初始化。

本机CPU入口：`supervise.py`→`owned_worker.py`导入前nonce/actualPID/RSS守护；Python `{cfg['CPU_python']}`，3.13.3/Torch2.10+cpu/NumPy2.2.6、4线程。实际顺序 `check_train_solver.py`→`generate_all.py`→首失败`evaluate_saved.py`→固定分区路径修复后的`evaluate_recovery_001.py`→`replay_all.py`→`summarize_and_plot.py`。原评价入口/配置/日志保留engineering_failure_001，恢复只补既有R26路径与独立失败文件名，不改科学定义。Pipeline为 `network.py/source_flow_core.py/flow_runtime.py`，权重 `SCALAR_TIME_FLOW_FINAL/FINAL_FLOW.pt` 和 `UNIT_CHARGE_CONSTANTS.npz`。读config中已绑定输入，无原始长波形/新G4。

独立CPU重放已实际执行4000条，原输入/white→新流32Heun→一次ADC及所有保存分量逐值通过；另一个进程从新初始化训练未重复执行，不能宣称CPU/CUDA训练逐位一致。初态比较是科学对照证据，正常scp/copy不做迁移一致性测试。复现应在新E/D目录，保留封存stage和 `.used` 原件；本轮结果失败不能用64步/额外编码或改门槛救援。完整私人数组/行流/预测/模型保留E/D，发布用METRICS_COMPACT。
''')
    items={p.relative_to(ROOT).as_posix():digest(p)for p in sorted(ROOT.rglob('*'))if p.is_file()and p.name not in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']}
    member_bytes=sum(v['bytes']for v in items.values());manifest={'stage':ROOT.name,'UTC':utc(),'files':items,'member_count':len(items),'member_bytes':member_bytes,
        'exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'self_reference':'terminal files excluded; external native tool binds both hashes','scientific_recomputation_in_seal':False,'old_stages_modified':False}
    raw=enc(manifest);msha=hashlib.sha256(raw).hexdigest();put('FINAL_MANIFEST.json',manifest)
    rec={'stage':ROOT.name,'status':'sealed','UTC':utc(),'PID':os.getpid(),'decision':gate['decision'],'retain':gate['retain'],'gates_in_order':[g['pass']for g in gates],
        'science_total_s':science,'manifest_sha256':msha,'manifest_bytes':len(raw),'manifest_member_count':len(items),'manifest_member_bytes':member_bytes,
        'stage_file_count_including_terminal_controls':len(items)+2,'stage_total_bytes_including_terminal_controls':0,'receipt_bytes':0,
        'terminal_hashes':{n:digest(ROOT/n)for n in ['REPORT.md','REPRODUCTION.md','METRICS_COMPACT.json','GATE_METRICS.json','RESOURCE_RECEIPT.json','NATIVE_CHAIN.json','FAILURE_ACCOUNTING.json','FIGURE_QA.json']},
        'deadline_UTC':cfg['deadline_UTC'],'before_deadline':True,'native_exit_evidence':'external tool result; this self receipt is not native exit','no_further_stage_writes':True,'scientific_job_active':False}
    for _ in range(10):
        n=len(enc(rec));total=member_bytes+len(raw)+n
        if rec['receipt_bytes']==n and rec['stage_total_bytes_including_terminal_controls']==total:break
        rec['receipt_bytes']=n;rec['stage_total_bytes_including_terminal_controls']=total
    else:raise RuntimeError('terminal size convergence')
    assert total<=536870912;put('FINAL_RECEIPT.json',rec)
    last=max(p.stat().st_mtime_ns for p in ROOT.rglob('*')if p.is_file());final=(ROOT/'FINAL_RECEIPT.json').read_bytes()
    print(json.dumps({'PID':os.getpid(),'status':'sealed','FINAL_MANIFEST_sha256':msha,'FINAL_RECEIPT_sha256':hashlib.sha256(final).hexdigest(),
        'last_stage_write_UTC':datetime.fromtimestamp(last/1e9,timezone.utc).isoformat(),'file_count':len(items)+2,'bytes':total,'science_s':science,
        'seal_control_elapsed_s':time.monotonic()-START,'no_further_stage_writes':True}),flush=True)
if __name__=='__main__':main()
