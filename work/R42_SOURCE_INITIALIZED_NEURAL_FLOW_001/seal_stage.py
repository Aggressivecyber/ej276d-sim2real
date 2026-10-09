"""Control-only negative-result report and one terminal byte manifest."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,time,math,hashlib
ROOT=Path(__file__).resolve().parent;START=time.monotonic()
def utc():return datetime.now(timezone.utc).isoformat()
def load(n):return json.loads((ROOT/n).read_text('utf8'))
def encode(v):return (json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
def put(n,v):
    p=ROOT/n;assert not p.exists(),n
    if datetime.now(timezone.utc)>=deadline:raise TimeoutError('R42 seal deadline')
    p.write_bytes(encode(v))
def text(n,s):
    p=ROOT/n;assert not p.exists(),n
    if datetime.now(timezone.utc)>=deadline:raise TimeoutError('R42 seal deadline')
    p.write_text(s.strip()+'\n','utf8')
def digest(p):
    h=hashlib.sha256();n=0
    with p.open('rb')as f:
        while b:=f.read(1024**2):h.update(b);n+=len(b)
    return {'bytes':n,'sha256':h.hexdigest()}
cfg=load('config.json');deadline=datetime.fromisoformat(cfg['deadline_UTC'])
def main():
    h=load('RUN_HISTORY.json');s=load('TRAIN_SOLVER_CHECK.json');t=load('SOURCE_NEURAL_FLOW_FINAL/TRAINING_RECEIPT.json')
    tn=load('TRAINING_NUMERIC_RECEIPT.json');flow=load('MATCHED_TRAINING_FLOW_RECEIPT.json');plot=load('PLOT_FAILURE_RECEIPT.json');qa=load('FIGURE_QA.json')
    assert len(h)==3 and [r['entry']for r in h]==['train_server.py','check_train_solver.py','plot_train_failure.py']
    assert all(r['actual_worker_native_exit']==r['launcher_native_exit']==0 for r in h)
    science=math.fsum(r['elapsed_s']for r in h);assert science<=2700 and h[0]['elapsed_s']<=900
    assert t['updates']==2950 and t['actual_rows']==94400 and all(t['original_7_audit_streams_match'].values())
    assert not s['pass'] and s['DEV_read_or_generated']==0
    assert not any((ROOT/n).exists()for n in ['GENERATION_RECEIPT.json','EVALUATION_NUMERIC_RECEIPT.json','REPLAY_RECEIPT.json','generate_all.py.used','evaluate_saved.py.used','replay_all.py.used','summarize_and_plot.py.used'])
    assert qa['all_final_figures_opened'] and set(qa['figures'])==set(plot['figures']) and plot['figure_count']==4
    for n,v in qa['figures'].items():assert v['sha256']==plot['figures'][n]['sha256'] and v['visual_pass']
    gates={f'gate{i}_{label}':{'status':'not_evaluated','pass':None,'reason':'fixed TRAIN32/64 solver prerequisite failed before DEV'}for i,label in enumerate(['complete_response','PSD_width_centers_noise','source_preservation','boundary_source','same_treated_REAL_control'],1)}
    gates['gate6_numerical_replay']={'status':'fail','pass':False,'TRAIN_fixed32_vs64_pass':False,'TRAIN_check':s,
        'charge_error_itself_pass':max(s['solver32']['max_relative_q_error_f32'],s['solver64']['max_relative_q_error_f32'])<=1e-5,
        'training_complete':True,'seven_audit_streams_match':flow['same_7_audit_streams'],'original_target_marginals_preserved':True,
        'independent_main4000_replay':'not_executed_after_prerequisite_failure',
        'not_applicable':['old G initialization equivalence','D/reader','J20','PatchNCE','fresh projections','style'],
        'actual_QA_and_resource_recorded':True}
    gate={'fixed_recipe':'new standalone fullSOURCE unit-charge convolutional neural flow,2950 updates,fixed32Heun',
        'decision':'close_this_fixed_source_initialized_neural_flow_recipe','retain':False,'scientific_distribution_effectiveness':'not_evaluated',
        'thresholds_changed_after_results':False,'no64step_rescue_or_prefix_candidate':True,'no_general_flow_impossibility_claim':True,**gates}
    put('GATE_METRICS.json',gate)
    put('STOP_CANDIDATE.json',{'UTC':utc(),'reason':'fixed32 TRAIN numerical prerequisite failed','allowed_remaining_work':'saved TRAIN figures/report/control seal only',
        'prohibited_current_recipe':['DEV generation','HGB/EM','4000 replay','solver/steps/noise/network/coupling scans'],'new_candidate_registered':False})
    compact_t={k:t[k]for k in ['updates','actual_rows','training_s','selection','new_model_no_pretrained_weights','white_used_z_style_NCE_unused',
        'actual_unique_source_REAL','loaded_source_REAL','marginal_preservation_checks','max_path_q_residual','max_field_q_residual','parameter_count',
        'GPU_peak_allocated_bytes','network_training_forward_rows','artificial_zero_check_rows','old_G_D_reader_forwards_updates']}
    counts={'velocity_models_fitted':1,'velocity_updates':2950,'training_rows_each_domain':94400,'TRAIN_solver_input_rows':128,
        'TRAIN_solver32_solution_rows':128,'TRAIN_solver64_numeric_reference_rows':128,'TRAIN_solver_field_forward_rows':24576,
        'network_zero_field_artificial_rows':4,'total_network_forward_rows_including_training_check_and_zero':118980,
        'DEV_waveform_scientific_decodes':0,'DEV_main_generation':0,'REAL_reference_control_generation':0,'joint_probe_extra_generation':0,
        'HGB':0,'EM':0,'density_slots':0,'main_replay':0,'EVAL_read':0,'CAL_waveform_decode':0,'new_G4_REAL_longwave_read':0,
        'additional_solver_calls_in_figures_or_seal':0}
    put('TRAINING_COMPACT.json',compact_t)
    put('METRICS_COMPACT.json',{'status':'closed_fixed_recipe_at_TRAIN_solver_prerequisite','training':compact_t,'solver':s,'gate':gate,
        'counters':counts,'figures':plot['figures'],'privacy':'aggregate only; exact random streams/TRAIN rows/weights remain in local artifacts',
        'source_labels_or_distribution_performance_not_inferred':True,'64step_not_a_DEV_candidate':True})
    external={'first_control':cfg['first_actual_command'],'preparation':{'tool':'e564f5','PID':45344,'native_exit':0},
        'preflight':{'tool':'88810f','PID':35328,'SSH_PID':56644,'server_PID':86395,'native_exit':0},
        'deploy':{'tool':'d9aa2b','PID':59932,'native_exit':0},
        'training':{'start_tool':'fd82fb','session_id':20521,'final_tool':'1e605f','local_PID':52176,'SSH_PID':37152,'worker_PID':86767,'supervisor_PID':86766,'native_exit':0},
        'normal_return':{'tool':'1cd27e','PID':20524,'native_exit':0,'pack_and_scp_native_exit':0,'pack_SSH_child_PID':'unknown','scp_child_PID':'unknown','migration_content_tests':0},
        'TRAIN_check':{'start_tool':'a9ad4d','session_id':68956,'final_tool':'cb3a28','worker_PID':1592,'supervisor_PID':36844,'native_exit':0},
        'report_source_preparation':{'tool':'19188d','PID':29148,'native_exit':0,'scientific_computation':0},
        'TRAIN_plots':{'tool':'942fc2','worker_PID':44312,'supervisor_PID':17368,'native_exit':0},
        'actual_visual_open_tool':'functions.exec(view_image) after942fc2; all4 images opened',
        'control_native_failures':[{'tool':'1c7620','native_exit':1,'cause':'source-only Get-Content of nonexistent R41 finalize.py; actual filename later found seal_stage.py',
            'native_PID':'unknown in original tool output','scientific_computation':0,'data_mutation':0,'original_failure_preserved':True}]}
    put('NATIVE_CHAIN.json',{'scientific_processes':h,'external_tools':external,'science_total_s':science,'self_report_is_not_external_native_exit':True})
    put('FAILURE_ACCOUNTING.json',{'scientific_native_failures':0,'scientific_result_failure':'fixed32 TRAIN solver accuracy',
        'control_native_failures':external['control_native_failures'],'RSS_query_gaps':[{'entry':r['entry'],'count':r['memory_probe_failures'],'events':r['memory_probe_failure_events']}for r in h if r['memory_probe_failures']],
        'RSS_query_gap_cause':'unknown; temporal proximity to exit is not proof','no_training_or_check_retries':True,'no_64step_rescue':True,
        'unused_prepared_pipeline':{'files':['generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py'],
            'status':'prepared/AST only; not scientifically executed or validated on this model','budget_cost':'ordinary source/control writes, not scientific generation'},
        'semantic_corrections':load('SEMANTIC_FIELD_MAP.json'),'frozen_remote_sources_not_rewritten':True})
    resource={'registered_UTC':cfg['registered_UTC'],'first_actual':cfg['first_actual_command'],'executor_deadline_UTC':cfg['deadline_UTC'],
        'science_total_s':science,'science_limit_s':2700,'unique_training_supervised_s':h[0]['elapsed_s'],'training_limit_s':900,
        'by_job':[{k:r[k]for k in ['entry','actual_worker_PID','supervisor_PID','start_UTC','stop_UTC','actual_worker_native_exit','elapsed_s','actual_worker_RSS_peak_bytes','memory_probe_failures']}for r in h],
        'actual_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes']for r in h),'actual_RSS_limit_bytes':4294967296,
        'GPU_peak_allocated_bytes':tn['GPU_peak_allocated_bytes'],'GPU_limit_bytes':4294967296,'threads':4,'single_science_job_at_a_time':True,
        'fit_generation_counters':counts,'native_failures_kept':True,'RSS_gap_causes':'unknown','source_control_failure_no_scientific_cost':True,
        'copy_location':'all local code/data/temp/cache E/D; server registered leaf','copied_existing_DEV_files':'ordinary bytes were copied during preparation, never waveform-decoded or used in scientific inference',
        'migration_content_tests':0,'output_cap_bytes':536870912,'output_bytes_before_terminal_manifest':sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file()),
        'final_exact_output_bytes':'FINAL_RECEIPT.json','system_library_import_IO_not_measured':'not represented as zero'}
    put('RESOURCE_RECEIPT.json',resource)
    text('REPORT.md',f'''
# R42：固定源初态神经流在 TRAIN 数值门停止

本轮关闭固定配方，**未进入 DEV 评价**。32/64步Heun在固定128条TRAIN输入的电压RMS差中位数 **{s['RMS_median_LSB']:.6f} LSB**、95分位 **{s['RMS_p95_LSB']:.6f} LSB**，超过预注册的 **0.25 / 1 LSB**。五项响应/PSD/源保真/边界/REAL控制科学门为 `not_evaluated`；第六数值门 `fail`。native0只说明训练与检查正常退出，不能作为模型有效性通过。64步仅数值参考，未作为候选或补救；没有延长训练、增加步数、重训或换初始化。

新模型由完整480点G4派生SOURCE启动，使用线性电荷 q=sum100:400−3sum0:100 的单位电荷表示、REAL TRAIN每Q区共同均值/逐点SD、秩一电荷切空间投影以及固定1mV原white初态。新32/64/128/128 U-Net、60点4-head attention、17维连续时间+8维区条件、末层零初始化，参数 **{t['parameter_count']:,}**；无旧G/reader/D/flow/diffusion权重。唯一训练目标是全部480维标准化速度MSE，新Adam lr.0002/.9,.999、clip5，仅2950末点。整体表示、架构和目标共同改变，不能解释成单部件消融。

训练完整2950更新、94400行/域，原七项行/随机审计流全部匹配，逐批2950次耦合边缘保留。载入6593源/21047实测，实际唯一消费3752/18739；重复使用和人工分组不增加独立采集数。Q内按未加噪原SOURCE与REAL PSD稳定一一排序，是统计耦合而非物理配对。原white与t实际参与；z/style/NCE仅未使用审计。新训练网络前向94400行，人工零场4行；没有GAN、NCE、identity、SW/MMD、style或J20。

在任何DEV之前，按TRAIN池顺序取首64支持内SOURCE+首64支持内REAL，使用 **首128 consumed-row white（ordinal[:128]，原噪声生成ordinal保存）**。32步与64步参考共24576网络前向行/384个批次场调用；两解和逐条输入/误差保留在 `TRAIN_SOLVER_CHECK_PRIVATE.npz`。坐标、投影及Heun累加使用float64，网络及实际preADC保存float32。32步实际float32 preADC最大相对电荷误差 {s['solver32']['max_relative_q_error_f32']:.10g}、64步 {s['solver64']['max_relative_q_error_f32']:.10g}，均小于1e-5；这只表明该守恒代理数值上成立，不能弥补步长误差或证明物理校准。没有最终重归一化修复，也没有ADC后电荷守恒结果。

4张图均来自已保存TRAIN：训练MSE/速度/梯度、电荷切空间残差、32/64步误差CDF和首/末检查行的完整波形、TRAIN公共坐标及电荷误差。已实际逐张打开，未平滑或挑DEV展示。训练MSE下降及图上两解接近都不替代预定数值门。64步也没有被证明为精确ODE真解，因此本轮只能结论固定32步检查失败，不能推断整个源驱动flow路线不可行或完整响应匹配成功/失败。

实际训练 worker86767/super86766、native0、{h[0]['elapsed_s']:.6f}s；本机TRAIN检查 worker1592/super36844、native0、{h[1]['elapsed_s']:.6f}s；保存证据绘图 worker44312/super17368、native0、{h[2]['elapsed_s']:.6f}s。累计科学 **{science:.6f}s / 2700s**，RSS峰 {resource['actual_RSS_peak_bytes']:,}B，GPU allocated峰 {resource['GPU_peak_allocated_bytes']:,}B，均在4GiB内。远端两次带时间戳RSS查询缺测保留，原因unknown，不能以发生在退出附近推定原因。一次纯源码读取尝试找不到旧 `finalize.py`，native1、科学0，已保留；无科学工程失败或重提。

DEV输入文件在准备中普通字节复制，但本轮DEV波形数值解码、候选生成、HGB、EM、4000回放均 **0**，EVAL/CAL波形/新G4/长实测读取均0。设计上417支持外只用边缘区条件并全部经过新流，但因数值前置失败未执行这项DEV外推，不能把它写成已验证。兼容 `flow_applied` 字段仅代表TRAIN支持内，真正执行字段为 `neural_flow_applied_all_rows`；旧functional字段详见 `SEMANTIC_FIELD_MAP.json`，不称物理signal/electronic独立分解。`FLOW_FORMULA_CHECK.field_projection_preserves_gradients` 为静态可微构图性质，没有另跑autograd测试。冻结远端科学核心原字节保留。

下一步由根登记新问题后执行。本轮不自行扫描求解器、步数、噪声、网络、耦合或额外初始化。项目目标继续，当前固定配方已关闭。
''')
    text('REPRODUCTION.md',f'''
# R42 复现与停止位置

完整执行模型为 `network.py` + `source_flow_core.py`，唯一权重 `SOURCE_NEURAL_FLOW_FINAL/FINAL_FLOW.pt` 和固定 `UNIT_CHARGE_CONSTANTS.npz`。训练入口 `train_server.py`，服务器 `{cfg['remote_root']}`，Python `{cfg['server_python']}`，实际NumPy1.26.4/Torch2.12.0+cu130/CUDA RTX5070；初态、optimizer、actual CFM行/耦合/t与未使用审计流均保存。输入从config中已绑定R23 TRAIN池/R26重组及R33原t流只读复用，没有重跑G4或获取新实测。

本机检查用 `{cfg['CPU_python']}`，Python3.13.3/Torch2.10+cpu/NumPy2.2.6，CPU4。入口 `check_train_solver.py` 由 `supervise.py`→`owned_worker.py` 导入前nonce/PID/RSS监督。固定chunk64对TRAIN128行做32/64检查，保存精确输入与解；这是数值参考检查，不是迁移测试。复现须在新E/D目录恢复，保留旧阶段和原 `.used` 文件，不能覆写本stage。

两环境不同，不宣称CPU/CUDA训练逐位一致。已保存末点CPU TRAIN检查正常执行且数值失败；未执行DEV生成、20HGB/16槽或4000主链回放，因此不声称推理主输出/训练初始化全链已通过独立复现。预备 `generate_all.py/evaluate_saved.py/replay_all.py/summarize_and_plot.py` 本轮未执行，必须尊重 `STOP_CANDIDATE.json`；不能将它们当作获准绕过检查的复现实验。

必要图入口 `plot_train_failure.py` 只读取本轮已保存TRAIN trace、128两解和固定坐标，不再求解、拟合或生成。正常迁移/取回pack和scp均native0，未做迁移内容一致性测试；原输入版本绑定保留。所有本机缓存/TEMP/MPL路径由监督器限定E盘。根负责Git及D盘交付。
''')
    put('CODE_STATUS.json',{'executed_scientific_entries':['train_server.py','check_train_solver.py','plot_train_failure.py'],
        'unexecuted_after_numerical_stop':['generate_all.py','evaluate_saved.py','replay_all.py','summarize_and_plot.py'],
        'inference_scientific_validation':'not reached','frozen_remote_sources':'PRETRAIN_FREEZE.json; preserved original',
        'new_source_model_weights':'SOURCE_NEURAL_FLOW_FINAL/FINAL_FLOW.pt','unused_historical_utility_modules_do_not_load_old_response_weights':True})
    # Seal after final resource/report files. No byte ledger framework or scientific array decoding.
    items={p.relative_to(ROOT).as_posix():digest(p)for p in sorted(ROOT.rglob('*'))if p.is_file() and p.name not in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']}
    member_bytes=sum(v['bytes']for v in items.values())
    manifest={'stage':ROOT.name,'UTC':utc(),'files':items,'member_count':len(items),'member_bytes':member_bytes,
        'exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'self_reference':'manifest excludes itself and terminal receipt; external tool binds terminal hashes',
        'old_stage_files_modified':False,'scientific_recomputation_in_seal':False}
    raw=encode(manifest);sha=hashlib.sha256(raw).hexdigest();put('FINAL_MANIFEST.json',manifest)
    rec={'stage':ROOT.name,'status':'sealed_negative_numeric_prerequisite','UTC':utc(),'PID':os.getpid(),'decision':gate['decision'],'retain':False,
        'gates_in_order':['not_evaluated']*5+['fail'],'science_total_s':science,'manifest_sha256':sha,'manifest_bytes':len(raw),
        'manifest_member_count':len(items),'manifest_member_bytes':member_bytes,'stage_file_count_including_terminal_controls':len(items)+2,
        'stage_total_bytes_including_terminal_controls':0,'receipt_bytes':0,'terminal_hashes':{n:digest(ROOT/n)for n in ['REPORT.md','REPRODUCTION.md','METRICS_COMPACT.json','GATE_METRICS.json','RESOURCE_RECEIPT.json','NATIVE_CHAIN.json','FAILURE_ACCOUNTING.json','FIGURE_QA.json']},
        'deadline_UTC':cfg['deadline_UTC'],'before_deadline':True,'native_exit_evidence':'external command native result; this self receipt is not native exit',
        'no_further_stage_writes':True,'scientific_job_active':False,'DEV_generation_HGB_EM_replay':0}
    for _ in range(8):
        n=len(encode(rec));total=member_bytes+len(raw)+n
        if rec['receipt_bytes']==n and rec['stage_total_bytes_including_terminal_controls']==total:break
        rec['receipt_bytes']=n;rec['stage_total_bytes_including_terminal_controls']=total
    else:raise RuntimeError('terminal size convergence')
    assert total<=536870912;put('FINAL_RECEIPT.json',rec)
    final=(ROOT/'FINAL_RECEIPT.json').read_bytes();last=max(p.stat().st_mtime_ns for p in ROOT.rglob('*')if p.is_file())
    print(json.dumps({'PID':os.getpid(),'status':rec['status'],'FINAL_MANIFEST_sha256':sha,'FINAL_RECEIPT_sha256':hashlib.sha256(final).hexdigest(),
        'last_stage_write_UTC':datetime.fromtimestamp(last/1e9,timezone.utc).isoformat(),'file_count':len(items)+2,'bytes':total,'science_s':science,
        'seal_control_elapsed_s':time.monotonic()-START,'no_further_stage_writes':True}),flush=True)
if __name__=='__main__':main()
