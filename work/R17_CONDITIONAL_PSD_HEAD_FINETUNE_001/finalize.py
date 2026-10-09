from __future__ import annotations
import pathlib,json,hashlib,datetime,os,time
ROOT=pathlib.Path(__file__).resolve().parent;start=time.monotonic();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def write(name,v):(ROOT/name).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
cfg=read('config.json');m=read('metrics.json');native=read('REMOTE_NATIVE_RECEIPT.json');replay=read('REPLAY_RECEIPT.json');redraw=read('CDF_LAYOUT_RECEIPT.json');counts=m['counters'];training=m['training'];psd=m['PSD_main'];cs=m['C2ST'];width=m['energy_conditional_width'];roles=m['data_roles'];targets=read('CONDITIONAL_TARGETS.json')
if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(cfg['deadline_UTC']):raise TimeoutError('R17 seal deadline passed; no further write')
compact={k:m[k]for k in ['status','models','complete_density_scores','step0_checks','gradient_check','data_roles','pairing','continuous_ADC_surrogate','PSD_main','domains','C2ST','environment','versions','counters','EVAL_read','same_acquisition_development','target_density_not_G4_mapping']}
compact['training']={n:{k:v for k,v in t.items()if k!='trace'}for n,t in training.items()}
compact['training']['trace_location']='each arm TRAINING_TRACE.json; aggregate validation records, complete local receipt retained'
compact['energy_conditional_width']={'edges_logQ':width['edges_logQ'],'common_support_errors':width['common_support_errors'],'old_REAL_and_STEP0_reused_no_fit':True,'not_particle_truth':True,'tables':{n:{'slices':[{'bin_id_1based':i+1,'valid':s['valid'],'failure_reasons':s['failure_reasons'],'components':s['components'],'fit_attempts':s['fit_attempts']}for i,s in enumerate(t['slices'])]}for n,t in width['tables'].items()}}
compact['source_conditioning']={'G4_inputs':0,'source_particle_or_PSD_conditioning':False,'draw_ids_are_G4_parent_events':False,'source_class_preservation_tested':False,'target_density_diagnostic_only':True,'future_target_only_continuation_not_authorized':True}
write('METRICS_COMPACT.json',compact)
figures=['conditional_PSD_CDF.png','charge_psd_full.png','charge_psd_main_body.png','valley_and_conditional_distances.png','training_validation.png','energy_peak_width.png','c2st_paired.png','waveform_noise_side_effects.png']
write('FIGURE_QA.json',{'UTC':utc(),'actually_viewed_in_tools':True,'selected_count':8,'results':[{'file':'results/figures/'+n,'viewed':True,'axes_legends_populations_readable':True,'layout_pass':True}for n in figures],'initial_visual_issue':{'file':'results/figures/original_conditional_PSD_CDF.png','long_adjacent_titles_overlap':True,'original_retained':True,'fix':'saved CDF values only, shorter multiline titles; updated main figure actually viewed','no_scientific_recompute':True},'interpretation':'full range and shared-axis body retained; width failures displayed as crosses; prompt100:150 includes peak/early falling segment'})
layout_native_wall_s=2.3073058
resource={'registered_UTC':cfg['registered_UTC'],'first_native_UTC':cfg['first_native_UTC'],'first_native_PID':cfg['first_native_PID'],'seal_deadline_UTC':cfg['deadline_UTC'],'stage_deadline_UTC':cfg['absolute_deadline_UTC'],'remote_root':cfg['remote_root'],'remote_history':native['history'],'remote_supervisor_PID':native['supervisor_PID'],'remote_numeric_cumulative_s':native['numeric_cumulative_s'],'local_saved_plot_redraw_native_wall_s':layout_native_wall_s,'local_saved_plot_redraw_inner_elapsed_s':redraw['elapsed_s'],'scientific_and_plot_cumulative_s_conservative':native['numeric_cumulative_s']+layout_native_wall_s,'limit_s':1800,'training_cumulative_s':counts['training_cumulative_s'],'training_arm_elapsed_s':{n:r['elapsed_s']for n,r in training.items()},'training_per_arm_limit_s':450,'training_combined_limit_s':900,'head_updates':{n:r['updates']for n,r in training.items()},'HGB_actual':counts['HGB_attempts'],'width_description_slots':counts['density_description_slots'],'width_EM_actual_attempts':counts['density_EM_attempts'],'main_generation_rows':counts['main_generation_rows'],'fresh_replay_rows':replay['total_rows'],'TRAIN_generated_inputs_consumed_per_arm':training['NLL_CONT']['updates']*cfg['batch'],'CONT_consumes_same_draws_but_does_not_optimize_them':True,'fixed_validation_checks_per_arm':{n:len(r['trace'])for n,r in training.items()},'fixed_validation_rows_per_check':2000,'reused_STEP0_rows':2000,'CPU_threads':4,'RSS_peak_bytes':max(r['RSS_peak_bytes']for r in native['history']),'local_redraw_RSS_peak_bytes':redraw['RSS_peak_bytes'],'RSS_limit_bytes':cfg['RSS_limit_bytes'],'GPU_allocated_peak_bytes':m['environment']['GPU_allocated_peak_bytes'],'GPU_reserved_peak_bytes':m['environment']['GPU_reserved_peak_bytes'],'GPU_allocated_limit_bytes':cfg['GPU_allocated_limit_bytes'],'execution_environment':m['versions'],'training_device':m['environment']['training_device'],'arithmetic':'TRAIN CUDAfloat32, validation/final/replay CPUfloat64','new_base_PCA_ridge_residual_fits':0,'EVAL_read':0,'CAL_scientific_decode':0,'raw_long_waveform_decode':0,'new_G4':0,'G4_source_conditions_used':0,'migration_content_tests':0,'source_provenance_rehash':0,'scientific_failures':[],'visual_issue_preserved_and_repaired':'overlapping CDF titles only','native_evidence':{'launch_tool':'e4309b','launch_terminal_session':19518,'completion_chunk':'0d1f9e','local_bridge_PID':23240,'SSH_PID':26048,'native_exit_code':0,'retrieval_tool':'a733d4','retrieval_PID':26560,'retrieval_native_exit_code':0,'local_plot_tool':'95ad75','local_plot_PID':1652,'local_plot_native_exit_code':0},'peak_RSS_sampling_failures':sum(r['RSS_probe_failures']for r in native['history']),'returned_tar_bytes':read('RETURN_TRANSFER_RECEIPT.json')['return_tar_bytes'],'output_limit_bytes':cfg['output_limit_bytes'],'system_dependency_loader_IO_not_measured':True,'final_outputs_hashing_is_pure_bytes_not_scientific_decode':True}
write('RESOURCE_RECEIPT.json',resource)
table='\n'.join('| '+str(i+1)+' | '+str(psd['tables']['REAL']['bins'][i]['N'])+' | '+' | '.join(f"{psd['vs_REAL'][n]['bins'][i]['W1']:.6f}"for n in ['STEP0','NLL_CONT','NLL_PSD'])+' |'for i in range(8))
vtable='\n'.join('| '+str(i+1)+' | '+' | '.join(f"{psd['tables'][n]['bins'][i]['valley_fraction']:.4f}"for n in ['REAL','STEP0','NLL_CONT','NLL_PSD'])+' |'for i in range(8))
reduction=1-psd['vs_REAL']['NLL_PSD']['eight_equal_mean_W1']/psd['vs_REAL']['NLL_CONT']['eight_equal_mean_W1']
report=f'''# R17：条件 PSD 约束有局部改善，未完成 G4→REAL 转换

本轮固定 GMM8 基分布与 R15 条件残差核，只微调34维 head。加入条件 PSD CDF/占比损失后，八区 PSD W1 都低于仅继续似然训练的分支，八区等权平均下降 {100*reduction:.2f}%。谷区仍过满，最高两区双成分拟合未收敛，完整波形仍明显可区分。本轮不能宣布项目目标完成。

**关键范围纠正：这里没有 G4 条件输入。** 这些波形是从实测 TRAIN 学到的目标密度随机采样，不是把某个 G4 事件转换为实测响应；draw_ids 是抽样序号，不是 G4 父记录。源 PSD、粒子类别、沉积组成条件均未传入，无法验证纯 γ 或纯 n 输入的类别保真。实测混合数据中的双带约束也不能替代源条件约束。用户已要求回到带源信息的转换问题；本阶段到此封存，不继续目标域自生成路线。

配方使用21047 TRAIN-fit、599组隔离内部验证、2000同采集 DEV；EVAL/CAL科学解码、原始长波形、新G4为0。基线窗0:100、快积分100:150、总积分100:400，PSD=1−Qfast/Qtotal，signed，单位与ADC流程均不变。旧 q4 参考失败、q1 高置信 neutron-like 数0保留。人工组不能视为独立采集批次，混合成分也不是粒子真值。

## 配对训练和选择

两臂相同 R14 EXPOSURES 初始 head、固定 R16 GMM8 与 R14/R15 条件核，Adam0.0003、batch256、各2000更新。最终 TRAIN 索引/生成随机流摘要相同，实际更新数相同。NLL_CONT只在训练中优化 head NLL/34；NLL_PSD加4×条件CDF L1和1×窗口占比L1。冻结网络参数仍传递输入梯度，因此条件核输出会随 head 改变。

两臂均用599真实验证与同一固定2000生成创新，在step0/每100步按同一个J选点；NLL_CONT的选择也使用PSD信息，不能称其完全不知PSD。DEV未用于选点。

| 分支 | 更新 | 所选step | 验证NLL/34 | 验证CDF L1 | 占比L1 | J |
|---|---:|---:|---:|---:|---:|---:|
| STEP0 | 0 | 0 | .904976 | .013151 | .017521 | .975100 |
| NLL_CONT | 2000 | 600 | .903657 | .012885 | .019430 | .974626 |
| NLL_PSD | 2000 | 1200 | .905071 | .006966 | .015773 | .948706 |

CONT最终J=.981942，PSD最终J=.951859，均不如各自早期所选点；轨迹完整保留，不能概括为全程单调改善。完整480维验证NLL为 STEP0 −2436.633954、CONT −2436.678799、PSD −2436.630734 nat/event。PSD在条件分布上改善而似然略差，不以复合目标下降替代分布验收。

## 完整条件PSD人口

所有域2000/2000 QC有效，无重抽或删除。硬电荷区间外低/高计数 REAL87/82、STEP0125/99、CONT115/98、PSD102/99；这些事件仍在总体结果与全范围散点中。CDF损失网格没有成为评价过滤器。TRAIN目标全21047一次构造，logQ范围外1120/1032，PSD低于−.1共30条；验证目标全599，logQ范围外34/29，无网格外PSD。原本就固定的 Gaussian 窗口宽度为一个完整log区间，邻区有混合；本轮不据此断言它是唯一残差原因。

| 固定bin | REAL n | STEP0 W1 | CONT W1 | PSD W1 |
|---|---:|---:|---:|---:|
{table}

八区等权平均 W1：.011774816 / .011073680 / .008315639；KS：.142453149 / .128498538 / .094465291。八区均非空，无因缺支持而删除区间。

谷区固定为旧 R3B TRAIN-fit 各区合格成分均值间的中间1/3，没有新拟合或把它直接作为训练目标。

| bin | REAL谷区比例 | STEP0 | CONT | PSD |
|---|---:|---:|---:|---:|
{vtable}

PSD分支在bin1略低于REAL，在bin2–8仍偏高；bin5相对STEP0甚至略高。二维图仍未恢复实测双带的清晰谷区，不能只看边际CDF或FoM。

## 峰宽与波形副作用

新增16宽度槽/16次EM。合格区：REAL1–8，STEP0仅4，CONT2/4，PSD1–6。PSD的7/8未在500次迭代内收敛，宽度undefined原样保留。三候选共同支持仅bin4；两个微调分支共同支持bin2/4。后者低PSD成分峰位MAE .033078→.019698、去趋势σ误差 .014493→.008679；高PSD成分去趋势σ误差 .004050→.005026反而变大。不能据覆盖更多区间就称所有峰宽匹配。

共同固定600/250位置，32次HGB、500同组配对区间。完整shape BA STEP0/CONT/PSD=.906/.908/.918，完整V=.968/.964/.956，tail=.922/.916/.924。PSD−CONT shapeΔ=.010（95%CI[−.00938,.02920]），VΔ=−.008（[−.02400,.00772]），均不能证明整体改进。基线RMS中位数 REAL .628mV，STEP0/CONT/PSD .677/.677/.675mV；噪声谱/ACF及高电荷prompt曲率仍有差距。prompt100:150包含峰和早期下降，不能都解释为纯上升沿。

二维Q/PSD HGB训练/留出 BA：STEP0 .730/.558，CONT .741/.528，PSD .707/.528；head34训练约.98而留出约.50，REAL–REAL同样训练很高。实际两列原始/预处理输入与预测仅保留本地，没有加拟合或改超参数。近机会水平不能证明PSD已经匹配，存在有限功效与泛化差距。FoM仅作补充：REAL .7024，STEP0 .5918，CONT .5916，PSD .6572；不以它替代完整PSD或源类别保真。

训练使用ADC前连续损失，最终评价ADC后。post−pre PSD RMS CONT .00020489、PSD .00017023，远小于主要条件距离。不能把当前差距直接归因量化。此次实验支持“显式条件PSD目标影响生成PSD”这一受控结论，未证明遗漏该项是全部失配的原因。

## 运行和交付

主worker185586于06:41:17.003405 UTC启动，06:42:46.130354结束，native0；replay186085于06:42:46.131225启动，06:42:47.934071结束，native0。科学累计90.930889s，含一次保存后4000重放；本地CDF标题重绘原生命令2.307306s保守计入，总93.238194s。两臂训练24.159254/47.007499s，累计71.166753s；RSS峰2,814,590,976B、GPU allocated峰97,098,752B，均在预算内。未发生科学进程失败；原重叠标题图保留，只从已保存CDF值重绘并实际查看新版。

step0全部599密度误差4.26e−14nat、旧2000波形preADC误差1.11e−16V/postADC完全一致；小批PSD梯度有限且非零，冻结参数无梯度/优化器更新。两个新臂4000条重放pre/post均逐位一致。迁移仅正常传输/加载，未做迁移内容一致性测试。

八张选定图已实际目检；源、配置、权重/优化器/RNG、实际验证与主输入、完整私有结果、聚合指标、资源/QA/重放回执及最终清单均在本目录。详细复现入口见REPRODUCTION.md。原目标域诊断得到部分改善，下一科学任务须由根代理围绕G4输入条件与源类别保真重新登记，本执行器不自动追加训练。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
repro=f'''# R17 reproduction

Executed server directory: `{cfg['remote_root']}` on rylan@100.82.231.102, Python `{cfg['execution']['python']}`. Main entry `numeric_worker.py`, supervised by `remote_supervise.py`; independent saved-weight entry `replay.py`. CPU4, training cuda:0, final/validation/replay float64 CPU. All actual versions are in metrics.json; no installation. Native bridge e4309b→0d1f9e exit0, worker185586/replay186085; transfer a733d4 exit0, ordinary copy with no migration equality test.

R14 TRAIN_ADDITION.npz/representation.npz and EXPOSURES weights are reused read-only; R15 RESIDUAL_FLOW.pt is reused; R16 completion_001/GMM8_BASE.npz, ACTUAL_GENERATION_INPUTS.npz, saved density terms and GMM8 generation are frozen inputs. R10 inputs provides processed old REAL TRAIN/DEV and role/QC/group controls; numeric view only first5200 REAL rows, not CAL/EVAL. INPUT_BINDINGS.json/config.json preserve existing input version references; originals were not rehashed for transfer.

Local authoritative result directory is `{ROOT}`. Initial/final/selected head, optimizer and actual RNG states are under NLL_CONT/ and NLL_PSD/. TRAINING_TRACE/RECEIPT records selection. VALIDATION_GENERATION_INPUTS.npz is the independent fixed2000 validation draw; ACTUAL_GENERATION_INPUTS.npz contains actual shared R16 latent/tailwhite/draw IDs. Private generated.npz includes preADC float64 and postADC float32 physical V. Private PSD/C2ST matrices/predictions remain local, not the aggregate Git payload.

Replay evidence: REPLAY_RECEIPT.json, fresh PID186085,4000 rows/1,920,000 points, pre/post bitwise. Replay loads SELECTED_HEAD.pt and the saved actual latent/tailwhite, not a reseeded substitute. To reproduce in a **new** permitted directory, copy current sources/config/saved selected heads/actual inputs using ordinary transport, update only filesystem paths to immutable upstream inputs, then invoke existing Python `-B replay.py`. Training reproduction additionally uses frozen initial head, TRAIN ordering, fixed targets, config seeds and saved initial RNG; environmental/version changes must be reported. The original directory has a consumed launch token and must not be resubmitted/overwritten.

No new scientific execution is requested by this document. It does not replay the upstream entire R11–R16 history or claim recovery of undocumented early initialization. This round is target-density sampling with no G4-source PSD/particle condition, so reproduction is not proof of G4→REAL physical mapping or source class preservation.
'''
(ROOT/'REPRODUCTION.md').write_text(repro,encoding='utf-8')
# Pure-byte final artifact hashing; no array decoding or old inputs opened.
excluded={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'};files=sorted(p for p in ROOT.rglob('*')if p.is_file()and p.name not in excluded);payload_bytes=sum(p.stat().st_size for p in files)
if payload_bytes+128*1024>cfg['output_limit_bytes']:raise RuntimeError('stage cumulative file output cap')
resource['actual_local_stage_bytes_before_final_control_files']=payload_bytes;resource['includes_upload_return_tar_and_transient_cache_bytes']=True;resource['final_manifest_receipt_reserved_bytes']=128*1024;resource['actual_output_with_final_control_reserve_within_cap']=payload_bytes+128*1024<=cfg['output_limit_bytes'];write('RESOURCE_RECEIPT.json',resource)
files=sorted(p for p in ROOT.rglob('*')if p.is_file()and p.name not in excluded);records=[]
for p in files:
    digest=hashlib.sha256();size=0
    with p.open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block);size+=len(block)
    records.append({'path':p.relative_to(ROOT).as_posix(),'bytes':size,'sha256':digest.hexdigest()})
write('FINAL_MANIFEST.json',{'stage':'R17','UTC':utc(),'files':records,'member_count':len(records),'member_bytes':sum(r['bytes']for r in records),'exclusions':sorted(excluded),'includes_transient_cache_and_return_tar':True,'hashing_only_current_stage_bytes':True,'migration_integrity_test':False,'frozen_scientific_inputs_not_reopened':True})
manifest_sha=hashlib.sha256((ROOT/'FINAL_MANIFEST.json').read_bytes()).hexdigest();write('FINAL_RECEIPT.json',{'stage':'R17','status':'sealed_complete_with_scientific_residuals','seal_PID':os.getpid(),'seal_UTC_before_last_write':utc(),'native_exit_proof':'external tool command return; no fabricated subprocess code','manifest_sha256':manifest_sha,'manifest_member_count':len(records),'manifest_member_bytes':sum(r['bytes']for r in records),'scientific_status':'conditional_PSD_improved; complete_response_and_source_class_preservation_not_verified','source_conditioning':'none; target-density draws only','EVAL_read':0,'next_scientific_task_started':False,'self_reference_exclusions':sorted(excluded),'seal_deadline_UTC':cfg['deadline_UTC']})
last_write=datetime.datetime.fromtimestamp(max(p.stat().st_mtime for p in ROOT.rglob('*')if p.is_file()),datetime.timezone.utc).isoformat();print(json.dumps({'seal_PID':os.getpid(),'manifest_sha256':manifest_sha,'manifest_members':len(records),'manifest_bytes':sum(r['bytes']for r in records),'actual_last_write_UTC':last_write,'seal_elapsed_s':time.monotonic()-start,'report_sha256':hashlib.sha256((ROOT/'REPORT.md').read_bytes()).hexdigest(),'compact_sha256':hashlib.sha256((ROOT/'METRICS_COMPACT.json').read_bytes()).hexdigest(),'resource_sha256':hashlib.sha256((ROOT/'RESOURCE_RECEIPT.json').read_bytes()).hexdigest(),'final_receipt_sha256':hashlib.sha256((ROOT/'FINAL_RECEIPT.json').read_bytes()).hexdigest(),'output_total_bytes':sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file())}),flush=True)
