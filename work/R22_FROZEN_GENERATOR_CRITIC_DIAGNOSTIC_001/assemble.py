from pathlib import Path
import os,json,datetime,hashlib,ast
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf8')
cfg=read('config.json');history=read('RUN_HISTORY.json');m=read('METRICS_COMPACT.json');tr=read('TRAINING_RECEIPT.json');pool=read('POOL_RECEIPT.json');gpu=read('GPU_ENVIRONMENT.json')
raw=m['critics']['RAW']['selected']['pre']['val']['equalbin_TILT_weighted'];std=m['critics']['STANDARDIZED']['selected']['pre']['val']['equalbin_TILT_weighted'];ci=m['bootstrap']['STANDARDIZED_minus_RAW_pre']['CI95']
names=['training_trace','validation_scores','classifier_comparison','per_bin_classification','fixed_pool_support','scaler_and_gradients','ADC_sensitivity']
qa={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'executor visual view_image; all7 actually displayed','figures':[{'path':'figures/'+n+'.png','viewed':True,'axes_legends_complete':True,'no_overlap_or_clipping':True}for n in names],'scientific_recompute_for_QA':False}
write('FIGURE_QA.json',qa)
resource={'stage':'R22','UTC':qa['UTC'],'science_known_native_wall_s':sum(r['elapsed_s']for r in history),
 'environment_check_elapsed_s':None,'environment_check_time_unknown_budget_reservation_s':60,
 'science_budget_debit_with_check_reservation_s':60+sum(r['elapsed_s']for r in history),'science_limit_s':1200,
 'D_training_s':tr['training_elapsed_s'],'D_training_limit_s':600,'D_updates':8000,'D_updates_per_arm':4000,'G_updates':0,'reader_updates':0,'generation_n':4048,'generation_limit':4096,
 'HGB_fits':2,'HGB_limit':2,'new_EM_fits':0,'CPU_threads':4,'CPU_peak_RSS_bytes':max(r['RSS_peak_bytes']for r in history if r['python']==cfg['CPU_python']),
 'GPU_actual_worker_RSS_bytes':None,'GPU_launcher_RSS_bytes':4214784,'RSS_budget_verification_status':'CPUverified; actualGPUworker unknown because venv launcher monitoring; no claim overallRSSpass',
 'GPU_actual_worker_PID':44800,'GPU_launcher_PID':57308,'GPU_allocated_peak_bytes':tr['GPU_peak_allocated_bytes'],'RSS_GPU_limits_bytes':4*1024**3,
 'numeric_failures':[{'entry':'train_critics.py','native_exit':1,'elapsed_s':history[1]['elapsed_s'],'launcher_PID':61204,'actual_science_PID':None,'D_updates':0,'G_updates':0,'error':'NumPy uint32 seed API rejects fullseed; full SeedSequence mapping repair'}],
 'control_failure':{'tool':'f68dc6','native_exit':1,'PID':None,'error':'AST SyntaxError else0 before scientific execution','scientific_fit':0},
 'all_native_history':history,'GPU_environment':gpu,'CPU_environment':{k:pool[k]for k in ['python','torch','numpy','platform']},
 'EVAL_numeric':0,'DEV_numeric':0,'CAL_numeric':0,'new_raw_long_decode':0,'SSH':0,'installation':0,'new_G4':0,'new_agents':0,'Git':0,
 'old_REAL_read':'seek rows2000:5200, no read of DEV/CAL waveform range','REAL_fit_scaler_population':21047,
 'support_excluded_sourcefit':6593-3753,'support_excluded_REALfit':21047-18895,'support_excluded_sourceval_fixed':512-295,'support_excluded_REALval':599-536,
 'bootstrap_attempted':500,'bootstrap_complete8bin':489,'bootstrap_missingbin':11,'REAL_artificial_val_groups':12,
 'output_cap_bytes':192*1024**2,'current_stage_file_bytes_before_final_controls':sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file()),
 'time_output_counts_within_limits':True,'resource_monitoring_defect_retained':True,'single_science_process_sequential':True,'no_rerun_to_fill_missing_RSS':True}
write('RESOURCE_RECEIPT.json',resource)
report=f'''R22 表明原训练 critic 在冻结生成池上已经能识别剩余域差。RAW 所选内部验证 BA={raw['BA_threshold0.5']:.6f}、AUC={raw['AUC_REAL1']:.6f}；固定逐坐标中心化与尺度标准化后 BA={std['BA_threshold0.5']:.6f}、AUC={std['AUC_REAL1']:.6f}。这否定了本诊断池中“原结构完全看不到残差”的解释，但没有解决实际转换质量，也没有证明在线对抗训练的唯一失效机制。

本轮将 R21 第2000步生成器完全冻结，只训练 RAW/STANDARDIZED 两个原同架构 critic。每臂4000更新，共8000D更新，生成器与style读取器更新均为0。两臂原R18初始权重、Adam、32样本/域及全部实际索引流相同；完整480点源输入保持，只有 critic 两通道输入的固定处理有差别。标准化把中心化和尺度变化作为一个干预包，不能再把收益归因于单独方差缩放。

|固定诊断|选中步|内部验证MSE|等Q/TILT BA|等Q/TILT AUC|
|---|---:|---:|---:|---:|
|RAW|4000|{raw['balanced_MSE']:.6f}|{raw['BA_threshold0.5']:.6f}|{raw['AUC_REAL1']:.6f}|
|STANDARDIZED|3750|{std['balanced_MSE']:.6f}|{std['BA_threshold0.5']:.6f}|{std['AUC_REAL1']:.6f}|
|HGB pre-ADC|固定100轮|{m['HGB']['pre']['val']['equalbin_TILT_weighted']['balanced_MSE']:.6f}|{m['HGB']['pre']['val']['equalbin_TILT_weighted']['BA_threshold0.5']:.6f}|1.000000|
|HGB post-ADC|固定100轮|{m['HGB']['post']['val']['equalbin_TILT_weighted']['balanced_MSE']:.6f}|1.000000|1.000000|

RAW 的训练/验证 BA 分别.917158/.922660，AUC .995938/.997843；标准化臂分别.999600/.997396、AUC均1。这不是只在训练行记住标签。未加权验证 BA 分别.924440/.998134，结论也保留。标准化相对RAW的配对 BA 差95%区间[{ci[0][0]:.6f},{ci[1][0]:.6f}]；AUC差区间[{ci[0][1]:.6f},{ci[1][1]:.6f}]。RAW的排序已经接近完全区分，因此标准化更大的BA/MSE收益也涉及LS分数校准和优化；不能仅据阈值BA说原critic没有信息。

RAW step250验证MSE约.249711，此后到step4000下降至.068173，但step3250有.141887反弹。标准化step250已降至.114057，step3750为最优.017686，末点4000回升至.020131。末点标准化BA虽为1，仍按预定MSE选3750，没有改用分类分数选点。所有轨迹、每区评分及末点均在METRICS.json保留。

源TRAIN6593条中3753共同Q支持父各生成一次，原512内部源验证子集中295父各一次，共4048输出。REAL原21047/599分别有18895/536共同支持。支持外2840源TRAIN、2152实测TRAIN、217固定源验证、63实测验证没有用于critic训练或本次评分；不是宣布这些人口已匹配。源验证各区40/41/56/68/44/27/15/4，实测48/75/72/73/71/70/75/52，完整八区结果保持。最高源验证区仅4父，不能作稳健局部推广。

标准化常数只由全部21047 REAL TRAIN-fit计算：原Torch float32两通道先转换，随后float64均值/总体标准差。通道尺度下界约.00166102/.00091332，没有坐标触底。两个critic以及两个HGB都用同一固定父与生成输出，生成端按原源Q/TILT权，实测端按观察Q区均匀；生成后Q迁移不改采样区。HGB每域相对权重先归一为1，再用公共Ntotal/2缩放为总体均重1以保留原l2=1的尺度。pre/post仅不同ADC表示，不是新父或重复独立采集；本次ADC裁剪0，critic的ADC前后BA基本不变。

区间由500次固定分类器的源父/REAL人工50记录组重采样得到。组 multiplicity 在各区共用，保留同一实测组跨区耦合，再按区重新归一；11次缺区，489次用于完整八区条件区间。实测12组来自同一采集，不是独立run，区间不包含模型拟合、采集或响应训练不确定性。协议初稿的分区独立重采样在评价前已明确改为全域组 multiplicity；原说明与澄清均保留。

两臂在相同64内部源验证行的输入梯度全有限。mean((D(gen)-1)^2)按64条求均值，所报每行导数含1/64，单位V^-1。RAW/标准化整体梯度RMS .108213/.323508，基线.087062/.630006，峰后.124452/.053194。梯度包含基线/峰归一化耦合，不能解释为物理因果重要性；本轮没有更新生成器检验这些梯度的修复价值。

数值按CPU2.10冻结生成→现有CUDA2.12顺序critic→CPU HGB/汇总执行。GPU训练39.921s，全部有原生回执的科学进程共95.248927s；环境import/check精确耗时未单独记录，按60s上界预留计入预算，累计记155.248927s。GPU峰allocated307647488B。CPU池峰RSS735461376B、评价456089600B。GPU科学实际PID44800由venv launcher57308启动；监督器仅量到launcher约4.2MB，实际GPU worker RSS未知且4GiB RSS guard没有覆盖它，不能写成本轮RSS预算已完全通过。该缺口不补跑，未来CUDA监督须绑定实际PID/子树或自报RSS。

保留一次科学启动失败：legacy NumPy seed接口拒绝超32bit种子，4.936926s、D/G均0，使用完整SeedSequence映射修复，不改已保存采样索引。另有训练前AST空白拼写错误，属于纯控制失败；两者原回执/源码或错误信息保留。没有SSH、安装、新G4、DEV/EVAL/CAL数值或追加密度拟合。七张图已实际逐张查看，坐标、图例和支持缺口可读。

优先下一问题是在线对抗优化，而非再断言原critic架构完全无力。可在完整源→NN转换主线固定REAL TRAIN中心化/尺度常数，检验critic更新平衡是否使判别梯度真正减少独立完整波形/PSD差距，继续保持源内容与共同Q采样。此处仅提出方向，不执行新训练。强固定池分类能力不保证在线随机生成、对手变化和源保真约束下会收敛；本輪也不证明真实粒子类别、唯一物理映射或跨采集成功。
'''
(ROOT/'REPORT.md').write_text(report,'utf8')
(ROOT/'REPRODUCTION.md').write_text('''R22 uses only local E/D. config.json defines all fixed input paths, seeds and ceilings. Old stages remain read-only.

CPU Python: C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe (Torch2.10.0+cpu/NumPy2.2.6). Existing GPU Python: E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe (Torch2.12.0+cu130/NumPy1.26.4). No installation. All child TEMP/TMP/MPL/JOBLIB/TORCH/CUDA cache variables point into this stage/cache.

In a fresh output directory copy the scripts/config and preserve input paths. Run CPU `supervise.py prepare_pool.py <CPU_python>`, then CPU `supervise.py train_attempt2.py <GPU_python>`, then CPU `supervise.py assess.py <CPU_python>`. One scientific process at a time. Do not run in the sealed original directory; used tokens prevent duplicates. No stage was scientifically replayed just for missingRSS.

FIXED_POOL_PRIVATE.npz preserves complete parent IDs/roles/random draws/ADC versions/standardizer and identical4000 sampling rows. RAW/ and STANDARDIZED/ store selected/final critic+optimizer. CRITIC_SCORES_PRIVATE/ALL_SCORES_PRIVATE retain all predictions; INPUT_GRADIENTS_PRIVATE includes64 actual input indices; BOOTSTRAP_PRIVATE stores paired replicates. These arrays/weights/models stay local. METRICS_COMPACT is aggregate-only. HGB pre/post are own newly fitted models.

This is a finite frozen TRAIN/internal-validation pool, not a new held-out acquisition or onlineGAN demonstration. CUDA supervision monitored a venv launcher, not actual scientificPID44800: actualRSSunknown; fix that before future CUDA use. Preserve failed_001 and native history. CPU/GPU version differences are explicit, not claimed bitwise equivalence.
''','utf8')
write('TOOL_RECEIPTS.json',{'first_prepare':{'tool':'509bf7','PID':60920,'exit':0,'UTC':cfg['start_UTC']},'bootstrap':{'tool':'0046e9','PID':59272,'exit':0},
 'GPUcheck':{'tools':['781005','ce4aa2'],'actualPID':54968,'exit':0,'duration_s':None},'pool':{'tools':['30cf3e','ee8510'],'supervisorPID':39604,'workerPID':60240,'exit':0},
 'controlAST_failure':{'tool':'f68dc6','PID':None,'exit':1},'Dfailure':{'tool':'a0fe77','supervisorPID':42224,'launcherPID':61204,'actualPID':None,'exit':1},
 'Dsuccess':{'tools':['7222c8','dc1eec'],'supervisorPID':60468,'launcherPID':57308,'actualPID':44800,'exit':0},
 'assessment':{'tools':['e3d3b2','70dca8'],'supervisorPID':38216,'workerPID':42144,'exit':0}})
bindings=[]
for p in sorted(ROOT.glob('*.py')):
    b=p.read_bytes();ast.parse(b);bindings.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
write('EXECUTED_SOURCE_BINDINGS.json',bindings)
write('DELIVERY_INDEX.json',{'report':'REPORT.md','compact':'METRICS_COMPACT.json','full_local':'METRICS.json','resource':'RESOURCE_RECEIPT.json','figures':['figures/'+n+'.png'for n in names],'reproduction':'REPRODUCTION.md','private_payloads_stay_local':True})
print(json.dumps({'PID':os.getpid(),'UTC':qa['UTC'],'status':'assembled','seven_figures_actually_viewed':True}),flush=True)
