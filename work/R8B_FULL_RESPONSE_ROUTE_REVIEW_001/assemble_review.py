from __future__ import annotations
import ast,datetime,hashlib,json,os,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT.parent.parent
def write(name,value):
 text=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
 (ROOT/name).write_text(text,encoding='utf-8')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def binding(path):
 b=path.read_bytes()
 return {'path':str(path),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
start=read(ROOT/'START_RECEIPT.json')
headers=read(ROOT/'INPUT_HEADERS.json')
roles=read(BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/results/DATA_ROLES.json')
old=pathlib.Path('E:/GAN/fullwave_execution_20261004_0930/executed_source/gan')
sources=[
 (BASE/'AGENTS.md','current simplified scope'),
 (BASE/'tasks/R8B_FULL_RESPONSE_ROUTE_REVIEW_001.md','current bounded review'),
 (BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/run.py','33-51 clean/condition; 54-101 identity/role/cache'),
 (BASE/'work/R3B_ENERGY_CONDITIONAL_WIDTH_001/model.py','18-103 descriptor-only generator and fixed family; 132-158 losses'),
 (BASE/'work/R8_LOCAL_JOINT_FLUCTUATION_001/model.py','8-40 peak-gated signal process'),
 (BASE/'work/R5_PREFIX_RECONSTRUCTION_UTILITY_001/run.py','42-67 saved noisy SIM TRAIN waveform'),
 (BASE/'work/R5_PREFIX_RECONSTRUCTION_UTILITY_001/sim_generator.py','fixed source scintillation formula and preprocessing'),
 (BASE/'work/R1_RESPONSE_SIMILARITY_BASELINE_001/REPORT.md','already executed W0 complete distribution/texture/noise failure'),
 (BASE/'work/R8_LOCAL_JOINT_FLUCTUATION_001/REPORT.md','already executed paired R8 whole-response failure'),
 (old/'scintcpunet/response_network.py','TaskMultiscaleResponse; signed normalized .15 residual; stochastic switch'),
 (old/'scintcpunet/response_losses.py','content/geometry/tail and 10% charge-drift constraints remain in W0'),
 (old/'scintcpunet/response_train.py','135-150 objectives and event preservation weight'),
 (old/'scintcpunet/response_data.py','122-147 noisy normalized waveform and peak conditions'),
 (old/'configs/fullwave/control.yaml','W0 stochastic_head false; preservation weight0; other constraints nonzero'),
]
source_bindings=[dict(binding(p),scope=s) for p,s in sources]
write('SOURCE_EVIDENCE.json',{'sources':source_bindings,'NPZ_evidence':'INPUT_HEADERS.json; current stat/headers only; existing container SHA reused without fresh whole-file verification','current_scientific_execution':False})
inputs={
 'TRAIN_clean':{'path':headers['files'][0]['path'],'bytes':headers['files'][0]['bytes'],'historic_sha256':'117152db28a16fcfc9a83012cc863e32f2d73bdef77bbf7b5f8e977c3e64818b','historic_sha_source':'R5 READINESS local_input_bindings','current_full_hash':False,'fields':headers['files'][0]['members'],'unit':'V; aligned480 points at1ns; CFD index120','source_N':8192,'fit_N':6593,'internal_validation_N':1599,'DEV_overlap':0,'waveform_origin':'EE and quenched hadron total deposits -> fixed512-point scintillation -> calibrated finite-window area -> integer CFD ->480-point clean window','time_resolved_G4_steps_available':False,'clean_waveform_information':'deterministic function of EE/hadron totals, hence of E_visible and h for E_visible>0; full wave is a structured description, not new independent timing information'},
 'TRAIN_noisy_existing_subset':{'path':headers['files'][1]['path'],'bytes':headers['files'][1]['bytes'],'fields':headers['files'][1]['members'],'role':'2000 parents selected from same8192 G4 TRAIN in R5; own internal-fit/validation membership must be inherited by parent identity, not reassigned','unit':'V','noise_prior_REAL_events_historically_used':79793,'clipping_in_R5_report':'248/2000; preserved, not discarded after viewing','primary_choice':False,'reason':'Available as source waveform reference; avoid making the current source-only review a new legacy-noise generation or switching all8192 to a2000 mixed subset'},
 'DEV_SIM_existing':{'path':headers['files'][2]['path'],'bytes':headers['files'][2]['bytes'],'fields':headers['files'][2]['members'],'role':'fixed2000 old SIM parents; historically exposed DEV','physical_waveform':'raw[i]*normalization_peak_V[i]; original baseline-subtracted/noisy/cropped SIM V proxy, not raw long trace','primary_clean_input':'reconstruct the same R2 clean formula from bound DEV EE/hadron in the subsequent authorized stage; retain same parent IDs/order; not new G4','source_label':'legacy neutron bucket2112 vs non2112 gamma bucket; not REAL truth'},
 'REAL_TRAIN_DEV':{'waveform_path':str(BASE/'work/P1K_REAL_REFERENCE_AND_REPEATABILITY_001/A2_ENGINEERING_001/run_001/window_V_f32.bin'),'stored_shape':[6800,480],'scope_for_next_fit':'only first5200 rows: DEV2000 + TRAIN3200; never use EVAL','waveform_dtype':'<f4','unit':'V; common1ns480-point window','TRAIN_QC':3191,'fit':2592,'internal_validation':599,'artificial_groups':'original_index//50; same acquisition, not independent run','CAL_reference':'existing frozen charge boundaries/reference only; q4 remains failed/allunknown; no fallback'},
 'original_source_role_receipt':roles,
 'this_stage':{'NPY_headers':21,'uncompressed_header_bytes':2688,'compressed_prefix_bytes_returned':5376,'scientific_body_inflated_or_decoded':0,'source_identity_strings_decoded':0,'NumPy_torch_imports':0,'fit':0,'generation':0,'EVAL':0,'SSH':0},
}
write('INPUT_STRUCTURE.json',inputs)
proposal={
 'status':'draft_for_root_next_stage_registration_not_executed','question':'Can a full-waveform-conditioned stochastic empirical map reduce the complete REAL/SIM gap while retaining source dependence?','primary_name':'FULL_WAVE_STOCHASTIC','one_ablation_name':'PULSE_STOCHASTIC_OFF','not_a_frontend_patch':True,'input':'all8192 existing clean480-point G4 TRAIN waves and c/Q; same R2 masks; no raw step timing claimed','REAL_targets':'only2592 TRAIN fit;599 grouped TRAIN validation; no pair labels or paired MSE','initialization':'fresh paired generator/critic initial parameters with seed set before constructors; save actual Torch/NumPy/train RNG states and copied initial bytes; old R3B/R8 weights remain reference inputs only',
 'representation':{'wave_grid':'input4801ns CFD-aligned in V; pad8 zeros before and24 zeros after to512 source-description slots, with a known-support mask; padding is not observed extra data','embedding':'x/.1V + support mask + relative grid coordinate, c4 and logQsource broadcast','encoder_decoder':'stride1 Conv1d kernel5,24channels,seven residual dilated blocks[1,2,4,8,16,32,64],SiLU; no transposed conv/strided upsampling/absolute sinusoid transformer/no batch normalization','mean':'mu=exp(g(c))*padded_source + .1V*f_mu(encoder,c); learned signed complete residual without .15 tanh bound, fixed exponential kernel, nonnegative clean constraint, RBF multiplicative correction or source FoM preservation','stochastic_pulse':'one16-D event latent plus independent512-D white into learned stride1 conditional temporal branch; .1V is a unit scale, not an amplitude cap; free throughout recorded domain, no clean-peak gate or sqrt law','baseline':'separate zero-mean31-tap learned conditional FIR driven by512-D white; condition/event latent may modulate RMS and correlation jointly with pulse; use TRAIN-fit baseline statistics for initialization only, filters remain learnable','interpretation':'mu is a baseline response branch, not automatically an identifiable physical conditional expectation; learned latent covariance is empirical, not photon statistics','output':'512 V -> one14bit[-.5,.5] ADC (STE during fit) -> original512 first20% baseline102-point CFD rule ->480 index120 window; padding/clipping/counts and invalid rows retained; no output peak normalization except separately declared shape loss/evaluation'},
 'loss':{'generator':'1*hinge_adversarial +1*full_joint_SW +2*soft_charge_CDF +.5*charge_occupancy +.5*baseline_joint','training_critic':'independent small spectral-normalized multiscale1-D CNN,24channels, no REAL source PDG/particle pseudolabel/c input; sees unit-bearing waveform and own-shape jointly; one critic step per generator step, hinge loss; not the heldout C2ST','full_joint_SW':'TRAIN-fit scaled vector of scalar Qt/PSD/peak/rise/tail plus complete unit-bearing480 curve, own-peak shape, first and second differences;128 fixed unsmoothed projections seed2026100892, blocks balanced by sqrt dimensionality; compare batch distributions only','soft_charge_CDF':'reuse R3B eight TRAIN-fit logQ windows/grids and unknown-inclusive conditional PSD CDF; do not maximize FoM or force1/sqrtE','baseline_joint':'joint RMS/lag1..8ACF/nonDC spectral shape with Qt/peak; TRAIN-fit scales; no paired waveform target','identity_or_source_PSD_preservation':0,'cycle':0,'random_event_MSE':0,'REAL_template_sampling':0,'validation_selection':'every100 including0, fixed384 TRAIN internal validation sources/REAL/noise; minimize fixed1SW+2CDF+.5occupancy+.5baseline only, no GAN critic loss/DEV C2ST; hold all projections/scales fixed'},
 'paired_ablation':{'mechanism':'same complete-wave architecture and loss, pulse event latent and pulse white disabled; baseline noise remains; draw then discard the same z/white so all source/REAL orders and RNG streams stay paired','same_initial_learnable_parameters':True,'same_updates':True,'same_TRAIN_fit_statistics':True,'purpose':'tests incremental conditional pulse variation within this full-wave route; existing R3B/R8 remain descriptive architecture controls, not a controlled proof of architecture causation','no_architecture_seed_weight_grid':True},
 'source_dependence_diagnostics':{'fixed_main_positions':'64 evenly spaced prespecified positions floor(j*1999/63),j0..63','control_1':'shuffle entire source bundle x,c,Q together by fixed permutation seed2026100893; keep z/baseline white/pulse white unchanged','control_2':'replace entire bundle by first allowed TRAIN-fit source in full-SHA parent ordering, keep noises unchanged; SIM-only control, not a REAL template','additional_outputs':128,'report':'paired waveform L2 and signed Qt/PSD/peak changes, source-to-output rank/conditional associations, original/all-donor parent provenance; source-ignoring if both controls leave all64 outputs bitwise invariant, otherwise magnitude is descriptive without invented physical tolerance','claim_boundary':'sensitivity demonstrates use of source, not unique correct per-event correspondence'},
 'joint_evaluation':['complete unweighted QC+unknown distributions','Qt-PSD2D, width/mean/occupancy/detrended width on fixed supported bins','unit-bearing and own-shape full waveform plus curve bands','baseline RMS/ACF/spectra and signal curvature by charge','source-nominal all parents and output-charge strata/migration separately','fixed independent HGB fullshape/fullvoltage plus strict local views; train/test split by parent/artificial REAL group','same-N REAL TRAIN-DEV reference','all invalid/clipped/padded/support failures','saved weights/random input independent main replay'],
 'references':['original SIM','old GEN_W0 with its proxy-scale limitation','R3B','R8 JOINT','R8 JOINT_SIGNAL'],
 'source_mixture_policy':'no posthoc reweighting/HC filtering; original G4 label/energy mix and REAL unlabelled mix can differ; complete population and observable-charge comparisons answer different questions; no claim that domain matching calibrates Geant4 particle labels/source spectrum',
 'budget_proposed':{'stage_s':7200,'numeric_cumulative_s':1800,'response_updates_per_arm_max':2000,'critic_updates_per_arm_max':2000,'single_seed':2026100891,'batch':64,'generator_lr':.0002,'critic_lr':.0002,'Adam_betas':[.5,.9],'weight_decay':.0001,'threads':4,'RSS_bytes':4294967296,'new_output_bytes':536870912,'new_HGB_max':12,'new_density_diagnostics_max':16,'new_generation_rows_main_and_extra':12000,'source_control_generation_rows':128,'SSH':0,'new_G4':0,'EVAL_before_freeze':0,'GPU_required':False,'performance':'runtime/memory feasibility is an unbenchmarked bounded estimate, not measured here; existing Torch CPU environment suffices to attempt; no install or remote migration'},
 'stop_rules':['hard wall/RSS/output/update caps','nonfinite loss/output or unrecoverable input role mismatch -> preserve failure, no recipe search','source input-insensitivity -> report failure of mapping requirement even if marginal similarity improves','source mixture/support conflicts cannot be silently solved by relabelling or reweighting','after one recipe/pair all scientific outcomes preserved; remaining gap is a negative result, not authorization for another patch/search','no EVAL for model choice; final frozen retrospective benchmark only after root registration'],
 'identifiability':'unpaired marginal agreement allows many source-to-output couplings and stochastic laws; full input being a deterministic source description adds an inductive bias, not missing G4 time information or independent detector calibration',
}
write('PROPOSED_CONFIG.json',proposal)
write('CONTROL_FAILURE_001.json',{'failure':'functions.exec ReferenceError: root is not defined after applying inspect_headers.py','native_header_subprocess_PID':'unknown','native_header_subprocess_exit':'unknown','followup_tool':'5e9f5c/native0','postcheck':'no INPUT_HEADERS.json and no matching inspect_headers.py process then visible; no success inferred for the discarded promise','later_explicit_header_command':'4217d2/PID5456/native0','mathematical_execution':0,'scientific_array_decode_by_header_program':0,'no_scientific_retry':True})
report='''完整 SIM 波形输入在本机已经可用，因此下一轮不需要等待服务器、读取原远端大库或识别单光电子响应。建议只尝试一个完整波形条件随机映射：原波形经过可学习的全时间卷积响应，并带可学习的事件级与局部随机过程；与关闭脉冲随机分支的一臂配对。它针对整组 REAL/SIM 的波形、条件分布和涨落差距，不把某个前沿核或峰后噪声开关当作整体路线。

输入事实与证据边界

R2 的 G4_TRAIN_CACHE.npz 为11,535,795B，已有历史SHA117152db28a16fcfc9a83012cc863e32f2d73bdef77bbf7b5f8e977c3e64818b；本轮只核对大小及头，未重哈希整个容器。clean_expectation_V 是 float32[8192,480]，单位V、1ns采样；四条件[8192,4]、Qsource、EE/hadron总沉积、可见能量/组成、PDG和父身份及内部划分都已保存。既存6593/1599 G4内部划分、2592/599实测内部划分继续使用，DEV仍固定2000，不把历史DEV改成TRAIN。R5另存完整SIM_V[2000,480]的有噪声TRAIN子集；248条量化截幅和79793条历史实测噪声先验成本属于旧事实，不换名为从3191条纯净建模。

缓存clean是固定闪烁配方从两类总沉积构造512点，再裁成480点：ee及.15*had分别进入固定[13,35,270]/[13,59,460]ns核，固定5ns上升，按(Evisible+.03918)/.000132506校准记录积分。没有step时间/位置/到达光子时间字段。Evisible和h在正能量时可反解两总沉积，clean因而也是四条件中E/h的确定函数。完整波形输入只改变结构化表示和学习偏置，不恢复被四条件舍弃的独立真实G4时序；声称它必然提供更多信息是错误的。

原DEV evaluation_inputs.npz 的raw[2000,480]乘同一SIM normalization_peak_V能恢复历史SIM扣基线/裁窗后的V，不能恢复未知REAL峰或未裁长波形。新主路线为了TRAIN/DEV输入配方一致，TRAIN使用缓存clean，DEV在下一授权阶段用同R2函数/既存EE-hadron构造对应clean，不用真实模板、不跑新G4。原有噪声SIM作为独立旧参照，不把clean和legacy noisy默默等同。

为什么四条件解析族与旧W0都未解决整体问题

当前R3B/R8 forward只接受c4、Qsource、z和white，没有接收cache clean波形。c=[log(Evisible)-center,h,cleanPSD,cleanTail]；完整480点的信息只有上述固定族的摘要。源码能证明：共同128ns onset与±.5tanh(z5)相位、非负差指数混合、2–14ns rise参数、有限范围衰减/宽度、乘法RBF及面积归一化预先限制了干净族。乘法纠正无法在原shape为零处造出干净响应；宽RBF限制细尺度变化的形式。平稳基线谱在TRAIN估计后固定，只有RMS分布学习；R8三抽头信号过程又被clean峰后10ns开关及sqrt包络限制。这些是结构事实；它们是否是全部失配的唯一原因，源码和R8结果都不能证明。

R8已完成：完整shape BA仍0.996，JOINT_SIGNAL的postpeak BA从JOINT的0.916降至0.826，但同支持bin2–7的两侧PSD宽度误差未改善。prompt100:150还含峰与早期下降，不能把其中曲率变化直接叫纯上升沿改善/恶化。局部修正不能替代整组相似性。

旧GEN_W0已经接收了完整noisy480点；它不是“从未用过全波形”的对照。response_network是多尺度CNN+位置Transformer+skip decoder，mean=x+.15*tanh(correction)，没有独立幅度输出，评价V默认乘SIM父峰。实际control.yaml关闭stochastic_head；event_preservation_weight=0也没有关闭content5、geometry2、low-tail1与超过10%电荷漂移罚。因此W0还保留其他工程结构，不能简单叫完全无约束或物理反事实。

R1既存结果证明W0总体PSD W1从SIM0.104627降到0.041618，而shape BA仍0.996；固定时刻纹理/振铃、窄条件带、baseline ACF1约0.477而REAL约-.031，低侧占比23.75%而REAL47.95%。完整输入本身不保证成功，stride1也不保证永不学出纹理。本方案与W0的实质差别是：完整有单位输入和可学习尺度、无.15残差界/父峰默认、全记录随机自由度与显式联合分布约束、独立完整波形验收及源依赖诊断，而不是给相同旧结构改名。对固定纹理的架构起因没有在本轮做新因果试验。

一个主方案和一个必要消融

主方案FULL_WAVE_STOCHASTIC接收完整clean480点及c/Q，不直接使用PDG作为实测真值。用24通道、kernel5的七个stride1扩张卷积块(1/2/4/8/16/32/64)，保留全域分辨率；不依靠差指数/RBF、跨区域峰后开关或反卷积上采样。将输入以已知mask扩至512，仅为兼容共同ADC/CFD接口；额外padding不是新增观测。均值基准是学习gain乘源波形加无.15硬界的signed全域残差，容许前沿、峰、尾和undershoot一起改变。独立16维事件latent和全时域white驱动条件时域随机分支；31抽头可学习基线FIR也由TRAIN初始化，RMS/相关性可与事件条件联动，不锁死Welch谱或指定sqrt(E)法律。所有分支返回真实V，再用一次14bit ADC、既有CFD和480窗，不按父峰默认缩放，不强制保持SIM FoM。

这仍是完整已处理波形上的经验随机映射，不是已识别的PMT单光电子卷积、唯一电子学传递函数或经过物理标定的光子统计。H3165-10/DS9000及20GS/s由根代理现有Gamma记录提供背景，不能从这些型号给未知响应常数补造数值，也不是这轮模型设计的前置阻塞。

训练采用一个小型谱归一化多尺度CNN的hinge分布critic，再加完整unit-V/own-shape/一二阶差分和Q/PSD/peak/rise/tail的联合SW、固定软电荷窗PSD CDF/占比、基线RMS/ACF/谱联合约束。不是随机事件MSE、真实波形模板匹配、粒子伪真值监督或FoM最大化。训练critic不是评价C2ST；后者另用冻结HGB及父/人工组留出。所有投影/尺度、CDF目标仅TRAIN-fit，一次固定；每100步用固定TRAIN验证384和非critic分布分数选checkpoint，DEV只报告。

唯一配对消融PULSE_STOCHASTIC_OFF保留相同完整表示、全部损失及基线随机过程，关闭事件latent与脉冲white自由度；两臂仍抽取相同随机流后丢弃关闭项，并共用初值、父/目标顺序、更新数和合法TRAIN统计。它检验完整映射内脉冲随机自由度的作用；旧R3B/R8只作既有参照，不能用不同架构的总差异当作严格机制归因。不再给多个前沿核、FIR长度、种子或系数网格。

整体匹配和输入关联是两个要求

在主生成2000中预定64个等间隔父位置，固定所有随机数，一次打乱整个源包x/c/Q，一次用按完整SHA排序的首个允许TRAIN-fit SIM源替换。额外仅128输出；保留原父/供体父身份，报告波形L2、电荷/PSD/peak改变与源到输出秩/条件关联。所有64输出均不变时必须报告当前检查下忽略源；非零变化只证明使用了源，不证明映射逐事件物理正确。不可仅打乱wave而不打乱条件制造不一致输入。

无配对REAL/G4两组存在多种可能耦合，即便完整边缘分布一致也不唯一识别输入—输出对应、潜变量或探测器物理。原源谱/混合与实测触发/能谱可不同；REAL没有独立粒子标签，不能把PSD两密度成分直接当真实n/g配比，并用生成器“纠正”PDG真值。主评价保留完整不加权QC总体/unknown，同时并列输出自己电荷桶与源名义桶全部父输出、迁移矩阵、条件宽度/占比/不支持理由。若条件相似而总谱仍不同，两项都报告，不暗中重加权或删HC困难区。

预计执行及停止

建议下一次只登记这一个两臂配方：每臂最多2000 G更新和2000 critic更新，batch64，Adam lr.0002/betas(.5,.9)，主seed2026100891，TRAIN选点；数值累计1800s、120min总阶段、4线程/4GiB/512MiB，至多12新HGB、16新密度描述拟合、12000主/附加视图+128源控制输出，保存权重/RNG后独立主2000重放。具体完整草案在PROPOSED_CONFIG.json，当前全部是未执行的下一阶段建议。

本地既有Torch CPU足以尝试这个轻量配方；速度/峰RSS尚未测量，不能把估算说成保证。4GiB/墙钟不足时保留终态，不因服务器已上线而重提或搬已封存作业。当前AGENTS仍local-only；本轮也未SSH。下一科学任务若根代理改变运行位置，应另登记同一配方与资源，不安装环境或悄悄切输入。

预算/非有限/角色冲突即停止并保留失败；若完整分布仍明显可区分、宽度失配或源依赖塌缩，记录负结果并停在这一个实验，不把科学残差当实现错误继续调配方。CAL q4失败/q1高PSD支持缺口保持，旧EVAL仅在最终冻结后回顾性比较，绝不选参。这次20分钟复审未拟合、生成、重新算指标或解码科学body。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
ast.parse((ROOT/'inspect_headers.py').read_text(encoding='utf-8'))
ast.parse((ROOT/'assemble_review.py').read_text(encoding='utf-8'))
now=datetime.datetime.now(datetime.timezone.utc)
assert now<datetime.datetime.fromisoformat(start['deadline_UTC'].replace('Z','+00:00'))
files=[dict(binding(p),path=str(p.relative_to(ROOT))) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']]
size=sum(r['bytes'] for r in files);assert size<5*1024*1024
write('FINAL_MANIFEST.json',{'files':files,'bytes':size,'self_exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'source_only':True})
receipt={'status':'source_only_route_review_complete','PID':os.getpid(),'start_UTC':start['start_UTC'],'stop_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_UTC':start['deadline_UTC'],'headers_native_tool':'4217d2','headers_PID':5456,'headers_exit':0,'science':inputs['this_stage'],'report':binding(ROOT/'REPORT.md'),'proposed_config':binding(ROOT/'PROPOSED_CONFIG.json'),'input_structure':binding(ROOT/'INPUT_STRUCTURE.json'),'manifest':binding(ROOT/'FINAL_MANIFEST.json'),'native_exit_recorded_by_calling_tool':True,'writes_stopped_after_receipt':True}
write('FINAL_RECEIPT.json',receipt)
print(json.dumps(dict(receipt,last_write_UTC=datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat(),total_files=len(files)+2,total_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())),ensure_ascii=False),flush=True)
