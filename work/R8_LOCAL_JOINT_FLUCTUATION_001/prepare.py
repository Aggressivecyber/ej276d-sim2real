from __future__ import annotations
import pathlib,json,hashlib,ast,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
R3B=PROJECT/'work/R3B_ENERGY_CONDITIONAL_WIDTH_001';R7=PROJECT/'work/R7_RESPONSE_RESIDUAL_DIAGNOSIS_001'
def j(p):return json.loads(p.read_text(encoding='utf-8'))
def bind(p):
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
old=j(R3B/'config.json');r7=j(R7/'config.json');start=j(ROOT/'START_RECEIPT.json');copied=[]
for name in ['evaluation.py','common_math.py','energy_width.py','diagnostic_core.py']:
    p=R7/name;(ROOT/name).write_bytes(p.read_bytes());copied.append(bind(p))
p=R3B/'model.py';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='5b4446ed033ce2018e952f17d516febf930e7f64c7ef5e0ba02770b18f127244'
text=b.decode('utf-8').replace('def forward(self,c,q_source,z,white,quantize=True):','def forward(self,c,q_source,z,white,signal_white=None,quantize=True):')
assert text.count('full=clean+rms[:,None]*noise')==1
text=text.replace('full=clean+rms[:,None]*noise','signal=self.signal_term(c,clean,signal_white)\n        full=clean+rms[:,None]*noise+signal')
assert text.count("'padding':(left<0)")==1
text=text.replace("'padding':(left<0)","'clean_V_512':clean,'signal_delta_V_512':signal,\n              'clean_peak_index':clean.argmax(dim=1),'signal_scale_V':self.signal_scale(c),\n              'signal_delta_RMS_V':signal.square().mean(dim=1).sqrt(),\n              'padding':(left<0)")
(ROOT/'response_base.py').write_text(text,encoding='utf-8');copied.append(bind(p))
launch=(R7/'launch.py').read_text(encoding='utf-8').replace('R7_diagnostics_started','R8_response_started').replace("'response_updates':0,","'response_updates':'see per-arm training receipts',").replace('fixed_response_residual_diagnosis','paired_joint_signal_response')
launch=launch.replace("str(ROOT/'run.py')","str(ROOT/(sys.argv[1] if len(sys.argv)>1 else 'run.py'))")
(ROOT/'launch.py').write_text(launch,encoding='utf-8');copied.append(bind(R7/'launch.py'))
keys=['model','generation_inputs','generation_rng','generated','REAL_V','REAL_QC','REAL_groups','REAL_original_index','REAL_roles','CAL_reference','TRAIN_roles','SIM_identity','width_definition']
files={k:r7['input_files'][k] for k in keys};files['G4_TRAIN_cache']={**old['R2_cache'],'bytes':pathlib.Path(old['R2_cache']['path']).stat().st_size}
for k,n in [('R7_metrics','METRICS_COMPACT.json'),('R7_selections','results/DIAGNOSTIC_SELECTIONS.json'),('R7_predictions','results/C2ST_HOLDOUT_PREDICTIONS.npz')]:files[k]=bind(R7/n)
cfg={**start,'seed':2026100881,'joint_projection_seed':2026100882,'signal_main_seed':2026100883,'extra_view_seeds':[2026100884,2026100885],
 'validation_seed':old['validation_seed'],'validation_n':384,'old_projection_seed':old['seed'],'signal_validation_seed':2026100886,
 'max_updates_per_arm':2500,'batch_size':256,'learning_rate':.0005,'training_wall_s_per_arm':900,'numeric_wall_limit_s':2400,
 'arms':['JOINT','JOINT_SIGNAL'],'HGB_max_fits':20,'C2ST':r7['C2ST'],'train_each_domain':600,'holdout_each_domain':250,'bootstrap_repeats':500,'bootstrap_seed':r7['bootstrap_seed'],
 'C2ST_views':['full_historical_shape','full_voltage','prompt_local','postpeak_local','tail_local','baseline_local'],
 'local_regions':[[100,150],[150,220],[220,400]],'joint_coefficient':1.,'local_scalar_IQR_floors':[.1,.05,.1,.1],'local_coordinate_IQR_floor':.05,'local_RMS_floor_V':1e-6,'joint_projections_per_region':64,
 'old_loss_coefficients':{'conditional_CDF':4.,'occupancy':1.,'composition':.2,'temporal_smooth':.002,'network_regular':.0001},
 'signal_process':{'amplitude_max_V':.02,'initial_amplitude_V':.002,'reference_voltage_V':.1,'postpeak_time_ns':10.,'FIR':'centered mathematical convolution, zero padding;[.5*tanh(a0),1,.5*tanh(a1)]/L2','initial_FIR_a':[0.,0.],'initial_w':[0.,0.,0.,0.],'not_photon_statistics_or_hardware_identification':True},
 'selection':'every100 incl0; old distribution+conditional CDF/occupancy+new joint only; exclusively TRAIN validation',
 'generation_chunk':256,'input_files':files,'evaluation':old['evaluation'],'task_sha256':start['task_sha256'],'source_nominal_region':'Qsource_Vns*1000 compared with fixed CAL edges; entire source group retained regardless output region',
 'CAL_q4_fallback':False,'EVAL_read':0,'SSH':0,'new_G4':0,'new_agents':0,'Git':0,'same_acquisition_retrospective_DEV':True,
 'numeric_ledger':'all run/replay child wall times in RUN_HISTORY; prepare no scientific arrays/fits'}
write('config.json',cfg)
protocol='''R8固定配对方案：JOINT、JOINT_SIGNAL；共同R3B初始化（bb45da6e...）、原全部损失、三区未平滑局部SW额外系数1；过程开关是两臂唯一科学配方差别。TRAIN fit2592/val599和G4缓存原internal_training划分，DEV不选点，EVAL/SSH/新G4为0。

局部向量[logQ_mVns,PSD,logpeak_V,loglocalRMS_V,z,dz,d2z]，本区均值/RMS，RMS下限1e-6V；TRAIN fit总体均值和type7 IQR一次固定；4标量IQR下限.1/.05/.1/.1，其余.05。各块除sqrt维度，各区64 Gaussian L2单位投影(seed2026100882)，排序batch平均绝对差，三区等权。此训练联合向量有全局条件标量，不能称R7严格局部分类输入。

保留原8维边际/.35平滑SW/.08协方差/4时间点边际、.2弱组成、.002时间系数平滑/.0001原mean/random正则、4条件CDF/1占比；旧全局投影按R3B原seed2026100836独立构造并保存。内部验证沿用2026100821的原父/真实384选序、z/white；signal white以独立2026100886固定。条件CDF fit2592及完整val599目标分别保存，局部尺度仅fit。验证值不包含原composition/regular/temporal罚项，与旧选择口径一致。

所有模型构造前Torch/NumPy2026100881。每臂训练Generator重置同2026100881，并按ii,jj,z,baseline white,signal white相同顺序每步抽取，即使过程关闭亦抽signal white。初始可学习参数与全部buffer逐位核对；JOINT新增未用参数没有梯度且保持初值。Adam lr.0005/batch256/2500上限，每100验证incl0；每臂900秒硬timer，接近墙钟留5秒用于末次验证/保存；每个有效checkpoint/失败全部保留，不DEV补训。

eta=.02sigmoid(b+w*c)*sqrt(max(clean,0)/.1V)*[1-exp(-max(t-first_argmax(clean),0)/10ns)]*xi；峰位整数detached。xi为3抽头数学卷积，核[.5tanh(a0),1,.5tanh(a1)]按L2单位化，零边缘；a/w零初值，b=logit(.1)。sqrt在clean=0处采用精确零且零梯度分支以避免0*infinity，正值公式不变。此为统计假设，不能称光子数/硬件已标定机制；原平稳噪声/ADC/CFD完全保留。

主2000使用R3B实际保存main z/white（从其绑定RNG恢复），signal独立2026100883；两额外视图2026100884/85顺序抽z/white/signal且共用父及两臂，保存全部实际输入。附加随机视图不是独立真实事件。原始R3B已保存三视图直接复用，不重复旧推理。

评价共同signed charge/PSD和冻结CAL后验0.99；q4参考失败无fallback，不筛unknown。8窄区TRAIN冻结，旧REAL/R3B描述EM结果复用、新臂各固定8区单初始化；未收敛/floor/支持不足均undefined。宽度误差必须用REAL/R3B/两臂共同合格bin集合并列缺口。源名义Qsource*1000四区的全部父与输出自身四区双报告、迁移和条件关联保留，不把输出分区当固定父条件。

12新HGB（每新臂6固定视图），与R7同600/250父/组索引和分类尺度规则。R7旧GEN、阴性TRAIN–DEV及阳性控制复用；若新臂在固定选择上无效不悄悄换样本，报告支持失败。500簇bootstrap、两新臂共测试父/REAL组配对差异；条件于固定评估器，不含响应训练不确定性。局部分类本区demean/RMS，不夹入全局峰/PSD/rise；源总体QC条件另披露。

一个本地科学作业依次训练/生成/评价/绘图；独立进程仅保存权重+全部主输入重放两臂各2000x480，逐位及max差/父顺序，不重训不重评。4CPU/4GiB/512MiB/累计2400秒，阶段从START_RECEIPT到120min。系统依赖装载不作原数据字节账本；日志、实际进程/退出/RSS/输出大小明示。
'''
(ROOT/'PROTOCOL.md').write_text(protocol,encoding='utf-8')
sources=[]
for p in sorted(ROOT.glob('*.py')):ast.parse(p.read_text(encoding='utf-8'));sources.append(bind(p))
write('READINESS.json',{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),'source_bindings':sources,'copied_immutable_sources':copied,'config':bind(ROOT/'config.json'),'protocol':bind(ROOT/'PROTOCOL.md'),'scientific_body_decode':0,'fit':0,'new_generation':0,'frozen_before_numeric':True})
print(json.dumps({'status':'ready','PID':os.getpid(),'source_files':len(sources),'config_sha256':bind(ROOT/'config.json')['sha256']}))
