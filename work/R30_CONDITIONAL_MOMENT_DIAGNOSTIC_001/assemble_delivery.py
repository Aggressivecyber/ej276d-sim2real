from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,os

ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def now():return datetime.now(timezone.utc).isoformat()
cfg=read('config.json');m=read('METRICS_COMPACT.json');runs=read('RUN_HISTORY.json');plot=read('PLOT_RECEIPT.json')
names=['FRESH_SELECTED','FRESH_FINAL','ENERGY_SELECTED','ENERGY_FINAL','TRAIN_REFERENCE']
labels={'FRESH_SELECTED':'R28 所选2250','FRESH_FINAL':'R28 末点2950','ENERGY_SELECTED':'R29 所选2750','ENERGY_FINAL':'R29 末点2950','TRAIN_REFERENCE':'REAL TRAIN–DEV'}
views=['RAW','GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE']
def f(x,n=4):return 'undefined' if x is None else format(x,f'.{n}f')
def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
classification=m['classification'];comp=classification['comparisons'];params=m['moment_parameters'];cov=m['holdout_covariance'];desc=m['descriptive_moments']
ba_rows=[[labels[n],*[f(comp[n][v]['balanced_accuracy'],3) for v in views]] for n in names]
auc_rows=[[labels[n],*[f(comp[n][v]['AUC'],4) for v in views]] for n in names]
paired_rows=[]
for n in names:
    for v in views[1:]:
        a=comp[n][v]['paired_vs_raw'];ci=a['BA_delta_95CI']
        paired_rows.append([labels[n],v,f(a['BA_delta_first_minus_second'],3),f'[{f(ci[0],4)}, {f(ci[1],4)}]'])
counts_rows=[]
for n,d in params['domains'].items():counts_rows.append([n,','.join(map(str,d['train_Qbin_counts'])),d['fallback_train_n'],d['fallback_holdout_n'],d['global_floor_columns'],','.join(map(str,d['bin_floor_columns'])),f(d['maximum_amplification_global'],1),f(d['maximum_amplification_supported_bins'],1)])
covrows=[]
for n in names:
    for v in ['RAW','Q_CENTER','Q_CENTER_SCALE']:
        a=cov[n][v];covrows.append([labels[n],v,f(a['trace'],4),f(a['effective_rank'],3),f(a['relative_Frobenius_difference_to_REAL'],4)])
momentrows=[]
for subset in ['population','holdout']:
    for v in ['RAW','GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE']:
        a=desc['FRESH_FINAL'][subset][v]['regions'];r=desc['REAL'][subset][v]['regions']
        momentrows.append([subset,v,*[f(a[k]['mean_residual_RMS_u'],6) for k in cfg['regions']],*[f(a[k]['pointwise_variance_mean_u2']/r[k]['pointwise_variance_mean_u2'],4) for k in cfg['regions']]])
equalrows=[[labels[n],*[f(comp[n][v]['common_holdout_bins_equal_mean_BA'],4) for v in views]] for n in names]
science=sum(r['elapsed_s'] for r in runs);peak=max(r['actual_worker_RSS_peak_bytes'] for r in runs)
start=datetime.fromisoformat(cfg['registered_UTC']);prepared=read('PREPARATION_RECEIPT.json')
resource={
    'stage':ROOT.name,'registered_UTC':cfg['registered_UTC'],'executor_seal_deadline_UTC':cfg['deadline_UTC'],
    'first_actual_command':cfg['first_actual_command'],'prepared_control_command':{'tool':'4ef0d3','PID':44956,'native_exit':0,'UTC':prepared['UTC']},
    'science_elapsed_s':science,'science_limit_s':600,'HGB_attempts':15,'HGB_completed':15,'HGB_limit':15,
    'response_updates':0,'response_inference':0,'response_generation':0,'EM':0,'new_REAL':0,'EVAL':0,'CAL_waveform':0,'new_G4':0,'SSH':0,'installation':0,
    'saved_existing_waveform_populations_decoded':6,'saved_existing_waveform_rows':12000,'saved_existing_waveform_points':5760000,
    'preprocessing_and_classifier_fits_are_diagnostic_not_response_models':True,
    'threads':4,'actual_worker_RSS_peak_bytes':peak,'RSS_limit_bytes':4*1024**3,'GPU_allocated_bytes':0,'device':'CPU',
    'actual_worker_PID_guard_before_imports':True,'memory_probe_failures':sum(r['memory_probe_failures'] for r in runs),
    'scientific_jobs':[
        {'entry':'numeric_worker.py','launch_tool':'de9974','terminal_tool':'35360c','session':16636,**runs[0]},
        {'entry':'render_figures.py','launch_and_terminal_tool':'7dc13b',**runs[1]},
    ],
    'scientific_failures':[],
    'control_failures':[{'tool':'9c1189','native_exit':1,'PID':'unknown (not reported by command)','elapsed_s':.4875173,
                         'cause':'Optional Get-Content NUMERIC_FAILURE.json requested an absent file after successfully printing existing metrics. No numeric failure file exists because the scientific job succeeded.',
                         'scientific_rerun':False,'cost_retained_separately':True}],
    'stage_output_limit_bytes':128*1024**2,'stage_output_bytes_before_seal':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
    'recorded_UTC':now(),'wall_elapsed_at_delivery_assembly_s':(datetime.now(timezone.utc)-start).total_seconds(),
    'system_library_load_bytes_not_measured':True,'no_additional_mathematical_probe':True,
}
write('RESOURCE_RECEIPT.json',resource)
report=f'''# R30 条件矩诊断

生成波形在减均值、按电荷区减均值和标准化后仍可明显区分；但同样的域别处理也使 REAL TRAIN–DEV 从原始 BA 0.510 升至 0.954 / 0.772 / 0.814。这个处理对照是本轮首要结论边界：不能把变换后的高 BA 直接解释为已经隔离了噪声、协方差或某个物理机制。本轮没有响应训练、推理、生成或密度拟合。

## 固定数据与定义

四候选为 R28 FRESH_SELECTED2250 / FRESH_FINAL2950、R29 ENERGY_SELECTED2750 / ENERGY_FINAL2950 的已保存 post-ADC 2000 行；REAL DEV2000 与既存 REAL TRAIN reference2000 同采集。四候选父事件顺序完全一致，源组成 417 pureEE / 1518 pure-had / 65 mixed 沿用历史证据。所有六域 QC=2000；保留原人口，不重标、不重加权。本轮不是新盲测。

原 600 个分类器训练行及 250 个留出行、人工分组/QC 和原始 Q 区间全部复用；每域训练/留出组交集为0，R28/R29 实际 phase 数组一致性检查通过。输入为完整480维 float64 u=(V−前100点均值)/max(扣基线峰值,0.0025V)，不含旧493列视图的13个标量。本表的 R28 pure-u BA .978 不同于旧完整493列 BA .970；二者特征定义不同，不能互换。

每域仅用600行估计矩：GLOBAL_CENTER 减全局均值；Q_CENTER 减对应原观察 Q-bin 均值；Q_CENTER_SCALE 再除该域该 bin 的样本 SD(ddof1)，下限固定 max(0.05×旧 REAL TRAIN shape480 尺度,1e−6)。六域每 bin 均 N≥20 才允许条件矩，统一合格为 bins1–7；bin8或区外/未定义 Q 对全部域回退各自全局矩。无丢弃、重归一化、新峰定位、重分区或变换后的新 PSD。原8区间为已冻结 TRAIN logQ 定义，不是实测真能量标定。

## 留出分类结果

{table(['域比较','原始u','GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE'],ba_rows)}

AUC 对应为：

{table(['域比较','原始u','GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE'],auc_rows)}

15个新 HGB 均按旧100步/learning-rate .06/15叶/min-leaf20/L2=1/random-state20261008，训练域各600、留出各250；训练 BA/AUC 全为1.000。原始 u 分类预测来自旧保存文件，未重拟合。所有新拟合在500次 bootstrap 前立即保存预测。CI 条件于固定分类器和同采集人工组，不包含响应重训或独立采集不确定性；不能把 REAL–REAL 数值当成预定等效容差。

变换减原始的配对 BA 差及95%区间（相同留出行/组）：

{table(['域比较','变换','ΔBA','95%组bootstrap CI'],paired_rows)}

全部比较的共同留出 N≥10 区为 bins1–6，不按结果选择区间。原 Q-bin 描述的等权 BA：

{table(['域比较','原始u','GLOBAL_CENTER','Q_CENTER','Q_CENTER_SCALE'],equalrows)}

每域每区计数、不合格原因为完整 CLASSIFIER_METRICS.json；源候选 bins7/8 留出不足的 n/a 全保留。这里没有分区重新训练或区内 bootstrap。

## 矩与剩余波动

R28 FRESH_FINAL 的预定区域描述如下。population 是全部2000个有效行；holdout 是固定250行，两者不混写。区域 baseline0:100 / prompt100:150 / postpeak150:220 / tail220:400；prompt 包含峰和早期下降。

{table(['样本','视图','base均值残差RMS','prompt RMS','postpeak RMS','tail RMS','base方差/REAL','prompt方差/REAL','postpeak方差/REAL','tail方差/REAL'],momentrows)}

Q_CENTER 在留出中把明显的均值差减小，但峰后仍有残差；原始/减均值后的局部方差也不相同。Q_CENTER_SCALE 的单位是各域训练 SD 单位，与前两种 u 单位不同，不能用跨表示绝对大小宣称物理改善。点方差相近也不代表联合波形分布相近。全体均值/SD及250行留出曲线均保存；全体曲线只是描述。

250行高维样本协方差（ddof1）的描述：

{table(['域比较','视图','trace','有效秩','相对REAL Frobenius差'],covrows)}

对应 REAL 的有效秩 RAW {f(cov['REAL']['RAW']['effective_rank'],3)}、Q_CENTER {f(cov['REAL']['Q_CENTER']['effective_rank'],3)}、Q_CENTER_SCALE {f(cov['REAL']['Q_CENTER_SCALE']['effective_rank'],3)}。这些是高维有限样本矩，且变化单位不同；既无协方差逆/白化，也没有因果定位。保留 REAL–REAL 描述作为估计和处理参照。

训练矩支持和回退：

{table(['域','8bin训练数','train回退','holdout回退','global floor列','8bin floor列','global最大放大','合格bin最大放大'],counts_rows)}

float64 全局去均值、合格区去均值、未截下限列 SD=1 的算术检查全部通过，精确误差/逐区数见 MOMENT_PARAMETER_RECEIPT.json。这是公式/角色检查，不是额外数据迁移测试。

## 能说明什么

原始生成域仍与 REAL 高度可分，四个候选在三种处理后也没有接近处理后的 REAL–REAL 对照；但对照从 .510 强烈偏移意味着此诊断不是“干净地只留下真实噪声”。由有限训练行分别估计域矩，再施加到带有逐事件峰归一化约束的波形，会改变各域几何与支持；这是可能的处理解释，当前未做额外机制实验，不能当已证实成因。

因此本轮不支持仅凭高残余 BA 加协方差/噪声项或把去均值当可部署修复。优先问题是厘清峰归一化锚点与域别矩估计引入的可分性，再依据原始条件均值/波动证据决定响应改进。剩余差距也可能包含时间依赖、高阶边际、源类混合与有限分区效应，不能唯一归因某个分支。纯组中心/峰宽/源保真和原始 PSD 继续引用 R28/R29 已封存结论；本轮不改这些结果。

## 运行与交付

首实际准备命令 PID21408/tool d87edf/native0，UTC16:10:24.564528；注册起点16:09:11.291876、封存期限16:59:11.291876 UTC。主统计 actual36276/supervisor25520/tool de9974→35360c/native0，16:20:59.432378→16:21:30.191905，30.7594174s。绘图 actual48664/supervisor50932/tool7dc13b/native0，4.9496507s。累计科学 {science:.7f}s/600，峰真实 worker RSS {peak}B/4GiB；0内存观测失败、GPU0；15/15HGB、0 EM、0响应更新/推理/生成、新 REAL/EVAL/CAL/G4/SSH/安装均0。

8张最终PNG均已实际打开检查，无裁切/重叠或必要修图。一次纯控制输出查询9c1189/native1因可选 NUMERIC_FAILURE.json 不存在而返回非零，科学任务成功；该控制失败及0.4875173s保留，未重跑计算。没有科学失败或科学重复。

必要产物：config.json / PROTOCOL.md / numeric_worker.py / render_figures.py；METRICS_COMPACT.json为无逐事件列表的聚合发布版；results/MOMENT_PARAMETERS_PRIVATE.npz、INPUT_ROWS_PRIVATE.npz、15份预测和FIGURE_DATA_PRIVATE.npz保留E/D；CLASSIFIER_METRICS、DESCRIPTIVE_MOMENTS、HOLDOUT_COVARIANCE及native/resource/QA/reproduction见同目录。未重复保存六域整套变换数组。最终清单/封存回执由单一收尾脚本产生，完成后本目录停止写入。
'''
(ROOT/'REPORT.md').write_text(report,'utf8')
qa={'PID':os.getpid(),'UTC':now(),'viewer':'tools.view_image actual rendered PNGs','figures':[{ 'path':str(ROOT/'figures'/n),'actually_viewed':True,'layout_ok':True,'numeric_scope_unchanged':True} for n in plot['figures']],
    'ROOT_also_viewed_all_eight':True,'ROOT_requested_plot_repairs':False,'displayed_CI_is_fixed_classifier_group_bootstrap':True,'unsupported_Qbins_explicit':True,'units_separate_RAW_and_Qscaled':True}
write('FIGURE_QA.json',qa)
repro='''# R30 reproduction

Existing local CPU environment: C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe. The immutable old saved inputs and exact phase0/phase1 rows are listed in config.json. R28/R29 phase arrays matched R27, and all four source ID orders matched the existing R28 input. No SSH or full source audit was performed.

In a new E/D directory, copy this stage's source files/config/protocol and retain the configured read-only input paths. Set PYTHONUTF8=1, PYTHONDONTWRITEBYTECODE=1 and TEMP/TMP to E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp. Set a fresh explicitly authorized deadline and budget for a future rerun; the original 2026-10-09 deadline and used markers are expired/consumed, so do not execute this sealed directory.

Commands actually executed once were python -B supervise.py numeric_worker.py <CPU_python>, then python -B supervise.py render_figures.py <CPU_python>. The supervisor sets four CPU threads and E-stage caches, binds os.getpid() and nonce before scientific imports, queries that actual PID's RSS, and owns termination. RUN_HISTORY.json and each .NATIVE_RECEIPT.json contain the authentic PID/exit/time chain. Numeric worker15 fits and plot8 images both native0. This stage has no response model, generation or fit replay requirement.

To rebuild moment features without storing all transformed arrays: read the fixed saved V and INPUT_ROWS_PRIVATE, calculate the frozen u representation, then apply MOMENT_PARAMETERS_PRIVATE. Unsupported/outside bins use domain global parameters; Q_CENTER_SCALE clips SD with the stored floor. Original Q is used for all conditioning. Raw probabilities remain in the configured old prediction files. New prediction/preprocessing vectors and curve arrays are saved locally; the published compact JSON does not include event identities or predictions.

The method is a domain-identity diagnostic, not an input-to-output model. The large REAL–REAL shift after transformation is preserved and blocks interpreting residual BA as a unique physical cause. No new PSD or deployment claim follows from these transformed features. Exact tool exits and resource costs, including the optional absent-file control query, are in RESOURCE_RECEIPT.json.
'''
(ROOT/'REPRODUCTION.md').write_text(repro,'utf8')
sources=[]
for p in ROOT.glob('*.py'):
    b=p.read_bytes();ast.parse(b.decode('utf8'));sources.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
write('SOURCE_BINDINGS.json',{'UTC':now(),'source_AST_parse':True,'sources':sources,'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),'source_arrays_version_by_reference':cfg['inputs'],'source_reaudit':False})
write('DELIVERY_ASSEMBLY_RECEIPT.json',{'PID':os.getpid(),'UTC':now(),'source_only_report_assembly':True,'new_scientific_compute':False,'science_elapsed_s':science,'all_figures_actual_viewed':True,'report_bytes':(ROOT/'REPORT.md').stat().st_size,'compact_bytes':(ROOT/'METRICS_COMPACT.json').stat().st_size})
print(json.dumps({'PID':os.getpid(),'UTC':now(),'report':str(ROOT/'REPORT.md'),'science_s':science,'eight_plots_QA':True}),flush=True)
