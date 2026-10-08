from __future__ import annotations
import pathlib,json,hashlib,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8'))
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def bind(p):
    h=hashlib.sha256();n=0
    with p.open('rb') as f:
        while b:=f.read(2*1024*1024):h.update(b);n+=len(b)
    return {'path':str(p.relative_to(ROOT)).replace('\\','/'),'bytes':n,'sha256':h.hexdigest()}
def f(x,d=5):return 'undefined' if x is None else f'{x:.{d}g}'
start=read('START_RECEIPT.json');cfg=read('config.json');m=read('metrics.json');history=read('RUN_HISTORY.json');replay=read('REPLAY_CHECK.json');qa=read('VISUAL_QA.json');roles=read('results/DATA_ROLES.json');layout=read('LAYOUT_REPAIR_RECEIPT.json')
numeric_total=sum(r['wall_s'] for r in history)+layout['elapsed_s']
assert m['status']=='complete' and replay['status']=='pass' and history[-1]['worker_native_exit']==0
assert m['counters']['HGB_attempts']==12 and m['counters']['generation_rows']==12000
assert m['counters']['EVAL_read']==m['counters']['SSH']==0 and qa['all_actual_viewed']
assert numeric_total<2400
views=cfg['C2ST_views'];names=['R3B','JOINT','JOINT_SIGNAL'];groups=['overall','q1','q2','q3','q4']
rows=['|视图|R3B BA|JOINT BA|JOINT_SIGNAL BA|SIGNAL−JOINT [簇95%]|','|---|---:|---:|---:|---:|']
for v in views:
    r=m['paired_C2ST_difference']['JOINT_SIGNAL_minus_JOINT|'+v];lo,hi=r['BA_delta_95CI']
    rows.append('|'+v+'|'+'|'.join(f(m['C2ST'][n+'|'+v]['balanced_accuracy'],4) for n in names)+f"|{f(r['BA_delta_first_minus_second'],4)} [{f(lo,4)},{f(hi,4)}]|")
distrows=['|候选/区域|完整QC数|PSD W1|PSD KS|Qt W1 (mV ns)|rise W1(ns)|','|---|---:|---:|---:|---:|---:|']
for n in names:
    for r in groups:
        d=m['domain_vs_REAL'][n][r];count=m['domains'][n]['regions'][r]['QC_n']
        distrows.append(f"|{n}/{r}|{count}|{f(d['PSD']['W1'])}|{f(d['PSD']['KS'])}|{f(d['Qtotal_mVns']['W1'])}|{f(d['rise_10_90_ns']['W1'])}|")
source=['|候选/源名义区域|全部父数/QC数|全部父输出PSD W1对REAL自身同q|输出自身同q PSD W1|','|---|---:|---:|---:|']
for n in names:
    for r in groups[1:]:
        x=m['source_nominal_condition'][n]['regions'][r]
        source.append(f"|{n}/{r}|{x['entire_source_parent_count']}/{x['QC_n']}|{f(x['vs_REAL_output_own_q']['PSD']['W1'])}|{f(m['domain_vs_REAL'][n][r]['PSD']['W1'])}|")
widthrows=['|密度成分/字段|四域共同合格bin IDs|R3B平均绝对误差|JOINT|JOINT_SIGNAL|','|---|---|---:|---:|---:|']
for r in m['energy_conditional_width']['common_support_paired_errors']:
    widthrows.append(f"|{r['component']}/{r['field']}|{r['common_qualified_bin_ids_1based']}|"+'|'.join(f(r['mean_abs_error'][n]) for n in names)+'|')
curves=['|自身电荷桶/共同N|prompt 曲率中位 REAL/R3B/JOINT/SIGNAL (mV)|postpeak 同口径 (mV)|postpeak W1 R3B/JOINT/SIGNAL/RR (mV)|','|---|---:|---:|---:|']
for r in groups:
    d=m['same_N_conditional_reference'][r];values={}
    for region in ['prompt','postpeak']:
        key=region+'_curvature_RMS_V';base=d['comparison']['R3B'][key]['REAL']['quantiles'];med=[1000*base[3] if base else None]
        for n in names:
            q=d['comparison'][n][key]['candidate']['quantiles'];med.append(1000*q[3] if q else None)
        values[region]='/'.join(f(x,4) for x in med)
    ww=[]
    for n in [*names,'TRAIN_REFERENCE']:
        q=d['comparison'][n]['postpeak_curvature_RMS_V']['distance']['W1'];ww.append(1000*q if q is not None else None)
    curves.append(f"|{r}/{d['N_per_domain']}|{values['prompt']}|{values['postpeak']}|"+'/'.join(f(x,4) for x in ww)+'|')
coverage=['|域/自身桶|QC|HC低PSD/高PSD/unknown|冻结cut FoM|参考有效|','|---|---:|---:|---:|---|']
for n in ['REAL',*names]:
    for r in groups:
        d=m['domains'][n]['regions'][r]
        coverage.append(f"|{n}/{r}|{d['QC_n']}|{d['HC_gamma_like_n']}/{d['HC_neutron_like_n']}/{d['unknown_n']}|{f(d['FoM'].get('FoM'))}|{d['reference_valid']}|")
trainrows=['|臂|更新/选中更新|初始/最佳TRAIN验证值|墙钟秒|','|---|---:|---:|---:|']
for n in cfg['arms']:
    r=m['training'][n];trainrows.append(f"|{n}|{r['optimizer_updates']}/{r['selected_update']}|{f(r['initial_validation'])}/{f(r['best_validation'])}|{f(r['training_wall_s'])}|")
failures={n:{'qualified_bins_1based':[s['bin']+1 for s in t['slices'] if s['valid']],'failure_reasons_1based':{str(s['bin']+1):s['failure_reasons'] for s in t['slices'] if not s['valid']}} for n,t in m['energy_conditional_width']['tables'].items()}
compact={k:m[k] for k in ['status','question','training','paired_training','domains','domain_vs_REAL','C2ST','paired_C2ST_difference','generation','CAL_charge_edges_mVns','CAL_reference_valid','q4_no_fallback','unknown_all_preserved','same_acquisition_historically_exposed_DEV','R3B_extra_training_vs_mechanism_effect_not_identified','R7_real_real_definition','EVAL_read','SSH','new_G4','CPU_threads','counters','worker_PID','numeric_elapsed_s']}
compact['common_width_errors']=m['energy_conditional_width']['common_support_paired_errors'];compact['all_width_support']=failures
compact['source_nominal']={n:{'migration':s['migration_rows_nominal_cols_output_q1_q2_q3_q4_invalid'],'regions':{r:{k:v for k,v in z.items() if k!='features'} for r,z in s['regions'].items()}} for n,s in m['source_nominal_condition'].items()}
compact['same_N_local_curvature']={r:{'N_per_domain':s['N_per_domain'],'source_counts':s['source_counts'],'comparison':{n:{k:v for k,v in z.items() if 'curvature' in k or k in ['baseline_RMS_V','ACF1']} for n,z in s['comparison'].items()}} for r,s in m['same_N_conditional_reference'].items()}
compact['parent_variation']=m['three_view_within_parent_variation'];compact['replay']=replay['arms'];compact['publication_payload']='aggregate results only; raw/identity/predictions/RNG/weights retained locally'
write('METRICS_COMPACT.json',compact)
local=read('TARGET_STATISTICS.json')['local_regions'];selected=m['training'];update_same=m['paired_training']['same_actual_update_count'];overall=m['domain_vs_REAL']
title=f"R8 的两臂按固定 TRAIN 配方完成。完整shape C2ST BA：R3B {f(m['C2ST']['R3B|full_historical_shape']['balanced_accuracy'],4)}，JOINT {f(m['C2ST']['JOINT|full_historical_shape']['balanced_accuracy'],4)}，JOINT_SIGNAL {f(m['C2ST']['JOINT_SIGNAL|full_historical_shape']['balanced_accuracy'],4)}。峰后局部 BA：{f(m['C2ST']['R3B|postpeak_local']['balanced_accuracy'],4)}/{f(m['C2ST']['JOINT|postpeak_local']['balanced_accuracy'],4)}/{f(m['C2ST']['JOINT_SIGNAL|postpeak_local']['balanced_accuracy'],4)}。信号过程对峰后局部差异有开发证据，但完整波形仍明显可区分；共同合格bin 2–7的两侧PSD宽度误差未改善，纯上升沿真实性仍未解决。各臂和不利指标均保留；这些开发统计不等同于响应完全相似、粒子真值或物理机制识别。"
report=title+f'''

唯一比较与数据用途

两臂均从同一R3B权重bb45da6e923088083ee7a3c3491f94909b81a24495c869f4553bf2379e8fbb7b开始、保留旧全部损失，并新增同一局部联合SW。JOINT_SIGNAL另开启峰后信号包络过程；JOINT关闭过程但同样抽取独立signal white。初始所有参数/buffer与验证输入逐位一致，前{m['paired_training']['common_prefix_updates']}步的c/Q源索引、REAL索引、z、baseline/signal white摘要完全一致；实际更新数相同={update_same}。如果两臂因墙钟更新数不同，应作为额外训练量混杂保留，不能称严格同更新效果。

REAL TRAIN source3200中QC3191，原组内fit2592/val599；G4缓存8192沿原父ID内部fit/val划分。只在TRAIN统计尺度/目标及选checkpoint；主DEV仍同原2000 G4父与REAL DEV2000。CAL仅复用已存参考，不新拟合；P1K全部字节hash但只解码前5200波形，后1600 CAL waveform不解码。EVAL/SSH/新G4/环境安装/新代理/Git为0。配方来自已查看的R7 DEV诊断，原DEV也长期被历史模型暴露，因此这是开发消融，不能包装成新盲测或外部采集验证。

预定实现

三区100:150、150:220、220:400；各区向量[logQ_mVns,PSD,logpeak_V,loglocalRMS_V,z,dz,d2z]。本区demean/populationRMS分母下限1e-6V，TRAIN-fit均值/IQR一次固定，四标量IQR底.1/.05/.1/.1、其余.05，各块除sqrt维数，每区64未平滑Gaussian单位投影seed2026100882，排序SW三区等权系数1。受IQR下限影响坐标数三区为{[x['floor_affected_count'] for x in local]}，逐块计数及实际均值/IQR/投影保存在TARGET_STATISTICS和TRAIN_TARGETS。

训练联合特征附带全局Q/PSD/peak用于条件关系；它与R7严格局部分类输入不同，不能声称训练特征只含本区域。共同旧损失为sorted标量边际、.35平滑shape投影SW、.08协方差、4时间点边际、.2弱组成关联、.002时间系数平滑、.0001原mean/random参数正则、4条件CDF+1占比。无FoM目标/真实波形模板/事件点对点配对MSE。CDF拟合2592和完整599验证目标分别保存，投影/目标只建一次，不按DEV更新。

eta(t)=.02V sigmoid(b+w*c) sqrt(max(clean(t),0)/.1V) [1-exp(-max(t-first_argmax(clean),0)/10ns)] xi(t)。首峰为整数argmax、无直通梯度；xi使用三抽头[.5tanh(a0),1,.5tanh(a1)]/L2的数学卷积与零边缘。w/a初0、b=logit(.1)，起始条件幅度.002V；JOINT未用参数没有梯度且保持初值。sqrt在clean=0分支精确零且零导数，正clean公式不变。此sqrt包络、10ns开启与.02V范围都是预设统计假设，不能叫已标定光子统计/电子学时间常数或唯一机制。

原平稳TRAIN基线噪声、ADC和整数CFD保持。过程在每条干净信号峰前/峰处严格零已经对全部主输出检查，但它可能改变后来的观测最大值和CFD，因此主波形统计、峰位、迁移/电荷/PSD都共同保留，不能只看新增噪声曲率。

训练与保存

{chr(10).join(trainrows)}

主Torch/NumPy2026100881在任何模型构造之前设定并保存；原R3B上游训练初始化的历史限制仍不因本轮确定性而被补造。每100更新含0按固定TRAIN验证384事件、原验证源/z/white种子2026100821及独立signal种子2026100886选点，唯一分数是旧分布+原CDF/占比+新联合，未加DEV指标。每臂900秒硬timer、2500上限、batch256、Adam .0005，不搜索配方/种子。初始、每100checkpoint、最终及选中权重、全部采样RNG/摘要与实际验证输入均留存。

主生成保存实际R3B main z/white及2026100883信号white；两额外视图2026100884/85共享两臂父、z、baseline/signal white。原R3B三个保存视图直接复用，其额外视图seed与新臂不同，父内3-view SD只作小样本描述。共12000新输出，额外视图不增加真实独立样本。主波形保存后新进程只重放两臂各2000x480，共{replay['values_compared']}值；两臂逐位相等和父序一致，maxabs均0，重放不重训/重评。

完整响应与区域C2ST

{chr(10).join(rows)}

12/20新HGB，没有调参；复用R7固定六视图、每域600分类训练/250父组留出及同一选序、training-only填补/标准化。R3B/同N REAL TRAIN–DEV阴性和尾部+噪声阳性直接复用R7，未重拟合。严格局部分类只从本区demean/RMS或baseline本区谱/ACF取得输入；完整历史shape明示含全局peak/PSD/rise。QC仍来自完整事件，因此局部结果是整体QC条件下的信息差异，不是独立物理归因。

表中差为同留出父及REAL人工组500次配对bootstrap；是固定两个评估器条件下的准确率差，没有覆盖响应训练不确定性。相对旧R3B的总变化混合了额外训练和新联合损失，不能将其全部因果归给新局部损失。JOINT_SIGNAL−JOINT仅是这一个初始/采样/配方条件下的过程开关开发比较；任何不利域/视图仍保留。

{chr(10).join(distrows)}

所有比较包括完整QC总体/重叠/unknown，无输出重加权或HC筛选。既有CAL q4参考失败仍全unknown/FoMundefined，其他置信仅冻结PSD密度后验≥.99，不是粒子标签。整体及每区覆盖如下；FoM不能独立代表完整波形/噪声相似。

{chr(10).join(coverage)}

能区宽度和局部曲率

8窄logQ区间来自原TRAIN-fit定义。旧REAL/R3B密度拟合直接复用，新两臂各最多8次同初始化EM，非CAL重拟合、不称粒子真值，不强制1/sqrt(E)趋势。未收敛500、触floor、有效Nw<20及分离不足原样undefined；各域完整资格/失败见ENERGY_WIDTH_METRICS。原宽度/区间内PSD~logQ去趋势宽度并列，以四域共同合格同一组bin计算配对误差，避免不同覆盖的平均值比较。

{chr(10).join(widthrows)}

曲率为区域原始电压二阶差分RMS，包含干净形状和随机变化；不能从它单独测量光子噪声。prompt=100:150包含主峰和早期下降段，新增过程可在该区的干净峰后非零，故prompt曲率增加不能全称为纯上升沿恶化；上升时间统计单独报告。自身Qt桶中REAL/R3B/JOINT/SIGNAL/TRAIN参照按固定身份散列取共同N，不因高置信删样本。完整baseline RMS/ACF/非DC谱和前沿/尾部仍在metrics及图；峰后过程在峰前为0不代表学习后的整个模型绝不影响前沿/基线参数。

{chr(10).join(curves)}

固定源条件与输出迁移

源名义桶明确按G4 Qsource_Vns*1000与冻结CAL界计算，是既有可见沉积/标定代理，不是入射能量/真实粒子。每一源桶全部父输出用于比较REAL观察自身同q，不筛掉跨桶父；同时报告输出自己落在同q的统计。这两种条件对应不同问题，排名不必相同。

{chr(10).join(source)}

完整四行×五列迁移（输出q1–q4/invalid）、G4四个条件与PSD/Q/peak/rise/局部曲率的秩关联、3-view父内波动和undefined数均在metrics。关联不等于粒子真值或独立传递函数校准，三个随机视图不能替代真实重复采集。

资源、失败和交付

首原生工具{start['first_tool']}/PID{start['first_native_PID']}/exit0，UTC{start['start_UTC']}，120分钟阶段截止{start['deadline_UTC']}。准备e56ca5/PID60720/native0。首数值015f68→262213，supervisor49372/worker23776/native1，10.084939s、RSS473260032B，在真实TRAIN/G4数组与目标统计已读取/计算后，目标JSON的Torch Tensor序列化失败；响应更新/生成/HGB均0。failed_numeric_1保留全部原件/源码/费用，不能写成没有数组访问。只修JSON Tensor→NumPy，未变任何目标数值/配方/config；config SHA仍6a1769a33dfdf772d174ad7942528b61dbfedc3698c2923e76d778326a836e15。修复后来源/AST绑定READINESS_ENGINEERING_REPAIR_001。

全部原生命令、成功/失败墙钟和RSS在RUN_HISTORY，另含仅绘图布局修复{layout['elapsed_s']:.6f}s；合计{numeric_total:.6f}s/2400。4线程、成功内存上限见RESOURCE_RECEIPT，512MiB新产物上限，全部源码/数据/缓存/日志E/D。源容器纯字节hash经过压缩body，不意味着未选科学数组全部被解码；系统Python/DLL依赖装载未作源数据IO逐字节计量。实际数值仅任务允许的TRAIN/DEV/CAL已存JSON及新结果。

{len(qa['figures'])}张关键图已实际查看并通过可读性检查。迁移图原色条遮住JOINT_SIGNAL的q4计数，原图保留，并且只从保存的metrics计数改成专属色条列重绘；不重训/不重算指标。模型在arms/JOINT与arms/JOINT_SIGNAL/model.pt，三视图同父波形generated.npz；GENERATION_INPUTS、VALIDATION_INPUTS与显式RNG可复现本次输入，replay.py是已经执行的独立主推理入口。完整metrics、紧凑METRICS_COMPACT、协议/实现、资源/原生回执及全阶段纯字节清单交付；身份/逐事件波形/预测/随机输入/模型留本地，ROOT只提交必要汇总与源码/图。

峰后局部改善没有解决纯上升沿的响应真实性：本轮仍用解析核和光滑时间RBF主干，新过程故意在干净峰前关闭；prompt整段的分类/曲率不能替代纯峰前结构验证。用户指出重建上升沿过于理想，需在后续明确的前沿响应实验中检验，而本轮不追加训练或重新解释为已解决。根代理暂缓尚未绑定的R9，先定点检查和登记前沿任务；R8全部有效候选保留。R8不读取EVAL、不隐藏不利臂、不自行进入下一任务。旧EVAL已是查看过的同采集基准，未来冻结回顾验证也不能写成新的外部盲测。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
write('RESOURCE_RECEIPT.json',{'numeric_history':history,'layout_repair':layout,'numeric_cumulative_wall_s':numeric_total,'numeric_limit_s':2400,'peak_RSS_bytes':max(x['peak_RSS_observed'] or 0 for x in history),'RSS_limit_bytes':4294967296,'threads':4,'counters':m['counters'],'replay_rows':4000,'replay_values_compared':replay['values_compared'],'output_limit_bytes':536870912,'deadline_UTC':start['deadline_UTC'],'all_files_and_cache_E_D':True,'first_target_serialization_failure_retained':True,'system_dependency_load_IO_not_instrumented':True})
assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(start['deadline_UTC'])
members=[bind(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']]
total=sum(x['bytes'] for x in members);assert total<cfg['output_limit_bytes']
write('FINAL_MANIFEST.json',{'files':members,'members':len(members),'bytes':total,'self_exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'pure_byte_hash_no_array_decode_at_seal':True})
receipt={'status':'sealed_complete','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_UTC':start['deadline_UTC'],'report':bind(ROOT/'REPORT.md'),'metrics':bind(ROOT/'metrics.json'),'compact_metrics':bind(ROOT/'METRICS_COMPACT.json'),'manifest':bind(ROOT/'FINAL_MANIFEST.json'),'numeric_cumulative_wall_s':numeric_total,'main_generation_rows':12000,'replay_rows':4000,'HGB_fits':12,'updates':m['counters']['response_updates'],'EVAL_read':0,'SSH':0,'native_exit_supplied_by_external_tool':True,'stage_writes_stopped_after_this_receipt':True}
write('FINAL_RECEIPT.json',receipt)
print(json.dumps({**receipt,'last_file_write_UTC':datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat(),'total_files':len(members)+2,'total_bytes':total+(ROOT/'FINAL_MANIFEST.json').stat().st_size+(ROOT/'FINAL_RECEIPT.json').stat().st_size},ensure_ascii=False),flush=True)
