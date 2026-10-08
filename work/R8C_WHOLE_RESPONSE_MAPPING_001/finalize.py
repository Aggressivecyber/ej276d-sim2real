from __future__ import annotations
import pathlib,json,hashlib,os,datetime,time,ast
ROOT=pathlib.Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()
cfg=read('config.json');m=read('metrics.json');hist=read('RUN_HISTORY.json');replay=read('REPLAY_RECEIPT.json');qa=read('FIGURE_QA.json');layout=read('LAYOUT_QA_RECEIPT.json')
arms=cfg['arms'];names=['SIM','GEN_W0','R3B','R8_JOINT','R8_JOINT_SIGNAL',*arms]
def fmt(x):return 'undefined' if x is None else f'{x:.8g}'
fields=['PSD','Qtotal_mVns','rise_10_90_ns','baseline_RMS_V','ACF1']
rows=['|域|PSD W1|电荷 W1 [mV ns]|rise W1 [ns]|基线RMS W1 [V]|ACF1 W1|','|---|---:|---:|---:|---:|---:|']
for n in names:rows.append('|'+n+'|'+'|'.join(fmt(m['domain_vs_REAL'][n]['overall'][k]['W1']) for k in fields)+'|')
crows=['|视图|R3B BA|R8 SIGNAL BA|FULL BA|OFF BA|FULL−OFF 95% CI|','|---|---:|---:|---:|---:|---|']
for v in cfg['C2ST_views']:
    delta=m['paired_C2ST_difference'].get(arms[0]+'_minus_'+arms[1]+'|'+v,{})
    vals=[m['C2ST'][n+'|'+v].get('balanced_accuracy') for n in ['R3B','R8_JOINT_SIGNAL',*arms]]
    crows.append('|'+v+'|'+'|'.join(fmt(x) for x in vals)+'|'+str(delta.get('BA_delta_95CI','unsupported'))+'|')
growth=[]
for n in arms:
    r=m['generation'][n];c=m['source_input_controls'][n]
    growth.append(f"- {n}: G/D各{m['training'][n]['optimizer_updates_G']}更新，TRAIN选点{m['training'][n]['selected_step']}；DEV QC={m['domains'][n]['QC_valid']}/2000，invalid={m['domains'][n]['invalid']}；主视图clip={r['per_view'][0]['ADC_clip_rows']}，padding={r['per_view'][0]['padding_rows']}。shuffle RMS变化中位数={fmt(c['shuffle']['RMS_delta_V']['quantiles'][3])}V，replace={fmt(c['replace_first_TRAIN_fit']['RMS_delta_V']['quantiles'][3])}V，两种控制完全忽略源={c['ignores_source_under_both_controls']}。")
coverage={n:[s['bin']+1 for s in t['slices'] if s['valid']] for n,t in m['energy_conditional_width']['tables'].items()}
failures={n:{str(s['bin']+1):s['failure_reasons'] for s in t['slices'] if not s['valid']} for n,t in m['energy_conditional_width']['tables'].items()}
width_rows=['|成分（均值顺序）|字段|共同合格 bin|R3B MAE|FULL MAE|OFF MAE|','|---|---|---|---:|---:|---:|']
for r in m['energy_conditional_width']['common_support_paired_errors']:
    width_rows.append('|'+str(r['component'])+'|'+r['field']+'|'+str(r['common_qualified_bin_ids_1based'])+'|'+'|'.join(fmt(r['mean_abs_error'][n]) for n in ['R3B',*arms])+'|')
output_counts={n:[m['domains'][n]['regions']['q'+str(i)]['source_n'] for i in range(1,5)] for n in ['REAL',*arms]}
variation={n:{k:m['three_view_within_parent_variation'][n]['features'][k]['within_parent_SD']['quantiles'][3] for k in ['PSD','Qtotal_mVns','peak_V','rise_10_90_ns']} for n in ['R3B',*arms]}
anomaly=next(r for r in m['training'][arms[0]]['validation_trace'] if r['step']==1900)
scientific_conclusion={
    'whole_response_similarity':'failed',
    'complete_shape_BA':{n:m['C2ST'][n+'|full_historical_shape']['balanced_accuracy'] for n in arms},
    'prompt_local_BA':{n:m['C2ST'][n+'|prompt_local']['balanced_accuracy'] for n in arms},
    'common_width_bins_1based':[3,4,5,6,7],
    'conditional_width_errors_larger_than_R3B_for_both_arms':True,
    'mean_waveforms_not_complete_distributions':True,
    'branch_ablation_identifies_full_branch_capacity_not_pure_variance':True,
    'source_sensitivity_not_correct_physical_mapping':True,
    'FULL_step1900_validation_anomaly':anomaly,
    'validation_anomaly_cause':'unknown; no additional diagnosis or recipe change'
}
codeclarification='均值分支不读噪声，但pulse非线性支路的条件期望没有约束为0。OFF移除整个新增支路，包含可能的确定性/均值表达能力，不能严格解释为纯方差消融。两臂共同实际更新数、初态、抽样和验证输入相同，选择检查点可不同；本轮只有一个配方和种子，未估计响应训练的不确定性。'
report=f'''# R8C 整体随机响应映射

本轮整体响应相似性失败。两臂完整形状、物理电压和prompt局部C2ST的留出平衡准确率均为1；峰后、尾部和基线也仍可区分。OFF的总体PSD W1降到0.00594147，电荷W1也改善，但10–90上升时间W1增至1.95591ns；FULL对应上升时间W1为1.81286ns，均显著大于既存R3B的0.134022ns。在共同合格窄电荷bin 3–7上，两臂的峰位、原宽度及区间内去趋势宽度误差都大于R3B。个别平均曲线或PSD边际接近，不能代表整体分布相似。

完整波形条件模型及预定支路关闭消融均按固定配方执行并保留。结果用于已有同采集、历史暴露DEV的诊断，不能当独立盲测或跨采集验证。完整数据集差距同时由总体/条件分布、波形、涨落和独立C2ST描述，PSD或FoM单独接近不代表完整响应成功。

## 执行与固定输入

阶段起点{cfg['start_UTC']}，硬截止{cfg['deadline_UTC']}。4CPU线程；累计数值{sum(r['wall_s'] for r in hist):.6f}s/2400s；峰RSS{max((r['peak_RSS_observed'] or 0) for r in hist)}B。每次原生进程及失败保留RUN_HISTORY和numeric_N目录。训练软上限1600s；公共完整轮次{m['paired_training']['common_completed_rounds']}，停止理由{m['paired_training']['stop_reason']}。

G4 clean8192中6593内部fit、1599val；真实TRAIN3200中2592fit、599分组val和9无效记录；固定DEV2000同父事件。只解码REAL首5200行，CAL边界/参考沿用，EVAL科学值0、SSH0、新G4/安装/新代理/Git0。完整clean是EE/hadron总沉积的固定结构描述，没有新增沉积时序。源谱/混合与响应仍混杂；历史SIM噪声先验使用79793实测暴露，未把其成本或暴露重写为本次2592。SIM/W0电压使用历史父峰代理，新模型输出直接物理V，单位相同不等于独立幅度标定。

## 模型及配对解释

24通道、七个stride1扩张残差块覆盖整段输入，signed自由均值残差、16维pulse latent与全记录白噪声、分离4维baseline latent和可学习31抽头条件FIR/RMS。输出一次14bit ADC和原512→480 CFD处理，clip/pad/失败全保留。优化损失adv+全联合SW+2条件PSD CDF+.5电荷占比+.5基线联合；尺度、投影、统计只取TRAIN，选点不看DEV。CFD/阈值 crossing索引不求导，crossing插值值仅分段可导，ADC训练STE；无FoM最大化/源PSD保持/真实模板/配对MSE。

{codeclarification}

{chr(10).join(growth)}

FULL选中step2000（内部验证分数0.110199504），OFF选中step1700（0.091728583），均只按TRAIN内部验证选点。FULL的step1900出现有限但异常大的验证分数{anomaly['score']:.9f}，其中SW项为{anomaly['parts']['SW']:.9f}；下一次记录回落不等于全程稳定，异常原因unknown。保存全部验证轨迹，图中用对数轴显示该点，不删除、不重新选点或追加诊断训练。

## 总体完整分布

{chr(10).join(rows)}

以上W1使用全部QC有效总体，包含unknown，不重新平衡出漂亮子集。metrics同时保存KS、不同分母、HC覆盖/unknown、固定q4参考失败和FoM不支持原因。q4参考无回退；窄能区密度拟合与HC标签参考是不同统计对象。

两臂主视图和REAL均QC有效2000/2000，两臂三视图所有行都保留，clip/padding/非有限输出均0。因此本轮差距没有靠拒绝不利生成事件缩小。charge_psd图中新模型的细窄弯曲带、response_features中的狭窄rise分布与条件宽度结果相互一致；这里描述的是观察到的条件分布不足，不能只凭这些结果唯一确定优化或结构原因。

## 完整与区域独立分类

{chr(10).join(crows)}

12个新HGB上限内使用固定600/250的R7源父/真实人工组划分；区域特征仅来自该区域，不借全局峰归一化泄漏其他区域。500次条件cluster bootstrap只反映评价样本的不确定性，不包含响应训练不确定性。REAL–REAL为同N TRAIN-fit与DEV的比较，均同次采集，人工50记录组不是独立采集批次；阳性对照及R3B/R8结果原样复用。旧SIM/W0没有在这轮重拟新HGB，不能将不同旧评价定义混进主配对矩阵。

BA=1是固定留出样本上的结果，其条件bootstrap区间在该样本上也退化为[1,1]，不能推广为任意实测总体的确定性识别率。FULL/ OFF的基线局部BA为0.848/0.812，峰后为0.994/0.996、尾部为0.980/0.994。区域与完整视图共同显示剩余差距广泛存在，而非只有某个单点前沿问题。

## 条件宽度、来源与涨落

固定8个窄logQ区间，仅两新臂各8次描述拟合，总实际尝试{m['counters']['density_EM_attempts']}，覆盖合格bin：{json.dumps(coverage,ensure_ascii=False)}。失败原因保留：{json.dumps(failures,ensure_ascii=False)}。共同合格的相同bin逐成分误差见energy_conditional_width.common_support_paired_errors；不拿不同覆盖集合的平均数比优劣，不把两密度成分当粒子真值。与输出自身电荷分区并列保存全部父事件的名义电荷分区及迁移，避免改变条件定义后隐去排序翻转。

{chr(10).join(width_rows)}

输出电荷q1–q4的完整计数为{json.dumps(output_counts,ensure_ascii=False)}。原父名义电荷计数[898,548,426,128]，既存SIM自身观测电荷计数[947,540,400,113]，两者定义不同；parent_migration保存4×5表（含invalid列），所有父记录保留。输出条件分箱是在模型输出后决定，不能等同固定父条件，也不能从迁移后的分布接近反推逐事件电荷校准。

三随机视图12000条保持同父，两新臂共用实际保存的16维pulse latent、4维baseline latent和两份白噪声输入；不声称这些噪声与旧架构逐项配对。OFF仍有随机baseline，随机重复不增加真实独立样本。三视图内同父PSD、电荷、峰和rise标准差的中位数为{json.dumps(variation,ensure_ascii=False)}。FULL相对OFF增加了随机变化，但相比R3B仍很小，且条件PSD宽度仍偏窄或形状不符；这种有限的三视图描述不是独立验证真实条件方差。

prompt100:150包含峰和早期下降，曲率不能概括为纯上升沿。pure_front_examples采用固定哈希抽取、实际first-peak之前的曲线，可见两新模型较一致的起跳与前沿形状；这只是一组固定示例的视觉证据，完整rise分布和独立分类才是总体比较。conditional_curvature中峰后细尺度起伏仍不足，conditional_waveforms平均曲线接近也没有消除上述联合分布差异。

固定源shuffle/replacement的非零影响只能说明模型使用了源，不证明逐事件物理映射正确。shuffle的64个目标中有1个自置换，位于控制索引59、DEV父位置1872；保存源ID与donor ID确实相同，两臂保存的unchanged_n也均为1。在已执行的确定性、同源bundle及同噪声推理路径下，该固定点解释唯一不变行，不能把它算作换成不同源后仍忽略源。此解释来自保存的身份字段和已有总计数，未重新推理或解码控制波形。其余63条shuffle及64条replacement均有变化，仍不验证正确物理对应；无配对分布匹配不唯一识别响应。

## 重放、QA及结论边界

保存权重与实际主输入后，独立新进程主4000事件重放：{json.dumps(replay['all_arms'],ensure_ascii=False)}。没有重训或重复完整评价。图表实际QA {qa['status']}，检查{len(qa['files'])}张，保留布局修订及失败（若有）。

主作业监督/worker为22836/53648，工具496785→a0fa0e，native0；独立重放30392/40340，db451b/native0；保存指标布局修订25176/53736，8d16ef/native0。训练与主评价1384.384414s，重放6.447790s，布局与身份解释2.069028s，累计1392.901232s。NUMERIC_RECEIPT和RUN_HISTORY保留真实时间、RSS及进程状态；首次准备和冻结回执另存。无科学失败重试。

两臂不依据DEV挑选删除，整体差距与局部改善分别描述。源敏感性、总体统计接近、逐事件物理对应、G4校准及跨采集能力相互不同，本轮只能支持已测到的部分。保留这个固定配方的负结果，按ROOT后续冻结任务进入回顾性比较；当前不读EVAL，不增加配方或消融。

## 交付及复现

配置config.json、冻结协议PROTOCOL.md、来源SOURCE_BINDINGS.json、实际输入回执INPUTS.json与执行源保留。训练入口为launch.py run.py，独立推理入口为launch.py replay.py，布局修订入口为launch.py layout_qa.py。已保存主随机输入和状态、初始化状态、每臂模型及逐轮训练记录，不为补报告再运行这些入口。

完整metrics.json及ENERGY_WIDTH_METRICS.json、PER_EVENT_FEATURES.npz、生成波形和源控制保存在E；METRICS_COMPACT.json用于关键统计交付。TARGET_STATISTICS.json含TRAIN逐事件目标向量，原始身份、随机输入、模型、逐事件结果/预测和科学载荷保留本地，Git由ROOT按明确路径管理。最终清单对产物作纯字节SHA，不重新解码数组；清单和最终回执自身明确排除自引用。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
compact={k:m[k] for k in ['status','question','paired_training','domains','domain_vs_REAL','local_and_processing_statistics','same_N_conditional_reference','C2ST','paired_C2ST_difference','source_nominal_condition','source_association','three_view_within_parent_variation','generation','CAL_reference_valid','q4_no_fallback','unknown_all_preserved','same_acquisition_historically_exposed_DEV','EVAL_read','SSH','new_G4','counters']}
compact['training']={n:{k:v for k,v in r.items() if k not in ['constructor_TRAIN_fit_only']} for n,r in m['training'].items()}
compact['source_input_controls']={n:{k:v for k,v in r.items() if k!='replacement_TRAIN_fit_id'} for n,r in m['source_input_controls'].items()}
compact['energy_conditional_width']={'edges_logQ':m['energy_conditional_width']['edges_logQ'],'common_support_paired_errors':m['energy_conditional_width']['common_support_paired_errors'],'qualified_bin_ids_1based':coverage,'failure_reasons':failures,
    'tables':m['energy_conditional_width']['tables'],'source_parent_three_view_width':m['energy_conditional_width']['source_parent_three_view_width']}
compact['interpretation']=codeclarification;compact['scientific_conclusion']=scientific_conclusion
compact['source_self_permutation']=layout['source_self_permutation'];write('METRICS_COMPACT.json',compact)
resources={'numeric_cumulative_wall_s':sum(r['wall_s'] for r in hist),'numeric_limit_s':2400,'native_processes':hist,
   'peak_RSS_observed_bytes':max((r['peak_RSS_observed'] or 0) for r in hist),'RSS_cap':4294967296,'memory_probe_failures':sum(r['memory_probe_failures'] for r in hist),
   'worker_counters':m['counters'],'replay_inference_rows':replay['inference_rows'],'replay_fit':0,'CPU_threads':4,
   'engineering_failures':[r for r in hist if r['worker_native_exit']!=0],'failures_not_refunded':True,
   'data_identity_acquisition_independence':'same acquisition, artificial groups; no independent run','EVAL':0,'SSH':0,'new_G4':0,'install':0,'Git':0,
   'postscience_layout_fit_forward_and_waveform_decode':0,'native_tool_links':read('DELIVERY_NATIVE_LINKS.json'),
   'peak_RSS_scope':'Observed worker peak working set, measured by the supervising Windows process; not whole-system memory.',
   'seal_hash_scope':'All stage files as pure bytes; scientific payloads are not decoded for sealing.'}
write('RESOURCE_RECEIPT.json',resources)
assert resources['numeric_cumulative_wall_s']<=2400
assert resources['peak_RSS_observed_bytes']<=4294967296
assert all(r['bitwise_equal'] for r in replay['all_arms'].values())
assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(cfg['deadline_UTC'])
members=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p.name not in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']:
        members.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
total=sum(m['bytes'] for m in members);assert total<536870912-65536
write('FINAL_MANIFEST.json',{'members':members,'member_count':len(members),'member_bytes':total,
    'self_reference_exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'pure_byte_hash_no_array_decode':True})
receipt={'status':'sealed','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'native_exit_expected':'caller tool actual exit authoritative',
    'members':len(members),'member_bytes':total,'manifest_sha256':sha(ROOT/'FINAL_MANIFEST.json'),'report_sha256':sha(ROOT/'REPORT.md'),
    'compact_sha256':sha(ROOT/'METRICS_COMPACT.json'),'resources_sha256':sha(ROOT/'RESOURCE_RECEIPT.json'),'replay_sha256':sha(ROOT/'REPLAY_RECEIPT.json'),
    'numeric_cumulative_wall_s':resources['numeric_cumulative_wall_s'],'scientific_conclusion':scientific_conclusion,
    'phase_after_seal':'no further stage writes; wait ROOT next task',
    'source_declaration':'final manifest contains actual executed source bytes, frozen recipe and preserved readiness/pre-numeric guard history'}
write('FINAL_RECEIPT.json',receipt)
print(json.dumps({**receipt,'last_write_UTC':datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat(),'receipt_sha256':sha(ROOT/'FINAL_RECEIPT.json')}))
