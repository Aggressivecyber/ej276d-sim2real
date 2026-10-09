from __future__ import annotations
import pathlib,json,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text('utf-8'))
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n','utf-8')
def compact(x):
    if isinstance(x,dict):return{k:compact(v)for k,v in x.items()if k not in ['validation_trace','source_parent_index','fixed_source_parent_index','histogram_counts','trace','initial','updates_trace']}
    if isinstance(x,list):return[compact(a)for a in x]
    return x
m=read('metrics.json');native=read('REMOTE_NATIVE_RECEIPT.json');replay=read('REPLAY_RECEIPT.json');pair=read('PAIRING_RECEIPT.json')
training={a:{k:v for k,v in t.items()if k!='validation_trace'}for a,t in m['training'].items()}
c={k:v for k,v in m.items()if k not in ['same_parent_noise','source_probes','training']};c['training']=training
c['same_parent_noise']={a:{k:v for k,v in r.items()if k!='rows'}for a,r in m['same_parent_noise'].items()};c['source_probes']=compact(m['source_probes'])
write('METRICS_COMPACT.json',compact(c))
resource={'stage':'R19','seal_deadline_UTC':'2026-10-09T09:24:33.361272+00:00','source_only_first_native':read('START_RECEIPT.json'),'science_cumulative_s':native['numeric_cumulative_s'],'scientific_history':native['history'],'counters':m['counters'],'GPU_allocated_peak_bytes':max(t['GPU_allocated_peak_bytes']for t in m['training'].values()),'GPU_reserved_peak_bytes':max(t['GPU_reserved_peak_bytes']for t in m['training'].values()),'RSS_peak_bytes':max(h['RSS_peak_bytes']for h in native['history']),'CPU_threads':4,'numeric_limit_s':1800,'training_each_limit_s':600,'training_total_limit_s':1200,'G_D_each_arm_max':4000,'HGB_plan':14,'HGB_limit':24,'width_EM_limit':16,'output_limit_bytes':268435456,'new_source_hash_audit':False,'migration_equality_test':False,'new_G4':0,'EVAL_scientific_decode':0,'CAL_waveform_decode':0,'raw_long_waveform_decode':0,'fresh_replay':replay,'normal_transfer':read('RETURN_TRANSFER_RECEIPT.json'),'failed_science_history':[h for h in native['history']if h['exit_code']!=0]}
write('RESOURCE_RECEIPT.json',resource)
def f(x):return 'undefined'if x is None else f'{x:.6g}'if isinstance(x,(int,float))else str(x)
lines=['# R19：完整源波形转换中的采样先验干预','',
'同一 R18 已选完整波形转换器继续训练，两臂仅改变共同电荷区内源采样概率。CONT 为均匀源采样，TILT 为 TRAIN 冻结指数倾斜。全部主评价保留原始 2000 个 G4 父记录及相同实际噪声，不重加权、不删样本、不重标。实测仍是同一次采集的 TRAIN/DEV，历史模型已接触该 DEV；本轮是诊断，非新盲测。',
'','## 执行与选点','',
'|臂|G/D更新|训练秒|所选step|初始加权J|所选加权J|末点加权J|所选原未加权J|末点原未加权J|',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for a,t in m['training'].items():lines.append('|'+ '|'.join([a,str(t['updates'])+'/'+str(t['D_updates']),f(t['training_s']),str(t['selected_step']),f(t['initial_validation']['J']),f(t['selected_validation']['J']),f(t['final_validation']['J']),f(t['selected_validation']['original_unweighted_J']),f(t['final_validation']['original_unweighted_J'])])+'|')
lines+=['','共同验证采用固定 512 源/599 实测、原 R18 噪声/投影/标准化；两臂共同使用 TRAIN-lambda 加权源验证。加权经验分位公式及全部损失见 PROTOCOL.md/config.json；step0、每250步与末点均参与选点，同时保留原未加权 J。最高区源验证仅4条，不足以支持精细分布结论。',
'','## 先验干预的实际覆盖','',
f"源 fit 加载 {m['data_roles']['G4_TRAIN_fit']} 条，共同八区内 {m['tilt']['source_inside_8bins']} 条、区外 {m['tilt']['source_outside_8bins']} 条。区外仍保留在总体主评价中；不能把加载数写成全部域损失覆盖。",
'','|bin|源N|实测N|原h均值|目标高带亲和量|lambda|倾斜h均值|残差|边界失败|Kish ESS|权重比|',
'|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
for r in m['tilt']['tables']:lines.append('|'+ '|'.join(f(r.get(k))for k in ['bin_id_1based','source_N','REAL_N','h_mean','target_affinity','lambda_value','achieved_affinity','affinity_bias','boundary_failure','Kish_ESS','weight_ratio'])+'|')
lines+=['','h=.15*had/(ee+.15*had)只用于源采样/明确标注的辅助分析，未作为新 NN 输入。低/高实测密度成分是操作性响应带，没有粒子真值；h 与高带的对应是待检验假设。正权重比最多10，未达目标的边界失败保留。采样改变同时影响全部源相关内容/电荷/排序损失的暴露频率，不能进一步归因于只有对抗项的类别先验。实际抽样组成与唯一源覆盖见 SOURCE_TILT_TABLE.json、PAIRING_RECEIPT.json 与完整训练回执。',
'','## 原总体 PSD 与预指定纯 had 结果','',
'|域|QC/2000|整体PSD W1|八区均值PSD W1|bin3谷区差|bin4谷区差|',
'|---|---:|---:|---:|---:|---:|']
for a in ['SOURCE_CLEAN','R18','CONT','TILT']:
    d=m['PSD_main']['vs_REAL'][a];globaldist=m['domain_vs_REAL'].get(a,{});pw=globaldist.get('overall',{}).get('PSD',{}).get('W1')if isinstance(globaldist,dict)else None
    lines.append('|'+ '|'.join([a,str(m['PSD_main']['tables'][a]['QC_n']),f(pw),f(d['mean8_W1']),f(d['bins'][2]['valley_fraction_difference']),f(d['bins'][3]['valley_fraction_difference'])])+'|')
lines+=['','纯 had 预先指定 source-bin3/4 的输出谷区比例（共同父条件，非输出筛选）：','',
'|域|bin3 N/谷区比例|bin4 N/谷区比例|','|---|---:|---:|']
for a,r in m['source_composition']['primary_pure_had_source_bins3_4'].items():lines.append('|'+a+'|'+str(r['bin3']['N'])+' / '+f(r['bin3']['valley_fraction'])+'|'+str(r['bin4']['N'])+' / '+f(r['bin4']['valley_fraction'])+'|')
lines+=['','所有八区、pureEE/had/mixed、源条件/输出条件及迁移表均保留在 metrics.json。TRAIN 冻结权重下的比例标准化仅为辅助比较，ESS明确列出，不替代主原总体。两成分峰位、原宽度、去趋势宽度仅在共同合格区间上比较，不合格区间保留 undefined。',
'','## 完整响应、重复噪声与源关联','',
'|视图|R18 BA|CONT BA|TILT BA|TILT−CONT BA /95%CI|','|---|---:|---:|---:|---|']
for v in m['C2ST']['results']['CONT']:
    r=m['C2ST']['TILT_minus_CONT_paired_CI'][v];lines.append('|'+v+'|'+f(m['C2ST']['reference_results_reused']['TRANSLATED'][v]['balanced_accuracy'])+'|'+f(m['C2ST']['results']['CONT'][v]['balanced_accuracy'])+'|'+f(m['C2ST']['results']['TILT'][v]['balanced_accuracy'])+'|'+f(r['BA_delta_first_minus_second'])+' / '+str(r['BA_delta_95CI'])+'|')
lines+=['','HGB为固定检验器，无调参；分类接近0.5不证明等效。SOURCE/REAL–REAL/R18结果复用，不重复拟合。区组置信区间条件于这些已拟合检验器，不含响应训练不确定性。主波形中位、噪声、自相关、上升沿和完整条件分布同看，不能以平均曲线或边际PSD改善代替完整响应相似。',
'','|臂|纯EE同父PSD SD均值|纯had同父PSD SD均值|mixed同父PSD SD均值|跨谷区父数|固定源换噪声RMS mV|','|---|---:|---:|---:|---:|---:|']
for a,r in m['same_parent_noise'].items():
    z=m['source_probes'][a]['rows']['fixed_source_changed_noise'];lines.append('|'+a+'|'+ '|'.join(f(r['by_composition'][c]['within_parent_PSD_SD']['mean'])for c in ['pure_EE','pure_had','mixed'])+'|'+str(sum(v['parents_straddling_valley']for v in r['by_composition'].values()))+'|'+f(z['waveform_RMS_vs_correct_reference_V']['mean']*1000)+'|')
lines+=['','重复噪声共享128父、每父8次，未增加独立 G4 父记录。探针复用 R18 已修正关系：固定源参考其一个实际父，源滚动替换与噪声来源/参考父分开保存。非零源敏感性不证明正确逐事件物理对应。源是G4总EE/had沉积经既有解析响应形成的480点波形，未含微观沉积时序；不能声称网络恢复了未提供的时序信息。DEV PDG由旧绑定仅22/2112的CSV及旧2112/non2112桶推定，区别于TRAIN显式PDG；纯沉积组不是实测真值标签。',
'','## 资源与复现','',
f"远端监督 PID {native['supervisor_PID']}；累计科学 {f(native['numeric_cumulative_s'])} s，CPU4，峰RSS {resource['RSS_peak_bytes']} B，GPU allocated峰 {resource['GPU_allocated_peak_bytes']} B。实际 {m['counters']['G_updates']} G/{m['counters']['D_updates']} D、{m['counters']['HGB_attempts']} HGB/{m['counters']['density_EM_attempts']} 新描述EM。全部失败保留；无EVAL/CAL波形/新G4/原始长波形。正常传输不做迁移一致性测试。",
f"独立新进程主生成重放 {replay['rows']} 条、{replay['points']} 点，pass={replay['pass']}；每臂 pre/post ADC bitwise 结果见 REPLAY_RECEIPT.json。R18模型、两臂所选/末权重、实际 source/z/white、训练真实索引/RNG、原生日志均保留。见 REPRODUCTION.md。",
'','本轮结论见下方人工解读；后续方向由根代理根据整体证据登记，本轮不追加第三臂或再调配方。']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf-8')
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'science_recomputed':False,'report_written':True,'resource':resource['science_cumulative_s']}))
