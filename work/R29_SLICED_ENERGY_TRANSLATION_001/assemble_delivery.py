from pathlib import Path
import os, json, datetime, hashlib

ROOT = Path(__file__).resolve().parent
def load(name): return json.loads((ROOT/name).read_text('utf8'))
def write(name, value):
    (ROOT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', 'utf8')
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def f(x, places=6): return 'undefined' if x is None else f'{x:.{places}f}'

m, history, cfg = load('metrics.json'), load('RUN_HISTORY.json'), load('config.json')
replay, plots = load('REPLAY_RECEIPT.json'), load('PLOT_RECEIPT.json')
t, old_t = m['training']['SLICED_ENERGY'], m['control_training_reused']
names = ['FRESH_SELECTED','FRESH_FINAL','ENERGY_SELECTED','ENERGY_FINAL']
views = [*cfg['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480']
science_s = sum(r['elapsed_s'] for r in history)

# Only aggregate summaries are published. Individual identities, predictions and
# the actual batch stream remain in the complete local scientific artifacts.
def trim_training(value):
    fields = ['arm','updates','D_updates','style_reader_updates','training_s','selected_step',
              'stream_hashes','validation_support','same_time_selected_G_reader_D','selected_using_DEV_or_D',
              'actually_sampled_unique_source','actually_sampled_unique_REAL','actually_consumed_rows',
              'completed_all_original_rows','training_stop_reason','GPU_peak_allocated_bytes','device',
              'critic_STANDARDIZED_input','fresh_projection_calls','fine_validation_consumes_fresh_RNG',
              'fine_training_distance','global_gradient_norms','energy_training_summary']
    answer = {k:value[k] for k in fields if k in value}
    vf = ['step','J','J19_weighted','fine_val','fine_groups','weighted_physical_W1','weighted_shape_SW',
          'source_logQ_abs_change','PSD_rank_violation','Q_rank_violation','online_D_weighted_BA']
    def tv(v): return {k:v[k] for k in vf if k in v}
    for key in ['initial_validation','selected_validation','final_validation']:
        answer[key] = tv(value[key])
    answer['validation_trace'] = [{'validation':tv(r['validation'])} for r in value['validation_trace']]
    return answer
def width_table(value):
    fields = ['bin','N','valid','converged','fit_attempts','failure_reasons','components',
              'charge_edges_mVns','actual_charge_range_mVns']
    return {'slices':[{k:r[k] for k in fields if k in r} for r in value['slices']]}
random_summary = {n:{k:v for k,v in value['groups']['joint8'].items() if k!='rows'}
                  for n,value in m['random_decomposition'].items()}
probe_summary = {n:{k:v for k,v in value.items() if k!='fixed_source_parent_index'}
                 for n,value in m['source_probes'].items()}
compact = {
    'stage':ROOT.name,'scientific_outcome':'local_PSD_width_gain_complete_response_worse_recipe_not_promoted',
    'training':{'SLICED_ENERGY':trim_training(t),'control_R28_FRESH_FINE':trim_training(old_t)},
    'data_roles':m['data_roles'],'domain_vs_REAL':m['domain_vs_REAL'],'PSD_main':m['PSD_main'],
    'energy_conditional_width':{'tables':{n:width_table(v) for n,v in m['energy_conditional_width']['tables'].items()},
                                'common_support_errors':m['energy_conditional_width']['common_support_errors'],
                                'density_components_not_particle_truth':True},
    'source_composition_primary':m['source_composition']['primary_pure_had_source_bins3_4'],
    'source_composition_full_aggregates_location':'metrics.json:source_composition',
    'source_probes_aggregate':probe_summary,'random_decomposition_aggregate':random_summary,
    'C2ST':m['C2ST'],'projection_coverage':m['projection_coverage'],
    'projection_common_support':m['projection_distances']['common_support_summary'],
    'projection_block_range_is_not_CI':True,'row_exposure':m['row_exposure'],
    'counter_evaluation_scope':m['counters'],
    'replay_summary':{'pass':replay['pass'],'rows':replay['rows'],'independent_process':True,'maxabs_V':0.0},
    'private_payload_policy':'individual source/REAL identities, parent records, predictions and actual batch/projection-call logs remain local',
    'source_association':{},
}
for n,v in m['source_association'].items():
    groups = {}
    for key,row in v['composition_and_PDG'].items():
        groups[key] = {k:row[k] for k in ['source_n','joint_QC_n','source_Q_to_output_Q','source_PSD_to_output_PSD','logQ_change'] if k in row}
        groups[key]['source_Q_bins'] = [{k:r[k] for k in ['source_bin_1based','N','PSD_ranking'] if k in r} for r in row.get('bins',[])]
    compact['source_association'][n] = {'composition_and_PDG':groups}

omitted = []
def aggregate_only(value, path='root'):
    if isinstance(value, dict):
        result = {}
        for k,v in value.items():
            kl = k.lower()
            if kl.endswith('_indices') or kl.endswith('_ids') or kl in ['predictions','labels','source_parent_index','parent_index','original_index']:
                omitted.append({'path':path+'.'+k,'reason':'individual payload field'})
                continue
            result[k] = aggregate_only(v, path+'.'+k)
        return result
    if isinstance(value, list):
        if len(value)>32 and all(not isinstance(x,(dict,list)) for x in value):
            raw = json.dumps(value,separators=(',',':'),ensure_ascii=False).encode('utf8')
            omitted.append({'path':path,'count':len(value),'sha256':hashlib.sha256(raw).hexdigest()})
            return {'private_payload_omitted':True,'count':len(value),'summary_sha256':hashlib.sha256(raw).hexdigest()}
        return [aggregate_only(v, path+'[]') for v in value]
    return value
compact = aggregate_only(compact)
compact['aggregation_policy_omissions'] = omitted
write('METRICS_COMPACT.json', compact)

failures = [
    {'tool_chunk_id':'1bd996','native_exit':1,'PID':'unknown_not_returned','elapsed_s':0.441103,
     'phase':'pure control inline JSON summary command','error':'SyntaxError: unexpected character after line continuation character',
     'cause':'literal backslash-n was appended to Python -c source; parse failed before file reading',
     'source_data_read_bytes':0,'science_seconds':0,'response_updates':0,'HGB':0,'EM':0,
     'original_tool_output':'original tool output was truncated by max_output_tokens=10; failure and real native exit remain in tool history',
     'repair':'corrected command quoting; no scientific job rerun'},
    {'tool_chunk_id':'1d8cd1','native_exit':1,'PID':'unknown_not_returned','elapsed_s':0.434353,
     'phase':'pure control inspection of saved metrics','error':"KeyError: 'observed_Q'",
     'cause':'projection_coverage also contains metadata entries; inspection assumed every entry was a domain',
     'source_data_read_bytes':(ROOT/'metrics.json').stat().st_size,'science_seconds':0,'response_updates':0,'HGB':0,'EM':0,
     'repair':'subsequent inspection restricted to dictionaries containing observed_Q; saved science unchanged'},
]
write('CONTROL_FAILURES.json',{'failures':failures,'total_elapsed_s':sum(v['elapsed_s'] for v in failures),'scientific_reruns':0})
tools = {
    'check_energy_formula.py':{'terminal_tool':'0e1fcc','tool_native_exit':0},
    'train_gpu.py':{'launch_tool':'32e417','session_id':94360,'terminal_tool':'09dfc0','tool_native_exit':0},
    'evaluate_endpoints.py':{'launch_tool':'377e59','session_id':36016,'terminal_tool':'28d267','tool_native_exit':0},
    'replay_endpoints.py':{'terminal_tool':'e18100','tool_native_exit':0},
    'render_figures.py':{'terminal_tool':'ebe49f','tool_native_exit':0},
}
write('NATIVE_CHAIN.json',{'jobs':[dict(r, tool_receipt=tools[r['entry']]) for r in history],
                          'control_failures':failures,'assembly_PID':os.getpid(),'assembly_started_UTC':utc(),
                          'assembly_native_exit_evidence':'external calling tool receipt, attached at seal'})
resource = {
    'stage':ROOT.name,'registered_UTC':'2026-10-09T14:52:30.236280+00:00',
    'first_preparation_UTC':'2026-10-09T14:53:11.9328658+00:00','first_preparation_PID':53192,'first_preparation_tool':'02cd4f',
    'executor_seal_deadline_UTC':'2026-10-09T16:12:30.236280+00:00','root_deadline_UTC':'2026-10-09T16:22:30.236280+00:00',
    'science_elapsed_s':science_s,'science_budget_s':1800,'training_s':t['training_s'],'training_limit_s':900,
    'response_fit_attempts':1,'G_updates':t['updates'],'D_updates':t['D_updates'],'reader_updates':t['style_reader_updates'],
    'formula_check_seconds':next(r['elapsed_s'] for r in history if r['entry']=='check_energy_formula.py'),
    'HGB_attempts':m['counters']['HGB_attempts'],'HGB_cap':20,
    'density_description_slots':m['counters']['density_description_slots'],'density_EM_attempts':m['counters']['density_EM_attempts'],'EM_cap':16,
    'main_generated_rows':m['counters']['main_generation_rows'],'independent_main_replay_rows':replay['rows'],
    'joint_repeat_rows':m['counters']['repeat_noise_rows'],'source_probe_rows':m['counters']['source_probe_rows'],
    'actual_worker_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes'] for r in history),'RSS_limit_bytes':4294967296,
    'GPU_peak_allocated_bytes':t['GPU_peak_allocated_bytes'],'GPU_allocated_limit_bytes':4294967296,
    'memory_probe_failures':sum(r['memory_probe_failures'] for r in history),'threads':4,
    'scientific_failures':sum(r['actual_worker_native_exit']!=0 for r in history),'control_failures':failures,
    'actual_PID_monitoring':'nonce and before-import PID handshake with real child handle/RSS acknowledgment; launcher RSS is separate',
    'worker_receipt_counters_scope':'evaluation response_updates=0 applies only to evaluation; stage has 2950 G/D/reader training updates each',
    'training_environment':{'python':'E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe','torch':'2.12.0+cu130','NumPy':'1.26.4','device':'cuda:0'},
    'evaluation_replay_environment':m['versions'],
    'no_SSH':True,'no_install':True,'no_new_G4':True,'EVAL_scientific_read':0,'CAL_waveform_decode':0,'raw_long_waveform_decode':0,
    'output_limit_bytes':268435456,'stage_output_bytes':'filled_by_final_seal','science_fits_or_array_recomputations_during_finalization':0,
}
write('RESOURCE_RECEIPT.json',resource)
write('FIGURE_QA.json',{'executor_visual_review_UTC':utc(),'all_11_final_figures_actually_viewed':True,
                       'root_all_11_final_figures_reviewed':True,'layout_fixes_needed':False,
                       'remaining_layout_issues_observed':[],'scientific_values_changed':False,'final_figures':plots['figures'],
                       'review_points':['common axes and complete charge/PSD populations','shortened C2ST labels readable',
                                        'bin8 support gap retained','fixed median tail structure visible',
                                        'source-bin3/4 complete parents','direction block range is not a CI',
                                        'conditional PSD widths remain narrower than REAL']})

c2rows = []
for view in views:
    new = m['C2ST']['results']['ENERGY_FINAL'][view]
    p = m['C2ST']['paired_vs_corresponding_R28']['ENERGY_FINAL'][view]
    c2rows.append([view,f(p['old_control_result']['balanced_accuracy'],3),f(new['balanced_accuracy'],3),
                  f(p['BA_delta_first_minus_second'],3),'['+', '.join(f(x,4) for x in p['BA_delta_95CI'])+']'])
projrows = []
for view,s in m['projection_distances']['common_support_summary'].items():
    d=s['domains'];projrows.append([view,f(d['FRESH_FINAL']['equal_mean_on_common_bins']['ORIGINAL']),
        f(d['ENERGY_FINAL']['equal_mean_on_common_bins']['ORIGINAL']),f(d['FRESH_FINAL']['new_block_mean']),
        f(d['ENERGY_FINAL']['new_block_mean']),f(d['TRAIN_REFERENCE']['new_block_mean'])])
widthrows = [['low PSD' if row['component']==0 else 'high PSD',row['field'],f(row['MAE']['FRESH_FINAL']),f(row['MAE']['ENERGY_FINAL'])]
             for row in m['energy_conditional_width']['common_support_errors']['final_equal_updates']]
pure_rows = []
for name in names:
    for binname,row in m['source_composition']['primary_pure_had_source_bins3_4'][name].items():
        pure_rows.append([name,binname,row['N'],f(row['output_PSD']['mean']),f(row['output_PSD']['sample_sd']),f(row['valley_fraction']*100,3)+'%'])
overall_rows = [[key,f(m['domain_vs_REAL']['FRESH_FINAL']['overall'][key]['W1']),f(m['domain_vs_REAL']['ENERGY_FINAL']['overall'][key]['W1'])]
                for key in ['PSD','Qtotal_mVns','baseline_RMS_V','ACF1','rise_10_90_ns','tail_charge_fraction','tail_mean_over_peak']]
energy_rows = [[name,f(row['mean_twice_cross']),f(row['mean_generated_self']),f(row['mean_REAL_self']),f(row['mean_U']),
                f(row['min_U']),str(row['negative_U_count'])+'/2950',f(row['mean_negative_direction_fraction'],4)]
               for name,row in t['energy_training_summary']['groups'].items()]
jobrows = [[r['entry'],r['actual_worker_PID'],r['supervisor_PID'],f(r['elapsed_s'],4),r['actual_worker_native_exit'],r['actual_worker_RSS_peak_bytes']] for r in history]
mean = {n:m['projection_coverage'][n]['observed_Q'][0] for n in ['FRESH_FINAL','ENERGY_FINAL']}
report = f'''R29 的单次 sliced energy 替换未改善完整响应，保留为负结果。2950对2950末点比较中，完整形状BA从0.970升至0.992，三组复用新方向W1均退化；PSD整体W1及部分条件峰宽误差改善，同时pure-had源条件谷区占比增加。不能把这些局部PSD收益当作整体仿真→实测转换成功，不推广此配方，不追加训练或挑选DEV端点。

本轮只把fine训练距离由分位数L1改为三对角线排除的energy U。每个128列投影块中的一列计算 sum(i≠j)[2|Ai−Bj|−|Ai−Aj|−|Bi−Bj|]/[n(n−1)]，再平均128列与三组。交叉、生成自项、实测自项三个对角线都排除，两个生成位置均反传；实测自项保留在报告值，负U不截断、不取绝对值/平方根。这同时改变目标函数和自然梯度尺度，不能把差异只归因于“去除有限批收缩偏差”，也不声称实际SGD流满足IID无偏估计前提。小张量向量/显式循环与两侧梯度检查已执行，误差4.44e−16/5.55e−17，负例U=−2按原定义保留。

完整480点G4派生源输入、网络、原其他loss/权重、STANDARDIZED critic、D:G=1:1、Adam与clip5冻结。复用R28/R26的2950×32=94400条实际行、2946同Q批次与4余数混合批次，source/REAL/bin/原ordinal/z/white/style保持绑定，每步NCE位置不改。NCE负例在同波形时间patch内，更新历史改变但没有新增跨事件NCE。独立PCG64(20261009281)每G步按shape480/diff479/baseline100顺序生成各128单位方向，两域共用；2950次逐ordinal摘要与R28一致，D/identity/验证不消费它，首末矩阵/状态/全部调用摘要保存。

源6593 TRAIN父、1599内部val中固定512源用于验证，实测21047 TRAIN与599 val；主2000源父与2000 REAL DEV一条不删、不重加权。417 pure-EE、1518 pure-had、65 mixed均保留，317/1683 PDG计数由既有标签规则及R2绑定CSV的仅22/2112证据推定，并非新逐事件读取。G4总沉积驱动的解析源没有新增微观沉积时序；历史噪声和采样先验暴露保留。DEV/人工50记录分组来自同采集，不能称独立run或外部盲测；EVAL/CAL波形/长原始波形/新G4/SSH为0。

原固定J20仍由W1项选点，step0/每250/末点严格小于才替换，不消费fresh流，不看DEV或D。Energy初J={t['initial_validation']['J']:.9f}，所选{t['selected_step']}步J={t['selected_validation']['J']:.9f}，末点J={t['final_validation']['J']:.9f}；Fresh旧选{old_t['selected_step']}步J={old_t['selected_validation']['J']:.9f}，末点J={old_t['final_validation']['J']:.9f}。两个同步G/reader/D及优化器端点都保存。所选的DEV PSD W1为0.033760，Fresh所选0.032535；预指定主要比较是2950对2950，不用DEV再选更好端点。

训练U三项按实际2950次更新汇总，单位来自冻结TRAIN逐坐标标准化，不能与分位数W1训练曲线的绝对量跨目标比较。负方向比例是每批128列中负U的比例均值。

{table(['组','2×交叉项均值','生成自项均值','REAL自项均值','U均值','最小U','负批数','平均负方向比例'],energy_rows)}

三组平均U的负批数为0；diff有7个负批、baseline有389个负批，原值均保留。4个余数批的原更新ordinal为137/268/2169/2837，各组三项/U及bin构成另存ENERGY_TERMS_PRIVATE与训练摘要，不混称全部同区IID批次。训练和梯度/clip轨迹均保存。

共同合格输出Q区为bins1–7，以下在相同区间等权汇总。原方向与四块R27方向继续按原W1评价；四块已用于DEV诊断，属于开发参照。块间范围不是CI或物理等效容差。

{table(['视图','Fresh末点原方向','Energy末点原方向','Fresh四块均值','Energy四块均值','TRAIN–DEV四块参照'],projrows)}

三个新方向均值都变差，且远高于TRAIN–DEV参照；不能由U优化证明联合时间结构已匹配。覆盖空间仍为rank390、正交补90，仅对应列明线性映射，不是整个loss的盲区。总体均值残差平方范数Fresh {mean['FRESH_FINAL']['mean_difference_sqnorm']:.6f}→Energy {mean['ENERGY_FINAL']['mean_difference_sqnorm']:.6f}，Energy约{100*mean['ENERGY_FINAL']['seen_fraction']:.2f}%在覆盖空间、{100*mean['ENERGY_FINAL']['complement_fraction']:.2f}%在补空间；两空间C2ST仍高度可分。

20个预定HGB逐fit落盘；每域600训练/250留出及实际组位置沿用，不重抽。500次配对区组bootstrap条件于固定已拟合分类器，不包括响应训练不确定性，不作等效阈值。

{table(['视图','Fresh末点BA','Energy末点BA','Energy−Fresh','配对95%CI'],c2rows)}

完整shape493变差的CI不含0；纯u480也变差。QPSD二维BA下降但配对CI含0；postpeak/tail仍约0.96–0.97，不能称匹配。shape493含13标量，不能顶替纯480点视图。

各观测量的整体W1并列，字段单位不同，不能跨字段比较绝对量。

{table(['观测量','Fresh末点W1','Energy末点W1'],overall_rows)}

PSD W1小改善与tail charge fraction改善保留；电荷、rise、基线RMS及ACF并未同步改善。条件中位波形仍呈实测中位不明显的固定尾段起伏，平均曲线接近不等于总体响应接近。

16个新密度槽实际14次EM；Energy selected bin8有36条、final有35条，均低于40支持门槛而undefined，旧Fresh两端点36/37同样缺口。不补拟合、不删缺口。两成分只是密度描述，不是粒子真值；四端点共同支持bins1–7的MAE如下。

{table(['密度成分','字段','Fresh末点MAE','Energy末点MAE'],widthrows)}

低带中心及两带原宽度/去趋势宽度MAE改善，高带中心略退化；各区宽度仍明显偏窄，最高区无足够新臂宽度证据。上述共同支持表不能代替完整8区分布/缺口。

源Q与输出Q条件分开。预指定pure-had源bin3/4保留全部207/182父记录：

{table(['端点','源Q区','父数','输出PSD均值','样本SD','冻结跨域谷区比例'],pure_rows)}

末点谷区占比bin3从10.628%增至14.976%，bin4从0.549%增至2.747%；这是源条件变化及潜在保真风险，不能仅凭跨域谷区断言粒子内容错误。Energy所选仍是0谷占比，但同样不能据此宣称正确保真或事后推广它。同父8次随机输出、分量方差和交叉项保存；更多PSD扩散不自动等同探测器真实涨落。固定源改噪声RMS末点4.639→5.132mV，源roll末点15.119→15.199mV，探针对应正确源/参考父；非零敏感性不证明物理关联。signal/electronic共享输出归一化且可能相关，signal不是条件期望，分解不证明物理独立噪声。

下一优先问题建议先对冻结输出按相同Q区分开“条件均值残差”和“去条件均值后的联合波动”：固定尾段起伏、两带窄宽与全波形高BA可能来自不同部分。用保存输出的有界诊断判断主要差距位于均值结构还是波动，再决定训练干预；本轮不再加loss/投影/权重搜索。当前负结果不证明energy距离一般无效、不证明W1必然收缩或某个架构伪影是唯一原因。

一次独立CPU保存模型主重放4000条通过，pre/post ADC、signal/electronic及源ID逐值一致，maxabs=0；没有重跑评价。主输出全有限、无ADC裁剪。CUDA训练与CPU主生成/重放的环境分别记录，未声称跨设备训练位级复现。11张最终PNG执行器与根均已实际查看，短C2ST标签可读，未追加布局计算。

{table(['入口','实际worker PID','监督PID','监督墙钟s','worker退出','RSS峰B'],jobrows)}

累计科学{science_s:.6f}s/1800s含公式检查/训练/评价/重放/绘图，训练{t['training_s']:.6f}s/900s，G/D/reader各2950。实际RSS峰{resource['actual_worker_RSS_peak_bytes']}B，GPU allocated峰{resource['GPU_peak_allocated_bytes']}B，均低于4GiB。科学失败0；两次纯控制命令错误分别为转义SyntaxError与覆盖metadata字段KeyError，原tool/native/费用保存，未引起任何科学重跑。完整逐事件/身份/预测/行流留本地；紧凑发布仅聚合统计。最终文件字节/SHA、真实封存退出和停写时间见FINAL_MANIFEST/FINAL_RECEIPT。
'''
(ROOT/'REPORT.md').write_text(report,'utf8')

reproduction = '''本轮在本机E盘执行，无SSH。训练Python：E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe；公式检查也使用该既有Torch。CPU评价/重放/绘图Python：C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe，只读解释器，数据/临时/缓存写stage/cache。线程4，supervise.py/owned_worker.py在科学导入前绑定真实PID/nonce并确认RSS守护，launcher单独记录。

顺序是supervise.py check_energy_formula.py GPU_PYTHON → train_gpu.py GPU_PYTHON → evaluate_endpoints.py CPU_PYTHON → replay_endpoints.py CPU_PYTHON → render_figures.py CPU_PYTHON；上述所有入口均受同supervise守护，具体native链在NATIVE_CHAIN/RUN_HISTORY。原目录.used禁止重复提交；复现应在新E/D目录复制必要文件并调整路径，独立注册预算/期限，不能删除原.used或覆盖旧阶段。

config/PROTOCOL/PRETRAIN_FREEZE保存首次训练前的方法和执行源SHA；EVALUATION_SOURCE_FREEZE固定评价与绘图源。R23 TRAIN_PRIVATE/VALIDATION_INPUTS/FINE_CONSTANTS/CRITIC_SCALER只读复用；reorder_core用原R23流和R26实际重排行/状态重建全部94400行，R21 step2000 G/reader与R18 INITIAL_CRITIC共同起点。SLICED_ENERGY/TRANSLATOR.pt、STYLE_READER.pt、CRITIC.pt、OPTIMIZER_SELECTED.pt为2750同步所选；FINAL_*和OPTIMIZER_FINAL.pt为2950末点。

energy_core.py只覆盖训练距离，validation继承原W1。ENERGY_TERMS_PRIVATE.json保存每次调用三个交叉/自项、U/负方向比例/混合bin上下文；FRESH_PROJECTION_STREAM.json保存逐ordinal与R28匹配，FIRST/LAST_PRIVATE和TRAINING_RANDOM_STATES存原随机状态。config记录独立PCG64种子，验证不消费它；训练项自然尺度未重权重。

ACTUAL_GENERATION_INPUTS/JOINT8_INPUTS_PRIVATE/PROBE_INPUTS_CORRECTED是保存的实际推理输入。ENERGY_SELECTED/generated.npz与ENERGY_FINAL/generated.npz各2000 pre/post ADC、signal/electronic、源ID；model.py纯推理重放用同步保存权重。REPLAY_RECEIPT及真实native回执证明4000条独立重放已执行且逐值一致；不再重做完整评价。

旧R28控制输出/预测/密度和R27固定线性空间/四投影块按config只读复用，固定600/250行及组保持，不重fit旧结果。这些DEV方向已暴露，不称新外部盲测。完整metrics/predictions/源身份/actual stream/模型留E/D，METRICS_COMPACT为聚合发布版。最终清单只封存本stage字节SHA，不重审3GB历史源或运行迁移一致性测试。
'''
(ROOT/'REPRODUCTION.md').write_text(reproduction,'utf8')
write('ASSEMBLY_RECEIPT.json',{'PID':os.getpid(),'UTC':utc(),'entry':'assemble_delivery.py','pure_control_only':True,
                             'new_fits_generation_updates':0,'scientific_inputs_decoded':0,
                             'files':['REPORT.md','REPRODUCTION.md','METRICS_COMPACT.json','RESOURCE_RECEIPT.json','FIGURE_QA.json','NATIVE_CHAIN.json','CONTROL_FAILURES.json'],
                             'native_exit_evidence':'calling exec tool receipt'})
print(json.dumps({'PID':os.getpid(),'UTC':utc(),'science_total_s':science_s,'report_bytes':(ROOT/'REPORT.md').stat().st_size,
                  'compact_bytes':(ROOT/'METRICS_COMPACT.json').stat().st_size,'omitted_payload_fields':len(omitted),
                  'scientific_computation':0}),flush=True)
