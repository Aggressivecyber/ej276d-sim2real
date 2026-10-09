from pathlib import Path
import os, json, datetime, hashlib

ROOT = Path(__file__).resolve().parent
def load(name):
    return json.loads((ROOT/name).read_text('utf8'))
def write(name, value):
    (ROOT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', 'utf8')
def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def f(x, places=6):
    return 'undefined' if x is None else f'{x:.{places}f}'

m = load('metrics.json')
history = load('RUN_HISTORY.json')
cfg = load('config.json')
replay = load('REPLAY_RECEIPT.json')
plots = load('PLOT_RECEIPT.json')
t = m['training']['FRESH_FINE']
old_t = m['control_training_reused']
names = ['HOMO_SELECTED','HOMO_FINAL','FRESH_SELECTED','FRESH_FINAL']
views = [*cfg['C2ST_views'],'normalized_u480','seen_reconstruction480','complement_reconstruction480']

# Publish aggregate statistics only; retain full per-parent, prediction and actual-stream payloads locally.
train_fields = ['arm','updates','D_updates','style_reader_updates','training_s','selected_step','initial_validation','selected_validation','final_validation','validation_trace','stream_hashes','validation_support','same_time_selected_G_reader_D','selected_using_DEV_or_D','actually_sampled_unique_source','actually_sampled_unique_REAL','actually_consumed_rows','completed_all_original_rows','training_stop_reason','GPU_peak_allocated_bytes','device','critic_STANDARDIZED_input','fresh_projection_calls','fine_validation_consumes_fresh_RNG']
def trim_training(value):
    answer = {k:value[k] for k in train_fields if k in value}
    val_fields = ['step','J','J19_weighted','fine_val','fine_groups','Q_relative','PSD_rank','online_D_weighted_BA','online_D_AUC']
    def trim_val(v):
        return {k:v[k] for k in val_fields if k in v}
    for key in ['initial_validation','selected_validation','final_validation']:
        if key in answer:answer[key]=trim_val(answer[key])
    if 'validation_trace' in answer:
        answer['validation_trace']=[{'validation':trim_val(r['validation'])} for r in answer['validation_trace']]
    return answer
def width_table(value):
    rows = []
    for r in value['slices']:
        rows.append({k:r[k] for k in ['bin','N','valid','converged','fit_attempts','failure_reasons','components','charge_edges_mVns','actual_charge_range_mVns'] if k in r})
    return {'slices':rows}
random_summary = {}
for n, value in m['random_decomposition'].items():
    g = value['groups']['joint8']
    random_summary[n] = {k:v for k,v in g.items() if k!='rows'}
probe_summary = {}
for n, value in m['source_probes'].items():
    probe_summary[n] = {k:v for k,v in value.items() if k!='fixed_source_parent_index'}
compact = {
    'stage':'R28_FRESH_PROJECTION_TRANSLATION_001',
    'scientific_outcome':'partial_direction_generalization_gain_complete_response_not_matched',
    'training':{'FRESH_FINE':trim_training(t),'control_R26':trim_training(old_t)},
    'data_roles':m['data_roles'],
    'domain_vs_REAL':m['domain_vs_REAL'],
    'PSD_main':m['PSD_main'],
    'energy_conditional_width':{'tables':{n:width_table(v) for n,v in m['energy_conditional_width']['tables'].items()},'common_support_errors':m['energy_conditional_width']['common_support_errors'],'density_components_not_particle_truth':True},
    'source_composition_primary':m['source_composition']['primary_pure_had_source_bins3_4'],
    'source_composition_full_aggregates_location':'metrics.json:source_composition',
    'source_probes_aggregate':probe_summary,
    'random_decomposition_aggregate':random_summary,
    'C2ST':m['C2ST'],
    'projection_coverage':m['projection_coverage'],
    'projection_common_support':m['projection_distances']['common_support_summary'],
    'projection_block_range_is_not_CI':True,
    'row_exposure':m['row_exposure'],
    'counter_evaluation_scope':m['counters'],
    'replay_summary':{'pass':replay['pass'],'rows':replay['rows'],'independent_process':True,'maxabs_V':0.0},
    'private_payload_policy':'actual source/REAL indices, per-parent records, predictions and projection-call logs remain in local full artifacts; this file contains aggregate summaries',
}
compact['source_association']={}
for n,v in m['source_association'].items():
    groups={}
    for key,row in v['composition_and_PDG'].items():
        groups[key]={k:row[k] for k in ['source_n','joint_QC_n','source_Q_to_output_Q','source_PSD_to_output_PSD','logQ_change'] if k in row}
        groups[key]['source_Q_bins']=[{k:r[k] for k in ['source_bin_1based','N','PSD_ranking'] if k in r} for r in row.get('bins',[])]
    compact['source_association'][n]={'composition_and_PDG':groups}
write('METRICS_COMPACT.json', compact)

control_failure = {
    'tool_chunk_id':'3cf639','native_exit':1,'PID':'unknown_not_returned',
    'elapsed_s':0.4509189,'phase':'stdlib-only metadata inspection of already saved metrics',
    'error':"AttributeError: 'str' object has no attribute 'get'",
    'cause':'treated source_probes.rows dictionary as a list; no scientific job, fit, array decode or update occurred',
    'cost_preserved':True,'science_seconds':0,'response_updates':0,'HGB':0,'EM':0,
    'input':'metrics.json','read_bytes':(ROOT/'metrics.json').stat().st_size,
    'repair':'subsequent control inspection used the existing dictionary schema; scientific artifacts unchanged',
}
write('CONTROL_FAILURE_001.json', control_failure)
tools = {
    'train_gpu.py':{'launch_tool':'56578e','session_id':59618,'terminal_tool':'e3a371','tool_native_exit':0},
    'evaluate_endpoints.py':{'launch_tool':'0785ad','session_id':70591,'terminal_tool':'6fcc0b','tool_native_exit':0},
    'replay_endpoints.py':{'terminal_tool':'c412d3','tool_native_exit':0},
    'render_figures.py':{'terminal_tool':'c8d22a','tool_native_exit':0},
    'fix_c2st_layout.py':{'terminal_tool':'5672b7','tool_native_exit':0},
}
write('NATIVE_CHAIN.json', {'jobs':[dict(r, tool_receipt=tools[r['entry']]) for r in history], 'control_failures':[control_failure], 'assembly_PID':os.getpid(),'assembly_started_UTC':utc(),'assembly_final_native_exit':'external_tool_receipt_required'})
science_s = sum(r['elapsed_s'] for r in history)
resource = {
    'stage':'R28_FRESH_PROJECTION_TRANSLATION_001','registered_UTC':'2026-10-09T14:11:09.766899+00:00',
    'executor_seal_deadline_UTC':'2026-10-09T15:31:09.766899+00:00','root_deadline_UTC':'2026-10-09T15:41:09.766899+00:00',
    'science_elapsed_s':science_s,'science_budget_s':1800,'training_s':t['training_s'],'training_limit_s':900,
    'response_fit_attempts':1,'G_updates':t['updates'],'D_updates':t['D_updates'],'reader_updates':t['style_reader_updates'],
    'HGB_attempts':m['counters']['HGB_attempts'],'HGB_cap':20,
    'density_description_slots':m['counters']['density_description_slots'],'density_EM_attempts':m['counters']['density_EM_attempts'],'EM_cap':16,
    'main_generated_rows':4000,'independent_main_replay_rows':4000,'joint_repeat_rows':2048,'source_probe_rows':1664,
    'actual_worker_RSS_peak_bytes':max(r['actual_worker_RSS_peak_bytes'] for r in history),'RSS_limit_bytes':4294967296,
    'GPU_peak_allocated_bytes':t['GPU_peak_allocated_bytes'],'GPU_allocated_limit_bytes':4294967296,
    'memory_probe_failures':sum(r['memory_probe_failures'] for r in history),'threads':4,
    'scientific_failures':sum(r['actual_worker_native_exit']!=0 for r in history),'control_failures':[control_failure],
    'layout_fix_only_s':next(r['elapsed_s'] for r in history if r['entry']=='fix_c2st_layout.py'),
    'actual_PID_monitoring':'nonce and before-import PID handshake, actual child process handle and RSS acknowledgment; venv launcher RSS is not the worker peak',
    'worker_receipt_counters_scope':'evaluation response_updates=0 describes evaluation only; whole-stage training updates are 2950 each',
    'training_environment':{'python':'E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe','torch':'2.12.0+cu130','NumPy':'1.26.4','device':'cuda:0'},
    'evaluation_replay_environment':m['versions'],
    'no_SSH':True,'no_install':True,'no_new_G4':True,'EVAL_scientific_read':0,'CAL_waveform_decode':0,'raw_long_waveform_decode':0,
    'output_limit_bytes':268435456,'stage_output_bytes':'filled_by_final_seal',
    'science_fits_or_array_recomputations_during_finalization':0,
}
write('RESOURCE_RECEIPT.json', resource)
qa = {'executor_visual_review_UTC':utc(),'all_11_final_figures_actually_viewed':True,'root_all_11_initial_figures_reviewed':True,
      'layout_correction':'c2st.png shortened x-axis labels; corrected figure actually viewed by executor, other 10 unchanged',
      'original_c2st_retained':'results/figures/c2st.layout_initial.png','original_plot_receipt_retained':'PLOT_RECEIPT.before_layout.json',
      'remaining_layout_issues_observed':[],'scientific_values_changed':False,'final_figures':plots['figures'],
      'review_points':['same axes and populations in charge/PSD plot','two PSD components and missing bin8 support shown','source-bin3/4 complete parents retained','time-axis and condition medians show fixed tail variation','projection-block range explicitly not CI','negative/undefined statistical states retained']}
write('FIGURE_QA.json',qa)

final_pairs = m['C2ST']['paired_vs_corresponding_R26']['FRESH_FINAL']
c2rows = []
for view in views:
    new = m['C2ST']['results']['FRESH_FINAL'][view]
    p = final_pairs[view]
    c2rows.append([view, f(p['old_control_result']['balanced_accuracy'],3), f(new['balanced_accuracy'],3), f(p['BA_delta_first_minus_second'],3), '['+', '.join(f(x,4) for x in p['BA_delta_95CI'])+']'])
projrows=[]
for view,s in m['projection_distances']['common_support_summary'].items():
    d=s['domains'];projrows.append([view,f(d['HOMO_FINAL']['equal_mean_on_common_bins']['ORIGINAL']),f(d['FRESH_FINAL']['equal_mean_on_common_bins']['ORIGINAL']),f(d['HOMO_FINAL']['new_block_mean']),f(d['FRESH_FINAL']['new_block_mean']),f(d['TRAIN_REFERENCE']['new_block_mean'])])
widthrows=[]
for row in m['energy_conditional_width']['common_support_errors']['final_equal_updates']:
    widthrows.append(['低PSD' if row['component']==0 else '高PSD',row['field'],f(row['MAE']['HOMO_FINAL']),f(row['MAE']['FRESH_FINAL'])])
pure_rows=[]
for name in names:
    for binname,row in m['source_composition']['primary_pure_had_source_bins3_4'][name].items():
        pure_rows.append([name,binname,str(row['N']),f(row['output_PSD']['mean']),f(row['output_PSD']['sample_sd']),f(row['valley_fraction']*100,3)+'%'])
overall_rows=[]
for key in ['PSD','Qtotal_mVns','baseline_RMS_V','ACF1','rise_10_90_ns','tail_charge_fraction','tail_mean_over_peak']:
    overall_rows.append([key,f(m['domain_vs_REAL']['HOMO_FINAL']['overall'][key]['W1']),f(m['domain_vs_REAL']['FRESH_FINAL']['overall'][key]['W1'])])
jobrows = [[r['entry'],r['actual_worker_PID'],r['supervisor_PID'],f(r['elapsed_s'],4),r['actual_worker_native_exit'],r['actual_worker_RSS_peak_bytes']] for r in history]

report = f'''R28 每步刷新细时间投影改善了对四块复用方向的推广，但完整响应仍未匹配实测。相同2950更新末点的完整形状分类BA从0.990降至0.970，仍高度可分；峰宽误差改善同时伴随峰中心退化、pure-had源条件谷区占比增加和固定尾段结构残留。保留所选2250步与预指定2950步末点两份模型，不用DEV重新择优。

本轮仅改变细时间项训练投影。完整480点源输入、网络、原损失、STANDARDIZED critic、D:G=1:1、Adam、clip5及R26同区流保持固定。复用2950×32=94400条实际行，2946个同区批次加4个余数混合批次；source/REAL/bin、原始行位置、z/white/style和每步NCE位置全部对应。NCE负例仍是同一波形内的时间patch；没有跨事件NCE新增机制。独立PCG64完整种子20261009281每G步按shape480、diff479、baseline100顺序各抽128个单位投影，两域共用同组方向。2950次G调用、8850个矩阵；D、identity、原验证及评价不消费此流。每次方向摘要、首末矩阵和随机状态已保存，未保存所有矩阵的大型重复载荷。

训练6593个G4派生源父与21047条实测TRAIN，内部验证固定512源/599实测；主评价保留2000源父和2000实测DEV，不重加权、不按输出删除父记录。来源为G4总沉积驱动的既有解析波形，没有新增微观沉积时序信息。DEV的417 pure-EE、1518 pure-had、65 mixed均保留；317/1683的PDG统计是已有标签规则与R2仅22/2112的绑定CSV证据推定，非新逐事件PDG读取。历史噪声/先验暴露继续存在；人工50记录组来自同一次采集，不称独立run或外部盲测。EVAL、CAL波形、新G4、长原始波形和SSH均为0。

原固定J20选点含step0、每250步及末点，严格小于才替换。新臂初值J={t['initial_validation']['J']:.9f}，2250步所选J={t['selected_validation']['J']:.9f}，末点J={t['final_validation']['J']:.9f}；旧R26选2750步J={old_t['selected_validation']['J']:.9f}，末点J={old_t['final_validation']['J']:.9f}。原验证仍使用原固定方向，分数不代表对新方向或完整响应的全排序；bin8只有4个验证源。主干预比较是预指定2950对2950，不把末点J较差直接当作所有科学指标较差。

共同合格输出Q区间为bins1–7。下表使用同一组区间等权汇总，原方向与四块R27方向保持原尺度/分位数定义；四块已用于DEV诊断，属于已查看的开发参照，范围是方向采样参照，不是置信区间或新盲测。

{table(['细时间视图','旧末点原方向','新末点原方向','旧末点四块均值','新末点四块均值','TRAIN–DEV四块参照'],projrows)}

shape的原方向反而变差，但四块均值0.408965→0.359454，原/新方向比从1.2534到0.9615；diff与baseline的新方向差距也缩小。仍远高于TRAIN–DEV参照，不能称联合波形匹配。参照范围没有被用作物理等效容差。覆盖投影复用实际rank390和90维正交补，仅对应列明的线性映射，不是整个损失的盲区。末点总体均值残差平方范数旧0.472697、新0.440470；新臂其中约90.07%仍在覆盖空间，约9.93%在补空间，两空间均高度可分，不能把剩余问题归为只在正交补。

固定每域600分类训练/250留出及相同隔离分组，20次预定HGB均已逐fit保存。以下CI是500次配对区组bootstrap，条件于两份已拟合分类器，不包含响应训练不确定性，不作为等效检验。

{table(['视图','旧末点BA','新末点BA','新−旧','配对95%CI'],c2rows)}

prompt局部下降较明确；postpeak和tail仍接近完全可分，baseline BA小幅升高，不能由全形状下降推断每个局部改善。纯480时间u、覆盖空间和补空间均保持高BA；历史shape493还含13个标量，二者单独报告。

整体W1并列如下，单位各自不同，不跨字段比较绝对量。

{table(['观测量','旧末点W1','新末点W1'],overall_rows)}

整体PSD W1改善不能替代条件峰宽成功。16个新密度描述槽实际14次EM；bin8新selected36条/final37条低于40门槛，保留undefined，未补拟合或删缺口。两成分是密度描述，不是粒子真值。三者及四端点共同支持仍为bins1–7，等权MAE如下。

{table(['密度成分','描述量','旧末点MAE','新末点MAE'],widthrows)}

低/高两成分原宽度与区间内去趋势宽度误差均改善，低/高中心误差却分别0.005935→0.009087、0.002923→0.006993；最高区无有效新臂宽度证据。输出高Q覆盖偏少也保留在共同坐标图与完整表中，未因共同支持摘要忽略。

预指定pure-had源Q区3/4分别保留全部207/182父事件，source条件与输出Q条件不同，不能混用。完整源条件统计如下。

{table(['端点','源Q区','父数','输出PSD均值','输出PSD样本SD','冻结跨域谷区占比'],pure_rows)}

末点源区3谷区从2.415%增至10.628%，区4从0增至0.549%；这表明源条件分布变化及潜在保真风险，不能仅凭跨域谷区判定粒子内容错误。selected无这项增加但仍有中心偏差。新末点同父8次PSD SD均值pure-had约0.025647（旧0.019715），pure-EE约0.016795（旧0.014284），更大随机展宽不能独立证明正确探测涨落。源交换和固定源变噪声探针均用已修正父语义；固定源改噪声平均波形RMS旧4.360mV、新4.639mV，source-roll旧14.881mV、新15.119mV，非零敏感性不能证明物理对应。signal/electronic共享输出归一化且可能相关，保存交叉项，不能称物理独立噪声或signal条件期望。

条件波形中位尾段仍有实测中位不明显的固定起伏，postpeak/tail分类接近1，且去趋势宽度仍明显偏窄；因此下一瓶颈更可能是条件联合尾部结构与源内容约束之间的权衡，而不只是固定方向数量。一个值得检验的具体问题是：保持已刷新方向、实际源行和随机容量不变，分位数L1目标在有限同区批次下是否仍偏向窄条件分布；可用注册后的单一替代距离干预，并在相同末点同时检验新方向、宽度/中心和pure-had源条件。当前证据不证明W1必然导致收缩、不证明架构伪影或唯一成因，也不授权本报告中直接追加训练。

独立保存权重CPU重放4000条，对pre/post ADC、signal/electronic和实际源ID逐值一致，maxabs=0。训练CUDA环境与CPU评价环境分别记录，重放比较的是同CPU生成入口；未声称跨设备训练位级复现。全部11最终图执行器已实际目检，根已查看原11图；仅C2ST轴标签修短，原PNG/原绘图回执另名保留，数值不变。

{table(['入口','实际worker PID','监督PID','监测墙钟s','worker退出','实际RSS峰B'],jobrows)}

累计科学{science_s:.6f}s/1800s（含绘图和一次布局修复），训练{t['training_s']:.6f}s/900s，G/D/reader各2950，真实RSS峰2,170,609,664B、GPU allocated峰788,881,920B，均在限内。科学失败0；有一次纯控制JSON字段检查失败3cf639/native1，费用/错误原样登记，未引起科学重算。完整逐事件/身份/预测/实际行流留本地；METRICS_COMPACT仅聚合统计。最终字节数、全部文件SHA与实际封存退出见FINAL_MANIFEST、FINAL_RECEIPT及资源回执。
'''
(ROOT/'REPORT.md').write_text(report,'utf8')
reproduction = '''本轮可复现入口与产物

执行在本机E盘，不需要SSH。训练使用 E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe；评价、独立重放、绘图使用 C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe（解释器只读，所有数据/缓存/临时文件写E盘stage/cache）。线程4；真实science PID导入前握手由supervise.py/owned_worker.py保存。

实际顺序：supervise.py train_gpu.py GPU_PYTHON → supervise.py evaluate_endpoints.py CPU_PYTHON → supervise.py replay_endpoints.py CPU_PYTHON → supervise.py render_figures.py CPU_PYTHON → supervise.py fix_c2st_layout.py CPU_PYTHON。入口的.used防止在原目录重复提交；复现应复制所需源、config及输入到新E/D目录并调整输出根，不删除原.used或覆盖旧结果。原绝对期限仅本次作业有效，复现需独立注册预算，不是直接解除旧作业守护。

config/PROTOCOL/PRETRAIN_FREEZE绑定训练源码和数据位置。R23 TRAIN_PRIVATE、VALIDATION_INPUTS、FINE_CONSTANTS、CRITIC_SCALER只读复用；原R23流、R26实际重排行载荷与状态由reorder_core读取并按原行绑定重建z/white/style。R21 step2000 G/reader与R18 INITIAL_CRITIC为共同起点。完整源波形继续作为网络输入，不退回目标域独立采样。

FRESH_FINE/TRANSLATOR.pt、STYLE_READER.pt、CRITIC.pt与OPTIMIZER_SELECTED.pt是2250所选同步状态；FINAL_*与OPTIMIZER_FINAL.pt为2950末点。FRESH_PROJECTION_STREAM.json保存2950次三矩阵摘要/范数界，FRESH_PROJECTION_FIRST_PRIVATE.npz、LAST及初末RNG状态用于原流核验；validation继承固定方向，fresh训练流不被验证消耗。

ACTUAL_GENERATION_INPUTS.npz、JOINT8_INPUTS_PRIVATE、PROBE_INPUTS_CORRECTED为保存的实际推理输入。FRESH_SELECTED/generated.npz与FRESH_FINAL/generated.npz保存各2000 pre/post ADC波形、两可加分量与source IDs。REPLAY_RECEIPT.json和replay_endpoints.py.NATIVE_RECEIPT.json证明一次独立CPU主生成重放4000条已执行、四数组及源ID一致；不重复完整评价。

旧R26四端点控制、R27实际投影空间/四块方向、原分类实际分组与预测按config路径只读复用；其中这些DEV方向已查看，不能作为新盲测。完整metrics/preds/父记录/权重留E/D；Git发布用METRICS_COMPACT与必要源码、报告、图、native摘要，不上传.private/逐事件载荷。最终交付清单由新stage字节SHA封存；不重审历史3GB源或做迁移一致性测试。
'''
(ROOT/'REPRODUCTION.md').write_text(reproduction,'utf8')
write('ASSEMBLY_RECEIPT.json',{'PID':os.getpid(),'UTC':utc(),'entry':'assemble_delivery.py','pure_control_only':True,'new_fits_generation_updates':0,'scientific_inputs_decoded':0,'files':['REPORT.md','REPRODUCTION.md','METRICS_COMPACT.json','RESOURCE_RECEIPT.json','FIGURE_QA.json','NATIVE_CHAIN.json','CONTROL_FAILURE_001.json'],'native_exit_evidence':'calling exec tool receipt'})
print(json.dumps({'PID':os.getpid(),'UTC':utc(),'science_total_s':science_s,'report_bytes':(ROOT/'REPORT.md').stat().st_size,'compact_bytes':(ROOT/'METRICS_COMPACT.json').stat().st_size,'scientific_computation':0}),flush=True)
