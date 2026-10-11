"""Prepare verification, report and delivery without reading a new endpoint result."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib
R=Path(__file__).resolve().parents[1];S=R/'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001';H=R/'work/R61_PRECISION_RESOLVED_SOURCE_ORDER_001'
def convert(t):
 t=t.replace('R61_PRECISION_RESOLVED_SOURCE_ORDER_001','@CURRENT_STAGE@').replace('R59_DIRECT_DISTRIBUTION_RESPONSE_001','@HISTORICAL_STAGE@')
 t=t.replace('PRECISION_RESOLVED_RESPONSE','COVERAGE_RESPONSE').replace('PRECISION_RESOLVED_REAL_CONTROL','COVERAGE_REAL_CONTROL')
 t=t.replace('R61','R63').replace('R59','R61').replace('r61','r63')
 return t.replace('@CURRENT_STAGE@',S.name).replace('@HISTORICAL_STAGE@',H.name)
def rep(t,a,b):
 assert t.count(a)==1,(t.count(a),a);return t.replace(a,b)
def save(n,t):
 ast.parse(t,filename=n);p=R/'reviews'/n;assert not p.exists();p.write_bytes(t.encode('utf8'))
t=convert((R/'reviews/review_R61_frozen_artifacts_root.py').read_text('utf8'))
start=t.index("repair=read('repairs/AUDIT_JSON_001/REPAIR_REVIEW.json')");end=t.index("P=read('config_post.json')",start);t=t[:start]+t[end:]
t=rep(t,"assert train['stream_hashes']==C['expected_R61_stream_hashes']","assert {k:train['stream_hashes'][k]for k in C['expected_R61_stream_hashes']}==C['expected_R61_stream_hashes']")
save('review_R63_frozen_artifacts_root.py',t)
t=convert((R/'reviews/finalize_R61_root.py').read_text('utf8'))
start=t.index("failed_attempt=local('repairs/AUDIT_JSON_001/numerical_check.py.NATIVE_RECEIPT.json')");end=t.index("science=sum(n['elapsed_s']for n in all_attempts)",start)
t=t[:start]+"""failed_attempts=[];seen_attempts=set()
for p in sorted((S/'repairs').rglob('*.NATIVE_RECEIPT.json')):
    n=json.loads(p.read_text('utf8'));key=(n.get('actual_worker_PID'),n.get('entry'),n.get('start_UTC'))
    if n.get('actual_worker_native_exit')not in [None,0]and key not in seen_attempts:
        failed_attempts.append(n);seen_attempts.add(key)
all_attempts=native+failed_attempts
"""+t[end:]
t=rep(t,"'# R63：存储精度排序单因素消融'","'# R63：全窗坐标与基线时序联合分布覆盖'")
t=rep(t,"'本轮只取消 float32 PSD 保守区间重叠父对的顺序罚，保留原非零父对分母。全部 6593 源和 21047 REAL 父事件、初始化、六条实际随机流、曝光次数、冻结常数、其他六项损失及唯一 50000 更新终点与 R61 匹配。该区间是存储精度包络，不是粒子标签或实测不确定性。'",
 "'本轮在冻结 R61 七项损失之上增加全坐标经验 W1 的 RMS 池化，以及31维基线RMS/lag1–20相关性/10频带的边缘与切片联合分布损失，两项系数均1。全部6593源和21047 REAL父事件、网络、初始化、六条原随机流、曝光次数、原标准化、噪声及电荷/ADC构造保持。新增TRAIN时序常数在首个更新前按原REAL区域固定；投影方向使用独立第七条随机流。唯一50000步末点，不按评价选参。'")
t=rep(t,"一致性审计支持本固定运行的单因素比较", "一致性审计支持同一网络和原随机流下的固定目标扩展比较，不能区分两个新增项各自的因果贡献")
t=rep(t,"precision=audit['single_factor_precision_order']", "precision=audit['coverage_extension']")
t=rep(t,"'| 单因素核验 | 结果 |'", "'| 冻结设计一致性核验 | 结果 |'")
t=rep(t,"precision['trace_summary']", "precision['precision_trace_summary']")
t=rep(t,"'R60 的三个预设冲突标记均未触发的结论保持；R63 是另行预登记的存储精度假设，不把本轮结果倒填为 R60 已定位原因。'", "'R60 三个预设冲突标记均未触发的结论保持；R62 两个描述标记为另行预登记的覆盖扩展提供依据，不能据此声称识别唯一原因。'")
start=t.index("lines+=['首次数值检查在保存训练审计 JSON 时");end=t.index("gen=local('GENERATION_RECEIPT.json')",start)
t=t[:start]+"""lines+=['| 新增覆盖训练记录 | 逐更新均值 |','|---|---:|']
for k,v in precision['coverage_trace_summary'].items():lines.append('| '+k+' | '+value(v['mean'])+' |')
lines+=['','首个本机人工检查因 sqrt(1e-20) 与 Python 常量相差 1.29247e-26 V 而触发精确相等断言；修正为 float64 epsilon 范围比较后，CPU/CUDA 全部人工检查通过。损失、系数、数据、原A–F未变；原失败和源文件已保存。科学 worker 失败尝试数量：'+str(len(failed_attempts))+'，全部纳入资源回执。','']
"""+t[end:]
t=t.replace('六条随机流、逐更新损失及父事件曝光次数完整保存。','原六条随机流与R61完全匹配；独立重构第七条流的全部消费字节和末状态。22列损失、9列新增覆盖分解、精度排序分解和父事件曝光次数完整保存。')
t=rep(t,"'六个成功阶段均原生退出 0；加上已保留的失败尝试，七个科学 worker 的实际耗时合计 '", "'六个成功阶段均原生退出 0；计入已保留失败尝试，共 '+str(len(all_attempts))+' 个科学 worker 的实际耗时合计 '")
t=t.replace("'precision_order_decomposition.png']","'precision_order_decomposition.png','coverage_loss_decomposition.png']")
t=rep(t,"'failed_native_attempts_preserved':[failed_attempt]", "'failed_native_attempts_preserved':failed_attempts")
t=rep(t,"    'serialization_repair':'repairs/AUDIT_JSON_001/REPAIR_REVIEW.json','admission_failure_without_scientific_worker':'repairs/AUDIT_JSON_001/ADMISSION_MARKER_RESOLUTION.json',\n", "    'local_artificial_assertion_repair':'reviews/R63_ARTIFICIAL_ASSERTION_REPAIR_001.json',\n")
t=rep(t,"'POSTPROCESS_FILE_BINDING.json','repairs/AUDIT_JSON_001/REPAIR_REVIEW.json','repairs/AUDIT_JSON_001/ADMISSION_MARKER_RESOLUTION.json',", "'POSTPROCESS_FILE_BINDING.json','TRAIN_TEMPORAL_CONSTANTS.npz','TRAIN_TEMPORAL_CONSTANTS.json',")
t=rep(t,"'precision_audit.py',ARM+'/PRECISION_ORDER_BY_UPDATE_PRIVATE.npz'", "'coverage_audit.py',ARM+'/COVERAGE_BY_UPDATE_PRIVATE.npz',ARM+'/PRECISION_ORDER_BY_UPDATE_PRIVATE.npz'")
t=t.replace('R63 存储精度排序消融科学终点完成','R63 全窗与基线联合覆盖科学终点完成').replace('R63 已完成唯一 50000 步存储精度排序消融训练','R63 已完成唯一 50000 步完整波形覆盖扩展训练')
save('finalize_R63_root.py',t)
t=convert((R/'reviews/deliver_R61_root.py').read_text('utf8')).replace('R60','R62')
t=t.replace('r63_precision_order_final_','r63_coverage_response_final_')
t=t.replace('R63_ARTIFICIAL_RECEIPT_REPAIR_001','R63_POSTPROCESS_PREPARATION_UNUSED_001')
t=rep(t,"rootfiles+=['reviews/R63_POSTPROCESS_READY_001.json']", "rootfiles+=['reviews/R63_POSTPROCESS_READY_001.json','reviews/R63_ARTIFICIAL_ASSERTION_REPAIR_001.json']")
t=t.replace('Record precision-resolved source-order ablation and unchanged joint acceptance','Record full-waveform temporal coverage extension and unchanged joint acceptance')
t=t.replace('R63 precision-resolved source-order incremental backup.','R63 full-waveform coverage extension incremental backup.')
t=t.replace('R61-matched parent exposure counts, precision-order traces and intervals','R61-matched parent exposure counts, original precision-order traces, new coverage traces and TRAIN temporal statistics')
t=t.replace('registered single-factor audit','registered objective-extension audit').replace('prior R62/R61 backup chain','prior R62/R61 backup chain')
save('deliver_R63_root.py',t)
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'owner':'/root','scientific_results_read':0,'new_scientific_execution':0,'closeout_executed':False,'publication_executed':False,'backup_executed':False,
 'prepared_files':[{'path':'reviews/'+n,'sha256':hashlib.sha256((R/'reviews'/n).read_bytes()).hexdigest()}for n in ['review_R63_frozen_artifacts_root.py','finalize_R63_root.py','deliver_R63_root.py']]}
(R/'reviews/R63_CLOSEOUT_PREPARATION_001.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode());print(json.dumps(receipt,indent=2))
