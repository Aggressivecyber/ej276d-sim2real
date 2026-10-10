"""Write one local R48 report/accounting/manifest/seal from saved results."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,hashlib

R=Path(__file__).resolve().parent
def utc():return datetime.now(timezone.utc).isoformat()
def read(n,default=None):
    p=R/n
    return json.loads(p.read_text('utf8'))if p.exists()else default
def write(n,x):(R/n).write_bytes((json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8'))
def fmt(x):return 'undefined'if x is None else f'{x:.8g}'
C=read('config.json');arms=C['arms'];num=read('TRAIN_SOLVER_CHECK.json');tc=read('TRAINING_COMPACT.json')
hist=read('REMOTE_RUN_HISTORY.json',[])+read('LOCAL_DISPLAY_HISTORY.json',[]);qa=read('FIGURE_QA.json')
assert num is not None and qa is not None and qa['all_final_images_actually_opened']
assert all(r['actual_worker_native_exit']==0 for r in hist)
assert len(hist)==6 and all(r['handshake_before_imports']for r in hist)
assert len(qa['figures'])==4 and all(f['actually_opened']for f in qa['figures'])
for a in arms:
    for n in ['FINAL_FLOW.pt','FINAL_OPTIMIZER.pt','FINAL_TORCH_RNG.pt',
              'INITIAL_RANDOM_STATES.json','FINAL_RANDOM_STATES.json','TRAINING_RECEIPT.json']:
        assert(R/a/n).is_file(),(a,n)
full=bool(num['both_pass']);gate=read('GATE_METRICS.json');accept=read('ACCEPTANCE_A_C.json')
metrics=read('METRICS_COMPACT.json');cal=read('NOISE_CALIBRATION.json')
if not full:
    assert not(R/'GENERATION_RECEIPT.json').exists(), 'DEV must not be opened after TRAIN gate failure'
    gate={'primary':arms[1],'decision':'close_fixed_R48_recipe_at_TRAIN_numerical_gate',
          'first_five_scientific_gates':'not_evaluated','numerical_prerequisite_pass':False,
          'models':num['models'],'new_DEV_generation_HGB_EM_replay':0,'control_not_fallback':True,
          'gate_status':{f'gate{i}':'not_evaluated' for i in range(1,6)},
          'gate6_numerical_prerequisite':'fail',
          'failure_scope':'fixed Heun64 numerical recipe; noise completion scientific efficacy not evaluated'}
    accept={'A':'not_evaluated','B':'not_evaluated','C':'numerical_prerequisite_failed',
            'D_E':'not_registered_or_evaluated','primary_A_C_pass':False}
    metrics={'status':'closed_at_TRAIN_gate','training':tc['training'],'TRAIN_solver':num,
             'calibration':cal,'gate':gate,'acceptance_A_C':accept,
             'counters':{'new_HGB':0,'new_EM':0,'new_DEV_main_rows':0,'independent_main_replay_rows':0},
             'EVAL_read':0,'CAL_waveform_decode':0}
    write('GATE_METRICS.json',gate);write('ACCEPTANCE_A_C.json',accept);write('METRICS_COMPACT.json',metrics)
else:
    assert gate is not None and accept is not None and metrics is not None
    assert read('REPLAY_RECEIPT.json')['pass']
science=sum(r['elapsed_s']for r in hist)
train=sum(r['elapsed_s']for r in hist if r['entry']in ['train_control.py','train_completion.py'])
failures=[{'name':str(p.relative_to(R)),'record':read(str(p.relative_to(R)))}for p in R.rglob('*FAILURE*.json')]
gaps=[{'entry':r['entry'],'events':r.get('memory_probe_failure_events',[]),'cause':'unknown'}for r in hist if r.get('memory_probe_failures',0)]
account={'PID':os.getpid(),'UTC':utc(),'scientific_wall_s':science,'science_limit_s':14400,
         'training_guard_wall_s':train,'training_total_limit_s':10800,'per_arm_train_limit_s':5400,
         'native_scientific_jobs':hist,'memory_query_gaps':gaps,'preserved_failures':failures,
         'maximum_actual_worker_RSS_bytes':max(r.get('actual_worker_RSS_peak_bytes',0)for r in hist),
         'maximum_job_tree_RSS_bytes':max(r.get('job_tree_RSS_peak_bytes',0)for r in hist),
         'GPU_allocated_and_reserved_bytes_by_arm':{a:{k:tc['training'][a][k]for k in ['GPU_allocated_peak_bytes','GPU_reserved_peak_bytes']}for a in arms},
         'completed_new_training_runs':2,'updates_per_arm':50000,'EVAL_read':0,'CAL_waveform_decode':0,
         'original_loaded_SOURCE_REAL':[6593,21047],'actually_consumed_unique_SOURCE_REAL':[3752,18739],
         'unique_parent_population_increased':False,'same_acquisition_DEV_is_not_blind_test':True,
         'source_only_control_operations_not_scientific_jobs':True,
         'local_remote_stage_cap_bytes':167772160,'remote_stat_inventory':read('REMOTE_OUTPUT_INVENTORY.json'),
         'normal_transfer_receipts':['TRANSFER_HISTORY.json','OUTPUT_RETURN_RECEIPT.json'],
         'scientific_inference_and_fit_counts':{'completed_training_updates':100000,
             'TRAIN_solver_field_rows':98304,'DEV_neural_inference':0,'new_HGB':0,'new_EM':0,'main_replay':0},
         'configuration_metadata_correction':'config.json inherited new_response_training_runs=0 is inactive historical metadata; two actual native receipts and config_cpu.json record 2',
         'migration_equivalence_tests':0,'shared_environment_installations':0,'root_Git_D_delivery':'pending_parent'}
assert science<=14400 and train<=10800
write('RESOURCE_RECEIPT.json',account)
lines=['# R48：固定guide与REAL身份路径的噪声规则对照','',
       '本轮固定主候选为GUIDED_NOISE_COMPLETION，GUIDED_FIXED_NOISE_CONTROL仅作同时训练对照。'+
       ('两臂通过TRAIN数值门并完成既定开发评价。'if full else'固定TRAIN数值门失败，按协议在DEV之前关闭；没有完整分布有效性结论。'),'',
       '两臂均从零完成50000次更新，共享guide、REAL身份轨迹、谱核、初始化、原统计耦合和四条PCG64流。唯一臂间区别是初态创新幅度：固定tau2与max(tau2-v(x),0)。SOURCE及IDENTITY损失各0.5；不选中间checkpoint。相对R46同时改变多个框架部分，历史差异不能全归因于噪声规则。','',
       '| 臂 | SOURCE首/末MSE | IDENTITY首/末MSE | 训练本体秒 | TRAIN RMS中位/p95（LSB） | 数值通过 |',
       '|---|---:|---:|---:|---:|---|']
for a in arms:
    r=tc['training'][a];f=r['initial_trace'];l=r['final_trace'];n=num['models'][a]
    lines.append(f"| {a} | {fmt(f['source_loss'])} / {fmt(l['source_loss'])} | {fmt(f['identity_loss'])} / {fmt(l['identity_loss'])} | {fmt(r['elapsed_s'])} | {fmt(n['RMS_median_LSB'])} / {fmt(n['RMS_p95_LSB'])} | {n['pass']} |")
lines+=['','噪声校准只使用18739条实际unique matched_target REAL TRAIN；实际SOURCE唯一父为3752。容器加载人数6593/21047与实际使用人数分开。新增状态曝光不增加独立父人口。100点居中样本方差分母99；tau2固定TRAIN低5%，两臂共同K0和各区投影后kappa归一。该规则是统计噪声种子，不是物理去噪或精确频带协方差补偿。','']
if full:
    lines+=['| 主候选阶段门 | 通过 |','|---|---|']
    for k,v in gate['models'][arms[1]].items():
        if k.startswith('gate'):lines.append(f"| {k} | {v['pass']} |")
    lines+=['',f"决定：{gate['decision']}。原A–C结果："+', '.join(f"{k}={accept[arms[1]][k]['pass']}"for k in ['A','B','C'])+'。D/E未另行登记或执行。',
            '', '| 域 | shape493 BA | pure u480 BA | Q–PSD BA | PSD总体W1 |', '|---|---:|---:|---:|---:|']
    for a in ['BASE',*arms,'GUIDED_FIXED_NOISE_REAL_CONTROL','GUIDED_COMPLETION_REAL_CONTROL']:
        sc=metrics['C2ST']['results'][a];dist=metrics['overall_domain_vs_REAL'].get(a,{})
        lines.append(f"| {a} | {fmt(sc['full_historical_shape']['balanced_accuracy'])} | {fmt(sc['normalized_u480']['balanced_accuracy'])} | {fmt(sc['Q_PSD_2D']['balanced_accuracy'])} | {fmt(dist.get('PSD',{}).get('W1'))} |")
    lines+=['','完整条件峰位/原宽度按原必需区保留失败与undefined，不删失败区后重算有利平均。源组谷区是冻结代理比例，不是粒子误分类率；actual未量化NN source-Q与历史ADC参考Q分列。SOURCE/REAL两臂primary−fixed的配对BA和500次区组95%CI见METRICS_COMPACT与paired_noise_rule_C2ST.png。同处理REAL的原三视图门保留，原始REAL恒等误差只作描述。','',
            '独立CPU完整链重放8000条通过；各端点均为chunk64/Heun64、float64坐标与float32网络，末端仅一次ADC。重放证明保存模型的计算复现，不证明真实粒子映射。']
else:
    lines+=['五项科学分布门及A/B均未评价；数值前提失败。禁止DEV、HGB、EM和8000条主输出重放。没有尝试更高步数或替代对照赢家。', '',
            '这一失败关闭的是固定Heun64数值配方，不能据此判定噪声补全机制无效。两臂IDENTITY速度目标会随创新幅度变化，较低的训练MSE不能单独证明REAL输出稳定；该输出科学门未执行。', '',
            '两臂中位数都低于0.25 LSB、电荷误差都低于1e-5，但p95分别为1.3327988与1.4498243 LSB，均高于1 LSB；不能因其他项通过替代联合数值门。方法名额3/3已用尽，本轮结束后不自动启动第四方法。']
lines+=['',f'科学原生守护总耗时{science:.6f}秒；训练守护合计{train:.6f}秒。实际RSS/GPU峰值与所有原生退出见RESOURCE_RECEIPT/NATIVE链。RSS查询间隙的原因保持unknown，不据接近退出时间宣称已解释。', '',
        '解释边界：两域无真实逐事件配对；身份路径仅同一REAL自身监督，没有域标签或特判回退。100点方差估计、加性/平稳假设及投影影响仍须保留。功能分量含末次float32 cast并保留交叉项，不能冒充独立物理电子噪声。已有同采集DEV用于开发诊断，不能称新盲测。原A–F完整模型目标仍须共同满足，阶段计算/重放成功不等于项目完成。', '',
        '代码、权重、原始身份与完整结果留在E/服务器新阶段；紧凑指标不包含逐事件身份或预测。图均由FIGURE_QA记录实际打开。REPRODUCTION.md给出同环境恢复入口。私有Git和D增量由根代理完成，本封存不宣称其已交付。']
(R/'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'))
exclude={'FINAL_MANIFEST.json','FINAL_RECEIPT.json'};files=[]
for p in sorted(R.rglob('*')):
    if not p.is_file()or p.relative_to(R).as_posix()in exclude:continue
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    files.append({'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
manifest={'PID':os.getpid(),'UTC':utc(),'files':files,'file_count':len(files),
          'payload_bytes':sum(x['bytes']for x in files),'self_reference_excluded':sorted(exclude),
          'hash_scope':'one local stage artifact inventory, not source/destination migration comparison'}
write('FINAL_MANIFEST.json',manifest)
now=datetime.now(timezone.utc)
assert now<=datetime.fromisoformat('2026-10-10T13:04:06.929385+00:00')
total=sum(p.stat().st_size for p in R.rglob('*')if p.is_file())
assert total+16384<=167772160
final={'PID':os.getpid(),'UTC':utc(),'status':'sealed','no_further_stage_writes':True,
       'manifest_sha256':hashlib.sha256((R/'FINAL_MANIFEST.json').read_bytes()).hexdigest(),
       'report_sha256':hashlib.sha256((R/'REPORT.md').read_bytes()).hexdigest(),
       'files_before_final_receipt':len(files)+1,'bytes_before_final_receipt':total,
       'scientific_wall_s':science,'training_runs':2,'updates_per_arm':50000,
       'decision':gate['decision'],'numerical_prerequisite_pass':full,
       'science_native_exits':[r['actual_worker_native_exit']for r in hist],
       'calling_native_tool_exit_reported_separately':True,'root_Git_D_delivery':'pending_parent'}
write('FINAL_RECEIPT.json',final)
print(json.dumps({'PID':os.getpid(),'UTC':final['UTC'],'status':'sealed',
                 'FINAL_RECEIPT_sha256':hashlib.sha256((R/'FINAL_RECEIPT.json').read_bytes()).hexdigest(),
                 'FINAL_MANIFEST_sha256':final['manifest_sha256'],
                 'final_file_count':len(files)+2,
                 'final_stage_bytes':sum(p.stat().st_size for p in R.rglob('*')if p.is_file()),
                 'scientific_wall_s':science,'no_further_stage_writes':True,
                 'actual_last_write_UTC':datetime.fromtimestamp((R/'FINAL_RECEIPT.json').stat().st_mtime,timezone.utc).isoformat()}),flush=True)
