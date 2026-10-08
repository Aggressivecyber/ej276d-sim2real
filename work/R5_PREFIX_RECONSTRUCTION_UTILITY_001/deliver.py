from __future__ import annotations
import pathlib,json,hashlib,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8'))
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def binding(p):
    b=p.read_bytes();return {'path':str(p.relative_to(ROOT)).replace('\\','/'),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
start=read('START_RECEIPT.json');m=read('metrics.json');history=read('RUN_HISTORY.json');replay=read('REPLAY_CHECK.json');ident=read('results/TASK_TRAIN_IDENTITIES.json')
assert m['fit_count']==44 and replay['status']=='pass' and history[-1]['worker_native_exit']==0
rows=['|任务实测|臂|模型|PSD MAE|PSD RMSE|Q MAE (mV ns)|logQ RMSE|翻侧率|','|---:|---|---|---:|---:|---:|---:|---:|']
for k,v in m['results'].items():
    b,a,model=k.split('__');o=v['overall'];rows.append(f"|{b[1:]}|{a}|{model}|{o['PSD']['MAE']:.6f}|{o['PSD']['RMSE']:.6f}|{o['Qtotal_mVns']['MAE']:.3f}|{o['logQ']['RMSE']:.6f}|{v['cut']['side_disagreement_rate']:.4f}|")
paired=['|预算/模型/混合臂|PSD MAE 差（负值改善）|95%人工组区间|','|---|---:|---|']
for k,v in m['paired_group_bootstrap'].items():
    d=v['targets']['PSD'];lo,hi=d['MAE_difference_95CI'];paired.append(f"|{k}|{d['paired_MAE_difference']:.6f}|[{lo:.6f}, {hi:.6f}]|")
report=f'''R5 完成了预定的已对齐保存窗前缀重建比较。合成增强对部分固定少样本估计器有帮助，但尚无稳健的采集节省或粒子判别证据；响应分布变得更近并不自动转化为本任务的更低误差。

256 条实测时，Ridge 的 PSD MAE 从 0.069210 降至原 SIM 增强的 0.057754；R3/R3B 为 0.059258/0.061745。1024 条时，Ridge 的混合 R3B 与纯实测几乎相同（0.055947/0.055960），完整 2592 条实测为 0.055386。HGB 的 1024 条 R3/R3B 增强结果为 0.058869/0.059307，相比纯实测 0.059864 只有小改善；完整实测为 0.059595。所有候选与两模型均保留，没有根据 EVAL 选择赢家。

输入、训练与评价

任务在 R4 打开 EVAL 前登记，任务 SHA 为 {start['task_sha256']}。本轮使用原 REAL TRAIN 的 2592 条内部拟合事件，599 条整人工组隔离的内部验证事件只检查、不调参。嵌套实测预算按 source_id 完整 SHA256 排序，人工组数为 {json.dumps(ident['subset_group_counts'],ensure_ascii=False)}。2592 个全实测任务对照同时运行。G4 TRAIN 的 8192 条来源绑定缓存按 R5-synth-v1|parent_id 全 SHA 排序选固定 2000 父事件；四个合成臂共用它们，全部 2000 条分别通过共同 QC，没有筛高置信样本。

原 SIM 使用已保存 active_prepare_sim_data 的核函数、原固定 Welch 谱和 0.6258014992938228 mV 全 512 点 RMS 归一噪声、14 bit 量化、原 float32 归一与 CFD/480 点裁窗。它是原发生器在新固定 TRAIN 父事件上的实现，主种子 2026100852；不是把无噪 G4 cache 当作含噪 SIM。原历史种子 42222 与这次新固定实现不同。R2/R3/R3B 权重与执行源已冻结且仅推理，无响应拟合，无 SSH、新 G4 或数据扩样。

每个输入只用保存窗 0:150：前 100 点基线与样本 RMS，100:150 扣基线/前缀峰归一的 50 点，加 log 峰和 log RMS，共 52 特征。数值峰/RMS 下限为 1e-8/1e-12 V，所有有效样本均未触发下限。标签来自相应全积分波形的 signed PSD 和 log(Qtotal)。每个预算共用仅该实测子集的均值与总体标准差。纯实测权重总和 1；混合臂实测和合成各 0.5。固定 Ridge alpha=.001；固定 HGB 150 步、学习率 .05、15 叶、最小 20、L2=1、无提前停止。权重不重缩放，HGB 正则随此冻结权重方案解释，未搜索其他模型或超参。

全部 44 个估计器、配置、训练身份与任务 SHA 已在 MODELS_FROZEN_BEFORE_EVAL 中落盘后，才对 R4 缓存的全部 7583 QC 有效 EVAL 事件、152 人工组作一次预测。没有将测试值用作训练、尺度、超参或候选选择。

完整结果

{chr(10).join(rows)}

完整各电荷分区的 MAE/RMSE/中位绝对误差/95%绝对误差与 logQ RMSE 在 metrics.json。所有 unknown 都保留。CAL q4 参考失败保持原样；本任务按真实电荷分区作回归误差描述，未在 q4 补类别参考。

{chr(10).join(paired)}

以上为固定估计器下 1000 次人工组配对 bootstrap（2026100853），没有包含响应训练不确定性、模型重拟合不确定性或跨采集不确定性；这些描述区间也没有作多重比较校正。1024 Ridge 混合 R2 的 PSD 误差反而变大；原 SIM 在 HGB 的 PSD 指标也未改善。合成辅助不能概括成所有模型、预算、目标共同成功。

容差与使用局限

沿用 CAL 全局切点 {m['CAL_global_cut']:.15f}。整体翻侧约 31%，近切点区间约一半翻侧。此处是与全积分实测切点判定的差异，不是 n/gamma 真值错误率。保存的完整容许绝对误差–覆盖曲线可用于具体用途，未规定任意统一 1%/25%成功线。全部估计器满足 |预测误差|<|truePSD-cut| 时不翻侧的几何保证，违例数均为 0。大于保证范围不等同必定翻侧。

保存窗曾经用完整波形 CFD 对齐，因此这是“已对齐保存窗的前缀重建”，尚未验证因果在线触发、硬件延迟或 FPGA；任何在线时间收益需要另行测量。前缀内 full charge/PSD 可辨识度有限，当前误差不能由相近 FoM 消除。

实测与历史成本

学习生成器已使用 3191 QC REAL TRAIN，并使用 1594 CAL 与 2000 DEV 开发参考。本轮任务仅从该已采集 TRAIN 内用 256/1024 条拟合，不能称只采集或仅暴露了这么多。原 SIM 噪声标度与固定 Welch 谱历史用了 79793 条实测，报告没有抹去此先验或将其计为免费物理输入。当前真实记录都来自一次采集，人工每 50 条组不是独立采集批次；没有粒子真值标签节省证据，也没有重算这些历史集合的独立采集时间/成本并集。

运行与复现

首个原生准备命令为 133091/PID44580、{start['start_UTC']}；准备 native3c9419/PID9960/exit0，兼容修复准备 PID59540/exit0。首次数值原生19d973→a3f033，监督43432/工作45124，exit1，用时21.942562s，默认 weights_only 拒绝旧自有检查点内 NumPy 标量；发生在任何任务拟合与 EVAL 数值打开前。FAILURE 与完整日志保留。随后在固定 SHA/大小核对后兼容读取自有检查点，未改科学定义。

主作业149c04→214f56，监督29044/工作40340，native0，19.199276s，44 fits。保存后独立进程 aff95d，监督23840/工作6392，native0，3.049185s，44 保存估计器/64 固定前缀/2816 数值逐位相同、maxabs=0；无重复训练或全套评价。累计数值 {sum(r['wall_s'] for r in history):.6f}s/1800，测得峰 RSS {max(r['peak_RSS_observed'] or 0 for r in history)} B/4 GiB，内存探针失败 0。全阶段本地 CPU4/数据缓存 E，无存活远端作业或新 SSH。所有原生失败费用保留。

五张关键 PNG 已由执行器实际查看，坐标/图例/所有臂/完整容差覆盖与分区均可读，无裁切。原模型、输入与 R4/R3B 封存文件未修改。主要交付为 task_models.joblib、synthetic_waveforms.npz、generation_inputs/RNG、predictions.npz、metrics、五图、源码和原生回执；下一轮依照已冻结 R6 执行连续电荷代理区间剔除实验，不按本结果改配方。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
write('VISUAL_QA.json',{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'five_actual_viewed':['prefix_errors','tolerance_coverage','cut_margin_flip','regional_errors','paired_bootstrap'],'readable_axes_and_legends':True,'no_clipping':True,'performed_via':'view_image tool','scientific_recalculation':0})
write('RESOURCE_RECEIPT.json',{'numeric_history':history,'numeric_cumulative_wall_s':sum(r['wall_s'] for r in history),'numeric_limit_s':1800,'RSS_measured_peak_bytes':max(r['peak_RSS_observed'] or 0 for r in history),'RSS_limit_bytes':4294967296,'threads':4,'planned_completed_task_fits':44,'response_training':0,'SSH':0,'one_ordinary_checkpoint_compatibility_failure_preserved':True,'data_or_cache_C_writes':0,'dependency_loading_IO_not_separately_instrumented':True,'stage_deadline_UTC':start['deadline_UTC']})
files=[binding(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']]
write('FINAL_MANIFEST.json',{'schema':'relative byte SHA256,ordinary final seal','files':files,'members':len(files),'bytes':sum(r['bytes'] for r in files),'self_exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'scientific_payloads_hashed_as_bytes_only_during_seal':True})
last=datetime.datetime.now(datetime.timezone.utc).isoformat();deadline=datetime.datetime.fromisoformat(start['deadline_UTC']);assert datetime.datetime.now(datetime.timezone.utc)<deadline
receipt={'status':'sealed_complete','seal_PID':os.getpid(),'seal_UTC':last,'deadline_UTC':start['deadline_UTC'],'report':binding(ROOT/'REPORT.md'),'metrics':binding(ROOT/'metrics.json'),'manifest':binding(ROOT/'FINAL_MANIFEST.json'),'numeric_cumulative_wall_s':sum(r['wall_s'] for r in history),'files_before_receipt':len(files)+1,'bytes_before_receipt':sum(r['bytes'] for r in files)+(ROOT/'FINAL_MANIFEST.json').stat().st_size,'native_seal_exit':'external command will report actual exit','further_stage_writes':False}
write('FINAL_RECEIPT.json',receipt)
print(json.dumps({'PID':os.getpid(),'status':'sealed_complete','last_file_write_UTC':datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat(),'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest':receipt['manifest'],'report':receipt['report'],'total_files':len(files)+2,'total_bytes':receipt['bytes_before_receipt']+(ROOT/'FINAL_RECEIPT.json').stat().st_size},ensure_ascii=False),flush=True)
