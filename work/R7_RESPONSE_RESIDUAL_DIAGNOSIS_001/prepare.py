from __future__ import annotations
import pathlib,json,hashlib,ast,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
WORK=PROJECT/'work'
start=json.loads((ROOT/'START_RECEIPT.json').read_text())
def bind(p):
    b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
old=WORK/'R3B_ENERGY_CONDITIONAL_WIDTH_001'
sources={
 'model.py':old/'model.py',
 'evaluation.py':WORK/'R6_CHARGE_GAP_GENERALIZATION_001/evaluation.py',
 'common_math.py':WORK/'R6_CHARGE_GAP_GENERALIZATION_001/common_math.py',
 'energy_width.py':old/'energy_width.py',
 'interface_core.py':WORK/'P1K_REAL_REFERENCE_AND_REPEATABILITY_001/interface_core.py'}
source_bindings=[]
for name,p in sources.items():
    b=p.read_bytes();ast.parse(b.decode('utf-8-sig'))
    (ROOT/name).write_bytes(b);source_bindings.append(bind(p))
s=(ROOT/'model.py').read_text(encoding='utf-8')
assert s.count('torch.arange(512,dtype=torch.float32)')==1
assert s.count('clean=shape/shape.sum')==1
s=s.replace('torch.arange(512,dtype=torch.float32)','torch.arange(512*20,dtype=torch.float32)/20')
s=s.replace('clean=shape/shape.sum','shape=shape.reshape(n,512,20).mean(dim=2)\n        clean=shape/shape.sum')
(ROOT/'model_block.py').write_text(s,encoding='utf-8')
ast.parse(s)
oc=json.loads((old/'config.json').read_text())
real=pathlib.Path(oc['REAL_prefix_dir']);ident=pathlib.Path(oc['REAL_identity_dir'])
files={
 'model':old/'model.pt','generation_inputs':old/'generation_inputs.npz','generation_rng':old/'GENERATION_RANDOM_STATE.pt','generated':old/'generated.npz',
 'REAL_V':real/'window_V_f32.bin','REAL_QC':real/'QC_reason_u16.bin','REAL_geometry':real/'geometry_i32.bin','REAL_normalization':real/'normalization_f32.bin',
 'REAL_groups':ident/'selection_group.bin','REAL_original_index':ident/'selection_original_index.bin','REAL_roles':ident/'selection_role.bin',
 'CAL_reference':real/'STATISTICS.json','TRAIN_roles':pathlib.Path(oc['R2_roles']),
 'SIM_identity':pathlib.Path(oc['evaluation']['input_npz']['SIM_METADATA']),
 'width_definition':old/'results/DIAGNOSTIC_DEFINITION.json'}
inputs={k:bind(p) for k,p in files.items()}
views=['full_voltage','full_historical_shape','baseline_voltage','baseline_local','prompt_voltage','prompt_local','postpeak_voltage','postpeak_local','tail_voltage','tail_local','no_prompt_voltage','no_prompt_local']
transforms=['fractional_align','coarse_ADC4','smooth_sigma1','align_then_smooth']
contrasts=['GEN_vs_REAL','REAL_TRAIN_vs_DEV','positive_vs_REAL']
matrix=[{'operator':'native','view':v,'contrast':c} for v in views for c in contrasts]
matrix += [{'operator':t,'view':'full_historical_shape','contrast':c} for t in transforms for c in contrasts]
matrix += [{'operator':t,'view':v,'contrast':'GEN_vs_REAL'} for t in ['ADC_off','signal_block_mean20'] for v in ['full_voltage','full_historical_shape']]
assert len(matrix)==52
cfg={**start,'seed':202610087,'bootstrap_seed':2026100871,'bootstrap_repeats':500,'positive_seed':202610081,
 'numeric_wall_limit_s':1800,'threads':4,'input_files':inputs,'source_bindings':source_bindings,'matrix':matrix,
 'C2ST':{k:oc['evaluation']['C2ST'][k] for k in ['max_iter','learning_rate','max_leaf_nodes','min_samples_leaf','l2_regularization','early_stopping','random_state']},
 'train_each_domain':600,'holdout_each_domain':250,'HGB_max_fits':60,'generation_max_rows':12000,
 'new_fixed_inference_rows':4000,'response_updates':0,'ADC_step_V':1/16383,'EVAL_read':0,'SSH':0,'new_G4':0,
 'source_probe_failures':[{'tool':'e63d61','detail':'R1 evaluation.py nonexistent; actual evaluate.py; no scientific read'},{'tool':'dd1ad6','native_exit':1,'detail':'R2 DATA_ROLES at root nonexistent; exact config results/DATA_ROLES.json used instead; science not started'}]}
write('config.json',cfg)
protocol='''R7 固定开发诊断；不是新盲测或响应优化。所有定义在本次新数值之前冻结。

输入：固定R3B模型/主DEV2000/主latent白噪声，P1K REAL DEV2000和TRAIN3200（QC fit2592/val599），既有CAL边界/失败参考和R3B固定窄logQ区间。只读取允许的本地文件。P1K字节文件可能含CAL6800行，浮点视图限定前5200；EVAL文件和预测均不访问。SIM元数据只读取real_event_ids/sim_event_ids。所有同父变体复用R1父/人工组划分及固定SHA选序。

52个固定HGB矩阵见config：12原始视图×三比较36；四共同操作×历史shape×三比较12；两干预×两完整视图4。无超参数搜索/响应优化。HGB参数为历史100迭代、lr.06、15叶、最小叶20、L2=1、不early-stop、seed20261008。分类训练侧中位填补/标准化，holdout无估计。目标每域600分类训练、250组/父隔离留出，若某固定池不足则所有核心比较共用可行上界并报告。500次簇bootstrap条件于固定分类器，不是训练不确定性。

视图：full_voltage原始480点电压；full_historical_shape复用全局peak归一化+PSD/rise/tail/RMS比/ACF的旧完整视图。区域0:100、100:150、150:220、220:400，以及去prompt的0:100+150:480，均直接来自原始V。voltage视图保持本段V，不先用区域外baseline/peak归一化；local视图仅减本段均值/除本段总体RMS。baseline另含本段ACF1..8与非零归一化功率谱，电压视图含本段log样本RMS，local不含绝对RMS。去prompt不含PSD/rise/全局peak/跨prompt特征。数值前做区域外改值不变性断言。完整视图特意允许全局信息，不能混为局部证据。

阴性：从允许REAL TRAIN fit2592按固定身份SHA取2000，对照REAL DEV2000；角色人工组不交叠，同一次采集。同N不增加独立采集数，也不是不保证chance的物理等效界。阳性：原REAL DEV加入原先规定2%自身peak exp(-(t-120)/80) t>=120尾和1.5%peak独立Gaussian噪声(seed202610081)；同父/组留出，簇bootstrap将两个视图视为同组。阴性及阳性在每个核心视图/共同操作各有独立拟合；干预的同定义完整视图控制直接引用native控制，不重复拟合。

共同操作：50%最后主峰上升支线性交点移到120，y_new(t)=interp(t+cross50-120,y)，边缘值填充且统计支持变化；ADC4=-.5+4*Delta*round((V+.5)/(4*Delta))，clip[-.5,.5]；Gaussian sigma1ns/truncate4/nearest；先fraction align再同Gaussian。两域和两控制完全同操作。以原始QC/固定身份保持比较对象，重新报告QC/电荷/PSD/rise/峰宽变化。信息丢失引起分类下降不是修复或物理原因。宽度只用原8窄logQ定义/单初始化500更新固定EM，失败undefined，不重拟合CAL参考。

固定模型干预：ADC_off调用保存model.forward(quantize=False)；signal_block_mean20仅将解析信号核/RBF按t=k+j/20,j=0..19计算并块均值到512，在原512噪声、ADC及CFD之前归一化；保存网络权重/latent/white不变，只有派生time buffer更长。它不是完整真实20GS/s采集链，未模拟0.05ns电子噪声/带宽，不能据此唯一归因。两干预共4000×480个输出，不重复已完成主回放；同主权重/latent/white，全部保留。无新采样种子/响应权重更新。

处理链：REAL raw10000→500均值→float64 ADC14bit→float32基线前100及全峰→主峰相连整数CFD→480窗；GEN连续解析信号1ns点采样+512噪声→Torch f32 ADC→前102基线/全峰→同主峰相连整数CFD→480窗。报告点采样/块平均、前100/102、float32量化、完整500/512支持的差异，代码不等于原因。检查phase、peak index、ADC lattice residual、padding/clipping；用固定人工脉冲比较NumPy/Torch整数分支以及边界，人工工程样本不用于物理结论。

条件图及同N距离按各自native CAL Qt桶，包括unknown，不假造粒子真值/q4参考。baseline RMS/ACF/非DC谱、phase、rise、局部二阶差分/相关给出TRAIN fit/val/DEV、GEN及每区同N REAL TRAIN/DEV参照。条件分组使用完整电荷只用于图/摘要，不带入区域分类器。次轮只建议一个优先修正配方和一个机制消融，本轮不执行。

资源：一次一个科学子进程/CPU4，累计数值1800s/90min阶段/RSS4GiB/产物512MiB。所有缓存E/D；失败费用保留，常规工程修复不改科学定义。旧阶段、根控制/Git不可写。
'''
(ROOT/'PROTOCOL.md').write_text(protocol,encoding='utf-8')
write('PREPARATION_RECEIPT.json',{'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_files':inputs,'source_bindings':source_bindings,'HGB_matrix_count':52,'response_updates':0,'EVAL_read':0,'numeric_started':False,'config_SHA256':bind(ROOT/'config.json')['sha256'],'protocol_SHA256':bind(ROOT/'PROTOCOL.md')['sha256']})
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'prepared','HGB_fits':52,'new_fixed_inference_rows':4000,'config_SHA256':bind(ROOT/'config.json')['sha256']}),flush=True)
