from __future__ import annotations
import pathlib,json,hashlib,datetime,os,sys,importlib.metadata
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def f(v):return 'undefined' if v is None else f'{v:.6g}'
m=read(ROOT/'metrics.json');w=read(ROOT/'results/ENERGY_WIDTH_METRICS.json');roles=read(ROOT/'results/DATA_ROLES.json');history=read(ROOT/'RUN_HISTORY.json');ready=read(ROOT/'READINESS.json')
names=['SIM','GEN_W0','R2','R3','R3B','REAL'];compact={'stage':'R4','EVAL_QC':m['EVAL_QC'],'main_N':2000,'domains':{},'C2ST':m['C2ST'],'joint_sliced_W1':m['joint_sliced_W1'],'REAL_REAL_summary':m['REAL_REAL_summary'],'REAL_REAL_joint':m['REAL_REAL_joint'],'width':{},'response_refit':0}
for name in names:
    d=m['domains'][name]
    compact['domains'][name]={'QC':d['QC_valid'],'counts':[d['regions']['q'+str(i+1)]['QC_n'] for i in range(4)],'FoM':d['regions']['overall']['FoM'],'scalar_summary':d['regions']['overall']['features'],'distances':m['domain_vs_REAL'].get(name)}
    compact['width'][name]=[{'bin':s['bin']+1,'N':s['N'],'valid':s['valid'],'failure_reasons':s['failure_reasons'],
                            'components':[{k:c[k] for k in ['mean','sigma_PSD','detrended_sigma_PSD','width_defined','effective_N_times_weight']} for c in s['components']]} for s in w['main'][name]['slices']]
write('METRICS_COMPACT.json',compact)
lines=['R4冻结留出响应验证已完成。响应新拟合为0，SIM、W0、R2、R3、R3B全部保留。R3B在留出实测上仍有明显边际改善，但完整响应可被独立评价分类器区分；本轮不据EVAL选择下一配方。',
       '',f'实际开始2026-10-08 11:22:42.929665 UTC，首工具b8883f/PID61868/native0；硬截止12:52:42.929665 UTC。FROZEN_METHOD SHA256 `{ready["FROZEN_METHOD_SHA256"]}` 与R4/R5/R6任务在首次EVAL解码前绑定。READINESS生成工具b70bd0/PID58648/native0，11:33:10.726957 UTC。',
       '', '来源与用途：P1J角色3共152个人工50记录组、7600条记录。原source_id及原始索引继承；本次QC有效7583，17条无效身份仍保留。主REAL2000按完整SHA256("R4-main-real-v1|"+source_id)排序从QC全集选取，实际覆盖152组，不按PSD、类别或输出选择。全部7583的补充CDF、统计与窄区间诊断保留。人工组不能称独立采集批次；同一历史采集的旧逐事件暴露unknown，当前拟合隔离不等于外部盲测。',
       '', '提取只使用既存原NPZ的time/waveforms两个成员：已绑定中央目录、局部头、NPY头、完整CRC和DEFLATE EOF。远端流式解压8,000,080,256B，但只数值解码选中76,000,000波形值及10,000时间值；未选波形和其他科学成员数值解码0，源filename字符串0。全容器SHA继承fddb9618…，本轮不重复3GB全量hash。压缩读取2,981,631,690B、元数据592B；实际只传回15,009,255B，未下载全库。转换复用P1K：20点f64块均值→14bit量化→500点f32基线/峰归一化→主峰CFD→480窗；物理V与归一化行均保存。',
       '', '共同指标使用自身前100点基线、fast100:150、total100:400、signed PSD=1-Qfast/Qtotal。原SIM/W0电压为SIM父峰标度代理，R2/R3/R3B为TRAIN学得的幅值；不称独立物理标定。二维sliced W1使用128个[0,π)等间隔方向，logQ/PSD尺度仅由2592个TRAIN拟合事件的均值和总体SD确定。完整unknown/重叠总体参与指标；CAL参考不重拟合，q4原失败继续unknown/FoM undefined。',
       '', '|固定域|PSD W1|PSD KS|Q W1 mV ns|rise W1 ns|二维sliced W1|全局固定切点FoM|',
       '|---|---:|---:|---:|---:|---:|---:|']
for name in names:
    dist=m['domain_vs_REAL'].get(name,{}).get('overall',{})
    lines.append('|'+name+'|'+ '|'.join(f(v) for v in [dist.get('PSD',{}).get('W1'),dist.get('PSD',{}).get('KS'),dist.get('Qtotal_mVns',{}).get('W1'),dist.get('rise_10_90_ns',{}).get('W1'),m['joint_sliced_W1'].get(name,{}).get('W1'),m['domains'][name]['regions']['overall']['FoM'].get('FoM')])+'|')
lines.extend(['','R3B主分区覆盖313/665/770/252，实测487/512/475/526，低电荷与高电荷仍不足。R3B PSD总体W1=0.013729，比原SIM/W0明显小，但当前留出R3的0.008206更小；这不触发候选删除或下一轮选配方。R3B电荷W1=945.34 mV ns和高端不足保留；rise与尾电荷等边际更近，不能由这些指标推出完整分布通过。',
              '', '实测–实测参照每次按完整固定salt SHA对152个组排序为互斥76/76组，再各按事件SHA取2000 QC。64次确为2000对2000；各分区数量不强制相同并逐次保存。PSD W1中位0.003135，5–95% 0.001902–0.006132；KS中位0.02725，5–95% 0.017575–0.03735。二维sliced W1中位0.036285，5–95% 0.024972–0.058952。R3B二维0.199078明显超出这一描述范围；参照仅同采集重复抽样，不是物理等效阈值、外部置信区间或成功线。',
              '', '|C2ST|形状BA（条件组bootstrap95%）|形状+幅值BA（95%）|prompt块置换下降（两视图）|',
              '|---|---|---|---|'])
for name,a in m['C2ST'].items():
    values=[]
    for view in ['shape_only','shape_plus_amplitude_proxy']:
        r=a[view];ci=r['heldout_CI']['balanced_accuracy_95CI'];values.append(f'{r["balanced_accuracy"]:.3f} [{ci[0]:.3f},{ci[1]:.3f}]')
    drops=[a[v]['heldout_permutation_block_accuracy_drop']['wave_prompt_100_150'] for v in ['shape_only','shape_plus_amplitude_proxy']]
    lines.append('|'+name+'|'+'|'.join(values)+'|'+','.join(f(x) for x in drops)+'|')
lines.extend(['', '每域600训练/250留出，原HGB配方和SHA源父/人工组隔离保持；训练中位数填补仅见评价训练分区。7比较×2视图=14个固定评价分类器，1000次条件组bootstrap。REAL–REAL用RR第0次的预定组半样本；扰动阳性对照复用原2%峰尾指数+1.5%峰白噪声、seed202610081。阳性别名与原实测的父组共用train/test和bootstrap群；不能靠重复视图跨分区。R3B BA0.988而REAL–REAL0.504/0.528，持续支持完整响应失败；prompt置换是相关块诊断，不是机制因果证据。',
              '', '电荷条件宽度使用R3B TRAIN冻结8个窄logQ区间、单初始化EM(max500/参数差1e-8/方差floor1e-8/有效N*w≥20/分离≥1)。均值升序的两个密度成分不是粒子真值；责任权重做区间内PSD~logQ去趋势。下面数字为原sigma及去趋势sigma；任一不合格留缺口，不改门槛或重启。',
              '', '|窄区间|REAL/R3/R3B N|REAL低/高原sigma|R3B低/高原sigma|REAL低/高去趋势sigma|R3B低/高去趋势sigma|',
              '|---|---|---|---|---|---|'])
def widths(name,i,field):
    c=w['main'][name]['slices'][i]['components']
    return '/'.join(f(a[field]) for a in c) if len(c)==2 else 'undefined'
for i in range(8):
    counts='/'.join(str(int(w['main'][n]['slices'][i]['N'])) for n in ['REAL','R3','R3B'])
    lines.append(f'|{i+1}|{counts}|{widths("REAL",i,"sigma_PSD")}|{widths("R3B",i,"sigma_PSD")}|{widths("REAL",i,"detrended_sigma_PSD")}|{widths("R3B",i,"detrended_sigma_PSD")}|')
lines.extend(['', 'R3B中段高PSD密度成分的收窄较原R3更接近实测，但首区峰位及宽度明显偏离，低PSD高区重新变宽仍保留。REAL最高第8区和R3B第8区本次单EM均未收敛，宽度undefined；原参数和直方图仍在JSON。主REAL的200次人工组bootstrap按原规则重拟合，只用于描述曲线不确定性，不用于改参考或选响应。全部7583的独立补充密度结果亦保留。没有事后删区间再做不同集合平均的优越性宣称，没有拟造1/sqrt(E)法律。',
              '', '波形图同时保留原V及自身峰归一化均值/中位/10–90%范围。R3B固定时间振铃相比W0消除，但前沿/峰后细节仍支持分类；基线ACF1 W1=0.061942，生成正lag1中心与实测负中心仍不同。noise RMS总体接近不等于全部尾部/相关谱相等。所有2000父事件组成保持原样，source_label是旧2112/neutron、else非2112桶，不能把gamma桶全称PDG22，也不能把CCM高置信计数当粒子真值。',
              '', '实际作业：提取ed9698→317edc，监督52288/SSH42460/远端214704，SSH native0，20.0542091s，远端peak RSS109,867,008B（封装前自测，最终序列化峰值未单测）。评价9829c6→b41b3d，监督7552/worker41800，native0，102.9007984s，本地监测peak RSS686,555,136B，探针失败0。累计读取/数值监督122.9550092s/1800；4线程，全部E/D输出，512MiB上限内。无失败或重试；无害X11提示原样保留。READY准备0.9926632s另记，不冒充科学。配置继承的旧no_SSH/EVAL_waveform_or_feature_read字段属于旧评价描述，当前顶层SSH=1/EVAL_read=7600及原生回执为准；没有隐去真实留出读取。',
              '', '交付metrics.json及METRICS_COMPACT.json，results/ENERGY_WIDTH_METRICS.json、REAL_REAL_REFERENCE.json、原始EVAL_EXTRACT.npz、固定身份/主选择、评价预测、10图及必要源码。10幅PNG均实际视觉检查，轴/图例完整，没有图外裁切；密集SIM/W0尖峰和失败缺口保持。当前方法仍不能称完整响应相似或可靠粒子识别，R5固定五臂/44拟合和R6固定两臂课程将按此前冻结顺序继续，EVAL不用于修改它们。'])
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
qa={'actual_image_views':['charge_psd','conditional_waveforms','energy_peak_width','energy_psd_slices','c2st','fluctuations','response_features','conditional_psd','parent_response','all_eval_supplement'],
    'axes_legends_not_clipped':True,'unknown_and_unqualified_preserved':True,'response_refit':0,'scientific_retry':0,'numeric_native_exits':[r.get('SSH_exit',r.get('worker_native_exit')) for r in history]}
write('DELIVERY_QA.json',qa)
write('RESOURCE_RECEIPT.json',{'history':history,'numeric_cumulative_wall_s':sum(r['wall_s'] for r in history),'numeric_cap_s':1800,
                             'prepare_native':{'tool':'b70bd0','PID':58648,'exit':0,'wall_s':.9926632},
                             'supervisor_runtime_includes_imports_and_graphs':True,'threads':4,'RSS_cap':4294967296,
                             'output_cap':536870912,'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','matplotlib']},
                             'failure_and_retries':0,'remote_state':'confirmed native0 receipt and matching transfer SHA'})
members=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p.name not in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']:
        b=p.read_bytes();members.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
write('FINAL_MANIFEST.json',{'stage':'R4','members':members,'self_reference_exclusions':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'only_byte_hash_for_scientific_payloads':True})
manifest=ROOT/'FINAL_MANIFEST.json';mb=manifest.read_bytes()
receipt={'stage':'R4','status':'sealed','seal_PID':os.getpid(),'numeric_success':True,'scientific_claim':'improved marginals;complete response remains distinguishable',
         'manifest_bytes':len(mb),'manifest_sha256':hashlib.sha256(mb).hexdigest(),'members_excluding_two_seal_files':len(members),
         'report_sha256':hashlib.sha256((ROOT/'REPORT.md').read_bytes()).hexdigest(),'metrics_sha256':hashlib.sha256((ROOT/'metrics.json').read_bytes()).hexdigest(),
         'EVAL_current_response_fit':0,'next':'predeclared R5; no further R4 changes','last_write_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
write('FINAL_RECEIPT.json',receipt)
total=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());count=sum(1 for p in ROOT.rglob('*') if p.is_file())
print(json.dumps({'seal_PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'last_file_write_UTC':datetime.datetime.fromtimestamp((ROOT/'FINAL_RECEIPT.json').stat().st_mtime,datetime.timezone.utc).isoformat(),'files':count,'bytes':total,**receipt}),flush=True)
