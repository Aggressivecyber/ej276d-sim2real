from pathlib import Path
import os, json, datetime, time, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text('utf8'))
OLD=Path(CFG['source_stage']); START=time.monotonic()
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,
                     'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','savefig.facecolor':'white'})

def load(path,keys):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in keys}
def write(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
def features(a):
    # The already frozen R18-R29 signed charge interface, used only to annotate
    # the three saved pairs; this is not a new evaluator or fit.
    x=np.asarray(a,dtype='f8'); x=x-x[:,:100].mean(1,keepdims=True)
    qt=x[:,100:400].sum(1)*1000
    qfast=x[:,100:150].sum(1)*1000
    return qt,1-qfast/qt

a=load(OLD/'ACTUAL_GENERATION_INPUTS.npz',['source_waveforms_V','source_ids'])
b=load(OLD/'FRESH_FINAL/generated.npz',['waveforms_V','source_ids'])
meta=load(OLD/'DEV_SOURCE_METADATA.npz',['source_ids','composition','edep_ee_MeV','edep_hadron_MeV'])
source=a['source_waveforms_V']; output=b['waveforms_V']; ids=a['source_ids']; comp=meta['composition']
assert source.shape==output.shape==(2000,480)
assert np.array_equal(ids,b['source_ids']) and np.array_equal(ids,meta['source_ids'])
assert len(np.unique(ids))==2000 and np.all(np.isfinite(source)) and np.all(np.isfinite(output))
qsource,_=features(source)
categories=['pure_EE','pure_had','mixed']
labels={'pure_EE':'电子沉积（pure_EE）','pure_had':'强子沉积（pure_had）','mixed':'混合沉积（mixed）'}
selected=[]
for category in categories:
    ix=np.flatnonzero(comp==category)
    median=float(np.median(qsource[ix]))
    index=int(ix[np.lexsort((ix,np.abs(qsource[ix]-median)))[0]])
    selected.append({'composition':category,'population_count':int(len(ix)),
                     'source_Q_median_mVns':median,'index_0based':index,
                     'source_id':str(ids[index]),'edep_ee_MeV':float(meta['edep_ee_MeV'][index]),
                     'edep_hadron_MeV':float(meta['edep_hadron_MeV'][index])})
pick=np.array([r['index_0based'] for r in selected]); sv=source[pick]; ov=output[pick]
sq,sp=features(sv); oq,op=features(ov)
for i,r in enumerate(selected):
    r.update(input_Q_mVns=float(sq[i]),input_signed_PSD=float(sp[i]),
             output_Q_mVns=float(oq[i]),output_signed_PSD=float(op[i]))

# This is a saved waveform coordinate, not an absolute acquisition timestamp.
# R28 data_support.clean_expectation crops at tp-120; the inherited grid is 1 ns
# (P1F PROTOCOL dt_ns=1 and R18 PROTOCOL's population metric definition).
sample=np.arange(480); t=(sample-120).astype('f8')
def draw(name, tail=False):
    fig,axs=plt.subplots(3,2,figsize=(12.8,10.4),sharex=True,facecolor='white')
    mask=(t>=30)&(t<=300) if tail else np.ones(480,dtype=bool)
    for i,r in enumerate(selected):
        values=np.r_[sv[i,mask],ov[i,mask]]*1000
        lo=float(values.min()); hi=float(values.max()); margin=max(.5,(hi-lo)*.07)
        for j,(v,color,psd,q) in enumerate([(sv,'#2463a6',sp,sq),(ov,'#dc7425',op,oq)]):
            ax=axs[i,j]; ax.plot(t,v[i]*1000,color=color,lw=1.55)
            ax.set_xlim(float(t[mask][0]),float(t[mask][-1])); ax.set_ylim(lo-margin,hi+margin)
            ax.grid(color='#dfe3e8',alpha=.65,lw=.6)
            if not tail: ax.axvline(0,color='#99a2aa',lw=.65,ls='--',alpha=.6)
            ax.text(.98,.95,f"源行 {r['index_0based']}  |  PSD {psd[i]:.4f}\nQ {q[i]:.1f} mV·ns",
                    transform=ax.transAxes,ha='right',va='top',fontsize=10.5,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':3})
            if j==0: ax.set_ylabel(labels[r['composition']]+'\n电压（mV）',fontsize=12)
            else: ax.set_ylabel('电压（mV）',fontsize=12)
            if i==2: ax.set_xlabel('相对对齐参考的时间（ns）',fontsize=12)
    axs[0,0].set_title('仿真波形（输入）',fontsize=18,pad=18)
    axs[0,1].set_title('重建波形（网络输出）',fontsize=18,pad=18)
    title='R28 FRESH_FINAL · 2950 步 · 同一源父事件配对'
    if tail: title+=' · 尾段放大'
    fig.suptitle(title,fontsize=14,y=.985)
    fig.subplots_adjust(left=.105,right=.975,bottom=.10,top=.91,wspace=.23,hspace=.30)
    caption='原始电压，未平滑、未独立归一化；每行左右使用相同坐标。索引120对应参考时间0。'
    if tail: caption+=' 尾段仅为辅助，完整480点见主图。'
    fig.text(.5,.025,caption,ha='center',fontsize=10,color='#414951')
    fig.savefig(ROOT/(name+'.png'),dpi=180)
    fig.savefig(ROOT/(name+'.svg'))
    plt.close(fig)
draw('simulation_to_reconstruction')
draw('simulation_to_reconstruction_tail',True)

for i,r in enumerate(selected):
    with (ROOT/(r['composition']+'_points.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['sample_index','time_relative_index120_ns','simulation_input_mV','reconstructed_post_ADC_mV'])
        w.writerows(zip(sample,t,sv[i]*1000,ov[i]*1000))
manifest=json.loads((OLD/'FINAL_MANIFEST.json').read_text('utf8'))
bindings={r['path']:r for r in manifest['files']}
checkpoint='FRESH_FINE/FINAL_TRANSLATOR.pt'
record={'preview':'R28_input_output_20261009_001','PID':os.getpid(),
        'source_file':str(OLD/'ACTUAL_GENERATION_INPUTS.npz'),'source_keys':['source_waveforms_V','source_ids'],
        'output_file':str(OLD/'FRESH_FINAL/generated.npz'),'output_keys':['waveforms_V','source_ids'],
        'metadata_file':str(OLD/'DEV_SOURCE_METADATA.npz'),
        'checkpoint_file':str(OLD/checkpoint),'checkpoint_existing_binding':bindings[checkpoint],
        'input_existing_binding':bindings['ACTUAL_GENERATION_INPUTS.npz'],
        'output_existing_binding':bindings['FRESH_FINAL/generated.npz'],
        'ID_order_equal':True,'unique_source_count':2000,'selected_rows':selected,
        'selection_rule':CFG['selection_rule'],'selection_Q_definition':'first100 baseline subtraction, signed sum100:400 ×1000 mV ns',
        'method_choice':'lowest saved complete DEV shape493 BA among the reviewed full-source R18-R29 route: R28 FRESH_FINAL BA0.970',
        'choice_is_development_display_not_final_success':True,'old_J20_selected_step_kept':2250,'display_checkpoint_step':2950,
        'time_grid':{'sample_interval_ns':1,'full_points':480,'sample_index120_reference':0,'absolute_acquisition_time_known':False,
                     'evidence':['P1F PROTOCOL.json dt_ns=1','R18 PROTOCOL.md first100/Qfast100:150/Qtotal100:400/1ns',
                                 'R28 data_support.py clean_expectation ix=tp-120+arange480']},
        'signed_PSD_definition':'1-Qfast/Qtotal with signed baseline-subtracted charge, no rectification',
        'waveform_plot_transform':'V to mV only; no smoothing/baseline subtraction/rescaling of plotted data',
        'source_semantics':'G4 deposit-derived analytic clean source input; not noisy source and not a measured waveform',
        'compositions_are_deposit_buckets_not_particle_truth':True,'new_REAL_read':0,'model_inference':0,'training':0,'new_fits':0,
        'selected_using_output':False,'fresh_migration_hash_test':False,
        'font':'C:/Windows/Fonts/msyh.ttc','elapsed_s':time.monotonic()-START,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
write('SOURCE_OUTPUT_REFERENCE.json',record)
README='''# 仿真输入与同父重建输出

主图为 `simulation_to_reconstruction.png`，SVG同名。三行分别为电子沉积、强子沉积和混合沉积；左列是实际进入完整波形网络的G4派生清洁仿真输入，右列是R28 FRESH_FINAL第2950步已保存的post-ADC输出。没有重新推理或训练，没有读取实测波形。

选择规则：每种源组成中，选择源侧冻结Qtotal窗口电荷最接近该组成中位数的事件，并列取原序号较小者。输出完全不参与选择。三个CSV保留完整480点，绘图只把V换为mV，不平滑、也不分别归一化；每行左右时间和电压轴相同。索引120为对齐参考0，网格间隔1ns，并非绝对采集时间。尾段图为辅助，不能替代完整主图。

按已完成DEV完整shape493 C2ST BA较低选择R28末点作开发展示（0.970仍明显可区分）。旧J20所选2250步模型保持不变；这不是最终验收通过，也不是按图挑好看的事件。来源组成桶不自动等同独立粒子真值，图形接近不能证明逐事件物理对应。完整输入/输出/权重身份与选中事件、PSD/Q口径见SOURCE_OUTPUT_REFERENCE.json。

复现：在新的E/D目录复制本预览脚本、config、supervise和owned_worker，将config source_stage指向只读R28存档。使用既有CPU Python，4线程，所有缓存写新目录；本次真实PID/退出/RSS见make_preview.py.NATIVE_RECEIPT.json。不要覆盖旧阶段或删除本次.used重提。
'''
(ROOT/'README.md').write_text(README,'utf8')
size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
assert size<CFG['output_cap_bytes']
print(json.dumps({'PID':os.getpid(),'selected_indices':pick.tolist(),'ID_order_equal':True,
                  'science_plot_elapsed_s':time.monotonic()-START,'output_bytes_so_far':size,
                  'figures':[str(ROOT/'simulation_to_reconstruction.png'),str(ROOT/'simulation_to_reconstruction_tail.png')],
                  'training_model_inference_new_REAL':0}),flush=True)
