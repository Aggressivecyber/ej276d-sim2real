from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path(__file__).resolve().parents[1];S=R/'work/R56_TRAIN_ENDPOINT_LEARNING_DIAGNOSTIC_001';now=datetime.now(timezone.utc).isoformat()
v=json.loads((S/'RESULT.json').read_text('utf8'))
assert v['decision']=='fixed_direction_rule_inconclusive_review_full_evidence' and v['learning_markers']==v['teacher_bias_markers']==0
assert all(m['teacher_REAL_W1']<.002 for m in v['markers'])
names=['coverage','TRAIN_PSD_CDF','TRAIN_path_residual','TRAIN_full_waveform','direction_markers']
observations=['Both support-outside source populations remain visible; exposure repeats are labeled.',
    'All eight PSD bins and the full CDF are visible; unweighted source parents are distinguished from teacher exposures.',
    'All 32 charge/time cells and four residual measures are readable; not-integrated-residual limitation is explicit.',
    'All 480 samples, voltage/shape means and SD curves remain visible, including full-population spread and tail differences.',
    'All six direction markers show no hit; the 0.20 reference and actual low ratios are displayed without clipping.']
qa={'UTC':now,'owner':'/root','all_figures_opened':True,'layout_pass':True,'method':'Actually opened all five final PNGs with the image viewer after terminal fetch; no plot regeneration.',
    'observations':observations,'figures':[{'path':n+'.png','bytes':(S/(n+'.png')).stat().st_size,'sha256':hashlib.sha256((S/(n+'.png')).read_bytes()).hexdigest(),'opened':True,'layout_pass':True} for n in names]}
(S/'FIGURE_QA.json').write_text(json.dumps(qa,indent=2)+'\n','utf8')
next_action='封存R56，不把0/6学习标记解释为通过，也不延长R54或直接加强PSD损失。下一轮只用合法TRAIN，预登记原TRAIN父事件等权的教师/耦合预检，去除旧exp(lambda*h)训练采样倾斜，并明确支持内外覆盖、源电荷/PSD关联、谷比例、完整波形与ADC/噪声约束。原验收2000个SOURCE的身份、顺序、组成和权重全部保持；训练采样政策的改变必须单列。先完成教师科学终点，有可用端点证据后再另行登记神经训练；不读取DEV/EVAL选参。'
d={'UTC':now,'owner':'/root','scientific_decision':v['decision'],'model_goal_complete':False,'application_goal_status':'active',
    'conclusion_zh':'R56的预登记学习残差标记为0/6，教师偏差标记也是0/6，不能据此启动显式PSD切向机制。教师Q2–Q7的PSD W1为0.000374–0.001770；按旧训练曝光统计，冻结R54在Q2–Q6的W1为0.00385–0.00976。相同源端点按原父事件等权统计后，Q2–Q6为0.01665–0.04285，显示原训练采样测度与等权源人口之间的明显差异。2840个支持外TRAIN源从未参加训练，且本轮其中647个preADC输出越过ADC电压范围。模型目标仍未完成。',
    'learning_markers':0,'teacher_bias_markers':0,'unweighted_Q2_Q7_W1':[r['features']['PSD']['W1'] for r in v['output_groups']['all']['preADC_vs_REAL_source_bins'][2:8]],
    'weighted_Q2_Q7_W1':[m['R54_REAL_W1'] for m in v['markers']],
    'original_A_F_changed':False,'original_evaluation_population_changed':False,'new_response_training':0,
    'next_action':next_action,'interpretation_limits':['Repeated occurrences share one fixed output realization and are not independent samples.',
        'Changing descriptive parent weights exposes a distribution sensitivity; it is not a new accepted response or unique causal proof.',
        'TRAIN path residuals do not equal integrated-trajectory residuals.',
        'Support-outside parents lack a teacher in the original training stream; no teacher accuracy can be invented for them.',
        'Full-wave moments and PSD descriptions do not establish fixed-HGB, source-valley, density, downstream-utility or independent-acquisition acceptance.']}
(R/'reviews/R56_ROOT_DIRECTION_001.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf8')
print(json.dumps({'layout_pass':True,'scientific_decision':d['scientific_decision'],'model_goal_complete':False}))
