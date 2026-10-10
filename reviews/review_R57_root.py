from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path(__file__).resolve().parents[1];S=R/'work/R57_EQUAL_PARENT_TEACHER_DIAGNOSTIC_001';now=datetime.now(timezone.utc).isoformat()
v=json.loads((S/'RESULT.json').read_text('utf8'));c=json.loads((S/'COUPLING_RECEIPT.json').read_text('utf8'))
assert v['decision']=='close_fixed_equal_parent_16_group_teacher' and not v['all_teacher_conditions_pass']
assert c['all_source_parents']==6593 and c['all_REAL_parents']==21047 and c['edges']==27724 and c['exact_integer_margins_all']
assert v['failed_gates']==['no_teacher_ADC_range_excess','preADC_overall_PSD','canonical_overall_PSD','source_h_one_Q3_Q4_valleys']
names=['population_and_margins','TRAIN_PSD_CDF','ADC_and_valley_conditions','TRAIN_full_waveform_V','TRAIN_full_waveform_U','fixed_teacher_examples']
observations=['All ten regions and both parent counts/masses are visible; unequal overall REAL region weights are explicit.',
    'All ten regions plus overall weighted CDFs retain full observed PSD tails; preADC and canonical metrics agree with RESULT.',
    'All 1217 out-of-range teacher edges are in the upper exterior region; original SOURCE overrange and weighted teacher fraction are distinct. Both valley failures and +0.02 limits are visible.',
    'All 480 voltage samples and population SD bands are visible for eleven groups, including large upper-exterior corrections.',
    'All 480 normalized-shape means and SD bands are visible, including positive/negative broad corrections in both exterior regions.',
    'First fixed edge per region is shown with SOURCE, REAL and teacher; no example was selected based on appearance.']
qa={'UTC':now,'owner':'/root','all_figures_opened':True,'layout_pass':True,'method':'Actually opened all six fetched final PNGs with the image viewer; no plot regeneration or scientific rerun.',
    'observations':observations,'figures':[{'path':n+'.png','bytes':(S/(n+'.png')).stat().st_size,'sha256':hashlib.sha256((S/(n+'.png')).read_bytes()).hexdigest(),'opened':True,'layout_pass':True} for n in names]}
(S/'FIGURE_QA.json').write_text(json.dumps(qa,indent=2)+'\n','utf8')
next_action='封存R57等权16分位教师，不调整组数、删除支持外事件或启动该教师的神经训练。下一轮仅用合法TRAIN预登记联合约束教师的可行性问题：保留原源父事件等权边缘，同时处理总体/条件PSD、源谷区概率及PSD秩关联；把ADC有界脉冲可行性单列，先区分原源电荷的硬界与当前脉冲基底的不足。不能把更严格的全TRAIN预检失败外推为原DEV A–F不可能；也不能通过改验收人口、谷定义或读取DEV/EVAL选参解决。只有获得完整端点证据后才另行登记神经训练。'
d={'UTC':now,'owner':'/root','scientific_decision':v['decision'],'model_goal_complete':False,'application_goal_status':'active',
    'conclusion_zh':'R57全6593源等权、21047 REAL条件等权的整数耦合通过，但固定教师未通过。共同Q1–7 preADC/规范ADC PSD W1均值为4.97e-15/0.0000851；总体为0.0304266/0.0317340。Q3/Q4 h==1 TRAIN谷概率由0增加到12.82%/12.41%，高于2个百分点的预检线。1217条教师边越界，关联575个高端支持外源，占总源测度8.69%；原输入本身有816个高端源越界，不能把全部教师越界称为新增裁剪。关闭此固定教师，0神经训练，原模型目标未完成。',
    'source_parents':6593,'REAL_parents':21047,'edges':27724,'failed_gates':v['failed_gates'],'PSD_summary':v['PSD_summary'],
    'exact_equal_source_margins':True,'original_A_F_changed':False,'evaluation_source_population_changed':False,'TRAIN_sampling_measure_changed':True,'new_response_training':0,
    'global_PSD_gap_identity':{'teacher_vs_matched_REAL_W1':v['teacher_vs_actual_matched_REAL'][0]['features']['PSD']['W1'],'matched_REAL_vs_full_REAL_W1':v['actual_matched_REAL_vs_full_REAL'][0]['features']['PSD']['W1']},
    'next_action':next_action,'interpretation_limits':['Fractional coupling is a conditional statistical teacher law, not physical cross-domain event pairs or changed SOURCE event weights.',
        'The full equal-parent TRAIN law differs from both the historical tilted exposure law and the fixed DEV evaluation population.',
        'The conservative TRAIN valley reference is original raw SOURCE, not the original DEV BASE reference; failure does not certify A-F infeasibility.',
        'Original source overrange-parent counts and teacher overrange-edge mass have different denominators; no clipping causality claim is made.',
        'Conditional exact PSD/noise matching does not imply overall matching, full waveform equivalence, source-rank acceptance, fixed-HGB acceptance, or D/E utility.',
        'All ten regions and every invalid/overrange edge remain in saved statistics; no failed region is removed.']}
(R/'reviews/R57_ROOT_DIRECTION_001.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf8')
print(json.dumps({'layout_pass':True,'scientific_decision':d['scientific_decision'],'model_goal_complete':False}))
