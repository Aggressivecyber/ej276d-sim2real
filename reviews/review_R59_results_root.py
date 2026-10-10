from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
R=Path(__file__).resolve().parents[1];S=R/'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001'
def read(p):return json.loads(p.read_text('utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
M=read(S/'METRICS_COMPACT.json');G=read(S/'GATE_METRICS.json');arm='DIRECT_DISTRIBUTION_RESPONSE';ctrl='DIRECT_DISTRIBUTION_REAL_CONTROL'
A=M['acceptance_A_C'][arm];g=G['models'][arm]
assert not any(A[k]['pass']for k in ['A','B','C']) and G['decision']=='close_fixed_R59_direct_distribution_recipe'
assert read(S/'FROZEN_ARTIFACT_REVIEW.json')['pass'] and read(S/'REPLAY_RECEIPT.json')['pass']
figs={
 'canonical_five_view_response.png':'All five SOURCE intervals exceed the acceptance band; all five same-treated REAL intervals remain inside it. Legend and intervals readable.',
 'PSD_all_eight_bins.png':'All eight fixed intervals visible; Q3/Q4 sharp shifted peaks and Q8 failed fit retained; shared axes, legends and density labels readable.',
 'PSD_centers_widths_and_W1.png':'All fixed-bin W1 values visible; failed density fits shown as gaps; required center and width discrepancies visible.',
 'training_and_checks.png':'All seven training losses and sole 50000 endpoint visible; training audit distinct from A/B/C failure.',
 'noise_and_source_association.png':'Noise RMS and ACF distributions plus original event PSD/Q associations visible; association not described as particle truth.',
 'complete_waveform_population.png':'Complete 480-point raw mean and SD visible; all source weights and relevant time boundaries retained.',
 'fixed_time_boundaries.png':'All nine boundaries and fixed limit visible; three violations near sample150 are visible.',
 'explicit_constraint_projection.png':'Reopened revised zero-count figure; readable counts and table, no empty or negative automatic axes; TRAIN trace distinguished from frozen main outputs.'}
now=datetime.now(timezone.utc).isoformat()
write(S/'FIGURE_QA.json',{'UTC':now,'owner':'/root','all_figures_opened':True,'layout_pass':True,
 'method':'Actual view_image inspection of all eight PNGs; revised projection figure separately reopened.',
 'figure_count':len(figs),'figures':[{'path':'figures/'+n,'SHA256':hashlib.sha256((S/'figures'/n).read_bytes()).hexdigest(),'opened':True,'layout_pass':True,'observation':v}for n,v in figs.items()],
 'repair_receipt':'reviews/R59_FIGURE_REPAIR_001/RECEIPT.json','scientific_acceptance_pass':False})
repair=read(R/'reviews/R59_FIGURE_REPAIR_001/RECEIPT.json');repair.update(visual_review_pending=False,revised_figure_opened=True,layout_pass=True,review_UTC=now)
write(R/'reviews/R59_FIGURE_REPAIR_001/RECEIPT.json',repair)
next_action='封存R59固定末点，不续训或选择其他检查点。下一轮先登记仅用合法TRAIN的冻结R59损失与梯度诊断：按原父事件权重完成一次全TRAIN固定前向，并在预先固定的小批量上分解七项损失及全窗四个视图的参数梯度，区分条件分布不足、源顺序/谷约束竞争、支持外投影和数值实现问题。保留全部源父事件及同处理REAL，据TRAIN证据决定下一项直接无配对机制；原A–F、最终源人口和阈值不变，不因本轮负结果启动D/E或宣称目标不可能。'
direction={'UTC':now,'owner':'/root','scientific_decision':G['decision'],'model_goal_complete':False,'application_goal_status':'active',
 'conclusion_zh':'R59完成唯一50000步直接无配对训练和全部原检查。SOURCE五视图BA为0.942/0.956/0.944/0.748/0.882，全部未通过；同处理REAL全部通过。总体/共同Q1–7 PSD W1为0.01445470/0.02647840，必需低/高峰位MAE为0.01599027/0.03823346，峰宽平均相对误差约39.95%/38.97%，均未达原阈值。源保持、数值和4000条独立重放通过，150点附近三项边界失败。原A/B/C全部失败，关闭该固定方法，完整模型目标未完成。',
 'A_B_C_pass':{k:A[k]['pass']for k in ['A','B','C']},'frozen_views':A['A']['views'],'PSD_and_density':A['B'],
 'original_six_gates':{k:v['pass']for k,v in g.items()if k.startswith('gate')},
 'boundary_failures':[x for x in g['gate4_boundary_source']['boundaries']if not x['pass']],
 'source_valley_counts':g['gate3_source_preservation']['actual_source_pure_had_valleys'],
 'source_associations':g['gate3_source_preservation']['whole_population_source_ranks'],
 'SOURCE_paired_BA_delta_vs_R54':{k:{'delta':v['BA_delta_first_minus_second'],'CI':v['BA_delta_95CI']}for k,v in M['paired_R59_minus_frozen_R54'][arm].items()},
 'projection_main':{n:{'events':M['explicit_constraint_projection'][n]['N'],'range_active_events':M['explicit_constraint_projection'][n]['range_active_events'],'rail_sample_total':M['explicit_constraint_projection'][n]['rail_sample_total'],'max_RMS_V':M['explicit_constraint_projection'][n]['fields']['constraint_projection_RMS_V']['quantiles'][-1]}for n in [arm,ctrl]},
 'TRAIN_trace_projection':M['training_audit']['constraint_loss_trace_summary'],
 'all_eight_figures_opened':True,'final_artifact_load_review':'work/R59_DIRECT_DISTRIBUTION_RESPONSE_001/FROZEN_ARTIFACT_REVIEW.json',
 'transfer_recovery':'reviews/R59_TRANSFER_RECOVERY_001/RECEIPT.json','scientific_restarts_due_to_transfer':0,
 'D_E_evaluated':False,'original_A_F_changed':False,'evaluation_population_changed':False,'next_action':next_action,
 'interpretation_limits':[
  'Passing numerical stability, source association and same-treated REAL does not imply joint response acceptance.',
  'All five SOURCE BA differences versus R54 are positive; this is one frozen classifier comparison, not a single-factor architecture ablation.',
  'Zero projection corrections on the 4000 saved main outputs do not establish zero projection during TRAIN or on every possible input.',
  'Loss plateaus and output spikes motivate TRAIN diagnosis but do not establish which loss or implementation caused the failure.',
  'This negative fixed method neither disproves feasibility of the full goal nor validates D/E or independent acquisitions.']}
write(R/'reviews/R59_ROOT_DIRECTION_001.json',direction)
print(json.dumps({'figure_QA':True,'A_B_C_pass':direction['A_B_C_pass'],'next_action':next_action},ensure_ascii=False,indent=2))
