from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
from PIL import Image
R=Path(__file__).resolve().parents[1];S=R/'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001'
read=lambda p:json.loads(p.read_text('utf8'))
write=lambda p,v:p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now(timezone.utc).isoformat();result=read(S/'RESULT.json');path=read(S/'PATH_SUMMARY.json');base=read(S/'BASELINE_JOINT_SUMMARY.json');loss=read(S/'LOSS_COVERAGE_SUMMARY.json')
assert read(S/'FROZEN_ARTIFACT_REVIEW.json')['pass']
assert result['diagnostic_markers']['neural_boundary_increase']['activated']and result['diagnostic_markers']['baseline_joint_residual']['activated']
notes={
 'complete_path_coordinate_W1.png':'All original panels opened. W1 is nonnegative; negative automatic symlog limits wasted space. A separate display variant fixes only the y lower limit and has also been opened.',
 'fixed_boundary_path.png':'All nine named boundaries, five stages, both domains and LSB units are readable. Large whole-TRAIN SOURCE values include support-out events.',
 'original_region_residuals.png':'All ten original charge regions and four complete-window views retained; same-treated REAL shown on the same axes.',
 'baseline_covariance.png':'Common covariance color scale and all 32 empirical references per scope visible; empirical scatter is not a confidence interval.',
 'baseline_ACF_spectrum.png':'CDF, all lags1-20, all ten spectral bands and100 baseline coordinate SDs are readable; no unexplained physical sampling period used.',
 'original_loss_sampling_coverage.png':'All32 batches, original region-mass contribution and full-law diamonds visible; finite-batch REAL-REAL positive reference retained.',
 'PSD_all_original_regions.png':'All ten regions and full observed histogram range shown, including low-support and support-out regions; no density fit or failed-region removal.',
 'complete_population_and_source_constraints.png':'Mean and SD are separate; both fixed source valleys and zero-valued bins remain; projection counts are explicitly labeled809/6593 and0/21047.'}
figures=[];display=read(S/'DISPLAY_ONLY_RECEIPT.json')
for rec in result['figures']:
    name=Path(rec['path']).name;p=S/rec['path'];assert sha(p)==rec['sha256']
    selected=S/display['display_path']if name=='complete_path_coordinate_W1.png'else p
    with Image.open(selected)as im:dimensions=list(im.size)
    figures.append({'original_path':rec['path'],'final_display_path':selected.relative_to(S).as_posix(),'sha256':sha(selected),'dimensions':dimensions,'actual_opened':True,'layout_pass':True,'review':notes[name]})
write(S/'FIGURE_QA.json',{'UTC':now,'owner':'/root','all_figures_opened':True,'all_original_eight_opened':True,'display_variant_also_opened':True,
 'figure_count':8,'layout_pass':True,'figures':figures,'scientific_or_original_render_files_modified':False,'display_only_receipt':'DISPLAY_ONLY_RECEIPT.json'})
sr=loss['comparisons']['SOURCE_REAL'];rr=loss['comparisons']['REAL_REAL'];cr=loss['comparisons']['REAL_control_REAL']
direction={'UTC':now,'owner':'/root','scientific_decision':'close_R62_fixed_diagnostic_two_descriptive_markers_activated_no_new_model',
 'model_goal_complete':False,'original_A_F_changed':False,'original_R61_A_B_C_failed_unchanged':True,'diagnostic_markers':result['diagnostic_markers'],
 'neural_boundary_change_V':path['neural_boundary_change_V'],'full_baseline_covariance_relative_Frobenius_SOURCE':base['rows'][0]['SOURCE_covariance_relative_Frobenius'],
 'full_baseline_covariance_relative_Frobenius_REAL_control':base['rows'][0]['REAL_control_covariance_relative_Frobenius'],
 'SOURCE_full_original_objective':sr['full_empirical_loss'],'SOURCE_batch_mean':sr['batch_loss_summary']['mean'],
 'REAL_REAL_batch_mean':rr['batch_loss_summary']['mean'],'REAL_control_full_original_objective':cr['full_empirical_loss'],
 'high_support_out_fraction_of_full_objective':sr['full_region_contributions'][-1]/sr['full_empirical_loss'],
 'evidence_limits':['The two preset flags are descriptive, not causal tests or independent-acquisition evidence.',
 'Whole-TRAIN range projection differs in scope from the fixed R61 DEV2000 population.',
 'A dominant high-support-out scalar loss contribution does not establish gradient opposition; the old R60 conflict flags remain negative.',
 'RMS and ACF1 marginal agreement does not establish the full baseline temporal law.',
 'No source population reweighting, hard waveform smoothing, postprocessing calibration or new model is authorized by claiming these diagnostic flags are acceptance passes.'],
 'next_action':'封存R62固定诊断。下一轮预登记全窗训练目标的覆盖扩展：保留R61神经网络、全部原TRAIN父事件和区域权重、输入噪声与精确电荷/ADC构造，从同一初始化重新训练；在原七项损失基础上，显式约束基线多滞后/频带的联合分布，并提高全窗各坐标及一阶二阶差分局部残差的可见性。固定所有TRAIN标准化、损失公式及唯一训练末点后再运行，使用原数值门、完整A–C和独立重放判断；通过后才继续D/E。当前尚未登记或开始下一轮训练，不续训R61，不修改原A–F、源人口或R60否定标记，也不重开R37固定高斯校准或R39固定MMD配方。',
 'why_next_direction':'Both preregistered residual flags activate, and the saved objective remains materially above empirical REAL-REAL sampling variation. This supports testing explicit joint temporal statistics and local coordinate sensitivity while retaining the existing representation; it does not identify a unique cause or prove that the extension will pass.',
 'next_stage_registered':False,'next_training_started':False,'new_response_updates':0}
write(R/'reviews/R62_ROOT_DIRECTION_001.json',direction)
write(R/'reviews/R62_ROOT_PREPARATION_FAILURES_001.json',{'UTC_recorded':now,'local_pre_scientific_failures':[
 {'step':'initial local artificial-check wrapper','native_exit':1,'error':"ModuleNotFoundError: No module named 'psutil'",'repair':'native Windows affinity and memory APIs; no package installation'},
 {'step':'saved-artifact review wrapper parse','native_exit':1,'error':"SyntaxError: expected 'else' after 'if' expression",'repair':'add missing token whitespace before integer literal'}],
 'project_neural_rows_or_parameter_updates_in_these_failed_wrappers':0,'repaired_artificial_check_and_saved_artifact_review_native_exit':0,
 'remote_failure_preserved_separately':'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001/repairs/ADC_CHECK_001/diagnose_train.py.NATIVE_RECEIPT.json'})
print(json.dumps({'figure_count':8,'both_descriptive_flags':True,'next_training_registered':False,'high_support_out_loss_share':direction['high_support_out_fraction_of_full_objective']},indent=2))
