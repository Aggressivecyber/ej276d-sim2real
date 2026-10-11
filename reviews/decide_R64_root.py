"""Record the root decision from the completed diagnostic, without training."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
S = R/'work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001'
target = R/'reviews/R64_ROOT_DIRECTION_001.json'
assert not target.exists()
read = lambda p: json.loads(p.read_text('utf8'))
result = read(S/'RESULT.json')
parameter = read(S/'PARAMETER_GRADIENT_SUMMARY.json')
endpoint = read(S/'ENDPOINT_GRADIENT_SUMMARY.json')
assert read(S/'FROZEN_ARTIFACT_REVIEW.json')['pass_review']
assert read(S/'ROOT_ARTIFACT_REVIEW.NATIVE_RECEIPT.json')['native_exit'] == 0
assert read(S/'FIGURE_QA.json')['layout_pass']

def binding(name):
    p = R/name
    return dict(path=name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())

names = ['work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001/'+n for n in (
    'RESULT.json', 'PARAMETER_GRADIENT_SUMMARY.json', 'ENDPOINT_GRADIENT_SUMMARY.json',
    'ENDPOINT_GEOMETRY_SUMMARY.json', 'SUPPORTING_SUMMARY.json', 'FROZEN_ARTIFACT_REVIEW.json',
    'ROOT_ARTIFACT_REVIEW.NATIVE_RECEIPT.json', 'FIGURE_QA.json')]
names += ['reviews/R64_RESULT_ANALYSIS_001/gradient/REPORT.md',
          'reviews/R64_RESULT_ANALYSIS_001/gradient/ANALYSIS.json',
          'reviews/R64_RESULT_ANALYSIS_001/geometry/REPORT.md',
          'reviews/R64_RESULT_ANALYSIS_001/geometry/SUMMARY.json',
          'reviews/R64_RESULT_ANALYSIS_001/literature_followup/REPORT.json',
          'reviews/R64_RESULT_ANALYSIS_001/literature_followup/SOURCES.json']
decision = ('封存R64固定诊断：联合128的参数total_D均方误差比四个32批的平均增加206.362353，'
            '原95%配对区间[99.051001,343.486026]。两空间全部分量的32宏均值更接近参照，'
            '但总经验离散度增加更多；本次不支持采用B128作为梯度误差改进。'
            '完整几何残差和全部支持外父事件保留，未证明原A–F不可行。')
next_action = ('下一轮预登记完整有限TRAIN分布训练：从原R63同一初始化重新训练，'
    '每次更新对全部6593 SOURCE父事件各取一份新噪声，并对全部21047 canonical REAL计算原五项D。'
    '按原64行调度完成两遍缓存和全部参数VJP，将五项D的端点梯度合并后注入，原区域质量、'
    '标准化、损失权重、ADC/Box路径和原四项C定义保持。C每更新保留一个原320 SOURCE+64 REAL identity微批，'
    '原索引及每次出现独立噪声流保持；完整D噪声使用另行声明的流。'
    '保留原优化器及学习率日程，预先固定与R63相同的50000次更新为唯一末点；'
    '先完成联合五D梯度、原C取项与真实网络回放的CPU/CUDA人工检查，再训练，'
    '按原数值门、完整A–C及独立重放评价；通过后才继续D/E。'
    '不续训R63、不选择中间检查点、不删除或重权支持外人口，不重开R60阴性冲突标记。')
record = dict(
    UTC=datetime.now(timezone.utc).isoformat(), owner='/root',
    stage='R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001',
    decision_status='fixed_diagnostic_closed_next_protocol_preparation_selected',
    model_goal_complete=False, application_goal_status='active', original_A_F_changed=False,
    original_R63_A_B_C=['failed','failed','failed'], original_A_F_reassessed=False,
    scientific_decision=decision, next_action=next_action,
    primary_parameter_gradient_comparison=result['primary_parameter_gradient_comparison'],
    parameter_total_D={arm:parameter['arms'][arm]['total_D'] for arm in ('A','B')},
    endpoint_total_D={arm:endpoint['arms'][arm]['total_D'] for arm in ('A','B')},
    evidence_bindings=[binding(n) for n in names],
    reasoning=[
        'The primary parameter-space total-D interval is strictly positive; the pre-result conditional B128 proposal is not activated.',
        'All five finite mean-gradient discrepancies decrease in B, but PSD dispersion and the aggregate cross-term dominate the total squared error increase.',
        'The A/B relative total-D errors are about 2.30/3.68 of the fixed reference squared norm; smaller A error alone does not show either estimator is adequate.',
        'Full-window and coordinate-RMS residuals exceed the saved REAL-pair descriptions in every original region, while the nominal charge bound covers only a specified fraction of measured coordinate discrepancy.',
        'Exact full finite-parent D evaluation removes this parent-subsampling reduction discrepancy conditional on each step noise/directions. It does not integrate conditional noise or all directions.',
        'One original C microbatch preserves its formula and marginal draw law. Four-C averaging would introduce another variance change and is not selected.',
        'D and C use distinct endpoint realizations; their joint stochastic coupling changes from R63. This is documented, not claimed as an identical full-objective stochastic law.',
        'The next candidate tests a new finite empirical objective using more data and compute per update. No equal-exposure, equal-compute, monotone batch-size or guaranteed A-F improvement claim is made.'
    ],
    implementation_baseline=dict(
        parent_order='original full SOURCE and REAL archive order',
        reference='all raw REAL passed only through original canonical_REAL_torch',
        full_D_gradient='one endpoint leaf, joint five-term scalar, one total endpoint derivative, full chunked parameter VJP',
        network_chunk=64, new_training_weight_loaded=False,
        fixed_terminal_updates=50000, optimizer_and_learning_rate_schedule='original R63 unchanged',
        C='one original DistributionLoss call; retain its four C pieces, discard its five microbatch D pieces',
        feature_cache='no additional REAL feature cache selected',
        extra_C_averaging=False, sampled_parameter_VJP=False,
        new_protocol_registered=False, new_scientific_execution_started=False),
    interpretation_limits=[
        'Fixed frozen-model finite-noise diagnostic, not an estimate of all original occurrence-noise training trajectories.',
        'Box selected backward, ADC STE and stable sorting define an algorithmic surrogate derivative, not a unique classical derivative of quantized output.',
        'Component and supplementary angle/cross-term intervals are secondary and unadjusted; original primary comparison remains parameter total_D.',
        'The nominal V bound is not a causal fraction, interval certificate or proof that original A-F cannot be satisfied.',
        'No new response weight, DEV/EVAL/CAL decode, density fit or acceptance result exists in R64.'
    ])
target.write_bytes((json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
print(json.dumps(dict(path=str(target), scientific_decision=decision,
                      new_scientific_execution_started=False), ensure_ascii=False, indent=2))
