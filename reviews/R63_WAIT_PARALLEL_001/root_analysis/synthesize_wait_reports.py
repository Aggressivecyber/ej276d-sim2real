"""Record the root's evidence synthesis; no scientific arrays or model execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[3]
P = R / 'reviews/R63_WAIT_PARALLEL_001'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text('utf8'))

report_paths = {
    'literature': P / 'literature/report.json',
    'train_evidence': P / 'train_evidence/report.json',
    'validation_preparation': P / 'validation_prep/report.json',
    'conditional_gradient_implementation': P / 'literature/C1_IMPLEMENTATION_NOTES.json',
    'charge_coordinate_loss_bound': P / 'root_analysis/CHARGE_LOSS_BOUND_STATIC.json',
}
reports = {key: load(path) for key, path in report_paths.items()}
expected = {
    'literature': '20891a32bca4b0d2eabcec295b9219eb7c863289d063ede925646cc314019e2f',
    'train_evidence': 'c3aab2871101d57ed71c082eefcf44da86060c54911de7ccd4c7f39ee9ca5bf7',
    'validation_preparation': 'e1eafdc353f90b4c56eb0ec2d4cbd125cd26707035d92a446130279e02640c7b',
    'conditional_gradient_implementation': '4b593215183c1c4893e243338a859582fd981cdde22bece76f1af9056688953f',
}
for key, digest in expected.items():
    assert sha(report_paths[key]) == digest, ('reported_document_changed', key)

references = {}
for entry in reports['literature']['project_evidence']:
    references[entry['path']] = entry['sha256']
for entry in reports['train_evidence']['evidence_index'].values():
    path = str(R / entry['file'])
    if path in references:
        assert references[path] == entry['sha256']
    references[path] = entry['sha256']
for entry in reports['conditional_gradient_implementation']['code_evidence']:
    if entry['path'] in references:
        assert references[entry['path']] == entry['sha256']
    references[entry['path']] = entry['sha256']
# AGENTS has an explicitly authorized parity-policy addition since the peer review.
known_policy_change = load(P / 'root_analysis/AGENT_PARITY_REQUIREMENT_RECEIPT.json')
verified = []
for name, digest in references.items():
    path = Path(name)
    actual = sha(path)
    relative = str(path.relative_to(R)).replace('\\', '/')
    if actual != digest:
        change = known_policy_change['changed_files'].get(relative)
        assert change and digest == change['before_SHA256'] and actual == change['after_SHA256'], relative
        disposition = 'user_authorized_model_parity_text_change_after_peer_review'
    else:
        disposition = 'same_bytes'
    verified.append({'path': relative, 'reviewed_SHA256': digest, 'current_SHA256': actual,
                     'disposition': disposition})

coverage = load(R / 'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001/LOSS_COVERAGE_SUMMARY.json')
psd = load(R / 'work/R62_FROZEN_FULL_WAVEFORM_RESIDUAL_001/PSD_SUMMARY.json')
region = reports['train_evidence']['region_and_representation_semantics']
source = coverage['comparisons']['SOURCE_REAL']
assert source['full_empirical_loss'] == region['SOURCE_full_window_total']
assert source['full_region_contributions'][9] == region['above_SOURCE_full_window_contribution']
current = load(R / 'work/R63_FULL_WAVEFORM_COVERAGE_RESPONSE_001/ROOT_CONTINUATION_OBSERVATION.json')
assert current['worker_PID'] == 218063 and current['identity_matches'] and current['exists']

result = {
    'schema_version': 1,
    'UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'root_synthesis_completed_while_frozen_R63_remains_running',
    'previous_goal_turn_classification': 'progress',
    'previous_goal_turn_evidence': 'Parallel literature and saved-TRAIN reviews completed; user-required model/effort parity was verified and persisted without changing 40 protected scientific files.',
    'authority': 'Current user-authorized auxiliary research and planning only; this is not the registration or execution of a new scientific method.',
    'current_execution_observation': current,
    'agent_configuration': {
        'same_model_and_reasoning_as_root_required': True,
        'verified_model': 'gpt-6-astra', 'verified_effort': 'max',
        'all_three_auxiliary_agents_match': True,
        'evidence': 'root_analysis/AGENT_MODEL_PARITY_OBSERVATION.json',
    },
    'reviewed_reports': {key: {'path': str(path.relative_to(R)).replace('\\', '/'),
                               'SHA256': sha(path)} for key, path in report_paths.items()},
    'reference_hash_checks': verified,
    'decisive_evidence': [
        {
            'finding': 'The large high-charge residual is a real saved TRAIN discrepancy with a same-region REAL reference, not an empty-reference artifact.',
            'SOURCE_N': 918, 'REAL_N': 1032,
            'SOURCE_mass': region['above_SOURCE_mass'],
            'full_window_contribution': source['full_region_contributions'][9],
            'full_window_total': source['full_empirical_loss'],
            'fraction': source['full_region_contributions'][9] / source['full_empirical_loss'],
            'boundary': 'Raw bin 8 is above a frozen conditioning boundary. Fine-grained charge/shape support and irreducible share remain unknown. Peak-normalized shape residuals also exist. Scalar loss share is not gradient share.',
        },
        {
            'finding': 'Finite-batch noise exists, but cannot by itself explain the saved SOURCE residual.',
            'SOURCE_batch_mean': source['batch_loss_summary']['mean'],
            'REAL_self_batch_mean': coverage['comparisons']['REAL_REAL']['batch_loss_summary']['mean'],
            'boundary': 'All values describe frozen R61 on R62 TRAIN, not the live R63 result. A larger batch is a hypothesis, not an adopted remedy.',
        },
        {
            'finding': 'RMS changes the concentration of coordinate-W1 gradients; the added 31-feature map is nonunique, while the retained whole-waveform objective still provides joint constraints.',
            'boundary': 'Do not infer a blind space for all nine objectives from the sign/reversal invariance of the 31-feature map. First determine whether the covered 31-dimensional distribution itself was fitted.',
        },
        {
            'finding': 'The original D/E obligations require current-method adapters and explicit treatment of an empty GAP Q6 training region.',
            'D': 'Restore the actual 21047 REAL row identities and all required paired relative/absolute comparisons; preserve the complete 2000 synthetic-parent design.',
            'E': 'CAL q3 contains the entire fixed narrow Q6 interval. Empty-region statistics, sampling, conditional inference, and FULL reuse versus matched retraining must be declared before science. Both nominal-source and output-region evaluations remain required.',
            'boundary': 'Static preparation is not D/E execution or completion. Actual identity intersections remain unverified.',
        },
    ],
    'primary_literature_root_checked': [
        {'url': 'https://arxiv.org/pdf/1705.10743', 'version': 'v1',
         'location': 'Theorem 1',
         'use': 'An empirical Wasserstein gradient can target a different expected objective; this is a general possibility, not proof of collapse here.'},
        {'url': 'https://arxiv.org/html/2101.01792v1',
         'location': 'Section 3.5 Definition 18; Section 4.2 Theorems 29-30',
         'use': 'Minibatch expectation and full-measure objectives must be distinguished. Subtracting a REAL self-loss constant is not a verified correction of the current compound objective.'},
    ],
    'root_decision': {
        'now': 'Finish the same R63 update50000 and frozen numerical/A-C/replay pipeline, then inspect actual evidence and all required figures.',
        'if_original_A_C_pass': 'Prepare and register the complete D/E adapters, identities, empty-region rules and original acceptance comparisons before final evaluation. Use the statistics preparation only after actual-design validation.',
        'if_numerical_gate_fails': 'Close this fixed numerical recipe with its real failure; address the specific numerical evidence before any new response evaluation or inference diagnostic.',
        'if_numerical_pass_and_A_C_fail': 'First register one shared TRAIN-only endpoint and charge/reference-geometry diagnostic. Reuse frozen R61/R63 scales and all original parents. Determine which covered statistics remain unfitted before adding more features.',
        'candidate_next_gradient_question': 'A paired 4x32 versus 128 distribution-gradient comparison may share that endpoint collection after its exact reference, masses, duplicate/noise semantics, ties and two-pass VJP are frozen and checked. It remains an unregistered candidate.',
        'local_feature_candidate': 'Keep the proposed 58-dimensional local/PSD extension as a secondary hypothesis. Do not train it merely because 31 features are mathematically nonunique, or combine it automatically with a batch-size change.',
        'not_adopted_automatically': 'Peer-proposed numerical diagnostic markers, batch plans and future training arms are not registered by this synthesis. Their full scientific definitions will be fixed only in a distinct protocol if evidence warrants it.',
    },
    'necessary_interpretation_corrections': [
        'The proposed per-parent fixed-noise full reference differs from the live training per-occurrence independent noise. It measures a conditional finite empirical objective, not the full stochastic training expectation.',
        'g_F is a distribution-only algorithmic proxy gradient under canonical ADC STE and the custom box/charge backward, not a unique classical gradient of the hard quantizer or a population truth.',
        'Global PSD/noise use separate original SOURCE and REAL masses; regional shape losses use SOURCE masses; conditional PSD averages the original seven regions equally.',
        'A 128-row-per-region union must preserve region-major layout. Do not merge source-order pairs or nonlinear valley terms across the four original microbatches when isolating distribution reduction.',
        'All R60 conflict markers remain untriggered. Any R63 gradient claim requires new preregistered R63 evidence.',
    ],
    'charge_bound_review': {
        'artifact': 'root_analysis/CHARGE_LOSS_BOUND_STATIC.json',
        'root_and_peer_algebra_agree_under_stated_conditions': True,
        'conditions': 'Same within-region weights and center/scale; exact W1 of represented coordinates; representation and numerical-reduction errors explicitly bounded. The source-charge form compares to canonical REAL charge, or includes the REAL ADC error too.',
        'numeric_bound_computed': False,
        'A_F_infeasibility_proven': False,
    },
    'scope_counters': {'new_scientific_jobs': 0, 'new_model_forwards': 0,
                       'new_gradient_computations': 0, 'new_training_updates': 0,
                       'DEV_EVAL_CAL_arrays_read': 0, 'new_scientific_stage_registered': False},
    'model_goal_complete': False,
}
out = P / 'ROOT_SYNTHESIS.json'
assert not out.exists(), 'Preserve the prior synthesis; use a dated follow-up for new evidence.'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
load(out)
print(json.dumps({'status': result['status'], 'output': str(out), 'reviewed_reports': len(report_paths),
                  'verified_reference_files': len(verified), 'actual_R63_step': current['step'],
                  'model_goal_complete': False}, indent=2))
