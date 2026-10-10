from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
S = R / 'work/R55_PSD_SOURCE_COMPATIBILITY_001'
now = datetime.now(timezone.utc).isoformat()
result = json.loads((S / 'RESULT.json').read_text('utf8'))
assert result['scientific_decision'] == 'relaxed_diagnostic_does_not_rule_out_compatibility'
certificate = result['certificate']
assert certificate['valid'] and not certificate['exact_lower_bound_exceeds_diagnostic_threshold']
p = S / 'compatibility_bounds.png'
qa = {'UTC': now, 'owner': '/root', 'all_figures_opened': True, 'layout_pass': True,
      'method': 'Actually opened the full R55 three-panel rendered PNG with image viewer after terminal fetch.',
      'observations': ['All source and REAL bin masses and allowed migration bars remain visible.',
                       'The fixed CDF and both valley bands are readable.',
                       'Actual lower bound and explicit lack of an A-F success claim are visible.'],
      'figures': [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size,
                   'opened': True, 'layout_pass': True}]}
(S / 'FIGURE_QA.json').write_text(json.dumps(qa, indent=2) + '\n', 'utf8')
direction = {'UTC': now, 'owner': '/root', 'scientific_decision': result['scientific_decision'],
    'model_goal_complete': False, 'application_goal_status': 'active',
    'conclusion_zh': 'R55 未证明验收目标冲突：一次固定 LP 的数值最优值为 0，精确对偶下界为 −1.2813960740059585e−23，未超过 0.010001。结合距离非负性只能得到平凡下界 0。这个放宽结果既不能排除原要求的可行性，也不能证明存在合格的完整波形或神经方法。',
    'certified_lower_bound': certificate['lower_bound'],
    'original_population_changed': False, 'original_A_F_changed': False, 'new_response_training': 0,
    'next_action': '封存 R55 必要条件诊断，不因其低下界放宽原 A–F，也不继续扫描下界设置。下一轮登记仅用 TRAIN 的教师端点与冻结 R54 学习误差诊断：先追溯固定匹配流的目标人口和原始源覆盖，再比较教师本身与全量合法 REAL TRAIN 的 PSD/完整响应描述及冻结模型对同一端点的误差；没有获得明确端点或学习瓶颈证据前，不启动新的响应训练。',
    'interpretation_limits': ['Fractional LP variables are a relaxation, not new event weights or generated waveforms.',
        'Interval-averaged CDF, McCormick and valley lower-integral relaxations can be loose.',
        'Fixed HGB, full waveform, density, source-rank, noise and D/E feasibility are not established.']}
(R / 'reviews/R55_ROOT_DIRECTION_001.json').write_text(json.dumps(direction, ensure_ascii=False, indent=2) + '\n', 'utf8')
print(json.dumps({'layout_pass': True, 'decision': direction['scientific_decision'], 'model_goal_complete': False}))
