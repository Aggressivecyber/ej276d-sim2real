from pathlib import Path
import ast
R=Path(__file__).resolve().parents[1]
text=(R/'reviews/deliver_R52_root.py').read_text('utf8')
text=text.replace('R52','R53').replace('R53_JOINT_CHARGE_PSD_COUPLING_001','R53_PULSE_ONLY_TEACHER_FLOW_001')
text=text.replace('R51','R52')
text=text.replace('r52_joint_coupling_final_','r53_pulse_teacher_final_').replace('r52_science_incremental.bundle','r53_science_incremental.bundle')
text=text.replace('Close fixed joint charge-PSD coupling after two noise-boundary failures','Record frozen pulse-only teacher flow and joint response evaluation')
text=text.replace('Record R53 negative delivery and pulse-only endpoint direction','Record R53 delivery and evidence-backed next action')
start=text.index("rootfiles=['R53_REPORT.md'")
end=text.index('\nrel=lambda',start)
text=text[:start]+"rootfiles=['R53_REPORT.md','tasks/'+STAGE+'.md','reviews/R53_ROOT_DIRECTION_001.json']+[str(p.relative_to(R)).replace('\\\\','/')for p in (R/'reviews').glob('*R53*root*.py')]"+text[end:]
text=text.replace('R53 private coupling and parent arrays are included. Original TRAIN and R43 stream remain dependencies identified in INPUT_BINDING.json and config.json.',
                  'R53 portable inference weights and constants, final optimizer/random state, private outputs, descriptions and predictions are included. Original TRAIN, R43 stream and R52 coupling remain training-reproduction dependencies identified in INPUT_BINDING.json and config.json.')
text=text.replace("'pulse_only_endpoint_protocol_preparation_root_only'","'root_following_R53_evidence_backed_direction'")
text=text.replace("'当前完成私有Git与D盘交付。','私有Git与D盘实际交付已完成，下一步由主助手按长期目标协议继续推进。'",
                  "'当前完成私有 Git 与 D 盘交付。','私有 Git 与 D 盘实际交付已完成，见 R53_DELIVERY_COMPLETION_RECEIPT.json。'")
ast.parse(text)
(R/'reviews/deliver_R53_root.py').write_text(text,'utf8')
print('R53 delivery controller prepared; not run until actual results and root review are sealed')
