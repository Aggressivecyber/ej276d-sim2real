"""Review already saved outputs; no network evaluation or fitting occurs here."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,math,shutil
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'work/R67_EQUAL_COST_REPAIR_001'
sys.path.insert(0,str(R/'reviews/R65_PREPARATION_001/training_audit_adapter'))
from local_resource_guard import apply,memory
guard=apply(P)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(f):return json.loads(f.read_text('utf8'))
def write(f,v):f.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
c=read(S/'config.json');v=read(S/'RESULT.json');col=read(P/'COLLECTION_RECEIPT.json');budget=read(P/'CUMULATIVE_BUDGET_AND_CPU_RECEIPT.json')
assert col['native']['actual_worker_native_exit'] is not None
assert col['old_checkpoint_unchanged']and col['downloaded_and_reopened']
assert budget['cumulative_native_wall_s']<=600
assert read(S/'ENVIRONMENT.json')['CPU_affinity']==list(range(12))
assert v['original_A_F_complete']==v['original_R65_50000_complete']==v['automatic_long_resume']==False
restore=read(S/'COMMON_START_RESTORATION.json')
assert all(row['state_hashes']==restore[0]['state_hashes']for row in restore)
completed=v['status']in['POSITIVE_LOCAL_ONLY','NEGATIVE_LOCAL_SUPPORT']
report=['# R67 同起点、等成本小试', '', '这是固定 TRAIN 人口、共同第500步状态和固定学习率下的短程诊断；原 A–F 与 R65 的50,000步终点均未完成。', '']
root=dict(UTC=datetime.now(timezone.utc).isoformat(),stage=c['stage'],native_status=v['status'],native_exit=col['native']['actual_worker_native_exit'],
    completed_comparison=completed,cumulative_compute=budget,common_start_restoration_verified=True,
    frozen_update_body_hashes_verified=all(sha(S/n)==sha(R/'work/R66_EQUAL_COST_PILOT_001'/n)for n in ['full_step.py','mini_step.py']),
    parent_population=dict(SOURCE=6593,REAL=21047),DEV_read=0,EVAL_read=0,CAL_decode=0,
    original_A_F_complete=False,original_R65_50000_complete=False,automatic_long_resume=False,independent_training_replicates=False,
    only_fixed_model_noise_MC=True,root_resource=dict(guard,**memory()))
if completed:
    assert col['native']['actual_worker_native_exit']==0 and col['native']['termination_reason']is None
    names=['baseline','full','mini'];evals=[read(S/n/'EVALUATION.json')for n in names]
    ft=read(S/'full/TRAINING_RECEIPT.json');mt=read(S/'mini/TRAINING_RECEIPT.json')
    assert ft['completed_updates']==20 and ft['training_elapsed_s']<=120
    assert mt['cost_match_pass']and abs(ft['training_elapsed_s']-mt['training_elapsed_s'])<=max(.5,.01*ft['training_elapsed_s'])
    assert ft['fixed_learning_rate']==mt['fixed_learning_rate']==c['fixed_learning_rate']
    for name,e in zip(names,evals):
        assert e['arm']==name and e['probe_count']==8 and e['SOURCE_N']==6593 and e['REAL_N']==21047
        assert [r['seed']for r in e['probes']]==c['probe_seeds']and e['train_streams_and_model_unchanged']
        assert set(e['REAL_control']['pieces'])=={'full_window','coordinate_RMS','PSD','noise','temporal_joint'}
        for row in e['probes']:
            assert set(row['pieces'])=={'full_window','coordinate_RMS','PSD','noise','temporal_joint','source_order','source_valley','REAL_identity','box_correction'}
            assert abs(row['score']-sum(row['pieces'].values()))<1e-10
            assert row['numeric']['q64']<=1e-5 and row['numeric']['q32']<=1e-5
            assert all('PSD_W1_region_'+str(b)in row['nonnegative']for b in [-1,*range(9)])
    scores=np.array([[x['score']for x in e['probes']]for e in evals]);b,f,m=scores
    primary=bool(f.mean()<=.9*m.mean()and f.mean()<=.9*b.mean()and int((f<m).sum())>=6)
    assert primary==v['primary']['pass_gate']
    assert bool(primary and all(x['pass_gate']for x in v['harm_checks']))==(v['status']=='POSITIVE_LOCAL_ONLY')
    # Recompute all harm comparisons from reported original measurements.
    for row in v['harm_checks']:
        if row.get('status')=='unknown':assert not row['pass_gate'];continue
        bv,fv,mv=row['baseline'],row['full'],row['mini'];group=row['group']
        if group=='correlations':bound=max(bv,mv)-.02;ok=fv>=bound
        elif group=='valleys':bound=min(bv,mv)+.01;ok=fv<=bound
        else:bound=min(bv,mv)+.05*max(bv,mv)+1e-10;ok=fv<=bound
        assert abs(bound-row['bound'])<1e-12 and ok==row['pass_gate']
    mean=lambda e,g,k:float(np.mean([r[g][k]for r in e['probes']]))
    metrics=[]
    for key in ['overall','region_0','region_1','region_2','region_3','region_4','region_5','region_6','region_7','region_-1','region_8']:
        metrics.append(dict(metric='PSD_W1_'+key,baseline=mean(evals[0],'nonnegative','PSD_W1_'+key),full=mean(evals[1],'nonnegative','PSD_W1_'+key),mini=mean(evals[2],'nonnegative','PSD_W1_'+key)))
    root.update(primary_recomputed=True,harm_checks_recomputed=True,primary=v['primary'],failed_harm_checks=v['failed_harm_checks'],
        full_training=ft,mini_training=mt,PSD_means=metrics,source_rank_means={n:mean(e,'correlations','overall')for n,e in zip(names,evals)},
        final_decision='Only separately capped confirmation could be considered; long R65 remains held'if v['status']=='POSITIVE_LOCAL_ONLY'else'No demonstrated joint equal-cost benefit; do not resume expensive R65')
    conclusion='本次出现局部联合收益信号，但只能支持另一次单独登记、有上限的确认。'if v['status']=='POSITIVE_LOCAL_ONLY'else'本次未建立全量路线的联合等成本收益，不恢复昂贵的 R65 长训练。'
    report.extend([f'**结论：{conclusion}**','',
        f"两臂从同一第500步模型、Adam 与全部随机状态开始。全量完成20步，训练 {ft['training_elapsed_s']:.3f} 秒；小批完成 {mt['completed_updates']} 步，训练 {mt['training_elapsed_s']:.3f} 秒。时间差 {mt['cost_match_absolute_s']:.6f} 秒，匹配门通过。固定学习率 {c['fixed_learning_rate']:.12g}。",'',
        '| 共同指标 | 起点 | 全量 | 小批 |','| --- | ---: | ---: | ---: |',
        f"| 九项共同损失均值 | {b.mean():.8g} | {f.mean():.8g} | {m.mean():.8g} |",
        f"| 总体 PSD W1 | {metrics[0]['baseline']:.8g} | {metrics[0]['full']:.8g} | {metrics[0]['mini']:.8g} |",
        f"| 源 PSD 秩相关 | {root['source_rank_means']['baseline']:.8g} | {root['source_rank_means']['full']:.8g} | {root['source_rank_means']['mini']:.8g} |",'',
        f"全量相对小批的共同损失改善为 {v['primary']['full_vs_mini_relative_improvement']*100:.3f}%（负值代表更差）；相对起点为 {v['primary']['full_vs_baseline_relative_improvement']*100:.3f}%。全量在8组中赢 {v['primary']['full_beats_mini_probes']} 组。主门通过：{primary}；无实质退化门通过：{v['harm_gate_pass']}。",'',
        f"共有 {len(v['failed_harm_checks'])}/{len(v['harm_checks'])} 项退化门未通过，全部保存在 ROOT_RESULT_REVIEW.json；不删除失败区域。",''])
    if v['failed_harm_checks']:
        report.extend(['| 未通过的退化门 | 起点 | 全量 | 小批 |','| --- | ---: | ---: | ---: |'])
        for h in v['failed_harm_checks']:
            if h.get('status')=='unknown':report.append(f"| {h['group']} / {h['key']} | unknown | unknown | unknown |")
            else:report.append(f"| {h['group']} / {h['key']} | {h['baseline']:.8g} | {h['full']:.8g} | {h['mini']:.8g} |")
        report.append('')
    report.extend(['| 所有原区域 PSD W1 | 起点 | 全量 | 小批 |','| --- | ---: | ---: | ---: |'])
    for x in metrics:report.append(f"| {x['metric']} | {x['baseline']:.8g} | {x['full']:.8g} | {x['mini']:.8g} |")
    report.append('')
    # Values are saved aggregate JSON only; this is visualization, not new science.
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    colors=['#8b949e','#d85d42','#2368a0'];labels=['Shared start','Full D (20 steps)',f"Mini ({mt['completed_updates']} steps)"]
    ax=axes[0];ax.bar(np.arange(3),scores.mean(1),color=colors,width=.6)
    for j,rows in enumerate(scores):
        ax.scatter(np.full(8,j)+np.linspace(-.13,.13,8),rows,color='#1d2935',s=14,zorder=3)
        ax.text(j,max(rows)+.025*max(scores.max(),1),f'{rows.mean():.3f}',ha='center',fontsize=11)
    ax.set_xticks(np.arange(3),labels,fontsize=9);ax.set_ylabel('Common nine-term score (lower is better)')
    ax.set_ylim(0,max(scores.max(),1)*1.2);ax.set_title('Eight shared full-parent TRAIN probes',fontsize=11)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax=axes[1];regions=['Overall','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Below','Above'];x=np.arange(len(regions))
    for j,(name,color)in enumerate(zip(names,colors)):ax.plot(x,[r[name]for r in metrics],marker='o',ms=4,color=color,label=labels[j])
    ax.axhline(.01,color='#424242',linestyle='--',linewidth=1,label='Original PSD target (not an A-F verdict)')
    ax.set_xticks(x,regions,rotation=40,ha='right',fontsize=9);ax.set_ylabel('Canonical PSD W1');ax.set_title('All original charge regions retained',fontsize=11)
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);ax.legend(fontsize=7.5,loc='best')
    fig.suptitle('Local equal-cost pilot: '+('joint gate passed'if v['status']=='POSITIVE_LOCAL_ONLY'else'joint gate not passed'),fontsize=14)
    fig.savefig(S/'R67_COMPARISON.png',dpi=170);fig.savefig(S/'R67_COMPARISON.svg');plt.close(fig)
    root['figure']='work/R67_EQUAL_COST_REPAIR_001/R67_COMPARISON.png'
else:
    root.update(final_decision='No valid complete comparison; no further retry under this cost assessment and no expensive R65 restart',error=v.get('error'))
    report.extend(['**结论：结果不足，未获得有效完整比较，不恢复昂贵 R65。**','',f"执行状态：{v['status']}。原因：{v.get('error',v.get('reason','unknown'))}。",''])
report.extend([f"本次原生运行 {col['native']['elapsed_s']:.3f} 秒；前次启动失败消耗 {budget['prior_attempt_native_wall_s']:.3f} 秒；累计 {budget['cumulative_native_wall_s']:.3f} 秒，未超过原600秒总上限。worker 与 supervisor 已由原生回执和进程消失核验结束。GPU 计算进程列表为空：{not bool(col['GPU_processes'])}。",'',
    'R66 初次试验因单核亲和性启动问题无效，已单独封存。本次启动前强制验证实际 CPU0–11，未在运行中改资源、方法或终点。把预定全量步数改为20，是在没有任何训练终点结果时按剩余预算登记的新试验；原 R66 的30步终点从未改写。', '',
    '这些数据只描述共同第500步起点、固定学习率和本次短程训练。八组是固定模型和 TRAIN 父人口上的噪声/方向重复，不是独立训练或独立采集重复。共同起点此前由 full-D 训练产生，也不能支持从原始初始化开始的优劣结论。原 A–F、峰位/峰宽、五视图判别、数值重放和任务效用没有在本次完成或重开。', '',
    '冻结方案：tasks/R67_EQUAL_COST_REPAIR_001.md。原始结果：work/R67_EQUAL_COST_REPAIR_001/RESULT.json。全部退化门、数值核验、成本与原生回执：reviews/R67_EQUAL_COST_REPAIR_001/ROOT_RESULT_REVIEW.json 与 COLLECTION_RECEIPT.json。', '',
    '后续仅整理此次已有证据并准备可检验的机制解释，不自动恢复 R65，不继续追加本次比较。总体模型目标仍未完成。',''])
(R/'R67_REPORT.md').write_text('\n'.join(report),encoding='utf8')
root.update(report_SHA256=sha(R/'R67_REPORT.md'),original_acceptance_hashes={n:sha(R/n)for n in ['PROJECT_ACCEPTANCE.json','PROJECT_ACCEPTANCE.md']})
write(P/'ROOT_RESULT_REVIEW.json',root)
before=P/'before_closure_metadata';before.mkdir()
for name in ['PROJECT_PROGRESS.json','PROGRAM.json','AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:shutil.copy2(R/name,before/name)
for name in ['PROJECT_PROGRESS.json','PROGRAM.json']:
    d=read(R/name);d.update(status='R67_cost_benefit_diagnostic_closed',project_status='R67_cost_benefit_diagnostic_closed',
        current_stage=c['stage'],current_stage_progress=v['status'],current_activity='completed_pilot_review_and_delivery',
        active_scientific_jobs=0,latest_completed_stage=c['stage'],latest_diagnostic_stage=c['stage'],latest_completed_report='R67_REPORT.md',
        latest_scientific_decision=root['final_decision'],latest_completed_review='reviews/R67_EQUAL_COST_REPAIR_001/ROOT_RESULT_REVIEW.json',
        next_action='Deliver this bounded evidence and analyze existing results only; no automatic long R65 continuation or further pilot retry.',
        model_goal_complete=False,project_model_goal_complete=False,application_goal_status='active',
        latest_delivery_status='R66_R67_root_review_complete_delivery_pending',
        current_stage_response_training_runs_started=2 if completed else None)
    d['cost_benefit_assessment']=dict(R66='invalid_execution',R67=v['status'],cumulative_native_wall_s=budget['cumulative_native_wall_s'],original_cap_s=600,
        original_12_CPU_allocation_restored=True,long_run_benefit_proven=False,automatic_resume=False)
    write(R/name,d)
header='# Current: bounded cost-benefit assessment closed; R65 remains on cost hold\n\nR67 result: '+v['status']+'. Root verified native completion, common starting states, full parent metrics and the original cumulative600-second cap. '+root['final_decision']+'. R66 failed execution is separately preserved. No active scientific worker and no further retry under this assessment. Original A-F and R65 fixed50000 endpoint remain incomplete. See R67_REPORT.md and reviews/R67_EQUAL_COST_REPAIR_001/ROOT_RESULT_REVIEW.json. All new local files stay on E/D.\n\n## Earlier records\n\n'
for name in ['AGENTS.md','PLAN.md','PROJECT_OBJECTIVE.md']:(R/name).write_text(header+(R/name).read_text('utf8'),encoding='utf8')
print(json.dumps(dict(status=v['status'],completed=completed,decision=root['final_decision'],primary=root.get('primary'),
    failed_harms=len(root.get('failed_harm_checks',[])),cumulative_native_wall_s=budget['cumulative_native_wall_s'],report='R67_REPORT.md'),indent=2))
