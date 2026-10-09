"""Control-only documentation and one final seal; no scientific recomputation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
DEADLINE = datetime.fromisoformat('2026-10-09T23:39:22.826524+00:00')
START = time.monotonic()


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(name):
    return json.loads((ROOT / name).read_text('utf8'))


def encode(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf8')


def put(name, obj):
    p = ROOT / name
    if p.exists():
        raise RuntimeError(('refuse existing final artifact', name))
    p.write_bytes(encode(obj))


def put_text(name, text):
    p = ROOT / name
    if p.exists():
        raise RuntimeError(('refuse existing final artifact', name))
    p.write_text(text.strip() + '\n', 'utf8')


def digest(p):
    h = hashlib.sha256()
    n = 0
    with p.open('rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            n += len(b)
            h.update(b)
    return {'bytes': n, 'sha256': h.hexdigest()}


def link(name):
    return f'[{name}]({(ROOT / name).as_posix()})'


def main():
    if datetime.now(timezone.utc) >= DEADLINE:
        raise RuntimeError('finalization deadline expired')
    for name in ['REPORT.md', 'REPRODUCTION.md', 'RESOURCE_RECEIPT.json',
                 'NATIVE_CHAIN.json', 'FAILURE_ACCOUNTING.json', 'FIGURE_QA.json',
                 'FINAL_MANIFEST.json', 'FINAL_RECEIPT.json']:
        if (ROOT / name).exists():
            raise RuntimeError(('already finalizing or sealed', name))
    history = load('RUN_HISTORY.json')
    gate = load('GATE_METRICS.json')
    train = load('TRAINING_COMPACT.json')
    train_native = load('TRAINING_NUMERIC_RECEIPT.json')
    gen = load('GENERATION_NUMERIC_RECEIPT.json')
    ev = load('EVALUATION_NUMERIC_RECEIPT.json')
    replay = load('REPLAY_RECEIPT.json')
    plot = load('PLOT_SUMMARY_RECEIPT.json')
    prep = load('PREPARATION_RECEIPT.json')
    transfer = load('TRANSFER_RECEIPT.json')
    returned = load('RETURN_TRANSFER_RECEIPT.json')
    module = load('FINAL_MODULE_ZERO_DESCRIPTION.json')
    changes = load('MODULE_PARAMETER_CHANGE.json')
    science_s = math.fsum(x['elapsed_s'] for x in history)
    assert len(history) == 5
    assert all(x['actual_worker_native_exit'] == 0 for x in history)
    assert science_s <= 1800
    assert train['updates'] == train['D_updates'] == train['style_reader_updates'] == 2950
    assert ev['counters']['HGB_attempts'] == 20
    assert ev['counters']['density_description_slots'] == 16
    assert ev['counters']['density_EM_attempts'] == 15
    assert replay['rows'] == 4000 and replay['pass']
    passes = [gate[k]['pass'] for k in gate if k.startswith('gate')]
    assert passes == [False, False, True, False, False, True]
    assert len(plot['figures']) == 13

    # These are notes from actual image openings, not automatic image checks.
    qa_notes = {
        'c2st.png': 'All 10 fixed views and paired intervals readable; legend outside data; missing RAW 2D prediction explicit.',
        'charge_psd.png': 'Common Q and PSD axes; full populations and narrow generated bands retained.',
        'conditional_psd_CDF.png': 'Short bin titles, support counts and bin8 support gap visible.',
        'energy_peak_width.png': 'Both component centers, raw/detrended widths and missing support shown without connecting gaps.',
        'conditional_waveforms.png': 'Full 480-point raw mV conditional mean/median; residual tail structure remains visible.',
        'noise_curvature.png': 'RMS, ACF and prompt/postpeak curvature units readable; prompt is not pure rising edge.',
        'source_preservation.png': 'Actual preADC source-Q populations 213/181 and source proxies distinguished from truth.',
        'boundary_differences.png': 'All nine prespecified boundaries and native mV distributions readable.',
        'training_time_mixer.png': 'Actual step2950 J20 plotted, selected1250 record-only, gradients and initial-to-final changes explicit.',
        'DEV_full_correlation.png': '480-coordinate description on common correlation scale; no DEV fit implied.',
        'source_and_random_decomposition.png': 'Source input perturbation and corrected probe references clear; functional covariance terms include cross terms.',
        'same_treated_REAL_controls.png': 'Gate lines apply only to shape/u/baseline; V480 descriptive; RAW identity not presumed.',
        'final_module_action.png': 'Final module disabled is descriptive action, not a second candidate or retraining ablation.'
    }
    assert set(qa_notes) == set(plot['figures'])
    put('FIGURE_QA.json', {
        'UTC': utc(), 'reviewer': 'sole executor',
        'method': 'Each of the 13 final PNGs actually opened with tools.view_image and visually inspected.',
        'all_final_figures_opened': True, 'figure_count': 13,
        'new_render_after_QA': False, 'display_repairs_required': [],
        'per_image_opening_UTC': 'not individually captured; no invented per-image timestamps',
        'figures': {k: dict(plot['figures'][k], visual_pass=True, note=v) for k, v in qa_notes.items()}
    })

    tools = {
        'train_server.py': {'launch_tool': 'e3e1b0', 'terminal_tool': 'b4def6', 'tool_native_exit': 0,
                            'local_launch_PID': 45344, 'SSH_PID': 28588,
                            'native_path': 'remote_receipts/train_server.py.NATIVE_RECEIPT.json'},
        'generate_all.py': {'launch_tool': '1a4da7', 'terminal_tool': '4c747b', 'tool_native_exit': 0,
                            'native_path': 'generate_all.py.NATIVE_RECEIPT.json'},
        'evaluate_saved.py': {'launch_tool': 'c2809a', 'terminal_tool': '210391', 'tool_native_exit': 0,
                             'native_path': 'evaluate_saved.py.NATIVE_RECEIPT.json'},
        'replay_all.py': {'launch_tool': 'f74f0f', 'terminal_tool': 'f74f0f', 'tool_native_exit': 0,
                         'native_path': 'replay_all.py.NATIVE_RECEIPT.json'},
        'summarize_and_plot.py': {'launch_tool': 'd70974', 'terminal_tool': 'f44ef3', 'tool_native_exit': 0,
                                 'native_path': 'summarize_and_plot.py.NATIVE_RECEIPT.json'}
    }
    put('NATIVE_CHAIN.json', {
        'first_actual_control': load('START_RECEIPT.json'),
        'scientific_processes': [dict(x, external_tool_evidence=tools[x['entry']]) for x in history],
        'one_scientific_process_at_a_time': True,
        'server_preflight_receipts': ['SERVER_PREFLIGHT.json', 'RESOURCE_CHECK_NATIVE.json'],
        'normal_transfer_receipts': ['TRANSFER_RECEIPT.json', 'RETURN_TRANSFER_RECEIPT.json'],
        'seal_native_exit_evidence': 'External tool response after finalize_stage.py exits; parent binds it without further stage writes.'
    })
    put('FAILURE_ACCOUNTING.json', {
        'scientific_attempts': 5, 'scientific_native_failures': 0,
        'unknown_scientific_job_state': False,
        'control_edit_failure': {
            'operation': 'apply_patch attempted DELETE and ADD for prepare_evaluation.py in one patch',
            'reason': 'Multiple operations target the same file',
            'files_written': 0, 'science_started_by_failed_edit': False,
            'correction': 'UPDATE-only patch disabled the historical R39 helper; prepare_evaluation_r40.py is the actual entry.',
            'original_tool_identifier': 'unknown', 'PID': 'not applicable to apply_patch'
        },
        'initial_J20_assert_order': {
            'actual_initial_absolute_difference': 0.0, 'tolerance': 1e-6,
            'component_equivalence_before_updates': True,
            'J20_comparison_assertion': 'executed at training completion, not immediately before updates',
            'mismatch': False, 'extra_training_or_restart': False
        },
        'RSS_query_failures': history[0]['memory_probe_failure_events'],
        'RSS_query_failure_cause': 'unknown; proximity to process exit is not proof of cause',
        'local_science_RSS_query_failures': 0,
        'all_original_stdout_stderr_native_receipts_retained': True,
        'expected_first_W1_zero_gradient': True,
        'scientific_negative_results_preserved': True
    })

    g2 = gate['gate2_PSD_width_centers_noise']['measures']
    metric_rows = []
    labels = {
        'PSD_overall_W1': '总体 PSD W1', 'PSD_common_Q_equal_mean_W1': '共同 Q bins1–7 等权 PSD W1',
        'component0_peak_position_MAE': '低侧峰位 MAE，bins2–7',
        'component1_peak_position_MAE': '高侧峰位 MAE，bins2–7',
        'component0_raw_width_MAE': '低侧 raw-width MAE，bins2–7',
        'component1_raw_width_MAE': '高侧 raw-width MAE，bins2–7',
        'baseline_RMS_V_overall_W1': '基线 RMS W1（V）', 'ACF1_overall_W1': '基线 ACF1 W1'
    }
    for name in labels:
        m = g2[name]
        metric_rows.append(f"| {labels[name]} | {m['BASE']:.9g} | {m['candidate']:.9g} | {100*m['relative_change']:+.2f}% |")
    same = gate['gate5_same_treated_REAL_control']['three_views']
    control_rows = []
    for name, label in [('full_historical_shape', 'shape493'), ('normalized_u480', '纯 u480'), ('baseline_local', 'baseline')]:
        v = same[name]
        ci = v['BA_delta_95CI']
        control_rows.append(f"| {label} | {v['BASE_G_REAL_BA']:.3f} | {v['new_G_REAL_BA']:.3f} | {v['BA_delta_first_minus_second']:+.3f} [{ci[0]:+.5f}, {ci[1]:+.5f}] | {'通过' if v['pass'] else '失败'} |")
    boundary_failures = [x for x in gate['gate4_boundary_source']['boundaries'] if not x['pass']]
    change_rows = []
    for name, v in changes['parameters'].items():
        change_rows.append(f"| {name} | {v['delta_L2_norm']:.7f} | {v['max_abs_delta']:.7f} |")
    report = f"""
# R40 固定全段时间混合层：关闭此配方

唯一2950末点未通过完整响应共同门槛，六门结果为 **失败／失败／通过／失败／失败／通过**。完整 shape493 BA=.984、纯u480 BA=.982，相对冻结R32 BASE=.978的配对差分别+.006（95%CI [−.008504,.021652]）与+.004（[−.011299,.020007]）；两区间上界均不小于0，且远高于预定.934/.928点值限额。保存模型与4000条独立完整链重放通过，不能把可复现代替科学通过。关闭这一固定480→128→480单层配方，不据此声称所有非局部模型无效。

本轮保持完整480点G4派生源波形进入原网络，不是实测目标密度独立抽样。唯一结构在最终16通道style FiLM后、signal/electronic两个头前共享一次 `h + Linear(128,480)(GELU(Linear(480,128)(h)))`，新增123488参数；W2零初始化。时间混合思想参考[Tolstikhin等，MLP-Mixer](https://proceedings.neurips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html)，只使用跨时间MLP思想，不是完整MLP-Mixer复现。原网络已有全局归一化、扩张卷积和style，新增连接也增加参数/FLOP，不能将对比唯一归因于感受野。

完整R32 CONTROL配置、原初始化、STANDARDIZED D、所有原损失及fresh分位数L1恢复；没有R39 MMD、均值额外项或后置校准。2950×32的94400行、七项实际源/实测/NCE/随机流、8850组fresh方向均匹配冻结对照。实际暴露为3752唯一SOURCE与18739唯一REAL，而非载入6593/21047全部参与更新。G/D/reader各2950。64 SOURCE与64 REAL的旧/新三功能分量初始逐值相同，共256额外模型前向；原CPU/CUDA RNG未被新层初始化推进。初始J20=.6166301966，差0；组件核对在更新前，J20差断言实际在训练完成后执行，此顺序局限保留。内部选择1250的J20=.5569948554；实际2950末点J20=.6295024157，作为唯一DEV端点，没有用DEV换成1250。

第一更新W1权重/偏置梯度0为零W2初始化的预期，W2权重/偏置梯度2.111989975/.882978976。四个新增参数都与保存initial比较，变化有限且非零：

| 新模块参数 | final−initial L2 | 最大绝对变化 |
| --- | ---: | ---: |
{chr(10).join(change_rows)}

两侧PSD峰位及宽度同时退化，不能用总体均值曲线或ACF单项改善作成功判据。下表所有PSD距离/宽度是同一冻结定义；峰宽拟合为密度成分描述，不是粒子标签真值。

| 指标 | 冻结 BASE | 本轮2950 | 相对变化 |
| --- | ---: | ---: | ---: |
{chr(10).join(metric_rows)}

总体和共同Q PSD W1分别恶化20.37%/11.71%，低/高raw-width误差恶化17.86%/33.99%，峰位误差恶化31.92%/16.54%；基线RMS W1恶化52.69%，ACF1 W1改善48.17%。共同峰位/宽度区是bins2–7，BASE这六区均继续合格，没有缩交集掩盖新增失败。主候选bin8只有38条，`total_support_below40`，不宣称覆盖最高区。新REAL控制bin1/8各因`not_converged500`不支持；所有旧密度缺口保留在 {link('GATE_METRICS.json')}。

2000原源父ID、组成、顺序、权重全部保持，417个TRAIN支持外父也用新G，没有BASE回退、删行、新无效或ADC裁剪。actual preADC源-Q的pure-had bins3/4分别213/181父，谷比例BASE5.6338%/.5525%→本轮2.8169%/0%；历史postADC SOURCE_CLEAN的207/182父另外保存，不能混称。全人口源Q→输出Q Spearman .9997263→.9997787，源PSD→输出PSD .4189917→.4243149，源代理门通过，但相关性与跨域谷区不是逐事件粒子保真真值。源交换有作用，也不证明物理对应正确。

9项固定边界中3个二阶量失败：center99 W1=.215528mV>限.106272mV；center149=.749649>.742689mV；center150=.807848>.703809mV。其余6项通过。前128个支持内完整SOURCE输入150:400零积分1mV扰动保持原噪声；输出全部非相同、preADC全480 RMS中位1.140239mV，输入Qfast严格不变、Qtotal/PSD仅保存float32微小误差。扰动敏感性单项通过不抵消边界失败，不解释成物理增益。

同处理REAL控制完全复用原2000 TRAIN_REFERENCE及主z/white/style；旧BASE_REAL波形、10评价器/密度/回放引用封存R39，零重生成/重拟合。新减旧的预定三门如下，baseline点差+.016超过+.01，故控制门失败。V480及其它视图只描述。对RAW的绝对失真完整报告；随机G可能对已含噪REAL再加响应，不能预设RAW恒等。

| 同处理REAL视图 | BASE BA | 新BA | 配对差与95%CI | 预定门 |
| --- | ---: | ---: | --- | --- |
{chr(10).join(control_rows)}

最终新模块关闭描述只对固定前128行进行，preADC/信号/电子全480 RMS差中位分别.883872/.876522/.086752mV；全部非相同。它表明新增层在最终网络中实际作用于两个头，不是第二候选、纯方差消融或从头无模块训练反事实，不对它拟合HGB/EM。signal/electronic共享总归一化及交叉项，仅为功能分量，不宣称独立物理噪声。条件均值/中位尾部仍有固定时间起伏，DEV相关矩阵亦保留结构差异；prompt100:150包含峰和早期下降，不能全称纯上升沿。

服务器唯一训练真实PID77574、监督77573，GPU allocated峰283249664B，RSS峰2707943424B；训练记录115.998228s，含导入/封存的原生过程118.756857s。两次RSS查询gap保留22:27:54.529290/.629565时间戳，原因unknown，不将靠近退出当成已证实原因。其余四个本机科学过程原生0；科学累计{science_s:.9f}s/1800s，20新HGB、16描述槽/15实际EM，旧评价器零新拟合。新增参数/初始化256前向单列，FLOP未测量，不声称等算力。所有13张最终PNG已实际打开检查，布局/单位/支持缺口清晰。

本轮使用已暴露的同采集TRAIN/DEV与人工50记录组，区组bootstrap条件于固定分类器和这些组，不涵盖响应重训或独立采集不确定性。没有新EVAL/CAL波形、REAL采集、G4、安装或迁移内容测试。结论限于这一固定全段时间MLP未解决总体响应差距；不追加本轮变体或新拟合，下一问题由根依据完整证据登记。

完整聚合结果 {link('METRICS_COMPACT.json')}；全门槛 {link('GATE_METRICS.json')}；训练 {link('TRAINING_COMPACT.json')}；重放 {link('REPLAY_RECEIPT.json')}；13图实际QA {link('FIGURE_QA.json')}；实际资源/原生链 {link('RESOURCE_RECEIPT.json')}、{link('NATIVE_CHAIN.json')}。完整逐事件输入、预测、随机流、模型与输出保留E/D，不进入聚合发布版。
"""
    put_text('REPORT.md', report)
    reproduce = f"""
# R40 执行与复现

实际训练在 `rylan@100.82.231.102` 的 `{history[0]['remote_root']}`，解释器 `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`；Python3.12.3 / NumPy1.26.4 / torch2.12.0+cu130 / scipy1.11.4 / cuda:0（RTX5070）。本机生成、20HGB/15EM、独立4000重放和绘图使用 `C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe`，torch2.10.0+cpu / NumPy2.2.6；两端不是逐位训练平台对照，CPU重放只与本轮CPU保存输出逐值比较。

实际输入与版本在 `SOURCE_BINDINGS.json`、`PRETRAIN_FREEZE.json`、`EVALUATION_SOURCE_FREEZE.json`。本轮 `ACTUAL_GENERATION_INPUTS.npz` 保留原2000完整source480、IDs、z/white/style；`REAL_DEV_AND_REFERENCE_PRIVATE.npz` 保留既有DEV与2000TRAIN_REFERENCE；`JOINT8_INPUTS_PRIVATE.npz`、`PROBE_INPUTS_CORRECTED.npz` 保留预定随机/源控制。服务器训练只读复用原R32/R33的训练载荷，原94400行和七流/8850方向的核对记录在 `MATCHED_TRAINING_FLOW_RECEIPT.json`。没有迁移逐文件相等测试；传输以正常scp/ssh退出与实际加载为据。

唯一科学模型为 `GLOBAL_TIME_MIXER_FINAL/FINAL_TRANSLATOR.pt`（2950）。`TRANSLATOR.pt` 是内部1250所选记录，不能替代本轮唯一DEV终点。`INITIAL_FULL_TRANSLATOR.pt`、`INITIAL_TIME_MIXER.pt`、reader/D初态与最终状态、optimizer和每步随机/梯度轨迹都保留。源结果在 `GLOBAL_TIME_MIXER_FINAL/generated.npz`，REAL处理控制在 `GLOBAL_TIME_MIXER_REAL_CONTROL/generated.npz`；旧BASE_REAL由R39封存目录引用，不能再次生成来替换历史对照。

本轮实际顺序：`prepare_stage.py` → `finish_prep.py` → 资源预检/`deploy_training.py` → `launch_remote.py train_server.py`（调用Linux `linux_supervise.py train_server.py`）→ `retrieve.py` → `prepare_evaluation_r40.py` → 本机监督 `generate_all.py`、`evaluate_saved.py`、`replay_all.py` → `prepare_summary.py`、监督 `summarize_and_plot.py`。历史模板 `prepare_evaluation.py` 明确禁用，不能用来覆盖本轮入口。

本机实际监督命令形式（cwd为本轮E目录，所有TEMP/TMP/缓存在项目E盘）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:TEMP='E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp'
$env:TMP=$env:TEMP
& 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe' -B ./supervise.py replay_all.py 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe'
```

该主重放已经独立执行：真实PID16360/监督40420、native0、7.0646635s；4000行全部preADC/postADC、signal/electronic/sum-rounding、父/参考ID与支持分支逐值通过。推理没有把保存输出或reader/评价器作为计算输入；保存输出仅作为最终比较对象。旧BASE_REAL独立回放直接引用R39证据。本轮再现事实见 `REPLAY_RECEIPT.json` 与完整原生日志。

未来恢复须使用新E/D目录及新资源/时间登记，保持科学config/源/权重/实际输入/随机流固定，只调整执行路径和时限。当前目录已封存，`.used`/SSH一次性标记与旧期限不得删掉后重复提交。训练从头再现仍受平台/库版本影响，不保证跨CPU/CUDA位等；本轮独立推理4000位等是已执行的有限证据。无需为此文档再启动计算。
"""
    put_text('REPRODUCTION.md', reproduce)

    local_files = [p for p in ROOT.rglob('*') if p.is_file()]
    local_bytes = sum(p.stat().st_size for p in local_files)
    transfer_bytes = transfer['upload_tar_bytes'] + returned['return_tar_bytes']
    remote_return_members_bytes = sum(x[1] for x in returned['remote_outputs']['files'])
    resource = {
        'UTC': utc(), 'PID': os.getpid(), 'stage': ROOT.name,
        'registered_UTC': '2026-10-09T22:19:22.826524+00:00',
        'first_actual_control': load('START_RECEIPT.json'),
        'executor_seal_deadline_UTC': DEADLINE.isoformat(),
        'science_total_s': science_s, 'science_cap_s': 1800,
        'training_supervised_s': history[0]['elapsed_s'], 'training_record_s': train['training_s'], 'training_cap_s': 900,
        'science_processes': history,
        'CPU_threads': 4, 'RSS_cap_bytes': 4 * 1024**3,
        'actual_worker_RSS_peak_observed_bytes': max(x['actual_worker_RSS_peak_bytes'] for x in history),
        'RSS_monitoring_complete': False,
        'RSS_query_gap_count': 2, 'RSS_query_gap_events': history[0]['memory_probe_failure_events'],
        'RSS_query_gap_cause': 'unknown',
        'worker_self_peak_RSS_bytes': history[0]['worker_self_receipt']['self_peak_RSS_bytes'],
        'GPU_peak_allocated_bytes': train_native['GPU_peak_allocated_bytes'], 'GPU_allocated_cap_bytes': 4*1024**3,
        'training_updates': {'G': 2950, 'D': 2950, 'style_reader': 2950, 'fits': 1},
        'actually_consumed_rows': 94400, 'actual_unique_source': 3752, 'actual_unique_REAL': 18739,
        'loaded_pool_source': 6593, 'loaded_pool_REAL': 21047,
        'initial_comparison_extra_model_forward_rows': 256,
        'validation_generation_rows': train['validation_generation_rows'],
        'fine_projection_calls': 2950, 'fresh_direction_groups': 8850,
        'new_parameters': 123488, 'FLOP_count': 'not measured; increased parameters/computation not claimed equal',
        'generation_counters': gen['counters'], 'evaluation_counters': ev['counters'],
        'new_complete_CPU_replay_rows': 4000, 'new_model_zero_description_rows': 128,
        'old_BASE_REAL_generation_or_fit': 0, 'old_RAW_RR_fit': 0,
        'scientific_native_failures': 0, 'control_edit_failures': 1,
        'EVAL_read': 0, 'CAL_waveform_decode': 0, 'new_REAL_acquisition': 0,
        'new_G4': 0, 'installation': 0, 'new_agents': 0, 'Git_operations': 0,
        'old_stage_writes': 0, 'migration_content_tests': 0,
        'local_file_count_before_resource_and_seal': len(local_files),
        'all_local_stage_bytes_before_resource_and_seal': local_bytes,
        'local_dependency_payload_copied_bytes': prep['dependency_payload_bytes'],
        'upload_tar_bytes': transfer['upload_tar_bytes'], 'return_tar_bytes': returned['return_tar_bytes'],
        'transport_packages_total_bytes': transfer_bytes,
        'remote_return_member_bytes': remote_return_members_bytes,
        'partition_note': 'Dependency and package amounts are reported separately; packages already exist in the all-local-stage total. Remote output may duplicate local downloaded files; no disjoint-subtraction claim.',
        'output_cap_bytes': 512*1024**2,
        'final_exact_stage_bytes_and_count': 'FINAL_RECEIPT.json includes manifest and itself',
        'finalization_scope': 'stdlib JSON/documentation/byte hashing only; no fit, inference, generation, array decode or re-render',
        'conclusion': gate['decision']
    }
    put('RESOURCE_RECEIPT.json', resource)

    # All immutable ordinary files are included; two terminal controls explicitly
    # exclude themselves to avoid a self-referential hash cycle.
    excluded = {'FINAL_MANIFEST.json', 'FINAL_RECEIPT.json'}
    items = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.name in excluded:
            continue
        if p.is_symlink():
            raise RuntimeError(('unexpected stage symlink', str(p)))
        items.append(dict(path=p.relative_to(ROOT).as_posix(), **digest(p)))
    total_member_bytes = sum(x['bytes'] for x in items)
    manifest = {
        'stage': ROOT.name, 'UTC': utc(), 'PID': os.getpid(),
        'files': items, 'file_count': len(items), 'member_bytes': total_member_bytes,
        'excluded_terminal_controls': sorted(excluded),
        'self_reference_rule': 'Manifest and final receipt are excluded from members. Final receipt anchors this manifest; external native exit is bound by the parent after return.',
        'old_stage_files_modified': False, 'scientific_recomputation_in_seal': False
    }
    manifest_raw = encode(manifest)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    put('FINAL_MANIFEST.json', manifest)
    receipt = {
        'stage': ROOT.name, 'status': 'sealed', 'PID': os.getpid(), 'UTC': utc(),
        'decision': gate['decision'], 'gates_in_order': passes,
        'retain': False, 'science_total_s': science_s,
        'manifest_sha256': manifest_sha, 'manifest_bytes': len(manifest_raw),
        'manifest_member_count': len(items), 'manifest_member_bytes': total_member_bytes,
        'stage_file_count_including_terminal_controls': len(items)+2,
        'stage_total_bytes_including_terminal_controls': 0,
        'receipt_bytes': 0, 'output_cap_bytes': 512*1024**2,
        'terminal_control_hashes': {name: digest(ROOT/name) for name in ['REPORT.md', 'REPRODUCTION.md', 'RESOURCE_RECEIPT.json', 'FIGURE_QA.json', 'NATIVE_CHAIN.json', 'FAILURE_ACCOUNTING.json']},
        'deadline_UTC': DEADLINE.isoformat(), 'before_deadline': True,
        'native_exit_evidence': 'Await actual external tool exit; this self receipt alone is not native exit evidence.',
        'no_further_stage_writes': True, 'scientific_job_active': False
    }
    for _ in range(8):
        n = len(encode(receipt))
        total = total_member_bytes+len(manifest_raw)+n
        if receipt['receipt_bytes'] == n and receipt['stage_total_bytes_including_terminal_controls'] == total:
            break
        receipt['receipt_bytes'] = n
        receipt['stage_total_bytes_including_terminal_controls'] = total
    else:
        raise RuntimeError('terminal size convergence')
    if datetime.now(timezone.utc) >= DEADLINE:
        raise RuntimeError('deadline reached before last write')
    assert receipt['stage_total_bytes_including_terminal_controls'] <= 512*1024**2
    put('FINAL_RECEIPT.json', receipt)
    # No file writes after this point. External process return is the native evidence.
    final_path = ROOT/'FINAL_RECEIPT.json'
    final_bytes = final_path.read_bytes()
    assert len(final_bytes) == receipt['receipt_bytes']
    last_ns = max(p.stat().st_mtime_ns for p in ROOT.rglob('*') if p.is_file())
    print(json.dumps({
        'PID': os.getpid(), 'status': 'sealed',
        'FINAL_MANIFEST_sha256': manifest_sha,
        'FINAL_RECEIPT_sha256': hashlib.sha256(final_bytes).hexdigest(),
        'last_stage_write_UTC': datetime.fromtimestamp(last_ns/1e9, timezone.utc).isoformat(),
        'file_count': receipt['stage_file_count_including_terminal_controls'],
        'bytes': receipt['stage_total_bytes_including_terminal_controls'],
        'science_s': science_s, 'seal_control_elapsed_s': time.monotonic()-START,
        'no_further_stage_writes': True
    }), flush=True)


if __name__ == '__main__':
    main()
