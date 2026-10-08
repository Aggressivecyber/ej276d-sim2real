from __future__ import annotations
import pathlib,json,hashlib,ast,datetime,os
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent.parent
FROZEN_SHA='83ed633a170471ef9013a5e70fda8fcd82a6d34de450bda61793f99a48156114'
START='2026-10-08T18:58:30.832822+00:00';END='2026-10-08T20:13:30.832822+00:00'
def dump(n,o):(ROOT/n).write_text(json.dumps(o,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def spec(p):
    p=pathlib.Path(p);h=hashlib.sha256();n=p.stat().st_size;count=0
    with p.open('rb') as f:
        while b:=f.read(1<<20):count+=len(b);h.update(b)
    if count!=n:raise RuntimeError('size changed '+str(p))
    return {'path':str(p),'bytes':n,'sha256':h.hexdigest()}
raw=(BASE/'CONTINUATION_FROZEN_METHOD.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==FROZEN_SHA
frozen=json.loads(raw);checked=[];seen={}
rows=frozen['model_and_generation_bindings']+frozen['shared_inference_and_stage_bindings']+frozen['prior_definition_bindings']+[frozen['R9_protocol'],frozen['R9_addendum']]
for r in rows:
    p=BASE/r['path'];key=str(p)
    if key not in seen:seen[key]=spec(p)
    current=seen[key]
    if (current['bytes'],current['sha256'])!=(r['bytes'],r['sha256']):raise RuntimeError('frozen drift '+key)
    checked.append({**r,'checked':True})
(ROOT/'CONTINUATION_FROZEN_METHOD.json').write_bytes(raw)
old=BASE/'work/R8C_WHOLE_RESPONSE_MAPPING_001'
for name in ['evaluation.py','diagnostic_core.py','common_math.py','assessment.py','energy_width.py']:
    (ROOT/name).write_bytes((old/name).read_bytes())
launch=(old/'launch.py').read_text(encoding='utf-8').replace('R8C_response_started','R9_evaluation_started').replace('whole_response_mapping','frozen_retrospective_validation')
(ROOT/'launch.py').write_text(launch,encoding='utf-8')
oldcfg=json.loads((old/'config.json').read_text());r4=BASE/'work/R4_FROZEN_HELDOUT_RESPONSE_001'
inputs={k:oldcfg['input_files'][k] for k in ['generation_inputs','SIM_identity','GEN_W0','CAL_reference','width_definition']}
paths={
 'EVAL_EXTRACT':r4/'EVAL_EXTRACT.npz', 'EVAL_identity':r4/'EVAL_IDENTITIES.json',
 'EVAL_MAIN_SELECTION':r4/'results/EVAL_MAIN_SELECTION.npz','RR_SELECTIONS':r4/'results/REAL_REAL_SELECTIONS.npz',
 'STANDARDIZATION':r4/'results/STANDARDIZATION.json','R4_WIDTHS':r4/'results/ENERGY_WIDTH_METRICS.json',
 'R8_metrics':BASE/'work/R8_LOCAL_JOINT_FLUCTUATION_001/metrics.json',
 'R8C_metrics':old/'metrics.json','R8_replay':BASE/'work/R8_LOCAL_JOINT_FLUCTUATION_001/REPLAY_CHECK.json',
 'R8C_replay':old/'REPLAY_RECEIPT.json','R3B_fixed_pairs':BASE/'work/R3B_ENERGY_CONDITIONAL_WIDTH_001/results/FIXED_CONDITION_PAIRS.json',
 'R8C_layout':old/'LAYOUT_QA_RECEIPT.json'}
for k,p in paths.items():inputs[k]=spec(p)
for n in ['R3B','JOINT','JOINT_SIGNAL','FULL_WAVE_STOCHASTIC','PULSE_STOCHASTIC_OFF']:
    r=next(s for s in frozen['model_and_generation_bindings'] if s['candidate']==n and s['kind']=='generation')
    inputs[n]={'path':str(BASE/r['path']),'bytes':r['bytes'],'sha256':r['sha256']}
ext=json.loads((r4/'EXTRACTION_RECEIPT.json').read_text())
assert inputs['EVAL_EXTRACT']['sha256']==ext['stdout_SHA256'] and ext['SSH_exit']==0
cfg={'stage':'R9_FROZEN_RETROSPECTIVE_VALIDATION_001','start_UTC':START,'deadline_UTC':END,
 'first_native_PID':60068,'first_native_chunk':'dcf735','first_native_exit':0,
 'numeric_wall_limit_s':1800,'threads':4,'RSS_limit_bytes':4294967296,'output_limit_bytes':536870912,
 'frozen_SHA256':FROZEN_SHA,'frozen_UTC':frozen['frozen_utc'],'candidates':frozen['candidates'],
 'input_files':inputs,'evaluation':oldcfg['evaluation'],'C2ST':oldcfg['C2ST'],
 'C2ST_views':oldcfg['C2ST_views'],'HGB_max_fits':42,'train_each_domain':600,'holdout_each_domain':250,
 'bootstrap_repeats':500,'bootstrap_seed':oldcfg['bootstrap_seed'],
 'REAL_main_rule':'Use exact saved R4 selected_positions/original_index/source_id; assert full SHA R4-main-real-v1 first2000 of original native QC matches; no redraw.',
 'REAL_all_supplement':'All7600 original rows, preserve17 original invalid; all7583 QC-valid, unknown included.',
 'C2ST_source_indices':'Common native-QC intersection of all5 candidates only for classifier sampling; original full populations separately retained. R1-SPLIT|SIM|parentID first8 SHA mod10<7, R1-C2ST-SAMPLE cap600/250.',
 'C2ST_REAL_indices':'Saved main2000, R1-SPLIT|REAL|original_index//50 group flag and frozen event ordering, cap600/250 across all candidate views.',
 'C2ST_RR':'Use saved R4 repeat0 halves; disjoint original artificial groups; same fixed split and feasible caps.',
 'C2ST_positive':'Same saved main2000, frozen seed202610081;2%peak exp tail t>=120 and1.5%peak Gaussian noise; same group partition as native REAL.',
 'same_N_RR':'Use original R4 saved64 disjoint whole-group selections,2000 QC rows per half. Fixed TRAIN R4 mean/popSD for128 angles [0,pi). Recompute R7 additional features without new fit.',
 'widths':'Reuse exact candidate native density tables from R4/R8/R8C and exact EVAL MAIN/all+bootstrap from R4; no new density fits. Assert unchanged counts/histograms for selected waves.',
 'width_common_support_sets':{'primary_five_candidates':['REAL',*frozen['candidates']],'all_domains_including_source_controls':['REAL','SIM','GEN_W0',*frozen['candidates']]},
 'source_controls_replay':'Quote sealed R8/R8C; no inference/replay without discrepancy. R8C branch OFF is whole capacity ablation, not pure variance.',
 'fixed_condition_pairs':'Quote saved R3B controlled energy/composition pairs; newer arms lack a corresponding separately controlled pair, report this gap instead of inventing one.',
 'response_updates':0,'new_G4':0,'SSH':0,'install':0,'Git':0,
 'previously_viewed_same_acquisition_retrospective':True,'EVAL_feedback_to_response':False}
cfg['evaluation'].update({'question':'R9 all5 frozen candidates on previously viewed same-acquisition EVAL','EVAL_waveform_or_feature_read':7600,'numeric_wall_limit_s':1800,'threads':4,'C2ST':{**cfg['evaluation']['C2ST'],'bootstrap_cluster_repeats':500}})
dump('config.json',cfg)
dump('START_RECEIPT.json',{'start_UTC':START,'deadline_UTC':END,'first_native_PID':60068,'first_tool_chunk':'dcf735','first_native_exit':0,'current_control_PID':os.getpid()})
dump('CONTROL_FAILURES.json',[{'chunk_id':'192da3','native_exit':1,'native_PID':'unknown','science_started':False,'error':'Metadata path STANDARDIZATION.json at R4 root absent; correct known location is results/STANDARDIZATION.json. No new scientific values were computed in this tool.'}])
dump('FROZEN_BINDING_CHECK.json',{'frozen_sha256':FROZEN_SHA,'frozen_UTC':frozen['frozen_utc'],'checked':checked,'unique_files':len(seen),'scientific_array_decodes':0,'current_config_sha256':spec(ROOT/'config.json')['sha256']})
print(json.dumps({'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_unique_bound_files':len(seen),'input_files':len(inputs),'EVAL_scientific_decodes':0}))
