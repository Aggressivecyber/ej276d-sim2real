"""One simple stdlib delivery pass, no scientific array decoding or fits."""
import ast, datetime as dt, hashlib, json, os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def utc():return dt.datetime.now(dt.timezone.utc)
def read(name):return json.loads((ROOT/name).read_text(encoding='utf8'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
start=read('START.json');deadline=dt.datetime.fromisoformat(start['stage_deadline_UTC'])
assert utc()<deadline and not (ROOT/'FINAL_RECEIPT.json').exists()
def write(name,obj):
    assert utc()<deadline
    b=(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
    assert len(b)<=131072
    (ROOT/name).write_bytes(b)

prep=read('PREPARATION_RECEIPT.json')
for name,expected in prep['source'].items():
    assert digest(ROOT/name)==expected,f'executed source changed:{name}'
    ast.parse((ROOT/name).read_text(encoding='utf8'),filename=name)
assert digest(ROOT/'config.json')['sha256']==prep['config_sha256']
assert digest(ROOT/'PROTOCOL.md')['sha256']==prep['protocol_sha256']
m=read('metrics.json');w=read('results/ENERGY_WIDTH_METRICS.json')
paired=read('results/PAIRED_WIDTH_ERROR_SUMMARY.json');h=read('RUN_HISTORY.json');r=read('REPLAY_CHECK.json')
assert len(h)==2 and all(x['worker_native_exit']==0 for x in h)
assert r['bitwise_equal'] and r['float32_values_compared']==960000 and r['max_absolute_difference_V']==0
assert r['parent_ID_order_equal']
for name,key in [('model.pt','model_SHA256'),('GENERATION_RANDOM_STATE.pt','generation_RNG_SHA256'),('generation_inputs.npz','generation_inputs_SHA256')]:
    assert digest(ROOT/name)['sha256']==r[key]
assert m['current_model_fit_DEV_events']==0 and m['EVAL_read']==0
assert m['training']['optimizer_updates']==2000 and m['training']['selected_update']==1700
assert paired['common_bins_1_based']==[2,3,4,5,6,7]
figures=sorted(p.name for p in (ROOT/'results/figures').glob('*.png'));assert len(figures)==12
write('DELIVERY_QA.json',{'UTC':utc().isoformat(),'PID':os.getpid(),'source_config_protocol_unchanged':True,'AST_pass':True,
                        'plots_actually_viewed':figures,'layout':'Legends/axes/complete distributions checked; no clipping.',
                        'preserved_visual_residuals':['Leading shoulders/local double peaks in fixed composition pair and stochastic examples.',
                                                     'Low-density-component high-charge width too large; lowest slice displaced; highest slice nonconverged.'],
                        'paired_width_error_summary_bins':[2,3,4,5,6,7],'unpaired6_vs7_means_not_directly_compared':True,
                        'R3B_QC_valid':m['domains']['R3B']['QC_valid'],'REAL_QC_valid':m['domains']['REAL']['QC_valid'],
                        'ADC_clips':m['generation']['ADC_clip_count'],'padding':m['generation']['padding_count'],
                        'fresh_saved_weight_replay':r,'R3B_seed_before_model_constructor':2026100836,
                        'R2_original_initialization':'unknown; inherited exact warm checkpoint chain retained',
                        'whole_response_similarity_pass':False,'EVAL_read':0,'SSH':0,'new_G4':0,
                        'no_new_fit_or_array_decode_for_delivery':True})
curves={}
for name,d in {**w['before'],'R3B_DEV':w['R3B_DEV']}.items():
    curves[name]={'N':d['N'],'out_charge_low_high':[d['below_logQ_range'],d['above_logQ_range']],
                  'complete_qualified_slices':sum(s['valid'] for s in d['slices']),
                  'slices':[{'bin_ID_1_based':s['bin']+1,'N':s['N'],'valid':s['valid'],'reasons':s['failure_reasons'],
                             'components':[{k:c[k] for k in ['order','mean','weight','effective_N_times_weight','width_defined','sigma_PSD','detrended_sigma_PSD']} for c in s['components']]} for s in d['slices']]}
write('METRICS_COMPACT.json',{'status':'conditional_middle_bins_improved;whole_response_similarity_failed',
                            'recipe':'one fixed R3 architecture/CDF recipe, seed2026100836',
                            'training':m['training'],'overall_distances_vs_REAL':{n:d['overall'] for n,d in m['domain_vs_REAL'].items()},
                            'C2ST_R3B':m['C2ST']['R3B_vs_REAL'],
                            'quartile_QC_counts':{n:{k:s['QC_n'] for k,s in m['domains'][n]['regions'].items()} for n in ['R3','R3B','REAL']},
                            'paired_width_errors':[{k:v for k,v in s.items() if k!='paired_values'} for s in paired['summary']],
                            'charge_edges_mVns':w['edges_mVns'],'conditional_density_curves':curves,
                            'REAL_bootstrap_width_trends':{n:b['trend_slopes'] for n,b in w['REAL_bootstrap'].items()},
                            'fixed_parent_width':w['fixed_parent_width'],'full_metrics':'metrics.json',
                            'full_width_metrics':'results/ENERGY_WIDTH_METRICS.json','paired_summary':'results/PAIRED_WIDTH_ERROR_SUMMARY.json',
                            'replay':r,'CAL_q4_reference_kept_failed':True,'EVAL_read':0,'external_independent_test':False})
numeric=h[-1]['numeric_cumulative_wall_s'];paired_summary_tool_wall=.5053878
peak=max(x['peak_RSS_observed'] for x in h)
before=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
assert numeric+paired_summary_tool_wall<=1800 and peak<=4*1024**3 and before+131072<=512*1024**2
write('RESOURCE_RECEIPT.json',{'UTC':utc().isoformat(),'PID':os.getpid(),'start':start,
                             'main_science_jobs':1,'fresh_saved_weight_replay_jobs':1,'maximum_concurrent_scientific_jobs':1,
                             'numeric_wall_s_science_and_replay':numeric,'paired_existing_JSON_summary_tool_wall_s':paired_summary_tool_wall,
                             'numeric_wall_s_conservative_including_small_summary':numeric+paired_summary_tool_wall,
                             'numeric_limit_s':1800,'optimizer_updates':2000,'selected_update':1700,'threads':4,
                             'science_child_peak_RSS_observed_B':peak,'RSS_limit_B':4*1024**3,
                             'RSS_probe_failures':sum(x['memory_probe_failures'] for x in h),
                             'RSS_scope':'Observed science-child peak; small stdlib report/metadata process RSS not separately measured.',
                             'new_stage_outputs_before_final_inventory_B':before,'final_inventory_write_upper_B':131072,
                             'new_stage_output_limit_B':512*1024**2,'native_runs':h,
                             'native_tool_receipts':[{'kind':'main_diagnosis_training_generation_evaluation','start_tool':'f7bba7','session_ID':71501,'completion_tool':'73496a','native_exit':0},
                                                     {'kind':'saved_weight_replay','tool':'705a83','native_exit':0},
                                                     {'kind':'small_paired_summary_from_existing_JSON','tool':'9648f8','PID':61776,'native_exit':0}],
                             'preparation_tool':{'tool':'b7186b','PID':44608,'native_exit':0},
                             'failures':0,'scientific_retries':0,'warning':'Tensor-to-scalar receipt warning retained in run_001/stderr.log.',
                             'density_fits':'Frozen eight slices; three REAL partitions each200 artificial-group bootstrap; nonconvergence not retried.',
                             'EVAL_read':0,'SSH':0,'new_G4':0,'old_stages_modified':False,
                             'output_and_cache_location':str(ROOT).replace('\\','/'),
                             'TEMP_TMP':'E:/GAN/research/sim2real_two_papers_20261008_0305/.runtime-cache/tmp',
                             'final_hash_verification':'Pure byte operation; scientific arrays not decoded.'})
members=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p.name not in {'FINAL_MANIFEST.json','FINAL_RECEIPT.json'}:
        members.append({'path':p.relative_to(ROOT).as_posix(),**digest(p)})
write('FINAL_MANIFEST.json',{'UTC':utc().isoformat(),'PID':os.getpid(),'member_count':len(members),
                           'member_bytes':sum(x['bytes'] for x in members),'self_reference_excluded':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'files':members})
anchors={name:digest(ROOT/name) for name in ['REPORT.md','metrics.json','METRICS_COMPACT.json','model.pt','generated.npz','REPLAY_CHECK.json','results/ENERGY_WIDTH_METRICS.json','results/PAIRED_WIDTH_ERROR_SUMMARY.json','RESOURCE_RECEIPT.json','FINAL_MANIFEST.json']}
write('FINAL_RECEIPT.json',{'seal_UTC':utc().isoformat(),'seal_PID':os.getpid(),'status':'complete;stage writes stop after this file',
                          'stage_deadline_UTC':start['stage_deadline_UTC'],'manifest_member_count':len(members),
                          'anchors':anchors,'native_exit_authority':'Outer calling tool receipt, not self-asserted here.',
                          'current_fit_DEV_events':0,'EVAL_read':0,'replay_pass':True,'new_fit_during_finalization':0,
                          'result':'REAL charge-narrowing supported;paired bins2–7 improved;edge/width/prompt residuals remain, whole C2ST fails.'})
files=[p for p in ROOT.rglob('*') if p.is_file()]
last=max(p.stat().st_mtime_ns for p in files)
print(json.dumps({'seal_PID':os.getpid(),'completed_UTC':utc().isoformat(),
                  'last_file_write_UTC':dt.datetime.fromtimestamp(last/1e9,dt.timezone.utc).isoformat(),
                  'file_count':len(files),'total_bytes':sum(p.stat().st_size for p in files),
                  'manifest':digest(ROOT/'FINAL_MANIFEST.json'),'receipt':digest(ROOT/'FINAL_RECEIPT.json'),
                  'anchors':anchors,'numeric_cumulative_s':numeric,'no_scientific_recompute':True},ensure_ascii=False))
