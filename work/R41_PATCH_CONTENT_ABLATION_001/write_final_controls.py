"""Actual visual QA and external tool identities; no scientific recomputation."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def put(n,v):
    p=ROOT/n;assert not p.exists()
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf8')
now=datetime.now(timezone.utc).isoformat()
plot=read('PLOT_SUMMARY_RECEIPT.json');repair=read('FIGURE_REPAIR_RECEIPT.json')
figures={**plot['figures'],**repair['final_figures']}
notes={
 'c2st.png':'Complete/local views, outside legend, short time-view labels and confidence intervals readable.',
 'charge_psd.png':'Common coordinates and complete source/REAL/candidate distributions visible.',
 'conditional_waveforms.png':'Complete 480-point means and medians, source composition and tail structure visible.',
 'conditional_psd_CDF.png':'Short conditional-bin titles and bin8 support gap retained.',
 'energy_peak_width.png':'Peak centers/raw-width gaps and deteriorations visible without filling unsupported bins.',
 'noise_curvature.png':'Complete CDFs, RMS and ACF tradeoffs, prompt/postpeak window meanings visible.',
 'training_no_patch_NCE.png':'Reopened repaired figure: project annotation below axes; actual final J20 and component curves retained.',
 'source_preservation.png':'Reopened repaired figure: final source-rank tick NO-NCE; actual input bins213/181 and valley caveat retained.',
 'boundary_differences.png':'All nine frozen differences and native mV scales readable.',
 'DEV_full_correlation.png':'Full 480 correlation matrices, common scale, all domains readable.',
 'source_and_random_decomposition.png':'Full-source perturbation, correctly referenced probes, functional branch variance/cross terms readable.',
 'same_treated_REAL_controls.png':'Same-treated paired differences; three gate views and descriptive V480 explicitly separated.'}
assert set(notes)==set(figures)and len(figures)==12
qa={}
for name,x in figures.items():
    p=ROOT/'results/figures'/name;b=p.read_bytes()
    assert len(b)==x['bytes']and hashlib.sha256(b).hexdigest()==x['sha256']
    qa[name]={**x,'visual_pass':True,'actual_open_method':'tools.view_image rendered through functions.exec image()',
        'display_repair_reopened':name in repair['final_figures'],'notes':notes[name]}
put('FIGURE_QA.json',{'UTC':now,'PID':os.getpid(),'all_final_figures_opened':True,'actual_final_figure_count':12,
    'figures':qa,'source':'executor actual image inspection; not inferred from matplotlib exit',
    'prior_initial_images_preserved':repair['preserved_initial_figures'],'executed_plot_script_unchanged':True,
    'new_fitting_generation_or_scientific_statistics':0})
history=read('RUN_HISTORY.json')
tools={
 'train_server.py':{'launch_tool':'549c70','session_id':76106,'terminal_tool':'6e3218','native_exit':0},
 'generate_all.py':{'launch_tool':'eb5655','session_id':54628,'terminal_tool':'c4c5d2','native_exit':0},
 'evaluate_saved.py':{'launch_tool':'16e218','session_id':37140,'terminal_tool':'e28294','native_exit':0},
 'replay_all.py':{'terminal_tool':'73584c','native_exit':0},
 'summarize_and_plot.py':{'launch_tool':'a68ced','session_id':62085,'terminal_tool':'921f4d','native_exit':0},
 'repair_figures.py':{'terminal_tool':'847633','native_exit':0}}
assert len(history)==6
chain=[]
for x in history:
    y=tools[x['entry']]
    chain.append({**y,'entry':x['entry'],'actual_worker_PID':x['actual_worker_PID'],
        'supervisor_PID':x['supervisor_PID'],'start_UTC':x['start_UTC'],'stop_UTC':x['stop_UTC'],
        'elapsed_s':x['elapsed_s'],'native_receipt':x['entry']+'.NATIVE_RECEIPT.json'})
put('EXTERNAL_NATIVE_CHAIN.json',{'UTC':now,'PID':os.getpid(),'source':'actual tools.exec_command and completed tools.write_stdin results; not self-inferred exit',
    'first_actual_control':{'tool':'d9d41c','PID':13652,'native_exit':0,'UTC':'2026-10-09T22:47:05.7590854Z'},
    'control_and_transfer_tools':[
        {'tool':'d115fd','PID':44272,'native_exit':0,'entry':'prepare_stage.py'},
        {'tool':'76a83a','PID':35096,'native_exit':0,'entry':'prepare_summary.py'},
        {'tool':'96adff','PID':21772,'SSH_PID':21408,'native_exit':0,'entry':'resource_check.py'},
        {'tool':'692186','PID':51256,'native_exit':0,'entry':'deploy_training.py'},
        {'tool':'cd9749','PID':13632,'native_exit':0,'entry':'retrieve.py'},
        {'tool':'6986ba','PID':23312,'native_exit':0,'entry':'AST repair_figures.py/seal_stage.py'}],
    'scientific_and_display_native_tools':chain,'control_native_failures':[],
    'final_seal_tool_native_exit':'pending actual final exec tool; returned after no further stage writes'})
print(json.dumps({'PID':os.getpid(),'UTC':now,'actual_final_figures_opened':len(qa),
    'final_controls':['FIGURE_QA.json','EXTERNAL_NATIVE_CHAIN.json']}),flush=True)
