from pathlib import Path
from datetime import datetime,timezone
import json,ast,hashlib,os
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'R38_SLICED_MONOTONE_TRANSPORT_001'
t=(OLD/'deploy_training.py').read_text('utf8')
start=t.index('files=');end=t.index('\nbindings=',start)
t=t[:start]+"files=['train_server.py','online_training.py','kernel_core.py','fine_core.py','prior.py','model.py','legacy_model.py','reorder_core.py','linux_supervise.py','linux_owned_worker.py','remote_config.json','PROTOCOL.md']"+t[end:]
t=t.replace("(old/'AFFINE_CONSTANTS.npz').is_file()and(old/'inputs/TRAIN_PRIVATE.npz').is_file()","(old/'inputs/TRAIN_PRIVATE.npz').is_file()and(old/'inputs/INITIAL_TRANSLATOR.pt').is_file()")
t=t.replace("repr(cfg['remote_R33_root'])","repr('/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R32_REMOTE_20261009_170201_UTC')")
(ROOT/'deploy_training.py').write_text(t,'utf8')
t=(OLD/'retrieve.py').read_text('utf8').replace('SLICED_MONOTONE_TRANSPORT','JOINT_KERNEL_FINAL')
t=t.replace("['STATISTICS_INPUT_BINDING.json','BASE_FORWARD_HASHES.json','SLICED_FORMULA_CHECK.json','STATISTICS_FAILURE.json','REORDER_CONSTRUCTION_RECEIPT.json','DIRECTION_INITIALIZATION.json']","['MATCHED_TRAINING_FLOW_RECEIPT.json','KERNEL_FORMULA_CHECK.json','TRAINING_FAILURE.json','REORDER_CONSTRUCTION_RECEIPT.json']")
t=t.replace("'training':json.loads((ROOT/'TRAINING_NUMERIC_RECEIPT.json').read_text('utf8'))","'training_counter':json.loads((ROOT/'TRAINING_NUMERIC_RECEIPT.json').read_text('utf8'))['counter']")
(ROOT/'retrieve.py').write_text(t,'utf8')
# Hard supervisor limit includes import/formula/calibration/update/validation/weights sealing.
p=ROOT/'linux_supervise.py';s=p.read_text('utf8').replace(',960)',',900)');p.write_text(s,'utf8')
# Update-only per-call evidence saved as checkpoint files every250 in addition to final.
p=ROOT/'online_training.py';s=p.read_text('utf8')
s=s.replace("if step%cfg['validation_every']==0:\n                valrow", "if step%cfg['validation_every']==0:\n                fine.persist();np.savez_compressed(root/'BASIC_LOSS_COMPONENTS_BY_UPDATE_PRIVATE.npz',values=np.asarray(component_rows,dtype='f8'));\n                valrow")
s=s.replace('grad_rows=[]','grad_rows=[];component_rows=[]')
s=s.replace("loss.backward();g_norm", "component_rows.append([*[float(v.detach()) for v in components.values()],float(style_loss.detach()),float(loss.detach())]);loss.backward();g_norm")
s=s.replace("fine.persist();assert fine.calls==updates", "fine.persist();np.savez_compressed(root/'BASIC_LOSS_COMPONENTS_BY_UPDATE_PRIVATE.npz',values=np.asarray(component_rows,dtype='f8'));assert fine.calls==updates")
p.write_text(s,'utf8')
protocol='''# R39 frozen joint-kernel neural translation

Task R39_FULL_JOINT_KERNEL_TRANSLATION_001 plus startup R39_REAL_CONTROL_SCOPE_ADDENDUM_001 are authoritative. One neural arm JOINT_KERNEL_FINAL. Original R21 selected2000 G/reader and R18 initial critic; new Adam lr.0002/betas(.5,.999), clip5, identical seed20261009230 and all original R26 2950x32 row/noise/style/NCE streams. Critic is explicitly STANDARDIZED. Only fine training projected-quantile objective changes; all basic terms/weights/style recovery/J20 validation stay unchanged. Mean extra coefficient0. No post-G transform, projection, smoothing or BASE fallback. Sole evaluated endpoint2950; J20 best checkpoint retained as record only.

Three original float32 standardized blocks (u480,diff479,sbaseline100) then float64 explicit pair difference. Distance=sum(block mean squared distance)/3; kernel mean(exp(-D/(2sigma²))) for fixed sigma .25,.5,1,2,4. Offdiagonal GG/RR means plus full cross GR mean*(-2). Negative finite U estimates retained; cross diagonal remains. Both generated positions retain derivatives. Formula/gradient/exchange/permutation/negative-identical-U toy checks execute once before training.

Exactly first64 actual TRAIN batches with original initial G and actual z/white/style, no_grad G, independent y output gradient. Original fine reference gets its own copied PCG64(20261009281) first64 calls/192 float32 direction hashes. c=sqrt(sum||grad old||²/sum||grad MMD||²), float64 aggregation, no clamp/epsilon/retry. Invalid sum<=1e-24/nonfinite/nonpositive c stops. Extra2048 source forwards count separately. Calibration does not consume original streams or optimizer updates. Every update persists raw/scaled loss and three Gram summaries plus base/style losses and clip evidence;3x2950=8850 Gram groups,5 kernel terms each group=44250 groups-bandwidth terms, no new projection stream during updates.

All original2000 source parents, including417 outside historical support, get new G. Two REAL controls use exact existing2000 TRAIN_REFERENCE and main z/white/style, separately frozen BASE G/new G. No composition or PDG labels assigned to REAL. Same-held-out selection and groups,30new HGB/24density slots at most24EM, reference zero refits. Source actual preADC Q and historical ADC reference Q separately. Source832/joint1024 noise records and128 first supported input perturbations saved exact. Input perturbation is full NN source150:400 zero-mean1mV sine; source original bin descriptive fixed, no post-G perturbation. One14bitADC. Independent CPU replay6000 using source/control saved inputs and weights, G chunk128, all components and pre/post bytes.

Six joint gates in task, gate5 specifically superseded BEFORE any science by addon: candidate REAL-control minus same-treated BASE REAL-control paired upper95%<=.10 and pointdelta<=.01 on full493/u480/baseline. Relative RAW effects all described, stochastic G(REAL) is not presumed identity. Other source/PSD/peak/width/noise/support/boundary gates unchanged. Undefined not pass. Fixed recipe closes at any failure, no scientific rescue. Same acquisition/unknown particle truth retained.

1800s total science,training/calibration/validation<=900s,CPU4,actualPID import-before handshake,RSS/GPUallocated4GiB,512MiB output. Executor23:04:32.204052UTC. Original stages immutable, E/D local cache, no new data/EVAL/CAL/G4/install/Git/agents. Normal transfer only; no migration content tests. Kernel references: Gretton2012 https://jmlr.org/papers/v13/gretton12a.html and Li2015 https://proceedings.mlr.press/v37/li15.html ; use of existing kernel criterion is not algorithm novelty.
'''
(ROOT/'PROTOCOL.md').write_text(protocol,'utf8')
bindings={}
for p in ROOT.glob('*.py'):
    b=p.read_bytes();ast.parse(b.decode('utf8'));bindings[p.name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
r={'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'source_bindings':bindings,'science_started':False,'task_read_and_addendum_bound':True}
(ROOT/'SOURCE_BINDINGS.json').write_text(json.dumps(r,indent=2)+'\n','utf8');print(json.dumps({'PID':os.getpid(),'UTC':r['UTC'],'AST_count':len(bindings),'training_entry_ready':True}),flush=True)
