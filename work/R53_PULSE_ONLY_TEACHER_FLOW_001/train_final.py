from pathlib import Path
from datetime import datetime,timezone
import json,os,time,sys,hashlib,copy,traceback
START=time.monotonic();R=Path(__file__).resolve().parent;C=json.loads((R/'config.json').read_text());ARM=C['arm'];D=R/ARM
import numpy as np,torch,scipy
from noise_core import q_np,bins_for,innovation
from source_flow_core import FlowNetwork,TangentField,q_torch
from teacher_core import validate,short_q
def write(p,x):(R/p).write_bytes((json.dumps(x,indent=2,allow_nan=False)+'\n').encode())
def utc():return datetime.now(timezone.utc).isoformat()
def state_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
updates=0;counter={'response_fit_attempts':0,'velocity_updates':0,'SOURCE_training_rows':0,'REAL_identity_rows':0,'REAL_matched_rows':0,'old_G_D_reader_forward_updates':0}
try:
 teacher,B=validate(R,C)
 if not teacher['passed']:
  write('TRAIN_NOT_STARTED.json',{'UTC':utc(),'reason':'fixed_teacher_structural_check_failed','counter':counter,'model_goal_complete':False});raise SystemExit(0)
 D.mkdir(exist_ok=False)
 torch.set_num_threads(C['CPU_threads']);torch.set_num_interop_threads(C['CPU_threads']);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 if not torch.cuda.is_available():raise RuntimeError('registered CUDA unavailable')
 dev=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(C['GPU_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
 env={'PID':os.getpid(),'UTC':utc(),'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'device':str(dev),'GPU':torch.cuda.get_device_name(0),'CPU_threads':torch.get_num_threads(),'cwd':str(R),'arm':ARM}
 write(ARM+'/ENVIRONMENT.json',env)
 assert all(env[k]==C['expected_R33_environment'][k] for k in ['python','numpy','torch','scipy','device','GPU']),('environment drift',env)
 with np.load(C['TRAIN_path'],allow_pickle=False) as z:sfull=z['source'].copy();rfull=z['target'].copy()
 with np.load(C['constants_path'],allow_pickle=False) as z:mu=z['mu_unit_q'].copy();scale=z['scale_unit_q'].copy();edges=z['edges_logQ'].copy()
 with np.load(C['original_stream'],allow_pickle=False) as z:si=z['source'].astype('i8');mi=z['matched_target'].astype('i8');oldbin=z['bin'].astype('i8');ordinal=z['original_row_ordinal'].copy()
 mappings={}
 for name,path,N in [('full',C['full_coupling_path'],94400),('partial',C['partial_coupling_path'],89600)]:
  with np.load(path,allow_pickle=False) as z:
   assert np.array_equal(si[:N],z['source']) and np.array_equal(mi[:N],z['matched_target_original']) and np.array_equal(ordinal[:N],z['original_row_ordinal'])
   mappings[name]=z['matched_target_joint'].astype('i8')
  assert np.array_equal(np.sort(mi[:N]),np.sort(mappings[name]))
 count_original=16*np.bincount(mi,minlength=len(rfull))+np.bincount(mi[:89600],minlength=len(rfull))
 count_joint=16*np.bincount(mappings['full'],minlength=len(rfull))+np.bincount(mappings['partial'],minlength=len(rfull));assert np.array_equal(count_original,count_joint)
 with np.load(C['noise_calibration_path'],allow_pickle=False) as z:noise={k:z[k].copy() for k in z.files}
 us,inv_s=np.unique(si,return_inverse=True);ur,inv_r=np.unique(mi,return_inverse=True);mapped={k:np.searchsorted(ur,v) for k,v in mappings.items()}
 assert all(np.array_equal(ur[mapped[k]],mappings[k]) for k in mapped)
 source=sfull[us].copy();real=rfull[ur].copy();loaded=[len(sfull),len(rfull)];del sfull,rfull
 sb,sraw,sq=bins_for(source,edges);rb,rraw,rq=bins_for(real,edges)
 assert np.array_equal(sb[inv_s],oldbin) and np.all(scale>0) and np.max(np.abs(q_np(mu)-1))<1e-10
 uS=torch.as_tensor(source.astype('f8')/sq[:,None],device=dev);rR=torch.as_tensor(real.astype('f8'),device=dev)
 rqt=torch.as_tensor(rq,device=dev);rs=torch.as_tensor(short_q(real),device=dev);uR=torch.as_tensor(real.astype('f8')/rq[:,None],device=dev)
 mt=torch.as_tensor(mu,dtype=torch.float64,device=dev);st=torch.as_tensor(scale,dtype=torch.float64,device=dev);Bt=torch.as_tensor(B,dtype=torch.float64,device=dev)
 torch.manual_seed(C['model_seed']);torch.cuda.manual_seed_all(C['model_seed']);model=FlowNetwork().to(dev)
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};ih=state_hash(initial)
 old_initial=torch.load(C['original_initial_state_path'],map_location='cpu',weights_only=True)
 assert set(old_initial)==set(initial) and all(torch.equal(v,old_initial[k]) for k,v in initial.items());del old_initial,initial
 inactive=model.embedding[0].weight[:,1:17].detach().clone()
 write(ARM+'/INITIALIZATION.json',{'UTC':utc(),'seed':C['model_seed'],'state_tensor_SHA':ih,'common_R48_initial_tensorwise_equal':True,'loaded_previous_trained_response':False,'parameter_count':sum(p.numel() for p in model.parameters()),'same_architecture':True})
 opt=torch.optim.Adam(model.parameters(),lr=C['lr'],betas=tuple(C['betas']),eps=1e-8,weight_decay=0)
 streams={k:np.random.Generator(np.random.PCG64(seed)) for k,seed in C['PCG64_seeds'].items()}
 write(ARM+'/INITIAL_RANDOM_STATES.json',{k:copy.deepcopy(v.bit_generator.state) for k,v in streams.items()})
 hashkeys=['source','matched_target','ordinal','source_t','source_epsilon','identity_t','identity_epsilon','matched_target_joint'];hashes={k:hashlib.sha256() for k in hashkeys}
 batch_counts=np.zeros(2950,dtype='u4');trace=[];dense=[];gradient=[];maxq=0.;zero_grad_count=zero_value_count=t_nonzero=bin_nonzero=0
 amplitude={'SOURCE_triggered':0,'REAL_triggered':0,'SOURCE_total':0,'REAL_total':0};counter['response_fit_attempts']=1
 write('TRAIN_STARTED.json',{'UTC':utc(),'PID':os.getpid(),'counter':counter,'sole_terminal_update':50000,'teacher_passed':True})
 for step in range(1,50001):
  batch=(step-1)%2950;sl=slice(batch*32,(batch+1)*32);ss=inv_s[sl];ri=inv_r[sl];mode='full' if step<=47200 else 'partial';rt=mapped[mode][sl]
  sx=source[ss];rx=real[ri]
  ts=streams['source_t'].uniform(0,1,32).astype('f4');es=streams['source_epsilon'].standard_normal((32,480)).astype('f4')
  ti=streams['identity_t'].uniform(0,1,32).astype('f4');ei=streams['identity_epsilon'].standard_normal((32,480)).astype('f4')
  ns,ms=innovation(sx,es,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
  ni,mr=innovation(rx,ei,mu,edges,noise['p480'],noise['kappa2'],float(noise['tau2']),True)
  bs=torch.as_tensor(sb[ss],device=dev);br=torch.as_tensor(rb[ri],device=dev);sqt=torch.as_tensor(sq[ss],device=dev)
  gS=(uS[ss]-mt[bs])/st[bs];gR=(uR[ri]-mt[br])/st[br]
  aS=gS+torch.as_tensor(ns/sq[ss,None],device=dev)/st[bs];aR=gR+torch.as_tensor(ni/rq[ri,None],device=dev)/st[br]
  alpha=sqt/rqt[rt]-1;coeff=torch.stack((alpha*rs[rt],alpha*(rqt[rt]-rs[rt])),1)
  target=rR[rt]+coeff@Bt.T;cS=(target/sqt[:,None]-mt[bs])/st[bs]
  tt=torch.as_tensor(np.concatenate((ts,ti)),dtype=torch.float64,device=dev);a=torch.cat((aS,aR));c=torch.cat((cS,gR));guide=torch.cat((gS,gR));bt=torch.cat((bs,br))
  y=(1-tt[:,None])*a+tt[:,None]*c;velocity=c-a;opt.zero_grad(set_to_none=True);pred=TangentField(model,mt,st,guide)(y,tt,bt)
  ls=(pred[:32]-velocity[:32]).square().mean();li=(pred[32:]-velocity[32:]).square().mean();loss=.5*ls+.5*li
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
  if step in [1,50000]:
   norms=[]
   for part in [ls,li]:
    grads=torch.autograd.grad(part,tuple(model.parameters()),retain_graph=True,allow_unused=True);norms.append(float(torch.sqrt(sum(g.double().square().sum() for g in grads if g is not None))))
   gradient.append({'step':step,'unweighted_SOURCE_gradient_norm':norms[0],'unweighted_IDENTITY_gradient_norm':norms[1],'weights':[.5,.5]})
  loss.backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite gradient')
  gg=model.embedding[0].weight.grad;assert gg is not None and torch.count_nonzero(gg[:,1:17])==0;zero_grad_count+=1
  t_nonzero+=int(torch.count_nonzero(gg[:,0])>0);bin_nonzero+=int(torch.count_nonzero(gg[:,17:])>0)
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),C['gradient_clip']);opt.step();updates=step
  counter['velocity_updates']+=1;counter['SOURCE_training_rows']+=32;counter['REAL_matched_rows']+=32;counter['REAL_identity_rows']+=32
  assert torch.equal(model.embedding[0].weight[:,1:17],inactive);zero_value_count+=1
  pathq=float(q_torch(st[bt]*velocity).abs().max());fieldq=float(q_torch(st[bt]*pred.detach()).abs().max());maxq=max(maxq,pathq,fieldq)
  dense.append([float(ls.detach()),float(li.detach()),float(loss.detach()),float(norm),pathq,fieldq])
  amplitude['SOURCE_triggered']+=int(np.count_nonzero(ms['noise_completion_triggered']));amplitude['REAL_triggered']+=int(np.count_nonzero(mr['noise_completion_triggered']));amplitude['SOURCE_total']+=32;amplitude['REAL_total']+=32
  for k,v in [('source',si[sl]),('matched_target',mi[sl]),('ordinal',ordinal[sl]),('source_t',ts),('source_epsilon',es),('identity_t',ti),('identity_epsilon',ei),('matched_target_joint',mappings[mode][sl])]:hashes[k].update(v.tobytes())
  batch_counts[batch]+=1
  if torch.cuda.max_memory_allocated()>C['GPU_limit_bytes'] or torch.cuda.max_memory_reserved()>C['GPU_limit_bytes']:raise MemoryError('GPU allocated/reserved cap')
  if step%1000==0 or step in [1,50000]:
   rec={'arm':ARM,'global_step':step,'source_loss':dense[-1][0],'identity_loss':dense[-1][1],'total_loss':dense[-1][2],'gradient_norm':dense[-1][3],'elapsed_s':time.monotonic()-START,'max_path_field_q_residual':maxq,'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved()}
   trace.append(rec);write(ARM+'/TRAINING_TRACE.json',trace);write('TRAIN_PROGRESS.json',rec);print(json.dumps(rec),flush=True)
 assert updates==50000 and all(torch.isfinite(p).all() for p in model.parameters())
 old=json.loads(Path(C['original_training_receipt']).read_text());streamhash={k:h.hexdigest() for k,h in hashes.items()}
 equality={k:streamhash[k]==old['stream_hashes'][k] for k in hashkeys if k!='matched_target_joint'}
 assert all(equality.values()) and ih==old['initial_state_SHA'] and old['updates']==50000 and batch_counts.tolist()==old['original_batch_occurrence_counts']
 torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},D/'FINAL_FLOW.pt');torch.save(opt.state_dict(),D/'FINAL_OPTIMIZER.pt')
 torch.save({'CPU':torch.get_rng_state(),'CUDA':torch.cuda.get_rng_state()},D/'FINAL_TORCH_RNG.pt')
 write(ARM+'/FINAL_RANDOM_STATES.json',{k:copy.deepcopy(v.bit_generator.state) for k,v in streams.items()})
 np.savez_compressed(D/'LOSS_BY_UPDATE_PRIVATE.npz',values=np.asarray(dense,dtype='f8'))
 rec={'PID':os.getpid(),'UTC':utc(),'arm':ARM,'updates':updates,'elapsed_s':time.monotonic()-START,'counter':counter,'selection':'sole50000 final','original_loaded_source_REAL':loaded,'actual_unique_SOURCE_REAL':[len(us),len(ur)],'original_rows':94400,
  'stream_hashes':streamhash,'same_R48_source_identity_and_random_streams':equality,'same_R48_target_total_occurrence_counts':True,'original_batch_occurrence_counts':batch_counts.tolist(),'initial_state_SHA':ih,'input_amplitudes':amplitude,'teacher_check_passed':True,
  'zero_input_gradient_checks':zero_grad_count,'zero_input_value_checks':zero_value_count,'t_nonzero_gradient_steps':t_nonzero,'bin_nonzero_gradient_steps':bin_nonzero,'GPU_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),
  'branch_gradient_descriptions':gradient,'unique_parent_population_increased':False,'partial_cycle_mapping_independent':True,'compute_budget_enforced':False,'model_goal_complete':False}
 write(ARM+'/TRAINING_RECEIPT.json',rec);print(json.dumps({'training_terminal':{'updates':updates,'stream_checks':equality,'elapsed_s':rec['elapsed_s']}}),flush=True)
except SystemExit:
 raise
except BaseException as exc:
 write('TRAINING_FAILURE.json',{'UTC':utc(),'PID':os.getpid(),'updates':updates,'counter':counter,'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-START});raise
