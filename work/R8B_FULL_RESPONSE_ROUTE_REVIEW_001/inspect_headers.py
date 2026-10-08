from __future__ import annotations
import ast,datetime,hashlib,json,os,pathlib,struct,zipfile,zlib
ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT.parent.parent
SPECS=[
 (BASE/'work/R2_SMOOTH_STOCHASTIC_RESPONSE_001/data/G4_TRAIN_CACHE.npz',None),
 (BASE/'work/R5_PREFIX_RECONSTRUCTION_UTILITY_001/synthetic_waveforms.npz',['SIM_V.npy','parent_ids.npy']),
 (pathlib.Path('E:/GAN/baseline_mean_fix_20261004_1125/inputs/evaluation_inputs.npz'),['raw.npy','normalization_peak_V.npy','sim_event_ids.npy','conditions.npy','edep_ee_MeV.npy','edep_hadron_MeV.npy','physical_r_h.npy','source_label.npy']),
]
records=[]
for path,selected in SPECS:
 row={'path':str(path),'bytes':path.stat().st_size,'members':[]}
 with zipfile.ZipFile(path) as archive,path.open('rb') as f:
  entries=archive.infolist()
  for info in entries:
   if selected is not None and info.filename not in selected:continue
   if not info.filename.endswith('.npy'):raise ValueError('unexpected member')
   f.seek(info.header_offset);local=f.read(30)
   if len(local)!=30 or local[:4]!=b'PK\x03\x04':raise ValueError('local ZIP header')
   name_n,extra_n=struct.unpack_from('<HH',local,26)
   f.seek(info.header_offset+30+name_n+extra_n)
   compressed_read=0;out=b'';pending=b'';dec=zlib.decompressobj(-15) if info.compress_type==8 else None
   if info.compress_type not in [0,8]:raise ValueError('compression')
   def take(n):
    global out,pending,compressed_read
    while len(out)<n:
     if dec is None:
      block=f.read(n-len(out));compressed_read+=len(block);out+=block
      if not block:raise EOFError('stored header')
     else:
      if not pending:
       if compressed_read>=info.compress_size:raise EOFError('compressed header')
       pending=f.read(min(256,info.compress_size-compressed_read));compressed_read+=len(pending)
       if not pending:raise EOFError('compressed source')
      previous=len(pending);part=dec.decompress(pending,n-len(out));pending=dec.unconsumed_tail;out+=part
      if not part and len(pending)==previous:raise ValueError('no header progress')
    return out
   first=take(10)
   if first[:6]!=b'\x93NUMPY':raise ValueError('NPY magic')
   version=tuple(first[6:8]);prefix=10 if version==(1,0) else 12
   if version not in [(1,0),(2,0),(3,0)]:raise ValueError('NPY version')
   take(prefix);length=struct.unpack_from('<H' if prefix==10 else '<I',out,8)[0]
   if length>4096:raise ValueError('NPY header cap')
   take(prefix+length)
   header=ast.literal_eval(out[prefix:].decode('utf-8' if version==(3,0) else 'latin1').strip())
   row['members'].append({'name':info.filename,'header':header,'header_bytes':len(out),'header_sha256':hashlib.sha256(out).hexdigest(),'compressed_prefix_bytes_returned':compressed_read,'uncompressed_body_bytes_returned':0,'central_uncompressed_bytes':info.file_size,'CRC_from_central':info.CRC,'full_CRC_not_rechecked_in_source_only_review':True})
 records.append(row)
result={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'headers_only','files':records,'NumPy_import':0,'scientific_body_decode':0,'scientific_body_bytes_inflated':0,'new_whole_container_hash':0,'fit':0,'generation':0,'EVAL':0,'SSH':0}
(ROOT/'INPUT_HEADERS.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
