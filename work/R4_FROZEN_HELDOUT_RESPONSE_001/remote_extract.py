from __future__ import annotations
import os,sys,time,signal,json,hashlib,io,zipfile,datetime,pathlib
START=time.monotonic()
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('R4 extraction600s')))
signal.alarm(600)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='4'
print(json.dumps({'kind':'R4_remote_started','PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}),file=sys.stderr,flush=True)
import numpy as np
if np.__version__!=REMOTE_NUMPY_VERSION:raise RuntimeError('historical NumPy environment drift; no replacement')
scope={'__name__':'r4_interface'}
exec(compile(INTERFACE_SOURCE,'bound_P1K_interface_for_R4','exec'),scope)
scope['np']=np
plan=PLAN;indices=INDICES;n=len(indices);positions={i:p for p,i in enumerate(indices)}
if n!=7600 or len(positions)!=n:raise RuntimeError('fixed7600 uniqueness')
a={'block_mean_V_f64':np.full((n,500),np.nan,dtype='<f8'),
   'window_normalized_f32':np.full((n,480),np.nan,dtype='<f4'),
   'window_V_f32':np.full((n,480),np.nan,dtype='<f4'),
   'normalization_f32':np.full((n,2),np.nan,dtype='<f4'),
   'geometry_i32':np.full((n,9),-1,dtype='<i4'),
   'signed_metrics_f64':np.full((n,8),np.nan,dtype='<f8'),
   'QC_reason_u16':np.zeros(n,dtype='<u2'),'DEV_bitwise_u8':np.zeros(2000,dtype='u1')}
cost={'reserved':dict.fromkeys(['hash','metadata','compressed','inflate'],0),
      'actual_returned':dict.fromkeys(['hash','metadata','compressed','inflate'],0),
      'seeks':0,'members':[],'hash_passes':0}
def on_time(i,b):
    t=np.frombuffer(b,dtype='<f8')
    if i!=0 or len(t)!=10000 or not np.all(np.isfinite(t)) or not np.all(np.diff(t)>0) or not np.all(np.abs(np.diff(t)-.05)<=1e-9):raise RuntimeError('bound time grid')
def on_wave(i,b):scope['convert_row'](positions[i],b,a,None,{})
spec=plan['container'];p=pathlib.Path(spec['path'])
if p.is_symlink() or p.stat().st_size!=spec['bytes']:raise RuntimeError('source size/link binding')
with p.open('rb') as raw:
    before=os.fstat(raw.fileno());f=scope['Counted'](raw,cost,plan,START+600)
    with zipfile.ZipFile(f,'r',allowZip64=True) as z:
        infos=z.infolist()
        if [i.filename for i in infos]!=[s['name'] for s in plan['members']]:raise RuntimeError('central member order binding')
        wants={s['name']:s for s in plan['members']}
        for info in infos:
            s=wants[info.filename]
            if (info.file_size,info.compress_size,info.CRC,info.compress_type)!=(s['uncompressed_bytes'],s['compressed_bytes'],int(s['crc32'],16),8):raise RuntimeError('central exact binding')
            if info.flag_bits not in (0,2048) or info.extract_version>45:raise RuntimeError('central unsupported flags/version')
        scope['stream_member'](f,z.getinfo('time_ns.npy'),wants['time_ns.npy'],[0],on_time)
        scope['stream_member'](f,z.getinfo('waveforms.npy'),wants['waveforms.npy'],indices,on_wave)
    after=os.fstat(raw.fileno());fields=['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']
    if any(getattr(before,k)!=getattr(after,k) for k in fields):raise RuntimeError('same-handle source drift')
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_before_save_s':time.monotonic()-START,
         'cost':cost,'selected_records':n,'selected_waveform_values':n*10000,'time_values':10000,
         'unselected_numeric_decodes':0,'source_identity_string_decodes':0,'whole_source_hash_passes':0,
         'source_SHA_inherited':spec['sha256'],'source_stat_before':{k:getattr(before,k) for k in fields},
         'python':sys.version,'numpy':np.__version__,'numpy_source':np.__file__,'peak_RSS_bytes':scope['peak_rss'](),
         'response_fits':0,'QC_valid':int(np.sum(a['QC_reason_u16']==0))}
buffer=io.BytesIO()
np.savez_compressed(buffer,window_V_f32=a['window_V_f32'],window_normalized_f32=a['window_normalized_f32'],
                    normalization_f32=a['normalization_f32'],geometry_i32=a['geometry_i32'],
                    QC_reason_u16=a['QC_reason_u16'],signed_metrics_f64=a['signed_metrics_f64'],
                    original_index=np.array(indices,dtype='<u4'))
payload=buffer.getvalue()
if len(payload)>33554432:raise RuntimeError('selected output32MiB cap')
receipt.update(output_bytes=len(payload),output_sha256=hashlib.sha256(payload).hexdigest(),elapsed_before_transfer_s=time.monotonic()-START)
print(json.dumps({'kind':'R4_remote_result','receipt':receipt}),file=sys.stderr,flush=True)
sys.stdout.buffer.write(payload);sys.stdout.buffer.flush()
