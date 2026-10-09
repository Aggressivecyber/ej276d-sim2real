import pathlib,json,hashlib,os,datetime,sys
ROOT=pathlib.Path(__file__).resolve().parent;cfg=json.loads((ROOT/'config.json').read_text());now=datetime.datetime.now(datetime.timezone.utc)
if now>=datetime.datetime.fromisoformat(cfg['deadline_UTC']):raise TimeoutError('executor seal deadline passed')
if (ROOT/'FINAL_RECEIPT.json').exists():raise RuntimeError('already sealed')
native=json.loads((ROOT/'REMOTE_NATIVE_RECEIPT.json').read_text());replay=json.loads((ROOT/'REPLAY_RECEIPT.json').read_text());resource=json.loads((ROOT/'RESOURCE_RECEIPT.json').read_text())
if native['status']!='complete' or not replay['pass']:raise RuntimeError('science not terminal complete or replay failed')
files=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file()or p.name in ['FINAL_MANIFEST.json','FINAL_RECEIPT.json']:continue
    with p.open('rb')as f:
        h=hashlib.sha256();n=0
        while True:
            b=f.read(1024*1024)
            if not b:break
            n+=len(b);h.update(b)
    files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':n,'sha256':h.hexdigest()})
manifest={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid(),'entries':files,'file_count':len(files),'member_bytes':sum(f['bytes']for f in files),'self_reference_excluded':['FINAL_MANIFEST.json','FINAL_RECEIPT.json'],'scientific_inputs_not_rehashed':True,'migration_equality_tests':False,'all_local_stage_files_including_failures_and_cache_listed':True}
raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8');(ROOT/'FINAL_MANIFEST.json').write_bytes(raw)
def entry(n):return next(f for f in files if f['path']==n)
receipt={'PID':os.getpid(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'sealed','science_status':'finite_negative_result; complete-wave source translation implemented, target joint response unmatched','file_count_manifest_members':len(files),'manifest_member_bytes':manifest['member_bytes'],'manifest_sha256':hashlib.sha256(raw).hexdigest(),'manifest_bytes':len(raw),'REPORT':entry('REPORT.md'),'METRICS_COMPACT':entry('METRICS_COMPACT.json'),'RESOURCE_RECEIPT':entry('RESOURCE_RECEIPT.json'),'science_cumulative_s':resource['science_cumulative_s'],'native_science_exit_codes':[x['exit_code']for x in native['history']],'replay_pass':replay['pass'],'ordinary_failures_preserved':True,'source_control_metadata_corrected':True,'deadline_UTC':cfg['deadline_UTC'],'no_new_task_started':True,'last_write_UTC_assigned_immediately_before_final_receipt':datetime.datetime.now(datetime.timezone.utc).isoformat()}
final=(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
if manifest['member_bytes']+len(raw)+len(final)>cfg['output_limit_bytes']:raise RuntimeError('stage output limit')
(ROOT/'FINAL_RECEIPT.json').write_bytes(final);print(json.dumps({**receipt,'final_receipt_sha256':hashlib.sha256(final).hexdigest(),'final_directory_file_count':len(files)+2,'final_directory_bytes':manifest['member_bytes']+len(raw)+len(final)}),flush=True)
